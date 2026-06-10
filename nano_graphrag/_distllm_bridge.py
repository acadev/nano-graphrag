"""
Bridge between distllm's HuggingFace-Arrow embedding output
(https://github.com/ramanathanlab/distllm) and nano-graphrag's ingestion
pipeline.

distllm's `embed` pipeline (semantic chunking with SFR-Embedding-Mistral,
see configs/aurora_sfr_mistral_semantic_chunking.yaml) writes one HF
`datasets.Dataset` (Arrow) per shard with at least:

  - "text"      : the semantically-chunked passage
  - "embedding" : the SFR-Mistral embedding for that chunk
  - "path"      : the source document path (from pdfwf JSONL output)

This module converts that output into nano-graphrag's `TextChunkSchema`
("text_chunks" / "full_docs"), wires the precomputed embeddings into an
`EmbeddingFunc` so entity/relationship descriptions written later by the
indexing LLM still get embedded normally, and runs the entity-extraction +
clustering steps directly (skipping nano-graphrag's own chunker, since
distllm already performed semantic chunking).

It also provides `align_communities_to_ontology`, which maps Leiden
communities onto the entity/relationship types defined in a domain schema
(e.g. `low-dose-radiation-cancer/ldr_dna_repair/graph_metadata.json`),
giving the graph a second, concept-cluster-level granularity on top of the
entity-level extraction.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from ._op import generate_community_report
from ._utils import EmbeddingFunc, compute_mdhash_id, logger
from .base import TextChunkSchema
from .graphrag import GraphRAG


# ---------------------------------------------------------------------------
# distllm Arrow -> nano-graphrag TextChunkSchema
# ---------------------------------------------------------------------------

def load_distllm_dataset(dataset_path: str | Path):
    """Load a distllm `writer: huggingface` output directory (or a single
    shard) as a `datasets.Dataset`."""
    from datasets import load_from_disk

    ds = load_from_disk(str(dataset_path))
    if hasattr(ds, "keys"):  # DatasetDict -> first split
        ds = ds[next(iter(ds.keys()))]
    return ds


def build_chunks_from_distllm(
    dataset,
    text_field: str = "text",
    embedding_field: str = "embedding",
    path_field: str = "path",
) -> tuple[dict[str, TextChunkSchema], dict[str, dict], dict[str, np.ndarray]]:
    """Convert a distllm semantic-chunk dataset into nano-graphrag's
    `text_chunks` / `full_docs` maps, plus a content -> embedding lookup of
    the precomputed SFR-Mistral embeddings.
    """
    full_docs: dict[str, dict] = {}
    chunks: dict[str, TextChunkSchema] = {}
    embedding_lookup: dict[str, np.ndarray] = {}
    order_by_doc: dict[str, int] = defaultdict(int)

    for row in dataset:
        text = (row.get(text_field) or "").strip()
        if not text:
            continue

        source_path = row.get(path_field) or "unknown"
        full_doc_id = compute_mdhash_id(source_path, prefix="doc-")
        full_docs.setdefault(full_doc_id, {"content": source_path})

        chunk_id = compute_mdhash_id(text, prefix="chunk-")
        chunks[chunk_id] = {
            "tokens": len(text.split()),
            "content": text,
            "full_doc_id": full_doc_id,
            "chunk_order_index": order_by_doc[full_doc_id],
        }
        order_by_doc[full_doc_id] += 1

        embedding = row.get(embedding_field)
        if embedding is not None:
            embedding_lookup[text] = np.asarray(embedding, dtype=np.float32)

    logger.info(
        f"distllm bridge: built {len(chunks)} chunks from {len(full_docs)} documents "
        f"({len(embedding_lookup)} with precomputed SFR-Mistral embeddings)"
    )
    return chunks, full_docs, embedding_lookup


def wrap_precomputed_embeddings(
    embedding_lookup: dict[str, np.ndarray],
    fallback_embedding_func: EmbeddingFunc,
) -> EmbeddingFunc:
    """Serve precomputed distllm/SFR-Mistral chunk embeddings by exact text
    match, falling back to `fallback_embedding_func` for anything distllm
    didn't produce (queries, entity/relationship summaries written by the
    indexing LLM, community-report text used for ontology alignment, ...).
    """

    async def _embed(texts: list[str]) -> np.ndarray:
        hits: list[Optional[np.ndarray]] = [embedding_lookup.get(t) for t in texts]
        missing_idx = [i for i, h in enumerate(hits) if h is None]
        if missing_idx:
            missing_vecs = await fallback_embedding_func.func(
                [texts[i] for i in missing_idx]
            )
            for i, vec in zip(missing_idx, missing_vecs):
                hits[i] = np.asarray(vec, dtype=np.float32)
        return np.stack(hits)

    return EmbeddingFunc(
        embedding_dim=fallback_embedding_func.embedding_dim,
        max_token_size=fallback_embedding_func.max_token_size,
        func=_embed,
    )


# ---------------------------------------------------------------------------
# Insert pre-chunked, pre-embedded distllm output
# ---------------------------------------------------------------------------

async def ainsert_distllm_chunks(
    rag: GraphRAG,
    chunks: dict[str, TextChunkSchema],
    full_docs: dict[str, dict],
) -> None:
    """Insert distllm-produced chunks directly, mirroring `GraphRAG.ainsert`
    but skipping nano-graphrag's own chunker (distllm already performed SFR-
    Mistral semantic chunking) and reusing precomputed embeddings via
    `rag.embedding_func` (see `wrap_precomputed_embeddings`).
    """
    await rag._insert_start()
    try:
        add_doc_keys = await rag.full_docs.filter_keys(list(full_docs.keys()))
        new_docs = {k: v for k, v in full_docs.items() if k in add_doc_keys}

        add_chunk_keys = await rag.text_chunks.filter_keys(list(chunks.keys()))
        new_chunks = {k: v for k, v in chunks.items() if k in add_chunk_keys}
        if not new_chunks:
            logger.warning("All distllm chunks are already in storage")
            return
        logger.info(
            f"[distllm bridge] inserting {len(new_chunks)} pre-chunked, pre-embedded chunks"
        )

        if rag.enable_naive_rag:
            await rag.chunks_vdb.upsert(new_chunks)

        # community reports are rebuilt from scratch on every insert, same as ainsert()
        await rag.community_reports.drop()

        logger.info("[Entity Extraction]...")
        maybe_new_kg = await rag.entity_extraction_func(
            new_chunks,
            knwoledge_graph_inst=rag.chunk_entity_relation_graph,
            entity_vdb=rag.entities_vdb,
            tokenizer_wrapper=rag.tokenizer_wrapper,
            global_config=asdict(rag),
            using_amazon_bedrock=rag.using_amazon_bedrock,
        )
        if maybe_new_kg is None:
            logger.warning("No new entities found")
            return
        rag.chunk_entity_relation_graph = maybe_new_kg

        logger.info("[Community Report]...")
        await rag.chunk_entity_relation_graph.clustering(rag.graph_cluster_algorithm)
        await generate_community_report(
            rag.community_reports,
            rag.chunk_entity_relation_graph,
            rag.tokenizer_wrapper,
            asdict(rag),
        )

        await rag.full_docs.upsert(new_docs)
        await rag.text_chunks.upsert(new_chunks)
    finally:
        await rag._insert_done()


async def ainsert_distllm_dataset(
    rag: GraphRAG,
    dataset_path: str | Path,
    text_field: str = "text",
    embedding_field: str = "embedding",
    path_field: str = "path",
) -> dict[str, np.ndarray]:
    """Convenience wrapper: load a distllm Arrow dataset, build chunks, and
    insert them into `rag`. Returns the content -> embedding lookup so the
    caller can pass it to `wrap_precomputed_embeddings` *before* calling this
    function (see `examples/radiation_biology_neo4j_pipeline.py`).
    """
    dataset = load_distllm_dataset(dataset_path)
    chunks, full_docs, embedding_lookup = build_chunks_from_distllm(
        dataset, text_field=text_field, embedding_field=embedding_field, path_field=path_field
    )
    await ainsert_distllm_chunks(rag, chunks, full_docs)
    return embedding_lookup


# ---------------------------------------------------------------------------
# Concept-cluster / ontology alignment layer
# ---------------------------------------------------------------------------

async def align_communities_to_ontology(
    rag: GraphRAG,
    ontology_schema_path: str | Path,
    write_back_to_graph: bool = True,
) -> dict[str, dict]:
    """Align Leiden communities (concept clusters) to the entity/relationship
    types declared in a `graph_metadata.json` domain schema, e.g.
    `low-dose-radiation-cancer/ldr_dna_repair/graph_metadata.json`.

    For each community, the community-report title+summary is embedded with
    `rag.embedding_func` (SFR-Mistral via `wrap_precomputed_embeddings`'s
    fallback) and matched by cosine similarity against the schema's
    `entity_types` + `relationship_types` descriptions. The best match is
    written back as `ontology_concept` / `ontology_kg_id` node properties
    (entity-level granularity already comes from extraction; this adds the
    concept-cluster / ontology layer on top) and as a JSON summary file in
    `working_dir`.
    """
    schema = json.loads(Path(ontology_schema_path).read_text())
    kg_id = schema.get("kg_id", Path(ontology_schema_path).stem)
    concept_types = schema["schema"]["entity_types"] + schema["schema"].get(
        "relationship_types", []
    )
    concept_names = [c["name"] for c in concept_types]
    concept_texts = [f"{c['name']}: {c['description']}" for c in concept_types]

    concept_vecs = np.asarray(await rag.embedding_func(concept_texts), dtype=np.float32)
    concept_vecs /= np.linalg.norm(concept_vecs, axis=1, keepdims=True)

    communities = await rag.chunk_entity_relation_graph.community_schema()
    community_ids = list(communities.keys())
    reports = await rag.community_reports.get_by_ids(community_ids)

    alignment: dict[str, dict] = {}
    for community_id, community, report in zip(community_ids, communities.values(), reports):
        if report is None:
            continue
        report_json = report.get("report_json", {})
        describe = f"{report_json.get('title', '')}\n{report_json.get('summary', '')}".strip()
        if not describe:
            continue

        vec = np.asarray((await rag.embedding_func([describe]))[0], dtype=np.float32)
        vec /= np.linalg.norm(vec)
        sims = concept_vecs @ vec
        best = int(np.argmax(sims))

        alignment[community_id] = {
            "kg_id": kg_id,
            "ontology_concept": concept_names[best],
            "score": float(sims[best]),
            "title": report_json.get("title", ""),
            "nodes": community["nodes"],
        }

        if write_back_to_graph:
            for node_id in community["nodes"]:
                node = await rag.chunk_entity_relation_graph.get_node(node_id)
                if node is None:
                    continue
                node["ontology_concept"] = concept_names[best]
                node["ontology_kg_id"] = kg_id
                await rag.chunk_entity_relation_graph.upsert_node(node_id, node)

    out_path = Path(rag.working_dir) / f"ontology_alignment_{kg_id}.json"
    out_path.write_text(json.dumps(alignment, indent=2))
    logger.info(
        f"[ontology alignment] mapped {len(alignment)} communities onto '{kg_id}' "
        f"schema, wrote {out_path}"
    )
    return alignment
