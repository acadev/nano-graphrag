"""
End-to-end pilot pipeline for the low-dose-radiation-cancer corpus:

  pdfwf JSONL --(distllm, SFR-Embedding-Mistral semantic chunking)-->
  HF Arrow shards --(_distllm_bridge)--> nano-graphrag (Neo4j graph storage)
  --(align_communities_to_ontology)--> concept-cluster / ontology layer

Prereqs:
  1. Run pdfwf over the radiation-biology PDFs to produce JSONL.
  2. Run `distllm embed --config configs/aurora_sfr_mistral_radiation_biology.yaml`
     to produce the HF Arrow embedding shards referenced by DISTLLM_DATASET below.
  3. Start Neo4j 5.x with the GDS plugin installed (see docs/use_neo4j_for_graphrag.md)
     and export NEO4J_URL / NEO4J_USER / NEO4J_PASSWORD.

This script processes a single domain sub-corpus (e.g. ldr_dna_repair); run it
once per sub-domain in `low-dose-radiation-cancer/`, each with its own
`working_dir` (-> its own Neo4j label namespace) and its own
`graph_metadata.json` for ontology alignment.
"""
import asyncio
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from nano_graphrag import GraphRAG
from nano_graphrag._distllm_bridge import (
    align_communities_to_ontology,
    build_chunks_from_distllm,
    load_distllm_dataset,
    wrap_precomputed_embeddings,
    ainsert_distllm_chunks,
)
from nano_graphrag._storage import Neo4jStorage
from nano_graphrag._utils import wrap_embedding_func_with_attrs

# ---------------------------------------------------------------------------
# Configuration — adjust per sub-domain
# ---------------------------------------------------------------------------
DOMAIN = "ldr_dna_repair"
DISTLLM_DATASET = f"/lus/flare/projects/<PROJECT>/radiation_biology/embeddings/sfr_mistral/{DOMAIN}"
ONTOLOGY_SCHEMA = f"low-dose-radiation-cancer/{DOMAIN}/graph_metadata.json"
WORKING_DIR = f"./nano_graphrag_cache_{DOMAIN}"

SFR_MODEL_NAME = "Salesforce/SFR-Embedding-Mistral"


# ---------------------------------------------------------------------------
# SFR-Embedding-Mistral fallback embedder.
#
# distllm precomputes chunk embeddings, so this is only invoked for text the
# distllm pipeline never saw: query strings at retrieval time, and the
# entity/relationship/community-report summaries the indexing LLM writes
# during graph construction. Using the same encoder keeps everything in one
# embedding space.
# ---------------------------------------------------------------------------
_tokenizer = AutoTokenizer.from_pretrained(SFR_MODEL_NAME)
_model = AutoModel.from_pretrained(SFR_MODEL_NAME)
_model.eval()


def _last_token_pool(last_hidden_state, attention_mask):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths
    ]


@wrap_embedding_func_with_attrs(embedding_dim=4096, max_token_size=4096)
async def sfr_mistral_embedding(texts: list[str]) -> np.ndarray:
    batch = _tokenizer(
        texts, max_length=4096, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = _model(**batch)
    embeddings = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy()


async def main() -> None:
    # 1. Load distllm's SFR-Mistral semantic chunks + precomputed embeddings
    dataset = load_distllm_dataset(DISTLLM_DATASET)
    chunks, full_docs, embedding_lookup = build_chunks_from_distllm(dataset)

    # 2. Serve those embeddings to nano-graphrag, falling back to a live
    #    SFR-Mistral call for anything distllm didn't produce.
    embedding_func = wrap_precomputed_embeddings(embedding_lookup, sfr_mistral_embedding)

    # 3. Wire up Neo4j as the entity-level graph store. Each domain
    #    sub-corpus gets its own working_dir, which Neo4jStorage uses as part
    #    of its node-label namespace, so the four LDR sub-graphs (and
    #    eventually the full radiation-biology corpus) coexist in one
    #    Neo4j instance without colliding.
    neo4j_config = {
        "neo4j_url": os.environ.get("NEO4J_URL", "neo4j://localhost:7687"),
        "neo4j_auth": (
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "neo4j"),
        ),
    }
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        graph_storage_cls=Neo4jStorage,
        addon_params=neo4j_config,
        embedding_func=embedding_func,
    )

    # 4. Insert pre-chunked, pre-embedded passages: skips nano-graphrag's own
    #    chunker, runs entity/relationship extraction, Leiden clustering
    #    (gds.leiden, via Neo4jStorage.clustering), and community reports.
    await ainsert_distllm_chunks(rag, chunks, full_docs)

    # 5. Concept-cluster / ontology-alignment layer: map each Leiden
    #    community onto the entity/relationship types declared in this
    #    domain's graph_metadata.json (DNA_LESION, REPAIR_PATHWAY,
    #    REPAIR_PROTEIN, DOSE_RATE, CELL_TYPE, FIDELITY, ...), writing
    #    `ontology_concept` / `ontology_kg_id` back onto each node in Neo4j.
    await align_communities_to_ontology(rag, ONTOLOGY_SCHEMA)


if __name__ == "__main__":
    asyncio.run(main())
