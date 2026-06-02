"""
Build per-group knowledge graphs from any corpus produced by a seed_*_corpus.py
script.

Replaces the two near-identical scripts:
  - build_haiqu_graphs.py   (pointed at data/haiqu_corpus)
  - build_evodevo_graphs.py (pointed at data/evo_devo_corpus)

For each group folder under <corpus-dir>/<group>/ this script:
  1. Loads domain_schemas/<group>.yaml as the typed extraction schema.
  2. Reads metadata.json to enumerate papers (<group>/papers/<uuid>.md).
  3. Runs the typed entity/relationship extractor (up to 3 LLM calls/chunk
     with self-refine on) over each paper.
  4. Merges entities and relationships into a single per-group NetworkX graph
     using graph_enrichment.graph_merger.
  5. Writes <output-dir>/<group>/<group>_graph.graphml plus a state file to
     allow resuming a partially-completed run.

Usage:
    # HAIQU corpus (default paths match the old build_haiqu_graphs.py)
    python build_domain_graphs.py --corpus-dir data/haiqu_corpus \\
                                  --output-dir haiqu_graphs

    # Evo-devo corpus (default paths match the old build_evodevo_graphs.py)
    python build_domain_graphs.py --corpus-dir data/evo_devo_corpus \\
                                  --output-dir evo_devo_graphs/v1

    # Smoke test: 3 papers, one group
    python build_domain_graphs.py --corpus-dir data/haiqu_corpus \\
                                  --groups haiqu_biosensor_detection \\
                                  --limit-papers 3

    export LLM_API_KEY=...
    export LLM_ENDPOINT=https://apps-dev.inside.anl.gov/argoapi/v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from domain_schemas.schema_loader import load_domain_schema
from nano_graphrag.entity_extraction.typed_module import (
    create_domain_extractor_from_schema,
)
from graph_enrichment.graph_merger import (
    add_entities_to_graph,
    add_relationships_to_graph,
)
from create_domain_typed_graph import chunk_text, extract_from_chunk
from gasl.llm import ArgoBridgeLLM
from graph_metadata import metadata_from_schema_and_corpus, save_graph_metadata
from nano_graphrag.graph_slots import get_salience_score, set_salience_score


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GroupResult:
    group: str
    schema: str
    papers_attempted: int
    papers_succeeded: int
    papers_skipped: int
    papers_failed: int
    nodes: int
    edges: int
    output_path: str
    elapsed_sec: float


# ---------------------------------------------------------------------------
# Corpus discovery & metadata helpers
# ---------------------------------------------------------------------------

def discover_groups(corpus_dir: Path) -> List[str]:
    """List <group> subdirectories that contain a metadata.json + papers/."""
    return [
        p.name
        for p in sorted(corpus_dir.iterdir())
        if p.is_dir()
        and (p / "metadata.json").exists()
        and (p / "papers").is_dir()
    ]


def load_group_metadata(corpus_dir: Path, group: str) -> dict:
    with open(corpus_dir / group / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Graph serialisation helpers
# ---------------------------------------------------------------------------

def _serialize_graph_for_graphml(g: nx.DiGraph) -> nx.DiGraph:
    """GraphML only accepts scalar attrs — flatten lists and arbitrary objects."""
    g = g.copy()
    for node in g.nodes():
        for k, v in list(g.nodes[node].items()):
            if isinstance(v, list):
                g.nodes[node][k] = ",".join(str(x) for x in v)
            elif not isinstance(v, (str, int, float, bool, type(None))):
                g.nodes[node][k] = str(v)
    for src, tgt in g.edges():
        for k, v in list(g.edges[src, tgt].items()):
            if isinstance(v, list):
                g.edges[src, tgt][k] = ",".join(str(x) for x in v)
            elif not isinstance(v, (str, int, float, bool, type(None))):
                g.edges[src, tgt][k] = str(v)
    return g


def save_graph(g: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(_serialize_graph_for_graphml(g), path)


# ---------------------------------------------------------------------------
# Resume state helpers
# ---------------------------------------------------------------------------

def load_state(state_path: Path) -> dict:
    if state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_uuids": []}


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(state_path)


# ---------------------------------------------------------------------------
# Per-paper extraction
# ---------------------------------------------------------------------------

async def _extract_one_chunk(
    i: int,
    chunk: str,
    paper_uuid: str,
    extractor,
    semaphore: Optional[asyncio.Semaphore],
) -> Tuple[Dict[str, Dict], List[Dict]]:
    chunk_id = f"{paper_uuid}_chunk_{i}"
    local_entities: Dict[str, Dict] = {}
    local_rels: List[Dict] = []
    try:
        if semaphore is not None:
            async with semaphore:
                await extract_from_chunk(chunk, chunk_id, extractor, local_entities, local_rels)
        else:
            await extract_from_chunk(chunk, chunk_id, extractor, local_entities, local_rels)
    except Exception as e:
        print(f"    ! chunk {i} failed: {e}")
    return local_entities, local_rels


async def extract_paper(
    text: str,
    paper_uuid: str,
    extractor,
    chunk_size: int,
    overlap: int,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """Chunk a paper, run the typed extractor on all chunks concurrently.

    Returns a merged-by-name entity dict and a flat relationship list.
    Entity merging keeps the highest salience score and accumulates source
    chunks — matching the more thorough strategy from build_haiqu_graphs.py.
    """
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    results = await asyncio.gather(
        *[_extract_one_chunk(i, c, paper_uuid, extractor, semaphore) for i, c in enumerate(chunks)]
    )

    entities: Dict[str, Dict] = {}
    relationships: List[Dict] = []
    for chunk_entities, chunk_rels in results:
        for name, data in chunk_entities.items():
            if name not in entities:
                entities[name] = data
            else:
                # Keep the highest salience score seen across chunks.
                existing_score = get_salience_score(entities[name], 0.0)
                new_score = get_salience_score(data, 0.0)
                if new_score > existing_score:
                    set_salience_score(entities[name], new_score)
                # Accumulate source chunk references.
                for sc in data.get("source_chunks", []):
                    if sc not in entities[name].get("source_chunks", []):
                        entities[name].setdefault("source_chunks", []).append(sc)
        relationships.extend(chunk_rels)

    return entities, relationships


# ---------------------------------------------------------------------------
# Per-group graph build
# ---------------------------------------------------------------------------

async def build_group(
    group: str,
    corpus_dir: Path,
    output_dir: Path,
    model: str,
    chunk_size: int,
    overlap: int,
    refine_turns: int,
    self_refine: bool,
    similarity_threshold: float,
    auto_merge: bool,
    limit_papers: Optional[int],
    min_paper_length: int,
    max_paper_length: Optional[int],
    resume: bool,
    save_every: int,
    chunk_concurrency: Optional[int],
) -> GroupResult:
    t0 = time.time()
    group_dir = corpus_dir / group
    metadata = load_group_metadata(corpus_dir, group)
    schema_name = metadata.get("schema") or group
    papers_meta: List[dict] = metadata.get("papers", [])
    if limit_papers is not None:
        papers_meta = papers_meta[:limit_papers]

    print(f"\n{'='*72}\nGROUP: {group}\n{'='*72}")
    print(f"  question : {metadata.get('question', '(none)')}")
    print(f"  schema   : {schema_name}")
    print(f"  papers   : {len(papers_meta)} (limit={limit_papers})")

    schema = load_domain_schema(schema_name)
    print(f"  entity_types       : {len(schema.entity_types)}")
    print(f"  relationship_types : {len(schema.relationship_types)}")

    llm = ArgoBridgeLLM(model=model)
    extractor = create_domain_extractor_from_schema(
        schema,
        llm_func=llm.call_async,
        num_refine_turns=refine_turns,
        self_refine=self_refine,
    )
    semaphore = asyncio.Semaphore(chunk_concurrency) if chunk_concurrency else None

    # Output paths — each group gets its own subdirectory.
    group_out_dir = output_dir / group
    group_out_dir.mkdir(parents=True, exist_ok=True)
    out_graph = group_out_dir / f"{group}_graph.graphml"
    state_path = group_out_dir / f"{group}_state.json"
    state = load_state(state_path) if resume else {"completed_uuids": []}
    completed_uuids: set = set(state.get("completed_uuids", []))

    # Optionally resume from an existing partial graph.
    if resume and out_graph.exists() and completed_uuids:
        graph = nx.read_graphml(out_graph)
        if not isinstance(graph, nx.DiGraph):
            graph = nx.DiGraph(graph)
        print(
            f"  resumed : {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges, {len(completed_uuids)} papers done"
        )
    else:
        graph = nx.DiGraph()

    succeeded = skipped = failed = attempted = 0

    for idx, paper in enumerate(papers_meta, 1):
        uuid = paper.get("uuid", "")
        if not uuid or uuid in completed_uuids:
            continue

        title = (paper.get("title") or "(untitled)")[:80]
        content_file = paper.get("content_file", f"{uuid}.md")
        path = group_dir / "papers" / content_file

        if not path.exists():
            print(f"  [{idx}/{len(papers_meta)}] MISSING: {path}")
            failed += 1
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  [{idx}/{len(papers_meta)}] READ ERROR: {e}")
            failed += 1
            continue

        if len(text) < min_paper_length:
            print(f"  [{idx}/{len(papers_meta)}] SKIP short ({len(text)} chars): {title}")
            skipped += 1
            completed_uuids.add(uuid)  # skip on subsequent runs too
            continue

        if max_paper_length is not None and len(text) > max_paper_length:
            print(f"  [{idx}/{len(papers_meta)}] SKIP oversized ({len(text)} chars): {title}")
            skipped += 1
            completed_uuids.add(uuid)
            continue

        attempted += 1
        print(f"  [{idx}/{len(papers_meta)}] {title} ({len(text)} chars)")
        try:
            entities, relationships = await extract_paper(
                text, uuid, extractor,
                chunk_size=chunk_size,
                overlap=overlap,
                semaphore=semaphore,
            )
            graph, name_mapping = add_entities_to_graph(
                graph, entities, uuid,
                similarity_threshold=similarity_threshold,
                auto_merge=auto_merge,
            )
            graph = add_relationships_to_graph(graph, relationships, name_mapping, uuid)
            succeeded += 1
            completed_uuids.add(uuid)
        except Exception as e:
            print(f"    ! extraction failed: {e}")
            traceback.print_exc()
            failed += 1
            continue

        # Periodic checkpoint so a long run isn't all-or-nothing.
        if save_every and succeeded % save_every == 0:
            save_graph(graph, out_graph)
            save_state(state_path, {
                "completed_uuids": sorted(completed_uuids),
                "model": model,
                "schema": schema_name,
                "last_checkpoint_paper": uuid,
            })
            print(
                f"    checkpoint: {graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges, {len(completed_uuids)} papers done"
            )

    # Final write.
    save_graph(graph, out_graph)
    save_state(state_path, {
        "completed_uuids": sorted(completed_uuids),
        "model": model,
        "schema": schema_name,
        "completed": True,
        "papers_succeeded": succeeded,
        "papers_failed": failed,
        "papers_skipped": skipped,
    })

    # Domain expertise metadata alongside the graph.
    try:
        gm = metadata_from_schema_and_corpus(
            kg_id=group,
            kg_version=output_dir.name,
            schema=schema,
            corpus_metadata=metadata,
        )
        meta_path = save_graph_metadata(group_out_dir, gm)
        print(f"  → wrote {meta_path.name}")
    except Exception as exc:
        print(f"  ! graph_metadata write failed (non-fatal): {exc}")

    elapsed = time.time() - t0
    print(f"\n  → {out_graph}")
    print(f"  nodes: {graph.number_of_nodes()}  edges: {graph.number_of_edges()}")
    print(
        f"  ok={succeeded}  skip={skipped}  fail={failed}  elapsed={elapsed:.1f}s"
        f"  llm_calls={llm.usage.get('calls', 0)}  tokens={llm.usage.get('total_tokens', 0)}"
    )
    return GroupResult(
        group=group,
        schema=schema_name,
        papers_attempted=attempted,
        papers_succeeded=succeeded,
        papers_skipped=skipped,
        papers_failed=failed,
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        output_path=str(out_graph),
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# Async main + CLI
# ---------------------------------------------------------------------------

async def amain(args: argparse.Namespace) -> None:
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    available = discover_groups(corpus_dir)

    if args.list_groups:
        for g in available:
            md = load_group_metadata(corpus_dir, g)
            print(
                f"{g:<40} schema={md.get('schema', '?'):<40} "
                f"papers={len(md.get('papers', []))}"
            )
        return

    if not available:
        sys.exit(f"No groups found in {corpus_dir}")

    if args.groups:
        unknown = [g for g in args.groups if g not in available]
        if unknown:
            sys.exit(f"Unknown group(s): {unknown}. Available: {available}")
        groups = args.groups
    else:
        groups = available

    print(f"corpus_dir : {corpus_dir}")
    print(f"output_dir : {output_dir}")
    print(f"model      : {args.model}")
    print(f"groups     : {groups}")
    print(f"limit/grp  : {args.limit_papers}")

    if args.dry_run:
        for g in groups:
            meta = load_group_metadata(corpus_dir, g)
            print(f"  {g}: {meta.get('paper_count', len(meta.get('papers', [])))} papers, "
                  f"schema={meta.get('schema', g)}")
        return

    results: List[GroupResult] = []
    for g in groups:
        try:
            r = await build_group(
                group=g,
                corpus_dir=corpus_dir,
                output_dir=output_dir,
                model=args.model,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                refine_turns=args.refine_turns,
                self_refine=not args.no_self_refine,
                similarity_threshold=args.similarity_threshold,
                auto_merge=not args.no_auto_merge,
                limit_papers=args.limit_papers,
                min_paper_length=args.min_paper_length,
                max_paper_length=args.max_paper_length,
                resume=not args.no_resume,
                save_every=args.save_every,
                chunk_concurrency=args.chunk_concurrency,
            )
            results.append(r)
        except Exception as e:
            print(f"\n!! GROUP {g} FAILED: {e}")
            traceback.print_exc()

    # Summary JSON.
    summary_path = output_dir / "build_summary.json"
    summary = {
        "model": args.model,
        "corpus_dir": str(corpus_dir),
        "results": [r.__dict__ for r in results],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*72}\nSUMMARY  →  {summary_path}\n{'='*72}")
    print(f"{'group':<40} {'ok':>4} {'skip':>4} {'fail':>4} {'nodes':>6} {'edges':>6} {'sec':>7}")
    for r in results:
        print(
            f"{r.group:<40} {r.papers_succeeded:>4} {r.papers_skipped:>4} "
            f"{r.papers_failed:>4} {r.nodes:>6} {r.edges:>6} {r.elapsed_sec:>7.1f}s"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--corpus-dir", required=True,
        help="Root of the corpus produced by a seed_*_corpus.py script",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Where to write per-group graphml + state files",
    )
    p.add_argument(
        "--groups", nargs="*",
        help="Only build these groups (default: all discovered)",
    )
    p.add_argument("--list-groups", action="store_true",
                   help="List available groups and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="List groups and paper counts without building")
    p.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt55"),
                   help="LLM model name passed to ArgoBridgeLLM (default: gpt55)")
    p.add_argument("--limit-papers", type=int, default=None,
                   help="Max papers per group (useful for smoke tests)")
    p.add_argument("--min-paper-length", type=int, default=500,
                   help="Skip papers shorter than this many chars (default: 500)")
    p.add_argument("--max-paper-length", type=int, default=None,
                   help="Skip papers longer than this many chars (default: unlimited)")
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--refine-turns", type=int, default=1)
    p.add_argument("--no-self-refine", action="store_true",
                   help="Disable critique→refine loop (1 LLM call/chunk instead of 3)")
    p.add_argument("--similarity-threshold", type=float, default=0.85,
                   help="Entity-merge similarity threshold 0..1 (default: 0.85)")
    p.add_argument("--no-auto-merge", action="store_true",
                   help="Disable automatic entity merging")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any existing state file / partial graph")
    p.add_argument("--save-every", type=int, default=10,
                   help="Checkpoint every N successful papers (default: 10)")
    p.add_argument("--chunk-concurrency", type=int, default=None,
                   help="Max concurrent LLM calls per paper (default: unlimited)")
    args = p.parse_args()

    if "--verbose" not in sys.argv:
        logging.getLogger("nano-graphrag").setLevel(logging.WARNING)

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
