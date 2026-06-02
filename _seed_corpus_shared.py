"""
Shared runner logic for seed_haiqu_corpus.py and seed_evodevo_corpus.py.

Both scripts define their own GROUPS list (corpus-specific queries) and then
delegate all I/O, deduplication, and argparse handling to the functions here.

Public API:
    SearchQuery   — one Firecrawl /v1/search call
    PaperGroup    — one group: name, question, schema, list of queries
    run_main()    — call from __main__ of each seed script
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from paper_fetching.firecrawl_client import (  # noqa: E402
    SCIENTIFIC_DOMAINS,
    EXCLUDED_DOMAINS,
    extract_text_from_result,
)

FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v1/search"


# ---------------------------------------------------------------------------
# Data classes (re-exported so seed scripts can import from here)
# ---------------------------------------------------------------------------

@dataclass
class SearchQuery:
    """One Firecrawl /search call within a group.

    ``tbs`` is Google's time-based filter passed straight through:
        qdr:m  → past month
        qdr:y  → past year
    """
    query: str
    tbs: Optional[str] = None
    notes: str = ""


@dataclass
class PaperGroup:
    """One corpus group: a driving question, a schema name, and search queries."""
    name: str       # folder-safe slug (also the schema name)
    question: str   # the scientific question this group answers
    schema: str     # exactly one domain_schema name
    queries: List[SearchQuery] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Firecrawl helpers
# ---------------------------------------------------------------------------

def _run_search(
    api_key: str,
    q: SearchQuery,
    max_results: int,
    only_main_content: bool = False,
) -> List[dict]:
    """POST /v1/search; return results filtered to scientific domains."""
    payload: dict = {
        "query": q.query,
        "limit": max_results,
        "scrapeOptions": {"formats": ["markdown"]},
    }
    if only_main_content:
        payload["scrapeOptions"]["onlyMainContent"] = True
    if q.tbs:
        payload["tbs"] = q.tbs
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        r = requests.post(FIRECRAWL_SEARCH_URL, headers=headers,
                          json=payload, timeout=180)
        r.raise_for_status()
        data = r.json().get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"     ! Firecrawl error: {e}")
        return []

    return [
        hit for hit in data
        if not any(d in (hit.get("url") or "").lower() for d in EXCLUDED_DOMAINS)
        and any(d in (hit.get("url") or "").lower() for d in SCIENTIFIC_DOMAINS)
    ]


def _save_paper(
    result: dict,
    dest_dir: Path,
    source_query: str,
    tbs: Optional[str],
) -> dict:
    """Write the paper's markdown body to disk; return its metadata record."""
    paper_uuid = str(uuid.uuid4())
    body = extract_text_from_result(result, format="markdown")
    out_path = dest_dir / f"{paper_uuid}.md"
    out_path.write_text(body, encoding="utf-8")
    md = result.get("metadata") or {}
    return {
        "uuid": paper_uuid,
        "title": md.get("title") or result.get("title", "(unknown)"),
        "url": result.get("url", ""),
        "description": md.get("description", ""),
        "language": md.get("language", ""),
        "source_query": source_query,
        "tbs": tbs,
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
        "content_file": out_path.name,
        "content_chars": len(body),
    }


def _run_group(
    g: PaperGroup,
    api_key: Optional[str],
    root: Path,
    max_per_query: int,
    dry_run: bool,
    only_main_content: bool = False,
) -> dict:
    print(f"\n=== {g.name} ===")
    print(f"  Q: {g.question}")
    print(f"  schema: {g.schema}")
    print(f"  queries: {len(g.queries)}")

    if dry_run:
        for i, q in enumerate(g.queries, 1):
            tag = f" [tbs={q.tbs}]" if q.tbs else ""
            print(f"    {i:>2}.{tag} {q.query[:100]}{'...' if len(q.query) > 100 else ''}")
        return {"group": g.name, "dry_run": True}

    group_dir = root / g.name
    papers_dir = group_dir / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    seen_urls: set = set()
    saved: List[dict] = []
    queries_log: List[dict] = []

    for i, q in enumerate(g.queries, 1):
        head = q.query[:80] + ("..." if len(q.query) > 80 else "")
        print(f"  [{i}/{len(g.queries)}] {head}")
        hits = _run_search(api_key, q, max_per_query, only_main_content=only_main_content)
        kept = 0
        for h in hits:
            u = (h.get("url") or "").strip()
            if not u or u in seen_urls:
                continue
            seen_urls.add(u)
            saved.append(_save_paper(h, papers_dir, q.query, q.tbs))
            kept += 1
        queries_log.append({
            "query": q.query,
            "tbs": q.tbs,
            "notes": q.notes,
            "raw_hits": len(hits),
            "saved_unique": kept,
        })
        print(f"     hits={len(hits)}  saved_new={kept}  running_total={len(saved)}")
        time.sleep(1)  # be gentle to the API

    metadata = {
        "group": g.name,
        "question": g.question,
        "schema": g.schema,
        "ran_at": datetime.utcnow().isoformat() + "Z",
        "max_per_query": max_per_query,
        "queries": queries_log,
        "paper_count": len(saved),
        "papers": saved,
    }
    (group_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  -> {group_dir / 'metadata.json'} ({len(saved)} unique papers)")
    return {"group": g.name, "paper_count": len(saved)}


# ---------------------------------------------------------------------------
# Entry point called by each seed script
# ---------------------------------------------------------------------------

def run_main(
    groups: List[PaperGroup],
    default_output_dir: str,
    default_max_per_query: int = 10,
    only_main_content: bool = False,
) -> None:
    """Parse CLI args, run Firecrawl searches, write corpus to disk.

    Args:
        groups:               The corpus-specific GROUPS list.
        default_output_dir:   Default value for --output-dir.
        default_max_per_query: Default --max-per-query value.
        only_main_content:    Pass ``onlyMainContent: True`` to Firecrawl
                              (useful for evo-devo where pages are heavy).
    """
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key", default=os.getenv("FIRECRAWL_API_KEY"),
                   help="Firecrawl API key (or set FIRECRAWL_API_KEY)")
    p.add_argument("--output-dir", default=default_output_dir,
                   help=f"Root output directory (default: {default_output_dir})")
    p.add_argument("--max-per-query", type=int, default=default_max_per_query,
                   help=f"Max results per Firecrawl query (default: {default_max_per_query})")
    p.add_argument("--groups", nargs="*",
                   help="Only run these group names (default: all)")
    p.add_argument("--list-groups", action="store_true",
                   help="List available groups and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="Print queries without making API calls")
    args = p.parse_args()

    if args.list_groups:
        for g in groups:
            print(f"{g.name:<48} schema: {g.schema:<40} {len(g.queries)} queries")
            print(f"  Q: {g.question}")
        return

    selected = groups
    if args.groups:
        wanted = set(args.groups)
        selected = [g for g in groups if g.name in wanted]
        unknown = wanted - {g.name for g in groups}
        if unknown:
            sys.exit(f"Unknown group(s): {sorted(unknown)}")
        if not selected:
            sys.exit(f"ERROR: no groups matched: {args.groups}")

    if not args.dry_run and not args.api_key:
        sys.exit("ERROR: FIRECRAWL_API_KEY not set and --api-key not provided")

    root = Path(args.output_dir)
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    print(f"output_dir   : {root}")
    print(f"max/query    : {args.max_per_query}")
    print(f"groups       : {len(selected)}")
    print(f"dry_run      : {args.dry_run}")

    summaries = [
        _run_group(g, args.api_key, root, args.max_per_query,
                   args.dry_run, only_main_content=only_main_content)
        for g in selected
    ]

    if not args.dry_run:
        index = {
            "ran_at": datetime.utcnow().isoformat() + "Z",
            "output_dir": str(root),
            "max_per_query": args.max_per_query,
            "groups": [
                {"group": r["group"], "paper_count": r.get("paper_count", 0)}
                for r in summaries
            ],
        }
        (root / "corpus_index.json").write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )
        total = sum(r.get("paper_count", 0) for r in summaries)
        print(f"\nDone. {total} papers across {len(summaries)} groups.")
        print(f"Index: {root / 'corpus_index.json'}")
