"""
_op.py — backward-compatibility re-export shim.

The original monolithic 1 300-line file has been split into three focused
modules:

  _extraction.py  — chunking, entity/relation parsing, graph upserts
  _reporting.py   — community-report generation
  _querying.py    — local / global / naive query pipelines

All public names remain importable from `nano_graphrag._op` so that existing
call-sites (graphrag.py, tests, external scripts) continue to work without
modification.
"""

from ._extraction import (
    FALLBACK_STOPWORDS,
    _heuristic_extract_records,
    chunking_by_token_size,
    chunking_by_seperators,
    get_chunks,
    _handle_entity_relation_summary,
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _merge_nodes_then_upsert,
    _merge_edges_then_upsert,
    extract_entities,
)

from ._reporting import (
    _fallback_community_report,
    _pack_single_community_by_sub_communities,
    _pack_single_community_describe,
    _community_report_json_to_str,
    generate_community_report,
)

from ._querying import (
    _find_most_related_community_from_entities,
    _find_most_related_text_unit_from_entities,
    _find_most_related_edges_from_entities,
    _build_local_query_context,
    local_query,
    _map_global_communities,
    global_query,
    naive_query,
)

__all__ = [
    # extraction
    "FALLBACK_STOPWORDS",
    "_heuristic_extract_records",
    "chunking_by_token_size",
    "chunking_by_seperators",
    "get_chunks",
    "_handle_entity_relation_summary",
    "_handle_single_entity_extraction",
    "_handle_single_relationship_extraction",
    "_merge_nodes_then_upsert",
    "_merge_edges_then_upsert",
    "extract_entities",
    # reporting
    "_fallback_community_report",
    "_pack_single_community_by_sub_communities",
    "_pack_single_community_describe",
    "_community_report_json_to_str",
    "generate_community_report",
    # querying
    "_find_most_related_community_from_entities",
    "_find_most_related_text_unit_from_entities",
    "_find_most_related_edges_from_entities",
    "_build_local_query_context",
    "local_query",
    "_map_global_communities",
    "global_query",
    "naive_query",
]
