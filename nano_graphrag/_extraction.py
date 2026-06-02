"""
Entity extraction, chunking, and graph-node/edge merging operations.

Extracted from the monolithic _op.py so each concern lives in its own module:
  _extraction.py  — chunking, entity/relation parsing, graph upserts
  _reporting.py   — community-report generation
  _querying.py    — local / global / naive query pipelines
"""
from __future__ import annotations

import re
import asyncio
from collections import Counter, defaultdict
from itertools import combinations
from typing import Union

from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    TokenizerWrapper,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


# ---------------------------------------------------------------------------
# Heuristic fallback (used when LLM entity extraction fails)
# ---------------------------------------------------------------------------

FALLBACK_STOPWORDS = {
    "The", "A", "An", "This", "That", "These", "Those", "He", "She", "It", "They",
    "His", "Her", "Their", "Its", "We", "I", "You", "Mr", "Mrs", "Ms", "Dr",
    "In", "On", "At", "For", "From", "To", "And", "But", "Or", "If", "Then",
    "As", "Of", "By", "With", "Without", "After", "Before", "During", "While",
}


def _heuristic_extract_records(
    content: str, chunk_key: str
) -> tuple[dict[str, list[dict]], dict[tuple[str, str], list[dict]]]:
    """Best-effort offline fallback when LLM-based indexing is unavailable.

    Extract title-case spans as entities and connect entities that co-occur in the
    same sentence.  Intentionally simple — only used when the indexing LLM fails so
    the graph remains usable rather than aborting the whole insert.
    """
    maybe_nodes: dict[str, list[dict]] = defaultdict(list)
    maybe_edges: dict[tuple[str, str], list[dict]] = defaultdict(list)
    sentence_spans = re.split(r"(?<=[.!?])\s+|\n{2,}", content)
    candidate_freq: Counter[str] = Counter()
    sentence_candidates: list[tuple[str, list[str]]] = []
    for sentence in sentence_spans:
        candidates = re.findall(
            r"\b(?:[A-Z][a-z]+(?:[-'][A-Za-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Za-z]+)?)*)\b",
            sentence,
        )
        normalized_names: list[str] = []
        for candidate in candidates:
            name = clean_str(candidate.upper())
            if not name:
                continue
            if candidate in FALLBACK_STOPWORDS:
                continue
            if len(candidate) < 3:
                continue
            candidate_freq[name] += 1
            normalized_names.append(name)
        if normalized_names:
            sentence_candidates.append((sentence, normalized_names))

    top_names = {
        name
        for name, _ in sorted(
            candidate_freq.items(), key=lambda kv: (-kv[1], kv[0])
        )[:12]
    }
    for sentence, names in sentence_candidates:
        sentence_entities = [name for name in names if name in top_names]
        for name in sorted(set(sentence_entities)):
            maybe_nodes[name].append(
                {
                    "entity_name": name,
                    "entity_type": '"UNKNOWN"',
                    "description": clean_str(sentence[:240]),
                    "source_id": chunk_key,
                }
            )
        for src, tgt in list(combinations(sorted(set(sentence_entities)), 2))[:24]:
            maybe_edges[(src, tgt)].append(
                {
                    "src_id": src,
                    "tgt_id": tgt,
                    "weight": 1.0,
                    "description": clean_str(sentence[:240]),
                    "source_id": chunk_key,
                }
            )
    return dict(maybe_nodes), dict(maybe_edges)


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tokenizer_wrapper: TokenizerWrapper,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) -> list[dict]:
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))
        chunk_texts = tokenizer_wrapper.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_texts):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )
    return results


def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tokenizer_wrapper: TokenizerWrapper,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) -> list[dict]:
    from .prompt import PROMPTS  # local import to avoid circular at module load

    separators = [tokenizer_wrapper.encode(s) for s in PROMPTS["default_text_separator"]]
    splitter = SeparatorSplitter(
        separators=separators,
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_tokens = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_tokens]
        decoded_chunks = tokenizer_wrapper.decode_batch(chunk_tokens)
        for i, chunk in enumerate(decoded_chunks):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )
    return results


def get_chunks(
    new_docs,
    chunk_func=chunking_by_token_size,
    tokenizer_wrapper: TokenizerWrapper = None,
    **chunk_func_params,
) -> dict:
    inserting_chunks = {}
    new_docs_list = list(new_docs.items())
    docs = [item[1]["content"] for item in new_docs_list]
    doc_keys = [item[0] for item in new_docs_list]
    tokens = [tokenizer_wrapper.encode(doc) for doc in docs]
    chunks = chunk_func(
        tokens,
        doc_keys=doc_keys,
        tokenizer_wrapper=tokenizer_wrapper,
        overlap_token_size=chunk_func_params.get("overlap_token_size", 128),
        max_token_size=chunk_func_params.get("max_token_size", 1024),
    )
    for chunk in chunks:
        inserting_chunks[compute_mdhash_id(chunk["content"], prefix="chunk-")] = chunk
    return inserting_chunks


# ---------------------------------------------------------------------------
# Entity / relationship parsing helpers
# ---------------------------------------------------------------------------

async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
    tokenizer_wrapper: TokenizerWrapper,
) -> str:
    use_llm_func: callable = global_config["cheap_model_func"]
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = tokenizer_wrapper.encode(description)
    if len(tokens) < summary_max_tokens:
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = tokenizer_wrapper.decode(tokens[:llm_max_tokens])
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    try:
        summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
        return summary
    except Exception as exc:
        logger.warning(
            "Entity/relation summary LLM failed for %s, using truncated description: %s",
            entity_or_relation_name,
            exc,
        )
        return use_description


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
) -> dict | None:
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=chunk_key,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
) -> dict | None:
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=chunk_key,
    )


# ---------------------------------------------------------------------------
# Graph upsert helpers (merge then write)
# ---------------------------------------------------------------------------

async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    tokenizer_wrapper: TokenizerWrapper,
) -> dict:
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knwoledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config, tokenizer_wrapper
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knwoledge_graph_inst.upsert_node(entity_name, node_data=node_data)
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    tokenizer_wrapper: TokenizerWrapper,
) -> None:
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knwoledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knwoledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config, tokenizer_wrapper
    )
    await knwoledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    tokenizer_wrapper: TokenizerWrapper,
    global_config: dict,
    using_amazon_bedrock: bool = False,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]

    # Dynamic entity-type generation per user task (query-aware).
    # Falls back to default entity types when the LLM path is unavailable.
    from .entity_type_generator import generate_entity_types_for_chunks
    from .prompt_system import QueryAwarePromptSystem

    task_text = (global_config.get("user_query", "") or "").strip()
    generated_entity_types = []
    if task_text:
        try:
            generated_entity_types = await generate_entity_types_for_chunks(
                task=task_text,
                chunks=chunks,
                llm_func=use_llm_func,
                per_chunk=True,
            )
        except Exception as exc:
            logger.warning(
                "Dynamic entity type generation failed, using defaults: %s", exc
            )
    if not generated_entity_types:
        generated_entity_types = QueryAwarePromptSystem()._static_prompts[
            "DEFAULT_ENTITY_TYPES"
        ]
    entity_types_value = ",".join(generated_entity_types)
    print(f"Dynamic entity types selected: {generated_entity_types}")

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=entity_types_value,
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0
    used_heuristic_fallback = False

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations, used_heuristic_fallback
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        try:
            final_result = await use_llm_func(hint_prompt)
            if isinstance(final_result, list):
                final_result = final_result[0]["text"]

            history = pack_user_ass_to_openai_messages(
                hint_prompt, final_result, using_amazon_bedrock
            )
            for now_glean_index in range(entity_extract_max_gleaning):
                glean_result = await use_llm_func(
                    continue_prompt, history_messages=history
                )
                history += pack_user_ass_to_openai_messages(
                    continue_prompt, glean_result, using_amazon_bedrock
                )
                final_result += glean_result
                if now_glean_index == entity_extract_max_gleaning - 1:
                    break
                if_loop_result: str = await use_llm_func(
                    if_loop_prompt, history_messages=history
                )
                if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                if if_loop_result != "yes":
                    break
        except Exception as exc:
            logger.warning(
                "Entity extraction LLM failed for chunk %s, using heuristic fallback: %s",
                chunk_key,
                exc,
            )
            used_heuristic_fallback = True
            maybe_nodes, maybe_edges = _heuristic_extract_records(content, chunk_key)
            already_processed += 1
            already_entities += len(maybe_nodes)
            already_relations += len(maybe_edges)
            now_ticks = PROMPTS["process_tickers"][
                already_processed % len(PROMPTS["process_tickers"])
            ]
            print(
                f"{now_ticks} Processed {already_processed}"
                f"({already_processed * 100 // len(ordered_chunks)}%) chunks, "
                f" {already_entities} entities(duplicated),"
                f" {already_relations} relations(duplicated)\r",
                end="",
                flush=True,
            )
            return maybe_nodes, maybe_edges

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes: dict = defaultdict(list)
        maybe_edges: dict = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue
            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}"
            f"({already_processed * 100 // len(ordered_chunks)}%) chunks, "
            f" {already_entities} entities(duplicated),"
            f" {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in asyncio.Semaphore, limiting max_async calls
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()  # clear progress bar

    maybe_nodes: dict = defaultdict(list)
    maybe_edges: dict = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # undirected graph — normalise edge key order
            maybe_edges[tuple(sorted(k))].extend(v)

    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(
                k, v, knwoledge_graph_inst, global_config, tokenizer_wrapper
            )
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(
                k[0], k[1], v, knwoledge_graph_inst, global_config, tokenizer_wrapper
            )
            for k, v in maybe_edges.items()
        ]
    )

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if used_heuristic_fallback and hasattr(knwoledge_graph_inst, "_graph"):
        knwoledge_graph_inst._graph.graph["heuristic_indexing"] = True
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    return knwoledge_graph_inst
