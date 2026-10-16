"""Deterministic, table-aware hybrid retrieval for application controls.

Retrieval changes model input and is an application alternative, not a same-mask
attention control. No gold answer/evidence is consulted by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class Chunk:
    index: int
    text: str
    source_start: int
    source_end: int
    tokens: int
    table_header_repeated: bool
    oversized_atomic_line: bool


def lexical_tokens(text):
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())


def make_chunks(text, tokenizer, target_tokens):
    """Pack whole lines, preserving table rows and repeating table headings.

    Target 512/1024 is a soft bound for an indivisible long row/utterance. Such
    oversize chunks are explicitly flagged, never silently truncated. All source
    characters are assigned once; repeated table header bytes are counted again.
    """
    assert target_tokens > 0
    lines = text.splitlines(keepends=True)
    chunks = []
    current, start, position = "", 0, 0
    table_header, previous_was_table, has_repeat = "", False, False

    def count(value):
        return len(tokenizer.encode(value, add_special_tokens=False))

    def emit(stop):
        nonlocal current, start, has_repeat
        if current:
            n = count(current)
            chunks.append(Chunk(len(chunks), current, start, stop, n, has_repeat, n > target_tokens))
        current, start, has_repeat = "", stop, False

    for i, line in enumerate(lines):
        is_table = line.count("|") >= 2
        if is_table and not previous_was_table:
            table_header = line
            if i + 1 < len(lines) and re.fullmatch(r"[\s|:\-]+", lines[i + 1]):
                table_header += lines[i + 1]
        if not is_table:
            table_header = ""
        if current and count(current + line) > target_tokens:
            emit(position)
            if is_table and previous_was_table and table_header and line not in table_header.splitlines(keepends=True):
                current = table_header
                has_repeat = True
        current += line
        position += len(line)
        previous_was_table = is_table
    emit(len(text))
    assert all(a.source_end == b.source_start for a, b in zip(chunks, chunks[1:]))
    assert not text or chunks[0].source_start == 0 and chunks[-1].source_end == len(text)
    return chunks


def ranking(scores):
    scores = np.asarray(scores)
    return np.lexsort((np.arange(len(scores)), -scores)).tolist()


def hybrid_rank(chunks, query, dense_scores, top_candidates=20, rrf_constant=60):
    """Stable BM25 + dense RRF over all chunks, then at most top20 candidates."""
    bm25 = BM25Okapi([lexical_tokens(chunk.text) for chunk in chunks])
    lexical = bm25.get_scores(lexical_tokens(query))
    dense_order, lexical_order = ranking(dense_scores), ranking(lexical)
    rrf = np.zeros(len(chunks), dtype=np.float64)
    for order in (dense_order, lexical_order):
        for rank, index in enumerate(order, 1):
            rrf[index] += 1 / (rrf_constant + rank)
    order = ranking(rrf)[:top_candidates]
    return order, {"bm25_scores": lexical.tolist(), "dense_scores": np.asarray(dense_scores).tolist(),
                   "rrf_scores": rrf.tolist(), "rrf_constant": rrf_constant,
                   "top_candidate_count": min(top_candidates, len(chunks))}


def select_context(chunks, ordered_candidates, tokenizer, budget):
    """Whole chunks, exact generation-tokenizer budget, original source order."""
    selected, seen_hashes = [], set()
    for index in ordered_candidates:
        key = hashlib.sha256(chunks[index].text.encode()).hexdigest()
        if key in seen_hashes:
            continue
        trial = sorted(selected + [index])
        candidate = "\n\n".join(chunks[i].text for i in trial)
        if len(tokenizer.encode(candidate, add_special_tokens=False)) <= budget:
            selected.append(index)
            seen_hashes.add(key)
    selected.sort()
    content = "\n\n".join(chunks[i].text for i in selected)
    n = len(tokenizer.encode(content, add_special_tokens=False))
    assert n <= budget
    return content, {"selected_chunks": selected, "retrieved_tokens": n,
                     "budget": budget, "empty_context": not bool(selected),
                     "selection_order": "rank for admission, original source order for generation"}
