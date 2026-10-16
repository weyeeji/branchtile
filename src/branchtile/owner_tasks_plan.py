"""Pure-CPU exact owner-region task packing; no tensor payload is copied."""
from __future__ import annotations
import hashlib
import json


def build_task_layout(offsets, row_map, counts, labels, batch, gqa, tokens, chunk_tokens=512):
    if batch not in (3, 4) or gqa != 8:
        raise ValueError("Owner-task control is qualified only for B3/B4 GQA8")
    if not isinstance(chunk_tokens, int) or chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be a positive integer")
    if labels != sorted(set(labels)) or not labels or any(not 0 < label < (1 << batch) for label in labels):
        raise ValueError("Owner labels must be unique, sorted, nonzero request bitmasks")
    heads, regions = len(offsets), len(labels)
    if not heads or len(counts) != heads or len(row_map) != heads:
        raise ValueError("Inconsistent head dimensions")
    tasks, packed_rows = [], 0
    contributions = [[[] for _ in range(batch)] for _ in range(heads)]
    region_task_counts = [[0] * regions for _ in range(heads)]
    for head in range(heads):
        if len(offsets[head]) != regions + 1 or len(counts[head]) != regions or len(row_map[head]) != regions:
            raise ValueError("Inconsistent region dimensions")
        if offsets[head][0] != 0 or not 0 <= offsets[head][-1] <= tokens:
            raise ValueError("Invalid packed index extent")
        for region, label in enumerate(labels):
            begin, end = offsets[head][region:region+2]
            if not 0 <= begin <= end <= tokens:
                raise ValueError("Offsets must be monotone and in range")
            requests = [request for request in range(batch) if label & (1 << request)]
            active = [request*gqa + group_head for request in requests for group_head in range(gqa)]
            if counts[head][region] != len(active) or list(row_map[head][region][:len(active)]) != active:
                raise ValueError("Active rows must be complete, ordered GQA8 request cohorts")
            for start in range(begin, end, chunk_tokens):
                stop = min(start + chunk_tokens, end)
                # head, region, packed-index begin/end, active rows, packed U row.
                tasks.append([head, region, start, stop, len(active), packed_rows])
                for slot, request in enumerate(requests):
                    contributions[head][request].append(packed_rows + slot*gqa)
                packed_rows += len(active)
                region_task_counts[head][region] += 1
    if packed_rows >= (1 << 31) or len(tasks) >= (1 << 31):
        raise ValueError("Task metadata exceeds signed int32 representation")
    contribution_counts = [[len(values) for values in head] for head in contributions]
    max_contributions = max((len(values) for head in contributions for values in head), default=0)
    merge_indices = [[values + [-1]*(max_contributions-len(values)) for values in head] for head in contributions]
    identity = {"offsets": offsets, "counts": counts, "labels": labels,
                "active_row_map": [[row_map[h][r][:counts[h][r]] for r in range(regions)] for h in range(heads)],
                "batch": batch, "gqa": gqa, "tokens": tokens, "chunk_tokens": chunk_tokens}
    return {"tasks": tasks, "packed_rows": packed_rows, "merge_indices": merge_indices,
            "contribution_counts": contribution_counts, "max_contributions": max_contributions,
            "region_task_counts": region_task_counts, "heads": heads, "regions": regions,
            "task_fields": ["head", "region", "begin", "end", "active_rows", "packed_row_base"],
            "batch": batch, "gqa": gqa, "tokens": tokens, "chunk_tokens": chunk_tokens,
            "structure_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}
