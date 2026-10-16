"""E4 exact hierarchical-intersection mechanism port, not a QUILT reproduction.

For every tree node, use the exact intersection of descendant request supports,
minus tokens assigned to ancestors. Leaves retain the exact residual support.
Sibling nodes may read the same token for disjoint requests, but each original
(request, token) pair occurs exactly once. All nonempty (head,node) items execute
in one native paged-prefill batch; stable LSE merge is part of the timed call.
"""
from functools import lru_cache
import itertools
import time

import torch

from partition_baseline import _NativeGroupedAttention


@lru_cache(None)
def binary_hierarchies(requests):
    requests = tuple(sorted(requests))
    if len(requests) == 1:
        return (requests[0],)
    result = []
    first, rest = requests[0], requests[1:]
    for count in range(len(rest)):
        for chosen in itertools.combinations(rest, count):
            left = (first,) + chosen
            right = tuple(request for request in rest if request not in chosen)
            for a in binary_hierarchies(left):
                for b in binary_hierarchies(right):
                    result.append((a, b))
    return tuple(result)


def descendants(tree):
    return (tree,) if isinstance(tree, int) else tuple(sorted(descendants(tree[0]) + descendants(tree[1])))


def exact_hierarchy_items(owners, tree, batch):
    owners = owners.to(device="cpu", dtype=torch.int64)
    if sorted(descendants(tree)) != list(range(batch)):
        raise ValueError("Hierarchy must contain each active request exactly once")
    if int(owners.min()) < 0 or int(owners.max()) >= 1 << batch:
        raise ValueError("Invalid owner bitmask")
    items, nodes = [], []
    for head in range(owners.shape[0]):
        def visit(node, assigned, depth):
            requests = descendants(node)
            bits = sum(1 << request for request in requests)
            selected = ((owners[head] & bits) == bits) & ~assigned
            ids = torch.where(selected)[0]
            items.append((head, requests, ids, None))
            nodes.append({"head": head, "requests": list(requests), "depth": depth, "tokens": len(ids)})
            if not isinstance(node, int):
                visit(node[0], assigned | selected, depth + 1)
                visit(node[1], assigned | selected, depth + 1)
        visit(tree, torch.zeros_like(owners[head], dtype=torch.bool), 0)
    return items, nodes


class ExactCascadeAttention(_NativeGroupedAttention):
    def __init__(self, q, k, v, owners, batch, tree, *, fixed_split_size=None):
        started = time.perf_counter()
        items, self.nodes = exact_hierarchy_items(owners, tree, batch)
        self.tree = tree
        self._initialize(q, k, v, batch, items, fixed_split_size=fixed_split_size,
                         started=started, metadata_ms=(time.perf_counter() - started) * 1000)
