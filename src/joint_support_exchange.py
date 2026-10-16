"""Opt-in CPU support exchange from the frozen 2026-10-16 pooled-score screen.

This changes masks. It is not an attention executor or a quality guarantee.
No GPU initialization, environment mutation, model output, or future Q access.
"""
from dataclasses import dataclass, asdict
import hashlib
import math
import time

import numpy as np

PROTOCOL_ID = "joint-support-exchange-cpu-v1-20261016"
PROTOCOL_SHA256 = "64ee6faf63a7f7ccec4cda0ba39d5242f464107054ce89eec3e89d754fa92e6f"
FROZEN_SOURCE_SHA256 = "97625c9207a2541ab76cef5eedf5200740ab6bf066abae5d9b89cd49b56aa651"


@dataclass(frozen=True)
class JointSupportExchangeConfig:
    protocol_id: str
    protocol_sha256: str
    epsilon: float
    request_order: str
    max_rounds: int
    protected_first: int
    protected_last: int

    def __post_init__(self):
        if self.protocol_id != PROTOCOL_ID or self.protocol_sha256 != PROTOCOL_SHA256:
            raise ValueError("Unrecognized frozen exchange protocol")
        if isinstance(self.epsilon, bool) or self.epsilon not in (0, .001, .005, .01):
            raise ValueError("Explicit frozen epsilon required")
        if self.request_order not in ("ascending", "descending"):
            raise ValueError("Explicit frozen request order required")
        if (self.max_rounds, self.protected_first, self.protected_last) != (2, 4, 32):
            raise ValueError("Frozen rounds/protection changed")

    @classmethod
    def from_mapping(cls, value):
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("All opt-in fields must be explicit; extra fields rejected")
        return cls(**value)


def score_sum(score, mask):
    return math.fsum(score[mask].tolist())

def stable_order(ids, scores, descending=False):
    return ids[np.lexsort((ids, -scores[ids] if descending else scores[ids]))]

def exchange(scores, baseline, protected, epsilon, order, rounds=2):
    """Sequential requests, prefix-batched swaps; owners refreshed each turn."""
    masks = baseline.copy()
    b, n = masks.shape
    original = np.array([score_sum(scores[q], baseline[q]) for q in range(b)])
    losses = np.zeros(b, dtype=np.float64)
    events = []
    for round_idx in range(rounds):
        round_swaps = 0
        for q in order:
            owners = masks.sum(axis=0)
            remove = np.flatnonzero(masks[q] & (owners == 1) & ~protected)
            add = np.flatnonzero(~masks[q] & (owners > 0) & ~protected)
            remove = stable_order(remove, scores[q])
            add = stable_order(add, scores[q], descending=True)
            removable_count, replacement_count = len(remove), len(add)
            count = min(len(remove), len(add))
            remove, add = remove[:count], add[:count]
            delta = scores[q, remove] - scores[q, add]
            # Removable entries remain original top-k members: shared entries
            # are never removed, so accepting a negative delta signals a bug.
            assert np.all(delta >= 0), "Original support is not protected top-k"
            cumulative = np.cumsum(delta, dtype=np.float64)
            remaining = max(0.0, epsilon * original[q] - losses[q])
            take = int(np.searchsorted(cumulative, remaining, side="right"))
            if epsilon == 0:
                # Exact float32 score equality, with no numerical slack.
                assert np.all(delta[:take] == 0)
            # Final acceptance uses fsum on the actual resulting selected set,
            # not just a potentially rounded cumulative-sum estimate.
            while take:
                candidate = masks[q].copy()
                candidate[remove[:take]] = False
                candidate[add[:take]] = True
                actual_loss = original[q] - score_sum(scores[q], candidate)
                if actual_loss <= epsilon * original[q] and actual_loss / original[q] <= epsilon:
                    break
                take -= 1
            if take:
                masks[q] = candidate
                losses[q] = actual_loss
            assert int(masks[q].sum()) == int(baseline[q].sum())
            assert np.array_equal(masks[q, protected], baseline[q, protected])
            if epsilon == 0:
                assert score_sum(scores[q], masks[q]) == original[q]
            events.append({"round": round_idx + 1, "question": int(q),
                           "removable_candidates": removable_count,
                           "replacement_candidates": replacement_count,
                           "swaps": take, "cumulative_score_loss": float(losses[q])})
            round_swaps += take
        if round_swaps == 0:
            break
    for q in range(b):
        loss = original[q] - score_sum(scores[q], masks[q])
        assert loss <= epsilon * original[q]
        if epsilon == 0:
            assert loss == 0
    return masks, events


def mask_hash(mask, document_start):
    ids = (np.flatnonzero(mask) + document_start).astype('<i8')
    return hashlib.sha256(ids.tobytes()).hexdigest()


def apply_joint_exchange(scores, baseline, config, *, document_start=0,
                         common_source_tokens=None):
    """Return independent/new [request,head,document] masks and charged CPU records.

    Input scores are the original float32 pooled selector scores. Promotion to
    float64 is exact; no re-normalization or per-document parameter selection.
    The caller must separately charge the preceding score GPU-to-CPU transfer.
    """
    wall, cpu = time.perf_counter(), time.process_time()
    if not isinstance(config, JointSupportExchangeConfig):
        raise TypeError('An explicit frozen config is required')
    score_input, mask_input = np.asarray(scores), np.asarray(baseline)
    if score_input.dtype != np.float32 or mask_input.dtype != np.bool_:
        raise TypeError('Expected original float32 scores and boolean masks')
    if score_input.ndim != 3 or score_input.shape != mask_input.shape:
        raise ValueError('Expected matching [request,head,document] arrays')
    b, heads, n = score_input.shape
    if not (1 <= b <= 8 and heads > 0 and n >= 36):
        raise ValueError('Unsupported empty/short/cohort dimensions')
    if document_start < 0 or not isinstance(document_start, int):
        raise ValueError('Document start must be a nonnegative integer')
    common_n = n if common_source_tokens is None else common_source_tokens
    if not isinstance(common_n, int) or not 0 <= n-common_n <= config.protected_last:
        raise ValueError('Unshared document tail must remain inside protected last32')
    if not np.isfinite(score_input).all() or (score_input < 0).any():
        raise ValueError('Selector scores must be finite and nonnegative')
    protected = np.zeros(n, dtype=bool)
    protected[:config.protected_first] = True
    protected[-config.protected_last:] = True
    counts = mask_input.sum(axis=-1)
    if not np.all(counts == counts[0, 0]) or not mask_input[:, :, protected].all():
        raise ValueError('Fixed per-request/head budget and protected positions required')
    scores64 = score_input.astype(np.float64)
    independent, result = mask_input.copy(), mask_input.copy()
    order = list(range(b))
    if config.request_order == 'descending':
        order.reverse()
    records = []
    for head in range(heads):
        score, initial = scores64[:, head], independent[:, head]
        for q in range(b):
            removable, excluded = score[q, initial[q] & ~protected], score[q, ~initial[q]]
            if len(removable) and len(excluded) and removable.min() < excluded.max():
                raise ValueError('Independent support is not protected top-k')
            if score_sum(score[q], initial[q]) <= 0:
                raise ValueError('Selected score mass must be positive')
        after, events = exchange(score, initial, protected, config.epsilon,
                                 order, config.max_rounds)
        result[:, head] = after
        per_request = []
        for q in range(b):
            original, retained = score_sum(score[q], initial[q]), score_sum(score[q], after[q])
            per_request.append({'question':q, 'budget':int(counts[q, head]),
                'independent_mask_sha256':mask_hash(initial[q], document_start),
                'new_mask_sha256':mask_hash(after[q], document_start),
                'original_selected_score':original, 'retained_score':retained,
                'score_loss_fraction':(original-retained)/original,
                'changed_positions':int(np.count_nonzero(initial[q] != after[q]))})
        def geometry(masks):
            owner = sum(masks[q, :common_n].astype(np.int64) << q for q in range(b))
            counts = np.bincount(owner, minlength=1 << b)
            return {'common_source_union_tokens':int(np.count_nonzero(owner)),
                'owner_bitmask_counts':{str(i):int(c) for i,c in enumerate(counts) if i and c}}
        records.append({'head':head, 'request_order':order, 'events':events,
            'requests':per_request, 'independent_geometry':geometry(initial),
            'new_geometry':geometry(after)})
    report = {'configuration':asdict(config), 'frozen_algorithm_source_sha256':FROZEN_SOURCE_SHA256,
        'shape':list(score_input.shape), 'document_start':document_start,
        'common_source_tokens':common_n, 'records':records,
        'cpu_wall_seconds':time.perf_counter()-wall, 'cpu_process_seconds':time.process_time()-cpu,
        'live_array_bytes_lower_bound':sum(x.nbytes for x in (score_input, mask_input, scores64, independent, result)),
        'scope':'Mask-changing CPU heuristic only; pooled-score loss does not bound O/LSE or task quality; no physical KV capacity released.'}
    return independent, result, report
