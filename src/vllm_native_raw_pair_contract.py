"""CPU-only admission and memory contract for a future B8 raw-pair model hook.

This module deliberately launches nothing and does not patch vLLM. The CUDA
pair implementation and the whole-model adapter require separate qualification.
"""
from dataclasses import dataclass, asdict
from typing import Protocol, Any


@dataclass(frozen=True)
class DecodeAdmission:
    actual_requests: int
    actual_tokens: int
    padded_tokens: int
    query_rows: int
    full_graph: bool
    page_table_rows: int
    page_table_columns: int
    page_table_row_stride: int
    page_table_column_stride: int
    page_table_dtype: str
    query_dtype: str
    cache_dtype: str
    kv_heads: int = 2
    query_heads: int = 16
    head_dim: int = 128
    page_size: int = 16
    sm_count: int = 170
    compute_capability: tuple[int, int] = (12, 0)
    common_prefix_tokens: int = 0
    physical_prefix_validated: bool = False
    cohort_epoch_unchanged: bool = False
    unit_kv_scales: bool = True
    decode_query_length: int = 1
    has_sink_or_custom_mask: bool = False
    sliding_window: bool = False
    dcp_world_size: int = 1


def assess_admission(info: DecodeAdmission) -> dict:
    """Explicit original-path fallback for every shape outside the first gate."""
    reasons = []
    if not info.full_graph:
        reasons.append('not_full_graph')
    if not (info.actual_requests == info.actual_tokens == info.padded_tokens == info.query_rows == 8):
        reasons.append('requires_actual_and_padded_B8_pure_decode')
    if info.page_table_rows != 8 or info.page_table_columns <= 0:
        reasons.append('page_table_shape')
    if info.page_table_column_stride != 1 or info.page_table_row_stride != info.page_table_columns:
        reasons.append('page_table_requires_original_full_contiguous_rows')
    if info.page_table_dtype != 'torch.int32':
        reasons.append('page_table_int32_ABI')
    if info.query_dtype != 'torch.bfloat16' or info.cache_dtype != 'torch.bfloat16':
        reasons.append('actual_BF16_only')
    if (info.kv_heads, info.query_heads, info.head_dim, info.page_size) != (2, 16, 128, 16):
        reasons.append('unqualified_attention_shape')
    if info.compute_capability != (12, 0) or info.sm_count <= 0:
        reasons.append('unqualified_device')
    if info.common_prefix_tokens < 256 or info.common_prefix_tokens % 16:
        reasons.append('no_complete_native_shared_tile')
    if info.common_prefix_tokens > info.page_table_columns * 16:
        reasons.append('prefix_outside_page_table')
    if not info.physical_prefix_validated or not info.cohort_epoch_unchanged:
        reasons.append('prefix_identity_or_epoch_unverified')
    if not info.unit_kv_scales or info.has_sink_or_custom_mask or info.sliding_window:
        reasons.append('unqualified_scale_or_mask')
    if info.decode_query_length != 1 or info.dcp_world_size != 1:
        reasons.append('unqualified_speculative_or_DCP_path')
    if reasons:
        return {'eligible': False, 'action': 'original_native_graph', 'reasons': reasons}
    # The public XQA ABI derives this extent from full table width. The C++
    # launcher also uses it as the physical page-table row pitch.
    extent = info.page_table_columns * info.page_size
    splits = min(max(1, info.sm_count // (8 * info.kv_heads)), (extent + 255) // 256)
    cut = info.common_prefix_tokens // 256 * 256
    return {'eligible': True, 'action': 'candidate_only_after_independent_qualification',
            'original_native_batch': 8, 'paired_batch': 4,
            'native_launch_extent': extent, 'native_splits': splits,
            'prefix_sm_argument': splits * 4 * info.kv_heads,
            'tail_sm_argument': info.sm_count, 'prefix_tiles': cut // 256,
            'shared_native_tile_tokens': cut,
            'shared_tokens_reprocessed_in_private_tail': info.common_prefix_tokens - cut,
            'prefix_phase': 1, 'tail_phase': 3, 'runtime_candidate_PDL': False}


def buffer_contract(info: DecodeAdmission, layer_count: int = 48) -> dict:
    decision = assess_admission(info)
    if not decision['eligible']:
        raise ValueError(decision['reasons'])
    S = decision['native_splits']
    buffers = {
        'workspace': {'shape': [64 * 1024 * 1024], 'dtype': 'uint8', 'bytes': 64 * 1024 * 1024},
        'paired_raw_state': {'shape': [4, 2, S, 4, 32, 35], 'dtype': 'float32', 'bytes': 4 * 2 * S * 4 * 32 * 35 * 4},
        'contiguous_Q': {'shape': [8, 16, 128], 'dtype': 'bfloat16', 'bytes': 8 * 16 * 128 * 2},
        'paired_Q': {'shape': [4, 1, 32, 128], 'dtype': 'bfloat16', 'bytes': 4 * 32 * 128 * 2},
        'native_O_temporary_if_needed': {'shape': [8, 16, 128], 'dtype': 'bfloat16', 'bytes': 8 * 16 * 128 * 2},
        'unused_prefix_O': {'shape': [4, 1, 32, 128], 'dtype': 'bfloat16', 'bytes': 4 * 32 * 128 * 2},
        'paired_full_page_table': {'shape': [4, info.page_table_columns], 'dtype': 'int32', 'bytes': 4 * info.page_table_columns * 4},
        'native_lengths': {'shape': [8, 1], 'dtype': 'uint32', 'bytes': 32},
        'paired_lengths': {'shape': [4, 1], 'dtype': 'uint32', 'bytes': 16},
    }
    return {'layers': layer_count, 'allocation_sets': 1,
            'lifetime_rule': 'One serial model graph stream: prefix L -> tail L -> prefix L+1. No concurrent replay on shared scratch/state.',
            'original_KV_and_full_page_table': 'aliases; no payload gather or per-layer duplication',
            'buffers': buffers, 'shared_production_bytes': sum(x['bytes'] for x in buffers.values()),
            'checked_graph_QO_snapshot_bytes': layer_count * 2 * 8 * 16 * 128 * 2,
            'snapshot_scope': 'Qualification graph only; absent from production graph.'}


class QualifiedRawPairLayerExecutor(Protocol):
    """Future adapter interface, not an assertion that this executor exists.

    prepare_step runs once before layer 0 on the graph stream. It refreshes
    full paired page rows and lengths from actual native metadata. run_layer
    charges Q stride canonicalization, Q packing, two native calls and O copy.
    Both must use preallocated buffers. Calls retain original per-row S.
    """
    def prepare_step(self, native_full_table: Any, native_lengths: Any) -> None: ...
    def run_layer(self, layer: int, query: Any, native_cache: Any, output: Any) -> Any: ...


REQUIRED_QUALIFICATION_FIELDS = (
    'all48_attention_Q_bits_equal', 'all48_attention_O_bits_equal',
    'hidden_bits_equal', 'local_vocab_logits_bits_equal',
    'gathered_logits_bits_equal_where_present', 'all48_current_KV_slot_bits_equal',
    'input_ids_positions_tables_bits_unchanged', 'native_AA_bits_equal',
    'candidate_graph_did_execute', 'same_state_replay_not_new_generated_input',
)


def qualify_same_state_receipt(receipt: dict) -> bool:
    """The driver checks both TP ranks after RPC; never strand one peer locally."""
    return all(receipt.get(field) is True for field in REQUIRED_QUALIFICATION_FIELDS)
