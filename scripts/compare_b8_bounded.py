"""Reject-only screen of B8 row compaction against native and fused controls.

A positive outcome requires further partition/hierarchy controls. Reuses the
frozen FP64 checks and random/reverse paired cache protocol without loosening.
"""
import compare_attention
from branchtile.attention_b8 import Attention, make_plan
from branchtile.union_b8 import FixedUnion
from flashinfer_union_b8 import NativeUnionPrefill

if __name__ == '__main__':
    compare_attention.Attention = Attention
    compare_attention.make_plan = make_plan
    compare_attention.FixedUnion = FixedUnion
    compare_attention.NativeUnionPrefill = NativeUnionPrefill
    compare_attention.main()
