#!/usr/bin/env python
"""Existing bench_kernel protocol, v2 with only an empty-shard guard.

Launch through the root-owned run_jobs GPU ledger, not directly from CPU audit.
"""
import bench_kernel
from branchtile.attention_emptyguard import Attention

if __name__ == "__main__":
    bench_kernel.Attention = Attention
    bench_kernel.main()
