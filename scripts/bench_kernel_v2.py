#!/usr/bin/env python
"""Same inputs and timing protocol as bench_kernel, explicit shared-buffer v2."""
import bench_kernel
from branchtile.attention_v2 import Attention

if __name__ == '__main__':
    bench_kernel.Attention = Attention
    bench_kernel.main()
