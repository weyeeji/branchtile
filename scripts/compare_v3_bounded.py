#!/usr/bin/env python
"""V3 bounded check: same paired protocol/strong baselines, register-u candidate.

The candidate supports N64 only; rejected N32 configurations are recorded by the
common tuner. Fixed E2 and native FlashInfer retain their full tuning choices.
"""
import compare_attention
from branchtile.attention_v3 import Attention

if __name__ == '__main__':
    compare_attention.Attention = Attention
    compare_attention.main()
