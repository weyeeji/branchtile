"""CPU-only compilation; no CUDA-visible device is allowed in this process."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('FLASHINFER_CUDA_ARCH_LIST') == '12.0'
import torch
from src.xqa_raw_pair_dynamic_cut_build import jit_specs

started = time.perf_counter()
assert not torch.cuda.is_initialized()
specs, receipt = jit_specs()
for spec in specs.values():
    spec.build_and_load()
assert not torch.cuda.is_initialized()
out = ROOT / 'results/xqa_raw_dynamic_cut_build_v1.json'
assert not out.exists()
out.write_text(json.dumps({'status': 'CPU_compiled_not_GPU_qualified', 'receipt': receipt,
                          'CUDA_initialized': torch.cuda.is_initialized(),
                          'wall_seconds': time.perf_counter()-started}, indent=2)+'\n')
