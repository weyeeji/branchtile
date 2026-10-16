"""Compile D64 prefix experiment with GPUs hidden; no numerical claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out = Path(a.output)
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    import torch
    from src.xqa_raw_direct_dslice_build import jit_specs, generate
    specs, receipt = jit_specs()
    dest, _ = generate()
    s = (dest / 'xqa/mha.cu').read_text()
    assert s.count('if (ownsPV)') == 3  # V copy, PV MMA, U export
    assert 'constexpr uint32_t nativeRows = 8;' in s
    assert s.count('if (!splitOutput || outputPartition == 0)') == 2
    # Exhaustively check raw-state addresses: each U/scalar word has one writer.
    writers = {}
    for partition in range(2):
        for warp in range(4):
            for lane in range(32):
                for field in range(35):
                    write = (warp % 2 == partition) if field < 32 else partition == 0
                    if write:
                        addr = (warp * 32 + lane) * 35 + field
                        writers[addr] = writers.get(addr, 0) + 1
    assert len(writers) == 4 * 32 * 35 and set(writers.values()) == {1}
    for spec in specs.values():
        spec.build_and_load()
    assert not torch.cuda.is_initialized()
    r = {'status': 'cpu_build_passed', 'cuda_initialized': False,
         'identity': receipt['identity'], 'modules': list(specs),
         'state_write_partition_exact': True,
         'generated_source_sha256': hashlib.sha256((dest / 'xqa/mha.cu').read_bytes()).hexdigest(),
         'source_sha256': {x: hashlib.sha256((ROOT / x).read_bytes()).hexdigest() for x in
            ['src/xqa_raw_direct_dslice_build.py', 'src/xqa_raw_direct_dslice_runtime.py',
             'scripts/build_xqa_raw_direct_dslice_cpu.py']},
         'scope': 'Compilation and address partition only; no GPU execution or correctness/performance claim.'}
    out.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r))


if __name__ == '__main__':
    main()
