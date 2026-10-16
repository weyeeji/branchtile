"""Offline compile of isolated shared-prefix experiment; never runs GPU code."""
import hashlib
import collections
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    import torch
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from src.branchtile.triton_kernel_query_components import producer, merge
    assert not torch.cuda.is_initialized()
    rows = []
    for m in (32, 64):
        for components in (False, True):
            constants = dict(ROWS=64, TOKENS=120506, SPLITS=118, CHUNK=1024, M=m, N=64, D=128, COMPONENTS=components)
            signature = dict(zip(producer.arg_names[:6], ('*bf16', '*fp8e4nv', '*fp8e4nv', '*fp32', '*fp32', '*fp32')))
            signature.update({k: 'constexpr' for k in constants})
            compiled = triton.compile(ASTSource(producer, signature, constants), target=GPUTarget('cuda', 120, 32), options={'num_warps': 4, 'num_stages': 1})
            ptx = compiled.asm['ptx']
            opcodes = collections.Counter(line.strip().split()[0] for line in ptx.splitlines()
                                          if line.strip().startswith(('cvt.', 'mma.sync.')))
            rows.append({'kernel': 'producer', 'constants': constants, 'metadata': compiled.metadata._asdict(),
                         'cubin_sha256': hashlib.sha256(compiled.asm['cubin']).hexdigest(),
                         'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(),
                         'static_conversion_and_mma_opcodes': dict(opcodes),
                         'mma_instructions': sorted({line.strip() for line in ptx.splitlines() if 'mma.sync' in line}),
                         'static_local_declarations': [line.strip() for line in ptx.splitlines() if '.local' in line]})
    constants = dict(ROWS=64, SPLITS=118, D=128, GROUP=16)
    signature = dict(zip(merge.arg_names[:4], ('*fp32', '*fp32', '*fp32', '*bf16')))
    signature.update({k: 'constexpr' for k in constants})
    compiled = triton.compile(ASTSource(merge, signature, constants), target=GPUTarget('cuda', 120, 32), options={'num_warps': 4, 'num_stages': 1})
    rows.append({'kernel': 'merge', 'constants': constants, 'metadata': compiled.metadata._asdict(),
                 'cubin_sha256': hashlib.sha256(compiled.asm['cubin']).hexdigest()})
    assert not torch.cuda.is_initialized()
    result = {'schema': 1, 'status': 'offline_SM120_compile_passed', 'cuda_initialized': False,
              'compiled': rows, 'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ('scripts/compile_query_components_cpu.py', 'src/branchtile/triton_kernel_query_components.py')},
              'limitations': ['No GPU execution, numerical qualification, measured speedup or paged/vLLM integration.', 'PTX local declarations do not prove final SASS has no spills; GPU qualification must collect resource metadata.']}
    target = ROOT / 'results/platform/query_components_compile_20261017.json'
    target.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps({'output': str(target.relative_to(ROOT)), 'variants': len(rows), 'cuda_initialized': False}))


if __name__ == '__main__':
    main()
