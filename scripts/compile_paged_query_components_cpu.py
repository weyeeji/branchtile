"""CPU-only SM120 compile, including strided native Q and fused KV addresses."""
import hashlib
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
    from src.branchtile.triton_kernel_paged_query_components import producer, merge
    assert not torch.cuda.is_initialized()
    rows = []
    for components in (False, True):
        constants = dict(BATCH=8, HEADS=2, GQA=8, TOKENS=120528, SPLITS=118, PAGE=16,
                         QS0=4096, QS1=128, QS2=1, CS0=8192, CS1=256, CS2=512, CS3=1,
                         CHUNK=1024, M=64, N=64, D=128, COMPONENTS=components)
        signature = dict(zip(producer.arg_names[:6], ('*bf16', '*fp8e4nv', '*i32', '*fp32', '*fp32', '*fp32')))
        signature.update({k: 'constexpr' for k in constants})
        compiled = triton.compile(ASTSource(producer, signature, constants), target=GPUTarget('cuda', 120, 32), options={'num_warps': 4, 'num_stages': 1})
        rows.append({'kernel': 'producer', 'constants': constants, 'metadata': compiled.metadata._asdict(),
                     'cubin_sha256': hashlib.sha256(compiled.asm['cubin']).hexdigest()})
    for write_lse in (False, True):
        constants = dict(BATCH=8, HEADS=2, GQA=8, SPLITS=118, D=128, OS0=2048, OS1=128, OS2=1, WRITE_LSE=write_lse, GROUP=16)
        signature = dict(zip(merge.arg_names[:5], ('*fp32', '*fp32', '*fp32', '*bf16', '*fp32')))
        signature.update({k: 'constexpr' for k in constants})
        compiled = triton.compile(ASTSource(merge, signature, constants), target=GPUTarget('cuda', 120, 32), options={'num_warps': 4, 'num_stages': 1})
        rows.append({'kernel': 'merge', 'constants': constants, 'metadata': compiled.metadata._asdict(),
                     'cubin_sha256': hashlib.sha256(compiled.asm['cubin']).hexdigest()})
    assert not torch.cuda.is_initialized()
    result = {'schema': 1, 'status': 'offline_SM120_compile_passed', 'cuda_initialized': False, 'compiled': rows,
              'sources': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                          ('scripts/compile_paged_query_components_cpu.py', 'scripts/qualify_paged_query_components_gpu.py',
                           'src/branchtile/triton_kernel_paged_query_components.py', 'src/branchtile/triton_kernel_query_components.py')},
              'limitations': ['No GPU execution or numerical qualification.', 'Strided Q compilation is not a tested engine Q-stride receipt.']}
    target = ROOT / 'results/platform/paged_query_components_compile_20261017.json'
    target.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps({'output': str(target.relative_to(ROOT)), 'variants': len(rows), 'cuda_initialized': False}))


if __name__ == '__main__':
    main()
