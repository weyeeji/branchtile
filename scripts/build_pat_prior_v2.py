"""Build PAT with one unused obsolete Thrust include removed for CUDA 13.2."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
os.environ['MAX_JOBS'] = '2'
os.environ['CUDA_HOME'] = '/usr/local/cuda-13.2'
prior = ROOT / 'build/pat_prior'
assert subprocess.check_output(['git', '-C', str(prior), 'rev-parse', 'HEAD'], text=True).strip() == '8cb067f0a31dabe800958f657093f17ac5d7366d'
assert not subprocess.check_output(['git', '-C', str(prior), 'status', '--porcelain'], text=True).strip()
output = ROOT / 'results/pat_prior_build_v2.json'
assert not output.exists()
started = time.perf_counter()
report = {'status': 'building', 'prior': 'https://github.com/flashserve/PAT',
          'commit': '8cb067f0a31dabe800958f657093f17ac5d7366d',
          'source_changes': ['Remove unused thrust/system/cuda/detail/par.h include; header removed in installed CCCL'], 'adaptation': 'SM120 compiler target, CUDA13.2 CCCL include path, isolated copied source with obsolete unused header removed',
          'source_sha256': {str(p.relative_to(prior)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((prior / 'csrc').glob('*')) if p.is_file()}}
output.write_text(json.dumps(report, indent=2) + '\n')
try:
    import torch
    import flashinfer
    from torch.utils.cpp_extension import load
    cutlass = Path(flashinfer.__file__).parent / 'data/cutlass'
    copied = ROOT / 'build/pat_cuda13_compat'
    assert not copied.exists()
    shutil.copytree(prior / 'csrc', copied)
    kernel = copied / 'pat_fwd_kernel.h'
    source = kernel.read_text()
    assert source.count('#include <thrust/system/cuda/detail/par.h>') == 1
    kernel.write_text(source.replace('#include <thrust/system/cuda/detail/par.h>', '// Unused obsolete Thrust header removed for CCCL compatibility.'))
    report['generated_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(copied.glob('*')) if p.is_file()}
    directory = ROOT / 'build/pat_sm120_extension_v2'
    directory.mkdir(exist_ok=True)
    sources = [copied / 'bindings.cpp'] + sorted(copied.glob('pat_fwd_split_*.cu'))
    flags = ['-O3', '-std=c++20', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
             '-U__CUDA_NO_HALF2_OPERATORS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
             '--expt-relaxed-constexpr', '--expt-extended-lambda', '--use_fast_math', '-lineinfo']
    module = load(name='_prefix_attn', sources=[str(p) for p in sources],
                  extra_include_paths=[str(copied), '/usr/local/cuda-13.2/targets/x86_64-linux/include/cccl', str(cutlass / 'include'), str(cutlass / 'tools/util/include')],
                  extra_cflags=['-O3', '-std=c++20'], extra_cuda_cflags=flags,
                  build_directory=str(directory), verbose=True)
    report.update(status='built_unqualified', module=module.__file__,
                  module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                  CUDA_initialized=torch.cuda.is_initialized(), torch=torch.__version__, cuda=torch.version.cuda)
    assert not report['CUDA_initialized']
except Exception:
    report.update(status='failed', error=traceback.format_exc())
    raise
finally:
    report['CPU_build_wall_seconds'] = time.perf_counter() - started
    output.write_text(json.dumps(report, indent=2) + '\n')
