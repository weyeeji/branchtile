"""CPU-only isolated PAT stream fork/join compatibility build; no math changes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide all GPUs before importing torch'
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
os.environ['FLASHINFER_CUDA_ARCH_LIST'] = '12.0'
os.environ['MAX_JOBS'] = '2'
os.environ['CUDA_HOME'] = '/usr/local/cuda-13.2'
prior = ROOT / 'build/pat_prior'
assert subprocess.check_output(['git', '-C', str(prior), 'rev-parse', 'HEAD'], text=True).strip() == '8cb067f0a31dabe800958f657093f17ac5d7366d'
assert not subprocess.check_output(['git', '-C', str(prior), 'status', '--porcelain'], text=True).strip()
output = ROOT / 'results/pat_stream_compat_build_v3.json'
assert not output.exists()
started = time.perf_counter()
report = {'status': 'building', 'prior': 'https://github.com/flashserve/PAT',
          'commit': '8cb067f0a31dabe800958f657093f17ac5d7366d',
          'scope': 'Explicit caller-stream fork/join compatibility hypothesis, not an established root-cause fix or numerical qualification.',
          'environment': {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','TORCH_CUDA_ARCH_LIST','FLASHINFER_CUDA_ARCH_LIST','CUDA_HOME','MAX_JOBS')},
          'source_changes': ['Remove one unused obsolete Thrust include for CUDA13 CCCL.',
             'Use current PyTorch CUDA stream as caller for all launches and gather.',
             'Record ready event on caller; each child waits before launch.',
             'Record done event on each child; caller waits for all children before gather.',
             'Check stream/event API return values and destroy per-call events/streams after enqueueing joins.'],
          'unchanged': ['Mathematics','MMA/reduction kernels','Tile choices','Tree scheduler','Metadata layout','Allocation strategy'],
          'source_sha256': {str(p.relative_to(prior)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((prior / 'csrc').glob('*')) if p.is_file()}}
output.write_text(json.dumps(report, indent=2) + '\n')
try:
    copied = ROOT / 'build/pat_stream_compat_v3'
    assert not copied.exists()
    shutil.copytree(prior / 'csrc', copied)
    kernel = copied / 'pat_fwd_kernel.h'
    source = kernel.read_text()
    assert source.count('#include <thrust/system/cuda/detail/par.h>') == 1
    kernel.write_text(source.replace('#include <thrust/system/cuda/detail/par.h>', '// Unused obsolete Thrust header removed for CCCL compatibility.'))
    launch = copied / 'pat_fwd_launch_template.h'
    source = launch.read_text()
    assert source.count('cudaStream_t stream = 0;') == 1
    source = source.replace('#include "c10/cuda/CUDAException.h"', '#include "c10/cuda/CUDAStream.h"\n#include "c10/cuda/CUDAException.h"')
    source = source.replace('cudaStream_t stream = 0;', 'cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();')
    source = source.replace('std::vector<cudaStream_t> streams;', 'std::vector<cudaStream_t> streams;\n            std::vector<cudaEvent_t> done_events;\n            cudaEvent_t ready_event = nullptr;')
    old = '''                for (int i = 0; i < num_kernels; i++) {
                    cudaStream_t stream;
                    cudaStreamCreate(&stream);
                    streams.push_back(stream);
                }
                for (int i = 0; i < num_kernels; i++) {
                    MNW_SWITCH(params[i].tile_q, params[i].tile_kv, params[i].Warps, kBlockM, kBlockN, Warps, [&]() {
                        launch<fwd_kernel_traits<elem_type, float, kBlockM, kBlockN, Headdim, Warps, HRatio> >(params[i], streams[i]);
                    });
                }'''
    new = '''                // Explicit fork: all input/metadata writes on the caller precede children.
                C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming));
                C10_CUDA_CHECK(cudaEventRecord(ready_event, stream));
                for (int i = 0; i < num_kernels; i++) {
                    cudaStream_t child_stream;
                    cudaEvent_t done_event;
                    C10_CUDA_CHECK(cudaStreamCreate(&child_stream));
                    C10_CUDA_CHECK(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming));
                    streams.push_back(child_stream);
                    done_events.push_back(done_event);
                    C10_CUDA_CHECK(cudaStreamWaitEvent(child_stream, ready_event, 0));
                }
                for (int i = 0; i < num_kernels; i++) {
                    MNW_SWITCH(params[i].tile_q, params[i].tile_kv, params[i].Warps, kBlockM, kBlockN, Warps, [&]() {
                        launch<fwd_kernel_traits<elem_type, float, kBlockM, kBlockN, Headdim, Warps, HRatio> >(params[i], streams[i]);
                    });
                    C10_CUDA_CHECK(cudaEventRecord(done_events[i], streams[i]));
                }
                // Explicit join: gather and allocator reuse on the caller follow every child.
                for (auto done_event : done_events) {
                    C10_CUDA_CHECK(cudaStreamWaitEvent(stream, done_event, 0));
                }'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    source = source.replace('// run gather_kernel on legacy default stream', '// Run gather on the caller stream after the explicit child joins.')
    old_gather = '<<<grid_gather, WARPS*32, 0>>>(params[0]);'
    assert source.count(old_gather) == 1
    source = source.replace(old_gather, '<<<grid_gather, WARPS*32, 0, stream>>>(params[0]);')
    source = source.replace('''            for (auto &stream: streams) {
                cudaStreamDestroy(stream);
            }''', '''            for (auto done_event : done_events) {
                C10_CUDA_CHECK(cudaEventDestroy(done_event));
            }
            if (ready_event != nullptr) C10_CUDA_CHECK(cudaEventDestroy(ready_event));
            for (auto child_stream : streams) {
                C10_CUDA_CHECK(cudaStreamDestroy(child_stream));
            }''')
    launch.write_text(source)
    import difflib
    patch = '\n'.join(difflib.unified_diff((prior/'csrc/pat_fwd_launch_template.h').read_text().splitlines(), source.splitlines(),
                                          fromfile='official/pat_fwd_launch_template.h', tofile='stream_compat/pat_fwd_launch_template.h'))+'\n'
    diff_path = ROOT / 'results/pat_stream_compat_v3.patch'
    assert not diff_path.exists()
    diff_path.write_text(patch)
    report['stream_patch'] = str(diff_path.relative_to(ROOT))
    report['stream_patch_sha256'] = hashlib.sha256(diff_path.read_bytes()).hexdigest()
    report['generated_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(copied.glob('*')) if p.is_file()}
    changed = sorted(p for p,h in report['generated_sha256'].items() if report['source_sha256']['csrc/'+p] != h)
    assert changed == ['pat_fwd_kernel.h','pat_fwd_launch_template.h'], changed
    report['changed_files'] = changed
    output.write_text(json.dumps(report, indent=2)+'\n')
    # find_spec locates installed headers without importing FlashInfer or probing GPUs.
    package_spec = importlib.util.find_spec('flashinfer')
    assert package_spec and package_spec.origin
    cutlass = Path(package_spec.origin).parent / 'data/cutlass'
    assert (cutlass/'include/cutlass/cutlass.h').exists()
    import torch
    from torch.utils.cpp_extension import load
    assert not torch.cuda.is_initialized()
    directory = ROOT / 'build/pat_stream_extension_v3'
    directory.mkdir(exist_ok=True)
    sources = [copied / 'bindings.cpp'] + sorted(copied.glob('pat_fwd_split_*.cu'))
    flags = ['-O3','-std=c++20','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
             '-U__CUDA_NO_HALF2_OPERATORS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
             '--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo']
    module = load(name='_prefix_attn', sources=[str(p) for p in sources],
                  extra_include_paths=[str(copied),'/usr/local/cuda-13.2/targets/x86_64-linux/include/cccl',str(cutlass/'include'),str(cutlass/'tools/util/include')],
                  extra_cflags=['-O3','-std=c++20'], extra_cuda_cflags=flags, build_directory=str(directory), verbose=True)
    report.update(status='built_unqualified_stream_compat', module=module.__file__,
                  module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                  CUDA_initialized=torch.cuda.is_initialized(), torch=torch.__version__, cuda=torch.version.cuda,
                  flashinfer_imported='flashinfer' in sys.modules)
    assert not report['CUDA_initialized'] and not report['flashinfer_imported']
except Exception:
    report.update(status='failed', error=traceback.format_exc())
    raise
finally:
    report['CPU_build_wall_seconds'] = time.perf_counter()-started
    output.write_text(json.dumps(report, indent=2)+'\n')
