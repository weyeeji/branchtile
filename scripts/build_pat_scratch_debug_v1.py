"""Independent NaN scratch + metadata visibility diagnostic; unchanged CUDA objects."""
import hashlib,importlib.util,json,os,pathlib,shutil,sys,time,traceback
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
os.environ.update(TORCH_CUDA_ARCH_LIST='12.0',FLASHINFER_CUDA_ARCH_LIST='12.0',MAX_JOBS='1',CUDA_HOME='/usr/local/cuda-13.2')
os.environ['PATH']=str(R/'.venv-engine/bin')+os.pathsep+os.environ['PATH']
output=R/'results/pat_scratch_debug_build_v1.json';assert not output.exists()
base=json.loads((R/'results/pat_stream_compat_build_v4.json').read_text()); src=R/'build/pat_scratch_debug_v1'; assert not src.exists()
for name,h in base['generated_sha256'].items(): assert sha(R/'build/pat_stream_compat_v4'/name)==h
shutil.copytree(R/'build/pat_stream_compat_v4',src)
api=src/'api.h';s=api.read_text();s=s.replace('namespace PAT_NAMESPACE {','namespace PAT_NAMESPACE {\ninline at::Tensor debug_last_o, debug_last_lse;\n',1)
needle='        base_param.oaccum_ptr = first_part_out.value().data_ptr();'
s=s.replace(needle,'        first_part_out.value().fill_(std::numeric_limits<float>::quiet_NaN());\n        first_part_lse.value().fill_(std::numeric_limits<float>::quiet_NaN());\n        debug_last_o = first_part_out.value();\n        debug_last_lse = first_part_lse.value();\n'+needle)
api.write_text('#include <limits>\n'+s)
bind=src/'bindings.cpp';s=bind.read_text();s=s.replace('    m.doc() =', '    m.def("debug_scratch", [](){ return std::make_tuple(pat::debug_last_o, pat::debug_last_lse); });\n    m.doc() =')
needle='.def("to_gpu", &KernelInfo::to_gpu, py::arg("device"));'
fields=['q_tables','block_tables','num_seqs_per_CTAs','CTA_ranks','kv_in_CTAs','MNWs','num_split_per_seq','max_split_per_seq','max_seqs_in_CTA','max_blocks_in_CTA']
s=s.replace(needle,'.def("to_gpu", &KernelInfo::to_gpu, py::arg("device"))\n'+''.join(f'        .def_readonly("{f}",&KernelInfo::{f})\n' for f in fields)+'        ;');bind.write_text(s)
objs=sorted((R/'build/pat_stream_extension_v4').glob('*.cuda.o'));assert len(objs)==4
rep={'status':'building','scope':__doc__,'base_receipt':{'path':'results/pat_stream_compat_build_v4.json','sha256':sha(R/'results/pat_stream_compat_build_v4.json')},'commit':base['commit'],'changes':['Fill FP32 partial O/LSE with NaN before launch.','Retain latest temporary tensors for diagnosis.','Expose read-only scheduler metadata.'],'unchanged':['CUDA objects reused byte-identically, including v4 explicit stream dependencies.','Scheduler, tiles, arithmetic kernels and gather.'],'generated_sha256':{p.name:sha(p) for p in src.iterdir() if p.is_file()},'reused_cuda_objects':{str(p.relative_to(R)):sha(p) for p in objs},'source_sha256':{str(pathlib.Path(__file__).relative_to(R)):sha(__file__)}}
t=time.perf_counter();output.write_text(json.dumps(rep,indent=2)+'\n')
try:
 import torch
 from torch.utils.cpp_extension import load
 fi=pathlib.Path(importlib.util.find_spec('flashinfer').origin).parent/'data/cutlass'
 directory=R/'build/pat_scratch_debug_extension_v1';directory.mkdir()
 module=load(name='_prefix_attn',sources=[str(bind)],extra_include_paths=[str(src),'/usr/local/cuda-13.2/targets/x86_64-linux/include/cccl',str(fi/'include'),str(fi/'tools/util/include')],extra_cflags=['-O3','-std=c++20'],extra_ldflags=[str(p) for p in objs],with_cuda=True,build_directory=str(directory),verbose=True)
 rep.update(status='built_unqualified_scratch_debug',module=module.__file__,module_sha256=sha(module.__file__),CUDA_initialized=torch.cuda.is_initialized(),flashinfer_imported='flashinfer' in sys.modules)
 assert not rep['CUDA_initialized'] and not rep['flashinfer_imported']
except Exception:
 rep.update(status='failed',error=traceback.format_exc());raise
finally:
 rep['CPU_build_seconds']=time.perf_counter()-t;output.write_text(json.dumps(rep,indent=2)+'\n')
