"""CPU-only graph diagnostic artifact: identical performance code plus metadata visibility."""
import hashlib,importlib.util,json,os,pathlib,shutil,sys,time,traceback
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
os.environ.update(TORCH_CUDA_ARCH_LIST='12.0',FLASHINFER_CUDA_ARCH_LIST='12.0',MAX_JOBS='1',CUDA_HOME='/usr/local/cuda-13.2')
os.environ['PATH']=str(R/'.venv-engine/bin')+os.pathsep+os.environ['PATH']
output=R/'results/pat_fixed_graph_build_v1.json';assert not output.exists()
basepath=R/'results/pat_fixed_scheduler_perf_build_v1.json';base=json.loads(basepath.read_text());assert base['status']=='built_unqualified_fixed_scheduler_perf'
stream=json.loads((R/'results/pat_stream_compat_build_v4.json').read_text())
src=R/'build/pat_fixed_graph_v1';assert not src.exists()
for name,h in base['generated_sha256'].items():assert sha(R/'build/pat_fixed_scheduler_perf_v1'/name)==h
shutil.copytree(R/'build/pat_fixed_scheduler_perf_v1',src)
binding=src/'bindings.cpp';text=binding.read_text()
needle='.def("to_gpu", &KernelInfo::to_gpu, py::arg("device"));'
fields=['q_tables','block_tables','num_seqs_per_CTAs','CTA_ranks','kv_in_CTAs','MNWs','num_split_per_seq','max_split_per_seq','max_seqs_in_CTA','max_blocks_in_CTA']
assert text.count(needle)==1
text=text.replace(needle,'.def("to_gpu", &KernelInfo::to_gpu, py::arg("device"))\n'+''.join(f'        .def_readonly("{f}",&KernelInfo::{f})\n' for f in fields)+'        ;')
binding.write_text(text)
assert sha(src/'api.h')==base['generated_sha256']['api.h']
objs=[R/p for p in base['reused_cuda_objects']]
for p,h in base['reused_cuda_objects'].items():assert sha(R/p)==h
rep={'status':'building','scope':__doc__,'label':'PAT compat fixed scheduler (local), not unchanged upstream','base_receipt':{'path':str(basepath.relative_to(R)),'sha256':sha(basepath)},'commit':base['commit'],'changes_from_performance':['Read-only pybind metadata exports only, for in-place graph replay qualification.'],'unchanged':['Fixed guarded scheduler source byte-identical.','CUDA arithmetic/gather and explicit fork/join objects byte-identical.','Original API allocation behavior restored.'],'generated_sha256':{p.name:sha(p) for p in src.iterdir() if p.is_file()},'reused_cuda_objects':base['reused_cuda_objects'],'source_sha256':{str(pathlib.Path(__file__).relative_to(R)):sha(__file__)}}
t=time.perf_counter();output.write_text(json.dumps(rep,indent=2)+'\n')
try:
 import torch
 from torch.utils.cpp_extension import load
 fi=pathlib.Path(importlib.util.find_spec('flashinfer').origin).parent/'data/cutlass'
 directory=R/'build/pat_fixed_graph_extension_v1';directory.mkdir()
 module=load(name='_prefix_attn',sources=[str(src/'bindings.cpp')],extra_include_paths=[str(src),'/usr/local/cuda-13.2/targets/x86_64-linux/include/cccl',str(fi/'include'),str(fi/'tools/util/include')],extra_cflags=['-O3','-std=c++20'],extra_ldflags=[str(p) for p in objs],with_cuda=True,build_directory=str(directory),verbose=True)
 rep.update(status='built_unqualified_fixed_graph',module=module.__file__,module_sha256=sha(module.__file__),CUDA_initialized=torch.cuda.is_initialized(),flashinfer_imported='flashinfer' in sys.modules)
 assert not rep['CUDA_initialized'] and not rep['flashinfer_imported']
except Exception:
 rep.update(status='failed',error=traceback.format_exc());raise
finally:
 rep['CPU_build_seconds']=time.perf_counter()-t;output.write_text(json.dumps(rep,indent=2)+'\n')
