"""CPU-only locally fixed PAT scheduler binding and exact logical scratch-write audit."""
import argparse, collections, hashlib, json, os, pathlib, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--capture',required=True); p.add_argument('--output',required=True); a=p.parse_args()
 assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 out=pathlib.Path(a.output); assert not out.exists(); out.parent.mkdir(parents=True,exist_ok=True)
 os.environ['PATH']=str(ROOT/'.venv-engine/bin')+os.pathsep+os.environ['PATH']; os.environ['MAX_JOBS']='1'
 import torch
 from torch.utils.cpp_extension import load
 assert not torch.cuda.is_initialized()
 directory=ROOT/'build/pat_fixed_schedule_cpu_v1'; directory.mkdir(exist_ok=True)
 header=ROOT/'build/pat_fixed_scheduler_v1/prefix_tree.h'
 fields=['q_tables','block_tables','num_seqs_per_CTAs','CTA_ranks','kv_in_CTAs','MNWs','num_split_per_seq','max_split_per_seq','max_seqs_in_CTA','max_blocks_in_CTA']
 cpp='''#include <torch/extension.h>
#include <pybind11/stl.h>
#include <optional>
#include "prefix_tree.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 py::class_<KernelInfo>(m,"KernelInfo")\n'''+''.join(f' .def_readonly("{f}",&KernelInfo::{f})\n' for f in fields)+''';
 py::class_<PrefixTree>(m,"PrefixTree")
 .def(py::init<int>())
 .def("build",[](PrefixTree&t,std::vector<int> ls,torch::Tensor bt){t.build_radix_tree(ls.data(),bt.data_ptr<int>(),bt.size(0),bt.size(1));})
 .def("pack",[](PrefixTree&t,std::vector<std::vector<int>> tiles,int g,int h){t.pack_schedule(tiles,g,h,false);})
 .def_readonly("info",&PrefixTree::_internal_info);
}
'''
 src=directory/'bindings.cpp'; src.write_text(cpp)
 start=time.perf_counter()
 mod=load(name='pat_fixed_schedule_cpu_v1',sources=[str(src)],extra_include_paths=[str(header.parent)],extra_cflags=['-O1','-std=c++20'],with_cuda=False,build_directory=str(directory),verbose=True)
 payload=torch.load(a.capture,map_location='cpu',weights_only=True)
 lengths=(payload['positions']+1).tolist(); table=payload['original_block_table'].int().contiguous(); b=len(lengths)
 tree=mod.PrefixTree(16); tree.build(lengths,table)
 tiles=[[64,128,4],[64,64,4],[64,32,4],[32,128,2],[32,64,2],[32,32,2],[32,16,2],[16,64,1],[16,32,1],[16,16,1]]
 tree.pack(tiles,8,2); info=tree.info
 slots=collections.defaultdict(list); coverage=[collections.Counter() for _ in lengths]; groups=[]; bad_bounds=[]
 for g,tile in enumerate(info.MNWs):
  qs=info.q_tables[g].tolist(); bs=info.block_tables[g].tolist(); ns=info.num_seqs_per_CTAs[g].tolist(); ranks=info.CTA_ranks[g].tolist(); ks=info.kv_in_CTAs[g].tolist()
  groups.append({'group':g,'tile':tile,'query_table':qs,'num_seqs':ns,'CTA_rank':ranks,'kv_tokens':ks,'block_table_shape':list(info.block_tables[g].shape)})
  for c,n in enumerate(ns):
   if not(0<n<=tile[0]//8 and 0<ks[c]<=16*len(bs[c])): bad_bounds.append([g,c,'tile_or_K_bound'])
   for row in qs[c][:n]:
    if not(0<=row<b and 0<=ranks[c]<info.num_split_per_seq[row]): bad_bounds.append([g,c,row,'slot_bound']); continue
    slots[row,ranks[c]].append([g,c])
    for i in range(ks[c]): coverage[row][(bs[c][i//16],i%16)]+=1
 expected={(row,rank) for row in range(b) for rank in range(int(info.num_split_per_seq[row]))}
 conflicts=[{'slot':list(k),'writers':v} for k,v in slots.items() if len(v)!=1]
 missing=[list(k) for k in sorted(expected-set(slots))]; extra=[list(k) for k in sorted(set(slots)-expected)]
 cover=[]
 for row,L in enumerate(lengths):
  want=collections.Counter((int(table[row,i//16]),i%16) for i in range(L)); got=coverage[row]
  cover.append({'row':row,'expected_tokens':L,'assigned_tokens':sum(got.values()),'missing_tokens':sum((want-got).values()),'extra_or_duplicate_tokens':sum((got-want).values()),'equal':got==want})
 report={'status':'passed' if not(conflicts or missing or extra or bad_bounds) and all(x['equal'] for x in cover) else 'failed','scope':'Locally fixed CPU scheduler; logical O/LSE slots, not actual CUDA store lane addressing or numerical correctness. Every (request,split) expands to all16heads/128Ocolumns and one LSE perhead.','source_sha256':{str(header.relative_to(ROOT)):sha(header),str(src.relative_to(ROOT)):sha(src),str(pathlib.Path(__file__).relative_to(ROOT)):sha(__file__)},'capture':{'path':a.capture,'sha256':sha(a.capture)},'module':{'path':mod.__file__,'sha256':sha(mod.__file__)},'groups':groups,'num_split_per_seq':info.num_split_per_seq.tolist(),'max_split_per_seq':info.max_split_per_seq,'logical_slot_count':len(expected),'conflicts':conflicts,'missing':missing,'extra':extra,'bad_bounds':bad_bounds,'physical_token_coverage':cover,'CUDA_initialized':torch.cuda.is_initialized(),'wall_seconds':time.perf_counter()-start}
 assert not torch.cuda.is_initialized(); out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:report[k] for k in ['status','logical_slot_count','conflicts','missing','bad_bounds','wall_seconds']}))
if __name__=='__main__': main()
