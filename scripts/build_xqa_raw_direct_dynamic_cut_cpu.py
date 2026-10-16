"""Build the independent direct-address/device-cut combination with no CUDA context."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.xqa_raw_direct_dynamic_cut_build import jit_specs,generate

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 out=Path(a.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
 specs,receipt=jit_specs();dest,_=generate();s=(dest/'xqa/mha.cu').read_text()
 assert s.count('ThrdRegRowMax hintToStore = globalRowMax;')==1 and 'constexpr uint32_t nativeRows = 8;' in s
 assert 'uint32_t const idxCacheReq = directPrefix ? idxReq * 2 : idxReq;' in s
 assert 'idxCacheReq + 1 >= originalBatch' in s and '*prefixTilesPtr' in s
 for spec in specs.values():spec.build_and_load()
 assert not torch.cuda.is_initialized()
 r={'status':'cpu_build_passed','cuda_initialized':False,'identity':receipt['identity'],'modules':list(specs),
    'generated_source_sha256':hashlib.sha256((dest/'xqa/mha.cu').read_bytes()).hexdigest(),
    'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/xqa_raw_direct_dynamic_cut_build.py',ROOT/'src/xqa_raw_direct_input_build.py']},
    'scope':'Only CPU compilation/static generated-source contract; combined GPU qualification remains required.'}
 out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
