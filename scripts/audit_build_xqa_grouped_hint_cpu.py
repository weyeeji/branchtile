"""Build all native-group hint variants without creating a CUDA context."""
import argparse,json,os,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.xqa_raw_grouped_hint_build import jit_specs,generate

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
 assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 out=Path(a.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
 r={'scope':__doc__,'builds':[]}
 for ng,fold in [(8,2),(4,2),(4,4)]:
  specs,receipt=jit_specs(ng,fold);dest,_=generate(ng,fold)
  source=(dest/'xqa/mha.cu').read_text()
  assert source.count('ThrdRegRowMax hintToStore = globalRowMax;')==1
  assert f'constexpr uint32_t nativeRows = {ng};' in source
  assert f'if constexpr (HEAD_GRP_SIZE == {ng*fold})' in source
  for spec in specs.values():spec.build_and_load()
  assert not torch.cuda.is_initialized()
  r['builds'].append({'native_group':ng,'fold':fold,'identity':receipt['identity'],'modules':list(specs),'source_path':str(dest/'xqa/mha.cu'),'generated_source_sha256':hashlib.sha256((dest/'xqa/mha.cu').read_bytes()).hexdigest(),'cuda_initialized':False})
 r['status']='cpu_build_passed';out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'builds':len(r['builds'])}))
if __name__=='__main__':main()
