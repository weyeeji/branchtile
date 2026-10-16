"""Short matched native/direct whole-model traces, never an E2E timing benchmark."""
import argparse,copy,hashlib,json,os,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--protocol',default='configs/native_natural_cohort16plus_training_v1.json')
 p.add_argument('--document-id',default='docfinqa-c1647beac4edf6d2')
 p.add_argument('--offline-qualification-records',nargs='+',required=True)
 p.add_argument('--tokens',type=int,choices=[8,16],default=16)
 p.add_argument('--output',required=True);a=p.parse_args()
 assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
 assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==2
 out=Path(a.output).resolve();assert not (out/'summary.json').exists();out.mkdir(parents=True,exist_ok=True)
 proto=json.loads(Path(a.protocol).read_text())
 from scripts.vllm_native_raw_pair_direct_worker import source_hashes
 from src.vllm_native_raw_pair_general_setup import validate_offline_records
 from scripts.benchmark_native_raw_pair_direct_e2e import answer
 base_sources=source_hashes()
 files=['scripts/profile_native_raw_pair_direct_model.py','scripts/vllm_native_raw_pair_direct_trace_worker.py',
        'scripts/benchmark_native_raw_pair_direct_e2e.py','scripts/capture_model.py','scripts/vllm_batch_admission.py']
 report={'status':'initializing','args':vars(a),'protocol':proto,'protocol_sha256':sha(a.protocol),
         'qualified_execution_sources':base_sources,'instrumentation_and_driver_sources':{f:sha(ROOT/f) for f in files},'runs':[],
         'scope':'Offline-qualified production execution plus host-only capture-address observer and per-rank profiler RPCs. Fixed short output for trace coverage, not task quality or performance.'}
 def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 save()
 try:
  paths=[Path(x).resolve() for x in a.offline_qualification_records];records=[json.loads(x.read_text()) for x in paths]
  fingerprint=records[0]['execution_fingerprint']
  validate_offline_records(records,base_sources,fingerprint,[],2)
  coverage={rank:{c['batch'] for r in records if r['rank']==rank for c in r['checks']} for rank in [0,1]}
  report['offline_inputs']=[{'path':str(x),'sha256':sha(x)} for x in paths]
  from transformers import AutoTokenizer
  from scripts.capture_model import make_inputs
  from scripts.vllm_batch_admission import generate_ready_group
  from vllm import LLM,SamplingParams
  manifest=Path(proto['manifest']);groups=json.loads(manifest.read_text())['groups']
  group=next(g for g in groups if g['document_id']==a.document_id)
  item=next(d for d in proto['documents'] if d['document_id']==a.document_id)
  prompts,begin,end=make_inputs(AutoTokenizer.from_pretrained(proto['model_path'],local_files_only=True),group)
  hashes=[hashlib.sha256(json.dumps(x).encode()).hexdigest() for x in prompts];b=len(prompts)
  assert b==group['natural_question_count'] and hashes==item['prompt_token_hashes']
  assert list(map(len,prompts))==item['prompt_lengths'] and [begin,end]==item['document_token_range']
  assert b in coverage[0]&coverage[1], 'Need new direct qualification for this full cohort B'
  enabled=sorted(set(range(6,b+1))&coverage[0]&coverage[1]);validate_offline_records(records,base_sources,fingerprint,enabled,2)
  common=0
  for values in zip(*prompts):
   if len(set(values))!=1:break
   common+=1
  common=common//16*16
  assert common>=65536 and (max(map(len,prompts))+a.tokens-common)*8<=common
  report.update(batch=b,prefix=common,enabled_batches=enabled,manifest_sha256=sha(manifest),prompt_hashes=hashes)
  engine=copy.deepcopy(proto['engine']);engine.update(model=proto['model_path'],kv_cache_dtype='auto',max_num_seqs=b,
      disable_log_stats=False,worker_extension_cls='scripts.vllm_native_raw_pair_direct_trace_worker.NativeRawPairDirectTraceWorker')
  engine['compilation_config']['cudagraph_capture_sizes']=list(range(1,b+1));report['engine']=engine
  llm=LLM(**engine)
  cfg={'output':str(out),'backend':'native_XQA_or_RawPair','batch':b,'enabled_batches':enabled,'setup_mode':'production_only',
       'offline_qualification_records':[str(x) for x in paths],'common_prefix_tokens':common,'prompt_token_hashes':hashes}
  report['configuration']=llm.collective_rpc('configure_dense_cascade',args=(cfg,))
  report['observer']=llm.collective_rpc('install_direct_capture_observer')
  report['capture']=llm.collective_rpc('capture_dense_cascade_graphs')
  assert all(x['captured_modes']==['production'] for x in report['capture'])
  report['captured_addresses']=llm.collective_rpc('direct_capture_address_receipt');save()
  for method in ['native','cascade']:
   phase='trace_'+method
   assert not llm.llm_engine.has_unfinished_requests()
   llm.collective_rpc('set_dense_cascade_phase',args=(phase+'_anchor','native',False))
   assert llm.reset_prefix_cache()
   anchors=llm.generate([{'prompt_token_ids':prompts[0]}],SamplingParams(temperature=0,max_tokens=1,seed=proto['seed']),use_tqdm=False)
   assert anchors[0].num_cached_tokens==0
   llm.collective_rpc('set_dense_cascade_phase',args=(phase,method,False))
   sampling=dict(proto['sampling']);sampling.update(max_tokens=a.tokens,min_tokens=a.tokens,ignore_eos=True)
   starts=llm.collective_rpc('start_direct_profile',args=(phase,))
   try:outputs=generate_ready_group(llm,[{'prompt_token_ids':x} for x in prompts],SamplingParams(**sampling))
   finally:traces=llm.collective_rpc('stop_direct_profile')
   workers=llm.collective_rpc('finish_dense_cascade_phase')
   answers=[answer(i,o) for i,o in enumerate(outputs)]
   run={'method':method,'sampling':sampling,'answers':answers,'workers':workers,'traces':traces,'profile_start_receipts':starts}
   report['runs'].append(run);save()
   assert len(answers)==b and all(len(x['tokens'])==a.tokens for x in answers)
   assert len(workers)==2 and all(w['phase_valid'] and w['qualification_steps']==0 for w in workers)
   assert all(w['candidate_replays']>0 for w in workers) if method=='cascade' else all(w['candidate_replays']==0 for w in workers)
   signatures=[]
   for w in workers:
    raw=json.loads(Path(w['raw_pair_record']).read_text());rows=json.loads(Path(w['graph_metadata']).read_text())['replays']
    assert raw['source_sha256']==base_sources and raw['setup_mode']=='production_only' and raw['offline_qualification']
    assert len(rows)==w['native_replays']+w['candidate_replays']
    assert sum(x['candidate_graph_replayed'] for x in rows)==w['candidate_replays']
    signatures.append(hashlib.sha256(json.dumps([(x['batch'],x['questions'],x['positions']) for x in rows]).encode()).hexdigest())
   assert len(set(signatures))==1;run['graph_trajectory_sha256_by_rank']=signatures
   save()
  assert [[x['tokens'] for x in r['answers']] for r in report['runs']][0]==[[x['tokens'] for x in r['answers']] for r in report['runs']][1]
  assert report['runs'][0]['graph_trajectory_sha256_by_rank']==report['runs'][1]['graph_trajectory_sha256_by_rank']
  report['status']='completed_trace_collection_pending_CUPTI_audit';save()
 except Exception:
  report['status']='failed';report['error']=traceback.format_exc();save();raise
 print(json.dumps({'status':report['status'],'output':str(out)}))
if __name__=='__main__':main()
