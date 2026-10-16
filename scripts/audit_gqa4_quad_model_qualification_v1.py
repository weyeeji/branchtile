"""Independent CPU audit of both-rank, all-layer GQA4 quad model qualification."""
import collections,hashlib,json,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.vllm_gqa4_raw_pair_setup import REQUIRED_QUALIFICATION_FIELDS,validate_offline_records
from src.gqa4_quad_scope import in_scope,receipt
D=R/'results/gqa4_thinking_quad_model_qual_v1/train1'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
summary=json.loads((D/'summary.json').read_text());assert summary['status']=='completed' and summary['enabled_batches']==[4,5,6,7,8]
assert summary['common_prefix_tokens']==120528 and summary['development_scope']==receipt()
for name,h in summary['source_sha256'].items():assert sha(R/name)==h
assert sha(R/summary['args']['protocol'])==summary['protocol_sha256']
run=summary['runs'];assert len(run)==1;run=run[0]
assert run['phase']=='qualification' and run['method']=='native' and run['validation'] and run['natural_eos_enabled']
assert run['sampling']['ignore_eos'] is False
records=[];outputs=[];steps_by_rank=[];graph_signatures=[]
for rank in (0,1):
 path=D/f'rank{rank}/qualification_raw_pair.json';r=json.loads(path.read_text());records.append(r)
 gp=D/f'rank{rank}/qualification_graph.json';rows=json.loads(gp.read_text())['replays']
 assert r['rank']==rank and r['setup_mode']=='strict' and r['qualification_passed'] and r['phase_valid']
 assert r['enabled_batches']==r['qualified_batches']==[4,5,6,7,8]
 assert r['qualification_steps']==len(r['checks'])==40 and r['qualification_states_by_batch']=={str(b):8 for b in range(4,9)}
 assert r['candidate_replays']==80 and r['native_replays']==len(rows)
 assert len(r['prepare_records'])==40
 for name,h in r['source_sha256'].items():assert sha(R/name)==h
 assert all(summary['source_sha256'][name]==h for name,h in r['source_sha256'].items())
 assert r['captured_layers_by_batch']=={mode:{str(b):list(range(36)) for b in range(4,9)} for mode in ('witness','checked','production')}
 build=r['build_receipt'];assert build['identity']=='28ecb045c76babdb' and build['native_group']==4 and build['fold']==4
 for name,h in build['generated_sha256'].items():assert sha(R/'build/xqa_raw_grouped_hint'/build['identity']/name)==h
 fp=r['execution_fingerprint'];assert fp['attention_shape']==[36,16,4,128,16] and fp['TP']==2 and fp['native_table_columns']==10240
 assert fp['model_path']=='data/models/Qwen3-4B-Thinking-2507'
 local_config=json.loads((R/fp['model_path']/'config.json').read_text())
 assert local_config['num_hidden_layers']==36 and local_config['num_attention_heads']==32 and local_config['num_key_value_heads']==8
 plan=r['plan'];assert plan['workspace_allocation_sets']==plan['raw_state_allocation_sets']==1 and plan['layers']==36
 for b in range(4,9):
  p=plan['plans'][str(b)];assert p['native_batch']==b and p['dummy_prefix_rows']==(4-b%4)%4
  expected_s=min(max(1,fp['SM']//(b*4)),(10240*16+255)//256)
  assert p['native_splits']==expected_s and p['prefix_SM']==expected_s*((b+3)//4)*4 and p['tail_SM']==expected_s*b*4
 eligible=collections.defaultdict(list)
 for row in rows:
  assert row['native_graph_replayed'] and not row['candidate_graph_replayed']
  if in_scope(row['batch'],120528,[x+1 for x in row['positions']]):eligible[row['batch']].append(row['positions'])
 counts=collections.Counter();signature=[]
 for i,c in enumerate(r['checks']):
  assert c['step']==i and c['qualification_passed'] and all(c.get(f) is True for f in REQUIRED_QUALIFICATION_FIELDS)
  b=c['batch'];assert c['positions']==eligible[b][counts[b]];counts[b]+=1
  prep=r['prepare_records'][i];assert prep['positions']==c['positions'] and prep['batch']==b and prep['development_scope_eligible']
  assert prep['native_GPU_lengths_aliased'] and not prep['host_length_copy']
  assert len(c['layer_checks'])==36
  for l,lc in enumerate(c['layer_checks']):assert lc=={'layer':l,'Q_bits_equal':True,'O_bits_equal':True}
  assert [m['mode'] for m in c['modes']]==['checked','production']
  assert all(value is True for m in c['modes'] for key,value in m.items() if key!='mode')
  signature.append((b,c['positions']))
 assert counts==collections.Counter({b:min(8,len(eligible[b])) for b in range(4,9)})==collections.Counter({b:8 for b in range(4,9)})
 steps_by_rank.append(signature);graph_signatures.append([(x['batch'],x['questions'],x['positions']) for x in rows])
 outputs.append({'rank':rank,'record':str(path.relative_to(R)),'sha256':sha(path),'graph_sha256':sha(gp),'verified_states':40,'by_batch':dict(counts),'native_natural_graph_steps':len(rows),'native_eligible_steps_by_batch':{b:len(eligible[b]) for b in range(4,9)},'verified_layer_QO_pairs':1440,'both_candidate_modes_verified':True,'shared_workspace_bytes':plan['workspace_bytes'],'shared_raw_state_bytes':plan['raw_state_bytes']})
assert steps_by_rank[0]==steps_by_rank[1] and graph_signatures[0]==graph_signatures[1]
assert records[0]['source_sha256']==records[1]['source_sha256'] and records[0]['execution_fingerprint']==records[1]['execution_fingerprint']
offline=validate_offline_records(records,records[0]['source_sha256'],records[0]['execution_fingerprint'],[4,5,6,7,8],2)
result={'status':'passed_for_bounded_production_trial','summary_path':str((D/'summary.json').relative_to(R)),'summary_sha256':sha(D/'summary.json'),'ranks':outputs,'global_model_states_verified_on_both_ranks':40,'rank_states':80,'layer_QO_pairs':2880,'offline_validation':offline,'scope':receipt(),'limitation':'Native trajectory and first8 naturally encountered states per enabled batch. This qualifies the bounded next production trial, not all later states, output trajectories, other prefixes or a speedup. Natural ABBA must independently retain token equality and report setup/candidate coverage.'}
p=R/'results/gqa4_quad_model_qualification_independent_audit_v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'rank_states':80,'layer_QO_pairs':2880,'counts':outputs[0]['by_batch']}))
