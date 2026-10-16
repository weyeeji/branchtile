"""CPU provenance, transfer, ownership and full-cost accounting contract checks."""
import copy,inspect,json,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.raw_pair_direct_training_shape_transfer import validate_training_shapes,TRAINING_PROTOCOL,load_training_shapes
from src.vllm_native_raw_pair_general_setup import REQUIRED_QUALIFICATION_FIELDS
from scripts.vllm_raw_pair_direct_transfer_worker import RawPairDirectTransferWorker
from scripts.vllm_raw_pair_direct_multidoc_production_worker import RawPairDirectMultiDocumentProductionWorker
from scripts.vllm_raw_pair_direct_multidoc_worker import source_hashes
from scripts.benchmark_raw_pair_direct_next8_transfer import first_task_components

def main():
    protocol=json.loads(TRAINING_PROTOCOL.read_text());sources={'test':'frozen'};fingerprint={'model':'same'};records=[]
    for doc,batches in zip(protocol['documents'],([6,8],[7,9])):
        for rank in (0,1):
            checks=[]
            for b in batches:
                check={k:True for k in REQUIRED_QUALIFICATION_FIELDS}
                check.update(batch=b,qualification_passed=True,layer_checks=[{'layer':l,'Q_bits_equal':True,'O_bits_equal':True} for l in range(48)],modes=[{'mode':m,'hidden_bits_equal':True} for m in ('checked','production')]);checks.append(check)
            records.append({'document_id':doc['document_id'],'rank':rank,'source_sha256':sources,'execution_fingerprint':fingerprint,
                'validation':True,'method':'native','phase_valid':True,'capture_invocations':1,'enabled_batches':protocol['candidate_enabled_batches'],
                'plan':{'nominal_active_prefix_tokens':doc['common_complete_page_tokens']},'prepare_records':[{'policy_prefix_tokens':doc['common_complete_page_tokens']}],
                'setup_mode':'strict','qualification_passed':True,'qualification_steps':len(checks),'checks':checks,
                'captured_layers_by_batch':{m:{str(b):list(range(48)) for b in batches} for m in ('witness','checked','production')}})
    enabled=list(range(6,20));result=validate_training_shapes(records,protocol,sources,fingerprint,enabled)
    assert result['eligible_training_qualified_shapes']==[6,7,8,9] and result['evaluation_document_locally_qualified'] is False
    partial=copy.deepcopy(records);partial[1]['checks']=partial[1]['checks'][:1];partial[1]['qualification_steps']=1
    assert validate_training_shapes(partial,protocol,sources,fingerprint,enabled)['eligible_training_qualified_shapes']==[6,7,9]
    flags={'per_rank_training_union_then_rank_intersection':True,'missing_shapes_fallback':True}
    for name,mutate in [('foreign_document',lambda x:x[0].update(document_id='eval-document')),('stale_source',lambda x:x[0].update(source_sha256={})),('failed_layer',lambda x:x[0]['checks'][0]['layer_checks'][3].update(O_bits_equal=False))]:
        bad=copy.deepcopy(records);mutate(bad)
        try:validate_training_shapes(bad,protocol,sources,fingerprint,enabled)
        except ValueError:flags['reject_'+name]=True
        else:raise AssertionError(name)
    old=ROOT/'results/raw_pair_multidoc_model_qual_v1/summary.json'
    if old.exists():
        try:load_training_shapes(old,source_hashes(),enabled)
        except ValueError:flags['reject_actual_old_constituent_qualification']=True
        else:raise AssertionError('Accepted old constituent qualification')
    bank=types.SimpleNamespace(activate_prefix=lambda p,upload:{'prefix':p,'uploaded':upload})
    docs={d['document_id']:dict(d,B=len(d['prompt_token_hashes'])) for d in protocol['documents']}
    state={'mode':None,'method':'native','armed':False,'documents':docs,'plan':bank,'config':{},'document_generation':0,'qualified_shapes_from_training':{6,8},'document_activations':[]}
    probe={'requests':{'old':1},'rank':0};worker=types.SimpleNamespace(raw_pair=state,native_probe=probe)
    for d in docs:
        state.update(cohort=['stale'],epochs={1:3},pending={'old':1},live_tables=[1],live_slots=[2]);probe['requests']['old']=1
        receipt=RawPairDirectTransferWorker.activate_raw_pair_document(worker,d,True)
        assert all(not state[k] for k in ('cohort','epochs','pending','live_tables','live_slots')) and probe['requests']=={}
        assert receipt['document_locally_qualified'] is False and receipt['eligible_training_qualified_batches']==[6,8]
    flags['cross_doc_activation_clears_ownership_without_claiming_local_qualification']=True
    for method in ('capture_dense_cascade_graphs','set_dense_cascade_phase','arm_raw_pair_qualification','_qualify_raw_pair_state'):
        assert inspect.getsource(getattr(RawPairDirectTransferWorker,method))==inspect.getsource(getattr(RawPairDirectMultiDocumentProductionWorker,method))
    anchor='        assert runner.batch_sharder is None'
    a=inspect.getsource(RawPairDirectTransferWorker.configure_dense_cascade);b=inspect.getsource(RawPairDirectMultiDocumentProductionWorker.configure_dense_cascade)
    assert a[a.index(anchor):]==b[b.index(anchor):]
    flags['computational_hooks_and_single_capture_unchanged']=True
    source=(ROOT/'scripts/benchmark_raw_pair_direct_next8_transfer.py').read_text()
    assert source.index("identities=llm.collective_rpc('raw_pair_device_identity')")<source.index("energy=PhaseEnergyCounter")<source.index("energy.sample('before_candidate_configuration')")<source.index("llm.collective_rpc('configure_dense_cascade'")<source.index("llm.collective_rpc('capture_dense_cascade_graphs')")
    flags['UUID_counter_init_precedes_candidate_setup']=True
    r={'runs':[],'offline_preflight_wall_seconds':1.,'candidate_setup_instrumented_wall_seconds':2.,'common_energy_initialization_and_identity_RPC_wall_seconds':.1,'common_precision_export_instrumented_wall_seconds':.2,'input_preparation_seconds':3.,'native_initialization_and_graph_setup_seconds':4.,'setup_energy_intervals':{'candidate_configuration_and_capture':{'total_GPU_joules':20.},'common_precision_receipt_export':{'total_GPU_joules':2.}}}
    for i,m in enumerate(('native','cascade','cascade','native')):
        for _ in range(2):r['runs'].append({'phase':f'natural_{m}_{i}','full_phase_control_wall_seconds':10. if m=='native' else 8.,'energy_intervals':{'complete_document_through_export':{'total_GPU_joules':100. if m=='native' else 80.}}})
    totals=first_task_components(r);assert totals['native_seconds']==20.3 and totals['candidate_seconds']==19.3
    assert totals['native_GPU_joules']==202 and totals['candidate_GPU_joules']==182
    r['setup_energy_intervals']['candidate_configuration_and_capture']['total_GPU_joules']=None
    assert first_task_components(r)['candidate_GPU_joules'] is None
    flags['full_phase_controls_setup_energy_once_and_null_preserved']=True
    assert not torch.cuda.is_initialized()
    print(json.dumps({'status':'passed','cuda_initialized':False,'checks':flags,'limitations':'Mock receipt/activation fixtures exercise contracts, not numerical model qualification.'},indent=2))
if __name__=='__main__':main()
