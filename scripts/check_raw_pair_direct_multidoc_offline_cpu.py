"""CPU negative provenance cases for document-indexed production graph reuse."""
import copy,inspect,json,sys,tempfile,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.raw_pair_direct_multidoc_offline import validate_document_records,load_completed_qualification
from src.vllm_native_raw_pair_general_setup import REQUIRED_QUALIFICATION_FIELDS
from scripts.vllm_raw_pair_direct_multidoc_worker import RawPairDirectMultiDocumentWorker
from scripts.vllm_raw_pair_direct_multidoc_production_worker import RawPairDirectMultiDocumentProductionWorker


def main():
    docs=[{'document_id':'A','common_complete_page_tokens':135376},{'document_id':'B','common_complete_page_tokens':123808}]
    enabled=[6,7,8,9];sources={'frozen_runtime':'correct'};fingerprint={'model':'same'};records=[]
    for doc,batch in zip(docs,(8,9)):
        for rank in (0,1):
            check={k:True for k in REQUIRED_QUALIFICATION_FIELDS}
            check.update(batch=batch,qualification_passed=True,layer_checks=[{'layer':l,'Q_bits_equal':True,'O_bits_equal':True} for l in range(48)],
                         modes=[{'mode':m,'hidden_bits_equal':True} for m in ('checked','production')])
            records.append({'document_id':doc['document_id'],'rank':rank,'source_sha256':sources,'execution_fingerprint':fingerprint,
                'validation':True,'method':'native','phase_valid':True,'capture_invocations':1,'enabled_batches':enabled,
                'plan':{'nominal_active_prefix_tokens':doc['common_complete_page_tokens']},'prepare_records':[{'policy_prefix_tokens':doc['common_complete_page_tokens']}],
                'setup_mode':'strict','qualification_passed':True,'qualification_steps':1,'checks':[check],
                'captured_layers_by_batch':{m:{str(batch):list(range(48))} for m in ('witness','checked','production')}})
    result=validate_document_records(records,docs,sources,fingerprint,enabled)
    assert result['A']['qualified_batches']==[8] and result['B']['qualified_batches']==[9]
    assert result['A']['unqualified_batches_fallback_native']==[6,7,9]
    checks={'valid_doc_indexed_partial_coverage_fallback':True}
    mutations={
      'reject_missing_rank':lambda r:r.pop(),
      'reject_cross_document':lambda r:r[0].update(document_id='B'),
      'reject_stale_source':lambda r:r[0].update(source_sha256={'frozen_runtime':'old'}),
      'reject_model_change':lambda r:r[0].update(execution_fingerprint={'model':'other'}),
      'reject_prefix_change':lambda r:r[0]['plan'].update(nominal_active_prefix_tokens=123808),
      'reject_failed_layer':lambda r:r[0]['checks'][0]['layer_checks'][0].update(O_bits_equal=False),
      'reject_omitted_capture_layer':lambda r:r[0]['captured_layers_by_batch']['production']['8'].pop(),
      'reject_production_as_qualification':lambda r:r[0].update(setup_mode='production_only'),
      'reject_foreign_prepare_prefix':lambda r:r[0]['prepare_records'][0].update(policy_prefix_tokens=16),
    }
    for name,mutate in mutations.items():
        altered=copy.deepcopy(records);mutate(altered)
        try:validate_document_records(altered,docs,sources,fingerprint,enabled)
        except ValueError:checks[name]=True
        else:raise AssertionError(name)
    with tempfile.TemporaryDirectory() as dirname:
        tmp=Path(dirname);runs=[]
        for doc in docs:
            workers=[];audits=[]
            for row in (r for r in records if r['document_id']==doc['document_id']):
                path=tmp/f'{doc["document_id"]}_{row["rank"]}.json';path.write_text(json.dumps(row))
                workers.append({'rank':row['rank'],'raw_pair_record':str(path)})
                audits.append({'rank':row['rank'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            runs.append({'validation':True,'document_id':doc['document_id'],'method':'native','workers':workers,
                         'qualification_joint_audit':audits,'answers':[{'tokens':[1,2]}]})
        protocol={'documents':docs,'candidate_enabled_batches':enabled}
        strict='scripts/benchmark_raw_pair_direct_multidoc.py'
        summary={'status':'completed','protocol':protocol,'source_sha256':{strict:hashlib.sha256((ROOT/strict).read_bytes()).hexdigest()},'runs':runs}
        sp=tmp/'summary.json';sp.write_text(json.dumps(summary))
        loaded=load_completed_qualification(sp,protocol,ROOT,sources)
        assert loaded['by_document']['A']['qualified_batches']==[8]
        rp=Path(runs[0]['workers'][0]['raw_pair_record']);rp.write_text(rp.read_text()+'\n')
        try:load_completed_qualification(sp,protocol,ROOT,sources)
        except ValueError as exc:assert 'bytes changed' in str(exc)
        else:raise AssertionError('Receipt digest replacement accepted')
        checks['completed_summary_and_audited_receipt_bytes_bound']=True
    for method in ('activate_raw_pair_document','capture_dense_cascade_graphs','raw_pair_device_identity','_qualify_raw_pair_state','finish_dense_cascade_phase'):
        assert inspect.getsource(getattr(RawPairDirectMultiDocumentWorker,method))==inspect.getsource(getattr(RawPairDirectMultiDocumentProductionWorker,method))
    old=inspect.getsource(RawPairDirectMultiDocumentWorker.configure_dense_cascade)
    new=inspect.getsource(RawPairDirectMultiDocumentProductionWorker.configure_dense_cascade)
    anchor='        assert runner.batch_sharder is None'
    assert old[old.index(anchor):]==new[new.index(anchor):]
    checks['runtime_hooks_and_capture_unchanged_from_strict']=True
    assert not torch.cuda.is_initialized()
    print(json.dumps({'status':'passed','cuda_initialized':False,'checks':checks,'limits':'Mock provenance fixtures test validation, not model numerical correctness.'},indent=2))
if __name__=='__main__':main()
