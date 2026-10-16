"""Independent terminal receipt audit of the combined direct/device-cut strict experiment."""
import argparse,collections,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    output=Path(a.output);assert not output.exists();output.parent.mkdir(parents=True,exist_ok=True)
    from src.raw_pair_direct_multidoc_offline import load_completed_qualification
    from scripts.vllm_raw_pair_direct_multidoc_worker import source_hashes
    from scripts.benchmark_native_raw_pair_general_policy_e2e import audit_qualification
    protocol=json.loads((ROOT/'configs/raw_pair_multidoc_training_v1.json').read_text())
    loaded=load_completed_qualification(a.summary,protocol,ROOT,source_hashes())
    summary=json.loads(Path(a.summary).read_text());assert len(summary['runs'])==2
    old=json.loads((ROOT/'results/raw_pair_multidoc_model_qual_v1/summary.json').read_text())
    old_tokens={r['document_id']:[x['tokens'] for x in r['answers']] for r in old['runs'] if r['validation']}
    result={'status':'auditing','summary_sha256':digest(a.summary),'auditor_sha256':digest(__file__),'records':[],'limits':['Strict training qualification checks the first at most eight naturally observed states per enabled batch, not every future state.','Checked/production shadow graphs are measured research overhead, not production performance.','Pointer/switch provenance is receipt/source evidence, not an independent CUDA event trace.']}
    cutptr={c['rank']:c['plan']['shared_cut_ptr'] for c in summary['capture']};assert set(cutptr)=={0,1}
    for config in summary['configuration']:
        receipt=config['raw_pair_build_receipt'];directory=ROOT/'build/xqa_raw_direct_dynamic_cut'/receipt['identity']
        for path,expected in receipt['generated_sha256'].items():assert digest(directory/path)==expected
    for index,run in enumerate(summary['runs']):
        assert run['validation'] and run['method']=='native' and run['natural_eos_enabled']
        recalculated=audit_qualification(run['workers'],protocol['candidate_enabled_batches'],8,run['P'])
        assert json.loads(json.dumps(recalculated))==run['qualification_joint_audit']
        assert [x['tokens'] for x in run['answers']]==old_tokens[run['document_id']]
        assert not run['cap_hits'] and run['total_request_preemptions']==0
        assert len(set(run['graph_trajectory_sha256_by_rank']))==1
        for worker in run['workers']:
            raw=json.loads(Path(worker['raw_pair_record']).read_text());rank=raw['rank'];plan=raw['plan']
            assert raw['capture_invocations']==1 and raw['document_generation']==index+1
            assert plan['shared_cut_ptr']==cutptr[rank] and plan['nominal_active_prefix_tokens']==run['P']
            assert len(raw['document_activations'])==index+1 and raw['document_activations'][-1]['document_id']==run['document_id']
            for by_layer in plan['query_layouts'].values():
                for layout in by_layer.values():assert layout['Q_native_alias'] and layout['O_native_alias'] and not layout['Q_canonicalization_copy_charged']
            result['records'].append({'document_id':run['document_id'],'rank':rank,'raw_sha256':digest(worker['raw_pair_record']),
                'states':len(raw['checks']),'layer_QO_pairs':sum(len(x['layer_checks']) for x in raw['checks']),
                'batches':sorted({x['batch'] for x in raw['checks']}),'all_required_fields_and_graph_modes_passed':True,'capture_invocations':1,'generation':index+1})
    result['total_states']=sum(x['states'] for x in result['records']);result['total_layer_QO_pairs']=sum(x['layer_QO_pairs'] for x in result['records'])
    result['by_document']=loaded['by_document'];result['all_native_tokens_equal_to_prior_native_training_trajectory']=True
    import torch
    assert not torch.cuda.is_initialized();result['cuda_initialized']=False;result['status']='combined_strict_qualification_audited'
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','total_states','total_layer_QO_pairs')}))
if __name__=='__main__':main()
