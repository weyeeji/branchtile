"""Actual-receipt mutation controls for the independent multi-document auditor."""
import contextlib,copy,io,json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_raw_pair_multidoc_production_cpu import main as audit,source_contract


def main():
    source=ROOT/'results/raw_pair_multidoc_production_v1/summary.json';summary=json.loads(source.read_text())
    complete=[x for x in summary['runs'] if 'graph_trajectory_sha256_by_rank' in x]
    assert len(complete)>=4
    # Preserve only four already-complete phases; this fixture intentionally remains interim.
    summary['runs']=complete[:4];summary['status']='initializing';checks={}
    with tempfile.TemporaryDirectory() as dirname:
        temp=Path(dirname)
        for name in ('positive','cut_pointer_changed','generation_reused','missing_physical_rebind','foreign_document_receipt','second_capture'):
            obj=copy.deepcopy(summary)
            if name=='generation_reused':obj['runs'][2]['activations'][0]['generation']=1
            if name in ('cut_pointer_changed','missing_physical_rebind','foreign_document_receipt','second_capture'):
                worker=obj['runs'][2]['workers'][0];raw=json.loads(Path(worker['raw_pair_record']).read_text())
                if name=='cut_pointer_changed':raw['plan']['shared_cut_ptr']+=16
                if name=='missing_physical_rebind':raw['prepare_records'][0]['host_physical_prefix_check']=False
                if name=='foreign_document_receipt':raw['document_id']=obj['runs'][3]['document_id']
                if name=='second_capture':raw['capture_invocations']=2
                path=temp/(name+'_raw.json');path.write_text(json.dumps(raw));worker['raw_pair_record']=str(path)
            sp=temp/(name+'_summary.json');sp.write_text(json.dumps(obj));out=temp/(name+'_audit.json')
            args=['audit','--summary',str(sp),'--variant','dynamic','--allow-interim','--output',str(out)]
            failed=False
            try:
                with patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):audit()
            except AssertionError:failed=True
            result=json.loads(out.read_text())
            if name=='positive':assert not failed and result['status']=='interim_exported_runs_audited'
            else:assert failed and result['status']=='audit_failed'
            checks[name]=True
    source_contract(ROOT/'scripts/benchmark_raw_pair_direct_multidoc_production.py',ROOT/'scripts/vllm_raw_pair_direct_multidoc_production_worker.py',ROOT/'src/vllm_raw_pair_direct_multidoc_plan.py')
    checks['combined_variant_static_single_capture_ownership_contract']=True
    import torch
    assert not torch.cuda.is_initialized()
    print(json.dumps({'status':'passed','cuda_initialized':False,'checks':checks,'scope':'Actual existing receipt mutations in temporary files; original GPU results and sources untouched.'},indent=2))
if __name__=='__main__':main()
