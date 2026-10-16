"""CPU-only structural and energy-null tests, not a completed-run audit."""
import copy,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
from scripts.audit_raw_pair_next8_transfer_cpu import source_contract,counter_difference

def main():
    driver=ROOT/'scripts/benchmark_raw_pair_direct_next8_transfer.py';worker=ROOT/'scripts/vllm_raw_pair_direct_transfer_worker.py';bank=ROOT/'src/vllm_raw_pair_direct_multidoc_plan.py'
    flags=source_contract(driver,worker,bank)
    with tempfile.TemporaryDirectory() as dirname:
        p=Path(dirname)/'worker.py';p.write_text(worker.read_text().replace("probe['requests'].clear()",'pass'))
        try:source_contract(driver,p,bank)
        except AssertionError:flags['reject_removed_request_owner_clear']=True
        else:raise AssertionError('Missing owner clearing accepted')
        p.write_text(worker.read_text().replace("state['capture_invocations']==0",'True'))
        try:source_contract(driver,p,bank)
        except AssertionError:flags['reject_removed_single_capture_guard']=True
        else:raise AssertionError('Missing capture guard accepted')
    a={'devices':[{'physical_selector':'2','total_energy_mJ':1000,'UUID':'GPU-a','expected_worker_UUID':'GPU-a','identity_verified':True}]}
    b={'devices':[dict(a['devices'][0],total_energy_mJ=2200)]}
    assert counter_difference(a,b)==1.2
    for mutation in ({'identity_verified':False},{'total_energy_mJ':None},{'total_energy_mJ':999},{'UUID':'GPU-b'}):
        bad=copy.deepcopy(b);bad['devices'][0].update(mutation);assert counter_difference(a,bad) is None
    flags['energy_UUID_null_and_counter_decrease_cases']=True
    import torch
    assert not torch.cuda.is_initialized()
    print(json.dumps({'status':'passed','cuda_initialized':False,'checks':flags,'limitations':'Structural/energy arithmetic unit checks only. Full terminal auditor remains to be exercised on actual next8 completed output; no actual new-document result is implied.'},indent=2))
if __name__=='__main__':main()
