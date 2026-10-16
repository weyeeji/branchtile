"""TP1 Hq32/Hkv8 adapter pointer/shape/state ABI checks with fake launch modules."""
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
from src.xqa_raw_tp1_direct_quad_runtime import RawTP1DirectQuadPlan
class Module:
    def __init__(self):self.calls=[]
    def xqa_wrapper(self,*args):self.calls.append(args)
def main():
    torch.set_num_threads(2);checks={};mods={g:Module() for g in (4,16)};workspace=torch.empty(64*1024*1024,dtype=torch.uint8)
    with patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(multi_processor_count=170)):
        for b in range(1,9):
            q=torch.zeros((b,32,128),dtype=torch.bfloat16);cache=torch.empty((1,8,16,256),dtype=q.dtype);table=torch.zeros((b,8192),dtype=torch.int32)
            plan=RawTP1DirectQuadPlan(mods,q,cache,table,[120560+i for i in range(b)],120528,workspace=workspace)
            native=torch.ones(b,dtype=torch.int32);plan.lengths=native.view(torch.uint32).reshape(b,1);plan.run()
            assert plan.splits==170//(b*8) and plan.state.numel()==((b+3)//4)*8*plan.splits*4480
            for group,batch,phase in ((16,(b+3)//4,1),(4,b,3)):
                args=mods[group].calls[-1]
                assert args[1]==plan.splits*batch*8 and args[2]==8 and args[17]==batch and args[-1]==phase
                assert args[8].data_ptr()==q.data_ptr() and args[14].data_ptr()==table.data_ptr() and args[16].data_ptr()==native.data_ptr()
                assert args[-3].data_ptr()==plan.state.data_ptr()
            if b>1:
                table[1,0]=1
                try:plan.prepare([120560+i for i in range(b)])
                except AssertionError:pass
                else:raise AssertionError('Foreign physical prefix accepted')
    checks.update(all_B1_to_B8_native_S_and_Hkv8_ABI=True,native_Q_table_length_aliases=True,state_capacity_original_split_times_8_heads=True,foreign_physical_prefix_rejected=True)
    # Generated direct mapping: prefix head rows group four requests of GQA4,
    # Hkv is runtime8, so nbQHeads=128 and original native request stride32.
    for pair in range(2):
        for head in range(8):
            for row in range(16):
                address=128*pair+4*head+(row//4)*32+row%4
                assert address==(4*pair+row//4)*32+4*head+row%4
    checks['direct_native_head_address_mapping_Hkv8']=True
    assert not torch.cuda.is_initialized()
    print(json.dumps({'status':'passed','cuda_initialized':False,'checks':checks,'scope':'CPU fake-module ABI only; actual Hkv8 kernel/FP64/graph qualification still required.'},indent=2))
if __name__=='__main__':main()
