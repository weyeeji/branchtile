"""CPU-only decision checks; does not initialize CUDA or claim model readiness."""
import json
from dataclasses import replace
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.vllm_native_raw_pair_contract import DecodeAdmission, assess_admission, buffer_contract, qualify_same_state_receipt, REQUIRED_QUALIFICATION_FIELDS
base=DecodeAdmission(8,8,8,8,True,8,10240,10240,1,'torch.int32','torch.bfloat16','torch.bfloat16',common_prefix_tokens=120528,physical_prefix_validated=True,cohort_epoch_unchanged=True)
d=assess_admission(base)
assert d['eligible'] and d['native_splits']==10 and d['prefix_tiles']==470 and d['shared_tokens_reprocessed_in_private_tail']==208
assert d['native_launch_extent']==163840 and d['prefix_sm_argument']==80
changes=[{'actual_requests':7},{'actual_tokens':16},{'padded_tokens':16},{'query_rows':16},{'page_table_dtype':'torch.int64'}, {'page_table_row_stride':20480},{'cache_dtype':'torch.float8_e4m3fn'},{'physical_prefix_validated':False},{'cohort_epoch_unchanged':False},{'common_prefix_tokens':128},{'decode_query_length':2},{'has_sink_or_custom_mask':True},{'dcp_world_size':2},{'unit_kv_scales':False},{'full_graph':False}]
for patch in changes:assert not assess_admission(replace(base,**patch))['eligible'],patch
for b in range(1,8):assert not assess_admission(replace(base,actual_requests=b,actual_tokens=b,padded_tokens=b,query_rows=b,page_table_rows=b))['eligible']
assert qualify_same_state_receipt({k:True for k in REQUIRED_QUALIFICATION_FIELDS})
assert not qualify_same_state_receipt({k:True for k in REQUIRED_QUALIFICATION_FIELDS[:-1]})
r={'status':'CPU_contract_only_not_runnable_model_adapter','admitted':d,'fallback_cases_checked':len(changes)+7,'buffer_contract':buffer_contract(base)}
path=ROOT/'results/native_raw_pair_model_contract_v1.json';path.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
