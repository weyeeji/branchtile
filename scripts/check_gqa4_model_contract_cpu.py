"""CPU guard checks: separate36-layer offline qualification and model geometry."""
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.vllm_gqa4_raw_pair_setup import REQUIRED_QUALIFICATION_FIELDS,validate_offline_records,capture_modes


def main():
    source={'example':'frozen'};fingerprint={'attention_shape':[36,16,4,128,16]}
    check={k:True for k in REQUIRED_QUALIFICATION_FIELDS}
    check.update(qualification_passed=True,batch=8,
                 layer_checks=[{'layer':i,'Q_bits_equal':True,'O_bits_equal':True} for i in range(36)],
                 modes=[{'mode':m,'hidden_bits_equal':True} for m in ['checked','production']])
    records=[{'rank':rank,'source_sha256':source,'execution_fingerprint':fingerprint,'setup_mode':'strict',
              'qualification_passed':True,'qualification_steps':1,'checks':[copy.deepcopy(check)],
              'captured_layers_by_batch':{m:{'8':list(range(36))} for m in ['witness','checked','production']}} for rank in [0,1]]
    validate_offline_records(records,source,fingerprint,[8],2)
    checks={'valid36layer_receipt':True,'strict_modes':capture_modes('strict')==('witness','checked','production'),
            'production_modes':capture_modes('production_only')==('production',)}
    variants={}
    v=copy.deepcopy(records);v[0]['checks'][0]['layer_checks'] += [{'layer':36,'Q_bits_equal':True,'O_bits_equal':True}];variants['extra_layer_rejected']=v
    v=copy.deepcopy(records);v[0]['checks'][0]['all36_attention_O_bits_equal']=False;variants['failed_O_rejected']=v
    v=copy.deepcopy(records);v[0]['source_sha256']={'example':'stale'};variants['stale_source_rejected']=v
    v=copy.deepcopy(records);v[0]['execution_fingerprint']={'attention_shape':[48,16,2,128,16]};variants['old_geometry_rejected']=v
    variants['missing_rank_rejected']=records[:1]
    for name,variant in variants.items():
        try:validate_offline_records(variant,source,fingerprint,[8],2)
        except ValueError:checks[name]=True
        else:checks[name]=False
    assert all(checks.values()),checks
    for name in ['Qwen3-4B-Instruct-2507','Qwen3-4B-Thinking-2507']:
        cfg=json.loads((ROOT/'data/models'/name/'config.json').read_text())
        assert (cfg['num_attention_heads']//2,cfg['num_key_value_heads']//2,cfg['head_dim'],cfg['num_hidden_layers'])==(16,4,128,36)
    print(json.dumps({'status':'passed','checks':checks},indent=2))
if __name__=='__main__':main()
