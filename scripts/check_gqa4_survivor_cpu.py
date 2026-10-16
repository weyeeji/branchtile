"""CPU scope/offline-coverage guards for the independent GQA4 survivor experiment."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.gqa4_survivor_scope import in_scope,receipt
from src.vllm_gqa4_raw_pair_setup import REQUIRED_QUALIFICATION_FIELDS,validate_offline_records


def main():
    checks={}
    for b in (6,7,8):
        checks[f'b{b}_inside']=in_scope(b,120528,[120529]*(b-1)+[128720])
        checks[f'b{b}_private8193_reject']=not in_scope(b,120528,[128721]*b)
    checks['P_not_120528_reject']=not in_scope(8,120320,[120400]*8)
    checks['B5_reject']=not in_scope(5,120528,[120600]*5)
    checks['prefix_not_full_reject']=not in_scope(8,120528,[120528]*8)
    records=[];fingerprint={'geometry':[36,16,4,128,16],'scope':receipt()};sources={'version':'new-survivor'}
    for rank in (0,1):
        steps=[]
        for b in (6,7,8):
            s={k:True for k in REQUIRED_QUALIFICATION_FIELDS}
            s.update(batch=b,qualification_passed=True,layer_checks=[{'layer':l,'Q_bits_equal':True,'O_bits_equal':True} for l in range(36)],
                     modes=[{'mode':m,'hidden_bits_equal':True} for m in ['checked','production']]);steps.append(s)
        records.append({'rank':rank,'source_sha256':sources,'execution_fingerprint':fingerprint,'setup_mode':'strict','qualification_passed':True,
                        'checks':steps,'qualification_steps':3,'captured_layers_by_batch':{m:{str(b):list(range(36)) for b in (6,7,8)} for m in ['witness','checked','production']}})
    validate_offline_records(records,sources,fingerprint,[6,7,8],2);checks['full36layer_threebatch_offline_pass']=True
    missing=copy.deepcopy(records);missing[1]['checks'].pop(1);missing[1]['qualification_steps']=2
    try:validate_offline_records(missing,sources,fingerprint,[6,7,8],2)
    except ValueError:checks['missing_rank1_B7_rejected']=True
    else:checks['missing_rank1_B7_rejected']=False
    assert all(checks.values()),checks
    print(json.dumps({'status':'passed','checks':checks},indent=2))
if __name__=='__main__':main()
