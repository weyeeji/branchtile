"""CPU admission/provenance tests; this does NOT qualify GPU numerics or speed."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vllm_native_raw_pair_general_plan import state_capacity
from src.vllm_native_raw_pair_general_setup import (
    REQUIRED_QUALIFICATION_FIELDS, candidate_descriptor, capture_modes, validate_offline_records,
)


def main():
    enabled = list(range(6, 22))
    count, sizes = state_capacity(enabled, 170, 10240)
    assert sizes[8]['native_splits'] == 10 and sizes[19]['native_splits'] == 4
    assert sizes[21]['prefix_groups'] == 11 and sizes[7]['prefix_groups'] == 4
    assert count * 4 < 2 * 1024 * 1024
    # Odd padding must not change S to the value selected for the padded B+1.
    assert sizes[7]['native_splits'] == 12 != sizes[8]['native_splits']
    assert capture_modes('strict') == ('witness', 'checked', 'production')
    assert capture_modes('production_only') == ('production',)
    descriptor = dict(cg_mode='FULL',num_tokens=7,num_reqs=7,uniform_token_count=1,
                      max_query_len=1,num_active_loras=0,num_ubatches=1)
    assert candidate_descriptor(SimpleNamespace(**descriptor),'FULL',enabled)
    descriptor_negatives = []
    for key, value in [('cg_mode','PIECEWISE'),('num_tokens',8),('num_reqs',6),
                       ('uniform_token_count',2),('max_query_len',2),
                       ('num_active_loras',1),('num_ubatches',2)]:
        bad = descriptor | {key:value}
        assert not candidate_descriptor(SimpleNamespace(**bad),'FULL',enabled)
        descriptor_negatives.append(key)
    assert not candidate_descriptor(SimpleNamespace(**(descriptor | {'num_tokens':5,'num_reqs':5})),'FULL',enabled)
    source = {'worker':'example_sha'}; fingerprint = {'model':'example_model','SM':170}
    def receipt(rank,batches):
        checks=[]
        for b in batches:
            check={key:True for key in REQUIRED_QUALIFICATION_FIELDS}
            check.update(batch=b,qualification_passed=True,
                layer_checks=[{'layer':i,'Q_bits_equal':True,'O_bits_equal':True} for i in range(48)],
                modes=[{'mode':mode,'hidden_bits_equal':True} for mode in ('checked','production')])
            checks.append(check)
        return {'rank':rank,'source_sha256':source,'execution_fingerprint':fingerprint,
            'setup_mode':'strict','qualification_passed':True,'qualification_steps':len(checks),'checks':checks,
            'captured_layers_by_batch':{mode:{str(b):list(range(48)) for b in batches}
                for mode in ('witness','checked','production')}}
    # Union across documents within each rank is permitted, never across ranks.
    records=[receipt(rank,[b]) for rank in (0,1) for b in (6,7)]
    validate_offline_records(records,source,fingerprint,[6,7],2)
    mutations = {
        'missing_rank': lambda r:r.__delitem__(slice(2,None)),
        'stale_source': lambda r:r[0].update(source_sha256={'worker':'old'}),
        'different_model': lambda r:r[0].update(execution_fingerprint={'model':'other'}),
        'wrong_setup': lambda r:r[0].update(setup_mode='production_only'),
        'failed_summary': lambda r:r[0].update(qualification_passed=False),
        'wrong_step_count': lambda r:r[0].update(qualification_steps=10),
        'failed_lengths': lambda r:r[0]['checks'][0].update(device_lengths_equal_native_and_positions=False),
        'failed_Q': lambda r:r[0]['checks'][0]['layer_checks'][1].update(Q_bits_equal=False),
        'missing_layer': lambda r:r[0]['checks'][0]['layer_checks'].pop(),
        'missing_production_check': lambda r:r[0]['checks'][0]['modes'].pop(),
        'failed_mode': lambda r:r[0]['checks'][0]['modes'][1].update(hidden_bits_equal=False),
        'missing_captured_layer': lambda r:r[0]['captured_layers_by_batch']['production']['6'].pop(),
        'missing_batch_rank': lambda r:r.pop(),
    }
    rejected=[]
    for name,mutate in mutations.items():
        bad=copy.deepcopy(records);mutate(bad)
        try: validate_offline_records(bad,source,fingerprint,[6,7],2)
        except ValueError: rejected.append(name)
        else: raise AssertionError(name)
    report={'status':'CPU_contract_checked_GPU_unqualified','enabled_batches':enabled,
        'state_capacity_floats':count,'state_capacity_bytes':count*4,'per_batch':sizes,
        'descriptor_negative_cases':descriptor_negatives+['B5_fallback'],
        'offline_negative_cases':rejected,'strict_record_union_checked':True,
        'limits':'Synthetic CPU receipts test the gate only; no GPU, model outputs, graph capture, or speed tested.'}
    output=Path('results/native_raw_pair_general_setup_cpu_v1.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(output),'status':report['status'],'negative_cases':len(rejected)+8}))


if __name__=='__main__':main()
