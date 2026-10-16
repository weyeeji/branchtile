"""Independent CPU audit of actual native-state RawPair bitwise qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.vllm_native_raw_pair_contract import REQUIRED_QUALIFICATION_FIELDS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(path):
    summary=json.loads(path.read_text())
    result={'status':'incomplete','summary_sha256':digest(path),'summary_status':summary['status'],
        'audit_source_sha256':digest(Path(__file__)),'ranks':[],'failures':[],
        'scope':'Actual native-trajectory same-state bitwise qualification. No performance or general model-quality claim.'}
    for name,expected in summary['source_sha256'].items():
        assert digest(ROOT/name)==expected,name
    result['runtime_source_hashes_match']=True
    runs=[r for r in summary['runs'] if r['validation']]
    if not runs:
        result['error']=summary.get('error','Qualification phase has not completed')
        return result
    assert len(runs)==1
    run=runs[0]
    requested=summary['args']['qualification_steps']
    assert run['method']=='native' and not run['natural_eos_enabled']
    assert run['sampling']['ignore_eos'] and run['sampling']['min_tokens']==run['sampling']['max_tokens']==requested+1
    assert len(run['answers'])==8 and all(len(a['tokens'])==requested+1 for a in run['answers'])
    assert len(run['workers'])==2
    trajectories=[]
    build_identities=[]
    for worker in run['workers']:
        source=Path(worker['raw_pair_record'])
        record=json.loads(source.read_text())
        graph=json.loads(Path(worker['graph_metadata']).read_text())
        assert worker['rank']==record['rank']==graph['rank']
        assert record['method']=='native' and record['validation'] and record['mechanism']=='raw_pair_native_state_continuation'
        assert record['qualification_steps']==worker['qualification_steps']==len(record['checks'])
        assert record['qualification_requested_steps']==requested
        if len(record['checks'])!=requested:
            result['failures'].append({'rank':record['rank'],'reason':'observed_qualification_step_count','observed':len(record['checks']),'required':requested})
        assert len(record['checks'])==len(record['prepare_records'])
        assert record['candidate_replays']==worker['candidate_replays']==2*len(record['checks'])
        assert record['native_replays']==worker['native_replays']==len(graph['replays'])
        assert all(r['native_graph_replayed'] and not r['candidate_graph_replayed'] for r in graph['replays'])
        assert record['captured_B8_layers']=={mode:list(range(48)) for mode in ('witness','checked','production')}
        for name,expected in record['source_sha256'].items():
            assert digest(ROOT/name)==expected,name
        build=record['build_receipt']
        build_identities.append(build['identity'])
        generated=ROOT/'build/xqa_raw_pair'/build['identity']
        for name,expected in build['generated_sha256'].items():
            assert digest(generated/name)==expected,(name,'generated build source changed')
        plan=record['plan']
        assert plan['layers']==48 and plan['workspace_allocation_sets']==plan['raw_state_allocation_sets']==1
        assert plan['native_table_shape'][0]==8 and plan['native_table_stride']==[plan['native_table_shape'][1],1]
        assert plan['native_launch_extent']==plan['native_table_shape'][1]*16
        splits=min(max(1,plan['native_SM']//16),(plan['native_launch_extent']+255)//256)
        assert plan['native_splits']==splits and plan['prefix_SM']==splits*8
        assert plan['prefix_tiles']==summary['common_prefix_tokens']//256
        assert plan['shared_tile_tokens']==plan['prefix_tiles']*256
        assert sorted(map(int,plan['query_layouts']))==list(range(48))
        fields={field:0 for field in REQUIRED_QUALIFICATION_FIELDS}
        positions=[]
        failed_layers=[]
        for index,(step,prepared) in enumerate(zip(record['checks'],record['prepare_records'],strict=True)):
            assert step['step']==index and step['batch']==prepared['batch']==prepared['actual_requests']==prepared['padded_tokens']==8
            assert step['positions']==prepared['positions']
            assert prepared['lengths']==[p+1 for p in prepared['positions']]
            assert prepared['host_physical_prefix_check'] and prepared['metadata_copy_cost_included']
            assert prepared['layers_sharing_state_and_scratch']==48 and prepared['native_splits']==splits
            assert prepared['native_table_columns']==plan['native_table_shape'][1]
            assert prepared['complete_prepare_host_seconds']>=0
            assert [r['layer'] for r in step['layer_checks']]==list(range(48))
            assert [m['mode'] for m in step['modes']]==['checked','production']
            positions.append(step['positions'])
            for field in REQUIRED_QUALIFICATION_FIELDS:
                if step.get(field) is True:
                    fields[field]+=1
                else:
                    result['failures'].append({'rank':record['rank'],'step':index,'field':field,'value':step.get(field)})
            for layer in step['layer_checks']:
                if not (layer['Q_bits_equal'] and layer['O_bits_equal']):
                    failed_layers.append({'step':index,**layer})
            mode_pass=all(all(value is True for key,value in m.items() if key!='mode') for m in step['modes'])
            expected_pass=all(step.get(field) is True for field in REQUIRED_QUALIFICATION_FIELDS)
            assert step['qualification_passed']==expected_pass
            if not mode_pass:
                result['failures'].append({'rank':record['rank'],'step':index,'mode_checks':step['modes']})
        for before,after in zip(positions,positions[1:]):
            assert after==[p+1 for p in before],'Qualification must continue actual native autoregressive positions'
        graph_positions=[r['positions'] for r in graph['replays'] if r['batch']==8]
        assert positions==graph_positions[:len(positions)]
        trajectories.append(positions)
        result['ranks'].append({'rank':record['rank'],'raw_record_sha256':digest(source),
            'observed_native_states':len(record['checks']),'checked_layer_QO_pairs':len(record['checks'])*48,
            'required_field_true_counts':fields,'failed_layer_checks':failed_layers,
            'native_sampler_graph_replays':record['native_replays'],'candidate_checked_and_production_replays':record['candidate_replays'],
            'fallbacks':record['fallbacks'],'build_identity':build['identity'],'generated_build_hashes_verified':True,
            'plan':plan,'total_qualification_prepare_host_seconds':sum(p['complete_prepare_host_seconds'] for p in record['prepare_records'])})
    assert trajectories[0]==trajectories[1] and len(set(build_identities))==1
    result['rank_native_positions_equal']=True
    result['requested_steps_per_rank']=requested
    result['status']='qualified' if not result['failures'] else 'rejected'
    result['costs']={'native_initialization_and_graph_setup_seconds':summary['native_initialization_and_graph_setup_wall_seconds'],
        'candidate_configuration_seconds':summary['candidate_configuration_wall_seconds'],
        'candidate_three_mode_graph_capture_seconds':summary['candidate_graph_capture_wall_seconds'],
        'qualification_full_phase_control_seconds':run['full_phase_control_wall_seconds'],
        'qualification_anchor_seconds':run['anchor_wall_seconds'],'qualification_generate_seconds':run['complete_generate_wall_seconds']}
    ledger=[json.loads(line) for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines()]
    ends=[r for r in ledger if r.get('event')=='end' and '--output' in r.get('argv',[]) and
          (ROOT/r['argv'][r['argv'].index('--output')+1]).resolve()==path.parent.resolve()]
    result['ledger_terminal_receipts']=ends
    if result['status']=='qualified':
        assert summary['status']=='completed'
        assert len(ends)==1 and ends[0]['exit_code']==0 and not ends[0]['timed_out']
    result['limits']=['Native witness graph records layer Q/O; candidate checked graph must match every layer. Uninstrumented native and witness final hidden are independently bitwise checked.',
        'Qualification graph copies, hidden/logit checks and KV snapshots are experimental overhead and are absent from production graphs.',
        'Current prototype requires session qualification before enabling candidate. Actual first activation therefore includes this cost; it is not silently free setup.',
        'Only actual B8 was qualified. No B21/B19 execution or naturally shrinking-batch performance follows from this gate.']
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    assert not output.exists(),output
    report=audit(Path(args.input))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'ranks':[{'rank':r['rank'],'states':r['observed_native_states']} for r in report['ranks']],
                      'failures':report['failures'][:8]},indent=2))


if __name__=='__main__':
    main()
