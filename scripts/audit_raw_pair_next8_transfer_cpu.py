"""Independent CPU audit of explicit training-shape transfer and setup-energy graph reuse.

Final ratios require completed ABBA. Interim mode audits only fully exported runs.
Physical page values are not stored: ownership evidence is the frozen runtime
check plus its per-document binding receipt, not an independent page snapshot.
"""
import argparse,ast,hashlib,json,statistics,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_contract(driver,worker,bank):
    source=Path(driver).read_text();tree=ast.parse(source);parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    captures=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='collective_rpc' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='capture_dense_cascade_graphs']
    assert len(captures)==1
    ancestors=[];n=captures[0]
    while n in parents:n=parents[n];ancestors.append(n)
    assert not any(isinstance(n,(ast.For,ast.While)) for n in ancestors)
    assert 'assert not llm.llm_engine.has_unfinished_requests()' in source
    w=Path(worker).read_text();b=Path(bank).read_text()
    for token in ("state['capture_invocations']==0", "state['capture_invocations']+=1", "probe['requests'].clear()", "cohort=None,epochs={},pending=None,live_tables=[],live_slots=[]", "state['qualified_shapes_from_training']", "state['cohort'].get(rid) != identity", "state['plan'].bind_prefix(actual_table)"):
        assert token in w,token
    assert 'self.device_cut.fill_(prefix_tokens//256)' in b and 'torch.cuda.synchronize()' in b
    activation=b[b.index('    def activate_prefix'):b.index('    def run_layer') if '    def run_layer' in b else b.index('    def receipt')]
    assert '.lengths.copy_' not in activation and '.table.copy_' not in activation
    return {'single_capture_RPC_outside_document_loops':True,'worker_second_capture_rejected':True,
            'empty_scheduler_assertions':True,'request_slot_epoch_ownership_reset_source_present':True,
            'explicit_training_shape_transfer_source_present':True,'activation_does_not_write_native_lengths_or_tables':True}


def recompute_energy(run):
    samples=run['energy_samples'];assert len(samples)==6
    rows=[]
    for lo,hi,key in ((0,1,'activation'),(1,2,'anchor_and_reset'),(3,4,'group_generate'),(0,5,'complete_document_through_export')):
        left={d['physical_selector']:d for d in samples[lo]['devices']};right={d['physical_selector']:d for d in samples[hi]['devices']}
        values=[]
        for device in sorted(left.keys()|right.keys()):
            a=left.get(device,{});b=right.get(device,{});x=a.get('total_energy_mJ');y=b.get('total_energy_mJ')
            valid=x is not None and y is not None and y>=x
            identity=a.get('identity_verified') is True and b.get('identity_verified') is True and a.get('UUID')==b.get('UUID')
            if identity:
                assert a['UUID'].lower()==a['expected_worker_UUID'].lower() and b['UUID'].lower()==b['expected_worker_UUID'].lower()
            values.append((y-x)/1000 if valid and identity else None)
        total=sum(values) if values and all(v is not None for v in values) else None
        assert run['energy_intervals'][key]['total_GPU_joules']==total
        rows.append({'interval':key,'UUID_verified_total_GPU_joules':total})
    assert abs(run['NVML_sampling_wall_seconds']-sum(x['NVML_sampling_wall_seconds'] for x in samples))<1e-9
    return rows



def counter_difference(begin,end):
    left={d['physical_selector']:d for d in begin['devices']};right={d['physical_selector']:d for d in end['devices']};values=[]
    for k in sorted(left.keys()|right.keys()):
        a=left.get(k,{});b=right.get(k,{});x=a.get('total_energy_mJ');y=b.get('total_energy_mJ')
        valid=x is not None and y is not None and y>=x
        identity=a.get('identity_verified') is True and b.get('identity_verified') is True and a.get('UUID')==b.get('UUID')
        if identity:assert a['UUID'].lower()==a['expected_worker_UUID'].lower() and b['UUID'].lower()==b['expected_worker_UUID'].lower()
        values.append((y-x)/1000 if valid and identity else None)
    return sum(values) if values and all(v is not None for v in values) else None

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--summary',required=True);ap.add_argument('--allow-interim',action='store_true');ap.add_argument('--output',required=True);a=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU audit requires CUDA hidden before imports'
    output=Path(a.output);assert not output.exists();output.parent.mkdir(parents=True,exist_ok=True)
    path=Path(a.summary).resolve();s=json.loads(path.read_text());complete=s['status']=='completed'
    r={'status':'auditing','auditor_sha256':digest(__file__),'input_summary':str(path),'input_summary_sha256_at_read':digest(path),'input_status':s['status'],'variant':'explicit_training_shape_transfer','runs':[],'comparisons':{},'limits':[
        'Receipt/source audit, not an independent CUDA event trace of every graph-capture call.',
        'Physical page IDs and per-request slot/epoch values are not persisted; physical ownership evidence is the executed checked binding and immutable source guards. No independent reconstruction of all page IDs is claimed.',
        'One seed and same-session ABBA; repeats are not independent documents. Component sums are not independent fresh-service trials.',
        'NVML counter differences include all activity on the device; exclusive GPU allocation must be audited in the root ledger separately.']}
    def save():output.write_text(json.dumps(r,indent=2)+'\n')
    try:
        assert complete or a.allow_interim, 'Final audit requires a completed summary'
        assert s['status'] in ('completed','initializing'), 'Failed runs cannot be promoted by interim audit'
        direct=True
        driver=ROOT/'scripts/benchmark_raw_pair_direct_next8_transfer.py';worker=ROOT/'scripts/vllm_raw_pair_direct_transfer_worker.py';bank=ROOT/'src/vllm_raw_pair_direct_multidoc_plan.py'
        assert str(driver.relative_to(ROOT)) in s['source_sha256']
        for name,expected in s['source_sha256'].items():assert digest(ROOT/name)==expected, name
        assert digest(ROOT/s['args']['protocol'])==s['protocol_sha256']
        assert digest(ROOT/s['protocol']['manifest'])==s['manifest_sha256']
        r['source_contract']=source_contract(driver,worker,bank)
        protocol=s['protocol'];all_docs={d['document_id']:d for d in protocol['documents']}
        selected=protocol['fixed_two_document_tasks'][s['args']['task_index']]
        assert s['selected_document_ids']==selected
        docs=[all_docs[k] for k in selected];enabled=protocol['candidate_enabled_batches'];bydoc={d['document_id']:d for d in docs}
        assert len(all_docs)==8 and len(protocol['predeclared_input_failures'])==2
        assert s['preserved_full8_context_failures']==protocol['predeclared_input_failures']
        capture={c['rank']:c for c in s['capture']};assert set(capture)=={0,1}
        cutptrs={rank:c['plan']['shared_cut_ptr'] for rank,c in capture.items()}
        for c in capture.values():assert c['captured_modes']==['production'] and c['setup_mode']=='production_only' and c['plan']['dynamic_cut']
        configs={c['rank']:c for c in s['configuration']}
        build_checks=[];seen=set()
        for configuration in configs.values():
            receipt=configuration['raw_pair_build_receipt'];identity=receipt['identity']
            if identity in seen:continue
            seen.add(identity)
            dirname='xqa_raw_direct_dynamic_cut' if direct else 'xqa_raw_grouped_dynamic_cut'
            directory=ROOT/'build'/dirname/identity
            for relative,expected_sha in receipt['generated_sha256'].items():assert digest(directory/relative)==expected_sha
            build_checks.append({'identity':identity,'generated_files_checked':len(receipt['generated_sha256']),'build_directory':str(directory)})
        r['generated_build_source_audit']=build_checks
        offline=s['offline_qualification'];prior=json.loads(Path(offline['summary']).read_text())
        assert prior['status']=='completed' and digest(offline['summary'])==offline['summary_sha256']
        from src.raw_pair_direct_training_shape_transfer import load_training_shapes
        from scripts.vllm_raw_pair_direct_multidoc_worker import source_hashes as strict_sources
        training=load_training_shapes(offline['summary'],strict_sources(),enabled)
        assert training['contract']==s['training_shape_transfer'] and training['loaded']['record_inputs']==offline['record_inputs']
        assert s['training_shape_transfer']['evaluation_document_locally_qualified'] is False
        native_tokens={x['document_id']:[y['tokens'] for y in x['answers']] for x in s['runs'] if x['phase']=='natural_native_0'}
        runs=[x for x in s['runs'] if 'graph_trajectory_sha256_by_rank' in x]
        if complete:assert len(runs)==len(s['runs'])==8
        expected=[(f'natural_{m}_{i}',d['document_id']) for i,m in enumerate(('native','cascade','cascade','native')) for d in docs]
        assert [(x['phase'],x['document_id']) for x in runs]==expected[:len(runs)]
        histories={0:[],1:[]};last_uploaded={rank:docs[0]['common_complete_page_tokens'] for rank in (0,1)}
        for index,run in enumerate(runs):
            doc=bydoc[run['document_id']];prefix=doc['common_complete_page_tokens'];candidate=run['method']=='cascade'
            assert run['validation'] is False and run['B']==doc['B'] and run['P']==prefix
            assert [x['tokens'] for x in run['answers']]==native_tokens[doc['document_id']]
            cold=sum(run[key] for key in ('activation_control_wall_seconds','prefix_reset_wall_seconds','anchor_wall_seconds','group_control_wall_seconds','complete_generate_wall_seconds'))
            assert abs(cold-run['cold_shared_document_total_wall_seconds'])<1e-7
            activations={x['rank']:x for x in run['activations']};workers={x['rank']:x for x in run['workers']};assert set(activations)==set(workers)=={0,1}
            receipts=[];trajectories=[]
            for rank,w in sorted(workers.items()):
                rawpath=Path(w['raw_pair_record']);raw=json.loads(rawpath.read_text());graphpath=Path(w['graph_metadata']);rows=json.loads(graphpath.read_text())['replays']
                act=activations[rank];assert act['document_id']==doc['document_id'] and act['generation']==index+1 and act['prefix']==prefix
                assert act['candidate_prepared']==candidate and act['old_request_slot_epoch_ownership_cleared'] is True
                assert act['bank']['shared_cut_ptr']==cutptrs[rank] and act['bank']['nominal_prefix_tokens']==prefix
                assert act['bank']['cut_tiles']==prefix//256 and act['bank']['candidate_cut_uploaded']==candidate and act['bank']['native_lengths_and_tables_untouched'] is True
                if candidate:last_uploaded[rank]=prefix
                else:assert act['bank']['prior_work_synchronize_seconds']==act['bank']['cut_upload_and_complete_seconds']==0
                histories[rank].append({k:v for k,v in act.items() if k!='rank'})
                assert raw['document_activations']==histories[rank]
                assert raw['plan']['document_switches']==[x['bank'] for x in histories[rank]]
                assert raw['document_id']==doc['document_id'] and raw['document_generation']==index+1 and raw['capture_invocations']==1
                assert raw['setup_mode']=='production_only' and raw['checks']==[] and raw['qualification_steps']==0 and raw['phase_valid']
                assert raw['source_sha256']=={k:s['source_sha256'][k] for k in raw['source_sha256']}
                assert raw['execution_fingerprint']==offline['execution_fingerprint']
                assert raw['build_receipt']['identity']==configs[rank]['raw_pair_build_receipt']['identity']
                plan=raw['plan'];assert plan['shared_cut_ptr']==cutptrs[rank] and plan['nominal_active_prefix_tokens']==prefix and plan['uploaded_prefix_tokens']==last_uploaded[rank]
                assert plan['workspace_allocation_sets']==plan['raw_state_allocation_sets']==1
                for btext,p in plan['plans'].items():
                    b=int(btext);extent=p['native_table_shape'][1]*16
                    splits=min(max(1,raw['execution_fingerprint']['SM']//(b*2)),(extent+255)//256)
                    assert p['native_splits']==splits and p['prefix_tiles']==prefix//256
                    assert p['prefix_SM']==splits*((b+1)//2)*2 and p['tail_SM']==splits*b*2
                assert raw['captured_layers_by_batch']=={'production':{str(b):list(range(48)) for b in enabled}}
                qualified=training['contract']['eligible_training_qualified_shapes']
                assert raw['qualified_batches']==act['eligible_training_qualified_batches']==qualified
                assert raw['training_qualified_shapes']==qualified
                assert raw['qualification_kind']==act['qualification_kind']=='explicit_training_shape_transfer'
                assert raw['document_locally_qualified'] is act['document_locally_qualified'] is False
                assert len(rows)==raw['candidate_replays']+raw['native_replays']
                selected=[x for x in rows if x['candidate_graph_replayed']];preps=raw['prepare_records']
                assert len(selected)==raw['candidate_replays']==len(preps)
                for row in rows:
                    assert row['native_graph_replayed']!=row['candidate_graph_replayed']
                    assert len(row['positions'])==len(row['questions'])==row['batch'] and len(set(row['questions']))==row['batch']
                    assert all(0<=q<doc['B'] for q in row['questions'])
                if candidate and preps:
                    assert preps[0]['host_physical_prefix_check'] and sum(p['host_physical_prefix_check'] for p in preps)==1
                    binding=preps[0]['prefix_binding'];assert binding['physical_prefix_checked'] and binding['prefix_pages']==prefix//16 and binding['rows']==preps[0]['batch']
                    cohort=set(selected[0]['questions'])
                    for p,row in zip(preps,selected):
                        assert set(row['questions']).issubset(cohort);cohort=set(row['questions'])
                        assert p['batch']==row['batch'] in qualified and p['positions']==row['positions']
                        assert p['lengths']==[x+1 for x in p['positions']] and p['policy_prefix_tokens']==prefix and p['actual_shared_tile_tokens']==prefix//256*256
                        assert p['native_GPU_lengths_aliased'] and not p['host_length_copy'] and p['cohort_request_slot_epoch_subset_guard']
                        private=max(p['lengths'])-prefix;assert p['policy_max_private_tokens']==private and 0<private*8<=prefix and p['policy_eligible']
                if not candidate:assert not selected and not preps
                if direct:
                    for layouts in plan['query_layouts'].values():
                        for layout in layouts.values():assert layout['Q_native_alias'] and layout['O_native_alias'] and not layout['Q_canonicalization_copy_charged']
                trajectory=hashlib.sha256(json.dumps([(x['batch'],x['questions'],x['positions']) for x in rows]).encode()).hexdigest();trajectories.append(trajectory)
                receipts.append({'rank':rank,'raw_record':str(rawpath),'raw_sha256':digest(rawpath),'graph_sha256':digest(graphpath),
                    'generation':index+1,'capture_invocations':1,'shared_cut_ptr':cutptrs[rank],'candidate_graph_steps':len(selected),
                    'initial_physical_binding_count':sum(p['host_physical_prefix_check'] for p in preps),
                    'observed_batches':sorted({x['batch'] for x in selected}),'transferred_training_qualified_batches':qualified})
            assert trajectories==run['graph_trajectory_sha256_by_rank'] and trajectories[0]==trajectories[1]
            r['runs'].append({'phase':run['phase'],'document_id':doc['document_id'],'receipts':receipts,'energy_recomputed':recompute_energy(run),
                'activation_seconds':run['activation_control_wall_seconds'],'generate_seconds':run['complete_generate_wall_seconds'],
                'cold_document_seconds':run['cold_shared_document_total_wall_seconds'],'NVML_sampling_seconds':run['NVML_sampling_wall_seconds']})
        r['capture_evidence']={'driver_capture_RPC_calls':1,'actual_captured_modes_by_rank':{str(k):v['captured_modes'] for k,v in capture.items()},
            'single_stable_cut_pointer_per_rank':cutptrs,'captured_production_seconds':s['candidate_capture_seconds'],
            'runtime_guard_rejects_second_capture':True,'audited_document_switches_per_rank':len(runs)}
        if complete:
            for doc in docs:
                rs=[x for x in runs if x['document_id']==doc['document_id']]
                assert [x['method'] for x in rs]==['native','cascade','cascade','native']
                for key in ('graph_trajectory_sha256_by_rank',):assert all(x[key]==rs[0][key] for x in rs)
                for field in ('tokens','finish_reason','num_cached_tokens'):assert all([y[field] for y in x['answers']]==[y[field] for y in rs[0]['answers']] for x in rs)
                executed=all(x['candidate_graph_executed'] for x in rs if x['method']=='cascade')
                ratios={key:statistics.mean(rs[i][key] for i in (0,3))/statistics.mean(rs[i][key] for i in (1,2)) for key in ('complete_generate_wall_seconds','cold_shared_document_total_wall_seconds')}
                energy=[x['energy_intervals']['group_generate']['total_GPU_joules'] for x in rs]
                r['comparisons'][doc['document_id']]={'native_over_candidate':ratios,'all_output_tokens_and_trajectories_equal':True,'candidate_executed':executed,
                    'group_energy_native_over_candidate':statistics.mean(energy[i] for i in (0,3))/statistics.mean(energy[i] for i in (1,2)) if all(x is not None and x>0 for x in energy) else None,
                    'candidate_speed_claim_eligible':executed}
            def task_mean(key,method):
                arm_indices=(0,3) if method=='native' else (1,2)
                return statistics.mean(sum(x[key] for x in runs if x['phase']==f'natural_{method}_{i}') for i in arm_indices)
            full={m:task_mean('full_phase_control_wall_seconds',m) for m in ('native','cascade')}
            ready={m:task_mean('complete_generate_wall_seconds',m) for m in ('native','cascade')}
            common=s['common_energy_initialization_and_identity_RPC_wall_seconds']+s['common_precision_export_instrumented_wall_seconds']
            extra=s['offline_preflight_wall_seconds']+s['candidate_setup_instrumented_wall_seconds']
            native=full['native']+common;candidate=full['cascade']+common+extra
            declared=s['first_two_document_task_component_sums']
            assert abs(declared['native_seconds']-native)<1e-8 and abs(declared['candidate_seconds']-candidate)<1e-8
            samples=s['setup_energy_samples'];assert [v['label'] for v in samples]==['before_candidate_configuration','after_candidate_configuration','after_candidate_capture','after_common_precision_receipt_export']
            assert all(x['monotonic_seconds']<=y['monotonic_seconds'] for x,y in zip(samples,samples[1:]))
            setup_j={}
            for lo,hi,key in ((0,1,'candidate_configuration'),(1,2,'candidate_capture'),(0,2,'candidate_configuration_and_capture'),(2,3,'common_precision_receipt_export')):
                setup_j[key]=counter_difference(samples[lo],samples[hi])
                assert setup_j[key]==s['setup_energy_intervals'][key]['total_GPU_joules']
            assert abs(sum(x['NVML_sampling_wall_seconds'] for x in samples)-s['setup_NVML_sampling_wall_seconds'])<1e-9
            energies={}
            for method,indices in (('native',(0,3)),('cascade',(1,2))):
                arms=[]
                for i in indices:
                    values=[x['energy_intervals']['complete_document_through_export']['total_GPU_joules'] for x in runs if x['phase']==f'natural_{method}_{i}']
                    assert len(values)==2
                    arms.append(sum(values) if all(x is not None for x in values) else None)
                energies[method]=statistics.mean(arms) if all(x is not None for x in arms) else None
            measured=all(x is not None for x in [*energies.values(),setup_j['candidate_configuration_and_capture'],setup_j['common_precision_receipt_export']])
            native_j=energies['native']+setup_j['common_precision_receipt_export'] if measured else None
            candidate_j=energies['cascade']+setup_j['common_precision_receipt_export']+setup_j['candidate_configuration_and_capture'] if measured else None
            assert declared['native_GPU_joules']==native_j and declared['candidate_GPU_joules']==candidate_j
            setup_source=driver.read_text()
            assert setup_source.index("identities=llm.collective_rpc('raw_pair_device_identity')")<setup_source.index('energy=PhaseEnergyCounter')<setup_source.index("energy.sample('before_candidate_configuration')")<setup_source.index("llm.collective_rpc('configure_dense_cascade'")
            r['two_document_primary']={'metric':'First two-document task after common engine initialization, measured component sum',
                'native_full_phase_task_mean_seconds':full['native'],'candidate_full_phase_task_mean_seconds':full['cascade'],
                'candidate_once_per_task_setup_seconds':extra,'common_controls_once_per_arm_seconds':common,
                'native_net_seconds':native,'candidate_net_seconds':candidate,'net_native_over_candidate':native/candidate,
                'ready_generation_native_over_candidate':ready['native']/ready['cascade'],
                'native_GPU_joules_including_common_export':native_j,'candidate_GPU_joules_including_setup_and_common_export':candidate_j,
                'GPU_energy_native_over_candidate':native_j/candidate_j if measured and candidate_j>0 else None,
                'setup_energy_intervals_independently_recomputed':setup_j,
                'full_eight_feasibility_denominator':8,'feasible_documents_in_full_protocol':6,'preserved_context_failures':protocol['predeclared_input_failures'],
                'evaluation_document_locally_qualified':False,'qualification_kind':'explicit_training_shape_transfer',
                'time_scope':'Full phases include switch/reset/anchor/group-control/generation/worker-export/all counter samples. Offline provenance/configuration/capture once for candidate; identity/init and precision-export once for both. Component estimate, not an independent fresh service trial.',
                'energy_scope':'Verified board counters include candidate GPU setup and common precision export. Common engine/NVML init and offline CPU preflight precede energy boundaries; not full-process startup Joules.'}
            r['status']='completed_production_audited'
        else:r['status']='interim_exported_runs_audited';r['limits'].append('Experiment incomplete: no ABBA aggregate or terminal performance conclusion.')
        import torch
        assert not torch.cuda.is_initialized();r['cuda_initialized']=False
    except Exception:
        import traceback
        r['status']='audit_failed';r['error']=traceback.format_exc();save();raise
    save();print(json.dumps({'status':r['status'],'audited_runs':len(r['runs'])}))
if __name__=='__main__':main()
