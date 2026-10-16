"""Bounded real-cache attention replay; no selector/executor optimization.

The engine is quiescent after snapshot. B1/B2 are explicit subsets of one real
B3 state, not separate natural trajectories. The model profiler is attribution
only; attention-only replay does not reproduce MoE/GEMM/NCCL interleaving.
"""
import hashlib
import json
import random
import time

import torch

from scripts.vllm_joint_support_worker import JointSupportWorkerExtension
from src.vllm_sparse_executor import PagedSparseExecutor, physical_supports


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class JointReplayWorkerExtension(JointSupportWorkerExtension):
    def configure_sparse_attention(self, config):
        result=super().configure_sparse_attention(config)
        state=self.sparse_state
        state['replay_model_outputs']={};state['replay_model_events']=[]
        state['replay_profile_count']=0;state['replay_profile_seen']=0
        state['replay_profile_armed']=False
        previous=self.model_runner.cudagraph_manager.run_fullgraph
        def measured(desc):
            pending=state['graph_pending']
            active=state['replay_profile_armed'] and pending is not None and state['method']=='e0'
            if not active:
                return previous(desc)
            assert pending['batch']==3
            seen=state['replay_profile_seen'];state['replay_profile_seen']+=1
            profiled=1<=seen<5
            if seen==1:
                profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False)
                profiler.start();state['replay_profiler']=profiler
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record()
            with torch.profiler.record_function('joint_replay_actual_E0_model_B3'):
                value=previous(desc)
            end.record()
            state['replay_model_events'].append({'begin':begin,'end':end,'profiled':profiled,
                'positions':list(pending['positions']),'questions':list(pending['questions'])})
            if profiled:
                state['replay_profile_count']+=1
            if seen==4:
                torch.cuda.synchronize();state['replay_profiler'].stop()
            if seen==6:
                # Capture while this actual input batch and metadata are still live,
                # before scheduler completion can recycle input-buffer entries.
                meta=self._sparse_metadata(state['graph_native_metadata'])
                assert [p+1 for p in pending['positions']]==meta.seq_lens[:3].cpu().tolist()
                plan=state['shared_plans']['e0',3]
                state['replay_live_receipt']={'tables':meta.block_table[:3].clone(),
                    'positions':list(pending['positions']),'questions':list(pending['questions']),
                    'queries':[state['graph_queries']['e0',l,3].clone() for l in range(48)],
                    'actual_outputs':[plan.outputs[l].clone() for l in range(48)],
                    'actual_indices':plan.indices.clone(),'actual_request_ids':list(state['batch'].req_ids),
                    'actual_live_index_count':int(plan.indptr[-1])}
            return value
        self.model_runner.cudagraph_manager.run_fullgraph=measured
        return result

    def _sparse_forward(self, layer, original, attn_layer, query, key, value, cache, meta,
                        output, output_scale, output_block_scale):
        state=self.sparse_state
        if state.get('graph_capturing')=='e0':
            state['replay_model_outputs'][layer,query.shape[0]]=output
        return super()._sparse_forward(layer,original,attn_layer,query,key,value,cache,meta,
            output,output_scale,output_block_scale)

    def arm_replay_profile(self):
        state=self.sparse_state
        assert state['production_enabled'] and not state['config'].get('graph_shadow_check')
        assert state['joint_active_variant']=='independent'
        state['replay_profile_armed']=True
        return {'rank':state['rank'],'skip_pure_B3_steps':1,'profile_pure_B3_steps':4,
            'scope':'Actual no-shadow sparse E0 model graph, includes all 48 layers, MoE and TP communication. Instrumented attribution only.'}

    @torch.inference_mode()
    def freeze_replay_snapshot(self):
        state=self.sparse_state;started=time.perf_counter()
        state['replay_profile_armed']=False
        assert state['replay_profile_count']==4
        pending=state['graph_pending'];assert pending['batch']==3
        assert state['replay_profile_seen']==7 and 'replay_live_receipt' in state
        assert state['joint_active_variant']=='independent'
        assert state['method']=='e0' and state['production_enabled']
        torch.cuda.synchronize()
        live_receipt=state.pop('replay_live_receipt')
        tables=live_receipt['tables']
        positions=live_receipt['positions'];questions=live_receipt['questions']
        assert questions==state['graph_phase_records'][-1]['questions']
        assert positions==state['graph_phase_records'][-1]['positions']
        qs=live_receipt['queries'];actual=live_receipt['actual_outputs']
        assert all(torch.isfinite(q).all() and float(q.float().norm())>0 for q in qs)
        state['replay_snapshot']={'tables':tables,'questions':questions,'positions':positions,
            'queries':qs,'actual_outputs':actual,'graphs':{},'outputs':{},'queries_by_batch':{}}
        # Verify the saved tables really reproduce the indices of the actual last model step.
        plan=state['shared_plans']['e0',3];before=live_receipt['actual_indices']
        prep=plan.prepare(tables,questions,positions,state['shared_history_id'])
        assert not prep['plan_topology_changed_requires_recapture']
        live=int(plan.indptr[-1]);assert live==live_receipt['actual_live_index_count']
        assert torch.equal(before[:,:live],plan.indices[:,:live])
        snapshot=state['replay_snapshot']
        rows=[]
        for b in (1,2,3):
            plan=state['shared_plans']['e0',b]
            prepared=plan.prepare(tables[:b],questions[:b],positions[:b],state['shared_history_id'])
            assert not prepared['plan_topology_changed_requires_recapture']
            qset=[q[:b].clone() for q in qs]
            outputs=[torch.empty_strided(state['replay_model_outputs'][l,b].shape,
                state['replay_model_outputs'][l,b].stride(),device=qset[l].device,dtype=qset[l].dtype) for l in range(48)]
            assert all(o.shape==q.shape for o,q in zip(outputs,qset))
            def run_layers():
                for l in range(48):
                    plan.run_layer(l,qset[l],output=outputs[l])
            for _ in range(3):run_layers()
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):run_layers()
            graph.replay();torch.cuda.synchronize()
            if b==3:
                assert all(torch.equal(a,o) for a,o in zip(actual,outputs)), 'Saved Q/cache/metadata must reproduce the final actual model attention outputs exactly'
            snapshot['graphs'][b]=graph;snapshot['outputs'][b]=outputs;snapshot['queries_by_batch'][b]=qset
            rows.append({'batch':b,'question_rows':questions[:b],'positions':positions[:b],
                'support_lengths':prepared['support_lengths'],'plan_info':prepared['plan_info'],
                'query_sha256':[tensor_sha(q) for q in qset],
                'query_strides':[list(q.stride()) for q in qset],
                'original_model_output_strides':[list(state['replay_model_outputs'][l,b].stride()) for l in range(48)],
                'replay_output_strides':[list(o.stride()) for o in outputs],
                'production_run_layer_used':True,'required_output_copy_included':True})
        # Allocate once; flush is outside timed intervals. No model/cache writes follow.
        snapshot['flush']=torch.empty(256*1024*1024,device=qs[0].device,dtype=torch.uint8)
        snapshot['table_cpu']=tables.cpu()
        (state['output']/'replay_actual_model_trace.json').parent.mkdir(parents=True,exist_ok=True)
        state['replay_profiler'].export_chrome_trace(str(state['output']/'replay_actual_model_trace.json'))
        model_events=[{k:v for k,v in r.items() if k not in ('begin','end')} | {'model_graph_cuda_us':r['begin'].elapsed_time(r['end'])*1000}
            for r in state['replay_model_events']]
        receipts={'rank':state['rank'],'snapshot_wall_seconds':time.perf_counter()-started,
            'source':'Last actual no-shadow E0 B3 graph step, 8-token bounded instrumented decode',
            'positions':positions,'questions':questions,'prompt_lengths':[len(state['selector'].prompts[q]) for q in questions],
            'generated_tail_lengths':[p-len(state['selector'].prompts[q])+1 for q,p in zip(questions,positions)],
            'actual_request_ids':live_receipt['actual_request_ids'],'block_table_sha256':tensor_sha(tables),
            'block_tables':snapshot['table_cpu'].tolist(),'rows':rows,'model_graph_events':model_events,
            'profiled_model_steps':4,'actual_model_phase_records':state['graph_phase_records'],
            'cache_layouts':{str(l):{'shape':list(c.shape),'stride':list(c.stride()),'storage_offset':c.storage_offset(),
                'data_ptr':c.data_ptr(),'dtype':str(c.dtype)} for l,c in state['graph_caches'].items()},
            'quiescence_contract':'No requests or model forwards after snapshot; no new KV writes; original physical cache retained. B1/B2 are explicit B3 row subsets.',
            'actual_B3_outputs_bitwise_reproduced':True,'actual_last_step_indices_bitwise_reproduced':True}
        receipts['independent_selector_summary']=state['joint_variants']['independent'].summary()
        import importlib.metadata
        receipts['packages']={p:importlib.metadata.version(p) for p in ('torch','flashinfer-python','vllm','triton')}
        props=torch.cuda.get_device_properties(qs[0].device)
        receipts['device']={'name':props.name,'uuid':str(getattr(props,'uuid',None)),
            'total_memory':props.total_memory,'compute_capability':[props.major,props.minor]}
        (state['output']/'replay_snapshot.json').write_text(json.dumps(receipts,indent=2)+'\n')
        return receipts

    @torch.inference_mode()
    def run_joint_replay(self, protocol):
        state=self.sparse_state;snapshot=state['replay_snapshot'];started=time.perf_counter()
        names=['independent','epsilon0','epsilon001','epsilon01']
        assert [v['name'] for v in protocol['variants']]==names
        report={'rank':state['rank'],'status':'qualifying','qualification':[],'measurements':[],
            'installations':[],'protocol':protocol,'scope':'Real original 48-layer cache attention chain; B1/B2 are actual B3 row subsets; no natural quality inference.'}
        path=state['output']/'replay.json'
        def save():path.write_text(json.dumps(report,indent=2)+'\n')
        def prepare(b):
            plan=state['shared_plans']['e0',b]
            record=plan.prepare(snapshot['tables'][:b],snapshot['questions'][:b],snapshot['positions'][:b],state['shared_history_id'])
            assert not record['plan_topology_changed_requires_recapture']
            return record
        original_outputs={}
        # Same-mask independent native SIMT checks use all 48 layers at every B.
        reference=PagedSparseExecutor(snapshot['queries'][0].device,'e0_simt')
        for name in names:
            installation=self.install_joint_variant(name)
            report['installations'].append({'stage':'qualification','variant':name,**installation})
            for b in (1,2,3):
                prepare(b);snapshot['graphs'][b].replay();torch.cuda.synchronize()
                for l in range(48):
                    q=snapshot['queries_by_batch'][b][l]
                    logical=[state['selector'].logical_supports(l,question,position)
                        for question,position in zip(snapshot['questions'][:b],snapshot['positions'][:b])]
                    ids=physical_supports(snapshot['table_cpu'][:b],logical,2,16,state['graph_caches'][l].stride())
                    reference.plan(q,state['graph_caches'][l],ids);expected=reference.run(q)
                    actual=snapshot['outputs'][b][l]
                    passed=bool(torch.isclose(actual.float(),expected.float(),atol=.006,rtol=.015).all())
                    row={'variant':name,'batch':b,'layer':l,'same_active_mask_SIMT':True,
                        'frozen_tolerance_passed':passed,'max_abs':float((actual.float()-expected.float()).abs().max())}
                    if name=='independent':original_outputs[b,l]=actual.clone()
                    if name=='epsilon0':
                        row['epsilon0_independent_bitwise_equal']=bool(torch.equal(actual,original_outputs[b,l]))
                        assert row['epsilon0_independent_bitwise_equal']
                    report['qualification'].append(row)
                    if not passed:
                        save();raise AssertionError(f'Same-mask replay qualification failed: {row}')
            save()
        del reference,original_outputs
        events=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        def sample(b,cache,complete):
            plan=state['shared_plans']['e0',b]
            # The measured chain is separate from the actual interleaved model profile.
            if cache=='cold':snapshot['flush'].fill_(1)
            torch.cuda.synchronize()
            begin,end=events;host=time.perf_counter();begin.record()
            prep=prepare(b) if complete else None
            snapshot['graphs'][b].replay();end.record();end.synchronize()
            return {'cuda_us':begin.elapsed_time(end)*1000,'complete_wall_us':(time.perf_counter()-host)*1e6,
                'prepare':dict(prep) if prep is not None else None}
        report['status']='measuring';save()
        rng=random.Random(protocol['measurement_seed'])
        for block in range(protocol['blocks']):
            order=names.copy();rng.shuffle(order)
            for direction,sequence in [('random',order),('reverse',order[::-1])]:
                for name in sequence:
                    installation=self.install_joint_variant(name)
                    report['installations'].append({'stage':'measurement','block':block,'direction':direction,'variant':name,**installation})
                    for b in (1,2,3):
                        # Explicit preparation is mandatory after each new identity installation.
                        t0=time.perf_counter();prep=prepare(b);torch.cuda.synchronize()
                        preparation_only_us=(time.perf_counter()-t0)*1e6
                        snapshot['graphs'][b].replay();torch.cuda.synchronize()
                        for cache in ('hot','cold'):
                            for complete in (False,True):
                                n=protocol['hot_repeats'] if cache=='hot' else protocol['cold_repeats']
                                samples=[sample(b,cache,complete) for _ in range(n)]
                                report['measurements'].append({'block':block,'direction':direction,'variant':name,
                                    'batch':b,'cache':cache,'contract':'prepare_plus_48_layers' if complete else '48_layers_only',
                                    'samples':samples,'standalone_prepare_complete_wall_us':preparation_only_us,
                                    'prepared_support_lengths':prep['support_lengths'],
                                    'active_mask_identity':installation['active_mask_identity']})
                save()
        report['status']='completed';report['complete_worker_replay_wall_seconds']=time.perf_counter()-started
        report['final_cache_layouts_unchanged']=all(c.data_ptr()==state['shared_plans']['e0',3].caches[l].data_ptr()
            for l,c in state['graph_caches'].items())
        save()
        return {'rank':state['rank'],'status':report['status'],'path':str(path),'qualification_checks':len(report['qualification']),
            'measurement_rows':len(report['measurements']),'complete_worker_replay_wall_seconds':report['complete_worker_replay_wall_seconds']}
