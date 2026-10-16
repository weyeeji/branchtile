"""Same real input/history native-vs-real-cascade whole-model graph intervention."""
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from scripts.vllm_dense_cascade_reference_worker import DenseCascadeReferenceWorker
from src.dense_cascade_fp64_reference import full_history_fp64, compare_to_reference


class DenseCascadeSnapshotWorker(DenseCascadeReferenceWorker):
    def configure_dense_cascade_snapshot(self, config):
        bare_native = self.model_runner.cudagraph_manager.run_fullgraph
        result = self.configure_dense_cascade(config)
        runner = self.model_runner
        diag = self.cascade_snapshot = {'native':bare_native,'armed':False,'seen':0,'config':config}
        previous_tables = runner.prepare_attn
        def tables(*args, **kwargs):
            value = previous_tables(*args, **kwargs)
            diag['slots'] = value[1]
            return value
        runner.prepare_attn = tables
        previous_prepare = runner.model_state.prepare_attn
        def prepare(*args, **kwargs):
            metadata = previous_prepare(*args, **kwargs)
            diag['metadata'] = metadata
            return metadata
        runner.model_state.prepare_attn = prepare
        previous = runner.cudagraph_manager.run_fullgraph
        def run(desc):
            value = previous(desc)
            batch = self.native_probe['batch']
            if diag['armed'] and desc.num_reqs == config['batch'] and batch.num_tokens == batch.num_reqs:
                diag['seen'] += 1
                if diag['seen'] == 3:
                    return self.measure_dense_cascade_snapshot(desc)
            return value
        runner.cudagraph_manager.run_fullgraph = run
        return result

    def arm_dense_cascade_snapshot(self):
        assert self.dense_cascade['method'] == 'native'
        self.cascade_snapshot['armed'] = True
        return {'rank':self.native_probe['rank'],'trigger':'third real full-batch decode graph; original native trajectory restored afterward'}

    @torch.inference_mode()
    def measure_dense_cascade_snapshot(self, desc):
        diag, state, runner = self.cascade_snapshot, self.dense_cascade, self.model_runner
        diag['armed'] = False
        b = desc.num_reqs
        native = diag['native']
        positions = runner.input_buffers.positions[:b].clone()
        tokens = runner.input_buffers.input_ids[:b].clone()
        tables = state['live_tables'].clone()
        assert len(diag['slots']) == 1
        slots = diag['slots'][0][:b].clone().long()
        assert bool((slots >= 0).all())
        seq_lens = [int(p)+1 for p in positions.cpu().tolist()]
        torch.cuda.synchronize()
        plan_receipt = state['plans'][b].prepare(state['live_tables'],seq_lens)
        torch.cuda.synchronize()
        def current_kv():
            return {l:c[slots//16,:,slots%16,:].clone() for l,c in state['caches'].items()}
        def local_logits(hidden):
            return runner.model.logits_processor(runner.model.lm_head,hidden,skip_gather=True).clone()
        native_hidden = native(desc).clone()
        native_kv = current_kv()
        native_global = runner.model.compute_logits(native_hidden)
        if native_global is not None:
            native_global = native_global.clone()
        native_local = local_logits(native_hidden)
        candidate_hidden = state['managers']['checked'].run_fullgraph(desc).clone()
        candidate_kv = current_kv()
        candidate_global = runner.model.compute_logits(candidate_hidden)
        if candidate_global is not None:
            candidate_global = candidate_global.clone()
        candidate_local = local_logits(candidate_hidden)
        metadata = next(iter(diag['metadata'].values()))
        layer_checks = []
        for layer in range(48):
            q = state['saved_q'][layer,b]
            actual = state['saved_o'][layer,b]
            expected = torch.empty_like(actual)
            attn_layer, original = state['native_layers'][layer]
            original(attn_layer,q,None,None,state['caches'][layer],metadata,output=expected)
            pair = bool(torch.isclose(actual.float(),expected.float(),atol=.006,rtol=.015).all())
            check = {'layer':layer,'same_Q_native_O_passed':pair,'qualification_passed':pair,
                     'max_abs':float((actual.float()-expected.float()).abs().max()),'reference_computed':False}
            if not pair:
                ref = full_history_fp64(q,state['caches'][layer],state['live_tables'],seq_lens,
                    k_scale=attn_layer._k_scale_float,v_scale=attn_layer._v_scale_float)
                c,n = compare_to_reference(actual,ref),compare_to_reference(expected,ref)
                check.update(reference_computed=True,candidate_vs_fp64=c,native_vs_fp64=n,
                    pair_failed_but_reference_passed=c['passed'] and n['passed'],qualification_passed=c['passed'] and n['passed'])
            layer_checks.append(check)
        production_hidden = state['managers']['production'].run_fullgraph(desc).clone()
        production_kv = current_kv()
        candidate_graphs_equal = bool(torch.equal(production_hidden,candidate_hidden)) and all(torch.equal(production_kv[l],candidate_kv[l]) for l in range(48))
        def difference(a,n):
            if a is None or n is None:
                assert a is n is None
                return None
            delta = a.float()-n.float()
            return {'bitwise_equal':bool(torch.equal(a,n)),'max_abs':float(delta.abs().max()),
                    'relative_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(n.float()).clamp_min(1e-30))}
        outputs = {'hidden':difference(candidate_hidden,native_hidden),
                   'local_logits':difference(candidate_local,native_local),
                   'gathered_logits':difference(candidate_global,native_global),
                   'global_greedy_argmax_native':native_global.argmax(-1).cpu().tolist() if native_global is not None else None,
                   'global_greedy_argmax_candidate':candidate_global.argmax(-1).cpu().tolist() if candidate_global is not None else None,
                   'checked_and_production_hidden_and_written_KV_bitwise_equal':candidate_graphs_equal}
        methods = {'native':lambda:native(desc),'cascade':lambda:state['managers']['production'].run_fullgraph(desc),'native_AA':lambda:native(desc)}
        for fn in methods.values():
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        flush = torch.zeros(256*1024*1024,device=runner.device,dtype=torch.uint8)
        rows=[]
        randomizer=random.Random(20261017)
        for cache_mode in ('model_warm','artificial_cold'):
            for block in range(diag['config'].get('blocks',12)):
                order=list(methods); randomizer.shuffle(order)
                for direction,names in (('random',order),('reverse',list(reversed(order)))):
                    row={'cache':cache_mode,'block':block,'direction':direction,'cuda_us':{},'submit_and_sync_wall_seconds':{}}
                    for name in names:
                        if cache_mode == 'artificial_cold':
                            flush.add_(1);torch.cuda.synchronize()
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        started=time.perf_counter();begin.record();methods[name]();end.record();end.synchronize()
                        row['submit_and_sync_wall_seconds'][name]=time.perf_counter()-started
                        row['cuda_us'][name]=begin.elapsed_time(end)*1000
                    rows.append(row)
        final=native(desc).clone()
        final_kv=current_kv()
        restored=(bool(torch.equal(final,native_hidden)) and bool(torch.equal(tokens,runner.input_buffers.input_ids[:b]))
            and bool(torch.equal(positions,runner.input_buffers.positions[:b])) and bool(torch.equal(tables,state['live_tables']))
            and all(torch.equal(final_kv[l],native_kv[l]) for l in range(48)))
        report={'status':'completed','rank':self.native_probe['rank'],'input_tokens':tokens.cpu().tolist(),
            'positions':positions.cpu().tolist(),'physical_slots':slots.cpu().tolist(),'seq_lens':seq_lens,
            'plan_receipt':plan_receipt,'layer_checks':layer_checks,'output_differences':outputs,
            'all_layers_qualified':all(c['qualification_passed'] for c in layer_checks),
            'final_native_inputs_tables_written_KV_hidden_bitwise_restored':restored,'measurements':rows,
            'scope':'Same real tokens, positions and prior KV history; both timed whole-model graphs perform actual full-history attention. Candidate hidden/logits and MoE routes may differ.',
            'limits':['One frozen state; candidate recomputes every layer O, not cached-output oracle.',
                      'Native and candidate each overwrite current-token KV before reading it; final native replay restores written slots and native hidden bitwise.',
                      'Only written KV slots are audited; no full-cache checksum.',
                      'Timers exclude per-step preparation, fresh prefill and graph construction; those costs remain in separate whole-generation experiments.',
                      'CUDA event intervals may include submission gaps; paired native AA and host times retained.',
                      'Numeric qualification does not prove identical next-token distribution or natural task accuracy.']}
        path=self.native_probe['output']/'dense_cascade_snapshot.json'
        path.write_text(json.dumps(report,indent=2)+'\n')
        diag['result_path']=str(path)
        return final

    def dense_cascade_snapshot_result(self):
        path=Path(self.cascade_snapshot['result_path'])
        report=json.loads(path.read_text())
        return {'rank':self.native_probe['rank'],'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'all_layers_qualified':report['all_layers_qualified'],
                'restored':report['final_native_inputs_tables_written_KV_hidden_bitwise_restored'],
                'candidate_graphs_equal':report['output_differences']['checked_and_production_hidden_and_written_KV_bitwise_equal']}
