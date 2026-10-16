"""Fixed real-state whole-model cached-O intervention, never a serving method."""
import hashlib
import json
from pathlib import Path
import random
import re
import time

import torch
from scripts.vllm_native_precision_worker import NativePrecisionWorkerExtension


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class CachedODiagnosticWorker(NativePrecisionWorkerExtension):
    def configure_cached_o_diagnostic(self, config):
        bare_native_run = self.model_runner.cudagraph_manager.run_fullgraph
        result = self.configure_native_precision_probe(config)
        state = self.cached_o = {'config': config, 'capture_mode': None, 'armed': False,
                                'done': False, 'seen': 0, 'managers': {}, 'cache': {},
                                'saved_o': {}, 'saved_q': {}, 'check_q': {}, 'check_o': {},
                                'captured_layers': {}}
        runner = self.model_runner
        for b in (1, 2, 3):
            for layer in range(48):
                for key in ('saved_o', 'saved_q', 'check_q', 'check_o'):
                    state[key][layer, b] = torch.zeros((b, 16, 128), device=runner.device, dtype=torch.bfloat16)
        for name, parent in runner.model.named_modules():
            match = re.search(r'(?:^|\.)layers\.(\d+)\.self_attn$', name)
            if not match:
                continue
            layer = int(match[1])
            assert not parent.attn.attn_backend.forward_includes_kv_cache_update
            state['cache'][layer] = parent.attn.kv_cache
            original = parent.attn.impl.forward
            def forward(attn_layer, query, key, value, kv_cache, attn_metadata, output,
                        output_scale=None, output_block_scale=None, layer=layer, original=original):
                mode = state['capture_mode']
                if mode is None:
                    return original(attn_layer, query, key, value, kv_cache, attn_metadata, output,
                                    output_scale=output_scale, output_block_scale=output_block_scale)
                assert output_scale is None and output_block_scale is None
                b = query.shape[0]
                assert query.shape == (b, 16, 128) and b in (1, 2, 3)
                state['captured_layers'].setdefault(mode, set()).add((layer, b))
                if mode == 'witness':
                    result = original(attn_layer, query, key, value, kv_cache, attn_metadata, output,
                                      output_scale=output_scale, output_block_scale=output_block_scale)
                    state['saved_q'][layer, b].copy_(query)
                    state['saved_o'][layer, b].copy_(output)
                    return result
                output.copy_(state['saved_o'][layer, b])
                if mode == 'checked_cached_o':
                    state['check_q'][layer, b].copy_(query)
                    state['check_o'][layer, b].copy_(output)
                return output
            parent.attn.impl.forward = forward
        assert len(state['cache']) == 48
        old_tables = runner.prepare_attn
        def tables(*args, **kwargs):
            value = old_tables(*args, **kwargs)
            state['live_block_tables'], state['live_slot_mappings'] = value
            return value
        runner.prepare_attn = tables
        previous = runner.cudagraph_manager.run_fullgraph
        state['original_graph_run'] = bare_native_run
        def run(desc):
            if state['armed'] and not state['done'] and desc.num_tokens == desc.num_reqs == 3:
                state['seen'] += 1
                if state['seen'] == 3:
                    self._measure_cached_o(desc)
            return previous(desc)
        runner.cudagraph_manager.run_fullgraph = run
        result['cached_o_worker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result

    @torch.inference_mode()
    def capture_cached_o_graphs(self):
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
        runner = self.model_runner
        state = self.cached_o
        assert not state['armed']
        started = time.perf_counter()
        for mode in ('witness', 'checked_cached_o', 'cached_o'):
            native = runner.cudagraph_manager
            manager = ModelCudaGraphManager(runner.vllm_config, runner.device, native.cudagraph_mode,
                decode_query_len=runner.decode_query_len, lora_capture_cases=runner.lora_capture_cases,
                varlen_decode=False, ubatch_runner=None)
            state['capture_mode'] = mode
            manager.capture(runner.model, runner.model_state, runner.input_buffers, runner.intermediate_tensors,
                runner.block_tables, runner.attn_groups, runner.kv_cache_config,
                pcp_manager=runner.pcp_manager, has_lora=False,
                use_aux_hidden_state_outputs=runner.use_aux_hidden_state_outputs,
                progress_bar_desc=f'Capturing diagnostic {mode}')
            state['capture_mode'] = None
            state['managers'][mode] = manager
            assert state['captured_layers'][mode] == {(l, b) for l in range(48) for b in (1, 2, 3)}
        torch.cuda.synchronize()
        return {'rank': self.native_probe['rank'], 'capture_seconds': time.perf_counter()-started,
                'captured_layer_shapes': {k: len(v) for k, v in state['captured_layers'].items()},
                'scope': 'All dummy captures finished before fresh real requests; never recapture after freezing the real state.'}

    def arm_cached_o_diagnostic(self):
        assert len(self.cached_o['managers']) == 3
        self.cached_o['armed'] = True
        return {'rank': self.native_probe['rank'], 'trigger': 'third actual B3 decode graph, before scheduler advances'}

    @torch.inference_mode()
    def _measure_cached_o(self, desc):
        state = self.cached_o
        state['armed'] = False
        runner = self.model_runner
        b = desc.num_reqs
        native = state['original_graph_run']
        batch = self.native_probe['batch']
        assert batch.num_reqs == batch.num_tokens == b == 3
        assert len(state['live_slot_mappings']) == 1
        slots = state['live_slot_mappings'][0][:b].clone().long()
        assert bool((slots >= 0).all())
        token_ids = runner.input_buffers.input_ids[:b].clone()
        positions = runner.input_buffers.positions[:b].clone()
        tables = [t.clone() for t in state['live_block_tables']]
        def kv_now():
            return {l: c[slots // 16, :, slots % 16, :].clone() for l, c in state['cache'].items()}
        native_hidden = native(desc).clone()
        native_kv = kv_now()
        native_logits_result = runner.model.compute_logits(native_hidden)
        native_logits = native_logits_result.clone() if native_logits_result is not None else None
        native_local_logits = runner.model.logits_processor(runner.model.lm_head, native_hidden, skip_gather=True).clone()
        witness_hidden = state['managers']['witness'].run_fullgraph(desc).clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(witness_hidden, native_hidden, atol=0, rtol=0)
        saved_q = {l: state['saved_q'][l, b].clone() for l in range(48)}
        saved_o = {l: state['saved_o'][l, b].clone() for l in range(48)}
        checks = []
        for mode in ('checked_cached_o', 'cached_o'):
            hidden = state['managers'][mode].run_fullgraph(desc).clone()
            logits_result = runner.model.compute_logits(hidden)
            logits = logits_result.clone() if logits_result is not None else None
            local_logits = runner.model.logits_processor(runner.model.lm_head, hidden, skip_gather=True).clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(hidden, native_hidden, atol=0, rtol=0)
            assert (logits is None) == (native_logits is None)
            if logits is not None:
                torch.testing.assert_close(logits, native_logits, atol=0, rtol=0)
            torch.testing.assert_close(local_logits, native_local_logits, atol=0, rtol=0)
            after_kv = kv_now()
            assert all(torch.equal(after_kv[l], native_kv[l]) for l in range(48))
            checks.append({'mode': mode, 'hidden_bitwise_equal': True,
                           'gathered_logits_bitwise_equal': True if logits is not None else None,
                           'local_vocab_logits_bitwise_equal': True,
                           'written_KV_slots_bitwise_equal_all48': True})
        layer_checks = []
        for l in range(48):
            torch.testing.assert_close(state['check_q'][l, b], saved_q[l], atol=0, rtol=0)
            torch.testing.assert_close(state['check_o'][l, b], saved_o[l], atol=0, rtol=0)
            layer_checks.append({'layer': l, 'q_bitwise_equal': True, 'o_bitwise_equal': True,
                                 'q_sha256': digest(saved_q[l]), 'o_sha256': digest(saved_o[l])})
        methods = {'native': lambda: native(desc), 'cached_o': lambda: state['managers']['cached_o'].run_fullgraph(desc),
                   'native_AA': lambda: native(desc)}
        for fn in methods.values():
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        rows = []
        randomizer = random.Random(20261016)
        flush = torch.zeros(256*1024*1024, device=runner.device, dtype=torch.uint8)
        for cache_mode in ('model_warm', 'artificial_cold'):
            for block in range(state['config'].get('blocks', 12)):
                order = list(methods)
                randomizer.shuffle(order)
                for direction, names in (('random', order), ('reverse', list(reversed(order)))):
                    row = {'cache': cache_mode, 'block': block, 'direction': direction, 'cuda_us': {}, 'submit_and_sync_wall_seconds': {}}
                    for name in names:
                        if cache_mode == 'artificial_cold':
                            flush.add_(1)
                            torch.cuda.synchronize()
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        started = time.perf_counter()
                        begin.record()
                        methods[name]()
                        end.record()
                        end.synchronize()
                        row['submit_and_sync_wall_seconds'][name] = time.perf_counter()-started
                        row['cuda_us'][name] = begin.elapsed_time(end)*1000
                    rows.append(row)
        final = native(desc).clone()
        torch.testing.assert_close(final, native_hidden, atol=0, rtol=0)
        assert torch.equal(runner.input_buffers.input_ids[:b], token_ids)
        assert torch.equal(runner.input_buffers.positions[:b], positions)
        assert all(torch.equal(t, old) for t, old in zip(state['live_block_tables'], tables))
        after_kv = kv_now()
        assert all(torch.equal(after_kv[l], native_kv[l]) for l in range(48))
        result = {'status': 'completed', 'rank': self.native_probe['rank'], 'positions': positions.cpu().tolist(),
                  'input_tokens': token_ids.cpu().tolist(), 'physical_slots': slots.cpu().tolist(),
                  'checks': checks, 'layer_checks': layer_checks,
                  'hidden_sha256': digest(native_hidden),
                  'gathered_logits_sha256': digest(native_logits) if native_logits is not None else None,
                  'local_vocab_logits_sha256': digest(native_local_logits),
                  'final_inputs_tables_KV_hidden_unchanged': True, 'measurements': rows,
                  'scope': 'One frozen real B3 state; cached per-layer O is invalid for future positions. Optimistic intervention, not a deployable algorithm, natural trajectory, or strict mathematical upper bound.',
                  'limitations': ['Both timed paths call bare graph managers without per-step CPU identity hooks. CUDA-event graph intervals can include CPU launch gaps; AA and host times are retained.',
                                  'Cached path still writes KV and copies saved O; graph dependency structure differs. Whole-model output is bitwise checked.',
                                  'Only written KV slots are compared, not a full-cache checksum. No preparation cost is amortized into the diagnostic.',
                                  'Witness/check graph copy overhead is excluded from paired graph timing.']} 
        path = self.native_probe['output'] / 'cached_o_diagnostic.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
        state['result'] = result
        state['done'] = True

    def cached_o_diagnostic_result(self):
        assert self.cached_o['done'], 'Expected real B3 state was not reached'
        return self.cached_o['result']
