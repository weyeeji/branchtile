"""Instrument native router boundaries; qualify an existing FlashInfer GEMM.

The live model always executes its original gate/router/expert kernels. TinyGEMM
is used only after the instrumented native trajectories, on frozen actual inputs.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import torch

from scripts.probe_engine_docfinqa import DeploymentWorkerExtension


def bitwise_equal(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def routing_identity(actual_weights, actual_ids, weights, ids):
    if actual_weights.shape != weights.shape or actual_ids.shape != ids.shape:
        return False
    return (bool(torch.isfinite(actual_weights).all() and torch.isfinite(weights).all())
            and bitwise_equal(actual_ids, ids) and bitwise_equal(actual_weights, weights))


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def error_metrics(output, reference):
    delta = output.double() - reference
    return {'finite': bool(torch.isfinite(output).all()),
            'max_abs_vs_FP64': float(delta.abs().max()),
            'relative_l2_vs_FP64': float(delta.norm() / reference.norm().clamp_min(1e-30)),
            'rmse_vs_FP64': float(delta.square().mean().sqrt())}


def bf16_comparison(output, native):
    """Descriptive differences; never a post-hoc numerical tolerance."""
    assert output.dtype == native.dtype == torch.bfloat16
    delta = output.double() - native.double()
    # Monotone BF16 encoding, including signed zero; report representable steps.
    def ordered(x):
        bits = x.contiguous().view(torch.int16).to(torch.int32) & 65535
        return torch.where(bits & 32768 != 0, 65535 - bits, bits + 32768)
    ulp = (ordered(output) - ordered(native)).abs()
    return {'max_abs': float(delta.abs().max()),
            'relative_l2': float(delta.norm() / native.double().norm().clamp_min(1e-30)),
            'different_elements': int((output != native).sum()),
            'total_elements': output.numel(), 'max_BF16_encoding_steps': int(ulp.max()),
            'encoding_steps_note': 'Adjacent BF16 encodings; signed zero occupies two adjacent encodings.'}


class TinyGemmGateWorkerExtension(DeploymentWorkerExtension):
    def _save(self, name, value):
        (self.tiny_gate['output'] / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')

    def configure_tiny_gate_probe(self, config):
        from vllm.distributed import get_tp_group
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
        from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
        assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
        assert torch.cuda.get_device_capability() == (12, 0)
        assert 'RTX 5090' in torch.cuda.get_device_name()
        runner = self.model_runner
        state = self.tiny_gate = dict(config=config, rank=get_tp_group().rank_in_group,
            phase='warmup', capture=False, refs={}, snapshots={}, layers={}, gates={}, routers={},
            capture_calls={}, requests={}, steps=[], manager=None)
        state['output'] = Path(config['output']) / f"rank{state['rank']}"
        state['output'].mkdir(parents=True, exist_ok=True)
        assert runner.batch_sharder is None
        prepare = runner.prepare_inputs
        def prepare_inputs(*args, **kwargs):
            state['batch'] = prepare(*args, **kwargs)
            return state['batch']
        runner.prepare_inputs = prepare_inputs
        for name, module in runner.model.named_modules():
            if not isinstance(module, MoERunner):
                continue
            match = re.search(r'(?:^|\.)layers\.(\d+)\.', name)
            assert match is not None, name
            layer = int(match[1]); gate = module.gate; router = module.router
            assert layer not in state['gates']
            assert not module._fse_fuse_gate
            assert not module.routed_experts.quant_method.is_monolithic
            assert isinstance(router, FusedTopKRouter)
            assert router.top_k == 8 and router.global_num_experts == 128
            assert router.scoring_func == 'softmax' and router.renormalize
            assert router.eplb_state is None and router.capture_fn is None
            assert router._routing_replay_out is None
            assert gate.is_unquantized and gate.bias is None
            assert gate.weight.shape == (128, 2048) and gate.weight.dtype == torch.bfloat16
            assert gate.weight.is_contiguous() and gate.out_dtype in (None, torch.bfloat16)
            assert not gate.allow_specialized_router_gemm and not gate.allow_ll_bf16_gemm
            gate_original, route_original = gate.forward, router.select_experts
            state['gates'][layer] = (gate, gate_original)
            state['routers'][layer] = (router, route_original)
            state['layers'][layer] = {'module': name, 'gate_class': type(gate).__module__ + '.' + type(gate).__name__,
                'weight_shape': list(gate.weight.shape), 'weight_stride': list(gate.weight.stride()),
                'weight_dtype': str(gate.weight.dtype), 'weight_sha256': tensor_sha(gate.weight),
                'gate_out_dtype': str(gate.out_dtype), 'gate_return_bias': gate.return_bias,
                'allow_ll_bf16_gemm': gate.allow_ll_bf16_gemm,
                'allow_specialized_router_gemm': gate.allow_specialized_router_gemm,
                'allow_cublas_router_gemm': gate.allow_cublas_router_gemm,
                'router_class': type(router).__module__ + '.' + type(router).__name__,
                'top_k': router.top_k, 'scoring_func': router.scoring_func, 'renormalize': router.renormalize,
                'monolithic_expert_path': False, 'gate_fusion': False, 'EPLB': False}
            def gate_forward(x, layer=layer, original=gate_original):
                result = original(x)
                output = result[0] if isinstance(result, tuple) else result
                if state['capture']:
                    b = x.shape[0]
                    assert x.shape == (b, 2048) and b in (1, 2, 3)
                    assert x.dtype == output.dtype == torch.bfloat16 and x.is_contiguous()
                    assert output.shape == (b, 128) and output.is_contiguous()
                    ref = state['refs'].setdefault((b, layer), {})
                    if 'x' not in ref:
                        ref.update(x=torch.empty_like(x), logits=torch.empty_like(output))
                    ref['x'].copy_(x); ref['logits'].copy_(output)
                    key = f'gate:B{b}:L{layer}'
                    state['capture_calls'][key] = state['capture_calls'].get(key, 0) + 1
                return result
            def select(hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None,
                       layer=layer, original=route_original):
                weights, ids = original(hidden_states, router_logits,
                    topk_indices_dtype=topk_indices_dtype, input_ids=input_ids)
                if state['capture']:
                    b = hidden_states.shape[0]; ref = state['refs'][b, layer]
                    if 'weights' not in ref:
                        ref.update(weights=torch.empty_like(weights), ids=torch.empty_like(ids))
                    ref['weights'].copy_(weights); ref['ids'].copy_(ids)
                    key = f'route:B{b}:L{layer}'
                    state['capture_calls'][key] = state['capture_calls'].get(key, 0) + 1
                return weights, ids
            gate.forward = gate_forward; router.select_experts = select
        assert sorted(state['gates']) == list(range(48))
        native = runner.cudagraph_manager; original_run = native.run_fullgraph
        def run(desc):
            before_case = self._pure_case()
            poison = (before_case is not None and state['phase'] in ('full_A', 'full_B')
                      and (state['phase'], before_case['batch']) not in state['snapshots'])
            if poison:
                for layer in range(48):
                    for name, t in state['refs'][before_case['batch'], layer].items():
                        t.fill_(-1 if name == 'ids' else float('nan'))
                torch.cuda.synchronize()
            result = state['manager'].run_fullgraph(desc) if state['manager'] is not None else original_run(desc)
            case = self._pure_case()
            if case is not None and state['phase'] in ('full_A', 'full_B'):
                state['steps'].append(case)
                key = (state['phase'], case['batch'])
                if key not in state['snapshots']:
                    assert poison and before_case == case
                    for layer in range(48):
                        for name, t in state['refs'][case['batch'], layer].items():
                            assert bool(((t >= 0) & (t < 128)).all() if name == 'ids' else torch.isfinite(t).all())
                    started = time.perf_counter()
                    state['snapshots'][key] = {'case': case, 'layers': {
                        layer: {name: t.clone() for name, t in state['refs'][case['batch'], layer].items()}
                        for layer in range(48)}}
                    torch.cuda.synchronize()
                    state['snapshots'][key]['instrument_snapshot_seconds'] = time.perf_counter() - started
                    state['snapshots'][key]['poison_then_actual_graph_overwrite_verified'] = True
            return result
        native.run_fullgraph = run
        result = {'rank': state['rank'], 'layers': state['layers'], 'deployment': self.deployment_state(),
            'packages': {p: importlib.metadata.version(p) for p in ('torch', 'vllm', 'flashinfer-python')},
            'live_model_uses_original_gate_router_and_experts': True,
            'tinygemm_not_installed_in_model': True}
        self._save('configuration', result)
        return result

    def _pure_case(self):
        state = self.tiny_gate
        if state['capture'] or 'batch' not in state:
            return None
        batch = state['batch']; b = batch.num_reqs
        starts = batch.query_start_loc_np[:b + 1].tolist()
        if b not in (1, 2, 3) or any(y - x != 1 for x, y in zip(starts, starts[1:])):
            return None
        infos = []
        for i, rid in enumerate(batch.req_ids[:b]):
            if rid not in state['requests']:
                slot = int(batch.idx_mapping_np[i]); runner = self.model_runner
                length = int(runner.req_states.prompt_len.np[slot])
                tokens = runner.req_states.all_token_ids.gpu[slot, :length].cpu().tolist()
                digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
                assert digest in state['config']['prompt_token_hashes']
                state['requests'][rid] = {'question': state['config']['prompt_token_hashes'].index(digest),
                    'prompt_length': length, 'request_id': rid, 'prompt_token_sha256': digest}
            infos.append(state['requests'][rid])
        positions = batch.num_computed_tokens_np[:b].tolist()
        if any(p < i['prompt_length'] for p, i in zip(positions, infos)):
            return None
        assert batch.num_tokens_after_padding == b
        return {'phase': state['phase'], 'batch': b, 'positions': positions,
                'questions': [i['question'] for i in infos], 'requests': infos}

    @torch.inference_mode()
    def capture_tiny_gate_native_graphs(self):
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
        state = self.tiny_gate; runner = self.model_runner; native = runner.cudagraph_manager
        before = torch.cuda.memory_allocated(); started = time.perf_counter()
        manager = ModelCudaGraphManager(runner.vllm_config, runner.device, native.cudagraph_mode,
            decode_query_len=runner.decode_query_len, lora_capture_cases=runner.lora_capture_cases,
            varlen_decode=False, ubatch_runner=None)
        state['capture'] = True
        try:
            manager.capture(runner.model, runner.model_state, runner.input_buffers, runner.intermediate_tensors,
                runner.block_tables, runner.attn_groups, runner.kv_cache_config, pcp_manager=runner.pcp_manager,
                has_lora=False, use_aux_hidden_state_outputs=runner.use_aux_hidden_state_outputs,
                progress_bar_desc='Capturing read-only native Gate/router receipts')
        finally:
            state['capture'] = False
        state['manager'] = manager; torch.cuda.synchronize()
        assert set(state['refs']) == {(b, l) for b in (1, 2, 3) for l in range(48)}
        assert all(set(r) == {'x', 'logits', 'weights', 'ids'} for r in state['refs'].values())
        result = {'rank': state['rank'], 'wall_seconds': time.perf_counter() - started,
            'additional_allocated_bytes': torch.cuda.memory_allocated() - before,
            'persistent_copy_buffer_bytes': sum(t.numel() * t.element_size() for r in state['refs'].values() for t in r.values()),
            'capture_calls': state['capture_calls'], 'descriptors': [str(d) for d in manager.graphs],
            'copies_inside_graph_prevent_compiled_intermediate_storage_reuse': True,
            'scope': 'Native execution plus instrumentation; not production latency'}
        self._save('graph_capture', result)
        return result

    def set_tiny_gate_phase(self, phase):
        self.tiny_gate.update(phase=phase, steps=[])
        return {'rank': self.tiny_gate['rank'], 'phase': phase}

    def finish_tiny_gate_phase(self):
        state = self.tiny_gate
        result = {'rank': state['rank'], 'phase': state['phase'], 'steps': state['steps'],
            'batches': sorted({r['batch'] for r in state['steps']})}
        self._save(state['phase'], result)
        return result

    @torch.inference_mode()
    def qualify_and_time_tiny_gate(self):
        from flashinfer.gemm import tinygemm_bf16
        state = self.tiny_gate; started = time.perf_counter()
        assert set(state['snapshots']) == {(phase, b) for phase in ('full_A', 'full_B') for b in (1, 2, 3)}
        checks = []; aa = []; state['tiny_api'] = tinygemm_bf16
        first = state['snapshots']['full_A', 3]['layers'][0]
        compile_start = time.perf_counter()
        warm_output = torch.empty_like(first['logits'])
        tinygemm_bf16(first['x'], state['gates'][0][0].weight, warm_output, bias=None, use_pdl=False)
        torch.cuda.synchronize()
        jit_and_first_call_seconds = time.perf_counter() - compile_start
        # Preserve complete actual tensor receipts, with original checkpoint
        # weights identified by SHA rather than redundantly storing 48 matrices.
        saved = {f'{phase}_B{b}': {'case': snap['case'], 'layers': {
            l: {n: t.cpu() for n, t in r.items()} for l, r in snap['layers'].items()}}
            for (phase, b), snap in state['snapshots'].items()}
        tensor_path = state['output'] / 'actual_gate_snapshots.pt'; torch.save(saved, tensor_path)
        for b in (1, 2, 3):
            a = state['snapshots']['full_A', b]; z = state['snapshots']['full_B', b]
            assert all(a['case'][k] == z['case'][k] for k in ('batch', 'positions', 'questions'))
            for layer in range(48):
                aa.append({'batch': b, 'layer': layer, 'bitwise_equal': all(
                    bitwise_equal(a['layers'][layer][n], z['layers'][layer][n]) for n in ('x', 'logits', 'weights', 'ids'))})
        for layer in range(48):
            gate, original = state['gates'][layer]
            assert tensor_sha(gate.weight) == state['layers'][layer]['weight_sha256']
            weight64 = gate.weight.double()
            for phase in ('full_A', 'full_B'):
                for b in (1, 2, 3):
                    ref = state['snapshots'][phase, b]['layers'][layer]; x = ref['x']
                    expected = x.double() @ weight64.T
                    assert bool(torch.isfinite(expected).all())
                    native_result = original(x)
                    native = native_result[0] if isinstance(native_result, tuple) else native_result
                    candidate = torch.empty_like(ref['logits'])
                    tinygemm_bf16(x, gate.weight, candidate, bias=None, use_pdl=False)
                    route = state['routers'][layer][1]
                    native_weights, native_ids = route(x, native, topk_indices_dtype=ref['ids'].dtype)
                    weights, ids = route(x, candidate, topk_indices_dtype=ref['ids'].dtype)
                    same_native = bitwise_equal(native, ref['logits']) and routing_identity(
                        ref['weights'], ref['ids'], native_weights, native_ids)
                    exact_rows = [routing_identity(ref['weights'][r], ref['ids'][r], weights[r], ids[r]) for r in range(b)]
                    row = {'phase': phase, 'batch': b, 'layer': layer,
                        'x_sha256': tensor_sha(x), 'actual_logits_sha256': tensor_sha(ref['logits']),
                        'candidate_logits_sha256': tensor_sha(candidate),
                        'native_same_input_matches_actual_graph_bitwise': same_native,
                        'candidate_logits_bitwise_equal_native': bitwise_equal(candidate, native),
                        'native_error': error_metrics(native, expected), 'candidate_error': error_metrics(candidate, expected),
                        'candidate_vs_native_BF16': bf16_comparison(candidate, native),
                        'selected_ids_native': ref['ids'].cpu().tolist(), 'selected_ids_candidate': ids.cpu().tolist(),
                        'selected_weights_native': ref['weights'].cpu().tolist(), 'selected_weights_candidate': weights.cpu().tolist(),
                        'exact_routing_identity_by_row': exact_rows,
                        'changed_selected_id_elements': int((ids != ref['ids']).sum()),
                        'changed_selected_weight_elements': int((weights != ref['weights']).sum()),
                        'max_selected_weight_abs_difference': float((weights - ref['weights']).abs().max()),
                        'passed_exact_contract': same_native and all(exact_rows) and bool(torch.isfinite(candidate).all())}
                    checks.append(row)
        qualified = all(r['bitwise_equal'] for r in aa) and all(r['passed_exact_contract'] for r in checks)
        result = {'rank': state['rank'], 'status': 'qualified_bitwise_drop_in' if qualified else 'not_qualified_bitwise_drop_in',
            'qualified': qualified, 'checks': checks, 'cold_AA_points': aa,
            'expected_points': 2 * 3 * 48, 'snapshot_file': str(tensor_path),
            'snapshot_sha256': hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
            'qualification_wall_seconds': time.perf_counter() - started,
            'tinygemm_JIT_and_first_call_seconds': jit_and_first_call_seconds,
            'poison_overwrite_receipts': {f'{p}_B{b}': s['poison_then_actual_graph_overwrite_verified']
                for (p, b), s in state['snapshots'].items()},
            'FP64_scope': 'Descriptive error for full logits on actual BF16 X and W; no borrowed attention tolerance or fitted threshold.',
            'exact_contract': 'Conservative direct-replacement gate: all captured rows use same actual official router, expert IDs including order and FP32 normalized weights bitwise identical; modular expert consumer receives only those routing outputs.',
            'failure_interpretation': 'Rounding differences do not establish incorrect TinyGEMM mathematics or permanent ineligibility. A non-bitwise implementation needs separately frozen model-level numerical/quality validation before substitution.'}
        self._save('qualification', result)
        state['qualified'] = qualified
        result['timing'] = {'status': 'pending_both_rank_qualification' if qualified
                            else 'skipped_all_candidate_timing_due_to_failed_exact_contract'}
        self._save('qualification', result)
        return {k: result[k] for k in ('rank', 'status', 'qualified', 'expected_points', 'qualification_wall_seconds', 'timing')}

    def time_qualified_tiny_gate(self):
        assert self.tiny_gate['qualified']
        return self._time_tiny_gate()

    @torch.inference_mode()
    def _time_tiny_gate(self):
        state = self.tiny_gate; config = state['config']['timing']; started = time.perf_counter()
        methods = ('native_A', 'native_B', 'tinygemm'); graphs = {}; outputs = {}
        for b in (1, 2, 3):
            for contract in ('gate_only', 'gate_and_router'):
                for method in methods:
                    def chain(b=b, contract=contract, method=method):
                        values = []
                        for layer in range(48):
                            ref = state['snapshots']['full_A', b]['layers'][layer]
                            x = ref['x']; gate, original = state['gates'][layer]
                            if method == 'tinygemm':
                                y = torch.empty_like(ref['logits'])
                                state['tiny_api'](x, gate.weight, y, bias=None, use_pdl=False)
                            else:
                                r = original(x); y = r[0] if isinstance(r, tuple) else r
                            if contract == 'gate_and_router':
                                w, ids = state['routers'][layer][1](x, y, topk_indices_dtype=ref['ids'].dtype)
                                values.append((y, w, ids))
                            else:
                                values.append((y,))
                        return values
                    chain(); torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        values = chain()
                    graphs[b, contract, method] = graph; outputs[b, contract, method] = values
                    graph.replay(); torch.cuda.synchronize()
                    if contract == 'gate_and_router':
                        for layer, (_, w, ids) in enumerate(values):
                            ref = state['snapshots']['full_A', b]['layers'][layer]
                            assert routing_identity(ref['weights'], ref['ids'], w, ids)
        pointers = {}
        for key, values in outputs.items():
            pointers[key] = {t.data_ptr() for group in values for t in group}
        keys = list(pointers)
        assert all(pointers[a].isdisjoint(pointers[b]) for i, a in enumerate(keys) for b in keys[i + 1:])
        preparation = time.perf_counter() - started
        flush = torch.empty(config['cache_flush_bytes'], dtype=torch.uint8, device='cuda')
        rng = random.Random(config['seed']); measurements = []
        begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        for b in (1, 2, 3):
            for contract in ('gate_only', 'gate_and_router'):
                for cache in ('hot', 'cold'):
                    for block in range(config['blocks']):
                        order = list(methods); rng.shuffle(order)
                        for direction, arranged in [('random', order), ('reverse', list(reversed(order)))]:
                            for method in arranged:
                                graph = graphs[b, contract, method]; samples = []
                                for _ in range(config['hot_repeats'] if cache == 'hot' else 1):
                                    if cache == 'cold': flush.fill_(1)
                                    else: graph.replay()
                                    torch.cuda.synchronize()
                                    wall = time.perf_counter(); begin.record(); graph.replay(); end.record(); end.synchronize()
                                    samples.append({'cuda_us': begin.elapsed_time(end) * 1000,
                                                    'complete_wall_us': (time.perf_counter() - wall) * 1e6})
                                measurements.append({'batch': b, 'contract': contract, 'cache': cache,
                                    'block': block, 'direction': direction, 'method': method, 'samples': samples})
        assert len(measurements) == 3 * 2 * 2 * config['blocks'] * 2 * 3
        assert sum(len(m['samples']) for m in measurements) == 3 * 2 * config['blocks'] * 2 * 3 * (config['hot_repeats'] + 1)
        result = {'rank': state['rank'], 'status': 'completed', 'graph_preparation_seconds': preparation,
            'timing_and_preparation_seconds': time.perf_counter() - started, 'measurements': measurements,
            'layer_order': list(range(48)), 'methods_have_disjoint_outputs': True,
            'same_frozen_X_and_live_W': True, 'PDL': False,
            'scope': '48 real gate weights and actual native inputs; same actual router optionally included. No extra output.copy_ in either arm because GateLinear returns its GEMM output directly. Outputs retained in separate graph pools.',
            'limits': ['Attention/MoE/TP interleaving is absent; this is not a whole-model step.',
                       'A/A and random/reverse samples are clustered within blocks; replicas/ranks are not independent documents.',
                       'Cold flush is outside timing and does not reproduce the actual model cache state.',
                       'Graph preparation and instrumented model capture are reported separately; no application speedup claim.']}
        self._save('timing', result)
        return {k: result[k] for k in ('status', 'graph_preparation_seconds', 'timing_and_preparation_seconds')}
