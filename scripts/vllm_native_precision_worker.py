"""Read-only precision receipts for the unchanged official native backend."""
import hashlib
import json
from pathlib import Path
import re
import time

from scripts.vllm_native_backend_worker import NativeBackendWorkerExtension


class NativePrecisionWorkerExtension(NativeBackendWorkerExtension):
    def configure_native_precision_probe(self, config):
        result = NativeBackendWorkerExtension.configure_native_probe(self, config)
        runner = self.model_runner
        state = self.native_probe
        state['admission_identity_records'] = []
        add_requests = runner.add_requests
        def add(scheduler_output):
            for request in scheduler_output.scheduled_new_reqs:
                started = time.perf_counter()
                ids = request.prefill_token_ids[:request.prompt_len]
                digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                assert digest in config['prompt_token_hashes']
                question = config['prompt_token_hashes'].index(digest)
                state['requests'][request.req_id] = question
                state['admission_identity_records'].append({
                    'request_id': request.req_id, 'question': question,
                    'phase': state['phase'], 'prompt_tokens': len(ids),
                    'source': 'scheduler CPU prefill_token_ids before original add_requests',
                    'CPU_identity_wall_seconds': time.perf_counter()-started})
            return add_requests(scheduler_output)
        runner.add_requests = add
        recorded_run = runner.cudagraph_manager.run_fullgraph
        def run(desc):
            batch = state['batch']
            assert all(rid in state['requests'] for rid in batch.req_ids[:batch.num_reqs])
            # This precondition makes the inherited GPU prompt-readback branch
            # unreachable, rather than assuming its cost is negligible.
            return recorded_run(desc)
        runner.cudagraph_manager.run_fullgraph = run
        result['precision_probe_identity'] = 'Prompt hash from CPU admission; no GPU token readback in graph wrapper. CPU hashing remains charged.'
        return result

    def native_precision_receipt(self, expected_dtype, require_decode=False):
        import torch
        from vllm.distributed import get_tp_group
        runner = self.model_runner
        assert self.vllm_config.cache_config.cache_dtype == expected_dtype
        assert self.vllm_config.cache_config.block_size == 16
        assert runner.batch_sharder is None
        layers = []
        storages = {}
        for name, parent in runner.model.named_modules():
            match = re.search(r'(?:^|\.)layers\.(\d+)\.self_attn$', name)
            if not match:
                continue
            module = parent.attn
            assert parent.num_heads == 16 and parent.num_kv_heads == 2 and parent.head_dim == 128
            assert module.attn_backend.get_name() == 'FLASHINFER'
            cache = module.kv_cache
            expected_storage = (torch.bfloat16,) if expected_dtype == 'auto' else (torch.uint8, torch.float8_e4m3fn)
            assert cache.dtype in expected_storage
            assert cache.shape[1:] == (2, 16, 256)
            scales = {key: float(getattr(module, key).item()) for key in ('_q_scale', '_k_scale', '_v_scale')}
            scale_floats = {key + '_float': float(getattr(module, key + '_float')) for key in scales}
            assert all(value == 1.0 for value in scales.values())
            assert all(scale_floats[key + '_float'] == value for key, value in scales.items())
            storage = cache.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
            layers.append({'layer': int(match[1]), 'name': name, 'backend': module.attn_backend.get_name(),
                           'implementation': type(module.impl).__module__ + '.' + type(module.impl).__name__,
                           'logical_cache_dtype': str(module.kv_cache_dtype), 'storage_dtype': str(cache.dtype),
                           'shape': list(cache.shape), 'stride': list(cache.stride()),
                           'payload_bytes': cache.numel() * cache.element_size(),
                           'storage_offset': cache.storage_offset(), 'scales': scales, 'host_scales': scale_floats})
        assert sorted(row['layer'] for row in layers) == list(range(48))
        metadata = list(self.native_probe['metadata'].values())
        if require_decode:
            assert metadata and any('decode' in meta for meta in metadata)
        for meta in metadata:
            assert meta['q_data_type_prefill'] == meta['q_data_type_decode'] == 'torch.bfloat16'
            if 'decode' in meta:
                assert 'xqa' in meta['decode']['kernel'].lower()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        result = {'rank': get_tp_group().rank_in_group, 'expected_cache_dtype': expected_dtype,
                  'layers': layers, 'cache_unique_storage_bytes': sum(storages.values()),
                  'allocated_cache_token_capacity': layers[0]['shape'][0] * 16,
                  'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'device_free_bytes': free,
                  'device_total_bytes': total, 'actual_metadata': metadata,
                  'graphs': [str(desc) for desc in runner.cudagraph_manager.graphs],
                  'admission_identity_records': list(self.native_probe['admission_identity_records']),
                  'worker_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'scope': 'Read-only RPC outside measured generation. No attention or CUDA graph alteration; inherited CPU identity/step hooks remain charged.'}
        return result
