"""Observe consecutive real native AR queries; never synthesize verifier Q."""
import hashlib
import json
from pathlib import Path
import re

import torch

from scripts.vllm_native_precision_worker import NativePrecisionWorkerExtension


class RaggedStateCaptureWorker(NativePrecisionWorkerExtension):
    def configure_ragged_state_capture(self, config):
        assert self.vllm_config.model_config.enforce_eager
        result = self.configure_native_precision_probe(config)
        runner = self.model_runner
        state = self.ragged_capture = dict(config=config, armed=False, layers={}, files=[], tables=None)
        old_tables = runner.prepare_attn

        def tables(*args, **kwargs):
            value = old_tables(*args, **kwargs)
            assert len(value[0]) == 1
            state['tables'] = value[0][0]
            return value

        runner.prepare_attn = tables
        for name, parent in runner.model.named_modules():
            match = re.search(r'(?:^|\.)layers\.(\d+)\.self_attn$', name)
            if not match or int(match[1]) not in config['layers']:
                continue
            layer = int(match[1])
            module = parent.attn
            assert not module.attn_backend.forward_includes_kv_cache_update
            assert module.attn_backend.get_name() == 'FLASHINFER'
            state['layers'][layer] = dict(q=[], o=[], positions=[], rows=[], done=False)
            original = module.impl.forward

            def forward(attn_layer, query, key, value, kv_cache, attn_metadata, output,
                        output_scale=None, output_block_scale=None, layer=layer, original=original):
                result = original(attn_layer, query, key, value, kv_cache, attn_metadata,
                                  output, output_scale=output_scale, output_block_scale=output_block_scale)
                record = state['layers'][layer]
                if not state['armed'] or record['done']:
                    return result
                batch = self.native_probe['batch']
                b = config['batch']
                if batch.num_reqs != b or batch.num_tokens != b or query.shape[0] != b:
                    return result
                rows = [self.native_probe['requests'][rid] for rid in batch.req_ids[:b]]
                assert sorted(rows) == list(range(b)), rows
                positions = runner.input_buffers.positions[:b].detach().cpu().long()
                order = torch.tensor(sorted(range(b), key=rows.__getitem__), dtype=torch.long)
                positions = positions[order]
                if record['positions']:
                    assert torch.equal(positions, record['positions'][-1]+1), 'Nonconsecutive native trajectory'
                record['q'].append(query.detach().cpu().reshape(b, 16, 128)[order].clone())
                record['o'].append(output.detach().cpu().reshape(b, 16, 128)[order].clone())
                record['positions'].append(positions)
                record['rows'].append(rows)
                if len(record['q']) == config['steps']:
                    table = state['tables'][:b].detach().cpu().long()[order]
                    lens = positions+1
                    old_lengths = record['positions'][0]
                    common_limit = int(old_lengths.min())//16
                    same = (table[:, :common_limit] == table[:1, :common_limit]).all(0)
                    mismatch = (~same).nonzero().flatten()
                    common = int(mismatch[0]) if mismatch.numel() else common_limit
                    assert common > 0
                    width = int(((lens+15)//16).max())
                    used = torch.unique(torch.cat([table[r, :(int(lens[r])+15)//16] for r in range(b)]), sorted=True)
                    compact = torch.zeros((b, width), dtype=torch.int32)
                    for r in range(b):
                        count = (int(lens[r])+15)//16
                        compact[r, :count] = torch.searchsorted(used, table[r, :count]).int()
                    assert kv_cache.ndim == 4 and tuple(kv_cache.shape[1:]) == (2, 16, 256)
                    source = kv_cache.view(torch.uint8) if kv_cache.dtype == torch.float8_e4m3fn else kv_cache
                    payload = source.index_select(0, used.to(kv_cache.device)).cpu()
                    directory = self.native_probe['output']/'ragged_capture'
                    directory.mkdir(exist_ok=True)
                    path = directory/f'layer{layer}.pt'
                    assert not path.exists()
                    torch.save(dict(schema='real_ar_multitoken_attention_v1', layer=layer,
                        rank=self.native_probe['rank'], q=torch.stack(record['q'], dim=1),
                        native_o=torch.stack(record['o'], dim=1),
                        positions=torch.stack(record['positions'], dim=1),
                        cache=payload, block_tables=compact, seq_lens=lens.tolist(),
                        common_prefix_tokens=common*16, original_page_ids=used,
                        original_cache_shape=list(kv_cache.shape), original_cache_stride=list(kv_cache.stride()),
                        original_cache_dtype=str(kv_cache.dtype),
                        kv_semantic_dtype='float8_e4m3fn' if kv_cache.dtype in (torch.uint8, torch.float8_e4m3fn) else 'bfloat16',
                        k_scale=float(attn_layer._k_scale_float), v_scale=float(attn_layer._v_scale_float),
                        prompt_token_hashes=config['prompt_token_hashes'], row_orders=record['rows'],
                        scope='Actual consecutive native AR states; causal attention replay, not a whole-model verifier or acceptance benchmark'), path)
                    state['files'].append(dict(layer=layer, path=str(path), bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), common_prefix_tokens=common*16,
                        query_steps=config['steps']))
                    record['done'] = True
                return result

            module.impl.forward = forward
        assert sorted(state['layers']) == sorted(config['layers'])
        return result

    def arm_ragged_state_capture(self):
        assert not self.ragged_capture['armed']
        self.ragged_capture['armed'] = True
        return dict(rank=self.native_probe['rank'], armed=True)

    def ragged_state_capture_result(self):
        state = self.ragged_capture
        complete = all(v['done'] for v in state['layers'].values())
        result = dict(status='completed' if complete else 'incomplete', rank=self.native_probe['rank'],
                      files=state['files'], steps={k:len(v['q']) for k,v in state['layers'].items()},
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (self.native_probe['output']/'ragged_capture_receipt.json').write_text(json.dumps(result, indent=2)+'\n')
        assert complete, result
        return result
