"""Independent failure-artifact extension; original numerical gate is unchanged."""
import hashlib
import json
from pathlib import Path
import torch
from scripts.vllm_dense_cascade_worker import DenseCascadeWorker


class DenseCascadeFailureWorker(DenseCascadeWorker):
    def configure_dense_cascade(self, config):
        result = super().configure_dense_cascade(config)
        state = self.dense_cascade
        for layer, (attn_layer, original) in list(state['native_layers'].items()):
            def reference(*args, layer=layer, original=original, **kwargs):
                returned = original(*args, **kwargs)
                expected = kwargs['output']
                b = expected.shape[0]
                actual = state['saved_o'][layer, b]
                passed = torch.isclose(actual.float(), expected.float(), atol=.006, rtol=.015)
                if not bool(passed.all()):
                    output = self.native_probe['output'] / 'failure_snapshot.pt'
                    assert not output.exists(), output
                    seq_lens = state['pending']['receipt']['seq_lens']
                    tables = state['live_tables'][:b].cpu()
                    page_counts = [(n+15)//16 for n in seq_lens]
                    used = sorted(set(p for row,n in enumerate(page_counts) for p in tables[row,:n].tolist()))
                    mapping = {physical:i for i,physical in enumerate(used)}
                    remapped = torch.zeros((b,max(page_counts)),dtype=torch.int32)
                    for row,count in enumerate(page_counts):
                        remapped[row,:count] = torch.tensor([mapping[p] for p in tables[row,:count].tolist()],dtype=torch.int32)
                    cache = state['caches'][layer]
                    layer_module = state['native_layers'][layer][0]
                    payload = {'schema':'dense_cascade_failure_v1','layer':layer,'rank':self.native_probe['rank'],
                        'positions':state['pending']['positions'],'seq_lens':seq_lens,
                        'common_prefix_tokens':config['common_prefix_tokens'],
                        'query':state['saved_q'][layer,b].cpu(), 'candidate_o':actual.cpu(),
                        'native_o':expected.cpu(), 'failed_mask':(~passed).cpu(),
                        'used_physical_pages':used,'block_tables':remapped,
                        'cache':cache[torch.tensor(used,device=cache.device,dtype=torch.long)].cpu(),
                        'original_cache_shape':list(cache.shape),'original_cache_stride':list(cache.stride()),
                        'k_scale':float(layer_module._k_scale_float),'v_scale':float(layer_module._v_scale_float),
                        'atol':.006,'rtol':.015,
                        'scope':'Captured before unchanged gate raises. Q and current complete physical KV retained for independent FP64 reference.'}
                    torch.save(payload,output)
                    record = {k:v for k,v in payload.items() if not isinstance(v,torch.Tensor) and k!='used_physical_pages'}
                    record.update(snapshot_path=str(output), failed_values=int((~passed).sum()),
                        sha256=hashlib.sha256(output.read_bytes()).hexdigest(), bytes=output.stat().st_size)
                    output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
                return returned
            state['native_layers'][layer] = (attn_layer, reference)
        result['failure_worker_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result
