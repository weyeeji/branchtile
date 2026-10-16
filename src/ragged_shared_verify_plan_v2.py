"""Full-support multi-query cascade control with causal private histories.

Rows must be grouped by group ID. Preparing and graph capture are explicit;
this G0 control neither schedules requests nor provides an online speed claim.
"""
import itertools
import time

import flashinfer
import torch

from src.vllm_dense_cascade_plan import fused_cache_hnd_views


def ragged_metadata(group_ids, prefix_tokens, query_counts, seq_lens, tables):
    """CPU-only checks and index construction; no CUDA requirement."""
    groups = list(map(int, group_ids))
    counts = list(map(int, query_counts))
    lens = list(map(int, seq_lens))
    b = len(groups)
    assert b > 0 and len(counts) == len(lens) == b
    assert groups == sorted(groups) and sorted(set(groups)) == list(range(len(prefix_tokens)))
    assert tables.device.type == 'cpu' and tables.ndim == 2 and tables.shape[0] == b
    assert all(n > 0 for n in counts)
    assert all(p > 0 and p % 16 == 0 for p in prefix_tokens)
    prefix_q, prefix_ind, prefix_ids = [0], [0], []
    tail_ind, tail_ids, tail_last = [0], [], []
    full_ind, full_ids, full_last = [0], [], []
    for g, rows_iterator in itertools.groupby(range(b), key=lambda r:groups[r]):
        rows = list(rows_iterator)
        pages = prefix_tokens[g]//16
        assert tables.shape[1] >= pages
        assert all(torch.equal(tables[r, :pages], tables[rows[0], :pages]) for r in rows), 'Prefix physical identity differs'
        prefix_ids.extend(tables[rows[0], :pages].tolist())
        prefix_ind.append(len(prefix_ids))
        prefix_q.append(prefix_q[-1]+sum(counts[r] for r in rows))
    for r, (g,n,length) in enumerate(zip(groups,counts,lens)):
        tail = length-prefix_tokens[g]
        assert tail >= n, 'All verification queries must lie after shared complete prefix'
        total_pages = (length+15)//16
        assert tables.shape[1] >= total_pages
        prefix_pages = prefix_tokens[g]//16
        tail_ids.extend(tables[r, prefix_pages:total_pages].tolist())
        tail_ind.append(len(tail_ids))
        tail_last.append((tail-1)%16+1)
        full_ids.extend(tables[r,:total_pages].tolist())
        full_ind.append(len(full_ids))
        full_last.append((length-1)%16+1)
    qind = [0]+list(itertools.accumulate(counts))
    return dict(query_indptr=qind, prefix_query_indptr=prefix_q,
                prefix_indptr=prefix_ind, prefix_ids=prefix_ids, prefix_last=[16]*len(prefix_tokens),
                tail_indptr=tail_ind, tail_ids=tail_ids, tail_last=tail_last,
                full_indptr=full_ind, full_ids=full_ids, full_last=full_last)


class RaggedSharedVerifyPlanV2:
    def __init__(self, cache, group_ids, prefix_tokens, query_counts, seq_lens,
                 tables_cpu, k_scale=1., v_scale=1.):
        started = time.perf_counter()
        self.cache = cache
        self.kv = fused_cache_hnd_views(cache)
        self.k_scale, self.v_scale = float(k_scale), float(v_scale)
        assert self.k_scale > 0 and self.v_scale > 0
        self.metadata = ragged_metadata(group_ids, prefix_tokens, query_counts, seq_lens, tables_cpu)
        self.device = cache.device
        self.workspace = torch.empty(512*1024**2, device=self.device, dtype=torch.uint8)
        self.wrappers = {}
        self.buffers = {}
        for name in ('prefix', 'tail', 'full'):
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, kv_layout='HND', backend='fa2')
            qkey = 'prefix_query_indptr' if name == 'prefix' else 'query_indptr'
            q = torch.tensor(self.metadata[qkey], dtype=torch.int32)
            ind = torch.tensor(self.metadata[name+'_indptr'], dtype=torch.int32)
            ids = torch.tensor(self.metadata[name+'_ids'], device=self.device, dtype=torch.int32)
            last = torch.tensor(self.metadata[name+'_last'], dtype=torch.int32)
            w.plan(q, ind, ids, last, 16, 2, 128, 16, causal=name!='prefix',
                   q_data_type=torch.bfloat16, kv_data_type=self.kv[0].dtype, sm_scale=128**-.5)
            self.wrappers[name] = w
            self.buffers[name] = (q,ind,ids,last)
        shape = (sum(query_counts),16,128)
        self.outputs = {n:torch.empty(shape, device=self.device, dtype=torch.bfloat16) for n in self.wrappers}
        self.lse = {n:torch.empty(shape[:2], device=self.device, dtype=torch.float32) for n in self.wrappers}
        self.receipt = dict(prepare_host_seconds=time.perf_counter()-started,
            workspace_bytes=512*1024**2, workspace_shared_by_all_three_wrappers=True,
            groups=group_ids, prefix_tokens=prefix_tokens, query_counts=query_counts,
            seq_lens=seq_lens, plan_calls=3, scope='Offline G0 static plan; no assumption of online reuse',
            causal_private_tail=True, complete_support=True)

    def _run(self, name, query):
        self.wrappers[name].run(query, self.kv, out=self.outputs[name], lse=self.lse[name],
            return_lse=True, k_scale=self.k_scale, v_scale=self.v_scale)
        return self.outputs[name]

    def native_full(self, query):
        return self._run('full', query)

    def shared(self, query):
        out = self._run('prefix', query)
        tail = self._run('tail', query)
        flashinfer.merge_state_in_place(out, self.lse['prefix'], tail, self.lse['tail'])
        return out
