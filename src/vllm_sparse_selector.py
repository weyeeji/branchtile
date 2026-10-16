"""Frozen QC-SnapKV history masks from actual vLLM suffix-prefill observations.

No selector novelty is claimed. APC-skipped Q may only be reused when the
complete token prefix and physical page identity agree. All temporary K gathers,
score passes and stable selection belong to preparation, not free execution.
"""
import hashlib
import json
import time

import torch
from .selector_baseline import qc_scores, select_support


class FrozenQCSelector:
    def __init__(self, prompt_tokens, document_range, budget, joint_support_exchange=None):
        self.prompts = prompt_tokens
        self.doc_start, self.doc_end = document_range
        self.budget = int(budget)
        if not 36 <= self.budget <= self.doc_end - self.doc_start:
            raise ValueError('Expected protected-tail-compatible fixed document budget')
        self.lcp = {}
        self.observation_windows = [(max(self.doc_end, len(p)-32), len(p)) for p in prompt_tokens]
        for a, prompt in enumerate(prompt_tokens):
            for b, other in enumerate(prompt_tokens):
                length = 0
                for x, y in zip(prompt, other):
                    if x != y:
                        break
                    length += 1
                self.lcp[a, b] = length
        self.observations, self.masks, self.records = {}, {}, []
        self.joint_config = None
        self.joint_pending_scores, self.joint_transfers = {}, []
        self.independent_masks = None
        self.joint_report = None
        if joint_support_exchange is not None:
            from .joint_support_exchange import JointSupportExchangeConfig
            self.joint_config = JointSupportExchangeConfig.from_mapping(joint_support_exchange)

    def observe(self, layer, question, positions, query, block_table, page_size):
        """Keep actual suffix Q, including Q later reused through APC full pages."""
        prompt_length = len(self.prompts[question])
        # A longer source prompt can contain the shorter target's observation
        # window outside its own last32 positions. Keep the union now: after APC
        # skips that complete page, its exact post-RoPE Q cannot be recomputed by
        # simply borrowing the target's final Q. freeze() still checks LCP and
        # the live physical page identity before any cross-question reuse.
        needed = torch.zeros_like(positions, dtype=torch.bool)
        for start, end in self.observation_windows:
            needed |= (positions >= start) & (positions < min(end, prompt_length))
        if not bool(needed.any()):
            return
        indices = torch.where(needed)[0]
        saved = query[indices.to(query.device)].detach().clone()
        for row, index in enumerate(indices.tolist()):
            position = int(positions[index])
            page_id = int(block_table[position // page_size])
            # Keep a per-question source so unrelated/evicted pages cannot match.
            self.observations[layer, position, question] = (page_id, saved[row])

    def freeze(self, layer, question, cache, block_table):
        key = layer, question
        if key in self.masks:
            return
        started = time.perf_counter()
        h, page_size, dim = cache.shape[1], cache.shape[2], cache.shape[3]//2
        prompt_length = len(self.prompts[question])
        start = max(self.doc_end, prompt_length-32)
        observations, reused = [], 0
        for position in range(start, prompt_length):
            page_id = int(block_table[position // page_size])
            found = None
            for source in [question] + [q for q in range(len(self.prompts)) if q != question]:
                candidate = self.observations.get((layer, position, source))
                if candidate is not None and candidate[0] == page_id and position < self.lcp[question, source]:
                    found = candidate[1]
                    reused += int(source != question)
                    break
            if found is None:
                raise RuntimeError(f'Missing exact APC observation Q: layer={layer}, question={question}, position={position}, page={page_id}')
            observations.append(found)
        assert observations, 'Question suffix cannot be empty'
        q = torch.stack(observations).permute(1, 0, 2).contiguous()[None]
        positions = torch.arange(start, prompt_length, device=cache.device)
        all_positions = torch.arange(prompt_length, device=cache.device)
        table = block_table.to(cache.device)
        pages = table[all_positions // page_size].long()
        offsets = all_positions % page_size
        # Explicit temporary gather for the existing selector. Decode execution
        # still references original physical cache without K/V payload copies.
        k = cache[pages, :, offsets, :dim].permute(1, 0, 2).contiguous()[None]
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            scores = qc_scores(q, k, positions, self.doc_start, self.doc_end)
            support = select_support(scores, self.budget, backend='argsort')
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        logical = [torch.where(support[head].cpu())[0] + self.doc_start for head in range(h)]
        assert all(len(ids) == self.budget for ids in logical)
        assert all(bool(torch.isin(torch.arange(self.doc_end-32, self.doc_end), ids).all()) for ids in logical)
        self.masks[key] = logical
        if self.joint_config is not None:
            transfer_started = time.perf_counter()
            assert scores.dtype == torch.float32
            self.joint_pending_scores[key] = scores.detach().cpu().contiguous().numpy().copy()
            self.joint_transfers.append({'layer':layer, 'question':question,
                'source_device':str(scores.device), 'score_bytes':scores.numel()*scores.element_size(),
                'GPU_to_CPU_bytes':scores.numel()*scores.element_size() if scores.is_cuda else 0,
                'transfer_and_owned_CPU_copy_wall_seconds':time.perf_counter()-transfer_started})
        torch.cuda.synchronize()
        self.records.append({'layer': layer, 'question': question, 'budget': self.budget,
            'observation_positions': list(range(start, prompt_length)), 'reused_other_question_prefix_Q': reused,
            'selector_prepare_seconds': time.perf_counter()-started,
            'temporary_K_gather_bytes': k.numel()*k.element_size(),
            'mask_sha256': [hashlib.sha256(ids.numpy().tobytes()).hexdigest() for ids in logical]})

    def logical_supports(self, layer, question, query_position):
        if query_position < len(self.prompts[question]):
            raise ValueError('Frozen history applies only after dense suffix prefill')
        self.require_finalized_joint_supports()
        ids = self.masks[layer, question]
        prefix = torch.arange(self.doc_start)
        suffix = torch.arange(self.doc_end, query_position+1)
        return [torch.cat((prefix, document, suffix)) for document in ids]

    def require_finalized_joint_supports(self):
        if self.joint_config is not None and self.joint_report is None:
            raise RuntimeError('Opt-in joint cohort must be finalized before sparse execution/capture')

    def mask_identity(self, masks=None):
        masks = self.masks if masks is None else masks
        rows = [{'layer':layer, 'question':question,
                 'head_sha256':[hashlib.sha256(x.numpy().astype('<i8',copy=False).tobytes()).hexdigest() for x in ids]}
                for (layer,question),ids in sorted(masks.items())]
        return {'sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(), 'rows':rows}

    def finalize_joint_supports(self, expected_layers=48, expected_heads=2):
        """One atomic exchange per frozen layer/head cohort, before any graph use."""
        if self.joint_config is None:
            return {'enabled':False, 'independent_selector_unchanged':True}
        if self.joint_report is not None:
            raise RuntimeError('Joint supports have already been finalized')
        import numpy as np
        from dataclasses import asdict
        from .joint_support_exchange import apply_joint_exchange
        started, cpu_started = time.perf_counter(), time.process_time()
        expected = {(l,q) for l in range(expected_layers) for q in range(len(self.prompts))}
        if set(self.masks) != expected or set(self.joint_pending_scores) != expected:
            raise RuntimeError('Incomplete frozen cohort; all layers and requests must be ready')
        if any(len(ids) != expected_heads for ids in self.masks.values()):
            raise RuntimeError('Unexpected KV head count in joint cohort')
        independent_identity = self.mask_identity()
        new_masks, reports = {}, []
        pending_bytes = sum(s.nbytes for s in self.joint_pending_scores.values())
        for layer in range(expected_layers):
            scores = np.stack([self.joint_pending_scores[layer,q] for q in range(len(self.prompts))])
            baseline = np.zeros_like(scores,dtype=bool)
            for q in range(len(self.prompts)):
                for h,ids in enumerate(self.masks[layer,q]):
                    baseline[q,h,(ids-self.doc_start).numpy()] = True
            _, result, report = apply_joint_exchange(scores,baseline,self.joint_config,
                document_start=self.doc_start,common_source_tokens=(self.doc_end//16)*16-self.doc_start)
            reports.append({'layer':layer,**report})
            for q in range(len(self.prompts)):
                new_masks[layer,q] = [torch.from_numpy(np.flatnonzero(result[q,h]).astype(np.int64))+self.doc_start
                                     for h in range(expected_heads)]
        new_identity = self.mask_identity(new_masks)
        # Commit only after every layer/head succeeds. The original records and
        # independent IDs remain unchanged and separately identifiable.
        self.independent_masks = self.masks
        self.masks = new_masks
        self.joint_pending_scores = {}
        self.joint_report = {'enabled':True, 'configuration':asdict(self.joint_config),
            'layers':expected_layers, 'requests':len(self.prompts), 'heads':expected_heads,
            'independent_mask_identity':independent_identity,'new_mask_identity':new_identity,
            'CPU_finalize_wall_seconds':time.perf_counter()-started,
            'CPU_finalize_process_seconds':time.process_time()-cpu_started,
            'score_transfers':self.joint_transfers,
            'score_transfer_and_owned_copy_wall_seconds':sum(r['transfer_and_owned_CPU_copy_wall_seconds'] for r in self.joint_transfers),
            'GPU_to_CPU_score_bytes':sum(r['GPU_to_CPU_bytes'] for r in self.joint_transfers),
            'peak_retained_pending_score_bytes':pending_bytes,
            'retained_independent_mask_bytes':sum(x.numel()*x.element_size() for ids in self.independent_masks.values() for x in ids),
            'layer_records':reports,
            'cost_accounting':'Score transfer/copy is already included in selector freeze time; add finalize wall once. No physical KV release or quality guarantee.'}
        return self.joint_report

    def summary(self):
        result = {'definition': 'proposal QC-SnapKV: last min32 suffix Q, full causal history denominator, head/position mean, avg_pool7, protected first4/last32, stable lower-position ties',
                'frozen_layer_question_masks': len(self.masks), 'observed_Q_records': len(self.observations),
                'observation_Q_bytes_upper_bound': sum(q.numel()*q.element_size() for _, q in self.observations.values()),
                'selector_prepare_seconds': sum(r['selector_prepare_seconds'] for r in self.records), 'records': self.records,
                'limitation': 'Selector gathers temporary full K per layer/request and is not optimized; its complete cost must be charged.'}
        if self.joint_config is not None:
            result['joint_support_exchange'] = self.joint_report or {'enabled':True,'status':'pending_complete_cohort'}
        return result
