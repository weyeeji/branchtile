"""Frozen QC-SnapKV history masks from actual vLLM suffix-prefill observations.

No selector novelty is claimed. APC-skipped Q may only be reused when the
complete token prefix and physical page identity agree. All temporary K gathers,
score passes and stable selection belong to preparation, not free execution.
"""
import hashlib
import time

import torch
from .selector_baseline import qc_scores, select_support


class FrozenQCSelector:
    def __init__(self, prompt_tokens, document_range, budget):
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
        torch.cuda.synchronize()
        self.records.append({'layer': layer, 'question': question, 'budget': self.budget,
            'observation_positions': list(range(start, prompt_length)), 'reused_other_question_prefix_Q': reused,
            'selector_prepare_seconds': time.perf_counter()-started,
            'temporary_K_gather_bytes': k.numel()*k.element_size(),
            'mask_sha256': [hashlib.sha256(ids.numpy().tobytes()).hexdigest() for ids in logical]})

    def logical_supports(self, layer, question, query_position):
        if query_position < len(self.prompts[question]):
            raise ValueError('Frozen history applies only after dense suffix prefill')
        ids = self.masks[layer, question]
        prefix = torch.arange(self.doc_start)
        suffix = torch.arange(self.doc_end, query_position+1)
        return [torch.cat((prefix, document, suffix)) for document in ids]

    def summary(self):
        return {'definition': 'proposal QC-SnapKV: last min32 suffix Q, full causal history denominator, head/position mean, avg_pool7, protected first4/last32, stable lower-position ties',
                'frozen_layer_question_masks': len(self.masks), 'observed_Q_records': len(self.observations),
                'observation_Q_bytes_upper_bound': sum(q.numel()*q.element_size() for _, q in self.observations.values()),
                'selector_prepare_seconds': sum(r['selector_prepare_seconds'] for r in self.records), 'records': self.records,
                'limitation': 'Selector gathers temporary full K per layer/request and is not optimized; its complete cost must be charged.'}
