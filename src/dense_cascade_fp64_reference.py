"""Chunked FP64 full-history mathematical reference over original fused KV pages."""
import torch


@torch.inference_mode()
def full_history_fp64(query, cache, block_tables, seq_lens, k_scale=1., v_scale=1., chunk=4096):
    assert query.ndim == 3 and cache.ndim == 4
    batch, qheads, dim = query.shape
    _, kvheads, page, fused = cache.shape
    assert dim*2 == fused and qheads % kvheads == 0
    assert len(seq_lens) == batch and chunk > 0
    q = query.double().reshape(batch, kvheads, qheads//kvheads, dim)
    outputs = []
    for row, length in enumerate(seq_lens):
        numerator = torch.zeros_like(q[row])
        denominator = torch.zeros(q[row].shape[:-1],device=query.device,dtype=torch.float64)
        maximum = torch.full_like(denominator,-torch.inf)
        for start in range(0,length,chunk):
            logical = torch.arange(start,min(start+chunk,length),device=query.device)
            physical = block_tables[row,logical//page].long()
            # Gather bytes before reinterpretation for float8 indexing support.
            source = cache.view(torch.uint8) if cache.dtype == torch.float8_e4m3fn else cache
            payload = source[physical,:,logical%page,:]
            if payload.dtype == torch.uint8:
                payload = payload.view(torch.float8_e4m3fn)
            payload = payload.double().permute(1,0,2)
            key, value = payload[...,:dim]*k_scale, payload[...,dim:]*v_scale
            scores = torch.einsum('hgd,htd->hgt',q[row],key)*(dim**-.5)
            updated = torch.maximum(maximum,scores.amax(-1))
            rescale = torch.exp(maximum-updated)
            probs = torch.exp(scores-updated[...,None])
            numerator = numerator*rescale[...,None]+torch.einsum('hgt,htd->hgd',probs,value)
            denominator = denominator*rescale+probs.sum(-1)
            maximum = updated
        outputs.append((numerator/denominator[...,None]).reshape(qheads,dim))
    result = torch.stack(outputs)
    assert bool(torch.isfinite(result).all())
    return result


def compare_to_reference(value, reference, atol=.006, rtol=.015):
    error = value.double()-reference
    limit = atol+rtol*reference.abs()
    passing = error.abs() <= limit
    return {'passed':bool(passing.all()), 'failed_values':int((~passing).sum()),
            'max_abs':float(error.abs().max()),
            'relative_l2':float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(reference).clamp_min(1e-30)),
            'max_normalized_tolerance':float((error.abs()/limit).max())}
