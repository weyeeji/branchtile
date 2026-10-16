"""CPU-only P-cast screen; no tensor-core or performance claim."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]

def cast(x, residual_scale):
    hi = (x * 256).to(torch.float8_e4m3fn).float() / 256
    lo = ((x - hi) / residual_scale).to(torch.float8_e4m3fn).float() * residual_scale
    return hi + lo

def error(x, ref):
    d = x.double() - ref.double()
    return {'finite': bool(torch.isfinite(x).all()), 'max_abs': float(d.abs().max()),
            'relative_l2': float(d.norm() / ref.double().norm().clamp_min(1e-30))}

def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    torch.set_num_threads(2)
    vals = torch.arange(0x3f81, dtype=torch.int16).view(torch.bfloat16).float()
    enumeration = []
    for label, scale in [('same_scale', 2.**-8), ('unsafe_residual_2^-14', 2.**-14), ('residual_2^-13', 2.**-13)]:
        rec = cast(vals, scale)
        bad = rec != vals
        enumeration.append({'method': label, 'values': len(vals), 'nonfinite': int((~torch.isfinite(rec)).sum()),
                            'nonexact': int(bad.sum()), 'residual_outside_E4M3_range': int((((vals-(vals*256).to(torch.float8_e4m3fn).float()/256)/scale).abs()>448).sum()), 'largest_nonexact_input': float(vals[bad].max()) if bad.any() else 0,
                            'max_abs_finite_error': float((rec[torch.isfinite(rec)] - vals[torch.isfinite(rec)]).abs().max())})
    rows = []
    for doc in (1, 7):
        path = ROOT / f'results/dense_capture_b8_train{doc}_fp8_v2/rank0/dense_capture/layer15.pt'
        data = torch.load(path, map_location='cpu', weights_only=True)
        q = data['q'].double().reshape(8, 2, 8, 128).permute(1, 0, 2, 3).reshape(2, 64, 128)
        kv = data['prefix_kv'].view(torch.float8_e4m3fn).double()
        tokens = kv.shape[1]; padded = math.ceil(tokens / 1024) * 1024
        for head in (0, 1):
            k, v = kv[0, :, head], kv[1, :, head]
            s = q[head] @ k.T / math.sqrt(128)
            sm = torch.nn.functional.pad(s, (0, padded-tokens), value=-float('inf')).reshape(64, -1, 16, 64)
            running_max = sm.amax(-1).cummax(-1).values
            running_max = torch.where(torch.isfinite(running_max), running_max, torch.zeros_like(running_max))
            local_p = (sm - running_max[..., None]).exp()
            pb = local_p.to(torch.bfloat16).float()
            global_max = s.amax(-1)
            final_weights = (running_max - global_max[:, None, None]).exp()[..., None]
            denominator = (s-global_max[:, None]).exp().sum(-1)
            exact = ((s-global_max[:, None]).exp() @ v) / denominator[:, None]
            def output(p):
                full = (p.double()*final_weights).reshape(64, -1)[:, :tokens]
                return (full @ v) / denominator[:, None]
            base = output(pb)
            row = {'document': doc, 'rank': 0, 'layer': 15, 'kv_head': head, 'tokens': tokens,
                   'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'BF16_P_vs_FP64': error(base, exact), 'variants': []}
            valid = torch.arange(padded).reshape(1, -1, 16, 64) < tokens
            for name, rec in [('single_FP8_256', (pb*256).to(torch.float8_e4m3fn).float()/256),
                              ('same_scale_components', cast(pb, 2.**-8)),
                              ('fixed_residual_components', cast(pb, 2.**-13))]:
                oo = output(rec)
                row['variants'].append({'method': name, 'nonexact_P': int(((rec != pb)&valid).sum()),
                    'zeroed_nonzero_P': int(((rec == 0)&(pb > 0)&valid).sum()),
                    'P_max_abs': float((rec-pb).abs().max()),
                    'O_vs_BF16_P': error(oo, base), 'O_vs_FP64': error(oo, exact)})
            rows.append(row)
    assert not torch.cuda.is_initialized()
    result = {'status': 'CPU_probability_representation_screen', 'cuda_initialized': False,
              'enumeration': enumeration, 'real_capture_rows': rows,
              'limitations': ['Only two documents, rank0, layer15, one actual decode state.',
               'FP64 QK and analytic online rescaling isolate P representation; GPU exponential, MMA accumulation and register allocation are not modeled.',
               'No end-to-end or performance result.']}
    out = ROOT/'results/platform/probability_components_cpu_20261017.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'path': str(out), 'enumeration': enumeration, 'rows': len(rows)}))
if __name__ == '__main__':
    main()
