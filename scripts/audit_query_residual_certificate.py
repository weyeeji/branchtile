"""CPU-only arithmetic audit; not a GPU implementation or a model quality test."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/platform/query_residual_certificate_20261016.json"


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def determinant(a):
    a = [row[:] for row in a]
    result = Fraction(1)
    for col in range(len(a)):
        pivot = next(i for i in range(col, len(a)) if a[i][col])
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            result = -result
        value = a[col][col]
        result *= value
        for row in range(col + 1, len(a)):
            scale = a[row][col] / value
            for j in range(col, len(a)):
                a[row][j] -= scale * a[col][j]
    return result


def bound_sanity():
    """Floating-point sanity only; the real-arithmetic proof is in the note."""
    rng = random.Random(20261016)
    peak_ratio, peak_mass_excess = 0.0, 0.0
    for _ in range(3000):
        n, d, dv = 7, 3, 2
        keys = [[rng.uniform(-2, 2) for _ in range(d)] for _ in range(n)]
        values = [[rng.uniform(-2, 2) for _ in range(dv)] for _ in range(n)]
        q0 = [rng.uniform(-1, 1) for _ in range(d)]
        h = [rng.uniform(-0.3, 0.3) for _ in range(d)]
        scale = math.sqrt(d)
        weights = [math.exp(dot(q0, k) / scale) for k in keys]
        z0 = sum(weights)
        p = [w / z0 for w in weights]
        mu = [sum(p[i] * keys[i][j] for i in range(n)) for j in range(d)]
        y = [sum(p[i] * values[i][j] for i in range(n)) for j in range(dv)]
        jmat = [[sum(p[i] * (values[i][a] - y[a]) * (keys[i][b] - mu[b])
                     for i in range(n)) / scale for b in range(d)] for a in range(dv)]
        # A rank-at-most-one approximation; Frobenius residual is an upper bound
        # on spectral residual, requiring no numerical SVD library.
        jr = [[row[0], 0.0, 0.0] for row in jmat]
        lam = norm([jmat[a][b] - jr[a][b] for a in range(dv) for b in range(d)])
        rk = max(norm([k[j] - mu[j] for j in range(d)]) for k in keys)
        rv = max(norm([v[j] - y[j] for j in range(dv)]) for v in values)
        rho = norm(h) * rk / scale
        remainder = math.expm1(rho) - rho
        update = [dot(row, h) for row in jr]
        approx = [y[j] + update[j] for j in range(dv)]
        bound = lam * norm(h) + (rv + norm(update) + lam * norm(h)) * remainder
        q = [q0[j] + h[j] for j in range(d)]
        current = [math.exp(dot(q, k) / scale) for k in keys]
        z = sum(current)
        exact = [sum(current[i] * values[i][j] for i in range(n)) / z for j in range(dv)]
        error = norm([exact[j] - approx[j] for j in range(dv)])
        mass_ratio = z / (z0 * math.exp(dot(h, mu) / scale))
        assert error <= bound + 1e-12
        assert 1 - 1e-12 <= mass_ratio <= 1 + remainder + 1e-12
        # Merge with an independently exact private-tail mass/output.
        zp = math.exp(rng.uniform(-2, 2))
        yp = [rng.uniform(-2, 2) for _ in range(dv)]
        zhat = z0 * math.exp(dot(h, mu) / scale)
        a = zhat / (zhat + zp)
        delta_a = remainder * a * (1 - a) / (1 + remainder * a)
        merged_bound = (a + delta_a) * bound + delta_a * norm([approx[j] - yp[j] for j in range(dv)])
        merged_exact = [(z * exact[j] + zp * yp[j]) / (z + zp) for j in range(dv)]
        merged_approx = [a * approx[j] + (1 - a) * yp[j] for j in range(dv)]
        assert norm([merged_exact[j] - merged_approx[j] for j in range(dv)]) <= merged_bound + 1e-12
        peak_ratio = max(peak_ratio, error / bound)
        peak_mass_excess = max(peak_mass_excess, mass_ratio - 1)
    return {"cases": 3000, "seed": 20261016, "failed": 0,
            "max_observed_error_over_bound": peak_ratio,
            "max_mass_ratio_minus_one": peak_mass_excess,
            "scope": "Synthetic float64 sanity, not a finite-precision certificate, natural hit rate, or performance result."}


def build():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    model_path = ROOT / "data/models/Qwen3-30B-A3B-Thinking-2507-FP8/config.json"
    model = json.loads(model_path.read_text())
    layers, heads, d = model["num_hidden_layers"], model["num_attention_heads"] // 2, model["head_dim"]
    assert (layers, heads, d) == (48, 16, 128)
    b, rank = 3, 8
    captures = {}
    for name in ("docfinqa256", "publicdev1_256", "publicdev6_256"):
        relative = f"results/captures/main30b_tp2_{name}_suffix_v2/rank0"
        files = sorted((ROOT / relative).glob("decode_q*_l*_s*.pt"))
        steps = sorted({int(re.fullmatch(r"decode_q\d+_l\d+_s(\d+)\.pt", f.name)[1]) for f in files})
        assert steps == [1, 16, 64, 256]
        captures[relative] = {"files": len(files), "prediction_steps": steps,
                              "filename_list_sha256": hashlib.sha256("\n".join(f.name for f in files).encode()).hexdigest(),
                              "scope": "Directory names only; tensor contents and answers not read."}
    matrix = [[Fraction(x ** j, sum(x ** k for k in range(4))) for j in range(4)] for x in range(1, 5)]
    det = determinant(matrix)
    assert det == Fraction(1, 17000)
    return {
        "date": "2026-10-16", "decision": "no_qualified_new_candidate_no_gpu",
        "model_config_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "cpu_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rank_one_logit_counterexample": {"q": "log(x), x=1..4", "k": "0,1,2,3", "softmax_determinant_exact": str(det)},
        "cosine_only_counterexample": {"q0": [1, 0], "q1": [1, 0.001], "keys": [[0, 10000], [0, -10000]], "values": [1, -1],
                                       "cosine": 1 / math.sqrt(1 + 0.001 ** 2), "output_absolute_change": math.tanh(10 / math.sqrt(2)),
                                       "scope": "Mathematical counterexample, not observed model norms."},
        "storage_designs_bytes_per_rank": {
            "assumptions": {"B": b, "query_heads": heads, "layers": layers, "d": d, "rank": rank, "element_bytes": 4},
            "full_jacobian": b * heads * layers * d * d * 4,
            "rank8_two_unshared_factors": b * heads * layers * 2 * d * rank * 4,
            "rank8_left_factors_only_excludes_shared_right": b * heads * layers * d * rank * 4,
            "rank8_two_factors_per_128token_block_at_32k": b * heads * layers * 2 * d * rank * 4 * 256,
            "scope": "Example designs, not an implementation or a storage lower bound."},
        "zero_overhead_acceptance_to_exceed_1p15": 1 - 1 / 1.15,
        "capture_inventory": captures, "bound_sanity": bound_sanity(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build()
    if args.check:
        assert json.loads(OUTPUT.read_text()) == result
        print("CPU residual audit matches saved result; all arithmetic sanity checks pass.")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(OUTPUT.relative_to(ROOT))
