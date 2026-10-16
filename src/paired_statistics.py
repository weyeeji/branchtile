"""Paired cluster bootstrap with seed collapse and a degenerate-CI guard.

Scores are bounded in [0, 1]. The caller supplies frozen cluster identities and
complete paired rows, including failures scored under the frozen task metric.
This module never establishes dataset independence or factual correctness.
The seed field labels experiment replicates, not resolved per-question RNG seeds.
"""
from collections import defaultdict
import math

import numpy as np


def paired_cluster_interval(rows, *, expected_seeds, resamples=10000, seed=20261016,
                            confidence=.95, margin=.02, formal=False):
    if resamples < 10000:
        raise ValueError("At least 10000 cluster resamples are required")
    expected_seeds = set(expected_seeds)
    if not expected_seeds or not 0 < confidence < 1 or not 0 < margin < 1:
        raise ValueError("Invalid seeds, confidence or margin")
    if formal and len(expected_seeds) < 3:
        raise ValueError("Formal quality requires at least three paired seeds")
    questions = defaultdict(dict)
    identities = {}
    for row in rows:
        question = (row["document_id"], row["question_id"])
        cluster = row["cluster_id"]
        if not cluster or row["seed"] not in expected_seeds:
            raise ValueError("Missing cluster or unexpected seed")
        if question in identities and identities[question] != cluster:
            raise ValueError("Question changes cluster across seeds")
        identities[question] = cluster
        if row["seed"] in questions[question]:
            raise ValueError("Duplicate question/seed pair")
        reference, candidate = float(row["reference"]), float(row["candidate"])
        if not (0 <= reference <= 1 and 0 <= candidate <= 1):
            raise ValueError("Scores must be finite and in [0,1]")
        questions[question][row["seed"]] = (reference, candidate)
    if not questions:
        raise ValueError("No paired questions")
    clusters = defaultdict(list)
    for question, values in questions.items():
        if set(values) != expected_seeds:
            raise ValueError("Incomplete paired seed denominator")
        # Average within each question first. Replicates do not create tasks.
        reference, candidate = np.mean(list(values.values()), axis=0)
        clusters[identities[question]].append((reference, candidate))
    cluster_ids = sorted(clusters)
    if len(cluster_ids) < 2:
        raise ValueError("At least two independent clusters are required")
    counts = np.array([len(clusters[k]) for k in cluster_ids], dtype=np.int64)
    sums = np.array([np.sum(clusters[k], axis=0) for k in cluster_ids])
    deltas = sums[:, 1] - sums[:, 0]
    means = deltas / counts
    weights = counts / counts.sum()
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(cluster_ids), size=(resamples, len(cluster_ids)))
    micro_draws = deltas[selections].sum(axis=1) / counts[selections].sum(axis=1)
    macro_draws = means[selections].mean(axis=1)
    tail = (1 - confidence) / 2
    all_pairs_equal = all(ref == cand for values in questions.values() for ref, cand in values.values())
    any_loss_by_cluster = [any(cand < ref for ref, cand in clusters[k]) for k in cluster_ids]
    result = {
        "scope": "formal interval component; requires external frozen-protocol and identity audit" if formal else "development diagnostic; never a formal noninferiority claim",
        "score_unit": "fraction; multiply by 100 for percentage points",
        "method": "paired percentile cluster bootstrap; same draws for both methods; seeds averaged within each question before resampling whole clusters",
        "seed": seed, "resamples": resamples, "confidence": confidence,
        "interval_tails": [tail, 1 - tail], "noninferiority_margin": margin,
        "paired_seeds": sorted(expected_seeds), "questions": len(questions),
        "clusters": len(cluster_ids), "cluster_ids": cluster_ids,
        "questions_per_cluster": counts.tolist(), "all_raw_pairs_equal": all_pairs_equal,
        "assumptions": ["Supplied clusters are independent; this function cannot verify that assumption.",
                        "All frozen questions, seeds, parse failures and service failures must be supplied by the caller.",
                        "Confidence intervals do not correct post-hoc method/budget selection or establish factual quality.",
                        "Unequal-cluster micro intervals target question-weighted performance; equal-cluster intervals target a different estimand."],
    }
    for name, estimate, draws, w in [
        ("question_micro", float(deltas.sum() / counts.sum()), micro_draws, weights),
        ("equal_cluster", float(means.mean()), macro_draws, np.full(len(cluster_ids), 1 / len(cluster_ids))),
    ]:
        low, high = np.quantile(draws, [tail, 1 - tail])
        # A cluster-mean paired difference is in [-1,1]. For fixed weights,
        # Hoeffding gives P(E[D] < observed - t) <= exp(-t²/(2 sum w²)).
        # This bound is deliberately conservative and needs independence, not
        # a nonzero empirical variance. It prevents a zero-width bootstrap
        # from certifying equivalence on a finite collection of equal pairs.
        conservative_lower = max(-1., estimate - math.sqrt(2 * math.log(1 / tail) * float(np.sum(w * w))))
        degenerate = bool(np.ptp(draws) <= 1e-15)
        decision_lower = min(float(low), conservative_lower) if degenerate else float(low)
        result[name] = {
            "paired_difference": estimate, "percentile_interval": [float(low), float(high)],
            "bootstrap_degenerate": degenerate,
            "fixed_weight_hoeffding_lower": conservative_lower,
            "decision_lower_after_degenerate_guard": decision_lower,
            "interval_above_margin": decision_lower > -margin,
            "formal_noninferiority_supported_by_interval_only": bool(formal and decision_lower > -margin),
        }
    result["no_observed_cluster_loss_sensitivity"] = {
        "clusters_with_any_seed_averaged_question_loss": sum(any_loss_by_cluster),
        "iid_cluster_event_loss_probability_upper_if_zero": 1 - tail ** (1 / len(cluster_ids)) if not any(any_loss_by_cluster) else None,
        "minimum_iid_zero_loss_clusters_to_bound_event_probability_below_margin": math.floor(math.log(tail) / math.log(1 - margin)) + 1,
        "limit": "Exact zero-event Bernoulli sensitivity assumes iid cluster loss events; bounds an event probability, not the question-weighted effect. It is not substituted for the primary interval.",
    }
    return result
