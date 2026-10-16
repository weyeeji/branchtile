"""Diagnostic graph timings of existing owner producer/merge; never additive costs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SOURCES = ("scripts/bench_owner_stages.py", "scripts/compare_attention.py",
           "src/branchtile/attention.py", "src/branchtile/attention_owner_tasks.py",
           "src/branchtile/owner_tasks_plan.py", "src/branchtile/triton_kernel_owner_tasks.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--step", type=int, default=64)
    parser.add_argument("--chunk-tokens", type=int, required=True)
    parser.add_argument("--block-n", type=int, default=64, choices=(32, 64))
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    output = ROOT / args.output
    assert not output.exists(), "Preserve every prior result"
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}
    import torch
    from compare_attention import load_capture, graph_of, elapsed
    from branchtile.attention import make_plan, reference
    from branchtile.attention_owner_tasks import Attention
    from branchtile.triton_kernel_owner_tasks import producer, merge
    torch.set_num_threads(4)
    torch.manual_seed(20261016)
    q, k, v, owners, batch, gqa, provenance = load_capture(
        args.capture, args.layer, args.budget, args.step, 2)
    started = time.perf_counter()
    plan = make_plan(owners, batch, gqa, "cuda")
    self = Attention(plan, block_n=args.block_n, chunk_tokens=args.chunk_tokens)
    prepare_ms = (time.perf_counter() - started) * 1000
    heads, rows, dim = self.shape
    tokens = plan.indices.shape[1]

    def produce():
        if self.num_tasks:
            return producer[(self.num_tasks,)](
                q, k, v, self.plan.indices, self.plan.row_map, self.tasks,
                self.u, self.m, self.l, ROWS=rows, TOKENS=tokens,
                REGIONS=len(self.plan.labels), N=self.block_n, D=dim,
                num_warps=4, num_stages=1)

    def reduce():
        return merge[(heads, rows)](
            self.u, self.m, self.l, self.merge_indices, self.contribution_counts,
            self.output, self.lse, BATCH=self.plan.batch, GQA=self.plan.gqa,
            ROWS=rows, MAX_CONTRIB=self.layout["max_contributions"], D=dim,
            GROUP=16, num_warps=4)

    expected, expected_lse = reference(q, k, v, owners, batch, gqa)
    checks = []

    def check(phase):
        torch.cuda.synchronize()
        torch.testing.assert_close(self.output.float(), expected.float(), atol=.006, rtol=.015)
        torch.testing.assert_close(self.lse, expected_lse.float(), atol=.0003, rtol=.0001)
        checks.append(dict(phase=phase, passed=True,
                           max_abs_O=float((self.output.double() - expected).abs().max()),
                           max_abs_LSE=float((self.lse.double() - expected_lse).nan_to_num().abs().max())))

    self(q, k, v)
    check("original_complete_operator")
    produce()
    reduce()
    check("independently_called_same_two_kernels")
    device = torch.cuda.get_device_properties(0)
    flush = torch.zeros(max(2 * device.L2_cache_size, 256 * 1024 * 1024) // 4,
                        dtype=torch.float32, device="cuda")
    measurements = []
    summary = {}
    rng = random.Random(20261016)
    for cache, repetitions in (("hot", 20), ("cold", 1)):
        graphs = {name: graph_of(fn, repetitions) for name, fn in
                  (("complete", lambda: self(q, k, v)), ("producer_only", produce),
                   ("merge_only", reduce), ("complete_AA", lambda: self(q, k, v)))}
        for block in range(args.blocks):
            order = list(graphs)
            rng.shuffle(order)
            for direction, sequence in (("random", order), ("reverse", list(reversed(order)))):
                measurements.append(dict(cache=cache, block=block, direction=direction,
                                         order=sequence, us={name: elapsed(graphs[name], repetitions,
                                             flush if cache == "cold" else None) for name in sequence}))
        values = [r for r in measurements if r["cache"] == cache]
        summary[cache] = {name: statistics.median(r["us"][name] for r in values) for name in graphs}
        check("after_" + cache + "_graphs")
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    result = dict(schema="owner-stage-diagnostic-v1", status="complete", uid=os.geteuid(),
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  virtual_environment=sys.prefix, code_sha256=hashes, arguments=vars(args),
                  provenance=provenance, checks=checks, prepare_ms=prepare_ms,
                  operator_metadata=self.metadata, measurements=measurements, summary_us=summary,
                  task_lengths=[t[3] - t[2] for t in self.layout["tasks"]],
                  contribution_counts=self.layout["contribution_counts"],
                  limitations="Stage-only graphs repeat one stage against the same buffers. They change residency, dependency and scheduling relative to the complete call. Their sum or difference is not removable launch/merge cost, an overlap measurement, or a predicted fused speedup. Cold merge starts with artificially flushed partials, unlike a real producer->merge chain. No hardware traffic counters or application inference.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), checks=len(checks), summary_us=summary)))


if __name__ == "__main__":
    main()
