"""Export actual TP-rank captures to selector and same-event replay formats."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import torch
from capture_model import qc_scores, select_support, opportunity, dump_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    torch.set_num_threads(4)
    source_dir = Path(a.input) / f"rank{a.rank}"
    out = Path(a.output) / f"rank{a.rank}"
    out.mkdir(parents=True, exist_ok=True)
    raw = json.loads((source_dir / "metadata.json").read_text())
    assert raw["status"] == "completed" and raw["rank"] == a.rank
    entries = {e["file"]:e for e in raw["files"]}
    def load(name):
        path = source_dir/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entries[name]["sha256"]
        return torch.load(path, map_location="cpu", weights_only=True)
    requests = raw["requests"]
    cold = next(r for r,i in requests.items() if i["phase"] == "cold_singleton")
    warm = sorted((r for r,i in requests.items() if i["phase"] == "apc_warm_natural_group"),
                  key=lambda r: requests[r]["question_index"])
    prompts = {r:load(f"request_{r}_prompt.pt")["prompt_token_ids"] for r in requests}
    ds,de = raw["config"]["document_token_range"]
    budgets = [4096,8192,16384,32768]
    summary = {"rank":a.rank,"raw_metadata_sha256":hashlib.sha256((source_dir/'metadata.json').read_bytes()).hexdigest(),
               "source":str(source_dir),"output":str(out),"observation_reconstruction":[],"ready_groups":[],
               "selector_scores_device":a.device,"all_hooked_kv_matches_actual_cache":True,
               "all_shared_full_prefix_pages_identical":True,"selector_seconds":[],
               "exporter_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "snapshot_checks":0}
    for name in entries:
        if "_step" in name or "_ready_anchor" in name:
            snapshot = load(name)
            assert snapshot["metadata"]["hooked_kv_matches_actual_cache"]
            assert snapshot["metadata"]["physical_full_prefix_pages_match"]
            summary["snapshot_checks"] += 1
    opportunities = []
    for layer in raw["config"]["layers"]:
        source = load(f"layer{layer}_shared_source.pt")
        se = source["shared_prefix_end"]
        tail = de-se
        assert tail >= 0 and tail < 16 and source["k"].shape == (2,se,128)
        torch.save({"prefix_k":source["k"][None,:,:ds],"prefix_v":source["v"][None,:,:ds],
                    "document_k":source["k"][None,:,ds:],"document_v":source["v"][None,:,ds:]},out/f"source_l{layer}.pt")
        observation = {}
        for r in requests:
            observation[r] = {}
            for name in entries:
                if name.startswith(f"layer{layer}_request{r}_observation_"):
                    x = load(name)
                    for i,pos in enumerate(x["positions"].tolist()):
                        assert pos not in observation[r]
                        observation[r][pos] = (x["q"][i], x["prompt_full_page_ids"], name)
        masks_by_budget = {b:[] for b in budgets}
        for qi,r in enumerate(warm):
            initial = load(f"layer{layer}_request{r}_step1.pt")
            pl = requests[r]["prompt_length"]
            positions = torch.arange(max(de,pl-32),pl)
            assert len(positions) == min(32, pl-de) and len(positions) > 0
            qs, reused = [], []
            for pos in positions.tolist():
                if pos in observation[r]:
                    qs.append(observation[r][pos][0])
                else:
                    page = pos//16
                    # A later request can also reuse another warm request's
                    # identical question prefix. Only an actually observed Q
                    # with identical token prefix AND physical page qualifies.
                    candidates = []
                    for origin_request in [cold, *warm]:
                        if origin_request == r or pos not in observation[origin_request]:
                            continue
                        value,pages,origin = observation[origin_request][pos]
                        if pos >= len(prompts[origin_request]):
                            continue
                        if not torch.equal(prompts[r][:pos+1], prompts[origin_request][:pos+1]):
                            continue
                        if page >= len(initial["prompt_full_page_ids"]) or page >= len(pages):
                            continue
                        if int(initial["prompt_full_page_ids"][page]) != int(pages[page]):
                            continue
                        candidates.append((origin_request, value, pages, origin))
                    assert candidates, "No observed Q with identical token prefix and cached physical page"
                    origin_request,value,pages,origin = candidates[0]
                    qs.append(value)
                    reused.append({"position":pos,"origin":origin,"origin_request_id":origin_request,
                                   "identical_prefix_through_position":True,
                                   "physical_page":int(pages[page])})
            observation_q = torch.stack(qs).permute(1,0,2).contiguous()[None]
            k = torch.cat([source["k"],initial["private_k"]],dim=1)[None]
            assert k.shape[2] == pl
            start = time.monotonic()
            scores = qc_scores(observation_q.to(a.device), k.to(a.device), positions.to(a.device),ds,de)
            supports = {budget:select_support(scores,budget).cpu() for budget in budgets}
            if a.device == "cuda": torch.cuda.synchronize()
            summary["selector_seconds"].append({"layer":layer,"question":qi,"seconds":time.monotonic()-start})
            torch.save({"observation_q":observation_q,"observation_positions":positions,
                        "pooled_scores":scores.cpu(),"supports":supports,
                        "observation_is_question_suffix_only":True,
                        "source_request_id":r,"reused_observation_positions_with_proof":reused},out/f"selector_q{qi}_l{layer}.pt")
            for budget in budgets:
                # Recent32 protects these actual request-private document tokens.
                assert tail == 0 or bool(supports[budget][:,-tail:].all())
                masks_by_budget[budget].append(supports[budget][:,:se-ds])
            torch.save({"query":initial["q"].reshape(1,16,1,128),
                        "private_suffix_k":initial["private_k"][None],"private_suffix_v":initial["private_v"][None],
                        "metadata":initial["metadata"]},out/f"decode_q{qi}_l{layer}_s1.pt")
            summary["observation_reconstruction"].append({"layer":layer,"question":qi,
                "reused_cached_Q_count":len(reused),"exact_token_prefix_and_physical_page_verified":True})
        for budget,supports in masks_by_budget.items():
            supports = torch.stack(supports)
            torch.save({"supports":supports,"budget":budget,"layer":layer,"gqa":8,
                        "private_protected_document_tail_tokens":tail},out/f"masks_l{layer}_b{budget}.pt")
            opportunities.extend({"rank":a.rank,"layer":layer,"budget":budget,**row}
                                 for row in opportunity(supports,8))
        for anchor in (16,64,256):
            names = [f"layer{layer}_ready_anchor{anchor}_request{r}.pt" for r in warm]
            if not all(name in entries for name in names):
                continue
            decoded = [load(name) for name in names]
            events = [x["metadata"]["same_event_query_positions"] for x in decoded]
            assert all(event == events[0] for event in events)
            for qi,x in enumerate(decoded):
                assert x["metadata"]["hooked_kv_matches_actual_cache"]
                assert x["metadata"]["physical_full_prefix_pages_match"]
                torch.save({"query":x["q"].reshape(1,16,1,128),
                            "private_suffix_k":x["private_k"][None],"private_suffix_v":x["private_v"][None],
                            "metadata":x["metadata"]},out/f"decode_q{qi}_l{layer}_s{anchor}.pt")
            summary["ready_groups"].append({"layer":layer,"anchor":anchor,"B":len(warm),
                "query_positions":events[0],"actual_prediction_steps":[x["metadata"]["prediction_step"] for x in decoded]})
    meta = {"status":"completed","B":len(warm),"gqa":8,"kv_heads":2,"tensor_parallel_size":2,
            "export_semantics_version":2,
            "observation_window":"Last min(32, prompt_length-document_end) actual suffix Q; no document Q backfill",
            "actual_tp_rank":a.rank,"model_path":"data/models/Qwen3-30B-A3B-Thinking-2507-FP8",
            "model_revision":raw["config"]["model_revision"],"document_sha256":raw["config"]["document_sha256"],
            "document_token_range":[ds,de],"actual_shared_document_token_range":[ds,se],
            "private_protected_document_tail_tokens":tail,"budgets":budgets,"layers":raw["config"]["layers"],
            "ready_lifetime_measured_by_step":{"1":False, **{str(s):sum(x["anchor"]==s for x in summary["ready_groups"])==len(raw["config"]["layers"]) for s in (16,64,256)}},
            "decode_step_definition":"s1 is each actual suffix-prefill query predicting first generated token (selector compatibility only); s16/s64/s256 are same-forward ready-group snapshots anchored on question0 prediction step, with each request's own actual position retained",
            "mask_definition":"Budget applies to the full original document; protected partial-page document tail resides in private suffix and is omitted from common-source masks only. All original selected tokens remain in execution.",
            "scope":"Actual official FP8 TP2 dense-trajectory instrumentation; no sparse quality or application speedup claim",
            "raw_rank_metadata":str(source_dir/'metadata.json'),"summary":"export_summary.json"}
    dump_json(out/'metadata.json',meta)
    dump_json(out/'export_summary.json',summary)
    dump_json(out/'opportunity.json',opportunities)
    print(json.dumps({"rank":a.rank,"status":"completed","ready_groups":len(summary['ready_groups']),"output":str(out)}),flush=True)


if __name__ == '__main__':
    main()
