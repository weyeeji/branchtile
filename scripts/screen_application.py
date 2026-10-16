"""Natural-EOS development screening on an actual vLLM engine.

This is an application-baseline development run, never a sparse candidate or
formal quality/noninferiority result. Every frozen document/question remains
in the denominator. Raw answers are persisted without looking at gold answers.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def extract_final(text: str, thinking: bool):
    """Do not score unfinished reasoning as a final answer."""
    if "</think>" in text:
        reasoning, final = text.rsplit("</think>", 1)
        return final.strip(), reasoning, "closed_thinking"
    if thinking:
        return "", text, "missing_thinking_end"
    return text.strip(), "", "nonthinking"


def parse_joint(final: str, count: int):
    """One exact JSON object; no nearest-gold extraction or partial rescue."""
    cleaned = final.strip()
    if cleaned.startswith("```json\n") and cleaned.endswith("\n```"):
        cleaned = cleaned[8:-4].strip()
    elif cleaned.startswith("```\n") and cleaned.endswith("\n```"):
        cleaned = cleaned[4:-4].strip()
    try:
        pairs = json.loads(cleaned, object_pairs_hook=lambda x: x)
    except (ValueError, TypeError):
        return [""] * count, "invalid_json"
    if not isinstance(pairs, list) or not all(isinstance(x, tuple) and len(x) == 2 for x in pairs):
        return [""] * count, "not_object"
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        return [""] * count, "duplicate_ids"
    obj = dict(pairs)
    expected = {f"q{i}" for i in range(count)}
    if set(obj) != expected or any(not isinstance(obj[k], str) or not obj[k].strip() for k in expected):
        return ["" for _ in range(count)], "missing_extra_or_empty_answer"
    return [obj[f"q{i}"].strip() for i in range(count)], "ok"


def make_joint(tokenizer, group):
    text = (ROOT / group["document_path"]).read_text()
    assert digest(text.encode()) == group["document_sha256"]
    instruction = ("Answer all questions using the meeting transcript. Give concise factual summaries "
                   "that address each question. Return the final answers as one JSON object with "
                   "exactly the question IDs as keys and answer strings as values. Include every question.")
    questions = "\n".join(f'q{i}: {q["question"]}' for i, q in enumerate(group["questions"]))
    content = "Document:\n" + text + "\n\nQuestions:\n" + questions
    encoded = tokenizer.apply_chat_template(
        [{"role": "system", "content": instruction}, {"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    return encoded["input_ids"] if hasattr(encoded, "keys") else encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default="data/manifests/qmsum_train_exploratory8.json")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--max-tokens-per-question", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--arms", default="dense,joint")
    parser.add_argument("--warm-repeats", type=int, default=0)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix

    import torch
    import pynvml as nv
    import vllm
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from capture_model import make_inputs

    torch.set_num_threads(4)
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text())
    group_indices = [int(x) for x in args.groups.split(",")]
    arms = args.arms.split(",")
    assert set(arms) <= {"dense", "joint"}
    assert all(manifest["groups"][i]["dataset"] == "QMSum" for i in group_indices)
    destination = ROOT / args.output
    nv.nvmlInit()
    handles = [nv.nvmlDeviceGetHandleByIndex(int(i)) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]

    def board_state():
        result = []
        for handle in handles:
            try:
                energy = nv.nvmlDeviceGetTotalEnergyConsumption(handle)
            except nv.NVMLError:
                energy = None
            result.append({"uuid": nv.nvmlDeviceGetUUID(handle), "energy_mj": energy,
                           "temperature_c": nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU),
                           "memory_used_bytes": nv.nvmlDeviceGetMemoryInfo(handle).used})
        return result

    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    report = {"schema": 1, "status": "initializing", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "scope": "QMSum frozen train development; natural generation application baseline screening; not formal quality/noninferiority or candidate speedup",
              "pid": os.getpid(), "uid": os.geteuid(), "python": sys.executable,
              "model": args.model, "torch": torch.__version__, "vllm": vllm.__version__,
              "manifest": args.manifest, "manifest_sha256": digest(manifest_path.read_bytes()),
              "script_sha256": digest(Path(__file__).read_bytes()), "config": vars(args),
              "sampling": {"temperature": .6, "top_p": .95, "top_k": 20, "min_p": 0,
                           "min_tokens": 0, "ignore_eos": False, "seed_rule": "seed + group_index*100 + question_index; joint uses group seed"},
              "engine": {"tp": 1, "dtype": "bfloat16", "kv_dtype": "auto (BF16)", "apc": True,
                         "enforce_eager": True, "max_num_seqs": 4, "gpu_memory_utilization": .80,
                         "max_num_batched_tokens": 4096, "max_model_len": args.max_model_len},
              "limitations": ["Single seed development screening; no quality conclusions before scoring and blind factual review.",
                              "Cold arms reset prefix cache. Warm repeats are explicitly labeled, never used as A/A.",
                              "First execution can include additional backend JIT; initialization and warmup are separately recorded.",
                              "Step output co-emission is an observer proxy, not exact attention ready/scheduler membership.",
                              "Board energy is gross and uncorrected; concurrent neighboring jobs prevent a formal energy comparison."],
              "board_before_initialization": board_state(), "runs": []}
    write_json(destination, report)
    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=args.max_model_len, max_num_seqs=4,
              max_num_batched_tokens=4096, enable_prefix_caching=True,
              enforce_eager=True, gpu_memory_utilization=.80,
              disable_custom_all_reduce=True, disable_log_stats=False, seed=args.seed)
    report["initialization_seconds"] = time.monotonic() - start
    warm_start = time.monotonic()
    llm.generate([{"prompt_token_ids": [1, 2, 3, 4]}] * 4,
                 SamplingParams(temperature=0, max_tokens=2), use_tqdm=False)
    report["warmup_seconds"] = time.monotonic() - warm_start
    report["status"] = "running"
    write_json(destination, report)

    for group_index in group_indices:
        group = manifest["groups"][group_index]
        prepare_start = time.monotonic()
        independent, doc_start, doc_end = make_inputs(tokenizer, group)
        joint = make_joint(tokenizer, group)
        tokenization_seconds = time.monotonic() - prepare_start
        b = len(independent)
        order = arms.copy()
        random.Random(args.seed + group_index).shuffle(order)
        for arm in order:
            for repeat in range(args.warm_repeats + 1):
                ids = independent if arm == "dense" else [joint]
                cap = args.max_tokens_per_question * (1 if arm == "dense" else b)
                record = {"group_index": group_index, "document_id": group["document_id"],
                          "document_sha256": group["document_sha256"], "question_ids": [q["question_sha256"] for q in group["questions"]],
                          "B": b, "arm": arm, "repeat": repeat, "cache": "cold" if repeat == 0 else "warm",
                          "prompt_tokens": list(map(len, ids)), "prompt_sha256": [digest(json.dumps(x).encode()) for x in ids],
                          "document_token_range_independent": [doc_start, doc_end],
                          "max_tokens_per_request": cap, "tokenization_seconds_shared_preparation": tokenization_seconds,
                          "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
                if any(len(x) + cap > args.max_model_len for x in ids):
                    record.update(status="unserviceable_context", answers=[""] * b)
                    report["runs"].append(record)
                    write_json(destination, report)
                    continue
                if repeat == 0:
                    assert llm.reset_prefix_cache(), "Cold-cache reset failed"
                sampling = [SamplingParams(temperature=.6, top_p=.95, top_k=20, min_p=0,
                                           max_tokens=cap, min_tokens=0, ignore_eos=False,
                                           seed=args.seed + group_index * 100 + i) for i in range(len(ids))]
                observations = []
                original_step = llm.llm_engine.step
                t0 = time.monotonic()

                def observed_step():
                    outputs = original_step()
                    observations.append({"elapsed_seconds": time.monotonic() - t0,
                                         "requests": [{"request_id": out.request_id, "finished": out.finished,
                                                       "generated_tokens": len(out.outputs[0].token_ids)}
                                                      for out in outputs if out.outputs]})
                    return outputs

                llm.llm_engine.step = observed_step
                record["board_before"] = board_state()
                try:
                    outputs = llm.generate([{"prompt_token_ids": x} for x in ids], sampling, use_tqdm=False)
                    record["group_jct_seconds"] = time.monotonic() - t0
                    record["status"] = "complete"
                except Exception as exc:
                    record.update(status="failed", error=repr(exc), group_jct_seconds=time.monotonic() - t0)
                    report["runs"].append(record)
                    write_json(destination, report)
                    raise
                finally:
                    llm.llm_engine.step = original_step
                    record["board_after"] = board_state()
                record["board_energy_joules"] = [
                    (after["energy_mj"] - before["energy_mj"]) / 1000
                    if after["energy_mj"] is not None and before["energy_mj"] is not None else None
                    for before, after in zip(record["board_before"], record["board_after"])]
                record["requests"] = []
                finals = []
                for out in outputs:
                    answer = out.outputs[0]
                    final, reasoning, parsing = extract_final(answer.text, args.thinking)
                    finals.append(final)
                    record["requests"].append({"request_id": out.request_id,
                        "generated_tokens": len(answer.token_ids), "output_token_ids": list(answer.token_ids),
                        "finish_reason": answer.finish_reason, "stop_reason": answer.stop_reason,
                        "text": answer.text, "final_text": final, "thinking_parse": parsing,
                        "reasoning_tokens_retokenized": len(tokenizer.encode(reasoning, add_special_tokens=False)),
                        "final_tokens_retokenized": len(tokenizer.encode(final, add_special_tokens=False)),
                        "num_cached_tokens": getattr(out, "num_cached_tokens", None),
                        "metrics": dataclasses.asdict(out.metrics) if dataclasses.is_dataclass(out.metrics) else repr(out.metrics)})
                if arm == "joint":
                    record["answers"], record["answer_parse_status"] = parse_joint(finals[0], b)
                else:
                    record["answers"] = finals
                    record["answer_parse_status"] = "ok" if all(finals) else "missing_final_answer"
                record["any_cap_hit"] = any(r["finish_reason"] == "length" for r in record["requests"])
                trace_path = ROOT / "results/raw/application_steps" / (destination.stem + f"_g{group_index}_{arm}_{repeat}_steps.jsonl")
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text("".join(json.dumps(x) + "\n" for x in observations))
                record["step_observer_trace"] = str(trace_path.relative_to(ROOT))
                record["step_observer_sha256"] = digest(trace_path.read_bytes())
                record["step_observer_calls"] = len(observations)
                report["runs"].append(record)
                write_json(destination, report)
                print(json.dumps({k: record[k] for k in ("group_index", "arm", "cache", "status", "group_jct_seconds", "any_cap_hit", "answer_parse_status")}), flush=True)
    report["status"] = "complete"
    report["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    report["total_process_seconds"] = time.monotonic() - start
    write_json(destination, report)
    nv.nvmlShutdown()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if "--output" in sys.argv:
            output_path = ROOT / sys.argv[sys.argv.index("--output") + 1]
            if output_path.exists():
                failed_report = json.loads(output_path.read_text())
                failed_report.update(status="failed", failure=repr(exc),
                                     failed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                write_json(output_path, failed_report)
        raise
