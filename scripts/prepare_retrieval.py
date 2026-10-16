"""Prepare frozen hybrid-RAG development grids with measured stage costs."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retrieval_baseline import make_chunks, hybrid_rank, select_context, ranking


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--groups", required=True)
    p.add_argument("--generator-tokenizer", required=True)
    p.add_argument("--embedding", default="data/models/Qwen3-Embedding-0.6B")
    p.add_argument("--reranker", default="data/models/Qwen3-Reranker-0.6B")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.manual_seed(20261016)
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temp = destination.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        temp.replace(destination)

    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text())
    report = {"schema": 1, "status": "initializing", "pid": os.getpid(), "uid": os.geteuid(),
              "python": sys.executable, "config": vars(args), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "development RAG preparation; no test-answer selection and no candidate attention mechanism",
              "protocol": {"chunk_targets": [512, 1024], "context_budgets": [4096, 8192],
                           "rrf": 60, "candidates": 20, "rankers": ["hybrid", "hybrid_rerank"],
                           "normalization": "L2 normalized full1024-dimensional last-token embeddings",
                           "precision": "BF16 weights, SDPA, FP32 normalized similarities and yes/no softmax",
                           "chunk_policy": "whole source lines; markdown table headers repeated across row boundaries; oversize atomic rows recorded; no silent truncation",
                           "context_order": "ranked admission under exact tokenizer budget, original source order"},
              "model_identities": {}, "groups": []}
    for name, model_path in (("embedding", args.embedding), ("reranker", args.reranker)):
        identity_path = ROOT / "results" / (Path(model_path).name + "_download.json")
        identity = json.loads(identity_path.read_text())
        assert identity["complete"] and identity.get("verified_shards")
        report["model_identities"][name] = identity
    save()
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.generator_tokenizer, local_files_only=True)
    etok = AutoTokenizer.from_pretrained(args.embedding, padding_side="left", local_files_only=True)
    rtok = AutoTokenizer.from_pretrained(args.reranker, padding_side="left", local_files_only=True)
    embedding = AutoModel.from_pretrained(args.embedding, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).cuda().eval()
    reranker = AutoModelForCausalLM.from_pretrained(args.reranker, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).cuda().eval()
    torch.cuda.synchronize()
    report["model_initialization_seconds"] = time.monotonic() - t0
    yes_id, no_id = rtok.convert_tokens_to_ids("yes"), rtok.convert_tokens_to_ids("no")

    @torch.inference_mode()
    def embed(texts):
        all_embeddings = []
        for first in range(0, len(texts), args.batch_size):
            encoded = etok(texts[first:first + args.batch_size], padding=True, truncation=False, return_tensors="pt").to("cuda")
            assert encoded.input_ids.shape[1] <= 32768
            hidden = embedding(**encoded, use_cache=False).last_hidden_state[:, -1].float()
            all_embeddings.append(F.normalize(hidden, p=2, dim=1))
        return torch.cat(all_embeddings)

    @torch.inference_mode()
    def rerank(query, texts, instruction):
        # Official Qwen reranker conversation markers and yes/no binary logits.
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        formatted = [prefix + f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {text}" + suffix for text in texts]
        output = []
        for first in range(0, len(formatted), args.batch_size):
            encoded = rtok(formatted[first:first + args.batch_size], padding=True, truncation=False, add_special_tokens=False, return_tensors="pt").to("cuda")
            assert encoded.input_ids.shape[1] <= 32768
            logits = reranker(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1]
            values = torch.stack((logits[:, no_id], logits[:, yes_id]), dim=-1).float().softmax(-1)[:, 1]
            output.extend(values.cpu().tolist())
        return output

    report["status"] = "running"
    fixture_queries = ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:What is the capital of China?",
                       "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:Explain gravity"]
    fixture_documents = ["The capital of China is Beijing.",
                         "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets in the solar system."]
    fixture_embeddings = embed(fixture_queries + fixture_documents)
    fixture_scores = (fixture_embeddings[:2] @ fixture_embeddings[2:].T).cpu().tolist()
    fixture_rerank = rerank("What is the capital of China?", fixture_documents,
                           "Given a web search query, retrieve relevant passages that answer the query")
    assert fixture_scores[0][0] > fixture_scores[0][1] and fixture_scores[1][1] > fixture_scores[1][0]
    assert fixture_rerank[0] > fixture_rerank[1]
    report["semantic_fixture"] = {"queries": fixture_queries, "documents": fixture_documents,
        "embedding_cosine": fixture_scores, "rerank_yes_probabilities": fixture_rerank,
        "purpose": "matching-pair ordering sanity check, not retrieval quality or bitwise parity qualification",
        "sources": ["https://huggingface.co/Qwen/Qwen3-Embedding-0.6B", "https://huggingface.co/Qwen/Qwen3-Reranker-0.6B"]}
    save()
    for index in map(int, args.groups.split(",")):
        group = manifest["groups"][index]
        text = (ROOT / group["document_path"]).read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == group["document_sha256"]
        task = ("Given a question about an annual report, retrieve passages and tables needed to answer it."
                if group["dataset"] == "DocFinQA" else
                "Given a question about a meeting, retrieve transcript passages needed to answer it.")
        group_record = {"group_index": index, "document_id": group["document_id"], "document_sha256": group["document_sha256"],
                        "question_ids": [q["question_sha256"] for q in group["questions"]], "chunk_grids": [], "variants": []}
        for target in [512, 1024]:
            start = time.monotonic()
            chunks = make_chunks(text, tokenizer, target)
            chunking_seconds = time.monotonic() - start
            start = time.monotonic()
            doc_embeddings = embed([chunk.text for chunk in chunks])
            torch.cuda.synchronize()
            indexing_seconds = time.monotonic() - start
            grid = {"target_tokens": target, "chunk_count": len(chunks),
                    "chunking_seconds": chunking_seconds, "document_embedding_seconds": indexing_seconds,
                    "chunks": [{k: v for k, v in dataclasses.asdict(c).items() if k != "text"} for c in chunks],
                    "chunk_text_sha256": [hashlib.sha256(c.text.encode()).hexdigest() for c in chunks],
                    "query_results": []}
            variant_questions = {(budget, ranker): [] for budget in [4096, 8192] for ranker in ["hybrid", "hybrid_rerank"]}
            for question in group["questions"]:
                start = time.monotonic()
                query_embedding = embed(["Instruct: " + task + "\nQuery:" + question["question"]])
                similarities = (query_embedding @ doc_embeddings.T)[0].cpu().numpy()
                torch.cuda.synchronize()
                query_embed_seconds = time.monotonic() - start
                start = time.monotonic()
                candidates, score_record = hybrid_rank(chunks, question["question"], similarities)
                hybrid_seconds = time.monotonic() - start
                start = time.monotonic()
                rerank_scores = rerank(question["question"], [chunks[i].text for i in candidates], task)
                torch.cuda.synchronize()
                rerank_seconds = time.monotonic() - start
                reranked = [candidates[i] for i in ranking(rerank_scores)]
                grid["query_results"].append({"question_id": question["question_sha256"],
                    "query_embedding_seconds": query_embed_seconds, "hybrid_ranking_seconds": hybrid_seconds,
                    "reranking_seconds": rerank_seconds, "hybrid_candidates": candidates,
                    "reranked_candidates": reranked, "rerank_scores": rerank_scores, **score_record})
                for budget, ranker in variant_questions:
                    start = time.monotonic()
                    context, selection = select_context(chunks, candidates if ranker == "hybrid" else reranked, tokenizer, budget)
                    selection_seconds = time.monotonic() - start
                    variant_questions[(budget, ranker)].append({"question_id": question["question_sha256"],
                        "context": context, "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                        "selection": selection, "context_assembly_seconds": selection_seconds,
                        "per_question_retrieval_seconds": query_embed_seconds + hybrid_seconds + selection_seconds + (rerank_seconds if ranker == "hybrid_rerank" else 0)})
            group_record["chunk_grids"].append(grid)
            for (budget, ranker), questions in variant_questions.items():
                group_record["variants"].append({"key": f"c{target}_b{budget}_{ranker}", "chunk_target": target,
                    "retrieval_budget": budget, "ranker": ranker,
                    "shared_document_preparation_seconds": chunking_seconds + indexing_seconds,
                    "serial_query_retrieval_seconds": sum(q["per_question_retrieval_seconds"] for q in questions),
                    "questions": questions})
        report["groups"].append(group_record)
        save()
        print(json.dumps({"group_index": index, "variants": len(group_record["variants"])}), flush=True)
    report["status"] = "complete"
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    save()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if "--output" in sys.argv:
            output_path = ROOT / sys.argv[sys.argv.index("--output") + 1]
            if output_path.exists():
                failed = json.loads(output_path.read_text())
                failed.update(status="failed", failure=repr(exc))
                output_path.write_text(json.dumps(failed, indent=2, ensure_ascii=False) + "\n")
        raise
