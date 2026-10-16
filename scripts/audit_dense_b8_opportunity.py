"""CPU-only natural B8 input/capacity audit; produces a proposed native pilot.

No GPU initialization, engine run, throughput prediction, or candidate claim.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def save(path, value):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    from transformers import AutoTokenizer
    from scripts.capture_model import make_inputs
    manifest_path = 'data/manifests/docfinqa_training_large_group_pilot_v1.json'
    manifest = json.loads((ROOT / manifest_path).read_text())
    model = 'data/models/Qwen3-30B-A3B-Thinking-2507-FP8'
    tokenizer = AutoTokenizer.from_pretrained(ROOT / model, local_files_only=True)
    documents, records = [], []
    inputs = [manifest_path, model + '/config.json',
              'data/models/Qwen3-4B-Instruct-2507/config.json',
              'scripts/capture_model.py', 'scripts/audit_dense_b8_opportunity.py']
    models = [('main30b_tp2', model, 2),
              ('instruct4b_tp1', 'data/models/Qwen3-4B-Instruct-2507', 1)]
    tokenizers = {model: tokenizer, models[1][1]: AutoTokenizer.from_pretrained(ROOT / models[1][1], local_files_only=True)}
    inputs.extend(path + '/' + filename for _, path, _ in models for filename in ['tokenizer.json', 'tokenizer_config.json'])
    for group, old_name in zip(manifest['groups'], ['train1', 'train7'], strict=True):
        prompts, begin, end = make_inputs(tokenizer, group)
        assert len(prompts) == len(group['questions']) == 8
        shared_tokens = end // 16 * 16
        docs = dict(document_id=group['document_id'], document_sha256=group['document_sha256'],
                    prompt_lengths=list(map(len, prompts)),
                    prompt_token_hashes=[hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts],
                    document_token_range=[begin, end])
        documents.append(docs)
        old_path = f'results/platform/capture_main30b_tp2_{old_name}_b8_256.json'
        inputs.extend([old_path, group['document_path']])
        old = json.loads((ROOT / old_path).read_text())
        warm = next(s for s in old['stages'] if s['name'] == 'apc_warm_natural_group')
        assert [r['input_tokens'] for r in warm['requests']] == docs['prompt_lengths']
        states = old['worker_state_after_initialization']
        record = dict(**docs, B=8, conservative_shared_complete_page_tokens=shared_tokens,
                      old_capture=dict(path=old_path, eager=old['engine_configuration']['enforce_eager'],
                                       warm_group_seconds=warm['seconds'],
                                       cached_tokens=[r['num_cached_tokens'] for r in warm['requests']],
                                       generated_tokens=[r['generated_tokens'] for r in warm['requests']],
                                       preemptions=[r['metrics']['num_preemptions'] for r in warm['requests']],
                                       pool_token_capacity_per_rank=[s['cache_num_gpu_blocks'] * s['cache_block_size'] for s in states],
                                       warning='Old eager instrumented capped run; not new graph latency or natural EOS.'),
                      payload_estimates=[])
        for label, path, tp in models:
            cfg = json.loads((ROOT / path / 'config.json').read_text())
            assert cfg['num_key_value_heads'] % tp == 0
            model_prompts, model_begin, model_end = make_inputs(tokenizers[path], group) if path != model else (prompts, begin, end)
            model_shared = model_end // 16 * 16
            for dtype, element_bytes in [('bf16', 2), ('fp8', 1)]:
                per_token = cfg['num_hidden_layers'] * (cfg['num_key_value_heads'] // tp) * cfg['head_dim'] * 2 * element_bytes
                estimates = []
                for suffix in [128, 256, 2048, 8192]:
                    private = sum(((len(p) - model_shared + suffix + 15) // 16) * 16 for p in model_prompts)
                    token_slots = model_shared + private
                    estimates.append(dict(output_cap_each=suffix, conservative_KV_token_slots=token_slots,
                                          KV_payload_GiB_per_rank=token_slots * per_token / 2**30,
                                          fits_old_main30b_pool_token_capacity=(token_slots <= min(record['old_capture']['pool_token_capacity_per_rank']))
                                          if label == 'main30b_tp2' and dtype == 'bf16' else None))
                record['payload_estimates'].append(dict(model=label, kv_dtype=dtype, TP=tp,
                    actual_model_prompt_lengths=list(map(len, model_prompts)), document_token_range=[model_begin, model_end],
                    bytes_per_token_all_layers_per_rank=per_token,
                    shared_prefix_payload_GiB_per_rank=model_shared * per_token / 2**30,
                    scenarios=estimates,
                    caveat='Payload only; excludes weights, workspace, graphs, allocator, scales, metadata and transient prefills. Model-specific tokenization is verified, but no new model was loaded or run.'))
        records.append(record)
    hist = {int(k): v for k, v in manifest['all_training_natural_question_count_histogram'].items()}
    report = dict(schema=1, status='CPU_audit_only_no_new_GPU', uid=os.geteuid(), python=sys.executable,
                  source_sha256={p: digest(p) for p in sorted(set(inputs))},
                  natural_group_distribution=dict(total=sum(hist.values()), at_least_8=sum(v for k,v in hist.items() if k>=8),
                                                  at_least_16=sum(v for k,v in hist.items() if k>=16), histogram=hist),
                  records=records,
                  interpretation='Natural simultaneous batch-query use case is possible. Dataset question counts do not prove agent fanout or real arrival concurrency. B8 is a scoped workload extension, not a new mechanism.')
    save('results/dense_b8_opportunity_v1/audit.json', report)
    base = json.loads((ROOT / 'configs/native_precision_natural_development_v1.json').read_text())
    engine = dict(base['engine'])
    engine.update(max_model_len=163840, max_num_seqs=8, gpu_memory_utilization=.90)
    engine['compilation_config'] = dict(mode=3, cudagraph_mode='FULL_DECODE_ONLY', cudagraph_capture_sizes=[1,2,3,4,5,6,7,8])
    protocol = dict(schema=1, protocol_id='native-opportunity-dense-b8-training-v1-20261016',
                    status='proposed_not_launched', model_path=model, model_revision=base['model_revision'],
                    manifest=manifest_path, documents=documents, engine=engine, seed=base['seed'], sampling=base['sampling'],
                    purpose='Measure dense APC-warm B8 whole-model opportunity on natural questions, not rerun failed sparse B8 dynamic kernel.',
                    source_audit='results/dense_b8_opportunity_v1/audit.json',
                    limitations=['Baseline-only measurement; neither a candidate speedup nor formal quality result.',
                                 'Fixed128 valid generation ignores EOS; same cap is not a guarantee of same cross-precision token work.',
                                 'Graph profiling is separate from unprofiled wall measurements; no rank-time summation.',
                                 'These two training documents do not establish general agent fanout or arrival distributions.',
                                 'Root must freeze launch sources and resource accounting before use; no GPU was launched by this audit.'])
    protocol_path = 'configs/native_opportunity_b8_training_v1.json'
    save(protocol_path, protocol)
    jobs = []
    for i, (doc_idx, dtype) in enumerate([(0,'auto'),(0,'fp8_e4m3'),(1,'fp8_e4m3'),(1,'auto')]):
        name = f'native_opportunity_b8_train{[1,7][doc_idx]}_{dtype}'
        jobs.append(dict(name=name, category='P1', gpus=[2*i,2*i+1], timeout_seconds=720,
            argv=['.venv-engine/bin/python','scripts/benchmark_native_opportunity.py','--protocol',protocol_path,
                  '--document-id', documents[doc_idx]['document_id'], '--kv-dtype',dtype,'--fixed-tokens','128',
                  '--natural-repeats','0','--output',f'results/native_opportunity_b8_v1/{name}'],
            env=dict(VLLM_WORKER_MULTIPROC_METHOD='spawn', NCCL_P2P_DISABLE='1', NCCL_DEBUG='WARN',
                     HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(ROOT), CUDA_HOME='/usr/local/cuda-13.2')))
    save('configs/native_opportunity_b8_training_jobs_v1.json', jobs)
    print(json.dumps(dict(status=report['status'], documents=[dict(id=r['document_id'], lengths=r['prompt_lengths']) for r in records],
                          distribution=report['natural_group_distribution'], proposed_job_count=len(jobs))))


if __name__ == '__main__':
    main()
