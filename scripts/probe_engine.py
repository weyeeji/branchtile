"""Real-weight native engine deployment qualification, not a speedup benchmark."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import vllm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from capture_model import make_inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--tp', type=int, default=1)
    p.add_argument('--output', required=True)
    p.add_argument('--max-model-len', type=int, default=32768)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    manifest = Path('data/manifests/qmsum_train_exploratory8.json')
    group = json.loads(manifest.read_text())['groups'][0]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    inputs, start, end = make_inputs(tokenizer, group)
    t0 = time.monotonic()
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype='bfloat16',
              max_model_len=args.max_model_len, max_num_seqs=4,
              max_num_batched_tokens=4096, enable_prefix_caching=True,
              enforce_eager=True, gpu_memory_utilization=0.80,
              disable_custom_all_reduce=True, disable_log_stats=False,
              seed=20261016)
    init_seconds = time.monotonic()-t0
    batches = []
    # A repeated native call qualifies cached execution, not a formal A/A trial.
    for repeat in range(2):
        t0 = time.monotonic()
        outputs = llm.generate([{'prompt_token_ids': ids} for ids in inputs],
                               SamplingParams(temperature=0, max_tokens=256), use_tqdm=False)
        elapsed = time.monotonic()-t0
        records = []
        for out in outputs:
            answer = out.outputs[0]
            records.append({'input_tokens': len(out.prompt_token_ids),
                            'generated_tokens': len(answer.token_ids),
                            'finish_reason': answer.finish_reason,
                            'output_sha256': hashlib.sha256(answer.text.encode()).hexdigest(),
                            'num_cached_tokens': getattr(out, 'num_cached_tokens', None),
                            'metrics': dataclasses.asdict(out.metrics) if dataclasses.is_dataclass(out.metrics) else repr(out.metrics)})
        batches.append({'repeat': repeat, 'seconds': elapsed, 'requests': records})
    record = {'scope': 'native real-weight deployment qualification only; no candidate integration, quality, or speedup claim',
              'uid': os.geteuid(), 'python': sys.executable, 'vllm': vllm.__version__,
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'visible_gpus': os.environ.get('CUDA_VISIBLE_DEVICES'), 'model': args.model,
              'tensor_parallel_size': args.tp, 'enforce_eager': True,
              'disable_custom_all_reduce': True, 'enable_prefix_caching': True,
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'document_sha256': group['document_sha256'], 'document_token_range': [start, end],
              'initialization_seconds': init_seconds, 'batches': batches}
    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
