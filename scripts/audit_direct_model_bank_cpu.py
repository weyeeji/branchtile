"""CPU check of native per-layer Q/O selection and rejection boundaries."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import torch
from src.vllm_native_raw_pair_direct_plan import LayerSharedRawPairDirectBank

records = []
for batch in range(6, 22):
    bank = LayerSharedRawPairDirectBank.__new__(LayerSharedRawPairDirectBank)
    bank.query_layouts = {}
    bank.caches = [torch.empty((1, 2, 16, 256), dtype=torch.bfloat16) for _ in range(48)]
    bank.views = [(c[..., :128], c[..., 128:], c[..., :128].transpose(1, 2),
                   c[..., 128:].transpose(1, 2)) for c in bank.caches]
    launches = []
    plan = SimpleNamespace()
    def record_call():
        launches.append((plan.q, plan.output, plan.k, plan.wrapper_v))
        return plan.output
    plan.run = record_call
    bank.plans = {batch: plan}
    inputs = [torch.empty((batch, 16, 128), dtype=torch.bfloat16) for _ in range(48)]
    outputs = [torch.empty_like(q) for q in inputs]
    for layer, (q, output) in enumerate(zip(inputs, outputs)):
        assert bank.run_layer(layer, q, output) is output
        assert launches[-1][0] is q and launches[-1][1] is output
        assert launches[-1][2] is bank.views[layer][0]
        assert launches[-1][3] is bank.views[layer][3]
    assert len({x[0].data_ptr() for x in launches}) == 48
    assert len({x[1].data_ptr() for x in launches}) == 48
    before = len(launches)
    bad = torch.empty((batch, 16, 256), dtype=torch.bfloat16)[..., ::2]
    for q, output in [(bad, outputs[0]), (inputs[0], bad)]:
        try:
            bank.run_layer(0, q, output)
        except AssertionError:
            pass
        else:
            raise AssertionError('Noncontiguous address accepted')
        assert len(launches) == before
    records.append({'batch': batch, 'layers': 48, 'native_Q_O_and_cache_pointer_selection': True,
                    'noncontiguous_Q_and_O_rejected_before_launch': True})
assert not torch.cuda.is_initialized()
files = ['src/vllm_native_raw_pair_direct_plan.py', 'scripts/vllm_native_raw_pair_direct_worker.py',
         'scripts/benchmark_native_raw_pair_direct_e2e.py', 'scripts/benchmark_native_raw_pair_direct_production.py',
         'scripts/audit_direct_model_bank_cpu.py']
report = {'status': 'CPU_pointer_selection_passed', 'cuda_initialized': False,
          'checks': records, 'source_sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
          'limitations': 'Recording plan checks Python pointer selection only. Actual CUDA graph pointer capture, numerical behavior and full bank construction require the model qualification run.'}
(ROOT/'results/raw_direct_model_bank_cpu_v1.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'status': report['status'], 'batch_layer_selections': 16*48}))
