"""CPU assertions for actual direct-model Q/O layout receipts; no CUDA import."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_direct_layout(record):
    source_names = ['scripts/vllm_native_raw_pair_direct_worker.py',
        'src/vllm_native_raw_pair_direct_plan.py', 'src/xqa_raw_direct_input_runtime.py',
        'src/xqa_raw_direct_input_build.py']
    for name in source_names:
        assert record['source_sha256'][name] == sha(ROOT / name)
    build = record['build_receipt']
    assert build['patch_sha256'] == sha(ROOT / 'src/xqa_raw_direct_input_build.py')
    assert build['base_receipt']['native_group'] == 8 and build['base_receipt']['fold'] == 2
    base = build['base_receipt']
    assert base['patch_sha256'] == sha(ROOT / 'src/xqa_raw_grouped_hint_build.py')
    for receipt, directory in [(build, 'build/xqa_raw_direct_input'), (base, 'build/xqa_raw_grouped_hint')]:
        for name, digest in receipt['generated_sha256'].items():
            assert sha(ROOT / directory / receipt['identity'] / name) == digest
    plan = record['plan']
    assert plan['layers'] == 48
    assert plan['workspace_allocation_sets'] == plan['raw_state_allocation_sets'] == 1
    assert plan['runtime_candidate_PDL'] is False
    fp = record['execution_fingerprint']
    assert fp['attention_shape'] == [48,16,2,128,16] and fp['TP'] == 2
    assert fp['model_path'] == 'data/models/Qwen3-30B-A3B-Thinking-2507-FP8'
    enabled = record['enabled_batches']
    assert set(plan['query_layouts']) == {str(b) for b in enabled}
    verified = 0
    for b in enabled:
        layers = plan['query_layouts'][str(b)]
        assert set(layers) == {str(i) for i in range(48)}
        for layout in layers.values():
            assert layout['shape'] == [b,16,128]
            assert layout['stride'] == layout['output_stride'] == [2048,128,1]
            assert layout['Q_native_alias'] is True and layout['O_native_alias'] is True
            assert layout['Q_canonicalization_copy_charged'] is False
            verified += 1
    tree = ast.parse((ROOT / 'src/vllm_native_raw_pair_direct_plan.py').read_text())
    run = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'run_layer')
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'copy_' for node in ast.walk(run))
    return {'rank': record['rank'], 'enabled_batches': enabled,
        'verified_batch_layer_QO_layout_receipts': verified, 'actual_query_and_output_stride': [2048,128,1],
        'Q_O_alias_flags_true': True, 'Q_canonicalization_copy_flags_false': True,
        'direct_build_identity': build['identity'], 'grouped_hint_base_identity': base['identity'],
        'scope': 'Actual capture receipts and source contract, not an independent CUDA-profiler kernel/copy count. State export, native tail/merge, KV writes and graph execution remain charged.'}
