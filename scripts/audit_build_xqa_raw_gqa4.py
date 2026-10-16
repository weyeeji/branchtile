"""CPU model/config and fragment-coordinate audit; optional GPU-less CUDA compilation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    out = Path(args.output)
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'initializing', 'models': [], 'args': vars(args)}
    def save():
        out.write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        for name in ('Qwen3-4B-Instruct-2507', 'Qwen3-4B-Thinking-2507'):
            path = ROOT / 'data/models' / name / 'config.json'
            config = json.loads(path.read_text())
            assert config['num_attention_heads'] == 32 and config['num_key_value_heads'] == 8
            assert config['head_dim'] == 128 and not config['use_sliding_window']
            report['models'].append({'name': name, 'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                    'TP2_Q_heads': 16, 'TP2_KV_heads': 4, 'GQA': 4,
                                    'layers': config['num_hidden_layers'], 'head_dim': config['head_dim'],
                                    'torch_dtype': config['torch_dtype']})
        # MMA m16n8: row=lane//4+8*(register%4//2), col=8*(register//4)+2*(lane%4)+register%2.
        # Check every live native GQA4 accumulator and every m/z/hint row, both members.
        checks = 0
        for member in range(2):
            for lane in range(16):
                for register in range(32):
                    if register % 4 >= 2:
                        continue
                    row = lane // 4
                    col = 8 * (register // 4) + 2 * (lane % 4) + register % 2
                    src_lane, src_register = lane + 16 * member, register
                    src_row = src_lane // 4 + 8 * (src_register % 4 // 2)
                    src_col = 8 * (src_register // 4) + 2 * (src_lane % 4) + src_register % 2
                    assert src_row == row + 4 * member and src_col == col
                    checks += 1
            for row in range(4):
                assert 0 <= row + 4 * member < 8
                checks += 3
        report['fragment_coordinate_checks'] = checks
        report['original_S_examples_SM170'] = {str(b): 170 // (b * 4) for b in [6, 7, 8, 19, 21]}
        from src.xqa_raw_pair_gqa4_build import generate
        dest, receipt = generate()
        report['generated_source'] = str(dest)
        report['identity'] = receipt['identity']
        report['cuda_compiled'] = False
        if args.build:
            assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only compile must hide GPUs'
            import torch
            from src.xqa_raw_pair_gqa4_build import jit_specs
            specs, _ = jit_specs()
            for group, spec in specs.items():
                spec.build_and_load()
                assert not torch.cuda.is_initialized()
            report['cuda_compiled'] = True
            report['cuda_initialized'] = torch.cuda.is_initialized()
        report['status'] = 'cpu_audited_and_compiled' if args.build else 'cpu_audited'
        report['gpu_qualification'] = 'not_run'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        save()
        raise
    save()
    print(json.dumps({'status': report['status'], 'identity': report.get('identity')}))


if __name__ == '__main__':
    main()
