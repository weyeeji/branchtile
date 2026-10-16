"""Freeze two extra repeats for every feasible selected document, without outcome selection."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    assert not p.exists(), p
    p.write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    primary_path = ROOT / 'results/grouped_hint_next8_production_launch_v1.json'
    primary = json.loads(primary_path.read_text())
    for name, digest in primary['frozen_source_sha256'].items():
        assert sha(ROOT / name) == digest
    new_sources = {name: sha(ROOT / name) for name in [
        'scripts/prepare_grouped_hint_next8_repetitions.py',
        'scripts/audit_grouped_hint_next8_repetitions.py',
        'docs/grouped_hint_next8_repetition_protocol_20261017.md']}
    repeats = []
    ledger = (ROOT / 'results/job_ledger.jsonl').read_text()
    for index in (1, 2):
        plan = copy.deepcopy(primary)
        plan['status'] = 'frozen_same_inputs_repeat_not_new_validation'
        plan['evaluation_role'] = 'all_six_feasible_documents_repeated_for_timing_stability'
        plan['repeat_index'] = index
        plan['primary_first_run_plan_sha256'] = sha(primary_path)
        plan['frozen_source_sha256'].update(new_sources)
        paths = []
        for document in plan['documents']:
            if not document['context_feasible']:
                continue
            number = document['evaluation_index']
            job_path = ROOT / f'configs/grouped_hint_next8_eval{number}_jobs.json'
            jobs = json.loads(job_path.read_text())
            assert len(jobs) == 1
            job = copy.deepcopy(jobs[0])
            job['name'] = f'grouped_hint_next8_eval{number}_repeat{index}_v1'
            assert job['name'] not in ledger
            output = f'results/grouped_hint_next8_repeat{index}_v1/eval{number}_bf16'
            assert not (ROOT / output).exists()
            job['argv'][job['argv'].index('--output') + 1] = output
            job['gpus'] = [6, 7]
            document['job_name'] = job['name']
            document['summary_path'] = output + '/summary.json'
            new_job_path = ROOT / f'configs/grouped_hint_next8_eval{number}_repeat{index}_jobs.json'
            save(new_job_path, [job])
            paths.append(str(new_job_path.relative_to(ROOT)))
        p = ROOT / f'results/grouped_hint_next8_repeat{index}_plan_v1.json'
        save(p, plan)
        repeats.append({'repeat_index': index, 'plan': str(p.relative_to(ROOT)), 'plan_sha256': sha(p), 'job_manifests': paths})
    output = ROOT / 'results/grouped_hint_next8_repetition_launch_v1.json'
    save(output, {'status': 'repetition_protocol_frozen_before_additional_runs', 'freeze_time_utc': datetime.now(timezone.utc).isoformat(),
        'primary_first_run_plan': str(primary_path.relative_to(ROOT)), 'primary_first_run_plan_sha256': sha(primary_path),
        'frozen_analysis_sources': new_sources, 'repeats': repeats,
        'all_selected_documents': 8, 'all_selected_questions': 98, 'feasible_documents_each_repeat': 6,
        'configuration_failures_each_repeat': 2, 'additional_runs_each_feasible_document': 2,
        'source_code_changed': False, 'sampling_seed_changed': False, 'reads_any_model_summary_or_output': False,
        'initial_results_role': 'Original first run remains the predeclared primary result; never replaced by better repeats.',
        'failure_rule': 'Every original selected input remains. Any failed repeat is retained; no replacement, silent retry, output selection, or fabricated ratio.',
        'statistical_unit': 'One whole document per new process per run. Phases, tokens, layers and ranks are not independent samples.',
        'GPU_assignment': '6,7 placeholders; root may assign free TP2 pairs without changing source, input, seed or output identities.'})
    print(output.relative_to(ROOT))


if __name__ == '__main__':
    main()
