"""Audit CUPTI launch structure; no privileged counters or traffic inference."""
import hashlib
import json
import os
import sys
from pathlib import Path

assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
root = Path(__file__).resolve().parents[1]
rows = []
for doc in (1, 7):
    for rank in (0, 1):
        for layer in (0, 15, 31, 47):
            name = f'train{doc}_r{rank}_l{layer}'
            path = root / f'results/xqa_raw_pair_perf_v1/{name}.trace.json'
            kernels = [e for e in json.loads(path.read_text())['traceEvents'] if e.get('cat') == 'kernel']
            # Benchmark replays its five arms serially: native, AA, no-PDL,
            # unshared checkpoint+resume, and copies+paired prefix+resume.
            assert len(kernels) == 10, name
            native, prefix, tail = kernels[0], kernels[8], kernels[9]
            assert native['args']['grid'] == tail['args']['grid'] == [10, 2, 8]
            assert prefix['args']['grid'] == [10, 2, 4]
            assert all(e['args']['registers per thread'] == 168 and
                       e['args']['shared memory'] == 79488 and
                       e['args']['block'] == [128, 1, 2] for e in (native, prefix, tail))
            assert all('copy_kernel' in e['name'] for e in kernels[5:8])
            rows.append({'case': name, 'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'native_CTA_count': 160, 'paired_prefix_CTA_count': 80, 'tail_CTA_count': 160,
                'registers_per_thread': 168, 'shared_memory_bytes_per_CTA': 79488,
                'threads_per_CTA': 256, 'charged_copy_kernel_count': 3,
                'single_profile_invocation_us_diagnostic_only': {
                    'native': native['dur'], 'prefix': prefix['dur'], 'tail': tail['dur'],
                    'copies': [e['dur'] for e in kernels[5:8]]}})
output = root / 'results/xqa_raw_pair_launch_resources_v1.json'
output.write_text(json.dumps({'status': 'verified_16_traces', 'rows': rows,
    'interpretation': 'Native and candidate have the same observed per-CTA register/shared-memory allocation. Pairing halves prefix CTAs while retaining original split count; a separate native-size tail launch is charged.',
    'limits': ['CUPTI launch metadata and activity only; no NCU counters or root privileges.',
               'Grid size is not measured SM occupancy. Do not use the profiler occupancy estimate as hardware evidence.',
               'Logical prefix load reuse does not prove a particular HBM traffic reduction.',
               'Single profile durations are diagnostic; randomized block timings are the performance evidence.']}, indent=2)+'\n')
print(output)
