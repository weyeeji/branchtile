"""Exercise CPU admission identity and block graph-time prompt readback."""
import ast
import hashlib
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

import pytest


def test_admission_identity_prevents_readback_and_rejects_unseen_requests():
    observed = []
    prompts = [[9, 2, 5], [7, 3]]
    config = {'prompt_token_hashes': [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]}
    runner = SimpleNamespace()
    def add_requests(scheduler):
        for request in scheduler.scheduled_new_reqs:
            assert request.req_id in worker.native_probe['requests']
        observed.append('add_original')
    runner.add_requests = add_requests
    def original_graph(desc):
        # A missing mapping in the inherited implementation would read the GPU.
        for rid in worker.native_probe['batch'].req_ids:
            if rid not in worker.native_probe['requests']:
                raise RuntimeError('GPU token readback attempted')
        observed.append('graph_original')
        return desc
    runner.cudagraph_manager = SimpleNamespace(run_fullgraph=original_graph)
    class Base:
        def configure_native_probe(self, cfg):
            self.native_probe = {'requests': {}, 'phase': 'test'}
            return {}
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'scripts/vllm_native_precision_worker.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = {'NativeBackendWorkerExtension': Base, 'hashlib': hashlib, 'json': json,
                 'Path': Path, 're': re, 'time': time}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<actual-worker-class>', 'exec'), namespace)
    worker = namespace['NativePrecisionWorkerExtension']()
    worker.model_runner = runner
    worker.configure_native_precision_probe(config)
    requests = [SimpleNamespace(req_id='r1', prompt_len=2, prefill_token_ids=[7, 3, 99]),
                SimpleNamespace(req_id='r0', prompt_len=3, prefill_token_ids=[9, 2, 5])]
    runner.add_requests(SimpleNamespace(scheduled_new_reqs=requests))
    assert worker.native_probe['requests'] == {'r1': 1, 'r0': 0}
    worker.native_probe['batch'] = SimpleNamespace(req_ids=['r1', 'r0'], num_reqs=2)
    assert runner.cudagraph_manager.run_fullgraph('native') == 'native'
    assert observed == ['add_original', 'graph_original']
    worker.native_probe['batch'] = SimpleNamespace(req_ids=['unknown'], num_reqs=1)
    with pytest.raises(AssertionError):
        runner.cudagraph_manager.run_fullgraph('native')
    assert observed == ['add_original', 'graph_original']
    bad = SimpleNamespace(req_id='bad', prompt_len=2, prefill_token_ids=[5, 5])
    with pytest.raises(AssertionError):
        runner.add_requests(SimpleNamespace(scheduled_new_reqs=[bad]))
    assert 'bad' not in worker.native_probe['requests']
