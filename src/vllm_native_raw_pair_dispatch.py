"""Frozen development rule. No runtime timings, future lengths, or per-input oracle."""
import hashlib
import json
from pathlib import Path

POLICY_PATH = 'results/xqa_raw_dispatch_policy_v1/policy.json'
EXPECTED_PARAMETERS = {'min_batch': 6, 'min_prefix': 65536, 'max_private_prefix_ratio': 0.125}


def frozen_policy():
    path = Path(POLICY_PATH)
    source = path.read_bytes()
    report = json.loads(source)
    assert report['status'] == 'development_policy_frozen'
    assert report['parameters'] == EXPECTED_PARAMETERS
    return {'parameters': dict(EXPECTED_PARAMETERS), 'path': POLICY_PATH,
            'sha256': hashlib.sha256(source).hexdigest(),
            'heldout_status_at_freeze': report['heldout_status']}


def eligible_from_current_state(batch, prefix_tokens, current_lengths):
    assert len(current_lengths) == batch and prefix_tokens > 0
    if batch < 6: return False, 'development_policy_small_batch'
    if prefix_tokens < 65536: return False, 'development_policy_short_prefix'
    max_private = max(current_lengths) - prefix_tokens
    if max_private <= 0: return False, 'prefix_not_fully_available'
    # Exactly .125, avoiding any threshold rounding ambiguity.
    if max_private * 8 > prefix_tokens: return False, 'development_policy_private_prefix_ratio'
    return True, None
