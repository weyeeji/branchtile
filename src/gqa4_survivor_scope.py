"""CPU-only fixed GQA4 development scope; deliberately not a fitted GQA8 policy."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'configs/gqa4_survivor_scope_v1.json'
SCOPE=json.loads(PATH.read_text())
assert SCOPE['enabled_batches']==[6,7,8] and SCOPE['nominal_shared_prefix_tokens']==120528 and SCOPE['max_actual_private_tokens']==8192


def in_scope(batch,prefix,lengths):
    return (batch in SCOPE['enabled_batches'] and len(lengths)==batch and prefix==SCOPE['nominal_shared_prefix_tokens']
            and min(lengths)>prefix and max(lengths)-prefix<=SCOPE['max_actual_private_tokens'])


def receipt():
    return dict(SCOPE,path=str(PATH.relative_to(ROOT)),sha256=hashlib.sha256(PATH.read_bytes()).hexdigest())
