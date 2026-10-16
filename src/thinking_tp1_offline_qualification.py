"""Completed TP1 strict-summary provenance; never accepts TP2 or raw-only partial receipts."""
import hashlib,json
from pathlib import Path
from src.vllm_gqa4_raw_pair_setup import validate_offline_records
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_tp1_qualification(path,source_sha256,protocol,enabled_batches):
    p=Path(path).resolve();summary=json.loads(p.read_text())
    if summary.get('status')!='completed' or summary.get('protocol')!=protocol:raise ValueError('Completed exact TP1 strict protocol required')
    if 'scripts/qualify_thinking_tp1_quad_model.py' not in summary.get('source_sha256',{}):raise ValueError('Not the independent TP1 strict driver')
    for name,expected in summary['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Strict source changed: '+name)
    runs=summary.get('runs',[])
    if len(runs)!=1 or runs[0].get('validation') is not True or runs[0].get('method')!='native':raise ValueError('Require the native strict natural trajectory')
    run=runs[0]
    if len(run.get('workers',[]))!=1 or len(run.get('qualification_joint_audit',[]))!=1:raise ValueError('TP1 requires exactly one worker/audit')
    worker=run['workers'][0];audit=run['qualification_joint_audit'][0]
    rp=Path(worker['raw_pair_record']).resolve();record=json.loads(rp.read_text())
    if worker['rank']!=0 or audit['rank']!=0 or digest(rp)!=audit['sha256']:raise ValueError('Strict rank receipt bytes changed')
    if record.get('phase_valid') is not True or record.get('validation') is not True or record.get('method')!='native':raise ValueError('Not a passing original-native strict record')
    fingerprint=record['execution_fingerprint']
    if fingerprint.get('TP')!=1 or fingerprint.get('attention_shape')!=[36,32,8,128,16]:raise ValueError('Require actual TP1 Hq32/Hkv8 fingerprint')
    validation=validate_offline_records([record],source_sha256,fingerprint,enabled_batches,1)
    return {'summary':str(p),'summary_sha256':digest(p),'record_path':str(rp),'record_sha256':digest(rp),
            'fingerprint':fingerprint,'validation':validation,'reference_tokens':[a['tokens'] for a in run['answers']]}
