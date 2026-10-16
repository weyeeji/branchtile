"""Fail-closed document-indexed provenance for the multi-document production path."""
import hashlib
import json
from pathlib import Path
from src.vllm_native_raw_pair_general_setup import validate_offline_records


def validate_document_records(records,documents,strict_sources,fingerprint,enabled_batches):
    expected={d['document_id']:d for d in documents}
    if set(r.get('document_id') for r in records)!=set(expected):
        raise ValueError('Offline document set differs from the frozen protocol')
    result={}
    for docid,doc in expected.items():
        rows=[r for r in records if r['document_id']==docid]
        if len(rows)!=2 or {r['rank'] for r in rows}!={0,1}:
            raise ValueError('Each document needs exactly two TP-rank receipts')
        for row in rows:
            if row.get('validation') is not True or row.get('method')!='native' or row.get('phase_valid') is not True:
                raise ValueError('Offline receipt is not a valid original-native qualification trajectory')
            if row.get('capture_invocations')!=1 or row.get('enabled_batches')!=list(enabled_batches):
                raise ValueError('Offline fixed graph bank contract mismatch')
            if row.get('plan',{}).get('nominal_active_prefix_tokens')!=doc['common_complete_page_tokens']:
                raise ValueError('Offline document prefix mismatch')
            if any(p.get('policy_prefix_tokens')!=doc['common_complete_page_tokens'] for p in row.get('prepare_records',[])):
                raise ValueError('Offline prepare record belongs to another prefix')
        audit=validate_offline_records(rows,strict_sources,fingerprint,[],2)
        coverage=[{c['batch'] for c in r['checks']} for r in rows]
        jointly=sorted(set.intersection(*coverage)&set(enabled_batches))
        if not jointly:raise ValueError('Document has no jointly qualified candidate batch')
        result[docid]={'qualified_batches':jointly,'unqualified_batches_fallback_native':sorted(set(enabled_batches)-set(jointly)),
                       'nominal_prefix_tokens':doc['common_complete_page_tokens'],'strict_audit':audit}
    return result


def load_completed_qualification(summary_path,protocol,root,strict_sources):
    """Verify the completed strict summary as well as every referenced rank receipt."""
    path=Path(summary_path).resolve();summary=json.loads(path.read_text())
    if summary.get('status')!='completed' or summary.get('protocol')!=protocol:
        raise ValueError('Require a completed qualification for the exact multi-document protocol')
    for source,expected in summary['source_sha256'].items():
        if hashlib.sha256((Path(root)/source).read_bytes()).hexdigest()!=expected:
            raise ValueError('Strict summary source changed: '+source)
    if 'scripts/benchmark_raw_pair_multidoc.py' not in summary['source_sha256']:
        raise ValueError('Qualification summary is not from the strict multi-document driver')
    runs=[r for r in summary['runs'] if r.get('validation')]
    if len(runs)!=len(protocol['documents']) or {r['document_id'] for r in runs}!={d['document_id'] for d in protocol['documents']}:
        raise ValueError('Missing or duplicate strict document trajectory')
    records=[];inputs=[]
    for run in runs:
        if run['method']!='native' or not run.get('qualification_joint_audit'):
            raise ValueError('Missing strict joint audit')
        audited={r['rank']:r for r in run['qualification_joint_audit']}
        for worker in run['workers']:
            record_path=Path(worker['raw_pair_record']);record=json.loads(record_path.read_text())
            expected=audited[worker['rank']]
            if hashlib.sha256(record_path.read_bytes()).hexdigest()!=expected['sha256']:
                raise ValueError('Strict-audited rank receipt bytes changed')
            if record['document_id']!=run['document_id']:raise ValueError('Cross-document receipt path')
            records.append(record);inputs.append({'path':str(record_path.resolve()),'sha256':hashlib.sha256(record_path.read_bytes()).hexdigest()})
    fingerprint=records[0]['execution_fingerprint']
    audit=validate_document_records(records,protocol['documents'],strict_sources,fingerprint,protocol['candidate_enabled_batches'])
    return {'summary':str(path),'summary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'record_inputs':inputs,
            'records':records,'execution_fingerprint':fingerprint,'by_document':audit,
            'native_tokens_by_document':{r['document_id']:[a['tokens'] for a in r['answers']] for r in runs}}
