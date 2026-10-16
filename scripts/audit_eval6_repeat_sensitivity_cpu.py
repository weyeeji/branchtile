"""Keep the frozen primary observation and its follow-up repeat separate."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    output=ROOT/'results/eval6_original_repeat_sensitivity_v1.json';assert not output.exists()
    paths=[ROOT/'results/native_raw_pair_heldout_production_v1/eval6_bf16/summary.json',
           ROOT/'results/native_raw_pair_heldout_repeat1_v1/eval6_bf16/summary.json']
    summaries=[json.loads(p.read_text()) for p in paths]
    assert all(s['status']=='completed' for s in summaries)
    assert summaries[0]['source_sha256']==summaries[1]['source_sha256']
    assert summaries[0]['protocol_sha256']==summaries[1]['protocol_sha256']
    assert summaries[0]['manifest_sha256']==summaries[1]['manifest_sha256']
    for a,b in zip(summaries[0]['runs'],summaries[1]['runs'],strict=True):
        assert a['phase']==b['phase']
        for field in ('tokens','finish_reason','stop_reason','num_cached_tokens'):
            assert [x[field] for x in a['answers']]==[x[field] for x in b['answers']]
        assert a['quality']==b['quality'] and a['total_request_preemptions']==b['total_request_preemptions']
        assert a['graph_trajectory_sha256_by_rank']==b['graph_trajectory_sha256_by_rank']
    rows=[]
    for label,path,s in zip(('frozen_primary','followup_repeat'),paths,summaries,strict=True):
        audit_path=path.with_name('analysis.json' if label=='frozen_primary' else 'independent_analysis.json')
        audit=json.loads(audit_path.read_text());assert audit['status']=='qualified' and audit['summary_sha256']==sha(path)
        extra=sum(s[k] for k in ('offline_preflight_wall_seconds','candidate_configuration_wall_seconds','candidate_graph_capture_wall_seconds','setup_receipt_export_wall_seconds'))
        native=statistics.mean(r['cold_shared_document_total_wall_seconds'] for r in s['runs'] if r['method']=='native')
        candidate=statistics.mean(r['cold_shared_document_total_wall_seconds'] for r in s['runs'] if r['method']=='cascade')
        rows.append({'label':label,'summary_path':str(path.relative_to(ROOT)),'summary_sha256':sha(path),
            'independent_audit_sha256':sha(audit_path),'ready_ratio':s['comparisons']['natural']['native_over_candidate']['complete_generate_wall_seconds'],
            'cold_ratio':native/candidate,'net_with_all_per_document_setup':native/(candidate+extra),
            'native_AA_max_over_min':s['comparisons']['natural']['native_AA_wall_max_over_min'],
            'phase_generation_seconds':{r['phase']:r['complete_generate_wall_seconds'] for r in s['runs']},
            'setup_seconds':extra})
    result={'status':'original_and_repeat_independently_retained','source_sha256':sha(Path(__file__)),
        'source_input_output_APC_finishes_quality_and_graph_trajectories_equal':True,'runs':rows,
        'limits':['The original frozen result remains the primary evaluation observation; this later repeat does not replace it or retroactively define an average.',
            'The follow-up shows timing sensitivity with unchanged output and sources; it does not identify a unique cause of original drift.',
            'Two processes of one document and one seed do not establish a population confidence interval.',
            'Both use the earlier raw-pair build whose separate eval2 counterexample remains. This repeat does not repair the bitwise guarantee.']}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'runs':rows},indent=2))


if __name__=='__main__':main()
