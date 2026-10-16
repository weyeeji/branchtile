"""CPU-only provenance/statistical audit: Thinking actual8cells and survivor6cells."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_xqa_raw_gqa4_perf import sha,interval


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);assert not out.exists();out.mkdir(parents=True)
    report={'status':'auditing','source_sha256':sha(__file__),'cells':[],'limitations':[
        'One development document; independent numerical records and per-cell block CIs, not population confidence.',
        'Survivor subset Q is a captured actual AR state, not later natural survivor Q. Extra-tail8192 is synthetic KV stress.',
        'Thinking actual captures remain independent from Instruct weights/outputs. No natural timing inferred from these kernels.']}
    files=[('thinking',p) for p in Path('results/xqa_raw_gqa4_thinking_perf_v1').glob('*.json')]
    files += [('survivor',p) for p in Path('results/gqa4_survivor_operator_v1').glob('*.json')]
    seen={'thinking':set(),'survivor':set()};hashed={}
    def cached_sha(path):
        key=str(path)
        if key not in hashed:hashed[key]=sha(path)
        return hashed[key]
    arms={'native','AA','native_PDL_off','unshared_checkpoint','candidate'}
    for kind,path in sorted(files):
        row=json.loads(path.read_text());assert row['status']=='qualified_and_measured'
        assert row['checks'] and all(v is True for v in row['checks'].values())
        assert row['prefix']==120528 and row['args']['blocks']==30
        for f,expected in row['source_sha256'].items():assert cached_sha(f)==expected
        assert cached_sha(row['capture']['path'])==row['capture']['sha256']
        if kind=='thinking':
            payload_path=Path(row['capture']['path']);rank=int(payload_path.parent.parent.name[4:]);layer=int(payload_path.stem[5:])
            key=(rank,layer);assert row['checks']['original_native_O_bits'] and row['batch']==8 and row['S']==5
            assert 'gqa4_thinking_native_capture_train1_v1' in str(payload_path)
        else:
            key=(row['batch'],row['args']['extra_tail']);assert row['S']=={6:7,7:6,8:5}[row['batch']]
            assert row['selected_original_rows']==[7,0,6,1,5,2,4,3][:row['batch']]
            if row['batch']<8:assert 'original_native_O_bits' not in row['checks']
            if row['batch']==7:assert row['checks']['odd_dummy_Q_zero']
        assert key not in seen[kind];seen[kind].add(key)
        cell={'kind':kind,'identity':list(key),'path':str(path),'sha256':sha(path),'capture_sha256':row['capture']['sha256'],
              'batch':row['batch'],'S':row['S'],'prefix':row['prefix'],'maxprivate':max(row['lengths'])-row['prefix'],'all_checks_pass':True}
        for regime in ['hot','cold']:
            data=row[regime];assert len(data['blocks_ms'])==len(data['orders'])==30
            assert all(set(x)==arms for x in data['blocks_ms']) and all(set(x)==arms and len(x)==5 for x in data['orders'])
            for field,num,den in [('native_over_candidate','native','candidate'),('AA','native','AA'),('native_over_unshared','native','unshared_checkpoint')]:
                assert interval(data['blocks_ms'],num,den)==data[field]
            cell[regime]={k:data[k] for k in ['native_over_candidate','AA','native_over_unshared']}
        report['cells'].append(cell)
    assert seen['thinking']=={(r,l) for r in (0,1) for l in (0,11,23,35)}
    assert seen['survivor']=={(b,t) for b in (6,7,8) for t in (0,8192)}
    capture=Path('results/gqa4_thinking_native_capture_train1_v1/summary.json');summary=json.loads(capture.read_text())
    assert summary['status']=='completed' and summary['model']=='data/models/Qwen3-4B-Thinking-2507'
    for worker in summary['workers']:
        assert len(worker['layers'])==36 and len(worker['layer_files'])==4
        assert all(m['decode']['kernel']=='FlashInferDecodeKernel.XQA' for m in worker['actual_metadata'].values())
    report['thinking_capture_summary_sha256']=sha(capture);report['status']='passed'
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# GQA4 follow-up operator audit','',
           'All14cells pass source/capture provenance,recorded numerical and graph-update checks,and independent paired-CI recomputation. No GQA8 policy was reused.','',
           '| Kind/cell | B | S | Maximum private tokens | Hot speedup [95% CI] | Cold speedup [95% CI] |','|---|---:|---:|---:|---|---|']
    for c in report['cells']:
        fields=[]
        for regime in ['hot','cold']:
            v=c[regime]['native_over_candidate'];lo,hi=v['bootstrap_95ci'];fields.append(f"{v['paired_median_ratio']:.3f} [{lo:.3f},{hi:.3f}]")
        lines.append(f"| {c['kind']}/{c['identity']} | {c['batch']} | {c['S']} | {c['maxprivate']} | {fields[0]} | {fields[1]} |")
    lines += ['', 'The weakest long-tail cell is B6 cold:1.089×,95% paired lower bound1.088×. All tested endpoints remain positive. This supports a bounded development model experiment at P120528 and private≤8192; it does not establish performance at other prefixes,models,batches or every intermediate length. Natural survivor states still require full-model qualification.', '']
    (out/'report.md').write_text('\n'.join(lines));print(json.dumps({'status':'passed','cells':len(files)}))
if __name__=='__main__':main()
