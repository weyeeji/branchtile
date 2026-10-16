"""CPU evidence that same-state diagnostics preserve an earlier native output."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--qualified-summary',required=True)
    parser.add_argument('--baseline-summary',required=True)
    parser.add_argument('--baseline-phase',default='native_natural')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    current_path,baseline_path=Path(args.qualified_summary),Path(args.baseline_summary)
    output=Path(args.output)
    assert not output.exists()
    current,baseline=json.loads(current_path.read_text()),json.loads(baseline_path.read_text())
    assert current['status']==baseline['status']=='completed'
    assert current['document_sha256']==baseline['document_sha256']
    assert current['prompt_token_hashes']==baseline['prompt_token_hashes']
    checked=next(r for r in current['runs'] if r['phase']=='qualification')
    original=next(r for r in baseline['runs'] if r['phase']==args.baseline_phase)
    assert checked['method']==original['method']=='native'
    assert checked['sampling']==original['sampling']
    fields=('tokens','num_cached_tokens','finish_reason','stop_reason')
    equal={field:[a[field] for a in checked['answers']]==[a[field] for a in original['answers']] for field in fields}
    assert all(equal.values())
    assert checked['quality']['score']==original['quality']['score']
    report={'status':'qualified','source_sha256':sha(Path(__file__)),
        'qualified_summary':str(current_path),'qualified_summary_sha256':sha(current_path),
        'baseline_summary':str(baseline_path),'baseline_summary_sha256':sha(baseline_path),
        'baseline_phase':args.baseline_phase,'document_sha256':current['document_sha256'],
        'natural_questions':len(checked['answers']),'equal':equal,'strict_quality_equal':True,
        'scope':'Full actual native outputs unchanged despite additional qualification replays. Qualification wall is diagnostic overhead and is not compared as performance.'}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
