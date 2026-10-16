"""Audit native backend identity and cold repeated natural trajectories."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args()
    root=Path(args.directory);source=root/'summary.json';data=json.loads(source.read_text())
    assert data['native_official_backend_control'] and data['status']=='completed'
    reference=data['runs'][0];pairs=[]
    def sequence(rank,run):
        graph=json.loads((root/f'rank{rank}/{run["phase"]}_graph.json').read_text())
        return [(r['batch'],r['questions'],r['positions']) for r in graph['replays']]
    for run in data['runs'][1:]:
        pairs.append({'phase':run['phase'],'reference':reference['phase'],
            'same_output_tokens_by_question':[a['tokens']==b['tokens'] for a,b in zip(reference['answers'],run['answers'])],
            'same_apc_hits_by_question':[a['num_cached_tokens']==b['num_cached_tokens'] for a,b in zip(reference['answers'],run['answers'])],
            'same_step_sequences_by_rank':[sequence(rank,reference)==sequence(rank,run) for rank in (0,1)]})
    identities=[]
    for rank,configured in enumerate(data['configured_workers']):
        meta=json.loads((root/f'rank{rank}/{reference["phase"]}.json').read_text())
        identities.append({'rank':rank,'all_48_layers_requested_backend':len(configured['layers'])==48 and all(l['backend']==data['methods'][0] for l in configured['layers']),
            'layer_impl':configured['layers'][0]['impl'],'runtime_metadata_types':meta['metadata_types'],
            'captured_graph_descriptors':configured['graph_descriptors'],'deployment':configured['deployment']})
    passed=bool(pairs) and all(all(p[k]) for p in pairs for k in ('same_output_tokens_by_question','same_apc_hits_by_question','same_step_sequences_by_rank'))
    passed &= all(i['all_48_layers_requested_backend'] for i in identities)
    result={'status':'natural_AA_exact' if passed else 'natural_AA_not_exact','source':str(source),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'pairs':pairs,'runtime_identity':identities,
        'scope':'Official native backend without candidate attention replacement; exact repeatability is local to this engine/document/sampling/admission configuration.',
        'limitations':['No cross-backend token equivalence is claimed.','One exploratory document is not formal quality noninferiority or a task speedup experiment.']}
    (root/'native_AA_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'pairs':len(pairs)}))


if __name__=='__main__':main()
