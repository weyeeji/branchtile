"""Recover source IDs without modifying any frozen benchmark manifest.

The official JSONL omits filenames. Reconstruct each JSON file's serialization
and match its Git blob SHA against the pinned repository tree, then independently
match the entire transcript SHA256 to the frozen document.
"""
import collections
import hashlib
import json
import re
from pathlib import Path
import sys
import urllib.request


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/qmsum_identity'
SOURCES={
    'qmsum_paper':'https://aclanthology.org/2021.naacl-main.472.pdf',
    'ami_ids':'https://groups.inf.ed.ac.uk/ami/corpus/meetingids.shtml',
    'icsi_project_series':'https://aclanthology.org/2007.sigdial-1.33.pdf',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob_sha(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()


def serialize_candidates(row):
    for ensure_ascii in (True,False):
        for newline in ('','\n'):
            yield (json.dumps(row,indent=4,ensure_ascii=ensure_ascii)+newline).encode()


def identity(meeting_id,domain):
    record={'meeting_id':meeting_id,'qmsum_domain':domain,
            'source_family':{'Product':'AMI','Academic':'ICSI','Committee':'Committee'}.get(domain,'unknown'),
            'higher_source_cluster':'unknown','cluster_evidence':'unknown',
            'calendar_session_identity':'unknown'}
    if domain=='Product' and re.fullmatch(r'[IET]S\d{4}[a-d]',meeting_id):
        record.update(higher_source_cluster='AMI:'+meeting_id[:-1],
            cluster_evidence='Verified AMI naming rule: a/b/c/d are four meetings in the same design trial.',
            cluster_source=SOURCES['ami_ids'],scenario_submeeting=meeting_id[-1])
    elif domain=='Academic' and re.fullmatch(r'(Bed|Bmr|Bro)\d+',meeting_id):
        record.update(higher_source_cluster='ICSI:'+meeting_id[:3],
            cluster_evidence='Published corpus analysis identifies Bed/Bmr/Bro as longitudinal project groups; some participants overlap between groups.',
            cluster_source=SOURCES['icsi_project_series'])
    elif domain=='Committee' and re.fullmatch(r'(covid|education)_\d+',meeting_id):
        record.update(higher_source_cluster='Committee:'+meeting_id.rsplit('_',1)[0],
            cluster_evidence='Conservative filename-series grouping from pinned official QMSum repository. It is not a verified calendar session or participant identity.',
            committee_institution_identity='unknown')
    return record


def main():
    assert sys.prefix!=sys.base_prefix
    revision=json.loads((ROOT/'data/raw/qmsum/source_commit.json').read_text())['sha']
    tree_url=f'https://api.github.com/repos/Yale-LILY/QMSum/git/trees/{revision}?recursive=1'
    with urllib.request.urlopen(tree_url,timeout=30) as response:
        tree_bytes=response.read()
    tree=json.loads(tree_bytes)
    assert not tree.get('truncated'), 'Incomplete tree cannot establish unique source matching'
    by_blob=collections.defaultdict(list)
    for entry in tree['tree']:
        if entry['type']=='blob' and entry['path'].startswith('data/') and entry['path'].endswith('.json'):
            by_blob[entry['sha']].append(entry['path'])
    OUT.mkdir(parents=True,exist_ok=True)
    tree_subset={key:tree[key] for key in ['sha','url','truncated']}
    tree_subset['tree']=[e for e in tree['tree'] if e['type']=='blob' and e['path'].startswith('data/') and e['path'].endswith('.json')]
    (OUT/'official_data_tree.json').write_text(json.dumps(tree_subset,indent=2)+'\n')
    reports=[];all_rows=[];frozen_hashes={}
    for split,name in [('train','qmsum_train_exploratory8'),('test','qmsum_official_test35')]:
        manifest_path=ROOT/f'data/manifests/{name}.json'
        frozen_hashes[str(manifest_path.relative_to(ROOT))]=sha(manifest_path.read_bytes())
        manifest=json.loads(manifest_path.read_text())
        raw_path=ROOT/f'data/raw/qmsum/{split}.jsonl'
        raw_rows=[json.loads(line) for line in raw_path.read_text().splitlines()]
        rows=[]
        for gi,group in enumerate(manifest['groups']):
            row=raw_rows[group['source_row']]
            transcript='\n'.join(f'{u["speaker"]}: {u["content"]}' for u in row['meeting_transcripts'])
            assert sha(transcript.encode())==group['document_sha256']
            assert sha((ROOT/group['document_path']).read_bytes())==group['document_sha256']
            matches={}
            for serialized in serialize_candidates(row):
                blob=blob_sha(serialized)
                for path in by_blob[blob]:
                    if path.startswith(f'data/ALL/{split}/'):
                        matches[path]=(blob,sha(serialized))
            record={'manifest':str(manifest_path.relative_to(ROOT)),'group_index':gi,
                    'document_id':group['document_id'],'document_sha256':group['document_sha256'],
                    'document_path':group['document_path'],'source_row':group['source_row'],
                    'questions':len(group['questions']),'status':'unknown',
                    'meeting_id':'unknown','source_family':'unknown','higher_source_cluster':'unknown',
                    'full_transcript_hash_matches':True}
            if len(matches)==1:
                source_path,(blob,digest)=next(iter(matches.items()))
                matching_domains={p.split('/')[1] for p in by_blob[blob]
                    if re.fullmatch(r'data/(Product|Academic|Committee)/(all|train|test|val)/[^/]+\.json',p)}
                assert len(matching_domains)==1
                domain=next(iter(matching_domains))
                record.update(identity(Path(source_path).stem,domain),status='exact_git_blob_and_transcript_match',
                    official_json_path=source_path,official_git_blob_sha1=blob,
                    reconstructed_official_json_sha256=digest,
                    official_source_url=f'https://github.com/Yale-LILY/QMSum/blob/{revision}/{source_path}',
                    matching_official_domain_paths=sorted(p for p in by_blob[blob] if f'data/{domain}/' in p))
            else:
                record['unknown_reason']=f'Expected one pinned official filename match; found {len(matches)}'
            rows.append(record)
        # ALL/jsonl and the individually named JSON occasionally differ in
        # annotations/format. Resolve remaining files only by full transcript
        # equality, verifying the downloaded bytes against the pinned Git blob.
        unknown=[r for r in rows if r['status']=='unknown']
        used={r.get('official_json_path') for r in rows}
        if unknown:
            candidates=[e for e in tree['tree'] if e['type']=='blob'
                and e['path'].startswith(f'data/ALL/{split}/') and e['path'].endswith('.json') and e['path'] not in used]
            for entry in candidates:
                url=f'https://raw.githubusercontent.com/Yale-LILY/QMSum/{revision}/{entry["path"]}'
                with urllib.request.urlopen(url,timeout=30) as response:
                    payload=response.read()
                assert blob_sha(payload)==entry['sha']
                official=json.loads(payload)
                transcript='\n'.join(f'{u["speaker"]}: {u["content"]}' for u in official['meeting_transcripts'])
                matching=[r for r in unknown if r['document_sha256']==sha(transcript.encode())]
                for record in matching:
                    assert record['status']=='unknown','Ambiguous duplicate source transcript'
                    domains={p.split('/')[1] for p in by_blob[entry['sha']]
                        if re.fullmatch(r'data/(Product|Academic|Committee)/(all|train|test|val)/[^/]+\.json',p)}
                    assert len(domains)==1
                    domain=next(iter(domains))
                    raw_row=raw_rows[record['source_row']]
                    record.update(identity(Path(entry['path']).stem,domain),
                        status='exact_downloaded_git_blob_and_full_transcript_match',
                        official_json_path=entry['path'],official_git_blob_sha1=entry['sha'],
                        downloaded_official_json_sha256=sha(payload),official_raw_url=url,
                        official_source_url=f'https://github.com/Yale-LILY/QMSum/blob/{revision}/{entry["path"]}',
                        raw_jsonl_object_equals_named_json=raw_row==official,
                        annotation_lists_equal_jsonl={key:raw_row.get(key)==official.get(key) for key in ['topic_list','general_query_list','specific_query_list']},
                        matching_official_domain_paths=sorted(p for p in by_blob[entry['sha']] if f'data/{domain}/' in p))
                    record.pop('unknown_reason')
        report={'schema':1,'scope':'Independent identity overlay only; original frozen manifest and question selection unchanged.',
                'original_manifest':str(manifest_path.relative_to(ROOT)),
                'original_manifest_sha256':frozen_hashes[str(manifest_path.relative_to(ROOT))],
                'official_repo':'Yale-LILY/QMSum','official_revision':revision,
                'raw_jsonl_sha256':sha(raw_path.read_bytes()),'groups':rows}
        output=OUT/f'{name}_identity.json'
        output.write_text(json.dumps(report,indent=2)+'\n')
        reports.append({'path':str(output.relative_to(ROOT)),'sha256':sha(output.read_bytes()),
                        'documents':len(rows),'exact_matches':sum(r['status'].startswith('exact_') for r in rows),
                        'source_family_counts':dict(collections.Counter(r['source_family'] for r in rows)),
                        'higher_source_clusters':dict(collections.Counter(r['higher_source_cluster'] for r in rows))})
        all_rows.extend(rows)
    for path,digest in frozen_hashes.items():
        assert sha((ROOT/path).read_bytes())==digest,'Frozen manifest changed during audit'
    clusters=collections.defaultdict(list)
    for row in all_rows:
        clusters[row['higher_source_cluster']].append({'manifest':row['manifest'],'meeting_id':row['meeting_id'],'group_index':row['group_index']})
    audit={'status':'completed','official_revision':revision,'tree_url':tree_url,'downloaded_tree_sha256':sha(tree_bytes),
        'method':'Exact official Git blob SHA1 from reconstructed indent=4 JSON plus independent full speaker:content transcript SHA256 match, pinned tree domain and filename; no model outputs used.',
        'sources':SOURCES,'frozen_manifests_unchanged':frozen_hashes,'reports':reports,
        'cross_split_higher_source_clusters':{k:v for k,v in clusters.items() if len({x['manifest'] for x in v})>1},
        'unknown_documents':[r['document_id'] for r in all_rows if r['status']=='unknown'],
        'limitations':['Source family and project-series identities do not prove statistical independence; ICSI projects can share participants.',
          'Committee filename prefixes provide conservative coarse groups; exact calendar session and institution are explicitly unknown here.',
          'AMI a/b/c/d submeetings belong to one design trial. Counting them as independent source clusters would inflate the effective sample size.',
          'Existing official train/test meeting split is preserved; higher-source overlap is reported rather than repaired by changing frozen selection.']}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'reports':reports,'cross_split_higher_source_clusters':list(audit['cross_split_higher_source_clusters']),'unknown':audit['unknown_documents']},indent=2))


if __name__=='__main__':
    main()
