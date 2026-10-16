"""Primary-source snapshot identifiers for metadata/plan prior-art review."""
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def get(url):
    with urllib.request.urlopen(url,timeout=30) as f:return f.read()

def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    remote=[]
    for repo,path in [('sgl-project/sglang','python/sglang/srt/layers/attention/flashinfer_backend.py'),
                      ('flashinfer-ai/flashinfer','include/flashinfer/attention/blackwell/plan.cuh')]:
        commits=json.loads(get(f'https://api.github.com/repos/{repo}/commits?path={path}&per_page=1'))
        c=commits[0]; sha=c['sha'];date=c['commit']['committer']['date']
        assert date<'2026-10-18', 'Source exceeds audit cutoff'
        raw=get(f'https://raw.githubusercontent.com/{repo}/{sha}/{path}')
        remote.append({'repository':repo,'path':path,'commit':sha,'commit_date':date,
                       'source_sha256':hashlib.sha256(raw).hexdigest(),
                       'url':f'https://github.com/{repo}/blob/{sha}/{path}'})
    local=[]
    for name in ('vllm/v1/attention/backends/flashinfer.py','flashinfer/decode.py','flashinfer/data/include/flashinfer/attention/blackwell/plan.cuh'):
        p=Path(sys.prefix)/'lib/python3.12/site-packages'/name
        local.append({'path':str(p),'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None})
    result={'status':'CPU_primary_source_metadata_prior_audit','cutoff':'2026-10-17','remote':remote,'local':local,
            'additional_primary_sources':[
                'https://github.com/orgs/flashinfer-ai/discussions/1743',
                'https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf',
                'https://vllm-project.github.io/2026/03/04/vllm-triton-backend-deep-dive.html',
                'https://arxiv.org/html/2606.26666v1'],
            'limitations':['Source inspection only, no installed backend equivalence or measured performance implied.',
                           'The Blackwell GPU planner is not automatically a drop-in FA2/SM120 implementation.']}
    out=ROOT/'results/platform/device_metadata_prior_20261017.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(remote))
if __name__=='__main__':main()
