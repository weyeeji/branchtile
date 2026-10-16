"""Fetch only pinned upstream source used by the A2 numerical reference."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCES={
    'fastkvzip':('Janghyun1230/FastKVzip','e04afaa6007c58d18500d0698d2d8ae592ff22c2',[
        'prefill/attention/gate.py','prefill/attention/score.py','prefill/model/wrapper.py']),
    'kvpress':('NVIDIA/kvpress','7136c224105ba7b046a6e538742c3589698680b8',[
        'kvpress/presses/fastkvzip_press.py','kvpress/attention_patch.py']),
}


def main():
    jobs=[]
    for name,(repo,revision,files) in SOURCES.items():
        url=f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1'
        with urllib.request.urlopen(url,timeout=30) as response:tree=json.load(response)
        assert not tree.get('truncated')
        lookup={entry['path']:entry for entry in tree['tree']}
        jobs.extend((name,repo,revision,path,lookup[path]['sha']) for path in files)
    def fetch(job):
        name,repo,revision,path,blob=job
        url=f'https://raw.githubusercontent.com/{repo}/{revision}/{path}'
        with urllib.request.urlopen(url,timeout=30) as response:data=response.read()
        assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==blob
        target=Path('cache/fastkvzip')/name/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        return {'path':str(target),'url':url,'git_blob_sha1':blob,'sha256':hashlib.sha256(data).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,jobs))
    Path('cache/fastkvzip/source_files.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
