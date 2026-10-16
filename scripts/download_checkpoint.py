"""Download immutable official model revisions, retaining provenance."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import concurrent.futures
from huggingface_hub import HfApi, snapshot_download, hf_hub_download, get_hf_file_metadata, hf_hub_url

p = argparse.ArgumentParser()
p.add_argument("repo")
p.add_argument("--revision", help="Immutable HF commit for reproducible resume")
p.add_argument("--output", required=True)
p.add_argument("--aria2", action="store_true")
p.add_argument("--modelscope-mirror", action="store_true",
               help="Download payload from Qwen's public mirror; verify HF revision SHA256")
a = p.parse_args()
assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
info = HfApi().model_info(a.repo, revision=a.revision)
record = {"repo": a.repo, "revision": info.sha,
          "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "path": str(Path(a.output).resolve()), "complete": False,
          "payload_transport": "modelscope_official_mirror_hf_sha256_verified" if a.modelscope_mirror else "huggingface"}
meta = Path("results") / (a.repo.split("/")[-1] + "_download.json")
meta.parent.mkdir(exist_ok=True)
meta.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
if not a.aria2:
    snapshot_download(a.repo, revision=info.sha, local_dir=a.output, max_workers=4)
else:
    dest = Path(a.output)
    dest.mkdir(parents=True, exist_ok=True)
    large = [s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors")]
    for s in info.siblings:
        if s.rfilename not in large:
            hf_hub_download(a.repo, s.rfilename, revision=info.sha, local_dir=a.output)
    def fetch(name):
        url = hf_hub_url(a.repo, name, revision=info.sha)
        metadata = get_hf_file_metadata(url)
        if a.modelscope_mirror:
            url = f"https://modelscope.cn/models/{a.repo}/resolve/master/{name}"
        final = dest / name
        incomplete = dest / ".cache/huggingface/download" / (name + "." + metadata.etag + ".incomplete")
        if incomplete.exists() and not final.exists():
            incomplete.rename(final)
        cmd = ["aria2c", "--continue=true", "--allow-overwrite=true", "--auto-file-renaming=false",
               "--file-allocation=none", "--max-connection-per-server=16", "--split=16",
               "--min-split-size=16M", "--summary-interval=30", "--console-log-level=warn",
               "--max-tries=8", "--retry-wait=3", "--dir="+str(dest), "--out="+name, url]
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy and not a.modelscope_mirror:
            cmd.insert(1, "--all-proxy=" + proxy)
        env = os.environ.copy()
        env.pop("all_proxy", None)
        env.pop("ALL_PROXY", None)
        if a.modelscope_mirror:
            for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                env.pop(key, None)
        subprocess.run(cmd, check=True, env=env)
        digest = hashlib.sha256()
        with final.open("rb") as f:
            for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        assert final.stat().st_size == metadata.size, "File length mismatch"
        assert digest.hexdigest() == metadata.etag, "LFS SHA256 mismatch"
        return {"file": name, "bytes": metadata.size, "sha256": digest.hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        record["verified_shards"] = list(pool.map(fetch, large))
record.update(complete=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
meta.write_text(json.dumps(record, indent=2) + "\n")
print("Completed " + a.repo, flush=True)
