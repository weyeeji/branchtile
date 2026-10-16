"""Download immutable official model revisions, retaining provenance."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
from huggingface_hub import HfApi, snapshot_download

p = argparse.ArgumentParser()
p.add_argument("repo")
p.add_argument("--output", required=True)
a = p.parse_args()
assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
info = HfApi().model_info(a.repo)
record = {"repo": a.repo, "revision": info.sha,
          "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "path": str(Path(a.output).resolve()), "complete": False}
meta = Path("results") / (a.repo.split("/")[-1] + "_download.json")
meta.parent.mkdir(exist_ok=True)
meta.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
snapshot_download(a.repo, revision=info.sha, local_dir=a.output, max_workers=4)
record.update(complete=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
meta.write_text(json.dumps(record, indent=2) + "\n")
print("Completed " + a.repo, flush=True)
