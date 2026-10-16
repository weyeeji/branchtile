"""Compress immutable, explicitly selected experiment evidence without deleting originals."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', required=True, help='JSON list of completed regular files')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    destination = Path(a.output)
    assert not destination.exists()
    destination.mkdir(parents=True)
    names = sorted(json.loads(Path(a.selection).read_text()))
    assert names and len(names) == len(set(names))
    records = []
    archive = destination / 'evidence.tar.gz'
    with archive.open('wb') as handle, gzip.GzipFile(fileobj=handle, mode='wb', mtime=0, filename='') as zipped:
        with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as tar:
            for name in names:
                path = Path(name)
                assert not path.is_absolute() and '..' not in path.parts
                assert path.is_file() and not path.is_symlink()
                raw = path.read_bytes()
                record = {'source_path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(raw))
                assert path.read_bytes() == raw, f'File changed during archive: {name}'
                records.append(record)
    with tarfile.open(archive, 'r:gz') as tar:
        assert tar.getnames() == names
        for record in records:
            assert hashlib.sha256(tar.extractfile(record['source_path']).read()).hexdigest() == record['sha256']
    receipt = {'status': 'roundtrip_verified', 'selection': a.selection,
               'selection_sha256': hashlib.sha256(Path(a.selection).read_bytes()).hexdigest(),
               'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
               'archive_bytes': archive.stat().st_size, 'files': records,
               'scope': 'Lossless archive of explicitly selected completed evidence. Originals retained. This script does not infer experiment success.'}
    (destination / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'output': str(destination), 'files': len(records), 'archive_bytes': archive.stat().st_size}))


if __name__ == '__main__':
    main()
