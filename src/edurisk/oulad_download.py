"""Reproducible acquisition of the public CC BY 4.0 OULAD archive from UCI."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

URL = 'https://archive.ics.uci.edu/static/public/349/open%2Buniversity%2Blearning%2Banalytics%2Bdataset.zip'
CITATION = 'Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2015). Open University Learning Analytics dataset. UCI. https://doi.org/10.24432/C5KK69'


def download(destination):
    from .oulad_external import REQUIRED_FILES, validate_files
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in (*REQUIRED_FILES, 'edurisk-source.json')):
        raise FileExistsError('Destination contains data or provenance; choose an empty directory')
    with tempfile.TemporaryDirectory(prefix='oulad-download-', dir=destination.parent) as temp:
        stage = Path(temp)
        archive = stage / 'source.zip'
        digest = hashlib.sha256()
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open('wb') as output:
            if response.url != URL:
                raise ValueError('Unexpected download redirect')
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > 150 * 1024 * 1024:
                    raise ValueError('Archive exceeds expected size limit')
                digest.update(chunk)
                output.write(chunk)
        with zipfile.ZipFile(archive) as bundle:
            # Extract exact filenames only; no arbitrary paths from the ZIP.
            for name in REQUIRED_FILES:
                matches = [entry for entry in bundle.infolist() if entry.filename == name]
                if len(matches) != 1 or matches[0].file_size > 600 * 1024 * 1024:
                    raise ValueError(f'Unexpected archive entry: {name}')
                with bundle.open(matches[0]) as src, (stage / name).open('wb') as dst:
                    shutil.copyfileobj(src, dst)
        files = validate_files(stage)
        receipt = {'kind':'uci_download', 'source_url':URL,
                   'retrieved_at':datetime.now(timezone.utc).isoformat(),
                   'archive_sha256':digest.hexdigest(), 'archive_bytes':count,
                   'files':files, 'license':'CC BY 4.0', 'citation':CITATION}
        destination.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_FILES:
            # Exclusive creation prevents replacing user-supplied data.
            with (stage / name).open('rb') as src, (destination / name).open('xb') as dst:
                shutil.copyfileobj(src, dst)
        with (destination / 'edurisk-source.json').open('x', encoding='utf-8') as f:
            json.dump(receipt, f, ensure_ascii=False, indent=2)
        return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='data/oulad/raw')
    args = parser.parse_args()
    result = download(args.out)
    print(json.dumps({'source':result['source_url'], 'bytes':result['archive_bytes'],
                      'files':len(result['files']), 'archive_sha256':result['archive_sha256']}))
