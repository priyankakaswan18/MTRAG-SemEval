"""Download and extract the full mt-rag-benchmark repository archive from GitHub.

Usage:
  python scripts/download_benchmark.py --out-dir mt-rag-benchmark

This will download the official IBM/mt-rag-benchmark main branch archive and extract
its contents into the specified output directory.
"""
import argparse
import shutil
import tempfile
import urllib.request
from pathlib import Path


def download_and_extract(url: str, dest: Path):
    import zipfile
    import tarfile

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='mt_rag_benchmark_')
    try:
        archive_path = Path(tmp) / 'benchmark_archive'
        print(f"Downloading {url} -> {archive_path}")
        with urllib.request.urlopen(url) as resp, open(archive_path, 'wb') as out:
            shutil.copyfileobj(resp, out)

        # Extract
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as z:
                z.extractall(dest.parent)
            # GitHub zip contains a top-level folder like mt-rag-benchmark-main; move its contents
            extracted_root = next(dest.parent.glob('mt-rag-benchmark-*'))
            if dest.exists():
                shutil.rmtree(dest)
            extracted_root.rename(dest)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as t:
                t.extractall(dest.parent)
            extracted_root = next(dest.parent.glob('mt-rag-benchmark-*'))
            if dest.exists():
                shutil.rmtree(dest)
            extracted_root.rename(dest)
        else:
            raise RuntimeError('Unsupported archive format for ' + str(archive_path))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='mt-rag-benchmark')
    parser.add_argument('--url', default='https://github.com/IBM/mt-rag-benchmark/archive/refs/heads/main.zip')
    args = parser.parse_args()
    dest = Path(args.out_dir).resolve()
    download_and_extract(args.url, dest)
    print(f'Extracted benchmark to {dest}')


if __name__ == '__main__':
    main()
