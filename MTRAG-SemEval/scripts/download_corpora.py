"""Download and extract corpora for mt_rag_solver.

Usage:
  python scripts/download_corpora.py --config mt_rag_solver/config.yaml

The script reads `data_sources.corpora_archive_url` from the config and downloads
the archive into `mt-rag-benchmark/` then extracts it. It supports .zip and .tar(.gz/.bz2).
If no URL is configured, the script exits with instructions.
"""
import argparse
import os
import sys
import shutil
import tempfile
import urllib.request
from pathlib import Path

import yaml


def download_file(url, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {out_path}")
    with urllib.request.urlopen(url) as resp, open(out_path, 'wb') as out:
        shutil.copyfileobj(resp, out)


def extract_archive(archive_path: Path, dest_dir: Path):
    import zipfile
    import tarfile

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as z:
            z.extractall(dest_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r:*') as t:
            t.extractall(dest_dir)
    else:
        raise RuntimeError("Unsupported archive format: " + str(archive_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mt_rag_solver/config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        sys.exit(1)
    cfg = yaml.safe_load(cfg_path.read_text())
    url = cfg.get('data_sources', {}).get('corpora_archive_url', '')
    # Default to official GitHub archive if not configured
    if not url:
        default_url = 'https://github.com/IBM/mt-rag-benchmark/archive/refs/heads/main.zip'
        print(f"No `data_sources.corpora_archive_url` configured; defaulting to {default_url}")
        url = default_url

    base = Path(__file__).resolve().parents[1]
    target_dir = base / 'mt-rag-benchmark'
    tmpdir = Path(tempfile.mkdtemp(prefix='corpora_dl_'))
    try:
        archive_path = tmpdir / 'corpora_archive'
        download_file(url, archive_path)
        print(f"Extracting archive to {target_dir}...")
        extract_archive(archive_path, target_dir)
        print("Extraction complete. Verify that `mt-rag-benchmark/corpora/` exists.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
