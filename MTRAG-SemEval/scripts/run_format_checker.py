"""Lightweight format checker wrapper for mt-rag-benchmark submission files.

This script performs a simple JSONL structural check and, if the official format_checker
script is available under `mt-rag-benchmark/scripts/evaluation/format_checker.py`, it will
attempt to run it (best-effort). Use this to validate your `predictions.jsonl` before submission.
"""
import argparse
import json
import os
import subprocess


def basic_check(path):
    required_fields = None
    ok = True
    with open(path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"Line {i}: invalid JSON: {e}")
                ok = False
                continue
            if 'id' not in obj:
                print(f"Line {i}: missing 'id' field")
                ok = False
    return ok


def try_official_checker(predictions_path):
    checker = os.path.join('mt-rag-benchmark', 'scripts', 'evaluation', 'format_checker.py')
    if os.path.exists(checker):
        try:
            subprocess.run(['python', checker, predictions_path], check=True)
        except subprocess.CalledProcessError as e:
            print('Official checker returned non-zero exit code:', e)
        except Exception as e:
            print('Could not run official checker:', e)
    else:
        print('Official format_checker not found at', checker)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions')
    args = parser.parse_args()
    p = args.predictions
    if not os.path.exists(p):
        print('Predictions file not found:', p)
        return
    print('Running basic JSONL checks...')
    ok = basic_check(p)
    print('Basic check:', 'PASSED' if ok else 'FAILED')
    print('Attempting to run official checker (best-effort)...')
    try_official_checker(p)


if __name__ == '__main__':
    main()
