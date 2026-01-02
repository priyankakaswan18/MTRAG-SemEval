import json
from pathlib import Path
from typing import List, Dict, Any


def read_jsonl(path: str):
    path = Path(path)
    with path.open("r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str, items: List[Dict[str, Any]]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_passages_from_file(path: str):
    """Load passages assumed as JSONL with fields `doc_id`, `passage_id`, `text` or similar.
    Returns list of dicts.
    """
    return list(read_jsonl(path))
