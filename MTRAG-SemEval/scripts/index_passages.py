"""Build and save a FAISS index and embeddings for passage collections.

Usage:
    python scripts/index_passages.py --input path/to/passages.jsonl --out-dir mt-rag-solver/data --model intfloat/e5-large-v2
"""
import argparse
import os
from mt_rag_solver.io_utils import read_jsonl
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="intfloat/e5-large-v2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    passages = list(read_jsonl(args.input))
    texts = [p.get('text') or p.get('body') or p.get('passage_text') or '' for p in passages]
    ids = [p.get('passage_id') or p.get('id') or str(i) for i, p in enumerate(passages)]

    print(f"Encoding {len(texts)} passages with {args.model}...")
    embedder = SentenceTransformer(args.model)
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = os.path.join(args.out_dir, 'faiss.index')
    emb_path = os.path.join(args.out_dir, 'embeddings.npy')
    map_path = os.path.join(args.out_dir, 'id_mapping.json')

    print(f"Saving index to {index_path} and embeddings to {emb_path}")
    faiss.write_index(index, index_path)
    np.save(emb_path, embeddings)
    with open(map_path, 'w', encoding='utf8') as f:
        json.dump(ids, f, ensure_ascii=False)

    print("Indexing complete.")


if __name__ == '__main__':
    main()
