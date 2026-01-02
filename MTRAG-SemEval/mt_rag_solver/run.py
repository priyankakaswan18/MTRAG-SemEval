import argparse
import os
import random
import yaml
from mt_rag_solver.io_utils import read_jsonl, write_jsonl
from mt_rag_solver.rewrite import Rewriter, simple_rewrite
from mt_rag_solver.retrieve import Retriever
from mt_rag_solver.rerank import Reranker
from mt_rag_solver.generate import Generator
from mt_rag_solver.verify import Verifier


def set_seed(seed: int):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", choices=["A", "B", "C"], required=True)
    parser.add_argument("--setting", choices=["reference", "reference+RAG", "RAG"], required=True)
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--input", type=str, default=None, help="Optional: override default input JSONL file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get('runtime', {}).get('seed', 42))

    # initialize modules
    # Prefer local model paths if they exist, otherwise use the configured id.
    def pick_model(name):
        val = cfg['models'].get(name)
        if val and isinstance(val, str) and os.path.exists(val):
            return val
        return val

    rewriter = Rewriter(model_name=pick_model('rewrite'))
    generator = Generator(model_name=pick_model('generator'))
    reranker = Reranker(model_name=pick_model('reranker'))
    verifier = Verifier(model_name=pick_model('nli'))

    # input paths: follow benchmark structure (assume mt-rag-benchmark/retrieval_tasks/<domain>/<domain>_lastturn.jsonl)
    base = os.path.join(os.path.dirname(__file__), '..', 'mt-rag-benchmark')
    if args.input:
        input_path = args.input
    else:
        input_path = os.path.join(base, 'retrieval_tasks', args.domain, f"{args.domain}_{'lastturn' if args.subtask in ['A'] else 'questions'}.jsonl")
        if not os.path.exists(input_path):
            # fallback to generation tasks
            gen_path = os.path.join(base, 'generation_tasks', f"{args.setting}.jsonl")
            if os.path.exists(gen_path):
                input_path = gen_path
            else:
                # try sample data folder
                sample_ac = os.path.join(base, 'mtrageval', 'sample_data', 'retrieval_taskac_input.jsonl')
                sample_b = os.path.join(base, 'mtrageval', 'sample_data', 'retrieval_taskb_input.jsonl')
                if os.path.exists(sample_b):
                    input_path = sample_b
                elif os.path.exists(sample_ac):
                    input_path = sample_ac
                else:
                    raise FileNotFoundError(f"Could not locate input for domain {args.domain}. Please pass --input <path>")

    items = list(read_jsonl(input_path))

    # For retrieval, we need a passage pool. We'll search corpora/passage_level/*.jsonl if exists.
    corpora_dir = os.path.join(base, 'corpora', 'passage_level')
    passage_files = []
    if os.path.isdir(corpora_dir):
        for fn in os.listdir(corpora_dir):
            if fn.endswith('.jsonl'):
                passage_files.append(os.path.join(corpora_dir, fn))
    # fallback: look into mt-rag-benchmark/human/generation_tasks/reference.jsonl as passages (not ideal)
    passages = []
    if passage_files:
        for pf in passage_files:
            for p in read_jsonl(pf):
                passages.append(p)
    else:
        # try sample data
        sample = os.path.join(base, 'mtrageval', 'sample_data', 'retrieval_taskac_input.jsonl')
        if os.path.exists(sample):
            for p in read_jsonl(sample):
                passages.append(p)

    faiss_index_path = cfg.get('retrieval', {}).get('faiss_index_path')
    embeddings_path = cfg.get('retrieval', {}).get('embeddings_path')
    mapping_path = cfg.get('retrieval', {}).get('mapping_path')
    retriever = Retriever(passages, dense_model=cfg['models']['dense'],
                          faiss_index_path=faiss_index_path, embeddings_path=embeddings_path, mapping_path=mapping_path)

    out_items = []
    for item in items:
        item_id = item.get('id') or item.get('question_id') or item.get('qid')
        history = item.get('history', '') or item.get('dialogue', '') or ''
        current = item.get('current_turn') or item.get('question') or item.get('last_turn') or ''
        # 1. rewrite
        try:
            query = rewriter.rewrite(history, current)
        except Exception:
            query = simple_rewrite(history, current)

        # 2. retrieve
        fused = retriever.hybrid_search(query, k=cfg['retrieval']['top_k'])

        # build candidate passage dicts
        candidate_passages = []
        id_to_passage = {p.get('passage_id') or p.get('id') or str(i): p for i, p in enumerate(passages)}
        for pid, score in fused:
            p = id_to_passage.get(pid)
            if p is None:
                # try simple struct
                p = {'passage_id': pid, 'text': ''}
            candidate_passages.append(p)

        # 3. rerank
        reranked = reranker.rerank(query, candidate_passages, topk=cfg['retrieval']['top_k'])

        # produce outputs per subtask
        if args.subtask == 'A':
            # output ranked passage IDs
            passage_ids = [pid for pid, _ in reranked]
            out_items.append({'id': item_id, 'passage_ids': passage_ids})
        else:
            # select top-k for generation
            select_k = cfg['retrieval']['select_k_for_generation']
            topk = reranked[:select_k]
            top_passages = []
            for pid, sc in topk:
                p = id_to_passage.get(pid)
                if p:
                    top_passages.append({'passage_id': pid, 'text': p.get('text') or p.get('body') or ''})

            # 4. generate with grounding
            ans = generator.generate(query, top_passages)

            # 5. verify
            passages_text = [p.get('text') for p in top_passages]
            try:
                ok = verifier.verify_answer(ans, passages_text)
            except Exception:
                ok = True
            if not ok:
                ans = "Insufficient information"

            out_items.append({'id': item_id, 'answer': ans, 'passages': [p['passage_id'] for p in top_passages]})

    write_jsonl(args.out, out_items)
    print(f"Wrote {len(out_items)} predictions to {args.out}")


if __name__ == '__main__':
    main()
