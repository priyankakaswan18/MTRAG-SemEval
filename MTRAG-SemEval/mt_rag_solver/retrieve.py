import os
import numpy as np
import faiss
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, passages: List[Dict], dense_model: str = "intfloat/e5-large-v2", seed: int = 42,
                 faiss_index_path: str = None, embeddings_path: str = None, mapping_path: str = None):
        # passages: list of dicts with 'passage_id' and 'text'
        self.passages = passages
        self.texts = [p.get("text") or p.get("passage_text") or p.get("body") for p in passages]
        self.ids = [p.get("passage_id") or p.get("id") or str(i) for i, p in enumerate(passages)]
        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        self.dense_model_name = dense_model
        self.embedder = SentenceTransformer(dense_model)

        # Try to load a prebuilt FAISS index and embeddings if provided
        if faiss_index_path and embeddings_path and mapping_path and \
           os.path.exists(faiss_index_path) and os.path.exists(embeddings_path) and os.path.exists(mapping_path):
            import numpy as _np
            import json as _json
            self.embeddings = _np.load(embeddings_path)
            with open(mapping_path, 'r', encoding='utf8') as f:
                mapping = _json.load(f)
            # mapping expected to be a list of ids aligned with embeddings
            self.ids = mapping
            dim = self.embeddings.shape[1]
            self.index = faiss.read_index(faiss_index_path)
        else:
            self.embeddings = self.embedder.encode(self.texts, convert_to_numpy=True, show_progress_bar=False)
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)

    def bm25_search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        topk = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in topk]

    def dense_search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append((self.ids[int(idx)], float(score)))
        return results

    def reciprocal_rank_fusion(self, lists: List[List[Tuple[str, float]]], k: int = 100, c: int = 60) -> List[Tuple[str, float]]:
        # lists: list of ranked lists (id, score). We only need ranks.
        scores = {}
        for lst in lists:
            for rank, (docid, _) in enumerate(lst, start=1):
                scores[docid] = scores.get(docid, 0.0) + 1.0 / (c + rank)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return ranked

    def hybrid_search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        bm25 = self.bm25_search(query, k=k)
        dense = self.dense_search(query, k=k)
        fused = self.reciprocal_rank_fusion([bm25, dense], k=k)
        return fused

    def save_index(self, faiss_index_path: str, embeddings_path: str, mapping_path: str):
        # Save FAISS index, embeddings npy, and id mapping (list)
        import numpy as _np
        import json as _json
        faiss.write_index(self.index, faiss_index_path)
        _np.save(embeddings_path, self.embeddings)
        with open(mapping_path, 'w', encoding='utf8') as f:
            _json.dump(self.ids, f, ensure_ascii=False)
