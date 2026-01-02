from typing import List, Tuple, Dict
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidate_passages: List[Dict], topk: int = 100) -> List[Tuple[str, float]]:
        # candidate_passages: list of dicts {'passage_id':..., 'text':...}
        pairs = [(query, p.get("text") or p.get("body")) for p in candidate_passages]
        scores = self.model.predict([p[1] for p in pairs], convert_to_numpy=True)
        scored = [(candidate_passages[i].get("passage_id") or str(i), float(scores[i])) for i in range(len(scores))]
        scored = sorted(scored, key=lambda x: x[1], reverse=True)[:topk]
        return scored
