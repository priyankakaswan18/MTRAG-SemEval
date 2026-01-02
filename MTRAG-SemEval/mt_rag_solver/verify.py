from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class Verifier:
    def __init__(self, model_name: str = "microsoft/deberta-v3-small-mnli", device: int = -1):
        self.device = 0 if torch.cuda.is_available() and device == -1 else device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if self.device >= 0:
            self.model.to(self.device)

    def is_entailed(self, premise: str, hypothesis: str, threshold: float = 0.7) -> bool:
        # returns True if model predicts entailment probability >= threshold
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        if self.device >= 0:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        # label mapping typically: 0: contradiction, 1: neutral, 2: entailment
        entail_prob = float(probs[2])
        return entail_prob >= threshold

    def verify_answer(self, answer: str, passages: List[str], threshold: float = 0.7) -> bool:
        # split answer into sentences
        sents = [s.strip() for s in answer.split('.') if s.strip()]
        for sent in sents:
            entailed = False
            for p in passages:
                if self.is_entailed(p, sent, threshold=threshold):
                    entailed = True
                    break
            if not entailed:
                return False
        return True
