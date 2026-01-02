import os
import random
import torch
from typing import List, Dict, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


class Generator:
    def __init__(self, model_name: Optional[str] = None, device: Optional[int] = None, seed: int = 42):
        self.model_name = model_name or os.environ.get("MT_RAG_GEN_MODEL", "OpenAssistant/replit-1b-instruct")
        self.device = 0 if torch.cuda.is_available() and device is None else (device or -1)
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
                if self.device >= 0:
                    self.model.to(self.device)
                self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device)
            except Exception:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self.generator = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device)
        except Exception:
            self.model_name = "google/flan-t5-small"
            self.generator = pipeline("text2text-generation", model=self.model_name, device=self.device)

    def make_prompt(self, query: str, passages: List[Dict], strict: bool = True) -> str:
        header = "Answer the question using ONLY the information in the following passages. If the answer cannot be derived from these passages, respond exactly: Insufficient information"
        passages_text = "\n\n".join([f"Passage {i+1}: {p.get('text') or p.get('body')}" for i, p in enumerate(passages)])
        prompt = f"{header}\n\nQuestion: {query}\n\n{passages_text}\n\nAnswer:"
        return prompt

    def generate(self, query: str, passages: List[Dict], max_new_tokens: int = 128) -> str:
        prompt = self.make_prompt(query, passages)
        out = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        if isinstance(out, list):
            text = out[0].get("generated_text") or out[0].get("summary_text") or out[0].get("text", "")
        else:
            text = str(out)
        return text.strip()
