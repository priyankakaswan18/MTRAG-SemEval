import os
import random
import torch
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline


class Rewriter:
    def __init__(self, model_name: str = None, device: Optional[int] = None, seed: int = 42):
        self.model_name = model_name or os.environ.get("MT_RAG_REWRITE_MODEL", "OpenAssistant/replit-1b-instruct")
        self.device = 0 if torch.cuda.is_available() and device is None else (device or -1)
        random.seed(seed)
        torch.manual_seed(seed)
        # Choose pipeline type depending on model class (causal vs seq2seq)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            # Try to load as causal LM first
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
                if self.device >= 0:
                    self.model.to(self.device)
                self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device)
            except Exception:
                # fallback: try seq2seq
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self.generator = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device)
        except Exception:
            # final fallback to a small T5 model
            self.model_name = "google/flan-t5-small"
            self.generator = pipeline("text2text-generation", model=self.model_name, device=self.device)

    def make_prompt(self, history: str, current_utterance: str) -> str:
        prompt = (
            "Rewrite the following conversational question into a concise, standalone search query. "
            "Keep it short and focused, include necessary context from the conversation history.\n\n"
            "Conversation history:\n" + history + "\n\n"
            "Current turn:\n" + current_utterance + "\n\n"
            "Standalone query:")
        return prompt

    def rewrite(self, history: str, current_utterance: str, max_new_tokens: int = 64) -> str:
        prompt = self.make_prompt(history, current_utterance)
        out = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        if isinstance(out, list):
            text = out[0]["generated_text"] if "generated_text" in out[0] else out[0]["generated_text"] if "generated_text" in out[0] else out[0].get("text", "")
        else:
            text = str(out)
        return text.strip()


def simple_rewrite(history: str, current_utterance: str) -> str:
    # lightweight fallback: concatenate last 2 turns
    context = history.strip().split("\n")[-4:]
    return " ".join(context + [current_utterance]).strip()
