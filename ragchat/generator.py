import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re, unicodedata

def _clean_text(s: str) -> str:
    """Normalize and strip invalid Unicode for GPT-2 input."""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[\ud800-\udfff]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

class Generator:
    """Arabic GPT-2 text generator"""
    def __init__(self, model_name: str, max_new_tokens: int = 64, temperature: float = 0.7, top_p: float = 0.95):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt: str) -> str:
        if isinstance(prompt, (list, tuple)):
            prompt = " ".join(str(x) for x in prompt)
        elif not isinstance(prompt, str):
            prompt = str(prompt)
        prompt = _clean_text(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=self.max_new_tokens,
                temperature=0.6,
                top_p=0.9,
                no_repeat_ngram_size=3,
                repetition_penalty=1.4,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # return only the continuation
        return full[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
