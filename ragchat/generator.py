# ragchat/generator.py
import re
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from .utils import clean_text

SEP = "\n- "

def _arabic_only(s: str) -> str:
    s = re.sub(r"[^ء-ي0-9\s.,؟!:؛\-\(\)\"']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

class Generator:
    """
    Hybrid generator that supports AraT5 (seq2seq) and AraGPT2 (causal).
    Defaults here are tuned for GPT-2 Arabic short answers.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 40,
        temperature: float = 0.3,
        top_p: float = 0.9,
        encoder_max_len: int = 512,  # for T5 encoders
        prompt_max_len: int = 640,   # for GPT-2 prompt budget
    ):
        self.model_name = model_name
        self.cfg = AutoConfig.from_pretrained(model_name)
        self.model_type = (self.cfg.model_type or "").lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_type == "t5":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        else:
            # default path: GPT-2
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.encoder_max_len = encoder_max_len
        self.prompt_max_len = prompt_max_len

    #  T5 path 
    def _t5_build(self, question: str, contexts: list[str] | None) -> str:
        q = clean_text(question)
        ctxs = [clean_text(c) for c in (contexts or []) if c and c.strip()]
        base = f"question: {q}"
        if not ctxs:
            return base

        prompt = base + " context: "
        max_len = self.encoder_max_len - 16
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids[0]
        for c in ctxs:
            candidate = prompt + (c if prompt.endswith(": ") else " </s> " + c)
            cand_len = len(self.tokenizer(candidate, return_tensors="pt", add_special_tokens=True).input_ids[0])
            if cand_len <= max_len:
                prompt = candidate
            else:
                break
        return prompt

    def _t5_generate(self, input_text: str) -> str:
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.encoder_max_len,
        ).to(self.device)

        do_sample = self.temperature and self.temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            early_stopping=True,
            no_repeat_ngram_size=3,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(dict(do_sample=True, temperature=self.temperature, top_p=self.top_p))
        else:
            gen_kwargs.update(dict(do_sample=False, num_beams=4, length_penalty=0.6))

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        ans = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return _arabic_only(clean_text(ans))

    #  GPT-2 path
    def _gpt2_build(self, question: str, contexts: list[str] | None) -> str:
        """Build AraGPT2 prompt with few-shot QA examples."""
        q = clean_text(question)
        ctxs = [clean_text(c) for c in (contexts or []) if c and c.strip()]

        # Simple few-shot instruction block
        fewshot = (
            "أجب بإيجاز وبشكل دقيق باللغة العربية فقط.\n"
            "الأمثلة:\n"
            "السؤال: من هو نجيب محفوظ؟\n"
            "الإجابة: هو كاتب وروائي مصري فاز بجائزة نوبل في الأدب.\n"
            "السؤال: ما هي عاصمة السعودية؟\n"
            "الإجابة: الرياض.\n"
            "السؤال: من اكتشف الكهرباء؟\n"
            "الإجابة: بنجامين فرانكلين.\n\n"
        )

        ctx_block = ""
        for c in ctxs:
            candidate = (ctx_block + (SEP if ctx_block else "") + c)
            skeleton = fewshot + f"السؤال: {q}\nالسياق:{candidate}\nالإجابة:"
            if len(self.tokenizer.encode(skeleton, add_special_tokens=False)) <= self.prompt_max_len:
                ctx_block = candidate
            else:
                break

        prompt = fewshot + f"السؤال: {q}\nالسياق:{ctx_block}\nالإجابة:"
        return prompt

    def _extract_after_answer_tag(self, text: str) -> str:
        # Keep only what comes after 'الإجابة:' (if present), and cut at the first blank line
        m = re.split(r"الإجابة:\s*", text, maxsplit=1)
        ans = m[-1] if len(m) > 1 else text
        ans = ans.strip()
        # cut when the model starts echoing a new prompt line
        ans = re.split(r"\n\s*(السؤال|context|context:|سياق|السياق)\b", ans)[0]
        # limit to first sentence or 25 tokens
        ans = re.split(r"[.!؟]\s", ans, maxsplit=1)[0]
        tokens = ans.split()
        if len(tokens) > 25:
            ans = " ".join(tokens[:25])
        return ans.strip()

    def _gpt2_generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)

        # For GPT-2, a tiny bit of sampling usually beats greedy/beam on coherence
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True if (self.temperature and self.temperature > 0.0) else False,
                temperature=max(self.temperature, 0.2),
                top_p=self.top_p,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # We only want the completion, not the whole prompt
        completion = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded
        ans = self._extract_after_answer_tag(completion)
        return _arabic_only(clean_text(ans))

    # Public
    def generate(self, question: str, contexts: list[str] | None = None) -> str:
        if self.model_type == "t5":
            input_text = self._t5_build(question, contexts)
            return self._t5_generate(input_text)
        else:
            prompt = self._gpt2_build(question, contexts)
            return self._gpt2_generate(prompt)
