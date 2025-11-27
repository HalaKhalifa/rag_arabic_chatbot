import os
import re
from typing import List, Optional
import google.generativeai as genai
from .config import RAGSettings
from .utils import normalize_arabic_text

SEP = "\n- "  # bullet separator for contexts


def _arabic_only(s: str) -> str:
    """
    Keep Arabic letters, digits, and basic punctuation only.
    This is a *final clean-up* step to avoid weird artifacts.
    """
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"[^ء-ي0-9\s.,؟!:؛\-\(\)\"']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

class Generator:
    """
    Gemini-based answer generator for Arabic RAG.

    - Takes a question + retrieved contexts.
    - Builds a clear Arabic prompt with instructions.
    - Calls Gemini and extracts the answer robustly.
    - Cleans the final text (Arabic only, no noise).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        self.model_name = model_name or RAGSettings.gen_model
        self.api_key = api_key or RAGSettings.gemini_api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Please export it as an environment variable or set RAGSettings.gemini_api_key."
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.max_tokens = max_tokens or RAGSettings.gen_max_new_tokens
        self.temperature = temperature or RAGSettings.temperature
        self.top_p = top_p or RAGSettings.top_p

    def _format_contexts(self, contexts: Optional[List]) -> str:
        """
        Accepts:
        - list[str]
        - list[dict] with keys like 'chunk', 'context_text', 'raw_context'
        and returns a single bullet-list string.
        """
        if not contexts:
            return "لا يوجد سياق متاح."

        pieces: List[str] = []

        for c in contexts:
            if isinstance(c, str):
                txt = c
            elif isinstance(c, dict):
                txt = (
                    c.get("chunk")
                    or c.get("context_text")
                    or c.get("raw_context")
                    or ""
                )
            else:
                txt = str(c)

            txt = normalize_arabic_text(txt)
            if txt:
                pieces.append(f"- {txt}")

        if not pieces:
            return "لا يوجد سياق متاح."

        return "\n".join(pieces)

    def _build_prompt(self, question: str, contexts: Optional[List]) -> str:
        """
        Build the full Arabic prompt for Gemini.
        """
        clean_question = normalize_arabic_text(question)
        context_block = self._format_contexts(contexts)

        prompt = f"""
            أنت مساعد ذكي للإجابة عن الأسئلة باللغة العربية بالاعتماد فقط على النصوص المعطاة في قسم (السياق).

            السياق:
            {context_block}

            السؤال:
            {clean_question}

            التعليمات:
            - أجب عن السؤال السابق باللغة العربية الفصحى.
            - اعتمد فقط على المعلومات الموجودة في (السياق).
            - إذا لم تجد الإجابة في السياق، قل بوضوح: "لا أجد إجابة واضحة في النص." ولا تحاول التخمين.
            - اجعل الإجابة مختصرة وواضحة ومباشرة.
        """.strip()

        return prompt

    def _gemini_generate(self, question: str, contexts: Optional[List]) -> str:
        prompt = self._build_prompt(question, contexts)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_output_tokens": self.max_tokens,
                },
            )
        except Exception as e:
            print("❌ Error while calling Gemini:", e)
            return "حدث خطأ أثناء الاتصال بنموذج Gemini."

        if getattr(response.candidates[0], "finish_reason", None) in ("MAX_TOKENS", "SAFETY"):
            return "لا أجد إجابة واضحة في النص."

        text = ""
        try:
            if hasattr(response, "text") and isinstance(response.text, str):
                text = response.text.strip()
        except Exception:
            text = ""

        # Fallback parsing if still empty
        if not text and getattr(response, "candidates", None):
            for c in response.candidates:
                content = getattr(c, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    for p in parts:
                        part_text = getattr(p, "text", None)
                        if part_text:
                            text = part_text.strip()
                            break
                if text:
                    break
        # Final fallback
        if not text:
            print("⚠️ Gemini returned empty or filtered text — raw response:", response)
            text = "لم أجد إجابة واضحة من النص المعطى."

        text = normalize_arabic_text(text)
        return _arabic_only(text)

    def generate(self, question: str, contexts: Optional[List] = None) -> str:
        """Main entry point for generation."""
        return self._gemini_generate(question, contexts)
