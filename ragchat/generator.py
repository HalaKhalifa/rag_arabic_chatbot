import os
import re
import google.generativeai as genai
from .utils import clean_text

SEP = "\n- "  # bullet sep for contexts


def _arabic_only(s: str) -> str:
    """Keep Arabic letters, digits, and basic punctuation only."""
    s = re.sub(r"[^ء-ي0-9\s.,؟!:؛\-\(\)\"']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class Generator:
    """
    Gemini-based generator for Arabic RAG chatbot.
    """

    def __init__(
        self,
        model_name: str = "models/gemini-2.5-flash",
        max_new_tokens: int = 512,
        temperature: float = 0.4,
        top_p: float = 0.9,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in environment.")

        genai.configure(api_key=api_key)
        self._gemini = None  # lazy initialization

    # Prompt Construction
    def _build_prompt(self, question: str, contexts: list[str] | None) -> str:
        """Combine Arabic contexts and question into one readable prompt."""
        unique_ctxs = []
        seen = set()
        for c in (contexts or []):
            c = c.strip()
            if c and c not in seen:
                seen.add(c)
                unique_ctxs.append(c)

        ctx = "\n\n".join(unique_ctxs[:2])
        ctx = ctx[:900]  # truncate long context
        prompt = (
            f"النص:\n{ctx}\n\n"
            f"السؤال: {question.strip()}\n"
            "أجب بإيجاز وبشكل دقيق باللغة العربية فقط."
        )
        return prompt

    # Gemini Generation
    def _gemini_generate(self, question: str, contexts: list[str] | None) -> str:
        if self._gemini is None:
            try:
                self._gemini = genai.GenerativeModel(self.model_name)
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini model: {e}")
                return "تعذر تهيئة نموذج Gemini."

        prompt = self._build_prompt(question, contexts)
        print("🔍 Gemini prompt preview:\n", prompt[:500])

        # Call Gemini safely
        try:
            response = self._gemini.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                },
            )
        except Exception as e:
            print("⚠️ Gemini API call failed:", e)
            return "حدث خطأ أثناء توليد الإجابة."

        # Try to read model text safely
        text = ""
        try:
            if getattr(response, "text", None):
                text = response.text.strip()
        except ValueError:
            # This happens when Gemini refuses for safety reasons
            finish = getattr(response.candidates[0], "finish_reason", None) if getattr(response, "candidates", None) else None
            print(f"⚠️ Gemini safety filter blocked output (finish_reason={finish}).")
            text = "النموذج لم يتمكن من توليد إجابة لأسباب تتعلق بالسلامة."

        # Fallback parsing if still empty
        if not text and getattr(response, "candidates", None):
            for c in response.candidates:
                if hasattr(c, "content") and getattr(c.content, "parts", None):
                    for p in c.content.parts:
                        if getattr(p, "text", None):
                            text = p.text.strip()
                            break
                if text:
                    break

        # Final fallback
        if not text:
            print("⚠️ Gemini returned empty or filtered text — raw response:", response)
            text = "لم أجد إجابة واضحة من النص المعطى."

        return _arabic_only(clean_text(text))
    # Public Interface
    def generate(self, question: str, contexts: list[str] | None = None) -> str:
        """Main entry point for generation."""
        return self._gemini_generate(question, contexts)
