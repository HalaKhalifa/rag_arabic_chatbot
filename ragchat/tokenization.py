from transformers import AutoTokenizer

class ArabicTokenizer:
    """Wrapper around Arabic GPT-2 tokenizer with padding/truncation."""
    def __init__(self, model_name: str = "aubmindlab/aragpt2-base", max_length: int = 256):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.max_length = max_length

    def encode(self, texts: list[str]):
        return self.tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
