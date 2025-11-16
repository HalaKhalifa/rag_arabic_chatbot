import typer
from datasets import load_from_disk, DatasetDict
from tqdm import tqdm

from .config import settings
from .embeddings import TextEmbedder
from .qdrant_index import QdrantIndex
from .retriever import Retriever
from .generator import Generator
from .pipeline import ArabicRAGPipeline
from .evaluation import bleu, f1


def main(
    ds_path: str = "data/processed/arcd_clean_prepared",
    n: int = typer.Option(50, "--n", "-n", help="Number of samples to evaluate"),
):
    """
    Evaluate the Arabic RAG (Gemini-based) pipeline on ARCD validation/test split.
    Computes BLEU and token-level F1.
    """
    # Load dataset
    ds = load_from_disk(ds_path)
    if isinstance(ds, DatasetDict):
        split = ds.get("validation") or ds.get("test") or next(iter(ds.values()))
    else:
        split = ds

    # Components
    emb = TextEmbedder(settings.emb_model)
    idx = QdrantIndex(settings.qdrant_url, settings.qdrant_api_key)
    retr = Retriever(emb, idx, settings.contexts_col, settings.top_k)

    # Gemini generator
    gen = Generator(
        model_name=settings.gen_model,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
    )

    pipe = ArabicRAGPipeline(retriever=retr, generator=gen, top_k=settings.top_k)

    preds, refs = [], []
    print(f"Evaluating {n} samples from {ds_path} ...", flush=True)

    for i, ex in enumerate(tqdm(split, total=n)):
        if i >= n:
            break

        q = ex["question"]
        gold = ex.get("answers", {}).get("text", [""])[0]

        out = pipe.run(q)
        preds.append(out["answer"])
        refs.append(gold)

    # Metrics
    b = bleu(preds, refs)
    f1_scores = [f1(p, r) for p, r in zip(preds, refs)]

    print(f"\nBLEU: {b:.2f}")
    print(f"F1:   {sum(f1_scores)/len(f1_scores):.3f}", flush=True)


if __name__ == "__main__":
    typer.run(main)
