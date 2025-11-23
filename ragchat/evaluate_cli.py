import typer
from datasets import load_from_disk, DatasetDict
from tqdm import tqdm
import time
from .config import settings
from .embeddings import TextEmbedder
from .qdrant_index import QdrantIndex
from .retriever import Retriever
from .generator import Generator
from .pipeline import RagPipeline
from .evaluation import bleu, f1


def main(
    ds_path: str = settings.clean_arcd_dir,
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

    # Components: use the same configuration as chat / pipeline
    embedder = TextEmbedder(settings.emb_model)
    index = QdrantIndex(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    retriever = Retriever(
        embedder=embedder,
        index=index,
        collection=settings.contexts_col,
        top_k=settings.top_k,
    )
    # Generator reads all configs (model name, API key, temperature, top_p, max tokens) from settings
    generator = Generator()
    pipeline = RagPipeline(
        embedder=embedder,
        retriever=retriever,
        generator=generator,
        top_k=settings.top_k,
    )

    preds, refs = [], []
    total = min(n, len(split))
    print(f"Evaluating {total} samples from {ds_path} ...", flush=True)

    for i, ex in enumerate(tqdm(split, total=total)):
        if i >= total:
            break

        q = ex["question"]
        # ARCD-style answers: {"text": [answer_str, ...], "answer_start": [...]}
        answers = ex.get("answers", {})
        if isinstance(answers, dict):
            gold_list = answers.get("text") or [""]
            gold = gold_list[0] if gold_list else ""
        else:
            gold = ""

        out = pipeline.answer(q)
        time.sleep(1.5)
        preds.append(out["answer"])
        refs.append(gold)

    # Metrics
    b = bleu(preds, refs)
    f1_scores = [f1(p, r) for p, r in zip(preds, refs)]
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    print("\n=== Evaluation Results ===")
    print(f"BLEU: {b:.2f}")
    print(f"F1:   {avg_f1:.3f}", flush=True)


if __name__ == "__main__":
    typer.run(main)