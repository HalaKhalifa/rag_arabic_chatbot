import typer
from datasets import load_dataset, load_from_disk, DatasetDict
from .config import settings
from .preprocessing import normalize_arabic_text

def prepare_arcd(out_dir: str = settings.processed_ds_dir):
    """
    Load ARCD from HF and save to disk (raw -> processed dir).
    """
    ds = load_dataset("hsseinmz/arcd")
    ds.save_to_disk(out_dir)
    typer.echo(f"Saved ARCD to {out_dir}")

def preprocess_arcd(
    in_dir: str = "data/processed/arcd_clean",
    out_dir: str = "data/processed/arcd_clean_prepared"
):
    """
    Normalize Arabic text (context, question, answer[0]).
    """
    ds = load_from_disk(in_dir)

    def _clean(example):
        example["context"] = normalize_arabic_text(example.get("context", ""))
        example["question"] = normalize_arabic_text(example.get("question", ""))
        if "answers" in example and example["answers"].get("text"):
            example["answers"]["text"][0] = normalize_arabic_text(example["answers"]["text"][0])
        return example

    if isinstance(ds, DatasetDict):
        ds_clean = DatasetDict({k: v.map(_clean) for k, v in ds.items()})
    else:
        ds_clean = ds.map(_clean)

    ds_clean.save_to_disk(out_dir)
    print(f"✅ Preprocessed ARCD saved to {out_dir}")


if __name__ == "__main__":
    typer.run(preprocess_arcd)