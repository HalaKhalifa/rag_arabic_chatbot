from datasets import load_from_disk, DatasetDict
from typing import Dict, Any, List
from .utils import normalize_arabic_text, split_into_sentences, chunk_sentences
from .config import settings


def preprocess_example(example: Dict[str, Any], group_size: int = 5) -> Dict[str, Any]:
    """
    Clean & chunk a single example from ARCD.
    Creates a new field: 'chunks' = list of sentence groups for embedding.
    """
    context = normalize_arabic_text(example.get("context", ""))
    question = normalize_arabic_text(example.get("question", ""))

    # Normalize first answer only (ARCD format)
    answers = example.get("answers", {})
    if isinstance(answers, dict) and "text" in answers and answers["text"]:
        ans_list = answers["text"]
        if isinstance(ans_list, list) and len(ans_list) > 0:
            ans_list[0] = normalize_arabic_text(ans_list[0])
            answers["text"] = ans_list

    example["context"] = context
    example["question"] = question
    example["answers"] = answers

    # Sentence segmentation
    sentences = split_into_sentences(context)

    # Chunking
    chunks = chunk_sentences(sentences, group_size=group_size)
    example["chunks"] = chunks

    return example


def preprocess_arcd(
    in_dir: str = settings.raw_arcd_dir,
    out_dir: str = settings.clean_arcd_dir,
    group_size: int = 5,
):
    """
    Load ARCD raw → clean text → split → chunk → save cleaned dataset to disk.
    """

    print(f"📥 Loading raw dataset from: {in_dir}")
    ds = load_from_disk(in_dir)

    print("🧹 Applying normalization + chunking...")

    if isinstance(ds, DatasetDict):
        ds_clean = DatasetDict({
            split: ds[split].map(
                lambda ex: preprocess_example(ex, group_size=group_size),
                desc=f"Processing {split}"
            )
            for split in ds.keys()
        })
    else:
        ds_clean = ds.map(
            lambda ex: preprocess_example(ex, group_size=group_size),
            desc="Processing dataset"
        )

    print(f"💾 Saving cleaned dataset to: {out_dir}")
    ds_clean.save_to_disk(out_dir)
    print("✅ Preprocessing completed successfully!")