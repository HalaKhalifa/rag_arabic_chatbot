import typer
from datasets import load_dataset
from .config import RAGSettings
from .preprocessing import preprocess_arcd
from .logger import logger

app = typer.Typer(help="Arabic RAG Data Preparation CLI")

@app.command()
def prepare_raw(out: str = RAGSettings.raw_arcd_dir):
    """
    Download ARCD dataset from HuggingFace and save to disk.
    """
    try:
        logger.info("Downloading ARCD dataset from HuggingFace..")
        ds = load_dataset("hsseinmz/arcd")
        logger.info(f"Saving raw ARCD dataset to: {out}")
        ds.save_to_disk(out)
        logger.info("Raw dataset saved successfully.")
    except Exception as e:
        logger.error(f"Failed to download or save ARCD dataset: {e}")
        raise

@app.command()
def preprocess(
    in_dir: str = RAGSettings.raw_arcd_dir,
    out_dir: str = RAGSettings.clean_arcd_dir,
    group_size: int = typer.Option(5, help="Number of sentences per chunk"),
):
    """
    Apply normalization + sentence splitting + chunking.
    """
    try:
        logger.info(
            f"Preprocessing ARCD dataset (in={in_dir}, out={out_dir}, group_size={group_size})"
        )
        preprocess_arcd(in_dir=in_dir, out_dir=out_dir, group_size=group_size)
    except Exception as e:
        logger.error(f"Failed to preprocess dataset: {e}")
        raise

if __name__ == "__main__":
    app()