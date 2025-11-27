import typer
from datasets import load_from_disk
from .config import RAGSettings
from .preprocessing import preprocess_arcd

app = typer.Typer(help="Arabic RAG Data Preparation CLI")

@app.command()
def prepare_raw(out: str = RAGSettings.raw_arcd_dir):
    """
    Download ARCD dataset from HuggingFace and save to disk.
    """
    from datasets import load_dataset
    ds = load_dataset("hsseinmz/arcd")
    ds.save_to_disk(out)
    typer.echo(f"📁 Raw ARCD dataset saved to {out}")

@app.command()
def preprocess(
    in_dir: str = RAGSettings.raw_arcd_dir,
    out_dir: str = RAGSettings.clean_arcd_dir,
    group_size: int = typer.Option(5, help="Number of sentences per chunk"),
):
    """
    Apply normalization + sentence splitting + chunking.
    """
    preprocess_arcd(in_dir=in_dir, out_dir=out_dir, group_size=group_size)

if __name__ == "__main__":
    app()