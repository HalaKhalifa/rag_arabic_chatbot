import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from .config import RAGSettings
from .embeddings import TextEmbedder
from .retriever import Retriever
from .qdrant_index import QdrantIndex
from .generator import Generator
from .pipeline import RagPipeline

console = Console()
app = typer.Typer(help="Arabic RAG Chatbot")

def _print_contexts(contexts):
    console.print("\n[bold cyan]--- Top Retrieved Contexts ---[/bold cyan]")
    for c in contexts:
        text = c.get("chunk") or c.get("context_text") or c.get("raw_context")
        score = c.get("score")
        idx = c.get("chunk_index")

        console.print(
            Panel(
                f"[bold]Chunk {idx}[/bold]\n"
                f"[yellow]Score:[/yellow] {score:.4f}\n\n"
                f"{text}",
                title="Context",
                border_style="cyan",
            )
        )

@app.command()
def chat():
    """
    Start an interactive Arabic RAG chat session.
    """
    console.print("\n💬 [bold green]Arabic RAG Chatbot[/bold green]")
    console.print("Type your question, or /exit to quit.\n")

    embedder = TextEmbedder(RAGSettings.emb_model)
    index = QdrantIndex(RAGSettings.qdrant_url, RAGSettings.qdrant_api_key)
    retriever = Retriever(embedder, index, RAGSettings.contexts_col, RAGSettings.top_k)
    generator = Generator(RAGSettings.gen_model)
    pipeline = RagPipeline(
        embedder=embedder,
        retriever=retriever,
        generator=generator,
        top_k=RAGSettings.top_k,
    )

    while True:
        q = console.input("\n[bold yellow]سؤالك:[/bold yellow] ").strip()
        if q.lower() in ["/exit", "exit", "quit"]:
            console.print("\n👋 Bye!\n")
            break

        console.print("\n[cyan]🔎 Retrieving relevant context...[/cyan]")
        result = pipeline.answer(q)
        answer = result["answer"]
        contexts = result["retrieved_contexts"]
        _print_contexts(contexts)

        console.print("\n[bold green]--- Answer ---[/bold green]\n")
        console.print(Panel(answer, border_style="green"))

if __name__ == "__main__":
    app()