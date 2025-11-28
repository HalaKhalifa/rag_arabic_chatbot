import typer
from rich.console import Console
from rich.panel import Panel
from ragchat.config import RAGSettings
from ragchat.core.embeddings import TextEmbedder
from ragchat.core.retriever import Retriever
from ragchat.storage.qdrant_index import QdrantIndex
from ragchat.core.generator import Generator
from ragchat.core.pipeline import RagPipeline
from ragchat.logger import logger

console = Console()
app = typer.Typer(help="Arabic RAG Chatbot")

def _print_contexts(contexts):
    try:
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
    except Exception as e:
        logger.error(f"Error while printing contexts: {e}")

@app.command()
def chat():
    """
    Start an interactive Arabic RAG chat session.
    """
    try:
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
    except Exception as e:
        logger.error(f"Failed to initialize chat pipeline: {e}")
        console.print("[red]تعذر تشغيل النظام، تحقق من الإعدادات أو قاعدة البيانات.[/red]")
        return

    while True:
        try:
            q = console.input("\n[bold yellow]سؤالك:[/bold yellow] ").strip()
        except Exception as e:
            logger.error(f"Failed to read user input: {e}")
            continue

        if q.lower() in ["/exit", "exit", "quit"]:
            console.print("\n👋 Bye!\n")
            break
        console.print("\n[cyan]🔎 Retrieving relevant context...[/cyan]")
        try:
            result = pipeline.answer(q)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            console.print("[red]حدث خطأ أثناء معالجة السؤال.[/red]")
            continue

        answer = result["answer"]
        contexts = result["retrieved_contexts"]
        _print_contexts(contexts)

        console.print("\n[bold green]--- Answer ---[/bold green]\n")
        console.print(Panel(answer, border_style="green"))

if __name__ == "__main__":
    app()