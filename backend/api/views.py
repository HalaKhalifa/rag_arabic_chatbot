from django.shortcuts import render
import time
from analytics.services import log_chat_event
from django.http import JsonResponse
from ragchat.core.pipeline import RagPipeline
from ragchat.core.embeddings import TextEmbedder
from ragchat.core.retriever import Retriever
from ragchat.core.generator import Generator
from ragchat.storage.qdrant_index import QdrantIndex
from django.views.decorators.csrf import csrf_exempt
from django.contrib.admin.views.decorators import staff_member_required
from ragchat.config import RAGSettings
from ragchat.logger import logger
import json
from .services.rag_service import ingest_text_to_qdrant
from .services.eval_service import evaluate_prediction
from django.contrib.auth.decorators import login_required
from .models import ChatHistory

try:
    embedder = TextEmbedder(RAGSettings.emb_model)
    index = QdrantIndex(RAGSettings.qdrant_url, RAGSettings.qdrant_api_key)
    retriever = Retriever(embedder, index, RAGSettings.contexts_col, RAGSettings.top_k)
    generator = Generator(RAGSettings.gen_model)
    pipeline = RagPipeline(embedder, retriever, generator, RAGSettings.top_k)
except Exception as e:
    pipeline = None
    logger.error(f"Failed to initialize RAG pipeline: {e}")

def health_check(request):
    return JsonResponse({"status": "ok"})

@csrf_exempt
def ask(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        body = json.loads(request.body.decode())
        question = body.get("question", "").strip()

        if not question:
            return JsonResponse({"error": "Question is required"}, status=400)

        if not pipeline:
            return JsonResponse({"error": "Pipeline not initialized"}, status=500)

        # measure latency
        start_ms = int(time.time() * 1000)
        result = pipeline.answer(question)
        end_ms = int(time.time() * 1000)
        latency_ms = end_ms - start_ms
        answer = result.get("answer", "")
        contexts = result.get("retrieved_contexts", []) or []

        # compute top_score from retrieval
        top_score = None
        if contexts:
            try:
                top_score = max(
                    float(c.get("score") or 0.0)
                    for c in contexts
                    if isinstance(c, dict)
                )
            except Exception:
                top_score = None

        # consider our Arabic fallback message as failure
        success = True
        error_type = None
        if answer.strip() == "حدث خطأ أثناء توليد الإجابة." or answer.strip() == "حدث خطأ أثناء الاتصال بنموذج Gemini." :
            success = False
            error_type = "generation_error"
        elif answer.strip() == "لا أجد إجابة واضحة في النص.":
            success = False
            error_type = "context_missing"
        
        # Create ChatHistory entry for authenticated users
        chat_history_entry = None
        if request.user.is_authenticated:
            try:
                chat_history_entry = ChatHistory.objects.create(
                    user=request.user,
                    question=question,
                    answer=answer,
                    sources=contexts
                )
            except Exception as save_exc:
                logger.warning(f"Failed to save chat history: {save_exc}")

        # Log analytics event
        try:
            log_chat_event(
                user=request.user if request.user.is_authenticated else None,
                channel="api",
                question=question,
                answer=answer,
                latency_ms=latency_ms,
                top_score=top_score,
                num_contexts=len(contexts),
                success=success,
                error_type=error_type,
                metadata={
                    "retrieved_contexts_count": len(contexts),
                    "chat_history_id": chat_history_entry.id if chat_history_entry else None
                },
                session_id=str(request.user.id) if request.user.is_authenticated else None
            )
        except Exception as log_exc:
            logger.warning(f"Failed to log analytics event: {log_exc}")

        return JsonResponse(result, safe=False)

    except Exception as e:
        logger.error(f"RAG answer endpoint failed: {e}")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@staff_member_required
def ingest(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        body = json.loads(request.body.decode())
        text = body.get("text", "").strip()

        if not text:
            return JsonResponse({"error": "Text is required"}, status=400)

        result = ingest_text_to_qdrant(text)
        return JsonResponse(result, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def evaluate(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
        question = data.get("question")
        expected = data.get("expected_answer")

        if not question or not expected:
            return JsonResponse(
                {"error": "Fields 'question' and 'expected_answer' are required."},
                status=400
            )

        if not pipeline:
            return JsonResponse({"error": "Pipeline not initialized"}, status=500)

        # run the RAG pipeline
        result = pipeline.answer(question)
        generated_answer = result.get("answer")

        # evaluate prediction
        scores = evaluate_prediction(expected, generated_answer)

        return JsonResponse({
            "question": question,
            "expected_answer": expected,
            "generated_answer": generated_answer,
            "bleu": scores["bleu"],
            "f1": scores["f1"]
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def chat_history(request):
    history = ChatHistory.objects.filter(user=request.user).order_by("-timestamp")[:20]
    results = [
        {"question": h.question, "answer": h.answer, "timestamp": h.timestamp.isoformat(), "sources": h.sources}
        for h in history
    ]
    return JsonResponse({"results": results})

@login_required
@csrf_exempt
def clear_chat_history(request):
    if request.method == "POST":
        ChatHistory.objects.filter(user=request.user).delete()
        return JsonResponse({"status": "ok"})
    return JsonResponse({"error": "POST required"}, status=405)

def chat_page(request):
    """
    Public chat page (end users)
    """
    return render(request, "api/chat.html")


@staff_member_required
def ingest_page(request):
    """
    Admin-only ingest page
    """
    return render(request, "api/ingest.html")
