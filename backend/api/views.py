from django.shortcuts import render

from django.http import JsonResponse
from django.conf import settings
from ragchat.core.pipeline import RagPipeline
from ragchat.core.embeddings import TextEmbedder
from ragchat.core.retriever import Retriever
from ragchat.core.generator import Generator
from ragchat.storage.qdrant_index import QdrantIndex
from django.views.decorators.csrf import csrf_exempt
from ragchat.config import RAGSettings
from ragchat.logger import logger
import json

try:
    embedder = TextEmbedder(RAGSettings.emb_model)
    index = QdrantIndex(RAGSettings.qdrant_url, RAGSettings.qdrant_api_key)
    retriever = Retriever(embedder, index, RAGSettings.contexts_col, RAGSettings.top_k)
    generator = Generator(RAGSettings.gen_model)
    pipeline = RagPipeline(embedder, retriever, generator, RAGSettings.top_k)
except Exception as e:
    pipeline = None
    logger.error(f"Failed to initialize RAG pipeline: {e}")

def require_api_key(request):
    token = request.headers.get("X-API-KEY")
    if not token or token != getattr(settings, "API_SECRET", None):
        return JsonResponse({"error": "Unauthorized"}, status=401)
    return None

def health_check(request):
    return JsonResponse({"status": "ok"})

@csrf_exempt
def ask(request):
    # Auth
    auth = require_api_key(request)
    if auth: 
        return auth

    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        body = json.loads(request.body.decode())
        question = body.get("question", "").strip()

        if not question:
            return JsonResponse({"error": "Question is required"}, status=400)

        if not pipeline:
            return JsonResponse({"error": "Pipeline not initialized"}, status=500)

        result = pipeline.answer(question)
        return JsonResponse(result, safe=False)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def ingest(request):
    # Auth
    auth = require_api_key(request)
    if auth: 
        return auth

    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        body = json.loads(request.body.decode())
        text = body.get("text", "").strip()

        if not text:
            return JsonResponse({"error": "Text is required"}, status=400)

        # TODO: integrate final ingestion logic
        return JsonResponse({"status": "ingestion not implemented yet, will be done later"})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
