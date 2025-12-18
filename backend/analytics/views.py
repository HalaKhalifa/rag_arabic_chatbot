import json
import csv
from io import StringIO
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db.models import Count, Avg, Q
from django.db.models.functions import TruncDate
from .models import ChatEvent
from ragchat.logger import logger
from django.shortcuts import render
from django.contrib.admin.views.decorators import staff_member_required

def _filter_events(request):
    """
    Common filtering for analytics endpoints.
    Supports query params:
      - start=YYYY-MM-DD
      - end=YYYY-MM-DD
      - channel=api|cli|all
      - success=true|false
    """
    qs = ChatEvent.objects.all()

    start = request.GET.get("start")
    end = request.GET.get("end")
    channel = request.GET.get("channel")
    success = request.GET.get("success")

    if start:
        qs = qs.filter(timestamp__date__gte=start)

    if end:
        qs = qs.filter(timestamp__date__lte=end)

    if channel and channel != "all":
        qs = qs.filter(channel=channel)

    if success == "true":
        qs = qs.filter(success=True)
    elif success == "false":
        qs = qs.filter(success=False)

    return qs

@csrf_exempt
@staff_member_required
def analytics_log(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    raw_body = request.body.decode().strip()
    if not raw_body:
        logger.error("[Analytics] Empty request body")
        return JsonResponse({"error": "Empty JSON body"}, status=400)

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as e:
        logger.error(f"[Analytics] Invalid JSON: {e}")
        return JsonResponse({"error": "Invalid JSON format"}, status=400)

    if "question" not in payload:
        return JsonResponse({"error": "Missing field: question"}, status=400)

    if "answer" not in payload:
        return JsonResponse({"error": "Missing field: answer"}, status=400)

    metadata = payload.get("metadata") or {}
    try:
        json.dumps(metadata)  # test serializability
    except TypeError:
        logger.error("[Analytics] Invalid metadata: not JSON serializable")
        return JsonResponse({"error": "Metadata must be JSON-serializable"}, status=400)

    try:
        event = ChatEvent.objects.create(
            channel = payload.get("channel") or "cli",
            question = payload.get("question") or "",
            answer = payload.get("answer") or "",
            latency_ms = payload.get("latency_ms") or 0,
            num_contexts = payload.get("num_contexts") or 0,
            top_score = payload.get("top_score") or 0,
            success = bool(payload.get("success", True)),
            error_type = payload.get("error_type"),
            session_id = payload.get("session_id"),
            metadata = metadata,
        )

        return JsonResponse({"status": "ok", "id": event.id})

    except Exception as e:
        logger.error(f"[Analytics] Failed to save event: {e}")
        return JsonResponse({"error": "Internal server error"}, status=500)

@csrf_exempt
@staff_member_required
def analytics_summary(request):
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)

    qs = _filter_events(request)

    total = qs.count()
    successful = qs.filter(success=True).count()
    failed = qs.filter(success=False).count()

    avg_latency = qs.aggregate(avg=Avg("latency_ms"))["avg"]
    avg_score = qs.aggregate(avg=Avg("top_score"))["avg"]

    by_channel = (
        qs.values("channel")
        .annotate(
            count=Count("id"),
            success_count=Count("id", filter=Q(success=True)),
        )
        .order_by("channel")
    )

    data = {
        "total_events": total,
        "successful": successful,
        "failed": failed,
        "success_rate": (successful / total) if total else 0.0,
        "avg_latency_ms": avg_latency or 0,
        "avg_top_score": avg_score or 0,
        "by_channel": list(by_channel),
    }
    return JsonResponse(data)

@csrf_exempt
@staff_member_required
def analytics_daily(request):
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)

    qs = _filter_events(request)
    qs = qs.annotate(day=TruncDate("timestamp"))
    daily_data = {}

    for event in qs:
        day = str(event.day)
        if day not in daily_data:
            daily_data[day] = {"total": 0, "success": 0, "failed": 0}

        daily_data[day]["total"] += 1
        if event.success:
            daily_data[day]["success"] += 1
        else:
            daily_data[day]["failed"] += 1

    response = [
        {"day": k, **v} for k, v in sorted(daily_data.items())
    ]

    return JsonResponse({"results": response})

@csrf_exempt
@staff_member_required
def analytics_top_questions(request):
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)

    qs = _filter_events(request)
    limit = int(request.GET.get("limit", 10))

    top_qs = (
        qs.values("question")
        .annotate(
            count=Count("id"),
            avg_latency=Avg("latency_ms"),
            avg_score=Avg("top_score"),
        )
        .order_by("-count")[:limit]
    )

    return JsonResponse({"results": list(top_qs)})

@csrf_exempt
@staff_member_required
def analytics_export(request):
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)

    qs = _filter_events(request)
    export_format = request.GET.get("format", "json").lower()

    fields = [
        "id",
        "timestamp",
        "channel",
        "question",
        "answer",
        "latency_ms",
        "top_score",
        "num_contexts",
        "success",
        "error_type",
        "session_id",
    ]

    if export_format == "csv":
        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(fields)

        for ev in qs.only(*fields):
            row = [getattr(ev, f) for f in fields]
            writer.writerow(row)

        return JsonResponse({"format": "csv", "data": buffer.getvalue()})

    # default JSON
    events = list(qs.values(*fields).order_by("-timestamp"))
    return JsonResponse({"format": "json", "results": events})

@csrf_exempt
@staff_member_required
def analytics_dashboard(request):
    return render(request, "analytics/dashboard.html")
