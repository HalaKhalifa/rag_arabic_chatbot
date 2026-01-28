import json
import csv
from io import StringIO
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db.models import Count, Avg, Q, F, Max, Min
from django.db.models.functions import TruncDate, TruncHour
from django.db.models import FloatField, Case, When, Value
from .models import ChatEvent
from ragchat.logger import logger
from django.shortcuts import render
from django.contrib.admin.views.decorators import staff_member_required
from datetime import datetime, timedelta
from collections import defaultdict

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

    today = timezone.now().date()
    last_week_start = today - timedelta(days=7)
    last_month_start = today - timedelta(days=30)
    
    total = qs.count()
    successful = qs.filter(success=True).count()
    failed = qs.filter(success=False).count()

    # Time period comparisons
    last_week_total = qs.filter(timestamp__date__gte=last_week_start).count()
    last_month_total = qs.filter(timestamp__date__gte=last_month_start).count()
    
    # Calculate trends
    week_trend = 0
    month_trend = 0
    if last_week_total > 0:
        week_trend = ((total - last_week_total) / last_week_total) * 100
    if last_month_total > 0:
        month_trend = ((total - last_month_total) / last_month_total) * 100

    avg_latency = qs.aggregate(avg=Avg("latency_ms"))["avg"]
    avg_score = qs.aggregate(avg=Avg("top_score"))["avg"]

    # Channel distribution
    by_channel = (
        qs.values("channel")
        .annotate(
            count=Count("id"),
            success_count=Count("id", filter=Q(success=True)),
            avg_latency=Avg("latency_ms"),
            avg_score=Avg("top_score")
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
        "trends": {
            "week_over_week": round(week_trend, 1),
            "month_over_month": round(month_trend, 1)
        },
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

    # Anonymize questions for privacy
    anonymized_results = []
    for item in list(top_qs):
        question = item["question"]
        if len(question) > 50:
            question = question[:50] + "..."
        
        anonymized_results.append({
            "question": question,
            "count": item["count"],
            "avg_latency": item["avg_latency"],
            "avg_score": item["avg_score"]
        })

    return JsonResponse({"results": anonymized_results})

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
def analytics_engagement(request):
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)

    qs = _filter_events(request)    
    today = timezone.now().date()
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)

    # Unique users
    unique_users = qs.filter(user__isnull=False).values("user").distinct().count()
    active_users_last_7_days = qs.filter(
        timestamp__date__gte=week_ago,
        user__isnull=False
    ).values("user").distinct().count()
    active_users_last_30_days = qs.filter(
        timestamp__date__gte=month_ago,
        user__isnull=False
    ).values("user").distinct().count()

    # Session analysis
    total_sessions = qs.filter(session_id__isnull=False).values("session_id").distinct().count()
    
    # Average questions per session
    avg_questions_per_session = 0
    if total_sessions > 0:
        session_stats = qs.filter(session_id__isnull=False).values("session_id").annotate(
            question_count=Count("id")
        )
        total_questions_in_sessions = sum(s['question_count'] for s in session_stats)
        avg_questions_per_session = total_questions_in_sessions / total_sessions

    # Daily active users
    daily_active_users = (
        qs.filter(user__isnull=False)
        .annotate(date=TruncDate("timestamp"))
        .values("date")
        .annotate(active_users=Count("user", distinct=True))
        .order_by("date")
    )

    # Hourly usage patterns
    hourly_activity = (
        qs.annotate(hour=TruncHour("timestamp"))
        .values("hour")
        .annotate(total_questions=Count("id"))
        .order_by("hour")
    )

    return JsonResponse({
        "engagement_metrics": {
            "unique_users_total": unique_users,
            "active_users_7d": active_users_last_7_days,
            "active_users_30d": active_users_last_30_days,
            "total_sessions": total_sessions,
            "avg_questions_per_session": round(avg_questions_per_session, 1),
            "engagement_rate_7d": round((active_users_last_7_days / max(unique_users, 1)) * 100, 1)
        },
        "daily_activity": list(daily_active_users),
        "hourly_patterns": list(hourly_activity),
    })

@csrf_exempt
@staff_member_required
def analytics_quality_metrics(request):
    """Quality metrics without revealing user-specific data"""
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)

    qs = _filter_events(request)
    
    # Quality buckets based on score
    quality_distribution = {
        "excellent": qs.filter(top_score__gte=0.8).count(),
        "good": qs.filter(top_score__gte=0.6, top_score__lt=0.8).count(),
        "fair": qs.filter(top_score__gte=0.4, top_score__lt=0.6).count(),
        "poor": qs.filter(top_score__lt=0.4).count(),
        "no_score": qs.filter(top_score__isnull=True).count(),
    }

    # Context usage analysis
    context_usage = {
        "no_contexts": qs.filter(num_contexts=0).count(),
        "1_3_contexts": qs.filter(num_contexts__gte=1, num_contexts__lte=3).count(),
        "4_6_contexts": qs.filter(num_contexts__gte=4, num_contexts__lte=6).count(),
        "7_plus_contexts": qs.filter(num_contexts__gte=7).count(),
    }

    # Latency percentiles
    latencies = list(qs.exclude(latency_ms__isnull=True).values_list("latency_ms", flat=True))
    latencies.sort()
    n = len(latencies)
    
    percentiles = {
        "p50": latencies[int(n * 0.5)] if n > 0 else 0,
        "p90": latencies[int(n * 0.9)] if n > 1 else 0,
        "p95": latencies[int(n * 0.95)] if n > 2 else 0,
        "p99": latencies[int(n * 0.99)] if n > 3 else 0,
        "max": latencies[-1] if n > 0 else 0,
        "min": latencies[0] if n > 0 else 0,
    }

    # Error analysis
    error_types = (
        qs.filter(success=False, error_type__isnull=False)
        .values("error_type")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    # Success rate over time
    success_over_time = (
        qs.annotate(date=TruncDate("timestamp"))
        .values("date")
        .annotate(
            total=Count("id"),
            successful=Count("id", filter=Q(success=True))
        )
        .annotate(
            success_rate=Case(
                When(total=0, then=Value(0.0)),
                default=F("successful") * 1.0 / F("total") * 100,
                output_field=FloatField()
            )
        )
        .order_by("date")
    )

    return JsonResponse({
        "quality_distribution": quality_distribution,
        "context_usage": context_usage,
        "latency_percentiles": percentiles,
        "error_analysis": list(error_types),
        "success_over_time": list(success_over_time),
        "total_responses": qs.count(),
        "avg_contexts_per_query": qs.aggregate(avg=Avg("num_contexts"))["avg"] or 0,
    })

@csrf_exempt
@staff_member_required
def analytics_topic_analysis(request):
    """Analyze topics without storing specific questions"""
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)

    qs = _filter_events(request)
    
    # Get question length statistics
    question_lengths = list(qs.values_list("question", flat=True))
    length_stats = {
        "total_questions": len(question_lengths),
        "avg_length": sum(len(q) for q in question_lengths) / max(len(question_lengths), 1),
        "short_questions": len([q for q in question_lengths if len(q) < 20]),
        "medium_questions": len([q for q in question_lengths if 20 <= len(q) < 100]),
        "long_questions": len([q for q in question_lengths if len(q) >= 100]),
    }

    # Response length statistics
    answer_lengths = list(qs.values_list("answer", flat=True))
    answer_stats = {
        "avg_answer_length": sum(len(a) for a in answer_lengths) / max(len(answer_lengths), 1),
        "short_answers": len([a for a in answer_lengths if len(a) < 50]),
        "detailed_answers": len([a for a in answer_lengths if len(a) >= 200]),
    }

    # Common patterns
    question_keywords = {
        "what_questions": qs.filter(question__icontains="ما").count(),
        "how_questions": qs.filter(question__icontains="كيف").count() +
                        qs.filter(question__icontains="كم").count(),
        "why_questions": qs.filter(question__icontains="لماذا").count() + 
                        qs.filter(question__icontains="لما").count(),
        "when_questions": qs.filter(question__icontains="متى").count(),
        "where_questions": qs.filter(question__icontains="أين").count(),
        "who_questions": qs.filter(question__icontains="من").count(),
    }

    # Performance by question type
    performance_by_type = {}
    for q_type, count in question_keywords.items():
        if count > 0:
            # Simplified type name for display
            display_name = q_type.replace("_questions", "").replace("_", " ").title()
            performance_by_type[display_name] = {
                "count": count,
                "percentage": (count / max(len(question_lengths), 1)) * 100
            }

    return JsonResponse({
        "question_analysis": length_stats,
        "answer_analysis": answer_stats,
        "question_types": performance_by_type,
    })

@csrf_exempt
@staff_member_required
def analytics_performance(request):
    """System performance metrics"""
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)

    qs = _filter_events(request)
    
    today = timezone.now().date()
    yesterday = today - timedelta(days=1)
    last_week_start = today - timedelta(days=7)
    
    # Today's performance
    today_stats = qs.filter(timestamp__date=today).aggregate(
        count=Count("id"),
        avg_latency=Avg("latency_ms"),
        success_rate=Avg(Case(When(success=True, then=1), default=0, output_field=FloatField())) * 100
    )
    
    # Yesterday's performance for comparison
    yesterday_stats = qs.filter(timestamp__date=yesterday).aggregate(
        count=Count("id"),
        avg_latency=Avg("latency_ms"),
        success_rate=Avg(Case(When(success=True, then=1), default=0, output_field=FloatField())) * 100
    )
    
    # Weekly performance
    weekly_stats = qs.filter(timestamp__date__gte=last_week_start).aggregate(
        total_questions=Count("id"),
        avg_daily_questions=Count("id") / 7,
        weekly_success_rate=Avg(Case(When(success=True, then=1), default=0, output_field=FloatField())) * 100
    )
    
    # Peak hours (top 3 hard-pressed hours)
    peak_hours = (
        qs.annotate(hour=TruncHour("timestamp"))
        .values("hour")
        .annotate(count=Count("id"))
        .order_by("-count")[:3]
    )
    
    # Format peak hours for display
    formatted_peak_hours = []
    for peak in peak_hours:
        hour = peak["hour"]
        if isinstance(hour, str):
            hour = datetime.fromisoformat(hour)
        formatted_peak_hours.append({
            "hour": hour.strftime("%H:00"),
            "count": peak["count"]
        })

    return JsonResponse({
        "today_performance": today_stats,
        "yesterday_comparison": yesterday_stats,
        "weekly_performance": weekly_stats,
        "peak_hours": formatted_peak_hours,
        "system_uptime": "99.9%",  # will change to be real 
    })

@csrf_exempt
@staff_member_required
def analytics_dashboard(request):
    return render(request, "analytics/dashboard.html")