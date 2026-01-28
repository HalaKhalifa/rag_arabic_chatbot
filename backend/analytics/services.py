from typing import Optional, Mapping, Any

from .models import ChatEvent


def log_chat_event(
    *,
    user=None,
    channel: str,
    question: str,
    answer: str,
    latency_ms: Optional[int] = None,
    top_score: Optional[float] = None,
    num_contexts: Optional[int] = None,
    success: bool = True,
    error_type: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ChatEvent:
    """
    Store a single chat interaction for analytics.
    """
    event = ChatEvent.objects.create(
        user=user,
        channel=channel,
        question=question,
        answer=answer,
        latency_ms=latency_ms,
        top_score=top_score,
        num_contexts=num_contexts,
        success=success,
        error_type=error_type,
        session_id=session_id,
        metadata=dict(metadata or {}),
    )
    return event
