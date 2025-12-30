from django.db import models
from django.conf import settings

class ChatEvent(models.Model):
    CHANNEL_API = "api"
    CHANNEL_CLI = "cli"

    CHANNEL_CHOICES = [
        (CHANNEL_API, "API"),
        (CHANNEL_CLI, "CLI"),
    ]

    timestamp = models.DateTimeField(auto_now_add=True)
    channel = models.CharField(max_length=16, choices=CHANNEL_CHOICES)

    question = models.TextField()
    answer = models.TextField()

    latency_ms = models.IntegerField(null=True, blank=True)
    top_score = models.FloatField(null=True, blank=True)
    num_contexts = models.IntegerField(null=True, blank=True)

    success = models.BooleanField(default=True)
    error_type = models.CharField(max_length=64, null=True, blank=True)

    session_id = models.CharField(
        max_length=64, null=True, blank=True,
        help_text="Optional identifier to group events by session."
    )

    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Extra data for future analytics (e.g., retrieval IDs, client info).",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="chat_events"
    )

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"[{self.channel}] {self.timestamp:%Y-%m-%d %H:%M} - {self.question[:50]}..."
