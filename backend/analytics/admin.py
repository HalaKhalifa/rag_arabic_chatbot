from django.contrib import admin
from .models import ChatEvent

@admin.register(ChatEvent)
class ChatEventAdmin(admin.ModelAdmin):
    list_display = (
        "timestamp",
        "channel",
        "question_short",
        "latency_ms",
        "top_score",
        "num_contexts",
        "success",
    )
    list_filter = ("channel", "success", "error_type")
    search_fields = ("question", "answer")
    date_hierarchy = "timestamp"

    def question_short(self, obj):
        return obj.question[:80]
    question_short.short_description = "Question"
