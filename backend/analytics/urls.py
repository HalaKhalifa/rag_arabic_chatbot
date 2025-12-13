from django.urls import path
from .views import (
    analytics_dashboard,
    analytics_log,
    analytics_summary,
    analytics_daily,
    analytics_top_questions,
    analytics_export,
)

urlpatterns = [
    path("", analytics_dashboard),
    path("log/", analytics_log),
    path("summary/", analytics_summary),
    path("daily/", analytics_daily),
    path("top-questions/", analytics_top_questions),
    path("export/", analytics_export),
]
