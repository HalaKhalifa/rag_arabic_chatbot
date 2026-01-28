from django.urls import path
from .views import (
    analytics_dashboard,
    analytics_log,
    analytics_summary,
    analytics_daily,
    analytics_top_questions,
    analytics_export,
    analytics_engagement,
    analytics_quality_metrics,
    analytics_topic_analysis,
    analytics_performance
)

urlpatterns = [
    path("", analytics_dashboard, name='analytics_dashboard'),
    path('log/', analytics_log, name='analytics_log'),
    path('summary/', analytics_summary, name='analytics_summary'),
    path('daily/', analytics_daily, name='analytics_daily'),
    path('top-questions/', analytics_top_questions, name='analytics_top_questions'),
    path('export/', analytics_export, name='analytics_export'),
    path('engagement/', analytics_engagement, name='analytics_engagement'),
    path('quality-metrics/', analytics_quality_metrics, name='analytics_quality_metrics'),
    path('topic-analysis/', analytics_topic_analysis, name='analytics_topic_analysis'),
    path('performance/', analytics_performance, name='analytics_performance'),
]