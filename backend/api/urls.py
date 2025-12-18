from django.urls import path
from . import views

urlpatterns = [
    # Pages
    path("chat/", views.chat_page),
    path("ingest/", views.ingest_page),

    # APIs
    path("health/", views.health_check),
    path("ask/", views.ask),
    path("ingest-api/", views.ingest),
    path("evaluate/", views.evaluate),
]
