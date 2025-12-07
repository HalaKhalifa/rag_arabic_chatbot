from django.urls import path
from .views import health_check, ask, ingest,evaluate

urlpatterns = [
    path("health/", health_check),
    path("ask/", ask),
    path("ingest/", ingest),
    path("evaluate/", evaluate),
]
