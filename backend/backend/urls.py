"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render

def home(request):
    return render(request, "home.html")

def chat_page(request):
    return render(request, "api/chat.html")

def ingest_page(request):
    return render(request, "api/ingest.html")

urlpatterns = [
    path("", home),
    path("chat/", chat_page),
    path("ingest/", ingest_page),
    path("api/", include("api.urls")),
    path("analytics/", include("analytics.urls"))
,

]
