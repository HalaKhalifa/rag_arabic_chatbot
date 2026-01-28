from django.conf import settings
from django.http import JsonResponse

def require_api_key(request):
    token = request.headers.get("X-API-KEY")
    if not token or token != settings.API_SECRET:
        return JsonResponse({"error": "Unauthorized"}, status=401)
    return None
