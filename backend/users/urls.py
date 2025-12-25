# backend/users/urls.py
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

app_name = 'users'

urlpatterns = [
    # Registration
    path('register/', views.UserRegistrationView.as_view(), name='register'),
    
    # Login/Logout
    path('login/', views.user_login_view, name='login'),
    path('logout/', views.user_logout_view, name='logout'),
    
    # Profile
    path('profile/', views.user_profile_view, name='profile'),
        
    # User list (for testing)
    path('list/', views.user_list_view, name='list'),
    
    # Password reset ' later'
]