# backend/users/urls.py
from django.urls import path
from . import views

app_name = 'users'

urlpatterns = [
    # Registration
    path('register/', views.UserRegistrationView.as_view(), name='register'),
    
    # Login/Logout
    path('login/', views.user_login_view, name='login'),
    path('logout/', views.user_logout_view, name='logout'),
    
    # Profile
    path('profile/', views.user_profile_view, name='profile'),
    path('delete-account/', views.delete_account_view, name='delete_account'),
       
    # Password reset ' later'
]