from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.utils.translation import gettext_lazy as _
from .forms import UserRegistrationForm, ProfileUpdateForm
from .models import User
from django.http import JsonResponse
from django.contrib.auth import get_user_model

class UserRegistrationView(CreateView):
    """View for user registration."""
    template_name = 'users/register.html'
    form_class = UserRegistrationForm
    success_url = reverse_lazy('users:profile')
    
    def dispatch(self, request, *args, **kwargs):
        # If user is already authenticated, redirect to profile
        if request.user.is_authenticated:
            messages.info(request, _('You are already logged in.'))
            return redirect('users:profile')
        return super().dispatch(request, *args, **kwargs)
    
    def form_valid(self, form):
        # Save the user
        user = form.save()
        
        # Log the user in
        login(self.request, user)
        
        # Show success message
        messages.success(
            self.request, 
            _('Registration successful! Welcome to Bosala AI.')
        )
        
        return super().form_valid(form)
    
    def form_invalid(self, form):
        # Show error messages
        for field, errors in form.errors.items():
            for error in errors:
                messages.error(self.request, f"{field}: {error}")
        return super().form_invalid(form)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


def user_login_view(request):
    """View for user login (separate from admin login)."""
    # If user is already authenticated, redirect to home
    if request.user.is_authenticated:
        messages.info(request, _('You are already logged in.'))
        return redirect('/')  # Changed from 'users:profile'
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        # Authenticate user
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            # Check if user is a regular user
            if user.is_staff or user.user_type == User.UserType.ADMIN:
                return redirect('admin:login')
            
            # Login successful
            login(request, user)
            messages.success(request, _('Login successful!'))
            
            # Redirect to home page
            next_page = request.GET.get('next', '/')
            return redirect(next_page)
        else:
            messages.error(request, _('Invalid username or password.'))

    return render(request, 'users/login.html')

@login_required
def user_profile_view(request):
    """View for user profile."""
    user = request.user
    
    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, _('Profile updated successfully!'))
            return redirect('users:profile')
        else:
            messages.error(request, _('Please correct the errors below.'))
    else:
        form = ProfileUpdateForm(instance=user)
    
    context = {
        'form': form,
        'user': user,
    }
    return render(request, 'users/profile.html', context)

@login_required
def delete_account_view(request):
    if request.method != 'POST':
        return redirect('users:profile')

    user = request.user
    password = request.POST.get('password')

    if not password:
        messages.error(request, 'يرجى إدخال كلمة المرور.')
        return redirect('users:profile')

    if not user.check_password(password):
        messages.error(request, 'كلمة المرور غير صحيحة. فشل حذف الحساب.')
        return redirect('users:profile')
    logout(request)
    user.delete()

    messages.success(request, 'تم حذف حسابك بنجاح.')
    return redirect('/')

@login_required
def user_logout_view(request):
    """View for user logout."""
    logout(request)
    return redirect('/')
