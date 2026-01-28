# backend/users/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils.translation import gettext_lazy as _

class User(AbstractUser):
    """Custom User model with additional profile fields."""
    
    # User type choices
    class UserType(models.TextChoices):
        ADMIN = 'admin', _('Admin')
        REGULAR = 'user', _('Regular User')
    
    # Additional fields
    user_type = models.CharField(
        max_length=20,
        choices=UserType.choices,
        default=UserType.REGULAR,
        verbose_name=_('User Type')
    )
    
    # Profile fields
    phone = models.CharField(
        max_length=20,
        blank=True,
        null=True,
        verbose_name=_('Phone Number')
    )
    
    bio = models.TextField(
        blank=True,
        null=True,
        verbose_name=_('Bio/Description')
    )
    
    # Timestamps for profile updates
    profile_updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name=_('Profile Last Updated')
    )
    
    class Meta:
        verbose_name = _('User')
        verbose_name_plural = _('Users')
    
    def __str__(self):
        return f"{self.username} ({self.get_user_type_display()})"
    
    def is_regular_user(self):
        """Check if user is a regular user (not admin/staff)."""
        return self.user_type == self.UserType.REGULAR and not self.is_staff
    
    def is_admin_user(self):
        """Check if user is an admin user."""
        return self.user_type == self.UserType.ADMIN or self.is_staff