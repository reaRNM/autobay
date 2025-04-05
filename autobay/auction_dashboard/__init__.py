"""
Dashboard & Mobile Alerts module for AI-powered auction research/resale automation.

This package provides a RESTful API backend, responsive dashboard UI, and mobile
notification functionality for viewing real-time auction data, historical results,
profit trends, and other key performance metrics.

Main components:
- api: Flask-based RESTful API backend
- models: Database models for auction data and user preferences
- services: Business logic and integration with external services
- utils: Utility functions and common data structures
- notifications: Mobile alert functionality
"""

__version__ = "0.1.0"

from .api import create_app
from .models import db, User, AuctionItem, AlertConfig
from .services import DashboardService, MetricsService
from .notifications import NotificationService

__all__ = [
    "create_app",
    "db",
    "User",
    "AuctionItem",
    "AlertConfig",
    "DashboardService",
    "MetricsService",
    "NotificationService",
]