"""
Daily Workflow Automation Module for AI-powered auction research and resale automation.

This module automates the entire daily process from data collection to bid recommendations
and notifications, ensuring efficiency and minimal manual intervention.
"""

__version__ = "1.0.0"

from workflow_automation.core import WorkflowManager
from workflow_automation.models import (
    WorkflowStatus,
    TaskStatus,
    NotificationType,
    BidRecommendation,
    WorkflowLog,
    NotificationChannel
)