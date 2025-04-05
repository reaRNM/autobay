"""
Learning & Feedback Systems Module

This package provides comprehensive learning and feedback capabilities including:
- Auction outcome feedback collection and processing
- User feedback collection and analysis
- Reinforcement learning for bidding strategy optimization
- Model performance monitoring and improvement
- Reporting and visualization tools

Author: Learning & Feedback Systems Team
Version: 1.0.0
"""

from learning_feedback.core import LearningSystem
from learning_feedback.models import (
    AuctionOutcome, UserFeedback, ModelPerformance, 
    BidStrategy, ShippingPerformance, ItemCategory
)
from learning_feedback.data_collection import (
    AuctionOutcomeCollector, UserFeedbackCollector
)
from learning_feedback.model_training import (
    PricingModelTrainer, BidModelTrainer, ShippingModelTrainer
)
from learning_feedback.reinforcement_learning import (
    BiddingEnvironment, BiddingAgent
)
from learning_feedback.feedback_processing import (
    FeedbackProcessor, UserPreferenceTracker
)
from learning_feedback.performance_monitoring import (
    PerformanceMonitor, DriftDetector
)
from learning_feedback.api import create_app
from learning_feedback.reporting import ReportGenerator

__all__ = [
    'LearningSystem',
    'AuctionOutcome',
    'UserFeedback',
    'ModelPerformance',
    'BidStrategy',
    'ShippingPerformance',
    'ItemCategory',
    'AuctionOutcomeCollector',
    'UserFeedbackCollector',
    'PricingModelTrainer',
    'BidModelTrainer',
    'ShippingModelTrainer',
    'BiddingEnvironment',
    'BiddingAgent',
    'FeedbackProcessor',
    'UserPreferenceTracker',
    'PerformanceMonitor',
    'DriftDetector',
    'create_app',
    'ReportGenerator'
]