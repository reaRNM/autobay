"""
AI-Driven Shipping Optimization Module

This package provides comprehensive shipping optimization capabilities including:
- Real-time shipping rate calculation from multiple carriers
- AI-powered shipping cost prediction
- Automated carrier selection based on cost, speed, and reliability
- Profit margin integration with shipping costs
- Historical shipping performance analysis and optimization

Author: Shipping Optimization Team
Version: 1.0.0
"""

from shipping_optimizer.core import ShippingOptimizer
from shipping_optimizer.models import (
    Package, ShippingRate, ShippingOption, 
    ShippingPreference, ShippingHistory
)
from shipping_optimizer.prediction import ShippingPredictor
from shipping_optimizer.api import create_app

__all__ = [
    'ShippingOptimizer',
    'ShippingPredictor',
    'Package',
    'ShippingRate',
    'ShippingOption',
    'ShippingPreference',
    'ShippingHistory',
    'create_app'
]