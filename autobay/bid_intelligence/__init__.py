"""
Bid Intelligence Core for AI-powered auction research and resale automation.

This package provides tools for real-time bid monitoring, adjustment, budget optimization,
and fraud detection in live auction environments.

Main components:
- real_time: Real-time bid monitoring and adjustment
- optimization: Budget optimization using knapsack algorithm
- fraud: Fraud detection and alerting
- utils: Utility functions and common data structures
"""

__version__ = "0.1.0"

from .real_time import BidMonitor, BidAdjuster
from .optimization import KnapsackOptimizer, ItemCombination
from .fraud import FraudDetector, FraudAlert
from .utils import AuctionItem, BidData, RiskScore, setup_logging

__all__ = [
    "BidMonitor",
    "BidAdjuster",
    "KnapsackOptimizer",
    "ItemCombination",
    "FraudDetector",
    "FraudAlert",
    "AuctionItem",
    "BidData",
    "RiskScore",
    "setup_logging",
]