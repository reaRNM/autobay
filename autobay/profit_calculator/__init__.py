"""
Profit Calculator for AI-powered auction research and resale automation.

This package provides tools for evaluating the profitability of potential auction
purchases by calculating the total cost (bid plus fees) and comparing it to
projected sales prices.

Main components:
- config: Configuration management for fee settings
- calculator: Core profit calculation functionality
- utils: Utility functions for validation and formatting
"""

__version__ = "0.1.0"

from .config import FeeConfig
from .calculator import ProfitCalculator, ProfitResult, CostBreakdown
from .utils import validate_numeric, format_currency

__all__ = [
    "FeeConfig",
    "ProfitCalculator",
    "ProfitResult",
    "CostBreakdown",
    "validate_numeric",
    "format_currency",
]