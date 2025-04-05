"""
Product Research Engine for AI-powered auction research and resale automation.

This package provides tools for gathering comprehensive product data from
multiple sources (Amazon, eBay, and Terapeaks) to inform pricing, SEO, and
sales strategy decisions.

Main components:
- amazon: Amazon product data integration
- ebay: eBay product data integration
- terapeaks: Terapeaks product data integration
- aggregator: Data aggregation and normalization
- config: Configuration management
- utils: Utility functions and classes
"""

__version__ = "0.1.0"

from .amazon import AmazonResearcher
from .ebay import EbayResearcher
from .terapeaks import TerapeaksResearcher
from .aggregator import ProductDataAggregator
from .config import ResearchConfig

__all__ = [
    "AmazonResearcher",
    "EbayResearcher",
    "TerapeaksResearcher",
    "ProductDataAggregator",
    "ResearchConfig",
]