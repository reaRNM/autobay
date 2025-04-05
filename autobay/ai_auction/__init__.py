"""
AI Scoring & NLP Interface for Auction Research/Resale Automation Tool.

This package provides two main components:
1. AI Scoring: Dynamically scores auction items based on multiple factors
2. NLP Interface: Provides a chat-based interface for natural language queries

Author: AI Auction Team
Version: 1.0.0
"""

from ai_auction.scoring import AIScoringEngine
from ai_auction.nlp import NLPInterface
from ai_auction.models import ItemData, ScoringResult, QueryResult
from ai_auction.utils import setup_logging

__all__ = [
    'AIScoringEngine',
    'NLPInterface',
    'ItemData',
    'ScoringResult',
    'QueryResult',
    'setup_logging'
]