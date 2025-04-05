"""
Data models for the AI Scoring & NLP Interface.

This module defines the data structures used throughout the package.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import json
from datetime import datetime


@dataclass
class ItemData:
    """Represents auction item data with all relevant attributes."""
    
    item_id: str
    title: str
    description: str = ""
    category: str = ""
    condition: str = ""
    
    # Auction details
    auction_id: str = ""
    auction_end_time: Optional[datetime] = None
    starting_bid: float = 0.0
    current_bid: float = 0.0
    bid_count: int = 0
    
    # Financial estimates
    estimated_value: float = 0.0
    estimated_profit: float = 0.0
    profit_margin: float = 0.0
    
    # Shipping details
    weight: Optional[float] = None
    dimensions: Optional[Dict[str, float]] = None
    estimated_shipping_cost: float = 0.0
    
    # Seller information
    seller_id: str = ""
    seller_rating: float = 0.0
    seller_feedback_count: int = 0
    
    # Historical data
    similar_items_sold: List[Dict[str, Any]] = field(default_factory=list)
    price_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    
    # Existing scores (if any)
    existing_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling datetime objects."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ItemData':
        """Create an ItemData instance from a dictionary."""
        # Handle datetime conversion
        if 'auction_end_time' in data and data['auction_end_time'] and isinstance(data['auction_end_time'], str):
            data['auction_end_time'] = datetime.fromisoformat(data['auction_end_time'])
        
        return cls(**data)


@dataclass
class ScoreComponent:
    """Represents a component of the overall score."""
    
    name: str
    score: float
    weight: float
    factors: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    
    @property
    def weighted_score(self) -> float:
        """Calculate the weighted score."""
        return self.score * self.weight


@dataclass
class ScoringResult:
    """Represents the result of the AI scoring process."""
    
    item_id: str
    priority_score: float
    components: List[ScoreComponent] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    model_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'item_id': self.item_id,
            'priority_score': self.priority_score,
            'components': [
                {
                    'name': comp.name,
                    'score': comp.score,
                    'weight': comp.weight,
                    'weighted_score': comp.weighted_score,
                    'factors': comp.factors,
                    'explanation': comp.explanation
                } for comp in self.components
            ],
            'timestamp': self.timestamp.isoformat(),
            'model_version': self.model_version
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class QueryResult:
    """Represents the result of an NLP query."""
    
    query: str
    parsed_intent: str
    entities: Dict[str, Any]
    items: List[Dict[str, Any]]
    total_items: int
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query': self.query,
            'parsed_intent': self.parsed_intent,
            'entities': self.entities,
            'items': self.items,
            'total_items': self.total_items,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class NLPEntity:
    """Represents an entity extracted from a natural language query."""
    
    name: str
    value: Any
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'confidence': self.confidence
        }