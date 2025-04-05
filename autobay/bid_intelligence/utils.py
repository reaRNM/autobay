"""
Utility functions and common data structures for the Bid Intelligence Core.

This module provides common data structures, configuration management,
and utility functions used throughout the Bid Intelligence Core.
"""

import logging
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
from datetime import datetime


class RiskLevel(Enum):
    """Risk level enumeration for auction items and bids."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class RiskScore:
    """
    Risk score for an auction item or bid.
    
    Attributes:
        score: Numerical risk score (0.0 to 1.0, higher means more risky)
        level: Risk level category
        factors: Dictionary of factors contributing to the risk score
    """
    score: float
    level: RiskLevel = RiskLevel.MEDIUM
    factors: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and set the risk level based on the score."""
        if not 0 <= self.score <= 1:
            raise ValueError("Risk score must be between 0 and 1")
        
        # Set risk level based on score
        if self.score < 0.25:
            self.level = RiskLevel.LOW
        elif self.score < 0.5:
            self.level = RiskLevel.MEDIUM
        elif self.score < 0.75:
            self.level = RiskLevel.HIGH
        else:
            self.level = RiskLevel.VERY_HIGH


@dataclass
class AuctionItem:
    """
    Representation of an auction item.
    
    Attributes:
        item_id: Unique identifier for the item
        title: Title or name of the item
        category: Category of the item
        current_bid: Current highest bid amount
        estimated_value: Estimated market value of the item
        estimated_profit: Estimated profit if purchased at current bid
        risk_score: Risk score for the item
        end_time: Auction end time (timestamp)
        additional_data: Additional item data
    """
    item_id: str
    title: str
    category: str
    current_bid: float
    estimated_value: float
    estimated_profit: float
    risk_score: RiskScore
    end_time: int  # Unix timestamp
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def time_remaining(self) -> int:
        """Get the time remaining in seconds until the auction ends."""
        return max(0, self.end_time - int(time.time()))
    
    @property
    def is_active(self) -> bool:
        """Check if the auction is still active."""
        return self.time_remaining > 0
    
    @property
    def profit_margin(self) -> float:
        """Calculate the profit margin as a percentage."""
        if self.current_bid <= 0:
            return 0.0
        return (self.estimated_profit / self.current_bid) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the auction item to a dictionary."""
        return {
            "item_id": self.item_id,
            "title": self.title,
            "category": self.category,
            "current_bid": self.current_bid,
            "estimated_value": self.estimated_value,
            "estimated_profit": self.estimated_profit,
            "risk_score": {
                "score": self.risk_score.score,
                "level": self.risk_score.level.value,
                "factors": self.risk_score.factors
            },
            "end_time": self.end_time,
            "time_remaining": self.time_remaining,
            "is_active": self.is_active,
            "profit_margin": self.profit_margin,
            "additional_data": self.additional_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuctionItem':
        """Create an auction item from a dictionary."""
        risk_data = data.get("risk_score", {})
        risk_score = RiskScore(
            score=risk_data.get("score", 0.5),
            factors=risk_data.get("factors", {})
        )
        
        return cls(
            item_id=data["item_id"],
            title=data["title"],
            category=data["category"],
            current_bid=data["current_bid"],
            estimated_value=data["estimated_value"],
            estimated_profit=data["estimated_profit"],
            risk_score=risk_score,
            end_time=data["end_time"],
            additional_data=data.get("additional_data", {})
        )


@dataclass
class BidData:
    """
    Representation of a bid event in an auction.
    
    Attributes:
        item_id: ID of the item being bid on
        bidder_id: ID of the bidder
        bid_amount: Bid amount
        timestamp: Time of the bid (Unix timestamp)
        is_auto_bid: Whether the bid was placed automatically
        previous_bid: Previous bid amount (if any)
    """
    item_id: str
    bidder_id: str
    bid_amount: float
    timestamp: int  # Unix timestamp
    is_auto_bid: bool = False
    previous_bid: Optional[float] = None
    
    @property
    def bid_increase(self) -> Optional[float]:
        """Calculate the increase from the previous bid."""
        if self.previous_bid is not None:
            return self.bid_amount - self.previous_bid
        return None
    
    @property
    def bid_increase_percentage(self) -> Optional[float]:
        """Calculate the percentage increase from the previous bid."""
        if self.previous_bid is not None and self.previous_bid > 0:
            return ((self.bid_amount - self.previous_bid) / self.previous_bid) * 100
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the bid data to a dictionary."""
        return {
            "item_id": self.item_id,
            "bidder_id": self.bidder_id,
            "bid_amount": self.bid_amount,
            "timestamp": self.timestamp,
            "is_auto_bid": self.is_auto_bid,
            "previous_bid": self.previous_bid,
            "bid_increase": self.bid_increase,
            "bid_increase_percentage": self.bid_increase_percentage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BidData':
        """Create bid data from a dictionary."""
        return cls(
            item_id=data["item_id"],
            bidder_id=data["bidder_id"],
            bid_amount=data["bid_amount"],
            timestamp=data["timestamp"],
            is_auto_bid=data.get("is_auto_bid", False),
            previous_bid=data.get("previous_bid")
        )


class KafkaSimulator:
    """
    Simulates a Kafka producer/consumer for testing and development.
    
    This class provides a simple in-memory message queue that simulates
    the behavior of Apache Kafka for development and testing purposes.
    """
    
    def __init__(self):
        """Initialize the Kafka simulator."""
        self.topics = {}
        self.consumers = {}
        self.running = False
        self._lock = asyncio.Lock()
    
    async def produce(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Produce a message to a topic.
        
        Args:
            topic: Topic to produce to
            message: Message to produce
        """
        async with self._lock:
            if topic not in self.topics:
                self.topics[topic] = []
            
            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = int(time.time())
            
            self.topics[topic].append(message)
            
            # Notify consumers
            if topic in self.consumers:
                for callback in self.consumers[topic]:
                    asyncio.create_task(callback(message))
    
    async def consume(self, topic: str, callback) -> None:
        """
        Register a consumer for a topic.
        
        Args:
            topic: Topic to consume from
            callback: Callback function to call when a message is produced
        """
        async with self._lock:
            if topic not in self.consumers:
                self.consumers[topic] = []
            
            self.consumers[topic].append(callback)
    
    async def start(self) -> None:
        """Start the Kafka simulator."""
        self.running = True
    
    async def stop(self) -> None:
        """Stop the Kafka simulator."""
        self.running = False
        
        # Clear all topics and consumers
        async with self._lock:
            self.topics = {}
            self.consumers = {}


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for the Bid Intelligence Core.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, log to console only)
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger("bid_intelligence")
    logger.setLevel(getattr(logging, log_level))
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def calculate_risk_score(
    price_volatility: float,
    historical_data_points: int,
    category_risk: float,
    seller_rating: float,
    time_pressure: float
) -> RiskScore:
    """
    Calculate a risk score for an auction item.
    
    Args:
        price_volatility: Volatility of the item's price (0.0 to 1.0)
        historical_data_points: Number of historical data points available
        category_risk: Risk associated with the item's category (0.0 to 1.0)
        seller_rating: Rating of the seller (0.0 to 1.0, higher is better)
        time_pressure: Time pressure to make a decision (0.0 to 1.0)
        
    Returns:
        RiskScore: Calculated risk score
    """
    # Validate inputs
    price_volatility = max(0.0, min(1.0, price_volatility))
    category_risk = max(0.0, min(1.0, category_risk))
    seller_rating = max(0.0, min(1.0, seller_rating))
    time_pressure = max(0.0, min(1.0, time_pressure))
    
    # Calculate data confidence factor (more data points = lower risk)
    data_confidence = 1.0 / (1.0 + 0.1 * historical_data_points)
    
    # Invert seller rating (higher rating = lower risk)
    seller_risk = 1.0 - seller_rating
    
    # Calculate weighted risk score
    factors = {
        "price_volatility": price_volatility * 0.3,
        "data_confidence": data_confidence * 0.2,
        "category_risk": category_risk * 0.15,
        "seller_risk": seller_risk * 0.25,
        "time_pressure": time_pressure * 0.1
    }
    
    score = sum(factors.values())
    
    return RiskScore(score=score, factors=factors)


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_file, 'r') as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_file: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_file: Path to the configuration file
    """
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)