"""
Data models for the Learning & Feedback Systems Module.

This module defines the data structures used throughout the package.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
from enum import Enum


class FeedbackType(str, Enum):
    """Types of user feedback."""
    PRICING = "pricing"
    LISTING = "listing"
    BIDDING = "bidding"
    SHIPPING = "shipping"
    GENERAL = "general"


class FeedbackSentiment(str, Enum):
    """Sentiment of user feedback."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class AuctionStatus(str, Enum):
    """Status of an auction."""
    ACTIVE = "active"
    SOLD = "sold"
    UNSOLD = "unsold"
    CANCELLED = "cancelled"


class ShippingStatus(str, Enum):
    """Status of a shipping transaction."""
    PENDING = "pending"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    DELAYED = "delayed"
    RETURNED = "returned"
    LOST = "lost"


@dataclass
class ItemCategory:
    """Represents a product category."""
    
    id: str
    name: str
    parent_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'parent_id': self.parent_id,
            'attributes': self.attributes,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class AuctionOutcome:
    """Represents the outcome of an auction."""
    
    id: str
    item_id: str
    user_id: str
    platform: str
    category_id: str
    listing_id: Optional[str] = None
    estimated_price: float = 0.0
    start_price: float = 0.0
    reserve_price: Optional[float] = None
    final_price: Optional[float] = None
    shipping_cost: Optional[float] = None
    fees: Optional[float] = None
    profit: Optional[float] = None
    roi: Optional[float] = None
    views: int = 0
    watchers: int = 0
    questions: int = 0
    bids: int = 0
    status: AuctionStatus = AuctionStatus.ACTIVE
    time_to_sale: Optional[int] = None  # in hours
    listing_quality_score: Optional[float] = None
    ai_confidence_score: Optional[float] = None
    listing_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'id': self.id,
            'item_id': self.item_id,
            'user_id': self.user_id,
            'platform': self.platform,
            'category_id': self.category_id,
            'listing_id': self.listing_id,
            'estimated_price': self.estimated_price,
            'start_price': self.start_price,
            'reserve_price': self.reserve_price,
            'final_price': self.final_price,
            'shipping_cost': self.shipping_cost,
            'fees': self.fees,
            'profit': self.profit,
            'roi': self.roi,
            'views': self.views,
            'watchers': self.watchers,
            'questions': self.questions,
            'bids': self.bids,
            'status': self.status.value,
            'time_to_sale': self.time_to_sale,
            'listing_quality_score': self.listing_quality_score,
            'ai_confidence_score': self.ai_confidence_score,
            'listing_date': self.listing_date.isoformat(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        
        if self.end_date:
            result['end_date'] = self.end_date.isoformat()
        
        return result
    
    @property
    def price_accuracy(self) -> Optional[float]:
        """Calculate price estimation accuracy."""
        if self.final_price is None or self.estimated_price == 0:
            return None
        
        # Calculate accuracy as a percentage (100% = perfect match)
        accuracy = 100 - abs((self.final_price - self.estimated_price) / self.estimated_price * 100)
        
        # Cap at 100% (in case estimated price was lower than final price)
        return min(100.0, max(0.0, accuracy))
    
    @property
    def is_successful(self) -> bool:
        """Determine if the auction was successful."""
        return self.status == AuctionStatus.SOLD and (self.profit or 0) > 0


@dataclass
class UserFeedback:
    """Represents feedback provided by a user."""
    
    id: str
    user_id: str
    item_id: Optional[str] = None
    auction_id: Optional[str] = None
    feedback_type: FeedbackType = FeedbackType.GENERAL
    sentiment: FeedbackSentiment = FeedbackSentiment.NEUTRAL
    rating: Optional[int] = None  # 1-5 scale
    comment: Optional[str] = None
    ai_suggestion: Optional[str] = None
    user_correction: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'item_id': self.item_id,
            'auction_id': self.auction_id,
            'feedback_type': self.feedback_type.value,
            'sentiment': self.sentiment.value,
            'rating': self.rating,
            'comment': self.comment,
            'ai_suggestion': self.ai_suggestion,
            'user_correction': self.user_correction,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ShippingPerformance:
    """Represents the performance of a shipping transaction."""
    
    id: str
    auction_id: str
    user_id: str
    carrier: str
    service_level: str
    package_weight: float  # in oz
    package_dimensions: Dict[str, float]  # in inches
    origin_zip: str
    destination_zip: str
    estimated_cost: float
    actual_cost: float
    estimated_delivery_days: int
    actual_delivery_days: Optional[int] = None
    status: ShippingStatus = ShippingStatus.PENDING
    issues: List[str] = field(default_factory=list)
    customer_satisfaction: Optional[int] = None  # 1-5 scale
    ship_date: Optional[datetime] = None
    delivery_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'id': self.id,
            'auction_id': self.auction_id,
            'user_id': self.user_id,
            'carrier': self.carrier,
            'service_level': self.service_level,
            'package_weight': self.package_weight,
            'package_dimensions': self.package_dimensions,
            'origin_zip': self.origin_zip,
            'destination_zip': self.destination_zip,
            'estimated_cost': self.estimated_cost,
            'actual_cost': self.actual_cost,
            'estimated_delivery_days': self.estimated_delivery_days,
            'actual_delivery_days': self.actual_delivery_days,
            'status': self.status.value,
            'issues': self.issues,
            'customer_satisfaction': self.customer_satisfaction,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        
        if self.ship_date:
            result['ship_date'] = self.ship_date.isoformat()
        
        if self.delivery_date:
            result['delivery_date'] = self.delivery_date.isoformat()
        
        return result
    
    @property
    def cost_accuracy(self) -> float:
        """Calculate shipping cost estimation accuracy."""
        if self.estimated_cost == 0:
            return 0.0
        
        # Calculate accuracy as a percentage (100% = perfect match)
        accuracy = 100 - abs((self.actual_cost - self.estimated_cost) / self.estimated_cost * 100)
        
        # Cap at 100% (in case estimated cost was lower than actual cost)
        return min(100.0, max(0.0, accuracy))
    
    @property
    def delivery_time_accuracy(self) -> Optional[float]:
        """Calculate delivery time estimation accuracy."""
        if self.actual_delivery_days is None or self.estimated_delivery_days == 0:
            return None
        
        # Calculate accuracy as a percentage (100% = perfect match)
        accuracy = 100 - abs((self.actual_delivery_days - self.estimated_delivery_days) / self.estimated_delivery_days * 100)
        
        # Cap at 100% (in case estimated time was lower than actual time)
        return min(100.0, max(0.0, accuracy))


@dataclass
class BidStrategy:
    """Represents a bidding strategy."""
    
    id: str
    user_id: str
    category_id: Optional[str] = None
    name: str = "Default Strategy"
    description: Optional[str] = None
    max_bid_percentage: float = 0.8  # percentage of estimated value
    early_bid_threshold: float = 0.3  # percentage of auction duration
    late_bid_threshold: float = 0.9  # percentage of auction duration
    bid_increment_factor: float = 1.05  # multiplier for minimum bid increment
    max_bid_count: int = 2  # maximum number of bids per auction
    risk_tolerance: float = 0.5  # 0.0 to 1.0 (conservative to aggressive)
    success_rate: float = 0.0  # historical success rate
    average_roi: float = 0.0  # historical average ROI
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'category_id': self.category_id,
            'name': self.name,
            'description': self.description,
            'max_bid_percentage': self.max_bid_percentage,
            'early_bid_threshold': self.early_bid_threshold,
            'late_bid_threshold': self.late_bid_threshold,
            'bid_increment_factor': self.bid_increment_factor,
            'max_bid_count': self.max_bid_count,
            'risk_tolerance': self.risk_tolerance,
            'success_rate': self.success_rate,
            'average_roi': self.average_roi,
            'metadata': self.metadata,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class ModelPerformance:
    """Represents the performance of a model."""
    
    id: str
    model_name: str
    model_version: str
    model_type: str  # pricing, bidding, shipping, etc.
    category_id: Optional[str] = None
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mean_absolute_error: Optional[float] = None
    mean_squared_error: Optional[float] = None
    r_squared: Optional[float] = None
    sample_count: int = 0
    training_duration: float = 0.0  # in seconds
    inference_latency: float = 0.0  # in milliseconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'model_type': self.model_type,
            'category_id': self.category_id,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mean_absolute_error': self.mean_absolute_error,
            'mean_squared_error': self.mean_squared_error,
            'r_squared': self.r_squared,
            'sample_count': self.sample_count,
            'training_duration': self.training_duration,
            'inference_latency': self.inference_latency,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class UserPreference:
    """Represents a user's preferences."""
    
    id: str
    user_id: str
    category_preferences: Dict[str, float] = field(default_factory=dict)  # category_id -> preference weight
    platform_preferences: Dict[str, float] = field(default_factory=dict)  # platform -> preference weight
    price_range_min: Optional[float] = None
    price_range_max: Optional[float] = None
    risk_tolerance: float = 0.5  # 0.0 to 1.0 (conservative to aggressive)
    preferred_condition: List[str] = field(default_factory=list)  # new, used, refurbished, etc.
    preferred_shipping_methods: List[str] = field(default_factory=list)
    preferred_carriers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'category_preferences': self.category_preferences,
            'platform_preferences': self.platform_preferences,
            'price_range_min': self.price_range_min,
            'price_range_max': self.price_range_max,
            'risk_tolerance': self.risk_tolerance,
            'preferred_condition': self.preferred_condition,
            'preferred_shipping_methods': self.preferred_shipping_methods,
            'preferred_carriers': self.preferred_carriers,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }