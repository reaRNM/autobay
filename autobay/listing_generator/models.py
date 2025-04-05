"""
Data models for the Listing Generator Module.

This module defines the data structures used throughout the package.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
from enum import Enum


class Marketplace(str, Enum):
    """Supported marketplaces."""
    AMAZON = "amazon"
    EBAY = "ebay"
    ETSY = "etsy"
    WALMART = "walmart"


class ListingStatus(str, Enum):
    """Listing status."""
    DRAFT = "draft"
    ACTIVE = "active"
    ENDED = "ended"
    SOLD = "sold"
    ARCHIVED = "archived"


@dataclass
class Product:
    """Represents a product to be listed."""
    
    id: str
    name: str
    brand: Optional[str] = None
    model: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    upc: Optional[str] = None
    ean: Optional[str] = None
    isbn: Optional[str] = None
    asin: Optional[str] = None
    mpn: Optional[str] = None  # Manufacturer Part Number
    features: List[str] = field(default_factory=list)
    specifications: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    condition: str = "New"
    weight_oz: Optional[float] = None
    dimensions: Optional[Dict[str, float]] = None
    image_urls: List[str] = field(default_factory=list)
    msrp: Optional[float] = None  # Manufacturer's Suggested Retail Price
    cost: Optional[float] = None  # Cost to acquire
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'id': self.id,
            'name': self.name,
            'brand': self.brand,
            'model': self.model,
            'category': self.category,
            'subcategory': self.subcategory,
            'upc': self.upc,
            'ean': self.ean,
            'isbn': self.isbn,
            'asin': self.asin,
            'mpn': self.mpn,
            'features': self.features,
            'specifications': self.specifications,
            'description': self.description,
            'condition': self.condition,
            'weight_oz': self.weight_oz,
            'dimensions': self.dimensions,
            'image_urls': self.image_urls,
            'msrp': self.msrp,
            'cost': self.cost,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        return result


@dataclass
class TitleVariation:
    """Represents a title variation for A/B testing."""
    
    id: str
    product_id: str
    title: str
    marketplace: Marketplace
    score: float = 0.0
    impressions: int = 0
    clicks: int = 0
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def ctr(self) -> float:
        """Calculate click-through rate."""
        if self.impressions == 0:
            return 0.0
        return self.clicks / self.impressions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'product_id': self.product_id,
            'title': self.title,
            'marketplace': self.marketplace,
            'score': self.score,
            'impressions': self.impressions,
            'clicks': self.clicks,
            'ctr': self.ctr,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class PricingRecommendation:
    """Represents a pricing recommendation."""
    
    id: str
    product_id: str
    marketplace: Marketplace
    min_price: float
    max_price: float
    recommended_price: float
    competitor_prices: List[float] = field(default_factory=list)
    historical_prices: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    factors: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'product_id': self.product_id,
            'marketplace': self.marketplace,
            'min_price': self.min_price,
            'max_price': self.max_price,
            'recommended_price': self.recommended_price,
            'competitor_prices': self.competitor_prices,
            'historical_prices': self.historical_prices,
            'confidence_score': self.confidence_score,
            'factors': self.factors,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ImageMetadata:
    """Represents metadata for a product image."""
    
    id: str
    product_id: str
    image_url: str
    alt_text: str
    caption: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    enhancement_suggestions: Dict[str, Any] = field(default_factory=dict)
    is_primary: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'product_id': self.product_id,
            'image_url': self.image_url,
            'alt_text': self.alt_text,
            'caption': self.caption,
            'tags': self.tags,
            'enhancement_suggestions': self.enhancement_suggestions,
            'is_primary': self.is_primary,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class Listing:
    """Represents a product listing."""
    
    id: str
    product_id: str
    marketplace: Marketplace
    title: str
    description: str
    price: float
    quantity: int
    status: ListingStatus = ListingStatus.DRAFT
    marketplace_id: Optional[str] = None  # ID assigned by marketplace
    marketplace_url: Optional[str] = None
    title_variations: List[TitleVariation] = field(default_factory=list)
    image_metadata: List[ImageMetadata] = field(default_factory=list)
    pricing_recommendation: Optional[PricingRecommendation] = None
    keywords: List[str] = field(default_factory=list)
    category_id: Optional[str] = None
    shipping_options: Dict[str, Any] = field(default_factory=dict)
    return_policy: Dict[str, Any] = field(default_factory=dict)
    item_specifics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'id': self.id,
            'product_id': self.product_id,
            'marketplace': self.marketplace,
            'title': self.title,
            'description': self.description,
            'price': self.price,
            'quantity': self.quantity,
            'status': self.status,
            'marketplace_id': self.marketplace_id,
            'marketplace_url': self.marketplace_url,
            'keywords': self.keywords,
            'category_id': self.category_id,
            'shipping_options': self.shipping_options,
            'return_policy': self.return_policy,
            'item_specifics': self.item_specifics,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        
        if self.title_variations:
            result['title_variations'] = [v.to_dict() for v in self.title_variations]
        
        if self.image_metadata:
            result['image_metadata'] = [m.to_dict() for m in self.image_metadata]
        
        if self.pricing_recommendation:
            result['pricing_recommendation'] = self.pricing_recommendation.to_dict()
        
        return result


@dataclass
class ListingPerformance:
    """Represents performance metrics for a listing."""
    
    id: str
    listing_id: str
    impressions: int = 0
    clicks: int = 0
    add_to_carts: int = 0
    purchases: int = 0
    revenue: float = 0.0
    search_rank: Optional[int] = None
    search_terms: List[str] = field(default_factory=list)
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    
    @property
    def ctr(self) -> float:
        """Calculate click-through rate."""
        if self.impressions == 0:
            return 0.0
        return self.clicks / self.impressions
    
    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        if self.clicks == 0:
            return 0.0
        return self.purchases / self.clicks
    
    @property
    def add_to_cart_rate(self) -> float:
        """Calculate add-to-cart rate."""
        if self.clicks == 0:
            return 0.0
        return self.add_to_carts / self.clicks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'id': self.id,
            'listing_id': self.listing_id,
            'impressions': self.impressions,
            'clicks': self.clicks,
            'add_to_carts': self.add_to_carts,
            'purchases': self.purchases,
            'revenue': self.revenue,
            'search_rank': self.search_rank,
            'search_terms': self.search_terms,
            'ctr': self.ctr,
            'conversion_rate': self.conversion_rate,
            'add_to_cart_rate': self.add_to_cart_rate,
            'start_date': self.start_date.isoformat()
        }
        
        if self.end_date:
            result['end_date'] = self.end_date.isoformat()
        
        return result


@dataclass
class ABTestResult:
    """Represents the result of an A/B test."""
    
    id: str
    listing_id: str
    variation_a_id: str
    variation_b_id: str
    winner_id: Optional[str] = None
    confidence_level: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'id': self.id,
            'listing_id': self.listing_id,
            'variation_a_id': self.variation_a_id,
            'variation_b_id': self.variation_b_id,
            'winner_id': self.winner_id,
            'confidence_level': self.confidence_level,
            'metrics': self.metrics,
            'start_date': self.start_date.isoformat()
        }
        
        if self.end_date:
            result['end_date'] = self.end_date.isoformat()
        
        return result