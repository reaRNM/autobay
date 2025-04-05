"""
Data models for the Shipping Optimization Module.

This module defines the data structures used throughout the package.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json


@dataclass
class Package:
    """Represents a package with weight and dimensions."""
    
    weight_oz: float
    length_in: float
    width_in: float
    height_in: float
    value: float = 0.0
    description: str = ""
    is_fragile: bool = False
    requires_signature: bool = False
    is_hazardous: bool = False
    
    @property
    def weight_lb(self) -> float:
        """Convert weight from ounces to pounds."""
        return self.weight_oz / 16.0
    
    @property
    def volume_cubic_in(self) -> float:
        """Calculate volume in cubic inches."""
        return self.length_in * self.width_in * self.height_in
    
    @property
    def dimensional_weight_lb(self) -> float:
        """Calculate dimensional weight in pounds (using 166 cubic inches per pound)."""
        return self.volume_cubic_in / 166.0
    
    @property
    def billable_weight_lb(self) -> float:
        """Return the greater of actual weight and dimensional weight."""
        return max(self.weight_lb, self.dimensional_weight_lb)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'weight_oz': self.weight_oz,
            'length_in': self.length_in,
            'width_in': self.width_in,
            'height_in': self.height_in,
            'value': self.value,
            'description': self.description,
            'is_fragile': self.is_fragile,
            'requires_signature': self.requires_signature,
            'is_hazardous': self.is_hazardous,
            'weight_lb': self.weight_lb,
            'volume_cubic_in': self.volume_cubic_in,
            'dimensional_weight_lb': self.dimensional_weight_lb,
            'billable_weight_lb': self.billable_weight_lb
        }


@dataclass
class Address:
    """Represents a shipping address."""
    
    street1: str
    city: str
    state: str
    postal_code: str
    country: str = "US"
    street2: Optional[str] = None
    residential: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'street1': self.street1,
            'street2': self.street2,
            'city': self.city,
            'state': self.state,
            'postal_code': self.postal_code,
            'country': self.country,
            'residential': self.residential
        }


@dataclass
class ShippingRate:
    """Represents a shipping rate from a carrier."""
    
    carrier: str
    service: str
    rate: float
    delivery_days: Optional[int] = None
    delivery_date: Optional[datetime] = None
    guaranteed: bool = False
    tracking_included: bool = True
    insurance_included: bool = False
    insurance_cost: float = 0.0
    signature_cost: float = 0.0
    fuel_surcharge: float = 0.0
    other_surcharges: Dict[str, float] = field(default_factory=dict)
    
    @property
    def total_cost(self) -> float:
        """Calculate total shipping cost including all surcharges."""
        return (
            self.rate + 
            self.insurance_cost + 
            self.signature_cost + 
            self.fuel_surcharge + 
            sum(self.other_surcharges.values())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'carrier': self.carrier,
            'service': self.service,
            'rate': self.rate,
            'delivery_days': self.delivery_days,
            'guaranteed': self.guaranteed,
            'tracking_included': self.tracking_included,
            'insurance_included': self.insurance_included,
            'insurance_cost': self.insurance_cost,
            'signature_cost': self.signature_cost,
            'fuel_surcharge': self.fuel_surcharge,
            'other_surcharges': self.other_surcharges,
            'total_cost': self.total_cost
        }
        
        if self.delivery_date:
            result['delivery_date'] = self.delivery_date.isoformat()
            
        return result


@dataclass
class ShippingOption:
    """Represents a shipping option with rate and score."""
    
    rate: ShippingRate
    cost_score: float
    speed_score: float
    reliability_score: float
    overall_score: float
    is_recommended: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rate': self.rate.to_dict(),
            'cost_score': self.cost_score,
            'speed_score': self.speed_score,
            'reliability_score': self.reliability_score,
            'overall_score': self.overall_score,
            'is_recommended': self.is_recommended
        }


@dataclass
class ShippingPreference:
    """Represents user preferences for shipping."""
    
    user_id: str
    preferred_carriers: List[str] = field(default_factory=list)
    excluded_carriers: List[str] = field(default_factory=list)
    cost_importance: float = 0.5  # 0.0 to 1.0
    speed_importance: float = 0.3  # 0.0 to 1.0
    reliability_importance: float = 0.2  # 0.0 to 1.0
    default_package_type: Optional[str] = None
    auto_insurance_threshold: float = 100.0
    require_signature_threshold: float = 200.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'preferred_carriers': self.preferred_carriers,
            'excluded_carriers': self.excluded_carriers,
            'cost_importance': self.cost_importance,
            'speed_importance': self.speed_importance,
            'reliability_importance': self.reliability_importance,
            'default_package_type': self.default_package_type,
            'auto_insurance_threshold': self.auto_insurance_threshold,
            'require_signature_threshold': self.require_signature_threshold
        }


@dataclass
class ShippingHistory:
    """Represents a historical shipping record."""
    
    user_id: str
    package: Package
    origin_address: Address
    destination_address: Address
    selected_rate: ShippingRate
    actual_cost: float
    ship_date: datetime
    delivery_date: Optional[datetime] = None
    tracking_number: Optional[str] = None
    delivery_status: str = "unknown"
    delivery_issues: List[str] = field(default_factory=list)
    customer_rating: Optional[int] = None
    
    @property
    def delivery_days(self) -> Optional[int]:
        """Calculate actual delivery days if both dates are available."""
        if self.ship_date and self.delivery_date:
            return (self.delivery_date - self.ship_date).days
        return None
    
    @property
    def cost_accuracy(self) -> Optional[float]:
        """Calculate the accuracy of the rate prediction."""
        if self.selected_rate and self.actual_cost > 0:
            return self.selected_rate.total_cost / self.actual_cost
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'user_id': self.user_id,
            'package': self.package.to_dict(),
            'origin_address': self.origin_address.to_dict(),
            'destination_address': self.destination_address.to_dict(),
            'selected_rate': self.selected_rate.to_dict(),
            'actual_cost': self.actual_cost,
            'ship_date': self.ship_date.isoformat(),
            'tracking_number': self.tracking_number,
            'delivery_status': self.delivery_status,
            'delivery_issues': self.delivery_issues,
            'customer_rating': self.customer_rating,
            'delivery_days': self.delivery_days,
            'cost_accuracy': self.cost_accuracy
        }
        
        if self.delivery_date:
            result['delivery_date'] = self.delivery_date.isoformat()
            
        return result


@dataclass
class CarrierPerformance:
    """Represents carrier performance metrics."""
    
    carrier: str
    service: str
    avg_delivery_days: float
    on_time_percentage: float
    damage_percentage: float
    loss_percentage: float
    cost_accuracy: float
    customer_satisfaction: float
    total_shipments: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'carrier': self.carrier,
            'service': self.service,
            'avg_delivery_days': self.avg_delivery_days,
            'on_time_percentage': self.on_time_percentage,
            'damage_percentage': self.damage_percentage,
            'loss_percentage': self.loss_percentage,
            'cost_accuracy': self.cost_accuracy,
            'customer_satisfaction': self.customer_satisfaction,
            'total_shipments': self.total_shipments
        }


@dataclass
class PriceAdjustment:
    """Represents a price adjustment based on shipping costs."""
    
    original_price: float
    adjusted_price: float
    shipping_cost: float
    handling_cost: float
    packaging_cost: float
    return_rate: float
    profit_margin: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'original_price': self.original_price,
            'adjusted_price': self.adjusted_price,
            'shipping_cost': self.shipping_cost,
            'handling_cost': self.handling_cost,
            'packaging_cost': self.packaging_cost,
            'return_rate': self.return_rate,
            'profit_margin': self.profit_margin
        }