"""
Configuration management for the Profit Calculator.

This module provides classes for managing fee configurations used in profit calculations.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Any
import json
import os


@dataclass
class FeeConfig:
    """
    Configuration for fee settings used in profit calculations.
    
    Attributes:
        buyer_premium_rate: Percentage rate for buyer premium (0.0 to 1.0)
        sales_tax_rate: Percentage rate for sales tax (0.0 to 1.0)
        platform_fee_rate: Percentage rate for selling platform fees (0.0 to 1.0)
        shipping_cost: Default shipping cost (flat fee)
        additional_fees: Dictionary of additional fees (name: amount)
        use_tiered_platform_fees: Whether to use tiered platform fees
        tiered_platform_fees: Dictionary of tiered platform fees (threshold: rate)
    """
    
    buyer_premium_rate: float = 0.15  # Default 15%
    sales_tax_rate: float = 0.07  # Default 7%
    platform_fee_rate: float = 0.1275  # Default 12.75% (eBay standard)
    shipping_cost: float = 10.0  # Default $10
    additional_fees: Dict[str, float] = field(default_factory=dict)
    use_tiered_platform_fees: bool = False
    tiered_platform_fees: Dict[float, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration values after initialization."""
        self._validate_rates()
    
    def _validate_rates(self):
        """Validate that all rate values are between 0 and 1."""
        if not 0 <= self.buyer_premium_rate <= 1:
            raise ValueError("Buyer premium rate must be between 0 and 1")
        
        if not 0 <= self.sales_tax_rate <= 1:
            raise ValueError("Sales tax rate must be between 0 and 1")
        
        if not 0 <= self.platform_fee_rate <= 1:
            raise ValueError("Platform fee rate must be between 0 and 1")
        
        if self.shipping_cost < 0:
            raise ValueError("Shipping cost cannot be negative")
        
        # Validate tiered platform fees
        for threshold, rate in self.tiered_platform_fees.items():
            if not isinstance(threshold, (int, float)) or threshold < 0:
                raise ValueError(f"Invalid tier threshold: {threshold}")
            if not 0 <= rate <= 1:
                raise ValueError(f"Tier rate must be between 0 and 1: {rate}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FeeConfig':
        """
        Create a configuration instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            FeeConfig: Configuration instance
        """
        # Filter out unknown keys
        valid_keys = {
            'buyer_premium_rate', 'sales_tax_rate', 'platform_fee_rate',
            'shipping_cost', 'additional_fees', 'use_tiered_platform_fees',
            'tiered_platform_fees'
        }
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'FeeConfig':
        """
        Create a configuration instance from a JSON string.
        
        Args:
            json_str: JSON string containing configuration values
            
        Returns:
            FeeConfig: Configuration instance
        """
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'FeeConfig':
        """
        Create a configuration instance from a JSON file.
        
        Args:
            file_path: Path to the JSON configuration file
            
        Returns:
            FeeConfig: Configuration instance
        """
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as a dictionary
        """
        return {
            'buyer_premium_rate': self.buyer_premium_rate,
            'sales_tax_rate': self.sales_tax_rate,
            'platform_fee_rate': self.platform_fee_rate,
            'shipping_cost': self.shipping_cost,
            'additional_fees': self.additional_fees,
            'use_tiered_platform_fees': self.use_tiered_platform_fees,
            'tiered_platform_fees': self.tiered_platform_fees,
        }
    
    def to_json(self) -> str:
        """
        Convert the configuration to a JSON string.
        
        Returns:
            str: Configuration as a JSON string
        """
        return json.dumps(self.to_dict(), indent=2)
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save the configuration to a JSON file.
        
        Args:
            file_path: Path to the JSON configuration file
        """
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_platform_fee_rate(self, sale_price: float) -> float:
        """
        Get the platform fee rate based on the sale price.
        
        If tiered platform fees are enabled, this will return the appropriate
        rate based on the sale price. Otherwise, it will return the default rate.
        
        Args:
            sale_price: Sale price to determine the fee rate
            
        Returns:
            float: Platform fee rate
        """
        if not self.use_tiered_platform_fees or not self.tiered_platform_fees:
            return self.platform_fee_rate
        
        # Find the appropriate tier
        applicable_rate = self.platform_fee_rate  # Default
        applicable_threshold = 0
        
        for threshold, rate in sorted(self.tiered_platform_fees.items()):
            if sale_price >= threshold and threshold > applicable_threshold:
                applicable_rate = rate
                applicable_threshold = threshold
        
        return applicable_rate