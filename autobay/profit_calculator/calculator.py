"""
Core profit calculation functionality.

This module provides classes and functions for calculating the total cost of
auction purchases and estimating profit based on projected sale prices.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import math

from .config import FeeConfig
from .utils import validate_numeric, format_currency


@dataclass
class CostBreakdown:
    """
    Detailed breakdown of costs for an auction purchase.
    
    Attributes:
        bid_amount: Original bid amount
        buyer_premium: Buyer premium fee
        sales_tax: Sales tax amount
        shipping_cost: Shipping cost
        additional_fees: Dictionary of additional fees
        total_cost: Total cost (sum of all fees)
    """
    
    bid_amount: float
    buyer_premium: float
    sales_tax: float
    shipping_cost: float
    additional_fees: Dict[str, float] = field(default_factory=dict)
    total_cost: float = 0.0
    
    def __post_init__(self):
        """Calculate the total cost after initialization."""
        self.total_cost = self.calculate_total()
    
    def calculate_total(self) -> float:
        """
        Calculate the total cost.
        
        Returns:
            float: Total cost
        """
        additional_fees_sum = sum(self.additional_fees.values())
        return (self.bid_amount + self.buyer_premium + self.sales_tax + 
                self.shipping_cost + additional_fees_sum)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the cost breakdown to a dictionary.
        
        Returns:
            Dict[str, Any]: Cost breakdown as a dictionary
        """
        return {
            'bid_amount': self.bid_amount,
            'buyer_premium': self.buyer_premium,
            'sales_tax': self.sales_tax,
            'shipping_cost': self.shipping_cost,
            'additional_fees': self.additional_fees,
            'total_cost': self.total_cost,
        }
    
    def format(self, currency_symbol: str = '$') -> Dict[str, str]:
        """
        Format the cost breakdown with currency symbols.
        
        Args:
            currency_symbol: Currency symbol to use
            
        Returns:
            Dict[str, str]: Formatted cost breakdown
        """
        result = {
            'bid_amount': format_currency(self.bid_amount, currency_symbol),
            'buyer_premium': format_currency(self.buyer_premium, currency_symbol),
            'sales_tax': format_currency(self.sales_tax, currency_symbol),
            'shipping_cost': format_currency(self.shipping_cost, currency_symbol),
            'total_cost': format_currency(self.total_cost, currency_symbol),
        }
        
        # Format additional fees
        formatted_additional_fees = {}
        for name, amount in self.additional_fees.items():
            formatted_additional_fees[name] = format_currency(amount, currency_symbol)
        
        result['additional_fees'] = formatted_additional_fees
        
        return result


@dataclass
class ProfitResult:
    """
    Result of a profit calculation.
    
    Attributes:
        cost_breakdown: Detailed breakdown of costs
        min_sale_price: Minimum projected sale price
        avg_sale_price: Average projected sale price
        max_sale_price: Maximum projected sale price
        min_profit: Profit based on minimum sale price
        avg_profit: Profit based on average sale price
        max_profit: Profit based on maximum sale price
        min_roi: ROI based on minimum sale price
        avg_roi: ROI based on average sale price
        max_roi: ROI based on maximum sale price
        min_profit_margin: Profit margin based on minimum sale price
        avg_profit_margin: Profit margin based on average sale price
        max_profit_margin: Profit margin based on maximum sale price
        platform_fees: Dictionary of platform fees for each sale price
    """
    
    cost_breakdown: CostBreakdown
    min_sale_price: float
    avg_sale_price: float
    max_sale_price: float
    min_profit: float
    avg_profit: float
    max_profit: float
    min_roi: float
    avg_roi: float
    max_roi: float
    min_profit_margin: float
    avg_profit_margin: float
    max_profit_margin: float
    platform_fees: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the profit result to a dictionary.
        
        Returns:
            Dict[str, Any]: Profit result as a dictionary
        """
        return {
            'cost_breakdown': self.cost_breakdown.to_dict(),
            'sale_prices': {
                'min': self.min_sale_price,
                'avg': self.avg_sale_price,
                'max': self.max_sale_price,
            },
            'profits': {
                'min': self.min_profit,
                'avg': self.avg_profit,
                'max': self.max_profit,
            },
            'roi': {
                'min': self.min_roi,
                'avg': self.avg_roi,
                'max': self.max_roi,
            },
            'profit_margins': {
                'min': self.min_profit_margin,
                'avg': self.avg_profit_margin,
                'max': self.max_profit_margin,
            },
            'platform_fees': self.platform_fees,
        }
    
    def to_json(self) -> str:
        """
        Convert the profit result to a JSON string.
        
        Returns:
            str: Profit result as a JSON string
        """
        return json.dumps(self.to_dict(), indent=2)
    
    def format(self, currency_symbol: str = '$') -> Dict[str, Any]:
        """
        Format the profit result with currency symbols.
        
        Args:
            currency_symbol: Currency symbol to use
            
        Returns:
            Dict[str, Any]: Formatted profit result
        """
        result = {
            'cost_breakdown': self.cost_breakdown.format(currency_symbol),
            'sale_prices': {
                'min': format_currency(self.min_sale_price, currency_symbol),
                'avg': format_currency(self.avg_sale_price, currency_symbol),
                'max': format_currency(self.max_sale_price, currency_symbol),
            },
            'profits': {
                'min': format_currency(self.min_profit, currency_symbol),
                'avg': format_currency(self.avg_profit, currency_symbol),
                'max': format_currency(self.max_profit, currency_symbol),
            },
            'roi': {
                'min': f"{self.min_roi:.2f}%",
                'avg': f"{self.avg_roi:.2f}%",
                'max': f"{self.max_roi:.2f}%",
            },
            'profit_margins': {
                'min': f"{self.min_profit_margin:.2f}%",
                'avg': f"{self.avg_profit_margin:.2f}%",
                'max': f"{self.max_profit_margin:.2f}%",
            },
        }
        
        # Format platform fees
        formatted_platform_fees = {}
        for name, amount in self.platform_fees.items():
            formatted_platform_fees[name] = format_currency(amount, currency_symbol)
        
        result['platform_fees'] = formatted_platform_fees
        
        return result
    
    def get_summary(self, currency_symbol: str = '$') -> Dict[str, str]:
        """
        Get a summary of the profit result.
        
        Args:
            currency_symbol: Currency symbol to use
            
        Returns:
            Dict[str, str]: Summary of the profit result
        """
        return {
            'total_cost': format_currency(self.cost_breakdown.total_cost, currency_symbol),
            'avg_sale_price': format_currency(self.avg_sale_price, currency_symbol),
            'avg_profit': format_currency(self.avg_profit, currency_symbol),
            'avg_roi': f"{self.avg_roi:.2f}%",
            'avg_profit_margin': f"{self.avg_profit_margin:.2f}%",
        }


class ProfitCalculator:
    """
    Calculator for estimating profit from auction purchases.
    
    This class provides methods for calculating the total cost of auction
    purchases and estimating profit based on projected sale prices.
    """
    
    def __init__(self, fee_config: Optional[FeeConfig] = None):
        """
        Initialize the profit calculator.
        
        Args:
            fee_config: Fee configuration to use
        """
        self.fee_config = fee_config or FeeConfig()
    
    def calculate_buyer_premium(self, bid_amount: float) -> float:
        """
        Calculate the buyer premium fee.
        
        Args:
            bid_amount: Bid amount
            
        Returns:
            float: Buyer premium fee
        """
        return bid_amount * self.fee_config.buyer_premium_rate
    
    def calculate_sales_tax(self, bid_amount: float, buyer_premium: float) -> float:
        """
        Calculate the sales tax.
        
        Args:
            bid_amount: Bid amount
            buyer_premium: Buyer premium fee
            
        Returns:
            float: Sales tax
        """
        taxable_amount = bid_amount + buyer_premium
        return taxable_amount * self.fee_config.sales_tax_rate
    
    def calculate_platform_fee(self, sale_price: float) -> float:
        """
        Calculate the platform fee.
        
        Args:
            sale_price: Sale price
            
        Returns:
            float: Platform fee
        """
        fee_rate = self.fee_config.get_platform_fee_rate(sale_price)
        return sale_price * fee_rate
    
    def calculate_total_cost(self, bid_amount: float, 
                            shipping_cost: Optional[float] = None,
                            additional_fees: Optional[Dict[str, float]] = None) -> CostBreakdown:
        """
        Calculate the total cost of an auction purchase.
        
        Args:
            bid_amount: Bid amount
            shipping_cost: Shipping cost (if None, use default from config)
            additional_fees: Dictionary of additional fees
            
        Returns:
            CostBreakdown: Detailed breakdown of costs
        """
        # Validate inputs
        bid_amount = validate_numeric(bid_amount, "bid_amount", min_value=0)
        
        if shipping_cost is None:
            shipping_cost = self.fee_config.shipping_cost
        else:
            shipping_cost = validate_numeric(shipping_cost, "shipping_cost", min_value=0)
        
        if additional_fees is None:
            additional_fees = {}
        
        # Calculate fees
        buyer_premium = self.calculate_buyer_premium(bid_amount)
        sales_tax = self.calculate_sales_tax(bid_amount, buyer_premium)
        
        # Create cost breakdown
        return CostBreakdown(
            bid_amount=bid_amount,
            buyer_premium=buyer_premium,
            sales_tax=sales_tax,
            shipping_cost=shipping_cost,
            additional_fees=additional_fees
        )
    
    def calculate_profit(self, bid_amount: float, 
                        min_sale_price: Optional[float] = None,
                        avg_sale_price: Optional[float] = None,
                        max_sale_price: Optional[float] = None,
                        shipping_cost: Optional[float] = None,
                        additional_fees: Optional[Dict[str, float]] = None) -> ProfitResult:
        """
        Calculate the profit from an auction purchase.
        
        Args:
            bid_amount: Bid amount
            min_sale_price: Minimum projected sale price
            avg_sale_price: Average projected sale price
            max_sale_price: Maximum projected sale price
            shipping_cost: Shipping cost (if None, use default from config)
            additional_fees: Dictionary of additional fees
            
        Returns:
            ProfitResult: Detailed profit calculation result
        """
        # Validate inputs
        bid_amount = validate_numeric(bid_amount, "bid_amount", min_value=0)
        
        # If only avg_sale_price is provided, use it for min and max
        if avg_sale_price is not None and min_sale_price is None and max_sale_price is None:
            avg_sale_price = validate_numeric(avg_sale_price, "avg_sale_price", min_value=0)
            min_sale_price = avg_sale_price
            max_sale_price = avg_sale_price
        # If only min and max are provided, calculate avg
        elif min_sale_price is not None and max_sale_price is not None and avg_sale_price is None:
            min_sale_price = validate_numeric(min_sale_price, "min_sale_price", min_value=0)
            max_sale_price = validate_numeric(max_sale_price, "max_sale_price", min_value=0)
            avg_sale_price = (min_sale_price + max_sale_price) / 2
        # Otherwise, validate all three
        else:
            if min_sale_price is None:
                raise ValueError("min_sale_price is required")
            if avg_sale_price is None:
                raise ValueError("avg_sale_price is required")
            if max_sale_price is None:
                raise ValueError("max_sale_price is required")
            
            min_sale_price = validate_numeric(min_sale_price, "min_sale_price", min_value=0)
            avg_sale_price = validate_numeric(avg_sale_price, "avg_sale_price", min_value=0)
            max_sale_price = validate_numeric(max_sale_price, "max_sale_price", min_value=0)
            
            # Ensure min <= avg <= max
            if not (min_sale_price <= avg_sale_price <= max_sale_price):
                raise ValueError(
                    f"Sale prices must satisfy min <= avg <= max, got "
                    f"min={min_sale_price}, avg={avg_sale_price}, max={max_sale_price}"
                )
        
        # Calculate total cost
        cost_breakdown = self.calculate_total_cost(
            bid_amount=bid_amount,
            shipping_cost=shipping_cost,
            additional_fees=additional_fees
        )
        
        # Calculate platform fees
        platform_fees = {
            'min': self.calculate_platform_fee(min_sale_price),
            'avg': self.calculate_platform_fee(avg_sale_price),
            'max': self.calculate_platform_fee(max_sale_price),
        }
        
        # Calculate profits
        min_profit = min_sale_price - platform_fees['min'] - cost_breakdown.total_cost
        avg_profit = avg_sale_price - platform_fees['avg'] - cost_breakdown.total_cost
        max_profit = max_sale_price - platform_fees['max'] - cost_breakdown.total_cost
        
        # Calculate ROI (Return on Investment)
        min_roi = (min_profit / cost_breakdown.total_cost) * 100 if cost_breakdown.total_cost > 0 else 0
        avg_roi = (avg_profit / cost_breakdown.total_cost) * 100 if cost_breakdown.total_cost > 0 else 0
        max_roi = (max_profit / cost_breakdown.total_cost) * 100 if cost_breakdown.total_cost > 0 else 0
        
        # Calculate profit margins
        min_profit_margin = (min_profit / min_sale_price) * 100 if min_sale_price > 0 else 0
        avg_profit_margin = (avg_profit / avg_sale_price) * 100 if avg_sale_price > 0 else 0
        max_profit_margin = (max_profit / max_sale_price) * 100 if max_sale_price > 0 else 0
        
        # Create profit result
        return ProfitResult(
            cost_breakdown=cost_breakdown,
            min_sale_price=min_sale_price,
            avg_sale_price=avg_sale_price,
            max_sale_price=max_sale_price,
            min_profit=min_profit,
            avg_profit=avg_profit,
            max_profit=max_profit,
            min_roi=min_roi,
            avg_roi=avg_roi,
            max_roi=max_roi,
            min_profit_margin=min_profit_margin,
            avg_profit_margin=avg_profit_margin,
            max_profit_margin=max_profit_margin,
            platform_fees=platform_fees
        )
    
    def calculate_break_even_price(self, bid_amount: float,
                                 shipping_cost: Optional[float] = None,
                                 additional_fees: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate the break-even sale price.
        
        Args:
            bid_amount: Bid amount
            shipping_cost: Shipping cost (if None, use default from config)
            additional_fees: Dictionary of additional fees
            
        Returns:
            float: Break-even sale price
        """
        # Calculate total cost
        cost_breakdown = self.calculate_total_cost(
            bid_amount=bid_amount,
            shipping_cost=shipping_cost,
            additional_fees=additional_fees
        )
        
        # Calculate break-even price using the standard platform fee rate
        # Solve for sale_price: sale_price - (sale_price * platform_fee_rate) = total_cost
        # sale_price * (1 - platform_fee_rate) = total_cost
        # sale_price = total_cost / (1 - platform_fee_rate)
        
        platform_fee_rate = self.fee_config.platform_fee_rate
        
        if platform_fee_rate >= 1:
            raise ValueError("Platform fee rate must be less than 1 for break-even calculation")
        
        break_even_price = cost_breakdown.total_cost / (1 - platform_fee_rate)
        
        # If using tiered platform fees, we need to iterate to find the correct break-even price
        if self.fee_config.use_tiered_platform_fees and self.fee_config.tiered_platform_fees:
            # Start with the initial estimate
            current_price = break_even_price
            
            # Iterate until convergence
            for _ in range(10):  # Limit iterations to avoid infinite loops
                fee_rate = self.fee_config.get_platform_fee_rate(current_price)
                new_price = cost_breakdown.total_cost / (1 - fee_rate)
                
                # Check for convergence
                if abs(new_price - current_price) < 0.01:
                    break
                
                current_price = new_price
              < 0.01:
                    break
                
                current_price = new_price
            
            break_even_price = current_price
        
        return break_even_price
    
    def calculate_target_bid(self, target_sale_price: float, target_profit: float,
                           shipping_cost: Optional[float] = None,
                           additional_fees: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate the maximum bid amount to achieve a target profit at a given sale price.
        
        Args:
            target_sale_price: Target sale price
            target_profit: Target profit amount
            shipping_cost: Shipping cost (if None, use default from config)
            additional_fees: Dictionary of additional fees
            
        Returns:
            float: Maximum bid amount
        """
        # Validate inputs
        target_sale_price = validate_numeric(target_sale_price, "target_sale_price", min_value=0)
        target_profit = validate_numeric(target_profit, "target_profit")
        
        if shipping_cost is None:
            shipping_cost = self.fee_config.shipping_cost
        else:
            shipping_cost = validate_numeric(shipping_cost, "shipping_cost", min_value=0)
        
        if additional_fees is None:
            additional_fees = {}
        
        additional_fees_sum = sum(additional_fees.values())
        
        # Calculate platform fee
        platform_fee = self.calculate_platform_fee(target_sale_price)
        
        # Calculate available amount for bid and fees
        available_amount = target_sale_price - platform_fee - target_profit
        
        # Calculate maximum bid amount
        # Solve for bid_amount:
        # available_amount = bid_amount + buyer_premium + sales_tax + shipping_cost + additional_fees
        # available_amount = bid_amount + (bid_amount * buyer_premium_rate) + 
        #                   ((bid_amount + bid_amount * buyer_premium_rate) * sales_tax_rate) + 
        #                   shipping_cost + additional_fees
        
        # Simplify:
        # available_amount = bid_amount * (1 + buyer_premium_rate) * (1 + sales_tax_rate) + 
        #                   shipping_cost + additional_fees
        
        # Solve for bid_amount:
        # bid_amount = (available_amount - shipping_cost - additional_fees) / 
        #             ((1 + buyer_premium_rate) * (1 + sales_tax_rate))
        
        buyer_premium_rate = self.fee_config.buyer_premium_rate
        sales_tax_rate = self.fee_config.sales_tax_rate
        
        bid_amount = (available_amount - shipping_cost - additional_fees_sum) / \
                    ((1 + buyer_premium_rate) * (1 + sales_tax_rate))
        
        # Ensure bid amount is not negative
        return max(0, bid_amount)