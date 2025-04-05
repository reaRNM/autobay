"""
Fee calculator for auction items.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ShippingStrategy(Enum):
    """Enum for shipping strategies."""
    CHARGE_SEPARATELY = "charge_separately"
    FREE_SHIPPING = "free_shipping"


@dataclass
class FeeSettings:
    """Settings for fee calculation."""
    # Auction fees
    buyer_premium_rate: float = 0.15  # 15% buyer premium
    sales_tax_rate: float = 0.07  # 7% sales tax
    
    # eBay fees
    ebay_listing_fee: float = 0.35  # $0.35 per listing
    ebay_final_value_fee_rate: float = 0.1275  # 12.75% final value fee
    ebay_final_value_fee_cap: Optional[float] = 750.0  # $750 cap on final value fee
    ebay_below_standard_seller_penalty: float = 0.05  # 5% additional fee for below standard sellers
    ebay_international_fee: float = 0.015  # 1.5% international fee
    
    # Shipping settings
    default_shipping_cost: float = 10.0  # Default shipping cost
    free_shipping_threshold: float = 100.0  # Offer free shipping for items over $100
    
    # Promotional settings
    promotional_discount_rate: float = 0.0  # Promotional discount rate
    
    # Other settings
    handling_fee: float = 1.0  # $1 handling fee
    packaging_cost: float = 2.0  # $2 packaging cost
    
    def __post_init__(self):
        """Validate settings after initialization."""
        if self.buyer_premium_rate < 0 or self.buyer_premium_rate > 1:
            raise ValueError("Buyer premium rate must be between 0 and 1")
        if self.sales_tax_rate < 0 or self.sales_tax_rate > 1:
            raise ValueError("Sales tax rate must be between 0 and 1")
        if self.ebay_final_value_fee_rate < 0 or self.ebay_final_value_fee_rate > 1:
            raise ValueError("eBay final value fee rate must be between 0 and 1")


@dataclass
class ProfitAnalysis:
    """Results of profit analysis."""
    # Input values
    bid_amount: float
    estimated_sale_price: float
    
    # Cost breakdown
    buyer_premium: float
    sales_tax: float
    shipping_cost: float
    ebay_listing_fee: float
    ebay_final_value_fee: float
    other_fees: float
    total_cost: float
    
    # Profit analysis
    min_profit: float
    avg_profit: float
    max_profit: float
    profit_margin: float  # As a percentage
    
    # Recommendations
    shipping_strategy: ShippingStrategy
    recommended_price: float
    
    def __str__(self) -> str:
        """String representation of profit analysis."""
        return (
            f"Profit Analysis:\n"
            f"  Bid Amount: ${self.bid_amount:.2f}\n"
            f"  Estimated Sale Price: ${self.estimated_sale_price:.2f}\n"
            f"  Costs:\n"
            f"    Buyer Premium: ${self.buyer_premium:.2f}\n"
            f"    Sales Tax: ${self.sales_tax:.2f}\n"
            f"    Shipping Cost: ${self.shipping_cost:.2f}\n"
            f"    eBay Listing Fee: ${self.ebay_listing_fee:.2f}\n"
            f"    eBay Final Value Fee: ${self.ebay_final_value_fee:.2f}\n"
            f"    Other Fees: ${self.other_fees:.2f}\n"
            f"    Total Cost: ${self.total_cost:.2f}\n"
            f"  Profit Analysis:\n"
            f"    Min Profit: ${self.min_profit:.2f}\n"
            f"    Avg Profit: ${self.avg_profit:.2f}\n"
            f"    Max Profit: ${self.max_profit:.2f}\n"
            f"    Profit Margin: {self.profit_margin:.2f}%\n"
            f"  Recommendations:\n"
            f"    Shipping Strategy: {self.shipping_strategy.value}\n"
            f"    Recommended Price: ${self.recommended_price:.2f}"
        )


class FeeCalculator:
    """
    Calculator for auction fees and profit analysis.
    
    This class calculates all fees related to purchasing an item from an auction
    and selling it on eBay, providing accurate profit estimates.
    """
    
    def __init__(self, settings: Optional[FeeSettings] = None):
        """
        Initialize the fee calculator.
        
        Args:
            settings: Fee calculation settings
        """
        self.settings = settings or FeeSettings()
        
    def calculate_buyer_premium(self, bid_amount: float) -> float:
        """
        Calculate buyer premium.
        
        Args:
            bid_amount: Auction bid amount
            
        Returns:
            float: Buyer premium amount
        """
        return bid_amount * self.settings.buyer_premium_rate
        
    def calculate_sales_tax(self, amount: float) -> float:
        """
        Calculate sales tax.
        
        Args:
            amount: Amount to calculate tax on
            
        Returns:
            float: Sales tax amount
        """
        return amount * self.settings.sales_tax_rate
        
    def calculate_ebay_listing_fee(self) -> float:
        """
        Calculate eBay listing fee.
        
        Returns:
            float: eBay listing fee
        """
        return self.settings.ebay_listing_fee
        
    def calculate_ebay_final_value_fee(
        self, 
        sale_price: float, 
        is_below_standard: bool = False,
        is_international: bool = False
    ) -> float:
        """
        Calculate eBay final value fee.
        
        Args:
            sale_price: Final sale price
            is_below_standard: Whether the seller is below standard
            is_international: Whether the sale is international
            
        Returns:
            float: eBay final value fee
        """
        rate = self.settings.ebay_final_value_fee_rate
        
        # Add penalty for below standard sellers
        if is_below_standard:
            rate += self.settings.ebay_below_standard_seller_penalty
            
        # Add international fee
        if is_international:
            rate += self.settings.ebay_international_fee
            
        fee = sale_price * rate
        
        # Apply cap if set
        if self.settings.ebay_final_value_fee_cap is not None:
            fee = min(fee, self.settings.ebay_final_value_fee_cap)
            
        return fee
        
    def determine_shipping_strategy(self, sale_price: float) -> ShippingStrategy:
        """
        Determine the optimal shipping strategy.
        
        Args:
            sale_price: Final sale price
            
        Returns:
            ShippingStrategy: Optimal shipping strategy
        """
        if sale_price >= self.settings.free_shipping_threshold:
            return ShippingStrategy.FREE_SHIPPING
        else:
            return ShippingStrategy.CHARGE_SEPARATELY
            
    def calculate_total_cost(
        self,
        bid_amount: float,
        shipping_cost: Optional[float] = None,
        is_below_standard: bool = False,
        is_international: bool = False,
        include_ebay_fees: bool = True
    ) -> Dict[str, float]:
        """
        Calculate the total cost of purchasing and selling an item.
        
        Args:
            bid_amount: Auction bid amount
            shipping_cost: Shipping cost (if None, use default)
            is_below_standard: Whether the seller is below standard
            is_international: Whether the sale is international
            include_ebay_fees: Whether to include eBay fees
            
        Returns:
            Dict[str, float]: Breakdown of costs
        """
        if shipping_cost is None:
            shipping_cost = self.settings.default_shipping_cost
            
        buyer_premium = self.calculate_buyer_premium(bid_amount)
        sales_tax = self.calculate_sales_tax(bid_amount + buyer_premium)
        
        costs = {
            "bid_amount": bid_amount,
            "buyer_premium": buyer_premium,
            "sales_tax": sales_tax,
            "shipping_cost": shipping_cost,
            "handling_fee": self.settings.handling_fee,
            "packaging_cost": self.settings.packaging_cost,
        }
        
        if include_ebay_fees:
            costs["ebay_listing_fee"] = self.calculate_ebay_listing_fee()
            # Note: Final value fee is calculated later when we know the sale price
            
        costs["subtotal"] = sum(costs.values())
        
        return costs
        
    def estimate_sale_price(
        self,
        bid_amount: float,
        markup_factor: float = 1.5,
        min_markup: float = 20.0,
        max_markup: Optional[float] = None
    ) -> float:
        """
        Estimate a reasonable sale price based on the bid amount.
        
        Args:
            bid_amount: Auction bid amount
            markup_factor: Markup factor (e.g., 1.5 for 50% markup)
            min_markup: Minimum markup amount
            max_markup: Maximum markup amount
            
        Returns:
            float: Estimated sale price
        """
        markup = max(bid_amount * (markup_factor - 1), min_markup)
        
        if max_markup is not None:
            markup = min(markup, max_markup)
            
        return bid_amount + markup
        
    def analyze_profit(
        self,
        bid_amount: float,
        estimated_sale_price: Optional[float] = None,
        shipping_cost: Optional[float] = None,
        is_below_standard: bool = False,
        is_international: bool = False,
        price_variation: float = 0.1  # 10% variation for min/max
    ) -> ProfitAnalysis:
        """
        Analyze profit potential for an item.
        
        Args:
            bid_amount: Auction bid amount
            estimated_sale_price: Estimated sale price (if None, will be calculated)
            shipping_cost: Shipping cost (if None, use default)
            is_below_standard: Whether the seller is below standard
            is_international: Whether the sale is international
            price_variation: Variation percentage for min/max profit calculation
            
        Returns:
            ProfitAnalysis: Profit analysis results
        """
        if estimated_sale_price is None:
            estimated_sale_price = self.estimate_sale_price(bid_amount)
            
        if shipping_cost is None:
            shipping_cost = self.settings.default_shipping_cost
            
        # Calculate costs
        costs = self.calculate_total_cost(
            bid_amount,
            shipping_cost,
            is_below_standard,
            is_international
        )
        
        # Determine shipping strategy
        shipping_strategy = self.determine_shipping_strategy(estimated_sale_price)
        
        # Calculate eBay final value fee
        ebay_final_value_fee = self.calculate_ebay_final_value_fee(
            estimated_sale_price,
            is_below_standard,
            is_international
        )
        
        # Calculate total cost
        total_cost = costs["subtotal"] + ebay_final_value_fee
        
        # Calculate profit
        profit = estimated_sale_price - total_cost
        
        # Calculate min/max profit based on price variation
        min_sale_price = estimated_sale_price * (1 - price_variation)
        max_sale_price = estimated_sale_price * (1 + price_variation)
        
        min_ebay_fee = self.calculate_ebay_final_value_fee(
            min_sale_price,
            is_below_standard,
            is_international
        )
        
        max_ebay_fee = self.calculate_ebay_final_value_fee(
            max_sale_price,
            is_below_standard,
            is_international
        )
        
        min_profit = min_sale_price - (costs["subtotal"] + min_ebay_fee)
        max_profit = max_sale_price - (costs["subtotal"] + max_ebay_fee)
        
        # Calculate profit margin
        profit_margin = (profit / estimated_sale_price) * 100
        
        # Calculate recommended price
        # Aim for at least 20% profit margin
        target_margin = 0.2
        min_recommended_price = total_cost / (1 - target_margin)
        
        # Use the higher of the estimated price or minimum recommended price
        recommended_price = max(estimated_sale_price, min_recommended_price)
        
        # Create and return profit analysis
        return ProfitAnalysis(
            bid_amount=bid_amount,
            estimated_sale_price=estimated_sale_price,
            buyer_premium=costs["buyer_premium"],
            sales_tax=costs["sales_tax"],
            shipping_cost=costs["shipping_cost"],
            ebay_listing_fee=costs["ebay_listing_fee"],
            ebay_final_value_fee=ebay_final_value_fee,
            other_fees=costs["handling_fee"] + costs["packaging_cost"],
            total_cost=total_cost,
            min_profit=min_profit,
            avg_profit=profit,
            max_profit=max_profit,
            profit_margin=profit_margin,
            shipping_strategy=shipping_strategy,
            recommended_price=recommended_price
        )
        
    def batch_analyze_profit(
        self,
        items: List[Dict[str, Union[float, str]]],
        **kwargs
    ) -> List[Tuple[Dict[str, Union[float, str]], ProfitAnalysis]]:
        """
        Analyze profit potential for multiple items.
        
        Args:
            items: List of item dictionaries with at least a 'bid_amount' key
            **kwargs: Additional arguments to pass to analyze_profit
            
        Returns:
            List[Tuple[Dict, ProfitAnalysis]]: List of (item, profit analysis) tuples
        """
        results = []
        
        for item in items:
            bid_amount = item.get("bid_amount")
            if bid_amount is None:
                logger.warning(f"Skipping item without bid amount: {item}")
                continue
                
            estimated_sale_price = item.get("estimated_sale_price")
            shipping_cost = item.get("shipping_cost")
            
            analysis = self.analyze_profit(
                bid_amount=bid_amount,
                estimated_sale_price=estimated_sale_price,
                shipping_cost=shipping_cost,
                **kwargs
            )
            
            results.append((item, analysis))
            
        return results