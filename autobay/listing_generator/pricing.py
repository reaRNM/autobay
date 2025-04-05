"""
Pricing module for the Listing Generator.

This module provides functionality for generating pricing recommendations
and dynamic price adjustments based on market data.
"""

import logging
import asyncio
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

from listing_generator.models import Product, Marketplace, PricingRecommendation, ListingPerformance


logger = logging.getLogger(__name__)


class PricingEngine:
    """
    Engine for generating pricing recommendations.
    
    This class provides methods for generating pricing recommendations
    based on market data and historical sales.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PricingEngine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.margin_target = config.get('margin_target', 0.3)
        self.competitor_weight = config.get('competitor_weight', 0.6)
        self.historical_weight = config.get('historical_weight', 0.3)
        self.trend_weight = config.get('trend_weight', 0.1)
        
        # Initialize marketplace-specific pricing factors
        self.marketplace_factors = {
            Marketplace.AMAZON: {
                "fee_percentage": 0.15,  # 15% referral fee
                "fixed_fee": 1.80,       # $1.80 closing fee
                "premium_factor": 1.05    # 5% premium for Amazon
            },
            Marketplace.EBAY: {
                "fee_percentage": 0.1275,  # 12.75% final value fee
                "fixed_fee": 0.30,         # $0.30 per order
                "premium_factor": 0.95      # 5% discount for eBay
            },
            Marketplace.ETSY: {
                "fee_percentage": 0.065,   # 6.5% transaction fee
                "fixed_fee": 0.20,         # $0.20 listing fee
                "premium_factor": 1.10      # 10% premium for Etsy (handmade/unique)
            },
            Marketplace.WALMART: {
                "fee_percentage": 0.15,    # 15% referral fee
                "fixed_fee": 0.0,          # No fixed fee
                "premium_factor": 0.90      # 10% discount for Walmart
            }
        }
        
        logger.info("PricingEngine initialized")
    
    async def get_pricing_recommendation(
        self,
        product: Product,
        marketplace: Marketplace
    ) -> PricingRecommendation:
        """
        Generate a pricing recommendation for a product.
        
        Args:
            product: Product to generate pricing for
            marketplace: Target marketplace
            
        Returns:
            Pricing recommendation
        """
        logger.info(f"Generating pricing recommendation for product {product.id} on {marketplace}")
        
        try:
            # Get competitor prices
            competitor_prices = await self._get_competitor_prices(product, marketplace)
            
            # Get historical prices
            historical_prices = await self._get_historical_prices(product, marketplace)
            
            # Calculate price range
            min_price, max_price = self._calculate_price_range(
                product, 
                competitor_prices, 
                historical_prices, 
                marketplace
            )
            
            # Calculate recommended price
            recommended_price = self._calculate_recommended_price(
                product,
                min_price,
                max_price,
                competitor_prices,
                historical_prices,
                marketplace
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                competitor_prices,
                historical_prices
            )
            
            # Create pricing recommendation
            recommendation = PricingRecommendation(
                id=str(random.randint(1000000, 9999999)),  # Simple ID generation
                product_id=product.id,
                marketplace=marketplace,
                min_price=min_price,
                max_price=max_price,
                recommended_price=recommended_price,
                competitor_prices=competitor_prices,
                historical_prices=[
                    {"date": date.isoformat(), "price": price}
                    for date, price in historical_prices
                ],
                confidence_score=confidence_score,
                factors=self._get_pricing_factors(
                    product,
                    competitor_prices,
                    historical_prices,
                    marketplace
                )
            )
            
            logger.info(f"Generated pricing recommendation: ${recommended_price:.2f} (${min_price:.2f} - ${max_price:.2f})")
            return recommendation
        
        except Exception as e:
            logger.error(f"Error generating pricing recommendation: {e}")
            # Fallback to basic pricing
            return self._generate_fallback_pricing(product, marketplace)
    
    async def get_dynamic_price_adjustment(
        self,
        current_price: float,
        performance: ListingPerformance
    ) -> float:
        """
        Generate a dynamic price adjustment based on performance.
        
        Args:
            current_price: Current price
            performance: Listing performance data
            
        Returns:
            Adjusted price
        """
        logger.info(f"Generating dynamic price adjustment for listing {performance.listing_id}")
        
        try:
            # Calculate adjustment factor based on performance
            adjustment_factor = 1.0  # Default: no change
            
            # If no impressions or very few, no adjustment
            if performance.impressions < 100:
                return current_price
            
            # Calculate metrics
            ctr = performance.ctr
            conversion_rate = performance.conversion_rate
            
            # Adjust based on CTR (click-through rate)
            if ctr < 0.01:  # Very low CTR
                adjustment_factor -= 0.05  # Reduce price by 5%
            elif ctr < 0.02:  # Low CTR
                adjustment_factor -= 0.02  # Reduce price by 2%
            elif ctr > 0.05:  # High CTR
                adjustment_factor += 0.02  # Increase price by 2%
            
            # Adjust based on conversion rate
            if conversion_rate < 0.01:  # Very low conversion
                adjustment_factor -= 0.05  # Reduce price by 5%
            elif conversion_rate < 0.02:  # Low conversion
                adjustment_factor -= 0.02  # Reduce price by 2%
            elif conversion_rate > 0.05:  # High conversion
                adjustment_factor += 0.03  # Increase price by 3%
            
            # Calculate adjusted price
            adjusted_price = current_price * adjustment_factor
            
            # Limit adjustment to reasonable range
            max_adjustment = 0.15  # 15% maximum adjustment
            if adjusted_price > current_price * (1 + max_adjustment):
                adjusted_price = current_price * (1 + max_adjustment)
            elif adjusted_price < current_price * (1 - max_adjustment):
                adjusted_price = current_price * (1 - max_adjustment)
            
            # Round to nearest $0.99
            adjusted_price = round(adjusted_price - 0.01, 0) + 0.99
            
            logger.info(f"Dynamic price adjustment: ${current_price:.2f} -> ${adjusted_price:.2f}")
            return adjusted_price
        
        except Exception as e:
            logger.error(f"Error generating dynamic price adjustment: {e}")
            return current_price
    
    async def _get_competitor_prices(
        self,
        product: Product,
        marketplace: Marketplace
    ) -> List[float]:
        """
        Get competitor prices for a product.
        
        Args:
            product: Product to get prices for
            marketplace: Target marketplace
            
        Returns:
            List of competitor prices
        """
        # In a real implementation, this would call an API or scrape websites
        # For this example, we'll simulate competitor prices
        
        # Base price (use MSRP or cost as reference)
        base_price = product.msrp or (product.cost * 2 if product.cost else 100.0)
        
        # Generate random competitor prices around the base price
        num_competitors = random.randint(5, 15)
        
        # Different marketplaces have different price distributions
        if marketplace == Marketplace.AMAZON:
            # Amazon tends to have more competitive pricing
            mean_factor = 0.95
            std_dev = 0.1
        elif marketplace == Marketplace.EBAY:
            # eBay tends to have more variable pricing
            mean_factor = 0.9
            std_dev = 0.15
        elif marketplace == Marketplace.ETSY:
            # Etsy tends to have higher pricing for unique items
            mean_factor = 1.1
            std_dev = 0.2
        else:
            # Default
            mean_factor = 1.0
            std_dev = 0.1
        
        # Generate prices
        competitor_prices = []
        for _ in range(num_competitors):
            factor = np.random.normal(mean_factor, std_dev)
            # Ensure factor is reasonable
            factor = max(0.7, min(1.3, factor))
            price = base_price * factor
            # Round to nearest $0.99
            price = round(price - 0.01, 0) + 0.99
            competitor_prices.append(price)
        
        return competitor_prices
    
    async def _get_historical_prices(
        self,
        product: Product,
        marketplace: Marketplace
    ) -> List[tuple]:
        """
        Get historical prices for a product.
        
        Args:
            product: Product to get prices for
            marketplace: Target marketplace
            
        Returns:
            List of (date, price) tuples
        """
        # In a real implementation, this would query a database
        # For this example, we'll simulate historical prices
        
        # Base price (use MSRP or cost as reference)
        base_price = product.msrp or (product.cost * 2 if product.cost else 100.0)
        
        # Generate historical prices for the last 90 days
        historical_prices = []
        
        # Create a price trend (slight decrease, increase, or stable)
        trend_type = random.choice(["decrease", "increase", "stable"])
        
        if trend_type == "decrease":
            start_factor = 1.1
            end_factor = 0.9
        elif trend_type == "increase":
            start_factor = 0.9
            end_factor = 1.1
        else:  # stable
            start_factor = 1.0
            end_factor = 1.0
        
        # Generate prices with some randomness
        for days_ago in range(90, 0, -5):  # Every 5 days
            date = datetime.now() - timedelta(days=days_ago)
            
            # Calculate trend factor
            progress = (90 - days_ago) / 90
            trend_factor = start_factor + (end_factor - start_factor) * progress
            
            # Add some noise
            noise = np.random.normal(0, 0.03)
            factor = trend_factor + noise
            
            # Ensure factor is reasonable
            factor = max(0.7, min(1.3, factor))
            
            price = base_price * factor
            # Round to nearest $0.99
            price = round(price - 0.01, 0) + 0.99
            
            historical_prices.append((date, price))
        
        return historical_prices
    
    def _calculate_price_range(
        self,
        product: Product,
        competitor_prices: List[float],
        historical_prices: List[tuple],
        marketplace: Marketplace
    ) -> tuple:
        """
        Calculate the recommended price range.
        
        Args:
            product: Product to calculate range for
            competitor_prices: List of competitor prices
            historical_prices: List of historical prices
            marketplace: Target marketplace
            
        Returns:
            Tuple of (min_price, max_price)
        """
        # Calculate minimum viable price (cost + fees + minimum margin)
        min_margin = 0.1  # 10% minimum margin
        
        # Get marketplace-specific factors
        marketplace_factor = self.marketplace_factors.get(
            marketplace, 
            {"fee_percentage": 0.15, "fixed_fee": 0.0, "premium_factor": 1.0}
        )
        
        fee_percentage = marketplace_factor["fee_percentage"]
        fixed_fee = marketplace_factor["fixed_fee"]
        
        # Calculate minimum price based on cost
        if product.cost:
            min_price = (product.cost + fixed_fee) / (1 - fee_percentage - min_margin)
        else:
            # If no cost available, use competitor prices
            if competitor_prices:
                min_price = min(competitor_prices) * 0.9
            else:
                # Fallback
                min_price = (product.msrp * 0.5) if product.msrp else 10.0
        
        # Calculate maximum price based on competitor prices and historical data
        if competitor_prices:
            competitor_max = np.percentile(competitor_prices, 90)  # 90th percentile
        else:
            competitor_max = min_price * 2
        
        if historical_prices:
            historical_prices_values = [price for _, price in historical_prices]
            historical_max = np.percentile(historical_prices_values, 90)  # 90th percentile
        else:
            historical_max = min_price * 2
        
        # Combine competitor and historical data
        max_price = (competitor_max * 0.7) + (historical_max * 0.3)
        
        # Ensure max price is greater than min price
        max_price = max(max_price, min_price * 1.2)
        
        # Round to nearest $0.99
        min_price = round(min_price - 0.01, 0) + 0.99
        max_price = round(max_price - 0.01, 0) + 0.99
        
        return (min_price, max_price)
    
    def _calculate_recommended_price(
        self,
        product: Product,
        min_price: float,
        max_price: float,
        competitor_prices: List[float],
        historical_prices: List[tuple],
        marketplace: Marketplace
    ) -> float:
        """
        Calculate the recommended price.
        
        Args:
            product: Product to calculate price for
            min_price: Minimum viable price
            max_price: Maximum recommended price
            competitor_prices: List of competitor prices
            historical_prices: List of historical prices
            marketplace: Target marketplace
            
        Returns:
            Recommended price
        """
        # Get marketplace-specific premium factor
        marketplace_factor = self.marketplace_factors.get(
            marketplace, 
            {"fee_percentage": 0.15, "fixed_fee": 0.0, "premium_factor": 1.0}
        )
        premium_factor = marketplace_factor["premium_factor"]
        
        # Calculate competitor price component
        if competitor_prices:
            # Use median price as reference
            competitor_price = np.median(competitor_prices)
        else:
            competitor_price = (min_price + max_price) / 2
        
        # Calculate historical price component
        if historical_prices:
            # Use recent prices with more weight
            recent_prices = [price for _, price in historical_prices[-6:]]  # Last 30 days
            historical_price = np.mean(recent_prices)
            
            # Check for price trends
            if len(historical_prices) >= 6:
                # Calculate trend
                old_prices = [price for _, price in historical_prices[:6]]  # First 30 days
                recent_prices = [price for _, price in historical_prices[-6:]]  # Last 30 days
                
                old_avg = np.mean(old_prices)
                recent_avg = np.mean(recent_prices)
                
                trend_factor = recent_avg / old_avg if old_avg > 0 else 1.0
                
                # Adjust historical price based on trend
                historical_price *= trend_factor
        else:
            historical_price = (min_price + max_price) / 2
        
        # Calculate target margin price
        if product.cost:
            margin_price = product.cost / (1 - self.margin_target)
        else:
            margin_price = (min_price + max_price) / 2
        
        # Combine components with weights
        recommended_price = (
            (competitor_price * self.competitor_weight) +
            (historical_price * self.historical_weight) +
            (margin_price * (1 - self.competitor_weight - self.historical_weight))
        )
        
        # Apply marketplace premium factor
        recommended_price *= premium_factor
        
        # Ensure price is within range
        recommended_price = max(min_price, min(max_price, recommended_price))
        
        # Round to nearest $0.99
        recommended_price = round(recommended_price - 0.01, 0) + 0.99
        
        return recommended_price
    
    def _calculate_confidence_score(
        self,
        competitor_prices: List[float],
        historical_prices: List[tuple]
    ) -> float:
        """
        Calculate confidence score for the pricing recommendation.
        
        Args:
            competitor_prices: List of competitor prices
            historical_prices: List of historical prices
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.5  # Base confidence
        
        # Adjust based on number of competitor prices
        if competitor_prices:
            num_competitors = len(competitor_prices)
            if num_competitors >= 10:
                confidence += 0.2
            elif num_competitors >= 5:
                confidence += 0.1
            
            # Adjust based on variance
            if num_competitors >= 3:
                variance = np.var(competitor_prices)
                mean = np.mean(competitor_prices)
                cv = np.sqrt(variance) / mean if mean > 0 else 0  # Coefficient of variation
                
                if cv < 0.1:  # Low variance
                    confidence += 0.1
                elif cv > 0.3:  # High variance
                    confidence -= 0.1
        else:
            confidence -= 0.2
        
        # Adjust based on historical data
        if historical_prices:
            num_historical = len(historical_prices)
            if num_historical >= 12:  # 60+ days of data
                confidence += 0.2
            elif num_historical >= 6:  # 30+ days of data
                confidence += 0.1
        else:
            confidence -= 0.2
        
        # Ensure confidence is within range
        confidence = max(0.1, min(0.9, confidence))
        
        return confidence
    
    def _get_pricing_factors(
        self,
        product: Product,
        competitor_prices: List[float],
        historical_prices: List[tuple],
        marketplace: Marketplace
    ) -> Dict[str, Any]:
        """
        Get factors that influenced the pricing recommendation.
        
        Args:
            product: Product to get factors for
            competitor_prices: List of competitor prices
            historical_prices: List of historical prices
            marketplace: Target marketplace
            
        Returns:
            Dictionary of pricing factors
        """
        factors = {}
        
        # Cost and margin factors
        if product.cost:
            factors["cost"] = product.cost
            factors["target_margin"] = self.margin_target
            
            # Get marketplace-specific factors
            marketplace_factor = self.marketplace_factors.get(
                marketplace, 
                {"fee_percentage": 0.15, "fixed_fee": 0.0, "premium_factor": 1.0}
            )
            
            factors["marketplace_fees"] = {
                "percentage": marketplace_factor["fee_percentage"],
                "fixed": marketplace_factor["fixed_fee"]
            }
        
        # Competitor factors
        if competitor_prices:
            factors["competitor_stats"] = {
                "count": len(competitor_prices),
                "min": min(competitor_prices),
                "max": max(competitor_prices),
                "median": np.median(competitor_prices),
                "mean": np.mean(competitor_prices)
            }
        
        # Historical factors
        if historical_prices:
            historical_prices_values = [price for _, price in historical_prices]
            
            factors["historical_stats"] = {
                "count": len(historical_prices),
                "min": min(historical_prices_values),
                "max": max(historical_prices_values),
                "median": np.median(historical_prices_values),
                "mean": np.mean(historical_prices_values)
            }
            
            # Calculate trend
            if len(historical_prices) >= 6:
                old_prices = [price for _, price in historical_prices[:6]]
                recent_prices = [price for _, price in historical_prices[-6:]]
                
                old_avg = np.mean(old_prices)
                recent_avg = np.mean(recent_prices)
                
                if old_avg > 0:
                    trend_percentage = ((recent_avg / old_avg) - 1) * 100
                    factors["price_trend"] = {
                        "direction": "up" if trend_percentage > 0 else "down",
                        "percentage": abs(trend_percentage)
                    }
        
        # Marketplace factors
        factors["marketplace_factor"] = self.marketplace_factors.get(
            marketplace, 
            {"premium_factor": 1.0}
        )["premium_factor"]
        
        return factors
    
    def _generate_fallback_pricing(
        self,
        product: Product,
        marketplace: Marketplace
    ) -> PricingRecommendation:
        """
        Generate fallback pricing if the normal calculation fails.
        
        Args:
            product: Product to generate pricing for
            marketplace: Target marketplace
            
        Returns:
            Fallback pricing recommendation
        """
        # Use MSRP or cost as reference
        if product.msrp:
            reference_price = product.msrp
        elif product.cost:
            reference_price = product.cost * 2
        else:
            reference_price = 100.0
        
        # Get marketplace-specific factors
        marketplace_factor = self.marketplace_factors.get(
            marketplace, 
            {"fee_percentage": 0.15, "fixed_fee": 0.0, "premium_factor": 1.0}
        )
        
        premium_factor = marketplace_factor["premium_factor"]
        
        # Apply marketplace factor
        recommended_price = reference_price * premium_factor
        
        # Calculate min and max prices
        min_price = recommended_price * 0.8
        max_price = recommended_price * 1.2
        
        # Round to nearest $0.99
        min_price = round(min_price - 0.01, 0) + 0.99
        max_price = round(max_price - 0.01, 0) + 0.99
        recommended_price = round(recommended_price - 0.01, 0) + 0.99
        
        # Create recommendation
        recommendation = PricingRecommendation(
            id=str(random.randint(1000000, 9999999)),
            product_id=product.id,
            marketplace=marketplace,
            min_price=min_price,
            max_price=max_price,
            recommended_price=recommended_price,
            competitor_prices=[],
            historical_prices=[],
            confidence_score=0.3,
            factors={
                "reference_price": reference_price,
                "marketplace_factor": premium_factor,
                "fallback": True
            }
        )
        
        logger.info(f"Generated fallback pricing recommendation: ${recommended_price:.2f}")
        return recommendation