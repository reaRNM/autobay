"""
Utility functions for the auction dashboard.
"""

import math
from typing import Dict, Any, Optional, List, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)

def calculate_risk_score(
    item_data: Dict[str, Any],
    seller_history: Optional[Dict[str, Any]] = None,
    market_volatility: Optional[float] = None,
    user_preferences: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate a risk score for an auction item.
    
    The risk score is a value between 0 and 1, where:
    - 0 represents the lowest risk
    - 1 represents the highest risk
    
    Parameters:
    -----------
    item_data : Dict[str, Any]
        Data about the auction item, including:
        - current_price: Current bid price
        - condition: Item condition
        - seller_rating: Rating of the seller (0-5)
        - days_until_end: Days until auction ends
        - category: Item category
        - has_returns: Whether returns are accepted
        - has_warranty: Whether item has warranty
        - description_length: Length of item description
        - num_bids: Number of bids placed
        - num_watchers: Number of people watching
        - shipping_cost: Cost of shipping
        
    seller_history : Optional[Dict[str, Any]]
        Historical data about the seller, including:
        - avg_rating: Average rating
        - num_reviews: Number of reviews
        - percent_positive: Percentage of positive reviews
        - time_on_platform: Time seller has been on platform (days)
        
    market_volatility : Optional[float]
        Market volatility factor (0-1)
        
    user_preferences : Optional[Dict[str, Any]]
        User preferences for risk calculation, including:
        - category_weights: Custom weights for different risk factors
        - risk_tolerance: User's risk tolerance (0-1)
    
    Returns:
    --------
    float
        Risk score between 0 and 1
    """
    # Default weights for different risk factors
    default_weights = {
        "seller_rating": 0.25,
        "item_condition": 0.15,
        "price_volatility": 0.15,
        "auction_dynamics": 0.15,
        "return_policy": 0.10,
        "description_quality": 0.10,
        "market_factors": 0.10
    }
    
    # Use custom weights if provided
    weights = default_weights
    if user_preferences and "category_weights" in user_preferences:
        weights = {**default_weights, **user_preferences["category_weights"]}
    
    # Initialize risk components
    risk_components = {}
    
    # 1. Seller Rating Risk
    seller_rating = item_data.get("seller_rating", 0)
    seller_risk = max(0, 1 - (seller_rating / 5.0))
    
    # Adjust seller risk based on history if available
    if seller_history:
        num_reviews = seller_history.get("num_reviews", 0)
        time_on_platform = seller_history.get("time_on_platform", 0)
        
        # More reviews and longer time on platform reduce risk
        review_factor = min(1, num_reviews / 100)  # Normalize to 0-1
        time_factor = min(1, time_on_platform / 365)  # Normalize to 0-1
        
        # Adjust seller risk (more history = lower risk)
        history_adjustment = 0.5 * (review_factor + time_factor)
        seller_risk = seller_risk * (1 - history_adjustment)
    
    risk_components["seller_rating"] = seller_risk
    
    # 2. Item Condition Risk
    condition_map = {
        "new": 0.0,
        "like_new": 0.1,
        "very_good": 0.3,
        "good": 0.5,
        "acceptable": 0.7,
        "for_parts": 0.9,
        "unknown": 1.0
    }
    condition = item_data.get("condition", "unknown").lower()
    condition_risk = condition_map.get(condition, 0.8)
    risk_components["item_condition"] = condition_risk
    
    # 3. Price Volatility Risk
    current_price = item_data.get("current_price", 0)
    estimated_value = item_data.get("estimated_value", current_price)
    
    if estimated_value > 0:
        # If price is much lower than estimated value, it might be too good to be true
        price_ratio = current_price / estimated_value
        if price_ratio < 0.5:
            price_risk = 0.8  # Suspiciously low price
        elif price_ratio > 1.5:
            price_risk = 0.7  # Significantly overpriced
        else:
            # Reasonable price range
            price_risk = 0.3 * abs(1 - price_ratio)
    else:
        price_risk = 0.5  # Default if we don't have estimated value
    
    risk_components["price_volatility"] = price_risk
    
    # 4. Auction Dynamics Risk
    days_until_end = item_data.get("days_until_end", 7)
    num_bids = item_data.get("num_bids", 0)
    num_watchers = item_data.get("num_watchers", 0)
    
    # Time pressure increases risk
    time_risk = max(0, 1 - (days_until_end / 7))
    
    # More bids/watchers usually means item is desirable but competitive
    competition_factor = min(1, (num_bids + num_watchers) / 20)
    
    auction_risk = 0.5 * time_risk + 0.5 * competition_factor
    risk_components["auction_dynamics"] = auction_risk
    
    # 5. Return Policy Risk
    has_returns = item_data.get("has_returns", False)
    has_warranty = item_data.get("has_warranty", False)
    
    return_risk = 0.8
    if has_returns:
        return_risk -= 0.5
    if has_warranty:
        return_risk -= 0.3
    
    return_risk = max(0, return_risk)
    risk_components["return_policy"] = return_risk
    
    # 6. Description Quality Risk
    description_length = item_data.get("description_length", 0)
    has_images = item_data.get("has_images", False)
    
    # Longer descriptions and presence of images reduce risk
    description_risk = 0.8
    if description_length > 500:
        description_risk -= 0.5
    elif description_length > 200:
        description_risk -= 0.3
    
    if has_images:
        description_risk -= 0.3
    
    description_risk = max(0, description_risk)
    risk_components["description_quality"] = description_risk
    
    # 7. Market Factors Risk
    category_risk_map = {
        "electronics": 0.6,
        "clothing": 0.4,
        "collectibles": 0.7,
        "home_goods": 0.3,
        "jewelry": 0.8,
        "toys": 0.5,
        "automotive": 0.6,
        "books": 0.2
    }
    
    category = item_data.get("category", "").lower()
    category_risk = category_risk_map.get(category, 0.5)
    
    # Adjust for market volatility if provided
    if market_volatility is not None:
        market_risk = 0.7 * category_risk + 0.3 * market_volatility
    else:
        market_risk = category_risk
    
    risk_components["market_factors"] = market_risk
    
    # Calculate weighted risk score
    risk_score = sum(weights[component] * risk for component, risk in risk_components.items())
    
    # Adjust based on user's risk tolerance if provided
    if user_preferences and "risk_tolerance" in user_preferences:
        risk_tolerance = user_preferences["risk_tolerance"]
        # Higher tolerance means we reduce the perceived risk
        risk_score = risk_score * (1 - 0.5 * risk_tolerance)
    
    # Ensure risk score is between 0 and 1
    risk_score = max(0, min(1, risk_score))
    
    # Log the risk calculation
    logger.debug(f"Risk score calculated: {risk_score:.2f} with components: {risk_components}")
    
    return risk_score

def calculate_confidence_score(
    data_points: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate a confidence score for a prediction or recommendation.
    
    Parameters:
    -----------
    data_points : List[Dict[str, Any]]
        List of data points used for the prediction, each with:
        - value: The data point value
        - source: Source of the data
        - age: Age of the data in days
        - relevance: Relevance score (0-1)
        
    weights : Optional[Dict[str, float]]
        Custom weights for different factors
    
    Returns:
    --------
    float
        Confidence score between 0 and 1
    """
    if not data_points:
        return 0.0
    
    default_weights = {
        "relevance": 0.4,
        "recency": 0.3,
        "source_reliability": 0.3
    }
    
    w = weights if weights else default_weights
    
    total_score = 0
    total_weight = 0
    
    for point in data_points:
        # Calculate recency score (newer is better)
        age = point.get("age", 30)  # Default to 30 days if not specified
        recency_score = math.exp(-0.05 * age)  # Exponential decay
        
        # Source reliability score
        source_map = {
            "official_api": 0.95,
            "verified_seller": 0.9,
            "historical_data": 0.85,
            "user_reported": 0.7,
            "estimated": 0.6,
            "unknown": 0.4
        }
        source = point.get("source", "unknown")
        source_score = source_map.get(source, 0.5)
        
        # Relevance score
        relevance = point.get("relevance", 0.5)
        
        # Calculate weighted score for this data point
        point_score = (
            w["relevance"] * relevance +
            w["recency"] * recency_score +
            w["source_reliability"] * source_score
        )
        
        total_score += point_score
        total_weight += sum(w.values())
    
    # Calculate final confidence score
    if total_weight > 0:
        confidence = total_score / total_weight
    else:
        confidence = 0.5  # Default if no weights
    
    return max(0, min(1, confidence))

def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format a number as currency.
    
    Parameters:
    -----------
    amount : float
        The amount to format
    currency : str
        Currency code (default: USD)
        
    Returns:
    --------
    str
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:.2f}"
    elif currency == "EUR":
        return f"€{amount:.2f}"
    elif currency == "GBP":
        return f"£{amount:.2f}"
    else:
        return f"{amount:.2f} {currency}"

def calculate_roi(cost: float, revenue: float) -> float:
    """
    Calculate Return on Investment (ROI).
    
    Parameters:
    -----------
    cost : float
        The investment cost
    revenue : float
        The revenue or return
        
    Returns:
    --------
    float
        ROI as a percentage
    """
    if cost == 0:
        return 0.0
    
    return ((revenue - cost) / cost) * 100

def parse_auction_end_time(end_time_str: str) -> Tuple[bool, str, int]:
    """
    Parse auction end time and determine urgency.
    
    Parameters:
    -----------
    end_time_str : str
        String representation of end time
        
    Returns:
    --------
    Tuple[bool, str, int]
        (is_urgent, formatted_time_remaining, minutes_remaining)
    """
    from datetime import datetime, timezone
    import dateutil.parser
    
    try:
        # Parse the end time string
        end_time = dateutil.parser.parse(end_time_str)
        
        # Ensure timezone awareness
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        
        # Calculate time remaining
        now = datetime.now(timezone.utc)
        time_diff = end_time - now
        minutes_remaining = int(time_diff.total_seconds() / 60)
        
        # Determine if urgent (less than 30 minutes)
        is_urgent = minutes_remaining < 30
        
        # Format time remaining
        if minutes_remaining < 0:
            formatted_time = "Ended"
        elif minutes_remaining < 60:
            formatted_time = f"{minutes_remaining}m remaining"
        elif minutes_remaining < 1440:  # Less than 24 hours
            hours = minutes_remaining // 60
            mins = minutes_remaining % 60
            formatted_time = f"{hours}h {mins}m remaining"
        else:
            days = minutes_remaining // 1440
            hours = (minutes_remaining % 1440) // 60
            formatted_time = f"{days}d {hours}h remaining"
        
        return is_urgent, formatted_time, minutes_remaining
    
    except Exception as e:
        logger.error(f"Error parsing auction end time: {e}")
        return False, "Unknown", 0