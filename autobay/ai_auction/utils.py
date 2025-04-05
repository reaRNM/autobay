"""
Utility functions for the AI Scoring & NLP Interface.

This module provides utility functions for logging, error handling, and data validation.
"""

import os
import logging
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RiskScore:
    """Represents a risk score with component factors."""
    
    score: float
    factors: Dict[str, float]


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        
    Returns:
        Logger instance
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure logging
    logging_config = {
        'level': numeric_level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if log_file:
        logging_config['filename'] = log_file
        logging_config['filemode'] = 'a'
    
    logging.basicConfig(**logging_config)
    
    # Create and return logger
    logger = logging.getLogger('ai_auction')
    
    return logger


def calculate_risk_score(
    price_volatility: float,
    historical_data_points: int,
    category_risk: float,
    seller_rating: float,
    time_pressure: float
) -> RiskScore:
    """
    Calculate a risk score based on multiple factors.
    
    Args:
        price_volatility: Price volatility (0.0-1.0)
        historical_data_points: Number of historical data points
        category_risk: Category risk factor (0.0-1.0)
        seller_rating: Seller rating (0.0-1.0)
        time_pressure: Time pressure factor (0.0-1.0)
        
    Returns:
        RiskScore object with overall score and factors
    """
    # Define weights for each factor
    weights = {
        'price_volatility': 0.25,
        'data_confidence': 0.15,
        'category_risk': 0.20,
        'seller_rating': 0.25,
        'time_pressure': 0.15
    }
    
    # Calculate data confidence based on historical data points
    # More data points = higher confidence = lower risk
    data_confidence = min(1.0, historical_data_points / 100)
    
    # Calculate individual risk factors (higher value = higher risk)
    factors = {
        'price_volatility': price_volatility,
        'data_confidence': 1.0 - data_confidence,  # Invert for risk
        'category_risk': category_risk,
        'seller_rating': 1.0 - seller_rating,  # Invert for risk
        'time_pressure': time_pressure
    }
    
    # Calculate weighted risk score
    weighted_score = sum(factors[factor] * weights[factor] for factor in factors)
    
    # Return risk score object
    return RiskScore(score=weighted_score, factors=factors)


def validate_item_data(item_data: Dict[str, Any]) -> bool:
    """
    Validate item data structure.
    
    Args:
        item_data: Item data dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['item_id', 'title', 'current_bid', 'estimated_value']
    
    # Check required fields
    for field in required_fields:
        if field not in item_data:
            return False
    
    # Validate numeric fields
    numeric_fields = ['current_bid', 'estimated_value', 'estimated_profit', 'profit_margin']
    for field in numeric_fields:
        if field in item_data and not isinstance(item_data[field], (int, float)):
            return False
    
    return True


def load_json_data(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data
    
    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data


def save_json_data(data: Any, file_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to JSON file
    
    Raises:
        TypeError: If data is not JSON serializable
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)