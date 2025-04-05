"""
Utility functions for the Learning & Feedback Systems Module.

This module provides utility functions used throughout the package.
"""

import os
import logging
import sys
from typing import Optional, Dict, Any
import json
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for the application.
    
    Args:
        log_level: Logging level
        log_file: Path to log file
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger("learning_feedback")
    logger.setLevel(getattr(logging, log_level))
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def json_serialize(obj: Any) -> Any:
    """
    Serialize objects to JSON.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    return str(obj)


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to file
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save data to file
    with open(file_path, 'w') as f:
        json.dump(data, f, default=json_serialize, indent=2)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Loaded data
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return {}
    
    # Load data from file
    with open(file_path, 'r') as f:
        return json.load(f)


def format_price(price: float, currency: str = "$") -> str:
    """
    Format a price with currency symbol.
    
    Args:
        price: Price to format
        currency: Currency symbol
        
    Returns:
        Formatted price
    """
    return f"{currency}{price:.2f}"


def calculate_accuracy(estimated: float, actual: float) -> float:
    """
    Calculate accuracy as a percentage.
    
    Args:
        estimated: Estimated value
        actual: Actual value
        
    Returns:
        Accuracy as a percentage
    """
    if estimated == 0:
        return 0.0
    
    # Calculate accuracy as a percentage (100% = perfect match)
    accuracy = 100 - abs((actual - estimated) / estimated * 100)
    
    # Cap at 100% (in case estimated value was lower than actual value)
    return min(100.0, max(0.0, accuracy))