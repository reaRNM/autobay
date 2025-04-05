"""
Utility functions for shipping optimization.

This module provides utility functions for shipping optimization.
"""

import os
import logging
import math
import re
import requests
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from shipping_optimizer.models import Address


logger = logging.getLogger(__name__)


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
    logger = logging.getLogger('shipping_optimizer')
    
    return logger


def calculate_distance(origin: Address, destination: Address) -> float:
    """
    Calculate distance between two addresses.
    
    Args:
        origin: Origin address
        destination: Destination address
        
    Returns:
        Distance in miles
    """
    try:
        # In a real implementation, use a geocoding and distance API
        # This is a simplified example using ZIP codes
        
        # Extract ZIP codes
        origin_zip = origin.postal_code[:5]
        destination_zip = destination.postal_code[:5]
        
        # Check if same ZIP code
        if origin_zip == destination_zip:
            return 5.0  # Assume 5 miles within same ZIP
        
        # Check if same state
        if origin.state == destination.state:
            return 50.0  # Assume 50 miles within same state
        
        # Simple distance calculation based on first digit of ZIP
        # (very rough approximation for example purposes only)
        origin_region = int(origin_zip[0])
        destination_region = int(destination_zip[0])
        
        region_distance = abs(origin_region - destination_region) * 300
        
        return max(10.0, region_distance)
    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return 500.0  # Default to 500 miles


def calculate_zone(origin_zip: str, destination_zip: str) -> int:
    """
    Calculate shipping zone based on ZIP codes.
    
    Args:
        origin_zip: Origin ZIP code
        destination_zip: Destination ZIP code
        
    Returns:
        Shipping zone (1-8)
    """
    try:
        # In a real implementation, use a zone calculation API or database
        # This is a simplified example
        
        # Extract first 3 digits of ZIP codes
        origin_prefix = origin_zip[:3]
        destination_prefix = destination_zip[:3]
        
        # Check if same 3-digit prefix
        if origin_prefix == destination_prefix:
            return 1
        
        # Check if same first digit
        if origin_zip[0] == destination_zip[0]:
            return 2
        
        # Calculate zone based on difference between first digits
        diff = abs(int(origin_zip[0]) - int(destination_zip[0]))
        
        if diff == 1:
            return 3
        elif diff == 2:
            return 4
        elif diff == 3:
            return 5
        elif diff == 4:
            return 6
        elif diff == 5:
            return 7
        else:
            return 8
    except Exception as e:
        logger.error(f"Error calculating zone: {e}")
        return 8  # Default to zone 8 (farthest)


def validate_address(address: Address) -> bool:
    """
    Validate an address.
    
    Args:
        address: Address to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required fields
        if not address.street1 or not address.city or not address.state or not address.postal_code:
            return False
        
        # Validate postal code format
        if 
            return False
        
        # Validate postal code format
        if address.country == "US":
            # US ZIP code format (5 digits or ZIP+4)
            if not re.match(r'^\d{5}(-\d{4})?$', address.postal_code):
                return False
        
        # Validate state format for US
        if address.country == "US":
            # US state code (2 letters)
            if not re.match(r'^[A-Z]{2}$', address.state):
                return False
        
        # In a real implementation, use an address validation API
        # This is a simplified example
        
        return True
    except Exception as e:
        logger.error(f"Error validating address: {e}")
        return False


def calculate_packaging_cost(item_price: float) -> float:
    """
    Calculate packaging cost based on item price.
    
    Args:
        item_price: Item price
        
    Returns:
        Packaging cost
    """
    try:
        # Base packaging cost
        base_cost = 1.0
        
        # Add percentage of item price
        percentage_cost = item_price * 0.01  # 1% of item price
        
        # Cap at reasonable amount
        max_cost = 10.0
        
        return min(base_cost + percentage_cost, max_cost)
    except Exception as e:
        logger.error(f"Error calculating packaging cost: {e}")
        return 1.0  # Default to $1


def log_api_request(carrier: str, endpoint: str) -> None:
    """
    Log an API request for rate limiting and monitoring.
    
    Args:
        carrier: Carrier name
        endpoint: API endpoint
    """
    try:
        # In a real implementation, log to database or monitoring system
        # This is a simplified example
        logger.debug(f"API request: {carrier} - {endpoint} - {datetime.now().isoformat()}")
    except Exception as e:
        logger.error(f"Error logging API request: {e}")


def estimate_delivery_date(
    ship_date: datetime,
    delivery_days: int,
    carrier: str,
    service: str
) -> datetime:
    """
    Estimate delivery date based on ship date and delivery days.
    
    Args:
        ship_date: Ship date
        delivery_days: Delivery days
        carrier: Carrier name
        service: Service name
        
    Returns:
        Estimated delivery date
    """
    try:
        # Start with ship date
        delivery_date = ship_date
        
        # Add delivery days (business days only)
        business_days_added = 0
        while business_days_added < delivery_days:
            delivery_date = delivery_date + timedelta(days=1)
            
            # Skip weekends
            if delivery_date.weekday() < 5:  # 0-4 are Monday-Friday
                business_days_added += 1
        
        # Adjust for carrier-specific factors
        if carrier.lower() == "usps" and service.lower() == "priority":
            # USPS Priority often delivers on Saturdays
            pass  # No adjustment needed
        elif carrier.lower() == "fedex" and "overnight" in service.lower():
            # FedEx Overnight delivers on Saturdays for additional fee
            pass  # No adjustment needed
        elif delivery_date.weekday() == 5:  # Saturday
            # Most carriers don't deliver on weekends
            delivery_date = delivery_date + timedelta(days=2)  # Skip to Monday
        elif delivery_date.weekday() == 6:  # Sunday
            # Most carriers don't deliver on weekends
            delivery_date = delivery_date + timedelta(days=1)  # Skip to Monday
        
        return delivery_date
    except Exception as e:
        logger.error(f"Error estimating delivery date: {e}")
        return ship_date + timedelta(days=delivery_days)