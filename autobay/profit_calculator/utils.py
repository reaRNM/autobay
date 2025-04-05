"""
Utility functions for the Profit Calculator.

This module provides utility functions for validation, formatting, and other
common tasks used in profit calculations.
"""

from typing import Union, Optional, Dict, Any
import re


def validate_numeric(value: Any, name: str, min_value: Optional[float] = None, 
                    max_value: Optional[float] = None) -> float:
    """
    Validate that a value is numeric and within the specified range.
    
    Args:
        value: Value to validate
        name: Name of the value (for error messages)
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        
    Returns:
        float: Validated numeric value
        
    Raises:
        TypeError: If the value is not numeric
        ValueError: If the value is outside the specified range
    """
    if value is None:
        raise TypeError(f"{name} cannot be None")
    
    if not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be at least {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be at most {max_value}, got {value}")
    
    return float(value)


def format_currency(value: float, currency_symbol: str = '$', 
                   decimal_places: int = 2) -> str:
    """
    Format a numeric value as currency.
    
    Args:
        value: Numeric value to format
        currency_symbol: Currency symbol to prepend
        decimal_places: Number of decimal places to include
        
    Returns:
        str: Formatted currency string
    """
    return f"{currency_symbol}{value:.{decimal_places}f}"


def parse_currency(value: str) -> float:
    """
    Parse a currency string into a float.
    
    Args:
        value: Currency string to parse
        
    Returns:
        float: Parsed numeric value
        
    Raises:
        ValueError: If the string cannot be parsed as currency
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        raise ValueError(f"Expected string or number, got {type(value).__name__}")
    
    # Remove currency symbols, commas, and whitespace
    clean_value = re.sub(r'[^\d.-]', '', value)
    
    try:
        return float(clean_value)
    except ValueError:
        raise ValueError(f"Could not parse '{value}' as currency")


def validate_dict_numeric(data: Dict[str, Any], required_keys: Optional[list] = None,
                         optional_keys: Optional[list] = None) -> Dict[str, float]:
    """
    Validate that a dictionary contains numeric values for specified keys.
    
    Args:
        data: Dictionary to validate
        required_keys: List of required keys
        optional_keys: List of optional keys
        
    Returns:
        Dict[str, float]: Validated dictionary with numeric values
        
    Raises:
        ValueError: If a required key is missing or a value is not numeric
    """
    result = {}
    
    if required_keys:
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Required key '{key}' is missing")
            result[key] = validate_numeric(data[key], key)
    
    if optional_keys:
        for key in optional_keys:
            if key in data:
                result[key] = validate_numeric(data[key], key)
    
    return result