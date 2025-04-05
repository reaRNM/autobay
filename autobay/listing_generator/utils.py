"""
Utility functions for the Listing Generator.

This module provides utility functions used throughout the package.
"""

import os
import logging
import sys
from typing import Optional


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
    logger = logging.getLogger("listing_generator")
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
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to ensure it's valid.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Ensure filename is not too long
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext
    
    return filename


def truncate_text(text: str, max_length: int, ellipsis: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        ellipsis: Ellipsis string to append
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Try to truncate at a word boundary
    truncated = text[:max_length - len(ellipsis)]
    last_space = truncated.rfind(" ")
    
    if last_space > max_length * 0.8:  # Only truncate at word boundary if it's not too far back
        truncated = truncated[:last_space]
    
    return truncated + ellipsis


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