"""
Utility functions for the workflow automation module.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional


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
    logger = logging.getLogger("workflow_automation")
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


def format_datetime(dt: datetime) -> str:
    """
    Format a datetime object as a string.
    
    Args:
        dt: Datetime object
    
    Returns:
        Formatted datetime string
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_datetime(dt_str: str) -> datetime:
    """
    Parse a datetime string.
    
    Args:
        dt_str: Datetime string
    
    Returns:
        Datetime object
    """
    return datetime.fromisoformat(dt_str)


def format_currency(amount: float) -> str:
    """
    Format a currency amount.
    
    Args:
        amount: Currency amount
    
    Returns:
        Formatted currency string
    """
    return f"${amount:.2f}"


def format_percentage(value: float) -> str:
    """
    Format a percentage value.
    
    Args:
        value: Percentage value (0-1)
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.2f}%"


def serialize_datetime(obj: Any) -> Any:
    """
    Serialize datetime objects for JSON.
    
    Args:
        obj: Object to serialize
    
    Returns:
        Serialized object
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def json_dumps(obj: Any) -> str:
    """
    Convert an object to a JSON string.
    
    Args:
        obj: Object to convert
    
    Returns:
        JSON string
    """
    return json.dumps(obj, default=serialize_datetime)


def json_loads(json_str: str) -> Any:
    """
    Convert a JSON string to an object.
    
    Args:
        json_str: JSON string
    
    Returns:
        Parsed object
    """
    return json.loads(json_str)


def truncate_string(s: str, max_length: int = 100) -> str:
    """
    Truncate a string to a maximum length.
    
    Args:
        s: String to truncate
        max_length: Maximum length
    
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."


def retry_on_exception(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry a function on exception.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
    
    Returns:
        Decorated function
    """
    import time
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for retry in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if retry < max_retries:
                        time.sleep(delay)
                    else:
                        raise last_exception
        return wrapper
    return decorator