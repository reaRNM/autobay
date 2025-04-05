"""
Caching utilities for the Product Research Engine.
"""

import functools
import hashlib
import json
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

T = TypeVar("T")

# Simple in-memory cache
_cache: Dict[str, Tuple[float, Any]] = {}


def cache_result(
    ttl: int = 3600,
    cache_dir: Optional[str] = None,
    prefix: str = "product_research_cache_"
) -> Callable:
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time-to-live in seconds
        cache_dir: Directory to store cache files (if None, use in-memory cache)
        prefix: Prefix for cache files
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create a cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            key_str = ":".join(key_parts)
            
            # Create a hash of the key string
            cache_key = hashlib.md5(key_str.encode()).hexdigest()
            
            # Check if result is in cache and not expired
            if cache_dir:
                # File-based cache
                cache_file = os.path.join(cache_dir, f"{prefix}{cache_key}.json")
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, "r") as f:
                            cache_data = json.load(f)
                        
                        timestamp = cache_data.get("timestamp", 0)
                        if time.time() - timestamp < ttl:
                            return cache_data.get("result")
                    except Exception:
                        # If there's an error reading the cache, ignore and proceed
                        pass
            else:
                # In-memory cache
                if cache_key in _cache:
                    timestamp, result = _cache[cache_key]
                    if time.time() - timestamp < ttl:
                        return result
            
            # Call the function and cache the result
            result = await func(*args, **kwargs)
            
            if cache_dir:
                # File-based cache
                os.makedirs(cache_dir, exist_ok=True)
                cache_data = {
                    "timestamp": time.time(),
                    "result": result
                }
                with open(cache_file, "w") as f:
                    json.dump(cache_data, f)
            else:
                # In-memory cache
                _cache[cache_key] = (time.time(), result)
            
            return result
        
        return wrapper
    
    return decorator


def clear_cache(cache_dir: Optional[str] = None, prefix: str = "product_research_cache_") -> None:
    """
    Clear the cache.
    
    Args:
        cache_dir: Directory where cache files are stored (if None, clear in-memory cache)
        prefix: Prefix for cache files
    """
    if cache_dir:
        # Clear file-based cache
        if os.path.exists(cache_dir):
            for filename in os.listdir(cache_dir):
                if filename.startswith(prefix):
                    os.remove(os.path.join(cache_dir, filename))
    else:
        # Clear in-memory cache
        _cache.clear()