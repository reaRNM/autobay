"""
Caching module for shipping rates.

This module provides functionality to cache shipping rates
to reduce API requests and improve response time.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from shipping_optimizer.models import ShippingRate


logger = logging.getLogger(__name__)


class RateCache:
    """
    Cache for shipping rates.
    
    This class provides functionality to cache shipping rates
    to reduce API requests and improve response time.
    """
    
    def __init__(self, ttl: int = 3600):
        """
        Initialize the rate cache.
        
        Args:
            ttl: Cache time-to-live in seconds (default: 1 hour)
        """
        self.cache = {}
        self.ttl = ttl
        logger.info(f"RateCache initialized with TTL of {ttl} seconds")
    
    def get(self, key: str) -> Optional[List[ShippingRate]]:
        """
        Get cached rates for a key.
        
        Args:
            key: Cache key
            
        Returns:
            List of shipping rates or None if not found or expired
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if entry is expired
        if time.time() > entry["expires"]:
            logger.debug(f"Cache entry expired for {key}")
            del self.cache[key]
            return None
        
        logger.debug(f"Cache hit for {key}")
        return entry["rates"]
    
    def set(self, key: str, rates: List[ShippingRate]) -> None:
        """
        Set cached rates for a key.
        
        Args:
            key: Cache key
            rates: List of shipping rates
        """
        self.cache[key] = {
            "rates": rates,
            "expires": time.time() + self.ttl
        }
        logger.debug(f"Cache set for {key} with {len(rates)} rates")
    
    def delete(self, key: str) -> None:
        """
        Delete a cache entry.
        
        Args:
            key: Cache key
        """
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Cache entry deleted for {key}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache = {}
        logger.debug("Cache cleared")
    
    def cleanup(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now > entry["expires"]
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        logger.debug(f"Cache cleanup removed {len(expired_keys)} expired entries")
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        now = time.time()
        total_entries = len(self.cache)
        expired_entries = sum(1 for entry in self.cache.values() if now > entry["expires"])
        valid_entries = total_entries - expired_entries
        
        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "ttl": self.ttl
        }