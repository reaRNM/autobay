"""
Terapeaks product researcher.

This module provides a class for researching Terapeaks product data using both
the Terapeaks API and web scraping.
"""

import logging
from typing import Dict, List, Optional, Any

import aiohttp

from ..config import ResearchConfig
from ..utils import cache_result, create_session, get_proxy_url
from .terapeaks_api import TerapeaksAPI
from .terapeaks_scraper import TerapeaksScraper

logger = logging.getLogger(__name__)


class TerapeaksResearcher:
    """
    Terapeaks product researcher.
    
    This class provides methods for researching Terapeaks product data using both
    the Terapeaks API and web scraping.
    """
    
    def __init__(self, config: ResearchConfig):
        """
        Initialize the Terapeaks researcher.
        
        Args:
            config: Research configuration
        """
        self.config = config
        self.api = TerapeaksAPI(config.terapeaks) if config.terapeaks.use_api else None
        self.scraper = TerapeaksScraper(config.terapeaks) if config.terapeaks.use_scraping else None
    
    @cache_result(ttl=3600)
    async def get_product_data(self, keywords: str) -> Dict[str, Any]:
        """
        Get product data from Terapeaks.
        
        Args:
            keywords: Search keywords
            
        Returns:
            Dict[str, Any]: Product data
        """
        logger.info(f"Getting Terapeaks product data for keywords: {keywords}")
        
        product_data = {}
        
        # Try API first if enabled
        if self.api:
            try:
                # Search for products
                search_results = await self.api.search_products(keywords)
                
                if search_results:
                    # Get the first product
                    products = search_results.get("products", [])
                    if products:
                        product_id = products[0].get("id")
                        
                        # Get product data
                        api_data = await self.api.get_product_data(product_id)
                        
                        # Get sold listings data
                        sold_data = await self.api.get_sold_listings(product_id)
                        
                        # Get active listings data
                        active_data = await self.api.get_active_listings(product_id)
                        
                        # Combine the data
                        product_data = {
                            "source": "terapeaks_api",
                            "product": api_data,
                            "sold_listings": sold_data,
                            "active_listings": active_data
                        }
                        
                        logger.info(f"Got Terapeaks product data from API for keywords: {keywords}")
            except Exception as e:
                logger.error(f"Error getting Terapeaks product data from API: {str(e)}")
        
        # Fall back to scraping if API failed or is disabled
        if not product_data and self.scraper:
            try:
                # Get proxy
                proxy = get_proxy_url(self.config.proxy) if self.config.proxy.enabled else None
                
                # Search for products
                search_results = await self.scraper.search_products(keywords, proxy=proxy)
                
                if search_results:
                    # Get the first product
                    product_id = search_results[0].get("product_id")
                    
                    # Get product details
                    scraper_data = await self.scraper.get_product_details(product_id, proxy=proxy)
                    
                    if scraper_data:
                        product_data = {
                            "source": "terapeaks_scraper",
                            "product": scraper_data
                        }
                        
                        logger.info(f"Got Terapeaks product data from scraper for keywords: {keywords}")
                
                # Close the scraper session
                await self.scraper.close()
            except Exception as e:
                logger.error(f"Error getting Terapeaks product data from scraper: {str(e)}")
        
        # Normalize the data
        if product_data:
            return self._normalize_product_data(product_data)
        
        return {"source": "terapeaks", "error": "No data found"}
    
    def _normalize_product_data(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Terapeaks product data to a common format.
        
        Args:
            product_data: Raw product data
            
        Returns:
            Dict[str, Any]: Normalized product data
        """
        normalized = {
            "source": "terapeaks",
            "raw_data": product_data
        }
        
        try:
            source = product_data.get("source")
            
            if source == "terapeaks_api":
                # Extract data from API response
                product = product_data.get("product", {})
                sold_listings = product_data.get("sold_listings", {})
                active_listings = product_data.get("active_listings", {})
                
                # Extract basic product info
                normalized["title"] = product.get("title")
                normalized["url"] = product.get("url")
                
                # Extract sold listings data
                sold_data = {}
                sold_items = sold_listings.get("items", [])
                if sold_items:
                    prices = [item.get("price") for item in sold_items if item.get("price")]
                    shipping_costs = [item.get("shipping_cost") for item in sold_items if item.get("shipping_cost")]
                    
                    if prices:
                        sold_data["avg_price"] = sum(prices) / len(prices)
                        sold_data["low_price"] = min(prices)
                        sold_data["high_price"] = max(prices)
                    
                    if shipping_costs:
                        sold_data["avg_shipping"] = sum(shipping_costs) / len(shipping_costs)
                    
                    sold_data["total_sold"] = len(sold_items)
                
                normalized["sold_listings"] = sold_data
                
                # Extract active listings data
                active_data = {}
                active_items = active_listings.get("items", [])
                if active_items:
                    prices = [item.get("price") for item in active_items if item.get("price")]
                    watchers = [item.get("watchers") for item in active_items if item.get("watchers")]
                    
                    if prices:
                        active_data["avg_price"] = sum(prices) / len(prices)
                        active_data["low_price"] = min(prices)
                        active_data["high_price"] = max(prices)
                    
                    if watchers:
                        active_data["total_watchers"] = sum(watchers)
                    
                    active_data["total_listings"] = len(active_items)
                
                normalized["active_listings"] = active_data
            
            elif source == "terapeaks_scraper":
                # Extract data from scraper response
                product = product_data.get("product", {})
                
                # Extract basic product info
                normalized["title"] = product.get("title")
                normalized["url"] = product.get("url")
                
                # Extract sold listings data
                sold_data = product.get("sold_listings", {})
                if sold_data:
                    normalized["sold_listings"] = sold_data
                
                # Extract active listings data
                active_data = product.get("active_listings", {})
                if active_data:
                    normalized["active_listings"] = active_data
        except Exception as e:
            logger.error(f"Error normalizing Terapeaks product data: {str(e)}")
        
        return normalized