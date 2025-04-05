"""
eBay product researcher.

This module provides a class for researching eBay product data using both
the eBay API and web scraping.
"""

import logging
import statistics
from typing import Dict, List, Optional, Any

import aiohttp

from ..config import ResearchConfig
from ..utils import cache_result, create_session, get_proxy_url
from .ebay_api import EbayAPI
from .ebay_scraper import EbayScraper

logger = logging.getLogger(__name__)


class EbayResearcher:
    """
    eBay product researcher.
    
    This class provides methods for researching eBay product data using both
    the eBay API and web scraping.
    """
    
    def __init__(self, config: ResearchConfig):
        """
        Initialize the eBay researcher.
        
        Args:
            config: Research configuration
        """
        self.config = config
        self.api = EbayAPI(config.ebay) if config.ebay.use_api else None
        self.scraper = EbayScraper(config.ebay) if config.ebay.use_scraping else None
    
    @cache_result(ttl=3600)
    async def get_product_data(
        self,
        identifier: str,
        identifier_type: str = "keywords"
    ) -> Dict[str, Any]:
        """
        Get product data from eBay.
        
        Args:
            identifier: Product identifier (keywords, UPC, etc.)
            identifier_type: Type of identifier ("keywords", "upc", "isbn", "ean")
            
        Returns:
            Dict[str, Any]: Product data
        """
        logger.info(f"Getting eBay product data for {identifier_type}: {identifier}")
        
        # Get raw listings data
        listings = await self._get_listings(identifier, identifier_type)
        
        # Aggregate the data
        return self._aggregate_listings_data(listings)
    
    async def _get_listings(
        self,
        identifier: str,
        identifier_type: str = "keywords"
    ) -> List[Dict[str, Any]]:
        """
        Get listings data from eBay.
        
        Args:
            identifier: Product identifier (keywords, UPC, etc.)
            identifier_type: Type of identifier ("keywords", "upc", "isbn", "ean")
            
        Returns:
            List[Dict[str, Any]]: List of listings data
        """
        listings = []
        
        # Try API first if enabled
        if self.api:
            try:
                if identifier_type == "keywords":
                    api_data = await self.api.find_items_by_keywords(identifier)
                else:
                    api_data = await self.api.find_items_by_product(
                        identifier, product_id_type=identifier_type.upper()
                    )
                
                if api_data:
                    api_listings = self._extract_listings_from_api(api_data)
                    listings.extend(api_listings)
                    logger.info(f"Got {len(api_listings)} eBay listings from API for {identifier_type}: {identifier}")
            except Exception as e:
                logger.error(f"Error getting eBay listings from API: {str(e)}")
        
        # Fall back to scraping if API failed or is disabled, or if we need more data
        if (not listings or len(listings) < 50) and self.scraper:
            try:
                # Create session and get proxy
                session = await create_session(self.config)
                proxy = get_proxy_url(self.config.proxy) if self.config.proxy.enabled else None
                
                # Scrape search results
                scraper_listings = await self.scraper.search_products(
                    identifier, session=session, proxy=proxy
                )
                
                if scraper_listings:
                    listings.extend(scraper_listings)
                    logger.info(f"Got {len(scraper_listings)} eBay listings from scraper for {identifier_type}: {identifier}")
                
                # Close the session
                await session.close()
            except Exception as e:
                logger.error(f"Error getting eBay listings from scraper: {str(e)}")
        
        return listings
    
    def _extract_listings_from_api(self, api_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract listings data from eBay API response.
        
        Args:
            api_data: Raw API data
            
        Returns:
            List[Dict[str, Any]]: List of listings data
        """
        listings = []
        
        try:
            # Extract items from API response
            find_items_response = api_data.get("findItemsByKeywordsResponse", [])
            if not find_items_response:
                find_items_response = api_data.get("findItemsByProductResponse", [])
            
            if not find_items_response:
                return listings
            
            search_result = find_items_response[0].get("searchResult", [])
            if not search_result:
                return listings
            
            items = search_result[0].get("item", [])
            if not items:
                return listings
            
            # Process each item
            for item in items:
                try:
                    # Extract basic information
                    item_id = item.get("itemId", [None])[0]
                    title = item.get("title", [None])[0]
                    url = item.get("viewItemURL", [None])[0]
                    
                    # Extract price
                    current_price = item.get("sellingStatus", [{}])[0].get("currentPrice", [{}])[0]
                    price = float(current_price.get("__value__", 0))
                    currency = current_price.get("@currencyId", "USD")
                    
                    # Extract shipping price
                    shipping_info = item.get("shippingInfo", [{}])[0]
                    shipping_cost = shipping_info.get("shippingServiceCost", [{}])[0]
                    shipping_price = float(shipping_cost.get("__value__", 0))
                    
                    # Extract condition
                    condition_info = item.get("condition", [{}])[0]
                    condition = condition_info.get("conditionDisplayName", [None])[0]
                    
                    # Extract listing type
                    listing_info = item.get("listingInfo", [{}])[0]
                    listing_type = listing_info.get("listingType", [None])[0]
                    buy_it_now = listing_info.get("buyItNowAvailable", ["false"])[0] == "true"
                    
                    # Extract watchers (if available)
                    watch_count = item.get("listingInfo", [{}])[0].get("watchCount", [None])[0]
                    watchers = int(watch_count) if watch_count else None
                    
                    # Create listing data dictionary
                    listing = {
                        "item_id": item_id,
                        "title": title,
                        "url": url,
                        "price": price,
                        "currency": currency,
                        "shipping_price": shipping_price,
                        "condition": condition,
                        "listing_type": listing_type,
                        "buy_it_now": buy_it_now,
                        "watchers": watchers
                    }
                    
                    listings.append(listing)
                except Exception as e:
                    logger.error(f"Error extracting listing data from API: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error extracting listings from eBay API data: {str(e)}")
        
        return listings
    
    def _aggregate_listings_data(self, listings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate listings data into a single product data dictionary.
        
        Args:
            listings: List of listings data
            
        Returns:
            Dict[str, Any]: Aggregated product data
        """
        if not listings:
            return {"source": "ebay", "listings_count": 0}
        
        # Initialize aggregated data
        aggregated = {
            "source": "ebay",
            "listings_count": len(listings),
            "raw_listings": listings
        }
        
        try:
            # Extract prices
            prices = [listing["price"] for listing in listings if listing.get("price") is not None]
            if prices:
                aggregated["price_range"] = {
                    "min": min(prices),
                    "max": max(prices),
                    "avg": statistics.mean(prices),
                    "median": statistics.median(prices)
                }
            
            # Extract shipping prices
            shipping_prices = [listing["shipping_price"] for listing in listings if listing.get("shipping_price") is not None]
            if shipping_prices:
                aggregated["shipping_price"] = {
                    "min": min(shipping_prices),
                    "max": max(shipping_prices),
                    "avg": statistics.mean(shipping_prices)
                }
            
            # Extract watchers
            watchers = [listing["watchers"] for listing in listings if listing.get("watchers") is not None]
            if watchers:
                aggregated["watchers"] = {
                    "total": sum(watchers),
                    "avg": statistics.mean(watchers),
                    "max": max(watchers)
                }
            
            # Extract conditions
            conditions = {}
            for listing in listings:
                condition = listing.get("condition")
                if condition:
                    conditions[condition] = conditions.get(condition, 0) + 1
            
            if conditions:
                aggregated["conditions"] = conditions
            
            # Extract listing types
            listing_types = {}
            for listing in listings:
                listing_type = listing.get("listing_type")
                if listing_type:
                    listing_types[listing_type] = listing_types.get(listing_type, 0) + 1
            
            if listing_types:
                aggregated["listing_types"] = listing_types
        except Exception as e:
            logger.error(f"Error aggregating eBay listings data: {str(e)}")
        
        return aggregated