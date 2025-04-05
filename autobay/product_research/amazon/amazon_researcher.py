"""
Amazon product researcher.

This module provides a class for researching Amazon product data using both
the Amazon Product Advertising API and web scraping.
"""

import logging
from typing import Dict, List, Optional, Any

import aiohttp

from ..config import ResearchConfig
from ..utils import cache_result, create_session, get_proxy_url
from .amazon_api import AmazonAPI
from .amazon_scraper import AmazonScraper

logger = logging.getLogger(__name__)


class AmazonResearcher:
    """
    Amazon product researcher.
    
    This class provides methods for researching Amazon product data using both
    the Amazon Product Advertising API and web scraping.
    """
    
    def __init__(self, config: ResearchConfig):
        """
        Initialize the Amazon researcher.
        
        Args:
            config: Research configuration
        """
        self.config = config
        self.api = AmazonAPI(config.amazon) if config.amazon.use_api else None
        self.scraper = AmazonScraper(config.amazon) if config.amazon.use_scraping else None
    
    @cache_result(ttl=3600)
    async def get_product_data(
        self,
        identifier: str,
        identifier_type: str = "asin"
    ) -> Dict[str, Any]:
        """
        Get product data from Amazon.
        
        Args:
            identifier: Product identifier (ASIN, UPC, etc.)
            identifier_type: Type of identifier ("asin", "upc", "keywords")
            
        Returns:
            Dict[str, Any]: Product data
        """
        logger.info(f"Getting Amazon product data for {identifier_type}: {identifier}")
        
        product_data = {}
        
        # Try API first if enabled
        if self.api:
            try:
                if identifier_type == "asin":
                    api_data = await self.api.get_product_by_asin(identifier)
                elif identifier_type == "upc":
                    api_data = await self.api.get_product_by_upc(identifier)
                else:
                    api_data = await self.api.search_items(keywords=identifier)
                
                if api_data:
                    product_data = self._normalize_api_data(api_data)
                    logger.info(f"Got Amazon product data from API for {identifier_type}: {identifier}")
            except Exception as e:
                logger.error(f"Error getting Amazon product data from API: {str(e)}")
        
        # Fall back to scraping if API failed or is disabled
        if not product_data and self.scraper:
            try:
                # Create session and get proxy
                session = await create_session(self.config)
                proxy = get_proxy_url(self.config.proxy) if self.config.proxy.enabled else None
                
                if identifier_type == "asin":
                    scraper_data = await self.scraper.get_product_details(
                        identifier, session=session, proxy=proxy
                    )
                else:
                    # Search for the product and get details for the first result
                    search_results = await self.scraper.search_products(
                        identifier, session=session, proxy=proxy
                    )
                    
                    if search_results:
                        first_result = search_results[0]
                        scraper_data = await self.scraper.get_product_details(
                            first_result["asin"], session=session, proxy=proxy
                        )
                    else:
                        scraper_data = {}
                
                if scraper_data:
                    product_data = self._normalize_scraper_data(scraper_data)
                    logger.info(f"Got Amazon product data from scraper for {identifier_type}: {identifier}")
                
                # Close the session
                await session.close()
            except Exception as e:
                logger.error(f"Error getting Amazon product data from scraper: {str(e)}")
        
        return product_data
    
    def _normalize_api_data(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Amazon API data to a common format.
        
        Args:
            api_data: Raw API data
            
        Returns:
            Dict[str, Any]: Normalized product data
        """
        normalized = {
            "source": "amazon_api",
            "raw_data": api_data
        }
        
        try:
            # Extract items from API response
            items = api_data.get("ItemsResult", {}).get("Items", [])
            if not items:
                return normalized
            
            # Use the first item
            item = items[0]
            
            # Extract basic information
            normalized["asin"] = item.get("ASIN")
            normalized["url"] = item.get("DetailPageURL")
            
            # Extract title
            item_info = item.get("ItemInfo", {})
            title_info = item_info.get("Title", {})
            normalized["title"] = title_info.get("DisplayValue")
            
            # Extract brand
            by_line_info = item_info.get("ByLineInfo", {})
            brand_info = by_line_info.get("Brand", {})
            normalized["brand"] = brand_info.get("DisplayValue")
            
            # Extract price
            offers = item.get("Offers", {})
            listings = offers.get("Listings", [])
            if listings:
                price_info = listings[0].get("Price", {})
                normalized["price"] = price_info.get("Amount")
                normalized["currency"] = price_info.get("Currency")
                
                # Extract discount
                saving_basis = listings[0].get("SavingBasis", {})
                if saving_basis:
                    normalized["was_price"] = saving_basis.get("Amount")
            
            # Extract images
            images = item.get("Images", {})
            primary_image = images.get("Primary", {})
            large_image = primary_image.get("Large", {})
            normalized["image_url"] = large_image.get("URL")
            
            # Extract reviews
            customer_reviews = item.get("CustomerReviews", {})
            normalized["rating"] = customer_reviews.get("StarRating", {}).get("Value")
            normalized["review_count"] = customer_reviews.get("Count")
            
            # Extract browse nodes (categories)
            browse_nodes_result = item.get("BrowseNodeInfo", {})
            browse_nodes = browse_nodes_result.get("BrowseNodes", [])
            if browse_nodes:
                categories = []
                for node in browse_nodes:
                    categories.append({
                        "id": node.get("Id"),
                        "name": node.get("DisplayName")
                    })
                normalized["categories"] = categories
            
            # Extract features
            features = item_info.get("Features", {})
            display_values = features.get("DisplayValues", [])
            if display_values:
                normalized["features"] = display_values
        except Exception as e:
            logger.error(f"Error normalizing Amazon API data: {str(e)}")
        
        return normalized
    
    def _normalize_scraper_data(self, scraper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Amazon scraper data to a common format.
        
        Args:
            scraper_data: Raw scraper data
            
        Returns:
            Dict[str, Any]: Normalized product data
        """
        normalized = {
            "source": "amazon_scraper",
            "raw_data": scraper_data
        }
        
        try:
            # Copy basic fields
            for field in ["asin", "title", "url", "price", "was_price", "rating", "review_count", "image_url"]:
                if field in scraper_data:
                    normalized[field] = scraper_data[field]
            
            # Extract ranks
            if "ranks" in scraper_data:
                normalized["ranks"] = scraper_data["ranks"]
                
                # Find the main category rank
                if scraper_data["ranks"]:
                    normalized["rank_in_category"] = scraper_data["ranks"][0]["rank"]
                    normalized["category"] = scraper_data["ranks"][0]["category"]
            
            # Extract details
            if "details" in scraper_data:
                details = scraper_data["details"]
                
                # Extract brand
                if "Brand" in details:
                    normalized["brand"] = details["Brand"]
                
                # Extract model
                if "Model" in details or "Model Name" in details or "Model Number" in details:
                    normalized["model"] = details.get("Model") or details.get("Model Name") or details.get("Model Number")
                
                # Extract UPC/EAN
                if "UPC" in details:
                    normalized["upc"] = details["UPC"]
                elif "EAN" in details:
                    normalized["ean"] = details["EAN"]
                
                # Extract dimensions
                if "Product Dimensions" in details:
                    normalized["dimensions"] = details["Product Dimensions"]
                
                # Extract weight
                if "Item Weight" in details:
                    normalized["weight"] = details["Item Weight"]
                
                # Extract date first available
                if "Date First Available" in details:
                    normalized["date_first_available"] = details["Date First Available"]
            
            # Extract frequently returned info
            if "frequently_returned" in scraper_data:
                normalized["frequently_returned"] = scraper_data["frequently_returned"]
        except Exception as e:
            logger.error(f"Error normalizing Amazon scraper data: {str(e)}")
        
        return normalized