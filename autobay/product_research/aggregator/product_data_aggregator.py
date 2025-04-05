"""
Product data aggregator.

This module provides a class for aggregating and normalizing product data
from multiple sources (Amazon, eBay, and Terapeaks).
"""

import logging
from typing import Dict, List, Optional, Any

from ..config import ResearchConfig
from ..amazon import AmazonResearcher
from ..ebay import EbayResearcher
from ..terapeaks import TerapeaksResearcher
from ..utils import cache_result

logger = logging.getLogger(__name__)


class ProductDataAggregator:
    """
    Product data aggregator.
    
    This class provides methods for aggregating and normalizing product data
    from multiple sources (Amazon, eBay, and Terapeaks).
    """
    
    def __init__(self, config: ResearchConfig):
        """
        Initialize the product data aggregator.
        
        Args:
            config: Research configuration
        """
        self.config = config
        self.amazon_researcher = AmazonResearcher(config)
        self.ebay_researcher = EbayResearcher(config)
        self.terapeaks_researcher = TerapeaksResearcher(config)
    
    @cache_result(ttl=3600)
    async def get_product_data(
        self,
        identifier: str,
        identifier_type: str = "keywords"
    ) -> Dict[str, Any]:
        """
        Get comprehensive product data from all sources.
        
        Args:
            identifier: Product identifier (keywords, UPC, ASIN, etc.)
            identifier_type: Type of identifier ("keywords", "upc", "asin", etc.)
            
        Returns:
            Dict[str, Any]: Aggregated product data
        """
        logger.info(f"Getting comprehensive product data for {identifier_type}: {identifier}")
        
        # Initialize result dictionary
        result = {
            "identifier": identifier,
            "identifier_type": identifier_type,
            "sources": {}
        }
        
        # Get data from Amazon
        try:
            amazon_data = await self.amazon_researcher.get_product_data(
                identifier,
                "asin" if identifier_type == "asin" else "upc" if identifier_type == "upc" else "keywords"
            )
            
            if amazon_data:
                result["sources"]["amazon"] = amazon_data
                logger.info(f"Got Amazon data for {identifier_type}: {identifier}")
        except Exception as e:
            logger.error(f"Error getting Amazon data: {str(e)}")
            result["sources"]["amazon"] = {"error": str(e)}
        
        # Get data from eBay
        try:
            ebay_data = await self.ebay_researcher.get_product_data(
                identifier,
                identifier_type
            )
            
            if ebay_data:
                result["sources"]["ebay"] = ebay_data
                logger.info(f"Got eBay data for {identifier_type}: {identifier}")
        except Exception as e:
            logger.error(f"Error getting eBay data: {str(e)}")
            result["sources"]["ebay"] = {"error": str(e)}
        
        # Get data from Terapeaks
        try:
            # Terapeaks only supports keyword search
            terapeaks_data = await self.terapeaks_researcher.get_product_data(identifier)
            
            if terapeaks_data:
                result["sources"]["terapeaks"] = terapeaks_data
                logger.info(f"Got Terapeaks data for {identifier_type}: {identifier}")
        except Exception as e:
            logger.error(f"Error getting Terapeaks data: {str(e)}")
            result["sources"]["terapeaks"] = {"error": str(e)}
        
        # Aggregate the data
        aggregated_data = self._aggregate_data(result["sources"])
        result["aggregated"] = aggregated_data
        
        return result
    
    def _aggregate_data(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate data from multiple sources.
        
        Args:
            sources: Dictionary of data from multiple sources
            
        Returns:
            Dict[str, Any]: Aggregated data
        """
        aggregated = {}
        
        try:
            # Extract basic product information
            aggregated["title"] = self._extract_title(sources)
            aggregated["brand"] = self._extract_brand(sources)
            aggregated["model"] = self._extract_model(sources)
            aggregated["upc"] = self._extract_upc(sources)
            
            # Extract pricing information
            aggregated["pricing"] = self._aggregate_pricing(sources)
            
            # Extract sales velocity information
            aggregated["sales_velocity"] = self._aggregate_sales_velocity(sources)
            
            # Extract SEO information
            aggregated["seo"] = self._aggregate_seo(sources)
            
            # Extract condition information
            aggregated["condition"] = self._aggregate_condition(sources)
            
            # Extract shipping information
            aggregated["shipping"] = self._aggregate_shipping(sources)
            
            # Extract review information
            aggregated["reviews"] = self._aggregate_reviews(sources)
        except Exception as e:
            logger.error(f"Error aggregating data: {str(e)}")
        
        return aggregated
    
    def _extract_title(self, sources: Dict[str, Any]) -> Optional[str]:
        """
        Extract product title from sources.
        
        Args:
            sources: Dictionary of data from multiple sources
            
        Returns:
            Optional[str]: Product title
        """
        # Try Amazon first
        amazon_data = sources.get("amazon", {})
        if amazon_data and "title" in amazon_data:
            return amazon_data["title"]
        
        # Try eBay next
        ebay_data = sources.get("ebay", {})
        if ebay_data and "raw_listings" in ebay_data and ebay_data["raw_listings"]:
            return ebay_data["raw_listings"][0].get("title")
        
        # Try Terapeaks last
        terapeaks_data = sources.get("terapeaks", {})
        if terapeaks_data and "title" in terapeaks_data:
            return terapeaks_data["title"]
        
        return None
    
    def _extract_brand(self, sources: Dict[str, Any]) -> Optional[str]:
        """
        Extract product brand from sources.
        
        Args:
            sources: Dictionary of data from multiple sources
            
        Returns:
            Optional[str]: Product brand
        """
        # Try Amazon first
        amazon_data = sources.get("amazon", {})
        if amazon_data and "brand" in amazon_data:
            return amazon_data["brand"]
        
        return None
    
    def _extract_model(self, sources: Dict[str, Any]) -> Optional[str]:
        """
        Extract product model from sources.
        
        Args:
            sources: Dictionary of data from multiple sources
            
        Returns:
            Optional[str]: Product model
        """
        # Try Amazon first
        amazon_data = sources.get("amazon", {})
        if amazon_data and "model" in amazon_data:
            return amazon_data["model"]
        
        return None
    
    def _extract_upc(self, sources: Dict[str, Any]) -> Optional[str]:
        """
        Extract product UPC from sources.
        
        Args:
            sources: Dictionary of data from multiple sources
            
        Returns:
            Optional[str]: Product UPC
        """
        # Try Amazon first
        amazon_data = sources.get("amazon", {})
        if amazon_data and "upc" in amazon_data:
            return amazon_data["upc"]
        
        return None
    
    def _aggregate_pricing(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate pricing information from sources.
        
        Args:
            sources: Dictionary of data from multiple sources
            
        Returns:
            Dict[str, Any]: Aggregated pricing information
        """
        pricing = {}
        
        # Extract Amazon pricing
        amazon_data = sources.get("amazon", {})
        if amazon_data and "price" in amazon_data:
            pricing["amazon"] = {
                "current_price": amazon_data["price"],
                "currency": amazon_data.get("currency", "USD")
            }
            
            if "was_price" in amazon_data:
                pricing["amazon"]["was_price"] = amazon_data["was_price"]
        
        # Extract eBay pricing
        ebay_data = sources.get("ebay", {})
        if ebay_data and "price_range" in ebay_data:
            pricing["ebay"] = ebay_data["price_range"]
        
        # Extract Terapeaks pricing
        terapeaks_data = sources.get("terapeaks", {})
        if terapeaks_data:
            # Extract sold listings pricing
            sold_listings = terapeaks_data.get("sold_listings", {})
            if sold_listings:
                pricing["terapeaks_sold"] = {
                    "avg_price": sold_listings.get("avg_price"),
                    "low_price": sold_listings.get("low_price"),
                    "high_price": sold_listings.get("high_price")
                }
            
            # Extract active listings pricing
            active_listings = terapeaks_data.get("active_listings", {})
            if active_listings:
                pricing["terapeaks_active"] = {
                    "avg_price": active_listings.get("avg_price"),
                    "low_price": active_listings.get("low_price"),
                    "high_price": active_listings.get("high_price")
                }
        
        # Calculate recommended pricing
        pricing["recommended"] = self._calculate_recommended_price(pricing)
        
        return pricing
    
    def _calculate_recommended_price(self, pricing: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate recommended pricing based on aggregated pricing information.
        
        Args:
            pricing: Aggregated pricing information
            
        Returns:
            Dict[str, Any]: Recommended pricing
        """
        recommended = {}
        
        # Get Amazon price
        amazon_price = None
        if "amazon" in pricing:
            amazon_price = pricing["amazon"].get("current_price")
        
        # Get eBay average price
        ebay_avg_price = None
        if "ebay" in pricing:
            ebay_avg_price = pricing["ebay"].get("avg")
        
        # Get Terapeaks sold average price
        terapeaks_sold_avg_price = None
        if "terapeaks_sold" in pricing:
            terapeaks_sold_avg_price = pricing["terapeaks_sold"].get("avg_price")
        
        # Calculate recommended price
        prices = [p for p in [amazon_price, ebay_avg_price, terapeaks_sold_avg_price] if p is not None]
        if prices:
            recommended["price"] = sum(prices) / len(prices)
        
        return recommended
    
    def _aggregate_sales_velocity(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate sales velocity information from sources.
        
        Args:
            sources: Dictionary of data from multiple sources
            
        Returns:
            Dict[str, Any]: Aggregated sales velocity information
        """
        sales_velocity = {}
        
        # Extract Amazon sales velocity
        amazon_data = sources.get("amazon", {})
        if amazon_data and "rank_in_category" in amazon_data:
            sales_velocity["amazon"] = {
                "rank_in_category": amazon_data["rank_in_category"],
                "category": amazon_data.get("category")
            }
        
        # Extract eBay sales velocity
        ebay_data = sources.get("ebay", {})
        if ebay_data and "watchers" in ebay_data:
            sales_velocity["ebay"] = {
                "total_watchers": ebay_data["watchers"].get("total"),
                "avg_watchers": ebay_data["watchers"].get("avg")
            }
        
        # Extract Terapeaks sales velocity
        terapeaks_data = sources.get("terapeaks", {})
        if terapeaks_data:
            # Extract sold listings data
            sold_listings = terapeaks_data.get("sold_listings", {})
            if sold_listings and "total_sold" in sold_listings:
                sales_velocity["terapeaks"] = {
                    "total_sold": sold_listings["total_sold"]
                }
            
            # Extract active listings data
            active_listings = terapeaks_data.get("active_listings", {})
            if active_listings and "total_watchers" in active_listings:
                if "terapeaks" not in sales_velocity:
                    sales_velocity["terapeaks"] = {}
                sales_velocity["terapeaks"]["total_watchers"] = active_listings["total_watchers"]
        
        return sales_velocity
    
    def _aggregate_seo(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate SEO information from sources.
        
        Args:
            sources: Dictionary of data from multiple sources
            
        Returns:
            Dict[str, Any]: Aggregated SEO information
        """
        seo = {}
        
        # Extract Amazon SEO
        amazon_data = sources.get("amazon", {})
        if amazon_data:
            amazon_seo = {}
            
            # Extract categories
            if "categories" in amazon_data:
                amazon_seo["categories"] = amazon_data["categories"]
            
            # Extract features
            if "features" in amazon_data:
                amazon_seo["features"] = amazon_data["features"]
            
            if amazon_seo:
                seo["amazon"] = amazon_seo
        
        # Extract eBay SEO
        # (eBay doesn't provide much SEO data in our current implementation)
        
        return seo
    
    def _aggregate_condition(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate condition information from sources.
        
        Args:
            sources: Dictionary of data from multiple sources
            
        Returns:
            Dict[str, Any]: Aggregated condition information
        """
        condition = {}
        
        # Extract Amazon condition
        amazon_data = sources.get("amazon", {})
        if amazon_data and "frequently_returned" in amazon_data:
            condition["amazon"] = {
                "frequently_returned": amazon_data["frequently_returned"]
            }
        
        # Extract eBay condition
        ebay_data = sources.get("ebay", {})
        if ebay_data and "conditions" in ebay_data:
            condition["ebay"] = {
                "conditions": ebay_data["conditions"]
            }
        
        return condition
    
    def _aggregate_shipping(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate shipping information from sources.
        
        Args:
            sources: Dictionary of data from multiple sources
            
        Returns:
            Dict[str, Any]: Aggregated shipping information
        """
        shipping = {}
        
        # Extract eBay shipping
        ebay_data = sources.get("ebay", {})
        if ebay_data and "shipping_price" in ebay_data:
            shipping["ebay"] = ebay_data["shipping_price"]
        
        # Extract Terapeaks shipping
        terapeaks_data = sources.get("terapeaks", {})
        if terapeaks_data:
            sold_listings = terapeaks_data.get("sold_listings", {})
            if sold_listings and "avg_shipping" in sold_listings:
                shipping["terapeaks"] = {
                    "avg_shipping": sold_listings["avg_shipping"]
                }
        
        return shipping
    
    def _aggregate_reviews(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate review information from sources.
        
        Args:
            sources: Dictionary of data from multiple sources
            
        Returns:
            Dict[str, Any]: Aggregated review information
        """
        reviews = {}
        
        # Extract Amazon reviews
        amazon_data = sources.get("amazon", {})
        if amazon_data:
            amazon_reviews = {}
            
            if "rating" in amazon_data:
                amazon_reviews["rating"] = amazon_data["rating"]
            
            if "review_count" in amazon_data:
                amazon_reviews["count"] = amazon_data["review_count"]
            
            if amazon_reviews:
                reviews["amazon"] = amazon_reviews
        
        return reviews