"""
eBay API integration.

This module provides a class for interacting with the eBay API
to retrieve product data.
"""

import logging
import time
from typing import Dict, List, Optional, Any

import aiohttp

from ..config import EbayConfig

logger = logging.getLogger(__name__)


class EbayAPI:
    """
    eBay API client.
    
    This class provides methods for retrieving product data from the eBay API.
    """
    
    # API endpoints
    FINDING_API_URL = "https://svcs.ebay.com/services/search/FindingService/v1"
    SHOPPING_API_URL = "https://open.api.ebay.com/shopping"
    
    def __init__(self, config: EbayConfig):
        """
        Initialize the eBay API client.
        
        Args:
            config: eBay API configuration
        """
        self.config = config
        self.app_id = config.app_id
        self.cert_id = config.cert_id
        self.dev_id = config.dev_id
    
    async def find_items_by_keywords(
        self,
        keywords: str,
        category_id: Optional[str] = None,
        page_number: int = 1,
        items_per_page: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Find items by keywords using the eBay Finding API.
        
        Args:
            keywords: Search keywords
            category_id: eBay category ID
            page_number: Page number
            items_per_page: Number of items per page
            
        Returns:
            Dict[str, Any]: Search results
        """
        logger.info(f"Searching eBay for '{keywords}'")
        
        if items_per_page is None:
            items_per_page = self.config.results_per_page
        
        # Prepare request parameters
        params = {
            "OPERATION-NAME": "findItemsByKeywords",
            "SERVICE-VERSION": "1.0.0",
            "SECURITY-APPNAME": self.app_id,
            "RESPONSE-DATA-FORMAT": "JSON",
            "REST-PAYLOAD": "",
            "keywords": keywords,
            "paginationInput.pageNumber": str(page_number),
            "paginationInput.entriesPerPage": str(items_per_page),
            "GLOBAL-ID": "EBAY-US",  # Default to US
            "siteid": "0",  # Default to US
        }
        
        if category_id:
            params["categoryId"] = category_id
        
        # Add additional parameters for more detailed results
        params["outputSelector"] = "SellerInfo,StoreInfo,AspectHistogram"
        params["sortOrder"] = "BestMatch"
        
        # Make the request
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.FINDING_API_URL,
                params=params,
                timeout=self.config.timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"eBay API error: {response.status} - {await response.text()}")
                    return {}
    
    async def find_items_by_product(
        self,
        product_id: str,
        product_id_type: str = "UPC",
        page_number: int = 1,
        items_per_page: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Find items by product ID using the eBay Finding API.
        
        Args:
            product_id: Product ID (UPC, ISBN, EAN)
            product_id_type: Type of product ID ("UPC", "ISBN", "EAN")
            page_number: Page number
            items_per_page: Number of items per page
            
        Returns:
            Dict[str, Any]: Search results
        """
        logger.info(f"Searching eBay for {product_id_type}: {product_id}")
        
        if items_per_page is None:
            items_per_page = self.config.results_per_page
        
        # Prepare request parameters
        params = {
            "OPERATION-NAME": "findItemsByProduct",
            "SERVICE-VERSION": "1.0.0",
            "SECURITY-APPNAME": self.app_id,
            "RESPONSE-DATA-FORMAT": "JSON",
            "REST-PAYLOAD": "",
            f"productId.@type": product_id_type,
            "productId": product_id,
            "paginationInput.pageNumber": str(page_number),
            "paginationInput.entriesPerPage": str(items_per_page),
            "GLOBAL-ID": "EBAY-US",  # Default to US
            "siteid": "0",  # Default to US
        }
        
        # Add additional parameters for more detailed results
        params["outputSelector"] = "SellerInfo,StoreInfo,AspectHistogram"
        params["sortOrder"] = "BestMatch"
        
        # Make the request
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.FINDING_API_URL,
                params=params,
                timeout=self.config.timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"eBay API error: {response.status} - {await response.text()}")
                    return {}
    
    async def get_item_details(self, item_id: str) -> Dict[str, Any]:
        """
        Get detailed item information using the eBay Shopping API.
        
        Args:
            item_id: eBay item ID
            
        Returns:
            Dict[str, Any]: Item details
        """
        logger.info(f"Getting eBay item details for item ID: {item_id}")
        
        # Prepare request parameters
        params = {
            "callname": "GetSingleItem",
            "responseencoding": "JSON",
            "appid": self.app_id,
            "siteid": "0",  # Default to US
            "version": "967",
            "ItemID": item_id,
            "IncludeSelector": "Details,Description,ItemSpecifics,ShippingCosts,Variations,Compatibility"
        }
        
        # Make the request
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.SHOPPING_API_URL,
                params=params,
                timeout=self.config.timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"eBay API error: {response.status} - {await response.text()}")
                    return {}
    
    async def get_multiple_items(self, item_ids: List[str]) -> Dict[str, Any]:
        """
        Get details for multiple items using the eBay Shopping API.
        
        Args:
            item_ids: List of eBay item IDs
            
        Returns:
            Dict[str, Any]: Multiple items details
        """
        logger.info(f"Getting eBay details for {len(item_ids)} items")
        
        # Prepare request parameters
        params = {
            "callname": "GetMultipleItems",
            "responseencoding": "JSON",
            "appid": self.app_id,
            "siteid": "0",  # Default to US
            "version": "967",
            "ItemID": ",".join(item_ids),
            "IncludeSelector": "Details,Description,ItemSpecifics,ShippingCosts"
        }
        
        # Make the request
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.SHOPPING_API_URL,
                params=params,
                timeout=self.config.timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"eBay API error: {response.status} - {await response.text()}")
                    return {}