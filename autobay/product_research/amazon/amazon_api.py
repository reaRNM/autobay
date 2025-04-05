"""
Amazon Product Advertising API integration.

This module provides a class for interacting with the Amazon Product Advertising API
to retrieve product data.
"""

import logging
import time
from typing import Dict, List, Optional, Any

import aiohttp
import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

from ..config import AmazonConfig

logger = logging.getLogger(__name__)


class AmazonAPI:
    """
    Amazon Product Advertising API client.
    
    This class provides methods for retrieving product data from the Amazon
    Product Advertising API.
    """
    
    # API endpoints by region
    ENDPOINTS = {
        "US": "webservices.amazon.com",
        "CA": "webservices.amazon.ca",
        "UK": "webservices.amazon.co.uk",
        "DE": "webservices.amazon.de",
        "FR": "webservices.amazon.fr",
        "JP": "webservices.amazon.co.jp",
        "IT": "webservices.amazon.it",
        "ES": "webservices.amazon.es",
        "IN": "webservices.amazon.in",
        "BR": "webservices.amazon.com.br",
        "MX": "webservices.amazon.com.mx",
        "AU": "webservices.amazon.com.au",
    }
    
    def __init__(self, config: AmazonConfig):
        """
        Initialize the Amazon API client.
        
        Args:
            config: Amazon API configuration
        """
        self.config = config
        self.credentials = Credentials(
            access_key=config.api_key,
            secret_key=config.api_secret,
            token=None
        )
        self.region = config.region
        self.partner_tag = config.partner_tag
        self.endpoint = self.ENDPOINTS.get(self.region, self.ENDPOINTS["US"])
        self.base_url = f"https://{self.endpoint}/paapi5/searchitems"
        
    async def search_items(
        self,
        keywords: str,
        search_index: str = "All",
        item_count: int = 10
    ) -> Dict[str, Any]:
        """
        Search for items on Amazon.
        
        Args:
            keywords: Search keywords
            search_index: Amazon search index (e.g., "All", "Electronics")
            item_count: Number of items to return
            
        Returns:
            Dict[str, Any]: Search results
        """
        logger.info(f"Searching Amazon for '{keywords}' in {search_index}")
        
        # Prepare request payload
        payload = {
            "Keywords": keywords,
            "SearchIndex": search_index,
            "ItemCount": item_count,
            "PartnerTag": self.partner_tag,
            "PartnerType": "Associates",
            "Resources": [
                "ItemInfo.Title",
                "ItemInfo.Features",
                "ItemInfo.ProductInfo",
                "ItemInfo.ByLineInfo",
                "Offers.Listings.Price",
                "Offers.Listings.SavingBasis",
                "Offers.Listings.Promotions",
                "Images.Primary.Large",
                "CustomerReviews",
                "BrowseNodeInfo.BrowseNodes"
            ]
        }
        
        # Sign the request
        request = AWSRequest(
            method="POST",
            url=self.base_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Host": self.endpoint
            }
        )
        
        SigV4Auth(self.credentials, "ProductAdvertisingAPI", self.region).add_auth(request)
        
        # Make the request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                headers=dict(request.headers),
                json=payload,
                timeout=self.config.timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Amazon API error: {response.status} - {await response.text()}")
                    return {}
    
    async def get_product_by_asin(self, asin: str) -> Dict[str, Any]:
        """
        Get product details by ASIN.
        
        Args:
            asin: Amazon Standard Identification Number
            
        Returns:
            Dict[str, Any]: Product details
        """
        logger.info(f"Getting Amazon product details for ASIN: {asin}")
        
        # Prepare request payload
        payload = {
            "ItemIds": [asin],
            "PartnerTag": self.partner_tag,
            "PartnerType": "Associates",
            "Resources": [
                "ItemInfo.Title",
                "ItemInfo.Features",
                "ItemInfo.ProductInfo",
                "ItemInfo.ByLineInfo",
                "ItemInfo.ContentInfo",
                "ItemInfo.ManufactureInfo",
                "ItemInfo.TechnicalInfo",
                "ItemInfo.Classifications",
                "Offers.Listings.Price",
                "Offers.Listings.SavingBasis",
                "Offers.Listings.Promotions",
                "Offers.Summaries",
                "Images.Primary.Large",
                "Images.Variants.Large",
                "CustomerReviews",
                "BrowseNodeInfo.BrowseNodes"
            ]
        }
        
        # Sign the request
        request = AWSRequest(
            method="POST",
            url=self.base_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Host": self.endpoint
            }
        )
        
        SigV4Auth(self.credentials, "ProductAdvertisingAPI", self.region).add_auth(request)
        
        # Make the request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                headers=dict(request.headers),
                json=payload,
                timeout=self.config.timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Amazon API error: {response.status} - {await response.text()}")
                    return {}
    
    async def get_product_by_upc(self, upc: str) -> Dict[str, Any]:
        """
        Get product details by UPC.
        
        Args:
            upc: Universal Product Code
            
        Returns:
            Dict[str, Any]: Product details
        """
        logger.info(f"Getting Amazon product details for UPC: {upc}")
        
        # Search for the product by UPC
        return await self.search_items(keywords=upc, item_count=1)