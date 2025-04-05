"""
eBay web scraper for product data.

This module provides a class for scraping product data from eBay.com
when the API is unavailable or insufficient.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple

import aiohttp
from bs4 import BeautifulSoup

from ..config import EbayConfig
from ..utils import rotate_user_agent, get_proxy_url

logger = logging.getLogger(__name__)


class EbayScraper:
    """
    eBay web scraper for product data.
    
    This class provides methods for scraping product data from eBay.com
    when the API is unavailable or insufficient.
    """
    
    BASE_URL = "https://www.ebay.com"
    SEARCH_URL = "https://www.ebay.com/sch/i.html"
    ITEM_URL = "https://www.ebay.com/itm/"
    
    def __init__(self, config: EbayConfig):
        """
        Initialize the eBay scraper.
        
        Args:
            config: eBay configuration
        """
        self.config = config
    
    async def search_products(
        self,
        keywords: str,
        session: Optional[aiohttp.ClientSession] = None,
        proxy: Optional[str] = None,
        page: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Search for products on eBay.
        
        Args:
            keywords: Search keywords
            session: aiohttp ClientSession (if None, a new session will be created)
            proxy: Proxy URL (if None, no proxy will be used)
            page: Page number
            
        Returns:
            List[Dict[str, Any]]: List of product data dictionaries
        """
        logger.info(f"Scraping eBay search results for '{keywords}' (page {page})")
        
        # Create a new session if not provided
        close_session = False
        if session is None:
            session = aiohttp.ClientSession(
                headers={"User-Agent": rotate_user_agent()}
            )
            close_session = True
        
        try:
            # Make the request
            params = {
                "_nkw": keywords,
                "_pgn": str(page),
                "_ipg": "200"  # Items per page
            }
            async with session.get(
                self.SEARCH_URL,
                params=params,
                proxy=proxy,
                timeout=self.config.timeout
            ) as response:
                if response.status != 200:
                    logger.error(f"eBay search error: {response.status}")
                    return []
                
                html = await response.text()
            
            # Parse the HTML
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract product data
            products = []
            product_elements = soup.select("li.s-item")
            
            for element in product_elements:
                try:
                    # Skip "More items like this" element
                    if "s-item--dynamic" in element.get("class", []):
                        continue
                    
                    # Extract item ID
                    link_element = element.select_one("a.s-item__link")
                    if not link_element:
                        continue
                    
                    item_url = link_element.get("href", "")
                    item_id = self._extract_item_id(item_url)
                    
                    # Extract title
                    title_element = element.select_one("h3.s-item__title")
                    title = title_element.text.strip() if title_element else "Unknown"
                    
                    # Extract price
                    price_element = element.select_one("span.s-item__price")
                    price_text = price_element.text.strip() if price_element else None
                    price = self._extract_price(price_text) if price_text else None
                    
                    # Extract shipping price
                    shipping_element = element.select_one("span.s-item__shipping")
                    shipping_text = shipping_element.text.strip() if shipping_element else None
                    shipping_price = self._extract_shipping_price(shipping_text) if shipping_text else None
                    
                    # Extract condition
                    condition_element = element.select_one("span.SECONDARY_INFO")
                    condition = condition_element.text.strip() if condition_element else None
                    
                    # Extract image URL
                    img_element = element.select_one("img.s-item__image-img")
                    img_url = img_element.get("src") if img_element else None
                    
                    # Extract "sold" status
                    sold_element = element.select_one("span.BOLD")
                    is_sold = sold_element and "sold" in sold_element.text.lower()
                    
                    # Extract "watchers" count
                    watchers_element = element.select_one("span.s-item__watchCount")
                    watchers_text = watchers_element.text.strip() if watchers_element else None
                    watchers = self._extract_number(watchers_text) if watchers_text else None
                    
                    # Create product data dictionary
                    product = {
                        "item_id": item_id,
                        "title": title,
                        "url": item_url,
                        "price": price,
                        "shipping_price": shipping_price,
                        "condition": condition,
                        "image_url": img_url,
                        "is_sold": is_sold,
                        "watchers": watchers
                    }
                    
                    products.append(product)
                except Exception as e:
                    logger.error(f"Error extracting product data: {str(e)}")
                    continue
            
            logger.info(f"Scraped {len(products)} products from eBay search results")
            return products
        except Exception as e:
            logger.error(f"Error scraping eBay search results: {str(e)}")
            return []
        finally:
            if close_session:
                await session.close()
    
    async def get_item_details(
        self,
        item_id: str,
        session: Optional[aiohttp.ClientSession] = None,
        proxy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed item information from eBay.
        
        Args:
            item_id: eBay item ID
            session: aiohttp ClientSession (if None, a new session will be created)
            proxy: Proxy URL (if None, no proxy will be used)
            
        Returns:
            Dict[str, Any]: Item details
        """
        logger.info(f"Scraping eBay item details for item ID: {item_id}")
        
        # Create a new session if not provided
        close_session = False
        if session is None:
            session = aiohttp.ClientSession(
                headers={"User-Agent": rotate_user_agent()}
            )
            close_session = True
        
        try:
            # Make the request
            url = f"{self.ITEM_URL}{item_id}"
            async with session.get(
                url,
                proxy=proxy,
                timeout=self.config.timeout
            ) as response:
                if response.status != 200:
                    logger.error(f"eBay item details error: {response.status}")
                    return {}
                
                html = await response.text()
            
            # Parse the HTML
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract item details
            item = {"item_id": item_id, "url": url}
            
            # Extract title
            title_element = soup.select_one("h1.x-item-title__mainTitle")
            if title_element:
                item["title"] = title_element.text.strip()
            
            # Extract price
            price_element = soup.select_one("span.x-price-primary")
            if price_element:
                price_text = price_element.text.strip()
                item["price"] = self._extract_price(price_text)
            
            # Extract shipping price
            shipping_element = soup.select_one("span.x-additional-info__shipping-cost")
            if shipping_element:
                shipping_text = shipping_element.text.strip()
                item["shipping_price"] = self._extract_shipping_price(shipping_text)
            
            # Extract condition
            condition_element = soup.select_one("div.x-item-condition-text")
            if condition_element:
                item["condition"] = condition_element.text.strip()
            
            # Extract watchers
            watchers_element = soup.select_one("span.vi-watchcount-value")
            if watchers_element:
                watchers_text = watchers_element.text.strip()
                item["watchers"] = self._extract_number(watchers_text)
            
            # Extract item specifics
            specifics_container = soup.select_one("div.x-about-this-item")
            if specifics_container:
                specifics = {}
                rows = specifics_container.select("div.ux-layout-section__row")
                for row in rows:
                    label_element = row.select_one("div.ux-labels-values__labels")
                    value_element = row.select_one("div.ux-labels-values__values")
                    if label_element and value_element:
                        label = label_element.text.strip().rstrip(":")
                        value = value_element.text.strip()
                        specifics[label] = value
                
                if specifics:
                    item["specifics"] = specifics
            
            # Extract seller information
            seller_element = soup.select_one("span.ux-seller-pseudonym__pseudonym")
            if seller_element:
                item["seller"] = seller_element.text.strip()
            
            # Extract seller feedback
            feedback_element = soup.select_one("span.ux-seller-pseudonym__feedback-score")
            if feedback_element:
                feedback_text = feedback_element.text.strip()
                item["seller_feedback"] = self._extract_number(feedback_text)
            
            # Extract seller rating
            rating_element = soup.select_one("span.ux-seller-pseudonym__rating-score")
            if rating_element:
                rating_text = rating_element.text.strip()
                item["seller_rating"] = self._extract_percentage(rating_text)
            
            logger.info(f"Scraped item details for item ID: {item_id}")
            return item
        except Exception as e:
            logger.error(f"Error scraping eBay item details: {str(e)}")
            return {"item_id": item_id, "error": str(e)}
        finally:
            if close_session:
                await session.close()
    
    def _extract_item_id(self, url: str) -> Optional[str]:
        """
        Extract item ID from URL.
        
        Args:
            url: Item URL
            
        Returns:
            Optional[str]: Item ID, or None if extraction fails
        """
        if not url:
            return None
        
        try:
            # Extract item ID from URL (e.g., https://www.ebay.com/itm/123456789)
            match = re.search(r'/itm/(\d+)', url)
            if match:
                return match.group(1)
            
            # Alternative format (e.g., https://www.ebay.com/itm/item/123456789)
            match = re.search(r'/itm/item/(\d+)', url)
            if match:
                return match.group(1)
            
            # Another alternative format with query parameter
            match = re.search(r'item=(\d+)', url)
            if match:
                return match.group(1)
            
            return None
        except Exception:
            logger.debug(f"Failed to extract item ID from URL: {url}")
            return None
    
    def _extract_price(self, price_text: str) -> Optional[float]:
        """
        Extract price from text.
        
        Args:
            price_text: Text containing a price
            
        Returns:
            Optional[float]: Extracted price, or None if extraction fails
        """
        if not price_text:
            return None
        
        try:
            # Handle price ranges (e.g., "$10.00 to $20.00")
            if " to " in price_text:
                # Use the lower price in the range
                price_text = price_text.split(" to ")[0]
            
            # Remove currency symbols and other non-numeric characters
            price_text = re.sub(r'[^\d.]', '', price_text)
            if price_text:
                return float(price_text)
            return None
        except ValueError:
            logger.debug(f"Failed to extract price from text: {price_text}")
            return None
    
    def _extract_shipping_price(self, shipping_text: str) -> Optional[float]:
        """
        Extract shipping price from text.
        
        Args:
            shipping_text: Text containing a shipping price
            
        Returns:
            Optional[float]: Extracted shipping price, or None if extraction fails
        """
        if not shipping_text:
            return None
        
        # Check for free shipping
        if "free" in shipping_text.lower():
            return 0.0
        
        try:
            # Extract price (e.g., "+$10.00 shipping" -> 10.00)
            match = re.search(r'[\$£€](\d+(\.\d+)?)', shipping_text)
            if match:
                return float(match.group(1))
            return None
        except ValueError:
            logger.debug(f"Failed to extract shipping price from text: {shipping_text}")
            return None
    
    def _extract_number(self, text: str) -> Optional[int]:
        """
        Extract number from text.
        
        Args:
            text: Text containing a number
            
        Returns:
            Optional[int]: Extracted number, or None if extraction fails
        """
        if not text:
            return None
        
        try:
            # Extract numeric value (e.g., "1,234 watchers" -> 1234)
            match = re.search(r'([\d,]+)', text)
            if match:
                return int(match.group(1).replace(",", ""))
            return None
        except ValueError:
            logger.debug(f"Failed to extract number from text: {text}")
            return None
    
    def _extract_percentage(self, text: str) -> Optional[float]:
        """
        Extract percentage from text.
        
        Args:
            text: Text containing a percentage
            
        Returns:
            Optional[float]: Extracted percentage, or None if extraction fails
        """
        if not text:
            return None
        
        try:
            # Extract percentage value (e.g., "98.7% positive feedback" -> 98.7)
            match = re.search(r'(\d+(\.\d+)?)%', text)
            if match:
                return float(match.group(1))
            return None
        except ValueError:
            logger.debug(f"Failed to extract percentage from text: {text}")
            return None