"""
Terapeaks web scraper for product data.

This module provides a class for scraping product data from Terapeaks
when the API is unavailable or insufficient.
"""

import logging
import re
import json
from typing import Dict, List, Optional, Any, Tuple

import aiohttp
from bs4 import BeautifulSoup

from ..config import TerapeaksConfig
from ..utils import rotate_user_agent, get_proxy_url

logger = logging.getLogger(__name__)


class TerapeaksScraper:
    """
    Terapeaks web scraper for product data.
    
    This class provides methods for scraping product data from Terapeaks
    when the API is unavailable or insufficient.
    """
    
    BASE_URL = "https://www.terapeaks.com"
    LOGIN_URL = "https://www.terapeaks.com/login"
    SEARCH_URL = "https://www.terapeaks.com/search"
    
    def __init__(self, config: TerapeaksConfig):
        """
        Initialize the Terapeaks scraper.
        
        Args:
            config: Terapeaks configuration
        """
        self.config = config
        self.username = config.username
        self.password = config.password
        self.session = None
        self.logged_in = False
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """
        Ensure that we have an active session.
        
        Returns:
            aiohttp.ClientSession: Active session
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"User-Agent": rotate_user_agent()}
            )
            self.logged_in = False
        
        return self.session
    
    async def _login(self, proxy: Optional[str] = None) -> bool:
        """
        Log in to Terapeaks.
        
        Args:
            proxy: Proxy URL (if None, no proxy will be used)
            
        Returns:
            bool: True if login was successful, False otherwise
        """
        if self.logged_in:
            return True
        
        logger.info("Logging in to Terapeaks")
        
        session = await self._ensure_session()
        
        try:
            # First, get the login page to extract CSRF token
            async with session.get(
                self.LOGIN_URL,
                proxy=proxy,
                timeout=self.config.timeout
            ) as response:
                if response.status != 200:
                    logger.error(f"Terapeaks login page error: {response.status}")
                    return False
                
                html = await response.text()
            
            # Parse the HTML
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract CSRF token
            csrf_token = None
            csrf_element = soup.select_one("input[name='_csrf_token']")
            if csrf_element:
                csrf_token = csrf_element.get("value")
            
            if not csrf_token:
                logger.error("Failed to extract CSRF token from Terapeaks login page")
                return False
            
            # Submit login form
            login_data = {
                "username": self.username,
                "password": self.password,
                "_csrf_token": csrf_token
            }
            
            async with session.post(
                self.LOGIN_URL,
                data=login_data,
                proxy=proxy,
                timeout=self.config.timeout,
                allow_redirects=True
            ) as response:
                if response.status != 200:
                    logger.error(f"Terapeaks login error: {response.status}")
                    return False
                
                # Check if login was successful
                html = await response.text()
                if "Invalid username or password" in html:
                    logger.error("Terapeaks login failed: Invalid username or password")
                    return False
                
                self.logged_in = True
                logger.info("Successfully logged in to Terapeaks")
                return True
        except Exception as e:
            logger.error(f"Error logging in to Terapeaks: {str(e)}")
            return False
    
    async def search_products(
        self,
        keywords: str,
        proxy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for products on Terapeaks.
        
        Args:
            keywords: Search keywords
            proxy: Proxy URL (if None, no proxy will be used)
            
        Returns:
            List[Dict[str, Any]]: List of product data dictionaries
        """
        logger.info(f"Scraping Terapeaks search results for '{keywords}'")
        
        # Ensure we have a session and are logged in
        await self._ensure_session()
        if not await self._login(proxy):
            return []
        
        try:
            # Make the search request
            params = {"q": keywords}
            async with self.session.get(
                self.SEARCH_URL,
                params=params,
                proxy=proxy,
                timeout=self.config.timeout
            ) as response:
                if response.status != 200:
                    logger.error(f"Terapeaks search error: {response.status}")
                    return []
                
                html = await response.text()
            
            # Parse the HTML
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract product data
            products = []
            product_elements = soup.select(".product-card")
            
            for element in product_elements:
                try:
                    # Extract product ID
                    product_id = element.get("data-product-id")
                    if not product_id:
                        continue
                    
                    # Extract title
                    title_element = element.select_one(".product-title")
                    title = title_element.text.strip() if title_element else "Unknown"
                    
                    # Extract URL
                    url_element = element.select_one("a.product-link")
                    url = self.BASE_URL + url_element.get("href") if url_element else None
                    
                    # Extract price data
                    price_element = element.select_one(".product-price")
                    price_text = price_element.text.strip() if price_element else None
                    price = self._extract_price(price_text) if price_text else None
                    
                    # Extract sold count
                    sold_element = element.select_one(".product-sold-count")
                    sold_text = sold_element.text.strip() if sold_element else None
                    sold_count = self._extract_number(sold_text) if sold_text else None
                    
                    # Extract active listings count
                    active_element = element.select_one(".product-active-count")
                    active_text = active_element.text.strip() if active_element else None
                    active_count = self._extract_number(active_text) if active_text else None
                    
                    # Create product data dictionary
                    product = {
                        "product_id": product_id,
                        "title": title,
                        "url": url,
                        "price": price,
                        "sold_count": sold_count,
                        "active_count": active_count
                    }
                    
                    products.append(product)
                except Exception as e:
                    logger.error(f"Error extracting product data: {str(e)}")
                    continue
            
            logger.info(f"Scraped {len(products)} products from Terapeaks search results")
            return products
        except Exception as e:
            logger.error(f"Error scraping Terapeaks search results: {str(e)}")
            return []
    
    async def get_product_details(
        self,
        product_id: str,
        proxy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed product information from Terapeaks.
        
        Args:
            product_id: Terapeaks product ID
            proxy: Proxy URL (if None, no proxy will be used)
            
        Returns:
            Dict[str, Any]: Product details
        """
        logger.info(f"Scraping Terapeaks product details for product ID: {product_id}")
        
        # Ensure we have a session and are logged in
        await self._ensure_session()
        if not await self._login(proxy):
            return {}
        
        try:
            # Make the product details request
            url = f"{self.BASE_URL}/products/{product_id}"
            async with self.session.get(
                url,
                proxy=proxy,
                timeout=self.config.timeout
            ) as response:
                if response.status != 200:
                    logger.error(f"Terapeaks product details error: {response.status}")
                    return {}
                
                html = await response.text()
            
            # Parse the HTML
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract product details
            product = {"product_id": product_id, "url": url}
            
            # Extract title
            title_element = soup.select_one("h1.product-title")
            if title_element:
                product["title"] = title_element.text.strip()
            
            # Extract product data from JSON-LD
            script_element = soup.select_one("script[type='application/ld+json']")
            if script_element:
                try:
                    json_data = json.loads(script_element.string)
                    if json_data:
                        product["json_ld"] = json_data
                except Exception:
                    pass
            
            # Extract sold listings data
            sold_data = await self._extract_sold_listings_data(soup)
            if sold_data:
                product["sold_listings"] = sold_data
            
            # Extract active listings data
            active_data = await self._extract_active_listings_data(soup)
            if active_data:
                product["active_listings"] = active_data
            
            logger.info(f"Scraped product details for product ID: {product_id}")
            return product
        except Exception as e:
            logger.error(f"Error scraping Terapeaks product details: {str(e)}")
            return {"product_id": product_id, "error": str(e)}
    
    async def _extract_sold_listings_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract sold listings data from product page.
        
        Args:
            soup: BeautifulSoup object for the product page
            
        Returns:
            Dict[str, Any]: Sold listings data
        """
        sold_data = {}
        
        try:
            # Extract sold listings container
            sold_container = soup.select_one("#sold-listings")
            if not sold_container:
                return sold_data
            
            # Extract average sold price
            avg_price_element = sold_container.select_one(".avg-price")
            if avg_price_element:
                avg_price_text = avg_price_element.text.strip()
                sold_data["avg_price"] = self._extract_price(avg_price_text)
            
            # Extract price range
            price_range_element = sold_container.select_one(".price-range")
            if price_range_element:
                price_range_text = price_range_element.text.strip()
                low_price, high_price = self._extract_price_range(price_range_text)
                sold_data["low_price"] = low_price
                sold_data["high_price"] = high_price
            
            # Extract total sold
            total_sold_element = sold_container.select_one(".total-sold")
            if total_sold_element:
                total_sold_text = total_sold_element.text.strip()
                sold_data["total_sold"] = self._extract_number(total_sold_text)
            
            # Extract average shipping cost
            shipping_element = sold_container.select_one(".avg-shipping")
            if shipping_element:
                shipping_text = shipping_element.text.strip()
                sold_data["avg_shipping"] = self._extract_price(shipping_text)
        except Exception as e:
            logger.error(f"Error extracting sold listings data: {str(e)}")
        
        return sold_data
    
    async def _extract_active_listings_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract active listings data from product page.
        
        Args:
            soup: BeautifulSoup object for the product page
            
        Returns:
            Dict[str, Any]: Active listings data
        """
        active_data = {}
        
        try:
            # Extract active listings container
            active_container = soup.select_one("#active-listings")
            if not active_container:
                return active_data
            
            # Extract average listing price
            avg_price_element = active_container.select_one(".avg-price")
            if avg_price_element:
                avg_price_text = avg_price_element.text.strip()
                active_data["avg_price"] = self._extract_price(avg_price_text)
            
            # Extract price range
            price_range_element = active_container.select_one(".price-range")
            if price_range_element:
                price_range_text = price_range_element.text.strip()
                low_price, high_price = self._extract_price_range(price_range_text)
                active_data["low_price"] = low_price
                active_data["high_price"] = high_price
            
            # Extract total listings
            total_listings_element = active_container.select_one(".total-listings")
            if total_listings_element:
                total_listings_text = total_listings_element.text.strip()
                active_data["total_listings"] = self._extract_number(total_listings_text)
            
            # Extract total watchers
            watchers_element = active_container.select_one(".total-watchers")
            if watchers_element:
                watchers_text = watchers_element.text.strip()
                active_data["total_watchers"] = self._extract_number(watchers_text)
        except Exception as e:
            logger.error(f"Error extracting active listings data: {str(e)}")
        
        return active_data
    
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
            # Remove currency symbols and other non-numeric characters
            price_text = re.sub(r'[^\d.]', '', price_text)
            if price_text:
                return float(price_text)
            return None
        except ValueError:
            logger.debug(f"Failed to extract price from text: {price_text}")
            return None
    
    def _extract_price_range(self, price_range_text: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract price range from text.
        
        Args:
            price_range_text: Text containing a price range
            
        Returns:
            Tuple[Optional[float], Optional[float]]: Tuple of (low price, high price)
        """
        if not price_range_text:
            return None, None
        
        try:
            # Extract low and high prices (e.g., "$10.00 - $20.00")
            match = re.search(r'[\$£€](\d+(\.\d+)?)\s*-\s*[\$£€](\d+(\.\d+)?)', price_range_text)
            if match:
                low_price = float(match.group(1))
                high_price = float(match.group(3))
                return low_price, high_price
            return None, None
        except ValueError:
            logger.debug(f"Failed to extract price range from text: {price_range_text}")
            return None, None
    
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
            # Extract numeric value (e.g., "1,234 items" -> 1234)
            match = re.search(r'([\d,]+)', text)
            if match:
                return int(match.group(1).replace(",", ""))
            return None
        except ValueError:
            logger.debug(f"Failed to extract number from text: {text}")
            return None
    
    async def close(self):
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            self.logged_in = False