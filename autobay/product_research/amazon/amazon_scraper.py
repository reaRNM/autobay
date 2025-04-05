"""
Amazon web scraper for product data.

This module provides a class for scraping product data from Amazon.com
when the API is unavailable or insufficient.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple

import aiohttp
from bs4 import BeautifulSoup

from ..config import AmazonConfig
from ..utils import rotate_user_agent, get_proxy_url

logger = logging.getLogger(__name__)


class AmazonScraper:
    """
    Amazon web scraper for product data.
    
    This class provides methods for scraping product data from Amazon.com
    when the API is unavailable or insufficient.
    """
    
    BASE_URL = "https://www.amazon.com"
    SEARCH_URL = "https://www.amazon.com/s"
    PRODUCT_URL = "https://www.amazon.com/dp/"
    
    def __init__(self, config: AmazonConfig):
        """
        Initialize the Amazon scraper.
        
        Args:
            config: Amazon configuration
        """
        self.config = config
        
    async def search_products(
        self,
        keywords: str,
        session: Optional[aiohttp.ClientSession] = None,
        proxy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for products on Amazon.
        
        Args:
            keywords: Search keywords
            session: aiohttp ClientSession (if None, a new session will be created)
            proxy: Proxy URL (if None, no proxy will be used)
            
        Returns:
            List[Dict[str, Any]]: List of product data dictionaries
        """
        logger.info(f"Scraping Amazon search results for '{keywords}'")
        
        # Create a new session if not provided
        close_session = False
        if session is None:
            session = aiohttp.ClientSession(
                headers={"User-Agent": rotate_user_agent()}
            )
            close_session = True
        
        try:
            # Make the request
            params = {"k": keywords, "ref": "nb_sb_noss"}
            async with session.get(
                self.SEARCH_URL,
                params=params,
                proxy=proxy,
                timeout=self.config.timeout
            ) as response:
                if response.status != 200:
                    logger.error(f"Amazon search error: {response.status}")
                    return []
                
                html = await response.text()
            
            # Parse the HTML
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract product data
            products = []
            product_elements = soup.select("div[data-component-type='s-search-result']")
            
            for element in product_elements:
                try:
                    # Extract ASIN
                    asin = element.get("data-asin")
                    if not asin:
                        continue
                    
                    # Extract title
                    title_element = element.select_one("h2 a span")
                    title = title_element.text.strip() if title_element else "Unknown"
                    
                    # Extract URL
                    url_element = element.select_one("h2 a")
                    url = self.BASE_URL + url_element.get("href") if url_element else None
                    
                    # Extract price
                    price_element = element.select_one("span.a-price span.a-offscreen")
                    price_text = price_element.text.strip() if price_element else None
                    price = self._extract_price(price_text) if price_text else None
                    
                    # Extract rating
                    rating_element = element.select_one("span.a-icon-alt")
                    rating_text = rating_element.text.strip() if rating_element else None
                    rating = self._extract_rating(rating_text) if rating_text else None
                    
                    # Extract review count
                    review_element = element.select_one("span.a-size-base.s-underline-text")
                    review_count = int(review_element.text.replace(",", "")) if review_element else 0
                    
                    # Extract image URL
                    img_element = element.select_one("img.s-image")
                    img_url = img_element.get("src") if img_element else None
                    
                    # Create product data dictionary
                    product = {
                        "asin": asin,
                        "title": title,
                        "url": url,
                        "price": price,
                        "rating": rating,
                        "review_count": review_count,
                        "image_url": img_url
                    }
                    
                    products.append(product)
                except Exception as e:
                    logger.error(f"Error extracting product data: {str(e)}")
                    continue
            
            logger.info(f"Scraped {len(products)} products from Amazon search results")
            return products
        except Exception as e:
            logger.error(f"Error scraping Amazon search results: {str(e)}")
            return []
        finally:
            if close_session:
                await session.close()
    
    async def get_product_details(
        self,
        asin: str,
        session: Optional[aiohttp.ClientSession] = None,
        proxy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed product information from Amazon.
        
        Args:
            asin: Amazon Standard Identification Number
            session: aiohttp ClientSession (if None, a new session will be created)
            proxy: Proxy URL (if None, no proxy will be used)
            
        Returns:
            Dict[str, Any]: Product details
        """
        logger.info(f"Scraping Amazon product details for ASIN: {asin}")
        
        # Create a new session if not provided
        close_session = False
        if session is None:
            session = aiohttp.ClientSession(
                headers={"User-Agent": rotate_user_agent()}
            )
            close_session = True
        
        try:
            # Make the request
            url = f"{self.PRODUCT_URL}{asin}"
            async with session.get(
                url,
                proxy=proxy,
                timeout=self.config.timeout
            ) as response:
                if response.status != 200:
                    logger.error(f"Amazon product details error: {response.status}")
                    return {}
                
                html = await response.text()
            
            # Parse the HTML
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract product details
            product = {"asin": asin, "url": url}
            
            # Extract title
            title_element = soup.select_one("#productTitle")
            if title_element:
                product["title"] = title_element.text.strip()
            
            # Extract price
            price_element = soup.select_one("#priceblock_ourprice, #priceblock_dealprice, .a-price .a-offscreen")
            if price_element:
                price_text = price_element.text.strip()
                product["price"] = self._extract_price(price_text)
            
            # Extract discounted price
            was_price_element = soup.select_one(".priceBlockStrikePriceString, .a-text-price .a-offscreen")
            if was_price_element:
                was_price_text = was_price_element.text.strip()
                product["was_price"] = self._extract_price(was_price_text)
            
            # Extract rating
            rating_element = soup.select_one("#acrPopover")
            if rating_element:
                rating_text = rating_element.get("title", "")
                product["rating"] = self._extract_rating(rating_text)
            
            # Extract review count
            review_count_element = soup.select_one("#acrCustomerReviewText")
            if review_count_element:
                review_text = review_count_element.text.strip()
                product["review_count"] = self._extract_number(review_text)
            
            # Extract category and rank
            rank_elements = soup.select("#productDetails_detailBullets_sections1 tr")
            for element in rank_elements:
                header = element.select_one("th")
                if header and "Best Sellers Rank" in header.text:
                    rank_text = element.select_one("td").text.strip()
                    ranks = self._extract_ranks(rank_text)
                    if ranks:
                        product["ranks"] = ranks
            
            # Extract product details
            detail_elements = soup.select("#productDetails_techSpec_section_1 tr, #productDetails_detailBullets_sections1 tr")
            details = {}
            for element in detail_elements:
                header = element.select_one("th, td:first-child")
                value = element.select_one("td:last-child")
                if header and value:
                    key = header.text.strip().rstrip(":")
                    details[key] = value.text.strip()
            
            if details:
                product["details"] = details
            
            # Extract frequently returned info
            returned_element = soup.select_one("#cr-summarization-attributes-list")
            if returned_element:
                returned_items = returned_element.select("div.a-row")
                returned_info = []
                for item in returned_items:
                    returned_info.append(item.text.strip())
                
                if returned_info:
                    product["frequently_returned"] = returned_info
            
            logger.info(f"Scraped product details for ASIN: {asin}")
            return product
        except Exception as e:
            logger.error(f"Error scraping Amazon product details: {str(e)}")
            return {"asin": asin, "error": str(e)}
        finally:
            if close_session:
                await session.close()
    
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
    
    def _extract_rating(self, rating_text: str) -> Optional[float]:
        """
        Extract rating from text.
        
        Args:
            rating_text: Text containing a rating
            
        Returns:
            Optional[float]: Extracted rating, or None if extraction fails
        """
        if not rating_text:
            return None
        
        try:
            # Extract rating value (e.g., "4.5 out of 5 stars" -> 4.5)
            match = re.search(r'(\d+(\.\d+)?)', rating_text)
            if match:
                return float(match.group(1))
            return None
        except ValueError:
            logger.debug(f"Failed to extract rating from text: {rating_text}")
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
            # Extract numeric value (e.g., "1,234 reviews" -> 1234)
            match = re.search(r'([\d,]+)', text)
            if match:
                return int(match.group(1).replace(",", ""))
            return None
        except ValueError:
            logger.debug(f"Failed to extract number from text: {text}")
            return None
    
    def _extract_ranks(self, rank_text: str) -> List[Dict[str, Any]]:
        """
        Extract category ranks from text.
        
        Args:
            rank_text: Text containing category ranks
            
        Returns:
            List[Dict[str, Any]]: List of category rank dictionaries
        """
        if not rank_text:
            return []
        
        ranks = []
        
        # Extract ranks (e.g., "#1,234 in Electronics (See Top 100 in Electronics)")
        pattern = r'#([\d,]+) in ([^(]+)'
        matches = re.findall(pattern, rank_text)
        
        for match in matches:
            rank_value = int(match[0].replace(",", ""))
            category = match[1].strip()
            ranks.append({
                "rank": rank_value,
                "category": category
            })
        
        return ranks