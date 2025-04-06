"""
Asynchronous scraper for HiBid.com auctions.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz

from .proxy_manager import ProxyManager
from .user_agent_manager import UserAgentManager

logger = logging.getLogger(__name__)


@dataclass
class AuctionItem:
    """Data class for storing auction item details."""
    item_id: str
    name: str
    brand: Optional[str] = None
    model: Optional[str] = None
    bid_amount: Optional[float] = None
    upc: Optional[str] = None
    condition: Optional[str] = None
    image_url: Optional[str] = None
    auction_url: Optional[str] = None
    amazon_url: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class HiBidScraper:
    """
    Asynchronous scraper for HiBid.com auctions.
    
    This class provides methods to scrape auction data from HiBid.com,
    including product details, bid amounts, and conditions.
    """

    BASE_URL = "https://www.hibid.com"
    SEARCH_URL = "https://www.hibid.com/search"
    
    def __init__(
        self,
        proxy_manager: Optional[ProxyManager] = None,
        user_agent_manager: Optional[UserAgentManager] = None,
        max_concurrent_requests: int = 5,
        request_delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """
        Initialize the HiBid scraper.

        Args:
            proxy_manager: ProxyManager instance for proxy rotation
            user_agent_manager: UserAgentManager instance for user agent rotation
            max_concurrent_requests: Maximum number of concurrent requests
            request_delay: Delay between requests in seconds
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
        """
        self.proxy_manager = proxy_manager or ProxyManager()
        self.user_agent_manager = user_agent_manager or UserAgentManager()
        self.max_concurrent_requests = max_concurrent_requests
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.timeout = timeout
        
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._last_request_time = 0
        self._session: Optional[aiohttp.ClientSession] = None
        self._upc_cache: Dict[Tuple[str, str], str] = {}  # (brand, model) -> UPC
        self._visited_urls: Set[str] = set()
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    async def start(self):
        """Initialize the scraper session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        
    async def close(self):
        """Close the scraper session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get the current session or create a new one.
        
        Returns:
            aiohttp.ClientSession: The current session
        """
        if self._session is None or self._session.closed:
            await self.start()
        return self._session
        
    async def _make_request(
        self, 
        url: str, 
        method: str = "GET", 
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        proxy: Optional[str] = None,
        retry_count: int = 0
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Make an HTTP request with retry logic and rate limiting.
        
        Args:
            url: URL to request
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Form data
            headers: HTTP headers
            proxy: Proxy URL to use
            retry_count: Current retry count
            
        Returns:
            Tuple[Optional[str], Optional[Dict]]: Tuple of (HTML content, response headers)
        """
        if retry_count > self.max_retries:
            logger.error(f"Max retries exceeded for URL: {url}")
            return None, None
            
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last_request)
        
        # Get or create session
        session = await self._get_session()
        
        # Prepare headers with random user agent
        if headers is None:
            headers = {}
        if "User-Agent" not in headers:
            headers["User-Agent"] = self.user_agent_manager.get_random_user_agent()
        
        # Get proxy if not provided
        if proxy is None and self.proxy_manager:
            proxy = await self.proxy_manager.get_proxy()
        
        try:
            async with self._semaphore:
                self._last_request_time = time.time()
                
                if method.upper() == "GET":
                    async with session.get(
                        url, params=params, headers=headers, proxy=proxy
                    ) as response:
                        if response.status == 200:
                            content = await response.text()
                            return content, dict(response.headers)
                        elif response.status in (429, 403):  # Rate limited or forbidden
                            logger.warning(f"Rate limited or forbidden: {response.status} for URL: {url}")
                            # Exponential backoff
                            wait_time = 2 ** retry_count * 5
                            logger.info(f"Waiting {wait_time} seconds before retry...")
                            await asyncio.sleep(wait_time)
                            return await self._make_request(
                                url, method, params, data, headers, proxy, retry_count + 1
                            )
                        else:
                            logger.error(f"HTTP error {response.status} for URL: {url}")
                            return None, None
                elif method.upper() == "POST":
                    async with session.post(
                        url, params=params, data=data, headers=headers, proxy=proxy
                    ) as response:
                        if response.status == 200:
                            content = await response.text()
                            return content, dict(response.headers)
                        elif response.status in (429, 403):  # Rate limited or forbidden
                            logger.warning(f"Rate limited or forbidden: {response.status} for URL: {url}")
                            # Exponential backoff
                            wait_time = 2 ** retry_count * 5
                            logger.info(f"Waiting {wait_time} seconds before retry...")
                            await asyncio.sleep(wait_time)
                            return await self._make_request(
                                url, method, params, data, headers, proxy, retry_count + 1
                            )
                        else:
                            logger.error(f"HTTP error {response.status} for URL: {url}")
                            return None, None
                else:
                    logger.error(f"Unsupported HTTP method: {method}")
                    return None, None
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout for URL: {url}")
            return await self._make_request(
                url, method, params, data, headers, proxy, retry_count + 1
            )
        except aiohttp.ClientError as e:
            logger.error(f"Request error for URL {url}: {str(e)}")
            return await self._make_request(
                url, method, params, data, headers, proxy, retry_count + 1
            )
        except Exception as e:
            logger.exception(f"Unexpected error for URL {url}: {str(e)}")
            return None, None
            
    async def search_auctions(
        self, 
        keyword: str, 
        category: Optional[str] = None,
        location: Optional[str] = None,
        max_pages: int = 5
    ) -> List[str]:
        """
        Search for auctions on HiBid.com.
        
        Args:
            keyword: Search keyword
            category: Auction category
            location: Location filter
            max_pages: Maximum number of search result pages to scrape
            
        Returns:
            List[str]: List of auction URLs
        """
        logger.info(f"Searching for auctions with keyword: {keyword}")
        
        params = {"q": keyword}
        if category:
            params["category"] = category
        if location:
            params["location"] = location
            
        auction_urls = []
        
        for page in range(1, max_pages + 1):
            params["page"] = page
            
            content, _ = await self._make_request(self.SEARCH_URL, params=params)
            if not content:
                logger.warning(f"Failed to get search results for page {page}")
                break
                
            soup = BeautifulSoup(content, "html.parser")
            auction_links = soup.select(".auction-item-container a.auction-item-link")
            
            if not auction_links:
                logger.info(f"No more auction links found on page {page}")
                break
                
            for link in auction_links:
                href = link.get("href")
                if href:
                    full_url = urljoin(self.BASE_URL, href)
                    if full_url not in auction_urls:
                        auction_urls.append(full_url)
                        
            logger.info(f"Found {len(auction_links)} auctions on page {page}")
            
            # Check if there are more pages
            next_page = soup.select_one("a.page-link[aria-label='Next']")
            if not next_page or "disabled" in next_page.get("class", []):
                logger.info("No more pages available")
                break
                
        logger.info(f"Found a total of {len(auction_urls)} auction URLs")
        return auction_urls
        
    async def scrape_auction_items(self, auction_url: str) -> List[AuctionItem]:
        """
        Scrape items from an auction page.
        
        Args:
            auction_url: URL of the auction page
            
        Returns:
            List[AuctionItem]: List of scraped auction items
        """
        logger.info(f"Scraping auction items from: {auction_url}")
        
        if auction_url in self._visited_urls:
            logger.info(f"Auction URL already visited: {auction_url}")
            return []
            
        self._visited_urls.add(auction_url)
        
        content, _ = await self._make_request(auction_url)
        if not content:
            logger.warning(f"Failed to get auction page: {auction_url}")
            return []
            
        soup = BeautifulSoup(content, "html.parser")
        item_containers = soup.select(".lot-tile-container")
        
        if not item_containers:
            logger.warning(f"No item containers found on auction page: {auction_url}")
            return []
            
        items = []
        tasks = []
        
        for container in item_containers:
            item_link = container.select_one("a.lot-tile-link")
            if not item_link:
                continue
                
            item_url = urljoin(self.BASE_URL, item_link.get("href", ""))
            if not item_url:
                continue
                
            tasks.append(self.scrape_item_details(item_url))
            
        if tasks:
            items = await asyncio.gather(*tasks)
            items = [item for item in items if item]
            
        logger.info(f"Scraped {len(items)} items from auction: {auction_url}")
        return items
        
    async def scrape_item_details(self, item_url: str) -> Optional[AuctionItem]:
        """
        Scrape details of a single auction item.
        
        Args:
            item_url: URL of the item page
            
        Returns:
            Optional[AuctionItem]: Scraped auction item, or None if scraping failed
        """
        logger.info(f"Scraping item details from: {item_url}")
        
        content, _ = await self._make_request(item_url)
        if not content:
            logger.warning(f"Failed to get item page: {item_url}")
            return None
            
        soup = BeautifulSoup(content, "html.parser")
        
        # Extract item ID from URL
        item_id = self._extract_item_id(item_url)
        
        # Extract item name
        name_elem = soup.select_one("h1.lot-title")
        name = name_elem.text.strip() if name_elem else "Unknown Item"
        
        # Extract current bid
        bid_elem = soup.select_one(".current-bid-value")
        bid_amount = None
        if bid_elem:
            bid_text = bid_elem.text.strip()
            bid_amount = self._extract_price(bid_text)
            
        # Extract item details
        details_table = soup.select_one(".lot-details-table")
        
        brand = None
        model = None
        upc = None
        condition = None
        amazon_url = None
        additional_info = {}
        
        if details_table:
            rows = details_table.select("tr")
            for row in rows:
                header = row.select_one("th")
                value = row.select_one("td")
                
                if not header or not value:
                    continue
                    
                header_text = header.text.strip().lower()
                value_text = value.text.strip()
                
                if "brand" in header_text:
                    brand = value_text
                elif "model" in header_text:
                    model = value_text
                elif "upc" in header_text or "barcode" in header_text:
                    upc = value_text
                elif "condition" in header_text:
                    condition = value_text
                elif "amazon" in header_text:
                    amazon_link = value.select_one("a")
                    if amazon_link:
                        amazon_url = amazon_link.get("href")
                else:
                    # Store other details in additional_info
                    additional_info[header_text] = value_text
        
        # Extract image URL
        image_elem = soup.select_one(".lot-image img")
        image_url = image_elem.get("src") if image_elem else None
        
        # If UPC is missing, try to find it using fuzzy matching
        if not upc and brand and model:
            upc = await self._find_upc_by_fuzzy_match(brand, model)
        
        # Create and return the AuctionItem
        item = AuctionItem(
            item_id=item_id,
            name=name,
            brand=brand,
            model=model,
            bid_amount=bid_amount,
            upc=upc,
            condition=condition,
            image_url=image_url,
            auction_url=item_url,
            amazon_url=amazon_url,
            additional_info=additional_info
        )
        
        logger.info(f"Scraped item: {item.name} (ID: {item.item_id})")
        return item
        
    def _extract_item_id(self, url: str) -> str:
        """
        Extract item ID from URL.
        
        Args:
            url: Item URL
            
        Returns:
            str: Item ID, or a generated ID if extraction fails
        """
        try:
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.split("/")
            
            # Try to find a numeric ID in the URL path
            for part in path_parts:
                if part.isdigit():
                    return part
                    
            # If no numeric ID found, use the last part of the path
            last_part = path_parts[-1]
            if last_part:
                return last_part
                
            # Fallback to a hash of the URL
            return str(hash(url))
        except Exception as e:
            logger.error(f"Error extracting item ID from URL {url}: {str(e)}")
            return str(hash(url))
            
    def _extract_price(self, price_text: str) -> Optional[float]:
        """
        Extract price from text.
        
        Args:
            price_text: Text containing a price
            
        Returns:
            Optional[float]: Extracted price, or None if extraction fails
        """
        try:
            # Remove currency symbols and other non-numeric characters
            price_text = re.sub(r'[^\d.]', '', price_text)
            if price_text:
                return float(price_text)
            return None
        except ValueError:
            logger.debug(f"Failed to extract price from text: {price_text}")
            return None
            
    async def _find_upc_by_fuzzy_match(self, brand: str, model: str) -> Optional[str]:
        """
        Find UPC by fuzzy matching brand and model.
        
        Args:
            brand: Product brand
            model: Product model
            
        Returns:
            Optional[str]: UPC if found, None otherwise
        """
        # Check cache first
        cache_key = (brand.lower(), model.lower())
        if cache_key in self._upc_cache:
            return self._upc_cache[cache_key]
            
        # In a real implementation, this would query a database or external API
        # For this example, we'll just return None
        logger.info(f"UPC not found for {brand} {model}, would query external source in production")
        return None
        
    async def scrape_multiple_auctions(self, auction_urls: List[str]) -> List[AuctionItem]:
        """
        Scrape multiple auctions concurrently.
        
        Args:
            auction_urls: List of auction URLs to scrape
            
        Returns:
            List[AuctionItem]: List of all scraped items
        """
        logger.info(f"Scraping {len(auction_urls)} auctions")
        
        all_items = []
        tasks = [self.scrape_auction_items(url) for url in auction_urls]
        
        results = await asyncio.gather(*tasks)
        for items in results:
            all_items.extend(items)
            
        logger.info(f"Scraped a total of {len(all_items)} items from {len(auction_urls)} auctions")
        return all_items