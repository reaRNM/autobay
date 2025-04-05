"""
Proxy rotation manager for web scraping.
"""

import asyncio
import logging
import random
from typing import List, Optional, Dict

import aiohttp

logger = logging.getLogger(__name__)


class ProxyManager:
    """
    Manages a pool of proxies for web scraping, with automatic rotation and validation.
    """

    def __init__(self, proxies: List[str] = None, validation_url: str = "https://httpbin.org/ip"):
        """
        Initialize the proxy manager.

        Args:
            proxies: List of proxy URLs in format "http://user:pass@host:port"
            validation_url: URL to use for validating proxies
        """
        self.proxies = proxies or []
        self.validation_url = validation_url
        self.working_proxies = []
        self.failed_proxies = set()
        self.proxy_stats: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()

    async def add_proxy(self, proxy: str) -> None:
        """
        Add a new proxy to the pool.

        Args:
            proxy: Proxy URL in format "http://user:pass@host:port"
        """
        async with self._lock:
            if proxy not in self.proxies:
                self.proxies.append(proxy)
                self.proxy_stats[proxy] = {"success": 0, "failure": 0}
                logger.info(f"Added new proxy: {self._mask_proxy(proxy)}")

    async def remove_proxy(self, proxy: str) -> None:
        """
        Remove a proxy from the pool.

        Args:
            proxy: Proxy URL to remove
        """
        async with self._lock:
            if proxy in self.proxies:
                self.proxies.remove(proxy)
                if proxy in self.working_proxies:
                    self.working_proxies.remove(proxy)
                if proxy in self.proxy_stats:
                    del self.proxy_stats[proxy]
                logger.info(f"Removed proxy: {self._mask_proxy(proxy)}")

    async def validate_proxies(self) -> None:
        """
        Validate all proxies in the pool and update the working_proxies list.
        """
        tasks = [self._validate_proxy(proxy) for proxy in self.proxies]
        await asyncio.gather(*tasks)
        
        async with self._lock:
            logger.info(f"Proxy validation complete. {len(self.working_proxies)}/{len(self.proxies)} proxies working.")

    async def _validate_proxy(self, proxy: str) -> bool:
        """
        Validate a single proxy by making a request to the validation URL.

        Args:
            proxy: Proxy URL to validate

        Returns:
            bool: True if the proxy is working, False otherwise
        """
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.validation_url, proxy=proxy) as response:
                    if response.status == 200:
                        async with self._lock:
                            if proxy not in self.working_proxies:
                                self.working_proxies.append(proxy)
                            if proxy in self.failed_proxies:
                                self.failed_proxies.remove(proxy)
                            self.proxy_stats[proxy]["success"] += 1
                        logger.debug(f"Proxy validated successfully: {self._mask_proxy(proxy)}")
                        return True
                    else:
                        self._mark_proxy_failed(proxy)
                        return False
        except Exception as e:
            logger.debug(f"Proxy validation failed for {self._mask_proxy(proxy)}: {str(e)}")
            self._mark_proxy_failed(proxy)
            return False

    def _mark_proxy_failed(self, proxy: str) -> None:
        """
        Mark a proxy as failed.

        Args:
            proxy: Proxy URL that failed
        """
        async def _update():
            async with self._lock:
                if proxy in self.working_proxies:
                    self.working_proxies.remove(proxy)
                self.failed_proxies.add(proxy)
                if proxy in self.proxy_stats:
                    self.proxy_stats[proxy]["failure"] += 1
        
        asyncio.create_task(_update())

    async def get_proxy(self) -> Optional[str]:
        """
        Get a working proxy from the pool.

        Returns:
            Optional[str]: A working proxy URL, or None if no working proxies are available
        """
        if not self.working_proxies:
            await self.validate_proxies()
            
        async with self._lock:
            if not self.working_proxies:
                logger.warning("No working proxies available")
                return None
                
            # Select a proxy with preference for those with higher success rates
            proxy_scores = []
            for proxy in self.working_proxies:
                stats = self.proxy_stats.get(proxy, {"success": 0, "failure": 0})
                success = stats["success"]
                failure = stats["failure"]
                
                # Calculate a score based on success rate
                if success + failure == 0:
                    score = 0.5  # Default score for new proxies
                else:
                    score = success / (success + failure)
                
                proxy_scores.append((proxy, score))
            
            # Sort by score and add some randomness
            proxy_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select from top 50% with higher probability
            top_half = max(1, len(proxy_scores) // 2)
            weights = [1.5 if i < top_half else 1.0 for i in range(len(proxy_scores))]
            
            selected_proxy = random.choices(
                [p[0] for p in proxy_scores],
                weights=weights,
                k=1
            )[0]
            
            logger.debug(f"Selected proxy: {self._mask_proxy(selected_proxy)}")
            return selected_proxy

    @staticmethod
    def _mask_proxy(proxy: str) -> str:
        """
        Mask sensitive information in proxy URL for logging.

        Args:
            proxy: Proxy URL

        Returns:
            str: Masked proxy URL
        """
        if '@' in proxy:
            # Mask username and password
            protocol, rest = proxy.split('://', 1)
            auth, host_port = rest.split('@', 1)
            return f"{protocol}://****:****@{host_port}"
        return proxy

    def get_stats(self) -> Dict:
        """
        Get statistics about proxy usage.

        Returns:
            Dict: Proxy statistics
        """
        return {
            "total": len(self.proxies),
            "working": len(self.working_proxies),
            "failed": len(self.failed_proxies),
            "stats": {self._mask_proxy(p): s for p, s in self.proxy_stats.items()}
        }