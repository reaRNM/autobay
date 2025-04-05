"""
HTTP utilities for the Product Research Engine.
"""

import random
import time
from typing import Dict, List, Optional, Tuple

import aiohttp
from aiohttp import ClientSession

from ..config import ProxyConfig, ResearchConfig


# List of common user agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
]


def rotate_user_agent() -> str:
    """
    Get a random user agent from the list.
    
    Returns:
        str: Random user agent
    """
    return random.choice(USER_AGENTS)


def get_proxy_url(proxy_config: ProxyConfig) -> Optional[str]:
    """
    Get a proxy URL from the configuration.
    
    Args:
        proxy_config: Proxy configuration
        
    Returns:
        Optional[str]: Proxy URL, or None if proxy is disabled or no URLs are available
    """
    if not proxy_config.enabled or not proxy_config.urls:
        return None
    
    proxy_url = random.choice(proxy_config.urls)
    
    # Add authentication if provided
    if proxy_config.username and proxy_config.password:
        # Parse the URL to add authentication
        if "://" in proxy_url:
            protocol, rest = proxy_url.split("://", 1)
            proxy_url = f"{protocol}://{proxy_config.username}:{proxy_config.password}@{rest}"
        else:
            proxy_url = f"{proxy_config.username}:{proxy_config.password}@{proxy_url}"
    
    return proxy_url


async def create_session(
    config: ResearchConfig,
    headers: Optional[Dict[str, str]] = None
) -> ClientSession:
    """
    Create an aiohttp ClientSession with the specified configuration.
    
    Args:
        config: Research configuration
        headers: Additional headers to include
        
    Returns:
        ClientSession: Configured aiohttp ClientSession
    """
    if headers is None:
        headers = {}
    
    # Add User-Agent header if not provided
    if "User-Agent" not in headers:
        headers["User-Agent"] = config.user_agent
    
    # Create session with timeout
    timeout = aiohttp.ClientTimeout(total=30)
    
    # Get proxy URL if enabled
    proxy = get_proxy_url(config.proxy) if config.proxy.enabled else None
    
    return ClientSession(
        headers=headers,
        timeout=timeout,
        trust_env=True,  # Trust environment variables for proxy settings
    )


async def make_request(
    session: ClientSession,
    url: str,
    method: str = "GET",
    params: Optional[Dict] = None,
    data: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    proxy: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Tuple[Optional[Dict], Optional[str], Optional[Dict]]:
    """
    Make an HTTP request with retry logic.
    
    Args:
        session: aiohttp ClientSession
        url: URL to request
        method: HTTP method (GET, POST, etc.)
        params: Query parameters
        data: Form data
        headers: HTTP headers
        proxy: Proxy URL
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        
    Returns:
        Tuple[Optional[Dict], Optional[str], Optional[Dict]]:
            Tuple of (JSON response, text response, response headers)
    """
    retries = 0
    
    while retries <= max_retries:
        try:
            if method.upper() == "GET":
                async with session.get(
                    url, params=params, headers=headers, proxy=proxy
                ) as response:
                    if response.status == 200:
                        # Try to parse as JSON first
                        try:
                            json_data = await response.json()
                            return json_data, None, dict(response.headers)
                        except:
                            # If not JSON, return text
                            text_data = await response.text()
                            return None, text_data, dict(response.headers)
                    elif response.status in (429, 403):  # Rate limited or forbidden
                        retries += 1
                        if retries > max_retries:
                            return None, None, dict(response.headers)
                        
                        # Exponential backoff
                        wait_time = retry_delay * (2 ** retries)
                        time.sleep(wait_time)
                        continue
                    else:
                        return None, None, dict(response.headers)
            elif method.upper() == "POST":
                async with session.post(
                    url, params=params, json=data, headers=headers, proxy=proxy
                ) as response:
                    if response.status == 200:
                        # Try to parse as JSON first
                        try:
                            json_data = await response.json()
                            return json_data, None, dict(response.headers)
                        except:
                            # If not JSON, return text
                            text_data = await response.text()
                            return None, text_data, dict(response.headers)
                    elif response.status in (429, 403):  # Rate limited or forbidden
                        retries += 1
                        if retries > max_retries:
                            return None, None, dict(response.headers)
                        
                        # Exponential backoff
                        wait_time = retry_delay * (2 ** retries)
                        time.sleep(wait_time)
                        continue
                    else:
                        return None, None, dict(response.headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        except Exception as e:
            retries += 1
            if retries > max_retries:
                return None, None, None
            
            # Exponential backoff
            wait_time = retry_delay * (2 ** retries)
            time.sleep(wait_time)
            continue
    
    return None, None, None