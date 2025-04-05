"""
Configuration management for the Product Research Engine.

This module provides a configuration class for managing API keys, proxy settings,
timeouts, and other configuration options for the Product Research Engine.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class ProxyConfig:
    """Configuration for proxy settings."""
    enabled: bool = False
    urls: List[str] = field(default_factory=list)
    username: Optional[str] = None
    password: Optional[str] = None
    rotate_requests: int = 10  # Rotate proxy every N requests


@dataclass
class AmazonConfig:
    """Configuration for Amazon integration."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    partner_tag: Optional[str] = None
    region: str = "US"
    use_api: bool = True
    use_scraping: bool = True  # Fallback to scraping if API fails or is unavailable
    timeout: int = 30  # Timeout in seconds
    max_retries: int = 3


@dataclass
class EbayConfig:
    """Configuration for eBay integration."""
    app_id: Optional[str] = None
    cert_id: Optional[str] = None
    dev_id: Optional[str] = None
    use_api: bool = True
    use_scraping: bool = True  # Fallback to scraping if API fails or is unavailable
    timeout: int = 30  # Timeout in seconds
    max_retries: int = 3
    results_per_page: int = 50
    max_pages: int = 5


@dataclass
class TerapeaksConfig:
    """Configuration for Terapeaks integration."""
    username: Optional[str] = None
    password: Optional[str] = None
    use_api: bool = False  # Terapeaks API may not be publicly available
    use_scraping: bool = True
    timeout: int = 45  # Timeout in seconds
    max_retries: int = 3


@dataclass
class ResearchConfig:
    """Main configuration for the Product Research Engine."""
    amazon: AmazonConfig = field(default_factory=AmazonConfig)
    ebay: EbayConfig = field(default_factory=EbayConfig)
    terapeaks: TerapeaksConfig = field(default_factory=TerapeaksConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    
    # General settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    cache_enabled: bool = True
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    
    @classmethod
    def from_env(cls) -> "ResearchConfig":
        """
        Create a configuration instance from environment variables.
        
        Returns:
            ResearchConfig: Configuration instance
        """
        config = cls()
        
        # Amazon config
        config.amazon.api_key = os.environ.get("AMAZON_API_KEY")
        config.amazon.api_secret = os.environ.get("AMAZON_API_SECRET")
        config.amazon.partner_tag = os.environ.get("AMAZON_PARTNER_TAG")
        config.amazon.region = os.environ.get("AMAZON_REGION", "US")
        config.amazon.use_api = os.environ.get("AMAZON_USE_API", "true").lower() == "true"
        config.amazon.use_scraping = os.environ.get("AMAZON_USE_SCRAPING", "true").lower() == "true"
        
        # eBay config
        config.ebay.app_id = os.environ.get("EBAY_APP_ID")
        config.ebay.cert_id = os.environ.get("EBAY_CERT_ID")
        config.ebay.dev_id = os.environ.get("EBAY_DEV_ID")
        config.ebay.use_api = os.environ.get("EBAY_USE_API", "true").lower() == "true"
        config.ebay.use_scraping = os.environ.get("EBAY_USE_SCRAPING", "true").lower() == "true"
        
        # Terapeaks config
        config.terapeaks.username = os.environ.get("TERAPEAKS_USERNAME")
        config.terapeaks.password = os.environ.get("TERAPEAKS_PASSWORD")
        config.terapeaks.use_api = os.environ.get("TERAPEAKS_USE_API", "false").lower() == "true"
        config.terapeaks.use_scraping = os.environ.get("TERAPEAKS_USE_SCRAPING", "true").lower() == "true"
        
        # Proxy config
        config.proxy.enabled = os.environ.get("PROXY_ENABLED", "false").lower() == "true"
        proxy_urls = os.environ.get("PROXY_URLS", "")
        if proxy_urls:
            config.proxy.urls = proxy_urls.split(",")
        config.proxy.username = os.environ.get("PROXY_USERNAME")
        config.proxy.password = os.environ.get("PROXY_PASSWORD")
        
        # General settings
        config.log_level = os.environ.get("LOG_LEVEL", "INFO")
        config.log_file = os.environ.get("LOG_FILE")
        config.cache_enabled = os.environ.get("CACHE_ENABLED", "true").lower() == "true"
        
        return config
    
    @classmethod
    def from_file(cls, file_path: str) -> "ResearchConfig":
        """
        Create a configuration instance from a JSON file.
        
        Args:
            file_path: Path to the JSON configuration file
            
        Returns:
            ResearchConfig: Configuration instance
        """
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        
        config = cls()
        
        # Amazon config
        amazon_dict = config_dict.get("amazon", {})
        config.amazon.api_key = amazon_dict.get("api_key")
        config.amazon.api_secret = amazon_dict.get("api_secret")
        config.amazon.partner_tag = amazon_dict.get("partner_tag")
        config.amazon.region = amazon_dict.get("region", "US")
        config.amazon.use_api = amazon_dict.get("use_api", True)
        config.amazon.use_scraping = amazon_dict.get("use_scraping", True)
        
        # eBay config
        ebay_dict = config_dict.get("ebay", {})
        config.ebay.app_id = ebay_dict.get("app_id")
        config.ebay.cert_id = ebay_dict.get("cert_id")
        config.ebay.dev_id = ebay_dict.get("dev_id")
        config.ebay.use_api = ebay_dict.get("use_api", True)
        config.ebay.use_scraping = ebay_dict.get("use_scraping", True)
        
        # Terapeaks config
        terapeaks_dict = config_dict.get("terapeaks", {})
        config.terapeaks.username = terapeaks_dict.get("username")
        config.terapeaks.password = terapeaks_dict.get("password")
        config.terapeaks.use_api = terapeaks_dict.get("use_api", False)
        config.terapeaks.use_scraping = terapeaks_dict.get("use_scraping", True)
        
        # Proxy config
        proxy_dict = config_dict.get("proxy", {})
        config.proxy.enabled = proxy_dict.get("enabled", False)
        config.proxy.urls = proxy_dict.get("urls", [])
        config.proxy.username = proxy_dict.get("username")
        config.proxy.password = proxy_dict.get("password")
        
        # General settings
        config.log_level = config_dict.get("log_level", "INFO")
        config.log_file = config_dict.get("log_file")
        config.cache_enabled = config_dict.get("cache_enabled", True)
        
        return config
    
    def to_dict(self) -> Dict:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dict: Configuration as a dictionary
        """
        return {
            "amazon": {
                "api_key": self.amazon.api_key,
                "api_secret": self.amazon.api_secret,
                "partner_tag": self.amazon.partner_tag,
                "region": self.amazon.region,
                "use_api": self.amazon.use_api,
                "use_scraping": self.amazon.use_scraping,
                "timeout": self.amazon.timeout,
                "max_retries": self.amazon.max_retries,
            },
            "ebay": {
                "app_id": self.ebay.app_id,
                "cert_id": self.ebay.cert_id,
                "dev_id": self.ebay.dev_id,
                "use_api": self.ebay.use_api,
                "use_scraping": self.ebay.use_scraping,
                "timeout": self.ebay.timeout,
                "max_retries": self.ebay.max_retries,
                "results_per_page": self.ebay.results_per_page,
                "max_pages": self.ebay.max_pages,
            },
            "terapeaks": {
                "username": self.terapeaks.username,
                "password": self.terapeaks.password,
                "use_api": self.terapeaks.use_api,
                "use_scraping": self.terapeaks.use_scraping,
                "timeout": self.terapeaks.timeout,
                "max_retries": self.terapeaks.max_retries,
            },
            "proxy": {
                "enabled": self.proxy.enabled,
                "urls": self.proxy.urls,
                "username": self.proxy.username,
                "password": self.proxy.password,
                "rotate_requests": self.proxy.rotate_requests,
            },
            "log_level": self.log_level,
            "log_file": self.log_file,
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "user_agent": self.user_agent,
        }
    
    def to_file(self, file_path: str) -> None:
        """
        Save the configuration to a JSON file.
        
        Args:
            file_path: Path to the JSON configuration file
        """
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> List[str]:
        """
        Validate the configuration.
        
        Returns:
            List[str]: List of validation errors, empty if valid
        """
        errors = []
        
        # Validate Amazon config
        if self.amazon.use_api:
            if not self.amazon.api_key:
                errors.append("Amazon API key is required when use_api is enabled")
            if not self.amazon.api_secret:
                errors.append("Amazon API secret is required when use_api is enabled")
            if not self.amazon.partner_tag:
                errors.append("Amazon partner tag is required when use_api is enabled")
        
        # Validate eBay config
        if self.ebay.use_api:
            if not self.ebay.app_id:
                errors.append("eBay app ID is required when use_api is enabled")
            if not self.ebay.cert_id:
                errors.append("eBay cert ID is required when use_api is enabled")
            if not self.ebay.dev_id:
                errors.append("eBay dev ID is required when use_api is enabled")
        
        # Validate Terapeaks config
        if self.terapeaks.use_api:
            if not self.terapeaks.username:
                errors.append("Terapeaks username is required when use_api is enabled")
            if not self.terapeaks.password:
                errors.append("Terapeaks password is required when use_api is enabled")
        
        # Validate proxy config
        if self.proxy.enabled and not self.proxy.urls:
            errors.append("Proxy URLs are required when proxy is enabled")
        
        return errors