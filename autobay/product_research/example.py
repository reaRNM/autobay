"""
Example usage of the Product Research Engine.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any

from product_research import (
    ResearchConfig,
    AmazonResearcher,
    EbayResearcher,
    TerapeaksResearcher,
    ProductDataAggregator
)
from product_research.utils import setup_logging


async def research_product(
    identifier: str,
    identifier_type: str = "keywords",
    config: ResearchConfig = None
) -> Dict[str, Any]:
    """
    Research a product using the Product Research Engine.
    
    Args:
        identifier: Product identifier (keywords, UPC, ASIN, etc.)
        identifier_type: Type of identifier ("keywords", "upc", "asin", etc.)
        config: Research configuration
        
    Returns:
        Dict[str, Any]: Product research data
    """
    # Create default configuration if not provided
    if config is None:
        config = ResearchConfig.from_env()
    
    # Create product data aggregator
    aggregator = ProductDataAggregator(config)
    
    # Get product data
    product_data = await aggregator.get_product_data(identifier, identifier_type)
    
    return product_data


async def research_amazon_only(
    identifier: str,
    identifier_type: str = "keywords",
    config: ResearchConfig = None
) -> Dict[str, Any]:
    """
    Research a product using only Amazon.
    
    Args:
        identifier: Product identifier (keywords, UPC, ASIN, etc.)
        identifier_type: Type of identifier ("keywords", "upc", "asin", etc.)
        config: Research configuration
        
    Returns:
        Dict[str, Any]: Amazon product data
    """
    # Create default configuration if not provided
    if config is None:
        config = ResearchConfig.from_env()
    
    # Create Amazon researcher
    researcher = AmazonResearcher(config)
    
    # Get product data
    product_data = await researcher.get_product_data(
        identifier,
        "asin" if identifier_type == "asin" else "upc" if identifier_type == "upc" else "keywords"
    )
    
    return product_data


async def research_ebay_only(
    identifier: str,
    identifier_type: str = "keywords",
    config: ResearchConfig = None
) -> Dict[str, Any]:
    """
    Research a product using only eBay.
    
    Args:
        identifier: Product identifier (keywords, UPC, ASIN, etc.)
        identifier_type: Type of identifier ("keywords", "upc", "asin", etc.)
        config: Research configuration
        
    Returns:
        Dict[str, Any]: eBay product data
    """
    # Create default configuration if not provided
    if config is None:
        config = ResearchConfig.from_env()
    
    # Create eBay researcher
    researcher = EbayResearcher(config)
    
    # Get product data
    product_data = await researcher.get_product_data(identifier, identifier_type)
    
    return product_data


async def research_terapeaks_only(
    identifier: str,
    config: ResearchConfig = None
) -> Dict[str, Any]:
    """
    Research a product using only Terapeaks.
    
    Args:
        identifier: Product identifier (keywords)
        config: Research configuration
        
    Returns:
        Dict[str, Any]: Terapeaks product data
    """
    # Create default configuration if not provided
    if config is None:
        config = ResearchConfig.from_env()
    
    # Create Terapeaks researcher
    researcher = TerapeaksResearcher(config)
    
    # Get product data
    product_data = await researcher.get_product_data(identifier)
    
    return product_data


async def main():
    """Main function."""
    # Set up logging
    logger = setup_logging(log_level="INFO")
    
    # Create configuration
    config = ResearchConfig.from_env()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        return
    
    try:
        # Get product identifier from command line arguments or use default
        identifier = os.environ.get("PRODUCT_IDENTIFIER", "iPhone 13 Pro")
        identifier_type = os.environ.get("IDENTIFIER_TYPE", "keywords")
        
        logger.info(f"Researching product: {identifier} ({identifier_type})")
        
        # Research product
        product_data = await research_product(identifier, identifier_type, config)
        
        # Print results
        logger.info(f"Research complete for: {identifier}")
        
        # Save results to file
        with open("product_research_results.json", "w") as f:
            json.dump(product_data, f, indent=2)
            
        logger.info("Results saved to product_research_results.json")
        
        # Print summary
        sources = product_data.get("sources", {})
        aggregated = product_data.get("aggregated", {})
        
        print("\nProduct Research Summary:")
        print(f"  Title: {aggregated.get('title')}")
        print(f"  Brand: {aggregated.get('brand')}")
        print(f"  Model: {aggregated.get('model')}")
        
        pricing = aggregated.get("pricing", {})
        if pricing:
            print("\nPricing:")
            if "amazon" in pricing:
                print(f"  Amazon: ${pricing['amazon'].get('current_price')}")
            if "ebay" in pricing and "avg" in pricing["ebay"]:
                print(f"  eBay (avg): ${pricing['ebay']['avg']}")
            if "terapeaks_sold" in pricing:
                print(f"  Terapeaks (sold avg): ${pricing['terapeaks_sold'].get('avg_price')}")
            if "recommended" in pricing and "price" in pricing["recommended"]:
                print(f"  Recommended: ${pricing['recommended']['price']:.2f}")
        
        sales_velocity = aggregated.get("sales_velocity", {})
        if sales_velocity:
            print("\nSales Velocity:")
            if "amazon" in sales_velocity:
                print(f"  Amazon Rank: #{sales_velocity['amazon'].get('rank_in_category')} in {sales_velocity['amazon'].get('category')}")
            if "ebay" in sales_velocity:
                print(f"  eBay Watchers: {sales_velocity['ebay'].get('total_watchers')}")
            if "terapeaks" in sales_velocity:
                print(f"  Terapeaks Sold: {sales_velocity['terapeaks'].get('total_sold')}")
        
    except Exception as e:
        logger.exception(f"Error in main function: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())