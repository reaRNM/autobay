"""
Example usage of the AI-Powered Listing Generator.

This script demonstrates how to use the Listing Generator
to create optimized product listings.
"""

import os
import json
import logging
import asyncio
from datetime import datetime

from listing_generator import (
    ListingGenerator,
    Product,
    Marketplace
)
from listing_generator.utils import setup_logging


# Set up logging
logger = setup_logging(log_level="INFO")


async def main():
    """Main function."""
    logger.info("Starting Listing Generator example")
    
    # Initialize listing generator
    generator = ListingGenerator()
    
    # Create a sample product
    product = Product(
        id="sample-product-001",
        name="Wireless Bluetooth Headphones",
        brand="SoundMaster",
        model="BT-500",
        category="Electronics",
        subcategory="Headphones",
        features=[
            "Bluetooth 5.0 connectivity",
            "Active noise cancellation",
            "40-hour battery life",
            "Comfortable over-ear design",
            "Built-in microphone for calls",
            "Quick charge (10 min charge = 5 hours playback)"
        ],
        specifications={
            "connectivity": "Bluetooth 5.0",
            "battery_life": "40 hours",
            "weight_oz": 8.5,
            "color": "Black",
            "charging": "USB-C",
            "frequency_response": "20Hz - 20kHz"
        },
        description="Experience premium sound quality with SoundMaster BT-500 wireless headphones. These over-ear headphones feature active noise cancellation and long battery life.",
        condition="New",
        weight_oz=8.5,
        dimensions={
            "length_in": 7.5,
            "width_in": 6.8,
            "height_in": 3.2
        },
        image_urls=[
            "https://example.com/images/headphones1.jpg",
            "https://example.com/images/headphones2.jpg",
            "https://example.com/images/headphones3.jpg"
        ],
        msrp=149.99,
        cost=65.00
    )
    
    # Save product to database
    await generator.db.save_product(product)
    
    # Generate listing for eBay
    logger.info("Generating eBay listing...")
    ebay_listing = await generator.generate_listing(
        product,
        Marketplace.EBAY,
        generate_variations=True,
        num_variations=3
    )
    
    logger.info(f"Generated eBay listing: {ebay_listing.id}")
    logger.info(f"Title: {ebay_listing.title}")
    logger.info(f"Price: ${ebay_listing.price:.2f}")
    
    # Print title variations
    logger.info("Title variations:")
    for i, variation in enumerate(ebay_listing.title_variations):
        logger.info(f"{i+1}. {variation.title}")
    
    # Generate listing for Amazon
    logger.info("\nGenerating Amazon listing...")
    amazon_listing = await generator.generate_listing(
        product,
        Marketplace.AMAZON,
        generate_variations=True,
        num_variations=3
    )
    
    logger.info(f"Generated Amazon listing: {amazon_listing.id}")
    logger.info(f"Title: {amazon_listing.title}")
    logger.info(f"Price: ${amazon_listing.price:.2f}")
    
    # Print title variations
    logger.info("Title variations:")
    for i, variation in enumerate(amazon_listing.title_variations):
        logger.info(f"{i+1}. {variation.title}")
    
    # Simulate tracking performance for eBay listing
    logger.info("\nSimulating performance tracking for eBay listing...")
    
    # Day 1 metrics
    day1_metrics = {
        "impressions": 120,
        "clicks": 8,
        "add_to_carts": 3,
        "purchases": 1,
        "revenue": ebay_listing.price,
        "search_terms": ["wireless headphones", "bluetooth headphones", "noise cancelling"]
    }
    
    performance = await generator.track_performance(ebay_listing.id, day1_metrics)
    
    logger.info(f"Day 1 CTR: {performance.ctr:.4f}")
    logger.info(f"Day 1 Conversion Rate: {performance.conversion_rate:.4f}")
    
    # Day 2 metrics (improved)
    day2_metrics = {
        "impressions": 150,
        "clicks": 12,
        "add_to_carts": 5,
        "purchases": 2,
        "revenue": ebay_listing.price * 2,
        "search_terms": ["soundmaster headphones", "bt-500 headphones"]
    }
    
    performance = await generator.track_performance(ebay_listing.id, day2_metrics)
    
    logger.info(f"Day 2 CTR: {performance.ctr:.4f}")
    logger.info(f"Day 2 Conversion Rate: {performance.conversion_rate:.4f}")
    
    # Optimize eBay listing
    logger.info("\nOptimizing eBay listing...")
    optimized_listing = await generator.optimize_title(ebay_listing.id, use_ab_testing=True)
    
    logger.info(f"Optimized title: {optimized_listing.title}")
    
    # Update pricing
    logger.info("\nUpdating pricing for eBay listing...")
    updated_listing = await generator.update_pricing(ebay_listing.id, dynamic_adjustment=True)
    
    logger.info(f"Updated price: ${updated_listing.price:.2f}")
    
    # Save listings to JSON files
    with open("ebay_listing.json", "w") as f:
        json.dump(ebay_listing.to_dict(), f, indent=2)
    
    with open("amazon_listing.json", "w") as f:
        json.dump(amazon_listing.to_dict(), f, indent=2)
    
    logger.info("\nListings saved to JSON files")
    
    # Start API server (in a real implementation, this would be in a separate script)
    logger.info("\nAPI server can be started with:")
    logger.info("from listing_generator.api import create_app")
    logger.info("app = create_app()")
    logger.info("app.run(host='0.0.0.0', port=5000)")


if __name__ == "__main__":
    asyncio.run(main())