"""
Example usage of the auction tool.
"""

import asyncio
import json
import os
from typing import List

from auction_tool.scraper import HiBidScraper, ProxyManager, UserAgentManager
from auction_tool.fee_calculator import FeeCalculator, FeeSettings
from auction_tool.utils import setup_logging


async def scrape_auctions(keyword: str, max_pages: int = 2) -> List[dict]:
    """
    Scrape auctions from HiBid.com.
    
    Args:
        keyword: Search keyword
        max_pages: Maximum number of search result pages to scrape
        
    Returns:
        List[dict]: List of scraped auction items
    """
    # Set up proxy manager (in a real scenario, you would provide actual proxies)
    proxy_manager = ProxyManager()
    
    # Set up user agent manager
    user_agent_manager = UserAgentManager()
    
    # Create scraper
    async with HiBidScraper(
        proxy_manager=proxy_manager,
        user_agent_manager=user_agent_manager,
        max_concurrent_requests=3,
        request_delay=2.0
    ) as scraper:
        # Search for auctions
        auction_urls = await scraper.search_auctions(
            keyword=keyword,
            max_pages=max_pages
        )
        
        # Limit to first 5 auctions for the example
        auction_urls = auction_urls[:5]
        
        # Scrape items from auctions
        items = await scraper.scrape_multiple_auctions(auction_urls)
        
        # Convert to dictionaries
        return [item.__dict__ for item in items]


def analyze_items(items: List[dict]) -> List[dict]:
    """
    Analyze profit potential for items.
    
    Args:
        items: List of scraped auction items
        
    Returns:
        List[dict]: List of items with profit analysis
    """
    # Create custom fee settings
    settings = FeeSettings(
        buyer_premium_rate=0.15,
        sales_tax_rate=0.07,
        ebay_final_value_fee_rate=0.1275,
        default_shipping_cost=12.0
    )
    
    # Create fee calculator
    calculator = FeeCalculator(settings=settings)
    
    results = []
    
    for item in items:
        # Skip items without bid amount
        if not item.get("bid_amount"):
            continue
            
        # Analyze profit
        analysis = calculator.analyze_profit(
            bid_amount=item["bid_amount"],
            shipping_cost=settings.default_shipping_cost
        )
        
        # Add analysis to item
        item_with_analysis = item.copy()
        item_with_analysis["profit_analysis"] = {
            "estimated_sale_price": analysis.estimated_sale_price,
            "total_cost": analysis.total_cost,
            "avg_profit": analysis.avg_profit,
            "profit_margin": analysis.profit_margin,
            "recommended_price": analysis.recommended_price
        }
        
        results.append(item_with_analysis)
        
    return results


async def main():
    """Main function."""
    # Set up logging
    logger = setup_logging(log_level=20)  # INFO level
    
    try:
        # Scrape auctions
        logger.info("Scraping auctions...")
        items = await scrape_auctions(keyword="electronics", max_pages=1)
        
        # If no items were scraped, use sample data
        if not items:
            logger.info("No items scraped, using sample data...")
            items = [
                {
                    "item_id": "12345",
                    "name": "Apple iPhone 12 Pro",
                    "brand": "Apple",
                    "model": "iPhone 12 Pro",
                    "bid_amount": 450.0,
                    "condition": "Used - Good",
                    "auction_url": "https://www.hibid.com/lot/12345"
                },
                {
                    "item_id": "67890",
                    "name": "Samsung Galaxy S21",
                    "brand": "Samsung",
                    "model": "Galaxy S21",
                    "bid_amount": 350.0,
                    "condition": "Refurbished",
                    "auction_url": "https://www.hibid.com/lot/67890"
                },
                {
                    "item_id": "24680",
                    "name": "Sony PlayStation 5",
                    "brand": "Sony",
                    "model": "PlayStation 5",
                    "bid_amount": 400.0,
                    "condition": "New",
                    "auction_url": "https://www.hibid.com/lot/24680"
                }
            ]
        
        # Analyze profit potential
        logger.info("Analyzing profit potential...")
        results = analyze_items(items)
        
        # Print results
        logger.info(f"Analyzed {len(results)} items")
        for item in results:
            analysis = item["profit_analysis"]
            logger.info(
                f"Item: {item['name']} - "
                f"Bid: ${item['bid_amount']:.2f}, "
                f"Est. Sale: ${analysis['estimated_sale_price']:.2f}, "
                f"Profit: ${analysis['avg_profit']:.2f} "
                f"({analysis['profit_margin']:.2f}%)"
            )
            
        # Save results to file
        with open("auction_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info("Results saved to auction_results.json")
        
    except Exception as e:
        logger.exception(f"Error in main function: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())