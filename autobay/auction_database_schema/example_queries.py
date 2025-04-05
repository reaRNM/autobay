"""
Example queries for the auction research tool database.

This script demonstrates how to:
1. Connect to the database
2. Perform various queries using SQLAlchemy ORM
3. Use fuzzy text matching with pg_trgm

Usage:
    python example_queries.py
"""

import os
from sqlalchemy import func, desc, cast, String
from models import get_engine, get_session, Item, AuctionHistory, SalesHistory, ItemCondition

def run_example_queries():
    """Run example queries against the database."""
    # Get database connection string from environment variable or use a default
    connection_string = os.environ.get(
        "DATABASE_URL", 
        "postgresql://username:password@localhost:5432/auction_db"
    )
    
    # Create engine and session
    engine = get_engine(connection_string)
    session = get_session(engine)
    
    try:
        # Example 1: Basic query - Get all items
        print("\n--- Example 1: All Items ---")
        items = session.query(Item).all()
        for item in items:
            print(f"Item: {item.brand} {item.model} ({item.condition.value})")
        
        # Example 2: Join query - Get items with their auction history
        print("\n--- Example 2: Items with Auction History ---")
        items_with_auctions = session.query(
            Item, AuctionHistory
        ).join(
            AuctionHistory
        ).all()
        
        for item, auction in items_with_auctions:
            print(f"Item: {item.brand} {item.model}, Auction Price: ${auction.purchase_price}")
        
        # Example 3: Aggregation - Calculate average purchase price by condition
        print("\n--- Example 3: Average Purchase Price by Condition ---")
        avg_price_by_condition = session.query(
            Item.condition,
            func.avg(AuctionHistory.purchase_price).label('avg_price')
        ).join(
            AuctionHistory
        ).group_by(
            Item.condition
        ).all()
        
        for condition, avg_price in avg_price_by_condition:
            print(f"Condition: {condition.value}, Average Price: ${avg_price:.2f}")
        
        # Example 4: Fuzzy text search - Find items with similar brand names
        print("\n--- Example 4: Fuzzy Text Search for Brands ---")
        search_term = "Aple"  # Misspelled "Apple"
        similar_brands = session.query(
            Item
        ).filter(
            func.similarity(Item.brand, search_term) > 0.3
        ).all()
        
        for item in similar_brands:
            print(f"Found similar to '{search_term}': {item.brand} {item.model}")
        
        # Example 5: Complex query - Profit analysis
        print("\n--- Example 5: Profit Analysis ---")
        profit_analysis = session.query(
            Item.brand,
            Item.model,
            AuctionHistory.purchase_price,
            SalesHistory.listing_price,
            (SalesHistory.listing_price - AuctionHistory.purchase_price).label('profit')
        ).join(
            AuctionHistory
        ).join(
            SalesHistory
        ).order_by(
            desc('profit')
        ).all()
        
        for brand, model, purchase, sale, profit in profit_analysis:
            print(f"{brand} {model}: Purchase ${purchase}, Sale ${sale}, Profit ${profit:.2f}")
        
    except Exception as e:
        print(f"Error running queries: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    run_example_queries()