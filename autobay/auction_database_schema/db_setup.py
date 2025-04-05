"""
Database setup and initialization script for the auction research tool.

This script demonstrates how to:
1. Connect to a PostgreSQL database
2. Enable the pg_trgm extension
3. Create the database schema
4. Add sample data

Usage:
    python db_setup.py
"""

import os
from sqlalchemy import text
from models import init_db, get_session, Item, AuctionHistory, SalesHistory, ItemCondition
from datetime import datetime, timedelta

def setup_database():
    """Set up the database with schema and sample data."""
    # Get database connection string from environment variable or use a default
    connection_string = os.environ.get(
        "DATABASE_URL", 
        "postgresql://username:password@localhost:5432/auction_db"
    )
    
    # Initialize the database and get the engine
    engine = init_db(connection_string)
    
    # Create a session
    session = get_session(engine)
    
    try:
        # Enable pg_trgm extension for fuzzy text matching
        session.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        session.commit()
        print("pg_trgm extension enabled successfully")
        
        # Add sample data
        add_sample_data(session)
        
        print("Database setup completed successfully")
    except Exception as e:
        session.rollback()
        print(f"Error setting up database: {e}")
    finally:
        session.close()

def add_sample_data(session):
    """Add sample data to the database."""
    # Create sample items
    item1 = Item(
        upc="123456789012",
        brand="Apple",
        model="iPhone 13 Pro",
        condition=ItemCondition.USED,
        additional_info={"color": "Graphite", "storage": "256GB"}
    )
    
    item2 = Item(
        upc="987654321098",
        brand="Samsung",
        model="Galaxy S21",
        condition=ItemCondition.REFURBISHED,
        additional_info={"color": "Phantom Black", "storage": "128GB"}
    )
    
    # Add items to session
    session.add_all([item1, item2])
    session.flush()  # Flush to get the IDs
    
    # Create sample auction history
    auction1 = AuctionHistory(
        item_id=item1.id,
        purchase_price=700.00,
        final_bid=680.00,
        buyer_premium=68.00,
        taxes=56.70,
        auction_fees=25.00,
        auction_date=datetime.utcnow() - timedelta(days=30),
        additional_details={"auction_house": "TechBid", "lot_number": "A123"}
    )
    
    auction2 = AuctionHistory(
        item_id=item2.id,
        purchase_price=450.00,
        final_bid=430.00,
        buyer_premium=43.00,
        taxes=35.45,
        auction_fees=20.00,
        auction_date=datetime.utcnow() - timedelta(days=45),
        additional_details={"auction_house": "GadgetAuctions", "lot_number": "B456"}
    )
    
    # Create sample sales history
    sale1 = SalesHistory(
        item_id=item1.id,
        listing_price=899.99,
        duration=7,
        accepted_offers=850.00,
        seo_keywords="iPhone 13 Pro Apple smartphone used excellent condition",
        shipping_costs=12.50,
        watcher_count=24,
        customer_feedback="Great phone, exactly as described!",
        sale_date=datetime.utcnow() - timedelta(days=15)
    )
    
    sale2 = SalesHistory(
        item_id=item2.id,
        listing_price=599.99,
        duration=10,
        accepted_offers=None,  # Sold at listing price
        seo_keywords="Samsung Galaxy S21 refurbished Android smartphone",
        shipping_costs=10.00,
        watcher_count=18,
        customer_feedback="Phone works perfectly, fast shipping",
        sale_date=datetime.utcnow() - timedelta(days=20)
    )
    
    # Add auction and sales history to session
    session.add_all([auction1, auction2, sale1, sale2])
    
    # Commit the changes
    session.commit()
    print("Sample data added successfully")

if __name__ == "__main__":
    setup_database()