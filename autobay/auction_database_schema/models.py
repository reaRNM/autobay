"""
Database schema for AI-powered auction research and resale automation tool.

This module defines the SQLAlchemy ORM models for the application's database schema.
The schema consists of three main tables:
1. Items - Core product data including UPC, brand, model, and condition
2. AuctionHistory - Records of auction transactions for items
3. SalesHistory - Records of sales transactions for items

The tables are designed with appropriate relationships, constraints, and indexes
to support efficient querying and data integrity.

Note: This schema requires the pg_trgm PostgreSQL extension to be enabled for
fuzzy text matching. Enable it by running: CREATE EXTENSION pg_trgm;
"""

from datetime import datetime
from typing import Optional, List
import enum

from sqlalchemy import (
    Column, Integer, String, Numeric, DateTime, 
    ForeignKey, Text, JSON, Enum, Index, func, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# Create the declarative base class
Base = declarative_base()

# Define an enum for item conditions
class ItemCondition(enum.Enum):
    NEW = "New"
    USED = "Used"
    REFURBISHED = "Refurbished"
    OPEN_BOX = "Open Box"
    DAMAGED = "Damaged"
    PARTS_ONLY = "Parts Only"

class Item(Base):
    """
    Core product data table storing information about items.
    This table serves as the central reference for both auction and sales history.
    """
    __tablename__ = 'items'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    upc = Column(String(50), index=True, unique=True, nullable=False, 
                comment="Universal Product Code - unique identifier for the product")
    brand = Column(String(100), nullable=False, index=True,
                 comment="Brand name of the product")
    model = Column(String(100), nullable=False, index=True,
                 comment="Model name/number of the product")
    condition = Column(Enum(ItemCondition), nullable=False, 
                     comment="Condition of the item (New, Used, Refurbished, etc.)")
    additional_info = Column(JSON, nullable=True, 
                           comment="Additional product details stored as JSON")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False,
                      comment="Timestamp when the record was created")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, 
                      nullable=False, comment="Timestamp when the record was last updated")
    
    # Relationships
    auction_history = relationship("AuctionHistory", back_populates="item", cascade="all, delete-orphan")
    sales_history = relationship("SalesHistory", back_populates="item", cascade="all, delete-orphan")
    
    # Create indexes for text fields that will use fuzzy matching
    __table_args__ = (
        Index('ix_items_upc_trgm', upc, postgresql_using='gin', 
              postgresql_ops={'upc': 'gin_trgm_ops'}),
        Index('ix_items_brand_trgm', brand, postgresql_using='gin', 
              postgresql_ops={'brand': 'gin_trgm_ops'}),
        Index('ix_items_model_trgm', model, postgresql_using='gin', 
              postgresql_ops={'model': 'gin_trgm_ops'}),
    )
    
    def __repr__(self):
        return f"<Item(id={self.id}, upc='{self.upc}', brand='{self.brand}', model='{self.model}')>"


class AuctionHistory(Base):
    """
    Records of auction transactions for items.
    Stores details about purchases made at auctions including prices, fees, and dates.
    """
    __tablename__ = 'auction_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(Integer, ForeignKey('items.id', ondelete='CASCADE'), nullable=False,
                   index=True, comment="Reference to the item that was auctioned")
    purchase_price = Column(Numeric(10, 2), nullable=False, 
                          comment="Base purchase price at auction")
    final_bid = Column(Numeric(10, 2), nullable=False, 
                     comment="Final bid amount")
    buyer_premium = Column(Numeric(10, 2), default=0.0, nullable=False, 
                         comment="Additional premium charged to the buyer")
    taxes = Column(Numeric(10, 2), default=0.0, nullable=False, 
                 comment="Taxes applied to the purchase")
    auction_fees = Column(Numeric(10, 2), default=0.0, nullable=False, 
                        comment="Additional auction fees")
    auction_date = Column(DateTime, nullable=False, index=True,
                        comment="Date and time when the auction took place")
    additional_details = Column(JSON, nullable=True, 
                              comment="Additional auction-specific details stored as JSON")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False,
                      comment="Timestamp when the record was created")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, 
                      nullable=False, comment="Timestamp when the record was last updated")
    
    # Relationship back to the Item
    item = relationship("Item", back_populates="auction_history")
    
    # Create an index on auction_date for efficient date-based queries
    __table_args__ = (
        Index('ix_auction_history_date', auction_date),
    )
    
    def __repr__(self):
        return f"<AuctionHistory(id={self.id}, item_id={self.item_id}, purchase_price={self.purchase_price})>"


class SalesHistory(Base):
    """
    Records of sales transactions for items.
    Stores details about sales including listing price, duration, shipping costs, and customer feedback.
    """
    __tablename__ = 'sales_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(Integer, ForeignKey('items.id', ondelete='CASCADE'), nullable=False,
                   index=True, comment="Reference to the item that was sold")
    listing_price = Column(Numeric(10, 2), nullable=False, 
                         comment="Price at which the item was listed for sale")
    duration = Column(Integer, nullable=True, 
                    comment="Duration in days that the item was listed")
    accepted_offers = Column(Numeric(10, 2), nullable=True, 
                           comment="Value of accepted offers, if applicable")
    seo_keywords = Column(Text, nullable=True, 
                        comment="SEO keywords used in the listing")
    shipping_costs = Column(Numeric(10, 2), default=0.0, nullable=False, 
                          comment="Costs associated with shipping the item")
    watcher_count = Column(Integer, default=0, nullable=False, 
                         comment="Number of watchers at listing time")
    customer_feedback = Column(Text, nullable=True, 
                             comment="Customer reviews, complaints, or return information")
    sale_date = Column(DateTime, nullable=False, index=True,
                     comment="Date and time when the sale occurred")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False,
                      comment="Timestamp when the record was created")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, 
                      nullable=False, comment="Timestamp when the record was last updated")
    
    # Relationship back to the Item
    item = relationship("Item", back_populates="sales_history")
    
    # Create indexes for efficient querying and fuzzy text search
    __table_args__ = (
        Index('ix_sales_history_date', sale_date),
        Index('ix_sales_history_seo_keywords_trgm', seo_keywords, postgresql_using='gin', 
              postgresql_ops={'seo_keywords': 'gin_trgm_ops'}),
        Index('ix_sales_history_customer_feedback_trgm', customer_feedback, postgresql_using='gin', 
              postgresql_ops={'customer_feedback': 'gin_trgm_ops'}),
    )
    
    def __repr__(self):
        return f"<SalesHistory(id={self.id}, item_id={self.item_id}, listing_price={self.listing_price})>"


# Database connection and session configuration
def get_engine(connection_string):
    """
    Create a SQLAlchemy engine using the provided connection string.
    
    Args:
        connection_string (str): PostgreSQL connection string
        
    Returns:
        Engine: SQLAlchemy engine instance
    """
    return create_engine(connection_string)


def get_session(engine):
    """
    Create a SQLAlchemy session factory bound to the provided engine.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Returns:
        sessionmaker: SQLAlchemy session factory
    """
    Session = sessionmaker(bind=engine)
    return Session()


def init_db(connection_string):
    """
    Initialize the database by creating all defined tables.
    
    Args:
        connection_string (str): PostgreSQL connection string
        
    Returns:
        Engine: SQLAlchemy engine instance
    """
    engine = get_engine(connection_string)
    Base.metadata.create_all(engine)
    return engine