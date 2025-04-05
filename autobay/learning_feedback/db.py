"""
Database module for the Learning & Feedback Systems Module.

This module provides functionality for interacting with the database.
"""

import logging
import sqlite3
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import os

from learning_feedback.models import (
    AuctionOutcome, UserFeedback, ModelPerformance, 
    BidStrategy, ShippingPerformance, ItemCategory,
    UserPreference, AuctionStatus, FeedbackType
)


logger = logging.getLogger(__name__)


class Database:
    """
    Database connection and operations.
    
    This class provides methods for interacting with the database.
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize the Database.
        
        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        
        # Extract database type and path
        if connection_string.startswith('sqlite:///'):
            self.db_type = 'sqlite'
            self.db_path = connection_string[10:]
        else:
            # Default to SQLite
            self.db_type = 'sqlite'
            self.db_path = 'learning.db'
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"Database initialized ({self.db_type})")
    
    def _initialize_database(self) -> None:
        """Initialize the database schema."""
        if self.db_type == 'sqlite':
            # Create database directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                parent_id TEXT,
                attributes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS auction_outcomes (
                id TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                category_id TEXT NOT NULL,
                listing_id TEXT,
                estimated_price REAL NOT NULL,
                start_price REAL NOT NULL,
                reserve_price REAL,
                final_price REAL,
                shipping_cost REAL,
                fees REAL,
                profit REAL,
                roi REAL,
                views INTEGER NOT NULL,
                watchers INTEGER NOT NULL,
                questions INTEGER NOT NULL,
                bids INTEGER NOT NULL,
                status TEXT NOT NULL,
                time_to_sale INTEGER,
                listing_quality_score REAL,
                ai_confidence_score REAL,
                listing_date TEXT NOT NULL,
                end_date TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                item_id TEXT,
                auction_id TEXT,
                feedback_type TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                rating INTEGER,
                comment TEXT,
                ai_suggestion TEXT,
                user_correction TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS shipping_performance (
                id TEXT PRIMARY KEY,
                auction_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                carrier TEXT NOT NULL,
                service_level TEXT NOT NULL,
                package_weight REAL NOT NULL,
                package_dimensions TEXT NOT NULL,
                origin_zip TEXT NOT NULL,
                destination_zip TEXT NOT NULL,
                estimated_cost REAL NOT NULL,
                actual_cost REAL NOT NULL,
                estimated_delivery_days INTEGER NOT NULL,
                actual_delivery_days INTEGER,
                status TEXT NOT NULL,
                issues TEXT,
                customer_satisfaction INTEGER,
                ship_date TEXT,
                delivery_date TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS bid_strategies (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                category_id TEXT,
                name TEXT NOT NULL,
                description TEXT,
                max_bid_percentage REAL NOT NULL,
                early_bid_threshold REAL NOT NULL,
                late_bid_threshold REAL NOT NULL,
                bid_increment_factor REAL NOT NULL,
                max_bid_count INTEGER NOT NULL,
                risk_tolerance REAL NOT NULL,
                success_rate REAL NOT NULL,
                average_roi REAL NOT NULL,
                metadata TEXT,
                is_active INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_type TEXT NOT NULL,
                category_id TEXT,
                accuracy REAL NOT NULL,
                precision REAL NOT NULL,
                recall REAL NOT NULL,
                f1_score REAL NOT NULL,
                mean_absolute_error REAL,
                mean_squared_error REAL,
                r_squared REAL,
                sample_count INTEGER NOT NULL,
                training_duration REAL NOT NULL,
                inference_latency REAL NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                category_preferences TEXT,
                platform_preferences TEXT,
                price_range_min REAL,
                price_range_max REAL,
                risk_tolerance REAL NOT NULL,
                preferred_condition TEXT,
                preferred_shipping_methods TEXT,
                preferred_carriers TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_auction_outcomes_user_id ON auction_outcomes(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_auction_outcomes_item_id ON auction_outcomes(item_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_auction_outcomes_category_id ON auction_outcomes(category_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_auction_outcomes_status ON auction_outcomes(status)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_feedback_user_id ON user_feedback(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_feedback_item_id ON user_feedback(item_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_feedback_auction_id ON user_feedback(auction_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_feedback_feedback_type ON user_feedback(feedback_type)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_shipping_performance_user_id ON shipping_performance(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_shipping_performance_auction_id ON shipping_performance(auction_id)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bid_strategies_user_id ON bid_strategies(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bid_strategies_category_id ON bid_strategies(category_id)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_performance_model_type ON model_performance(model_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_performance_category_id ON model_performance(category_id)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id)')
            
            # Commit changes
            conn.commit()
            conn.close()
    
    async def get_categories(self) -> List[ItemCategory]:
        """
        Get all categories.
        
        Returns:
            List of ItemCategory objects
        """
        if self.db_type == 'sqlite':
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM categories')
            rows = cursor.fetchall()
            
            categories = []
            for row in rows:
                categories.append(ItemCategory(
                    id=row['id'],
                    name=row['name'],
                    parent_id=row['parent_id'],
                    attributes=json.loads(row['attributes']) if row['attributes'] else {},
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                ))
            
            conn.close()
            return categories
    
    async def get_category(self, category_id: str) -> Optional[ItemCategory]:
        """
        Get a category by ID.
        
        Args:
            category_id: Category ID
            
        Returns:
            ItemCategory object or None if not found
        """
        if self.db_type == 'sqlite':
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM categories WHERE id = ?', (category_id,))
            row = cursor.fetchone()
            
            if row:
                category = ItemCategory(
                    id=row['id'],
                    name=row['name'],
                    parent_id=row['parent_id'],
                    attributes=json.loads(row['attributes']) if row['attributes'] else {},
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
                
                conn.close()
                return category
            
            conn.close()
            return None
    
    async def save_category(self, category: ItemCategory) -> None:
        """
        Save a category.
        
        Args:
            category: ItemCategory object
        """
        if self.db_type == 'sqlite':
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO categories (
                id, name, parent_id, attributes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                category.id,
                category.name,
                category.parent_id,
                json.dumps(category.attributes),
                category.created_at.isoformat(),
                category.updated_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
    
    async def get_auction_outcomes(
        self,
        user_id: Optional[str] = None,
        item_id: Optional[str] = None,
        category_id: Optional[str] = None,
        status: Optional[Union[AuctionStatus, str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuctionOutcome]:
        """
        Get auction outcomes with optional filtering.
        
        Args:
            user_id: Filter by user ID
            item_id: Filter by item ID
            category_id: Filter by category ID
            status: Filter by status
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of AuctionOutcome objects
        """
        if self.db_type == 'sqlite':
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM auction_outcomes WHERE 1=1'
            params = []
            
            if user_id:
                query += ' AND user_id = ?'
                params.append(user_id)
            
            if item_id:
                query += ' AND item_id = ?'
                params.append(item_id)
            
            if category_id:
                query += ' AND category_id = ?'
                params.append(category_id)
            
            if status:
                if isinstance(status, AuctionStatus):
                    query += ' AND status = ?'
                    params.append(status.value)
                else:
                    query += ' AND status = ?'
                    params.append(status)
            
            query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            auctions = []
            for row in rows:
                auctions.append(AuctionOutcome(
                    id=row['id'],
                    item_id=row['item_id'],
                    user_id=row['user_id'],
                    platform=row['platform'],
                    category_id=row['category_id'],
                    listing_id=row['listing_id'],
                    estimated_price=row['estimated_price'],
                    start_price=row['start_price'],
                    reserve_price=row['reserve_price'],
                    final_price=row['final_price'],
                    shipping_cost=row['shipping_cost'],
                    fees=row['fees'],
                    profit=row['profit'],
                    roi=row['roi'],
                    views=row['views'],
                    watchers=row['watchers'],
                    questions=row['questions'],
                    bids=row['bids'],
                    status=AuctionStatus(row['status']),
                    time_to_sale=row['time_to_sale'],
                    listing_quality_score=row['listing_quality_score'],
                    ai_confidence_score=row['ai_confidence_score'],
                    listing_date=datetime.fromisoformat(row['listing_date']),
                    end_date=datetime.fromisoformat(row['end_date']) if row['end_date'] else None,
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                ))
            
            conn.close()
            return auctions
    
    async def get_auction_outcome(self, auction_id: str) -> Optional[AuctionOutcome]:
        """
        Get an auction outcome by ID.
        
        Args:
            auction_id: Auction ID
            
        Returns:
            AuctionOutcome object or None if not found
        """
        if self.db_type == 'sqlite':
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM auction_outcomes WHERE id = ?', (auction_id,))
            row = cursor.fetchone()
            
            if row:
                auction = AuctionOutcome(
                    id=row['id'],
                    item_id=row['item_id'],
                    user_id=row['user_id'],
                    platform=row['platform'],
                    category_id=row['category_id'],
                    listing_id=row['listing_id'],
                    estimated_price=row['estimated_price'],
                    start_price=row['start_price'],
                    reserve_price=row['reserve_price'],
                    final_price=row['final_price'],
                    shipping_cost=row['shipping_cost'],
                    fees=row['fees'],
                    profit=row['profit'],
                    roi=row['roi'],
                    views=row['views'],
                    watchers=row['watchers'],
                    questions=row['questions'],
                    bids=row['bids'],
                    status=AuctionStatus(row['status']),
                    time_to_sale=row['time_to_sale'],
                    listing_quality_score=row['listing_quality_score'],
                    ai_confidence_score=row['ai_confidence_score'],
                    listing_date=datetime.fromisoformat(row['listing_date']),
                    end_date=datetime.fromisoformat(row['end_date']) if row['end_date'] else None,
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
                
                conn.close()
                return auction
            
            conn.close()
            return None
    
    async def save_auction_outcome(self, auction: AuctionOutcome) -> None:
        """
        Save an auction outcome.
        
        Args:
            auction: AuctionOutcome object
        """
        if self.db_type == 'sqlite':
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO auction_outcomes (
                id, item_id, user_id, platform, category_id, listing_id,
                estimated_price, start_price, reserve_price, final_price,
                shipping_cost, fees, profit, roi, views, watchers, questions,
                bids, status, time_to_sale, listing_quality_score,
                ai_confidence_score, listing_date, end_date, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                auction.id,
                auction.item_id,
                auction.user_id,
                auction.platform,
                auction.category_id,
                auction.listing_id,
                auction.estimated_price,
                auction.start_price,
                auction.reserve_price,
                auction.final_price,
                auction.shipping_cost,
                auction.fees,
                auction.profit,
                auction.roi,
                auction.views,
                auction.watchers,
                auction.questions,
                auction.bids,
                auction.status.value,
                auction.time_to_sale,
                auction.listing_quality_score,
                auction.ai_confidence_score,
                auction.listing_date.isoformat(),
                auction.end_date.isoformat() if auction.end_date else None,
                auction.created_at.isoformat(),
                auction.updated_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
    
    async def get_user_feedback(
        self,
        user_id: Optional[str] = None,
        item_id: Optional[str] = None,
        auction_id: Optional[str] = None,
        feedback_type: Optional[Union[FeedbackType, str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[UserFeedback]:
        """
        Get user feedback with optional filtering.
        
        Args:
            user_id: Filter by user ID
            item_id: Filter by item ID
            auction_id: Filter by auction ID
            feedback_type: Filter by feedback type
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of UserFeedback objects
        """
        if self.db_type == 'sqlite':
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM user_feedback WHERE 1=1'
            params = []
            
            if user_id:
                query += ' AND user_id = ?'
                params.append(user_id)
            
            if item_id:
                query += ' AND item_id = ?'
                params.append(item_id)
            
            if auction_id:
                query += ' AND auction_id = ?'
                params.append(auction_id)
            
            if feedback_type:
                if isinstance(feedback_type, FeedbackType):
                    query += ' AND feedback_type = ?'
                    params.append(feedback_type.value)
                else:
                    query += ' AND feedback_type = ?'
                    params.append(feedback_type)
            
            query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            feedback_list = []
            for row in rows:
                feedback_list.append(UserFeedback(
                    id=row['id'],
                    user_id=row['user_id'],
                    item_id=row['item_id'],
                    auction_id=row['auction_id'],
                    feedback_type=FeedbackType(row['feedback_type']),
                    sentiment=FeedbackSentiment(row['sentiment']),
                    rating=row['rating'],
                    comment=row['comment'],
                    ai_suggestion=row['ai_suggestion'],
                    user_correction=row['user_correction'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    created_at=datetime.fromisoformat(row['created_at'])
                ))
            
            conn.close()
            return feedback_list
    
    async def get_user_feedback_by_id(self, feedback_id: str) -> Optional[UserFeedback]:
        """
        Get user feedback by ID.
        
        Args:
            feedback_id: Feedback ID
            
        Returns:
            UserFeedback object or None if not found
        """
        if self.db_type == 'sqlite':
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM user_feedback WHERE id = ?', (feedback_id,))
            row = cursor.fetchone()
            
            if row:
                feedback = UserFeedback(
                    id=row['id'],
                    user_id=row['user_id'],
                    item_id=row['item_id'],
                    auction_id=row['auction_id'],
                    feedback_type=FeedbackType(row['feedback_type']),
                    sentiment=FeedbackSentiment(row['sentiment']),
                    rating=row['rating'],
                    comment=row['comment'],
                    ai_suggestion=row['ai_suggestion'],
                    user_correction=row['user_correction'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    created_at=datetime.fromisoformat(row['created_at'])
                )
                
                conn.close()
                return feedback
            
            conn.close()
            return None
    
    async def save_user_feedback(self, feedback: UserFeedback) -> None:
        """
        Save user feedback.
        
        Args:
            feedback: UserFeedback object
        """
        if self.db_type == 'sqlite':
            conn