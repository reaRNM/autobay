"""
Data collection module for the Learning & Feedback Systems Module.

This module provides functionality for collecting auction outcomes
and user feedback.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from learning_feedback.models import (
    AuctionOutcome, UserFeedback, AuctionStatus,
    FeedbackType, FeedbackSentiment
)
from learning_feedback.db import Database


logger = logging.getLogger(__name__)


class AuctionOutcomeCollector:
    """
    Collector for auction outcomes.
    
    This class provides methods for collecting and validating
    auction outcome data.
    """
    
    def __init__(self, db: Database):
        """
        Initialize the AuctionOutcomeCollector.
        
        Args:
            db: Database connection
        """
        self.db = db
        logger.info("AuctionOutcomeCollector initialized")
    
    async def save_auction_outcome(self, data: Dict[str, Any]) -> AuctionOutcome:
        """
        Save an auction outcome.
        
        Args:
            data: Auction outcome data
            
        Returns:
            AuctionOutcome object
        """
        logger.debug(f"Saving auction outcome for item {data.get('item_id')}")
        
        try:
            # Validate required fields
            required_fields = ['item_id', 'user_id', 'platform', 'category_id']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Get existing auction outcome if available
            existing = None
            if 'id' in data:
                existing = await self.db.get_auction_outcome(data['id'])
            
            # Create or update auction outcome
            if existing:
                # Update existing auction outcome
                for key, value in data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                
                # Update timestamps
                existing.updated_at = datetime.now()
                
                # If status changed to SOLD or UNSOLD, set end_date
                if existing.status in [AuctionStatus.SOLD, AuctionStatus.UNSOLD] and not existing.end_date:
                    existing.end_date = datetime.now()
                
                # Calculate profit and ROI if not provided
                if existing.status == AuctionStatus.SOLD and existing.final_price is not None:
                    if 'profit' not in data and existing.final_price is not None:
                        # Calculate profit (final_price - fees - shipping_cost)
                        cost = 0
                        if hasattr(existing, 'cost') and existing.cost is not None:
                            cost = existing.cost
                        
                        fees = existing.fees or 0
                        shipping_cost = existing.shipping_cost or 0
                        
                        existing.profit = existing.final_price - cost - fees - shipping_cost
                    
                    if 'roi' not in data and existing.profit is not None and hasattr(existing, 'cost') and existing.cost is not None and existing.cost > 0:
                        # Calculate ROI (profit / cost)
                        existing.roi = existing.profit / existing.cost
                
                # Calculate time to sale if not provided
                if existing.status == AuctionStatus.SOLD and 'time_to_sale' not in data and existing.end_date and existing.listing_date:
                    # Calculate time to sale in hours
                    time_diff = existing.end_date - existing.listing_date
                    existing.time_to_sale = int(time_diff.total_seconds() / 3600)
                
                # Save updated auction outcome
                await self.db.save_auction_outcome(existing)
                
                logger.info(f"Updated auction outcome {existing.id}")
                return existing
            else:
                # Create new auction outcome
                auction_id = data.get('id', str(uuid.uuid4()))
                
                # Convert status string to enum if needed
                status = data.get('status', AuctionStatus.ACTIVE)
                if isinstance(status, str):
                    status = AuctionStatus(status)
                
                # Create auction outcome
                auction = AuctionOutcome(
                    id=auction_id,
                    item_id=data['item_id'],
                    user_id=data['user_id'],
                    platform=data['platform'],
                    category_id=data['category_id'],
                    listing_id=data.get('listing_id'),
                    estimated_price=data.get('estimated_price', 0.0),
                    start_price=data.get('start_price', 0.0),
                    reserve_price=data.get('reserve_price'),
                    final_price=data.get('final_price'),
                    shipping_cost=data.get('shipping_cost'),
                    fees=data.get('fees'),
                    profit=data.get('profit'),
                    roi=data.get('roi'),
                    views=data.get('views', 0),
                    watchers=data.get('watchers', 0),
                    questions=data.get('questions', 0),
                    bids=data.get('bids', 0),
                    status=status,
                    time_to_sale=data.get('time_to_sale'),
                    listing_quality_score=data.get('listing_quality_score'),
                    ai_confidence_score=data.get('ai_confidence_score'),
                    listing_date=data.get('listing_date', datetime.now()),
                    end_date=data.get('end_date')
                )
                
                # Calculate profit and ROI if not provided
                if auction.status == AuctionStatus.SOLD and auction.final_price is not None:
                    if 'profit' not in data and hasattr(data, 'cost') and data.cost is not None:
                        # Calculate profit (final_price - cost - fees - shipping_cost)
                        cost = data.cost
                        fees = auction.fees or 0
                        shipping_cost = auction.shipping_cost or 0
                        
                        auction.profit = auction.final_price - cost - fees - shipping_cost
                    
                    if 'roi' not in data and auction.profit is not None and hasattr(data, 'cost') and data.cost is not None and data.cost > 0:
                        # Calculate ROI (profit / cost)
                        auction.roi = auction.profit / data.cost
                
                # Calculate time to sale if not provided
                if auction.status == AuctionStatus.SOLD and 'time_to_sale' not in data and auction.end_date and auction.listing_date:
                    # Calculate time to sale in hours
                    time_diff = auction.end_date - auction.listing_date
                    auction.time_to_sale = int(time_diff.total_seconds() / 3600)
                
                # Save new auction outcome
                await self.db.save_auction_outcome(auction)
                
                logger.info(f"Created new auction outcome {auction.id}")
                return auction
        
        except Exception as e:
            logger.error(f"Error saving auction outcome: {e}")
            raise
    
    async def get_auction_outcomes(
        self,
        user_id: Optional[str] = None,
        item_id: Optional[str] = None,
        category_id: Optional[str] = None,
        status: Optional[AuctionStatus] = None,
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
        logger.debug(f"Getting auction outcomes (user_id={user_id}, item_id={item_id}, category_id={category_id}, status={status})")
        
        try:
            # Get auction outcomes from database
            auctions = await self.db.get_auction_outcomes(
                user_id=user_id,
                item_id=item_id,
                category_id=category_id,
                status=status,
                limit=limit,
                offset=offset
            )
            
            return auctions
        
        except Exception as e:
            logger.error(f"Error getting auction outcomes: {e}")
            raise


class UserFeedbackCollector:
    """
    Collector for user feedback.
    
    This class provides methods for collecting and validating
    user feedback data.
    """
    
    def __init__(self, db: Database):
        """
        Initialize the UserFeedbackCollector.
        
        Args:
            db: Database connection
        """
        self.db = db
        logger.info("UserFeedbackCollector initialized")
    
    async def save_user_feedback(self, data: Dict[str, Any]) -> UserFeedback:
        """
        Save user feedback.
        
        Args:
            data: User feedback data
            
        Returns:
            UserFeedback object
        """
        logger.debug(f"Saving user feedback from user {data.get('user_id')}")
        
        try:
            # Validate required fields
            required_fields = ['user_id']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Get existing feedback if available
            existing = None
            if 'id' in data:
                existing = await self.db.get_user_feedback(data['id'])
            
            # Create or update feedback
            if existing:
                # Update existing feedback
                for key, value in data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                
                # Save updated feedback
                await self.db.save_user_feedback(existing)
                
                logger.info(f"Updated user feedback {existing.id}")
                return existing
            else:
                # Create new feedback
                feedback_id = data.get('id', str(uuid.uuid4()))
                
                # Convert feedback_type and sentiment strings to enums if needed
                feedback_type = data.get('feedback_type', FeedbackType.GENERAL)
                if isinstance(feedback_type, str):
                    feedback_type = FeedbackType(feedback_type)
                
                sentiment = data.get('sentiment', FeedbackSentiment.NEUTRAL)
                if isinstance(sentiment, str):
                    sentiment = FeedbackSentiment(sentiment)
                
                # Create feedback
                feedback = UserFeedback(
                    id=feedback_id,
                    user_id=data['user_id'],
                    item_id=data.get('item_id'),
                    auction_id=data.get('auction_id'),
                    feedback_type=feedback_type,
                    sentiment=sentiment,
                    rating=data.get('rating'),
                    comment=data.get('comment'),
                    ai_suggestion=data.get('ai_suggestion'),
                    user_correction=data.get('user_correction'),
                    metadata=data.get('metadata', {})
                )
                
                # Save new feedback
                await self.db.save_user_feedback(feedback)
                
                logger.info(f"Created new user feedback {feedback.id}")
                return feedback
        
        except Exception as e:
            logger.error(f"Error saving user feedback: {e}")
            raise
    
    async def get_user_feedback(
        self,
        user_id: Optional[str] = None,
        item_id: Optional[str] = None,
        auction_id: Optional[str] = None,
        feedback_type: Optional[FeedbackType] = None,
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
        logger.debug(f"Getting user feedback (user_id={user_id}, item_id={item_id}, auction_id={auction_id}, feedback_type={feedback_type})")
        
        try:
            # Get user feedback from database
            feedback = await self.db.get_user_feedback(
                user_id=user_id,
                item_id=item_id,
                auction_id=auction_id,
                feedback_type=feedback_type,
                limit=limit,
                offset=offset
            )
            
            return feedback
        
        except Exception as e:
            logger.error(f"Error getting user feedback: {e}")
            raise