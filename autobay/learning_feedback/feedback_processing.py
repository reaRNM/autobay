"""
Feedback processing module for the Learning & Feedback Systems Module.

This module provides functionality for processing user feedback
and tracking user preferences.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from learning_feedback.models import (
    UserFeedback, UserPreference, FeedbackType, FeedbackSentiment
)
from learning_feedback.db import Database


logger = logging.getLogger(__name__)


class FeedbackProcessor:
    """
    Processor for user feedback.
    
    This class provides methods for processing and analyzing
    user feedback.
    """
    
    def __init__(self, db: Database):
        """
        Initialize the FeedbackProcessor.
        
        Args:
            db: Database connection
        """
        self.db = db
        logger.info("FeedbackProcessor initialized")
    
    async def process_feedback(self, feedback: UserFeedback) -> None:
        """
        Process user feedback.
        
        Args:
            feedback: User feedback
        """
        logger.debug(f"Processing feedback {feedback.id}")
        
        try:
            # Analyze feedback sentiment if not provided
            if not feedback.sentiment:
                feedback.sentiment = await self._analyze_sentiment(feedback)
                
                # Save updated feedback
                await self.db.save_user_feedback(feedback)
            
            # Process feedback based on type
            if feedback.feedback_type == FeedbackType.PRICING:
                await self._process_pricing_feedback(feedback)
            elif feedback.feedback_type == FeedbackType.LISTING:
                await self._process_listing_feedback(feedback)
            elif feedback.feedback_type == FeedbackType.BIDDING:
                await self._process_bidding_feedback(feedback)
            elif feedback.feedback_type == FeedbackType.SHIPPING:
                await self._process_shipping_feedback(feedback)
            else:
                await self._process_general_feedback(feedback)
        
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
    
    async def get_feedback_summary(
        self,
        user_id: Optional[str] = None,
        feedback_type: Optional[FeedbackType] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get a summary of user feedback.
        
        Args:
            user_id: Filter by user ID (optional)
            feedback_type: Filter by feedback type (optional)
            limit: Maximum number of feedback items to analyze
            
        Returns:
            Feedback summary
        """
        logger.debug(f"Getting feedback summary (user_id={user_id}, feedback_type={feedback_type})")
        
        try:
            # Get feedback
            feedback = await self.db.get_user_feedback(
                user_id=user_id,
                feedback_type=feedback_type,
                limit=limit
            )
            
            if not feedback:
                return {
                    'count': 0,
                    'average_rating': None,
                    'sentiment_distribution': {},
                    'common_issues': []
                }
            
            # Calculate average rating
            ratings = [f.rating for f in feedback if f.rating is not None]
            avg_rating = sum(ratings) / len(ratings) if ratings else None
            
            # Calculate sentiment distribution
            sentiment_dist = {}
            for f in feedback:
                sentiment = f.sentiment.value
                sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1
            
            # Convert to percentages
            for sentiment, count in sentiment_dist.items():
                sentiment_dist[sentiment] = count / len(feedback) * 100
            
            # Identify common issues (simple keyword analysis)
            issues = {}
            for f in feedback:
                if f.comment and f.sentiment == FeedbackSentiment.NEGATIVE:
                    comment = f.comment.lower()
                    
                    # Check for common issue keywords
                    keywords = {
                        'price': ['price', 'expensive', 'cost', 'cheap', 'overpriced', 'underpriced'],
                        'shipping': ['shipping', 'delivery', 'package', 'late', 'delay'],
                        'quality': ['quality', 'condition', 'damaged', 'broken', 'poor'],
                        'accuracy': ['accuracy', 'incorrect', 'wrong', 'inaccurate', 'error'],
                        'usability': ['usability', 'difficult', 'confusing', 'complex', 'hard to use']
                    }
                    
                    for issue, terms in keywords.items():
                        if any(term in comment for term in terms):
                            issues[issue] = issues.get(issue, 0) + 1
            
            # Sort issues by frequency
            common_issues = sorted(
                [{'issue': issue, 'count': count} for issue, count in issues.items()],
                key=lambda x: x['count'],
                reverse=True
            )
            
            return {
                'count': len(feedback),
                'average_rating': avg_rating,
                'sentiment_distribution': sentiment_dist,
                'common_issues': common_issues
            }
        
        except Exception as e:
            logger.error(f"Error getting feedback summary: {e}")
            raise
    
    async def _analyze_sentiment(self, feedback: UserFeedback) -> FeedbackSentiment:
        """
        Analyze sentiment of feedback.
        
        Args:
            feedback: User feedback
            
        Returns:
            Sentiment
        """
        # If rating is provided, use it to determine sentiment
        if feedback.rating is not None:
            if feedback.rating >= 4:
                return FeedbackSentiment.POSITIVE
            elif feedback.rating <= 2:
                return FeedbackSentiment.NEGATIVE
            else:
                return FeedbackSentiment.NEUTRAL
        
        # If comment is provided, analyze it
        if feedback.comment:
            # Simple keyword-based sentiment analysis
            comment = feedback.comment.lower()
            
            # Positive keywords
            positive_keywords = [
                'good', 'great', 'excellent', 'amazing', 'awesome', 'love',
                'perfect', 'best', 'helpful', 'easy', 'recommend', 'thanks',
                'thank you', 'fantastic', 'wonderful', 'happy', 'pleased'
            ]
            
            # Negative keywords
            negative_keywords = [
                'bad', 'poor', 'terrible', 'awful', 'horrible', 'hate',
                'difficult', 'confusing', 'frustrating', 'disappointed',
                'waste', 'useless', 'expensive', 'overpriced', 'error',
                'wrong', 'incorrect', 'not working', 'broken', 'issue'
            ]
            
            # Count positive and negative keywords
            positive_count = sum(1 for keyword in positive_keywords if keyword in comment)
            negative_count = sum(1 for keyword in negative_keywords if keyword in comment)
            
            # Determine sentiment
            if positive_count > negative_count:
                return FeedbackSentiment.POSITIVE
            elif negative_count > positive_count:
                return FeedbackSentiment.NEGATIVE
            else:
                return FeedbackSentiment.NEUTRAL
        
        # Default to neutral
        return FeedbackSentiment.NEUTRAL
    
    async def _process_pricing_feedback(self, feedback: UserFeedback) -> None:
        """
        Process pricing feedback.
        
        Args:
            feedback: User feedback
        """
        logger.debug(f"Processing pricing feedback {feedback.id}")
        
        # In a real implementation, this would update pricing models
        # For now, just log the feedback
        if feedback.ai_suggestion and feedback.user_correction:
            logger.info(f"Pricing feedback: AI suggested {feedback.ai_suggestion}, user corrected to {feedback.user_correction}")
    
    async def _process_listing_feedback(self, feedback: UserFeedback) -> None:
        """
        Process listing feedback.
        
        Args:
            feedback: User feedback
        """
        logger.debug(f"Processing listing feedback {feedback.id}")
        
        # In a real implementation, this would update listing models
        # For now, just log the feedback
        if feedback.comment:
            logger.info(f"Listing feedback: {feedback.comment}")
    
    async def _process_bidding_feedback(self, feedback: UserFeedback) -> None:
        """
        Process bidding feedback.
        
        Args:
            feedback: User feedback
        """
        logger.debug(f"Processing bidding feedback {feedback.id}")
        
        # In a real implementation, this would update bidding models
        # For now, just log the feedback
        if feedback.ai_suggestion and feedback.user_correction:
            logger.info(f"Bidding feedback: AI suggested {feedback.ai_suggestion}, user corrected to {feedback.user_correction}")
    
    async def _process_shipping_feedback(self, feedback: UserFeedback) -> None:
        """
        Process shipping feedback.
        
        Args:
            feedback: User feedback
        """
        logger.debug(f"Processing shipping feedback {feedback.id}")
        
        # In a real implementation, this would update shipping models
        # For now, just log the feedback
        if feedback.ai_suggestion and feedback.user_correction:
            logger.info(f"Shipping feedback: AI suggested {feedback.ai_suggestion}, user corrected to {feedback.user_correction}")
    
    async def _process_general_feedback(self, feedback: UserFeedback) -> None:
        """
        Process general feedback.
        
        Args:
            feedback: User feedback
        """
        logger.debug(f"Processing general feedback {feedback.id}")
        
        # In a real implementation, this would update general models
        # For now, just log the feedback
        if feedback.comment:
            logger.info(f"General feedback: {feedback.comment}")


class UserPreferenceTracker:
    """
    Tracker for user preferences.
    
    This class provides methods for tracking and updating
    user preferences based on feedback and behavior.
    """
    
    def __init__(self, db: Database):
        """
        Initialize the UserPreferenceTracker.
        
        Args:
            db: Database connection
        """
        self.db = db
        logger.info("UserPreferenceTracker initialized")
    
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreference]:
        """
        Get preferences for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            UserPreference object or None if not found
        """
        logger.debug(f"Getting preferences for user {user_id}")
        
        try:
            # Get user preferences from database
            preferences = await self.db.get_user_preference(user_id)
            
            return preferences
        
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return None
    
    async def update_preferences(self, feedback: UserFeedback) -> None:
        """
        Update user preferences based on feedback.
        
        Args:
            feedback: User feedback
        """
        logger.debug(f"Updating preferences for user {feedback.user_id}")
        
        try:
            # Get existing preferences
            preferences = await self.get_user_preferences(feedback.user_id)
            
            # Create new preferences if not found
            if not preferences:
                preferences = UserPreference(
                    id=str(uuid.uuid4()),
                    user_id=feedback.user_id
                )
            
            # Update preferences based on feedback type
            if feedback.feedback_type == FeedbackType.PRICING:
                await self._update_pricing_preferences(preferences, feedback)
            elif feedback.feedback_type == FeedbackType.LISTING:
                await self._update_listing_preferences(preferences, feedback)
            elif feedback.feedback_type == FeedbackType.BIDDING:
                await self._update_bidding_preferences(preferences, feedback)
            elif feedback.feedback_type == FeedbackType.SHIPPING:
                await self._update_shipping_preferences(preferences, feedback)
            
            # Update timestamp
            preferences.updated_at = datetime.now()
            
            # Save updated preferences
            await self.db.save_user_preference(preferences)
        
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
    
    async def update_preferences_from_auction(self, auction: Dict[str, Any]) -> None:
        """
        Update user preferences based on auction outcome.
        
        Args:
            auction: Auction outcome
        """
        logger.debug(f"Updating preferences from auction for user {auction.get('user_id')}")
        
        try:
            # Get user ID
            user_id = auction.get('user_id')
            if not user_id:
                return
            
            # Get existing preferences
            preferences = await self.get_user_preferences(user_id)
            
            # Create new preferences if not found
            if not preferences:
                preferences = UserPreference(
                    id=str(uuid.uuid4()),
                    user_id=user_id
                )
            
            # Update category preferences
            category_id = auction.get('category_id')
            if category_id:
                # Increase preference weight for successful auctions
                if auction.get('status') == 'sold' and auction.get('profit', 0) > 0:
                    preferences.category_preferences[category_id] = preferences.category_preferences.get(category_id, 0) + 0.1
                # Decrease preference weight for unsuccessful auctions
                elif auction.get('status') == 'unsold':
                    preferences.category_preferences[category_id] = max(0, preferences.category_preferences.get(category_id, 0) - 0.05)
            
            # Update platform preferences
            platform = auction.get('platform')
            if platform:
                # Increase preference weight for successful auctions
                if auction.get('status') == 'sold' and auction.get('profit', 0) > 0:
                    preferences.platform_preferences[platform] = preferences.platform_preferences.get(platform, 0) + 0.1
                # Decrease preference weight for unsuccessful auctions
                elif auction.get('status') == 'unsold':
                    preferences.platform_preferences[platform] = max(0, preferences.platform_preferences.get(platform, 0) - 0.05)
            
            # Update price range
            if auction.get('final_price'):
                if preferences.price_range_min is None or auction['final_price'] < preferences.price_range_min:
                    preferences.price_range_min = auction['final_price']
                
                if preferences.price_range_max is None or auction['final_price'] > preferences.price_range_max:
                    preferences.price_range_max = auction['final_price']
            
            # Update timestamp
            preferences.updated_at = datetime.now()
            
            # Save updated preferences
            await self.db.save_user_preference(preferences)
        
        except Exception as e:
            logger.error(f"Error updating user preferences from auction: {e}")
    
    async def _update_pricing_preferences(self, preferences: UserPreference, feedback: UserFeedback) -> None:
        """
        Update pricing preferences.
        
        Args:
            preferences: User preferences
            feedback: User feedback
        """
        # Update risk tolerance based on pricing feedback
        if feedback.ai_suggestion and feedback.user_correction:
            try:
                ai_price = float(feedback.ai_suggestion)
                user_price = float(feedback.user_correction)
                
                # If user consistently prices higher than AI, they're more risk-tolerant
                if user_price > ai_price:
                    preferences.risk_tolerance = min(1.0, preferences.risk_tolerance + 0.05)
                # If user consistently prices lower than AI, they're more risk-averse
                elif user_price < ai_price:
                    preferences.risk_tolerance = max(0.0, preferences.risk_tolerance - 0.05)
            except ValueError:
                # Not numeric values, skip
                pass
    
    async def _update_listing_preferences(self, preferences: UserPreference, feedback: UserFeedback) -> None:
        """
        Update listing preferences.
        
        Args:
            preferences: User preferences
            feedback: User feedback
        """
        # No specific updates for now
        pass
    
    async def _update_bidding_preferences(self, preferences: UserPreference, feedback: UserFeedback) -> None:
        """
        Update bidding preferences.
        
        Args:
            preferences: User preferences
            feedback: User feedback
        """
        # Update risk tolerance based on bidding feedback
        if feedback.ai_suggestion and feedback.user_correction:
            # Simple heuristic: if user consistently bids higher, they're more risk-tolerant
            if "higher" in feedback.user_correction.lower():
                preferences.risk_tolerance = min(1.0, preferences.risk_tolerance + 0.05)
            # If user consistently bids lower, they're more risk-averse
            elif "lower" in feedback.user_correction.lower():
                preferences.risk_tolerance = max(0.0, preferences.risk_tolerance - 0.05)
    
    async def _update_shipping_preferences(self, preferences: UserPreference, feedback: UserFeedback) -> None:
        """
        Update shipping preferences.
        
        Args:
            preferences: User preferences
            feedback: User feedback
        """
        # Update shipping preferences
        if feedback.ai_suggestion and feedback.user_correction:
            # Extract carrier from correction
            carriers = ["usps", "fedex", "ups", "dhl"]
            for carrier in carriers:
                if carrier.upper() in feedback.user_correction:
                    if carrier.upper() not in preferences.preferred_carriers:
                        preferences.preferred_carriers.append(carrier.upper())
            
            # Extract shipping method from correction
            methods = ["priority", "first class", "ground", "express", "overnight"]
            for method in methods:
                if method.lower() in feedback.user_correction.lower():
                    if method not in preferences.preferred_shipping_methods:
                        preferences.preferred_shipping_methods.append(method)