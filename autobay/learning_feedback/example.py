"""
Example usage of the Learning & Feedback Systems Module.

This script demonstrates how to use the Learning & Feedback Systems Module
to collect feedback, train models, and monitor performance.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
import uuid

from learning_feedback import (
    LearningSystem,
    AuctionOutcome,
    UserFeedback,
    AuctionStatus,
    FeedbackType,
    FeedbackSentiment
)
from learning_feedback.utils import setup_logging


# Set up logging
logger = setup_logging(log_level="INFO")


async def main():
    """Main function."""
    logger.info("Starting Learning & Feedback Systems example")
    
    # Initialize learning system
    learning_system = LearningSystem()
    
    # Example 1: Update auction outcome
    logger.info("Example 1: Update auction outcome")
    
    # Create sample auction outcome
    auction_data = {
        'id': str(uuid.uuid4()),
        'item_id': 'item-001',
        'user_id': 'user-001',
        'platform': 'ebay',
        'category_id': 'electronics',
        'listing_id': 'listing-001',
        'estimated_price': 100.0,
        'start_price': 50.0,
        'final_price': 95.0,
        'shipping_cost': 10.0,
        'fees': 5.0,
        'profit': 30.0,
        'roi': 0.3,
        'views': 100,
        'watchers': 10,
        'questions': 2,
        'bids': 5,
        'status': AuctionStatus.SOLD,
        'time_to_sale': 48,  # hours
        'listing_quality_score': 0.8,
        'ai_confidence_score': 0.9,
        'listing_date': datetime.now() - timedelta(days=2),
        'end_date': datetime.now() - timedelta(days=1)
    }
    
    # Update auction outcome
    auction = await learning_system.update_auction_outcome(auction_data)
    
    logger.info(f"Updated auction outcome: {auction.id}")
    logger.info(f"Price accuracy: {auction.price_accuracy:.2f}%")
    
    # Example 2: Track user feedback
    logger.info("\nExample 2: Track user feedback")
    
    # Create sample user feedback
    feedback_data = {
        'id': str(uuid.uuid4()),
        'user_id': 'user-001',
        'item_id': 'item-001',
        'auction_id': auction.id,
        'feedback_type': FeedbackType.PRICING,
        'sentiment': FeedbackSentiment.POSITIVE,
        'rating': 4,
        'comment': 'The price estimate was very accurate!',
        'ai_suggestion': '100.0',
        'user_correction': '95.0',
        'metadata': {
            'source': 'web'
        }
    }
    
    # Track user feedback
    feedback = await learning_system.track_user_feedback(feedback_data)
    
    logger.info(f"Tracked user feedback: {feedback.id}")
    
    # Example 3: Recommend adjustments
    logger.info("\nExample 3: Recommend adjustments")
    
    # Get adjustment recommendations
    recommendations = await learning_system.recommend_adjustments(
        user_id='user-001',
        category_id='electronics'
    )
    
    logger.info(f"Recommended adjustments: {json.dumps(recommendations, indent=2)}")
    
    # Example 4: Get model performance
    logger.info("\nExample 4: Get model performance")
    
    # Get pricing model performance
    pricing_performance = await learning_system.get_model_performance(
        model_type='pricing',
        category_id='electronics'
    )
    
    if pricing_performance:
        logger.info(f"Pricing model performance: {pricing_performance[0].accuracy:.4f}")
    else:
        logger.info("No pricing model performance data available")
    
    # Example 5: Get user preferences
    logger.info("\nExample 5: Get user preferences")
    
    # Get user preferences
    preferences = await learning_system.get_user_preferences('user-001')
    
    if preferences:
        logger.info(f"User preferences: {json.dumps(preferences, indent=2)}")
    else:
        logger.info("No user preferences available")
    
    # Example 6: Start API server
    logger.info("\nExample 6: Start API server")
    logger.info("To start the API server, run:")
    logger.info("from learning_feedback.api import create_app")
    logger.info("app = create_app()")
    logger.info("import uvicorn")
    logger.info("uvicorn.run(app, host='0.0.0.0', port=8000)")


if __name__ == "__main__":
    asyncio.run(main())