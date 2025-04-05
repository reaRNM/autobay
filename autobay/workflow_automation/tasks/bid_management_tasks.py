"""
Bid management tasks for the workflow automation module.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from workflow_automation.db import Database
from workflow_automation.models import BidRecommendation
from workflow_automation.utils import setup_logging

# Import from existing modules
from bid_intelligence import BidIntelligence
from fraud_detection import FraudDetector


logger = setup_logging()


async def generate_bid_recommendations(db: Database, execution_id: str) -> Dict[str, Any]:
    """
    Generate bid recommendations for high-score items.
    
    Args:
        db: Database instance
        execution_id: Workflow execution ID
    
    Returns:
        Dictionary with recommendation results
    """
    logger.info("Starting bid recommendation generation task")
    
    try:
        # Get high-score items
        high_score_items = await db.get_recent_auction_items(
            hours=24,
            status="high_score"
        )
        
        logger.info(f"Found {len(high_score_items)} high-score items for bid recommendations")
        
        # Initialize bid intelligence
        bid_intelligence = BidIntelligence()
        
        # Initialize fraud detector
        fraud_detector = FraudDetector()
        
        # Track recommendation results
        recommendations = 0
        flagged_for_review = 0
        
        # Process each item
        for item in high_score_items:
            # Get profit calculation
            profit_calc = await db.get_profit_calculation(item.id)
            
            if not profit_calc:
                logger.warning(f"No profit calculation found for item {item.id}")
                continue
            
            # Check for fraud
            fraud_check = await fraud_detector.check_item(item)
            
            # Generate bid recommendation
            bid_data = await bid_intelligence.generate_bid_recommendation(
                item,
                profit_calc,
                fraud_check
            )
            
            # Create bid recommendation record
            recommendation = BidRecommendation(
                item_id=item.id,
                auction_id=item.id.split("_")[1],  # Extract auction ID from item ID
                recommended_bid=bid_data["recommended_bid"],
                max_bid=bid_data["max_bid"],
                confidence_score=bid_data["confidence_score"],
                profit_potential=bid_data["profit_potential"],
                roi_potential=bid_data["roi_potential"],
                risk_score=bid_data["risk_score"],
                time_sensitivity=bid_data["time_sensitivity"],
                requires_review=bid_data["requires_review"],
                review_reason=bid_data.get("review_reason"),
                bid_time=bid_data["bid_time"],
                metadata={
                    "generation_time": datetime.now().isoformat(),
                    "bid_strategy": bid_data.get("bid_strategy", {}),
                    "market_factors": bid_data.get("market_factors", {})
                }
            )
            
            # Save recommendation
            await db.save_bid_recommendation(recommendation)
            recommendations += 1
            
            # Track items flagged for review
            if recommendation.requires_review:
                flagged_for_review += 1
                await db.update_auction_item_status(item.id, "review_required")
            else:
                await db.update_auction_item_status(item.id, "bid_recommended")
        
        return {
            "total_items": len(high_score_items),
            "recommendations": recommendations,
            "flagged_for_review": flagged_for_review,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating bid recommendations: {str(e)}")
        raise


async def schedule_bids(db: Database, execution_id: str) -> Dict[str, Any]:
    """
    Schedule bids for recommended items.
    
    Args:
        db: Database instance
        execution_id: Workflow execution ID
    
    Returns:
        Dictionary with scheduling results
    """
    logger.info("Starting bid scheduling task")
    
    try:
        # Get items with bid recommendations
        recommended_items = await db.get_recent_auction_items(
            hours=24,
            status="bid_recommended"
        )
        
        logger.info(f"Found {len(recommended_items)} items with bid recommendations for scheduling")
        
        # Track scheduling results
        scheduled_bids = 0
        urgent_bids = 0
        
        # Current time
        now = datetime.now()
        
        # Process each item
        for item in recommended_items:
            # Get bid recommendation
            recommendation = await db.get_bid_recommendation(item.id)
            
            if not recommendation:
                logger.warning(f"No bid recommendation found for item {item.id}")
                continue
            
            # Determine if bid is urgent (ending within 6 hours)
            is_urgent = item.end_time - now < timedelta(hours=6)
            
            # Schedule bid
            if is_urgent:
                # Schedule for immediate execution
                await db.schedule_bid(
                    item_id=item.id,
                    bid_amount=recommendation.recommended_bid,
                    max_bid=recommendation.max_bid,
                    scheduled_time=now + timedelta(minutes=5),  # Schedule in 5 minutes
                    is_urgent=True
                )
                urgent_bids += 1
                await db.update_auction_item_status(item.id, "urgent_bid")
            else:
                # Schedule based on recommendation
                await db.schedule_bid(
                    item_id=item.id,
                    bid_amount=recommendation.recommended_bid,
                    max_bid=recommendation.max_bid,
                    scheduled_time=recommendation.bid_time,
                    is_urgent=False
                )
                scheduled_bids += 1
                await db.update_auction_item_status(item.id, "bid_scheduled")
        
        return {
            "total_items": len(recommended_items),
            "scheduled_bids": scheduled_bids,
            "urgent_bids": urgent_bids,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error scheduling bids: {str(e)}")
        raise


def get_tasks():
    """Get all bid management tasks."""
    return {
        "generate_bid_recommendations": generate_bid_recommendations,
        "schedule_bids": schedule_bids
    }