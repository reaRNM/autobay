"""
Notification tasks for the workflow automation module.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from workflow_automation.db import Database
from workflow_automation.models import (
    Notification,
    NotificationType,
    NotificationChannel,
    DailySummary
)
from workflow_automation.utils import setup_logging

# Import from existing modules
from notification_service import NotificationService


logger = setup_logging()


async def send_daily_summary(db: Database, execution_id: str) -> Dict[str, Any]:
    """
    Send daily summary of auction opportunities and bid outcomes.
    
    Args:
        db: Database instance
        execution_id: Workflow execution ID
    
    Returns:
        Dictionary with notification results
    """
    logger.info("Starting daily summary notification task")
    
    try:
        # Get today's date (start of day)
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get workflow execution data
        workflow_data = await db.get_workflow_executions_by_date(today)
        
        # Get auction data
        auction_data = await db.get_auction_stats_by_date(today)
        
        # Get bid data
        bid_data = await db.get_bid_stats_by_date(today)
        
        # Get top opportunities
        top_opportunities = await db.get_bid_recommendations(
            limit=10,
            min_confidence=0.7,
            min_profit=20.0,
            require_review=False
        )
        
        # Create daily summary
        summary = DailySummary(
            date=today,
            total_items_scraped=auction_data.get("total_scraped", 0),
            new_items_found=auction_data.get("new_items", 0),
            items_processed=auction_data.get("processed_items", 0),
            bid_recommendations_generated=bid_data.get("recommendations_generated", 0),
            bids_placed=bid_data.get("bids_placed", 0),
            successful_bids=bid_data.get("successful_bids", 0),
            total_potential_profit=bid_data.get("total_potential_profit", 0.0),
            top_opportunities=top_opportunities,
            errors_encountered=workflow_data.get("errors", 0),
            metadata={
                "generation_time": datetime.now().isoformat(),
                "workflow_executions": workflow_data.get("executions", []),
                "platform_breakdown": auction_data.get("platform_breakdown", {})
            }
        )
        
        # Save summary
        await db.save_daily_summary(summary)
        
        # Initialize notification service
        notification_service = NotificationService()
        
        # Get users to notify
        users = await db.get_users_with_notification_preferences(
            notification_type=NotificationType.DAILY_SUMMARY
        )
        
        # Track notification results
        notifications_sent = 0
        
        # Send notifications
        for user in users:
            # Create notification content
            notification_content = {
                "title": f"Daily Auction Summary - {today.strftime('%Y-%m-%d')}",
                "message": f"Today's summary: {summary.new_items_found} new items, {summary.bid_recommendations_generated} bid recommendations, ${summary.total_potential_profit:.2f} potential profit.",
                "data": summary.dict()
            }
            
            # Create notification record
            notification = Notification(
                id=f"summary_{today.strftime('%Y%m%d')}_{user['id']}",
                user_id=user["id"],
                type=NotificationType.DAILY_SUMMARY,
                title=notification_content["title"],
                message=notification_content["message"],
                channel=user["preferred_channel"],
                priority=3,  # Medium priority
                data=notification_content["data"]
            )
            
            # Save notification
            await db.save_notification(notification)
            
            # Send notification
            await notification_service.send_notification(
                user_id=user["id"],
                notification=notification,
                channel=user["preferred_channel"]
            )
            
            notifications_sent += 1
        
        return {
            "summary_date": today.isoformat(),
            "users_notified": notifications_sent,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error sending daily summary: {str(e)}")
        raise


async def send_urgent_notifications(db: Database, execution_id: str) -> Dict[str, Any]:
    """
    Send urgent notifications for high-priority bids or items requiring review.
    
    Args:
        db: Database instance
        execution_id: Workflow execution ID
    
    Returns:
        Dictionary with notification results
    """
    logger.info("Starting urgent notifications task")
    
    try:
        # Get urgent items
        urgent_items = await db.get_recent_auction_items(
            hours=24,
            status="urgent_bid"
        )
        
        # Get items requiring review
        review_items = await db.get_recent_auction_items(
            hours=24,
            status="review_required"
        )
        
        logger.info(f"Found {len(urgent_items)} urgent items and {len(review_items)} items requiring review")
        
        # Initialize notification service
        notification_service = NotificationService()
        
        # Track notification results
        urgent_notifications = 0
        review_notifications = 0
        
        # Send urgent bid notifications
        for item in urgent_items:
            # Get bid recommendation
            recommendation = await db.get_bid_recommendation(item.id)
            
            if not recommendation:
                logger.warning(f"No bid recommendation found for urgent item {item.id}")
                continue
            
            # Get users to notify
            users = await db.get_users_with_notification_preferences(
                notification_type=NotificationType.HIGH_PRIORITY_BID
            )
            
            for user in users:
                # Create notification content
                notification_content = {
                    "title": "Urgent Bid Opportunity",
                    "message": f"Urgent bid opportunity: {item.title} ending soon. Recommended bid: ${recommendation.recommended_bid:.2f}, potential profit: ${recommendation.profit_potential:.2f}",
                    "data": {
                        "item": item.dict(),
                        "recommendation": recommendation.dict()
                    }
                }
                
                # Create notification record
                notification = Notification(
                    id=f"urgent_{item.id}_{user['id']}",
                    user_id=user["id"],
                    type=NotificationType.HIGH_PRIORITY_BID,
                    title=notification_content["title"],
                    message=notification_content["message"],
                    channel=user["preferred_channel"],
                    priority=5,  # Highest priority
                    data=notification_content["data"]
                )
                
                # Save notification
                await db.save_notification(notification)
                
                # Send notification
                await notification_service.send_notification(
                    user_id=user["id"],
                    notification=notification,
                    channel=user["preferred_channel"]
                )
                
                urgent_notifications += 1
        
        # Send review required notifications
        for item in review_items:
            # Get bid recommendation
            recommendation = await db.get_bid_recommendation(item.id)
            
            if not recommendation:
                logger.warning(f"No bid recommendation found for review item {item.id}")
                continue
            
            # Get users to notify
            users = await db.get_users_with_notification_preferences(
                notification_type=NotificationType.MANUAL_REVIEW_REQUIRED
            )
            
            for user in users:
                # Create notification content
                notification_content = {
                    "title": "Manual Review Required",
                    "message": f"Item requires manual review: {item.title}. Reason: {recommendation.review_reason}",
                    "data": {
                        "item": item.dict(),
                        "recommendation": recommendation.dict()
                    }
                }
                
                # Create notification record
                notification = Notification(
                    id=f"review_{item.id}_{user['id']}",
                    user_id=user["id"],
                    type=NotificationType.MANUAL_REVIEW_REQUIRED,
                    title=notification_content["title"],
                    message=notification_content["message"],
                    channel=user["preferred_channel"],
                    priority=4,  # High priority
                    data=notification_content["data"]
                )
                
                # Save notification
                await db.save_notification(notification)
                
                # Send notification
                await notification_service.send_notification(
                    user_id=user["id"],
                    notification=notification,
                    channel=user["preferred_channel"]
                )
                
                review_notifications += 1
        
        return {
            "urgent_notifications": urgent_notifications,
            "review_notifications": review_notifications,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error sending urgent notifications: {str(e)}")
        raise


def get_tasks():
    """Get all notification tasks."""
    return {
        "send_daily_summary": send_daily_summary,
        "send_urgent_notifications": send_urgent_notifications
    }