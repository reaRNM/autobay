"""
Database module for the workflow automation system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING

from workflow_automation.models import (
    WorkflowExecution,
    TaskResult,
    WorkflowLog,
    AuctionItem,
    ProfitCalculation,
    BidRecommendation,
    Notification,
    DailySummary
)
from workflow_automation.utils import setup_logging


logger = setup_logging()


class Database:
    """
    Database interface for the workflow automation system.
    
    This class provides methods for interacting with the database,
    including saving and retrieving workflow executions, tasks,
    logs, auction items, and other data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the database connection.
        
        Args:
            config: Database configuration
        """
        self.config = config or {}
        self.client = None
        self.db = None
        
        # Connect to database
        self._connect()
    
    def _connect(self):
        """Connect to the database."""
        try:
            # Get connection string
            connection_string = self.config.get(
                "connection_string",
                "mongodb://localhost:27017"
            )
            
            # Get database name
            db_name = self.config.get("db_name", "auction_automation")
            
            # Connect to MongoDB
            self.client = AsyncIOMotorClient(connection_string)
            self.db = self.client[db_name]
            
            logger.info(f"Connected to database {db_name}")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    async def save_workflow_execution(self, execution: WorkflowExecution) -> str:
        """
        Save a workflow execution to the database.
        
        Args:
            execution: Workflow execution to save
        
        Returns:
            Execution ID
        """
        try:
            # Convert to dict
            execution_dict = execution.dict()
            
            # Save to database
            await self.db.workflow_executions.update_one(
                {"id": execution.id},
                {"$set": execution_dict},
                upsert=True
            )
            
            return execution.id
        except Exception as e:
            logger.error(f"Error saving workflow execution: {str(e)}")
            raise
    
    async def get_workflow_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """
        Get a workflow execution by ID.
        
        Args:
            execution_id: Workflow execution ID
        
        Returns:
            Workflow execution or None if not found
        """
        try:
            # Get from database
            execution_dict = await self.db.workflow_executions.find_one({"id": execution_id})
            
            if not execution_dict:
                return None
            
            # Convert to model
            return WorkflowExecution(**execution_dict)
        except Exception as e:
            logger.error(f"Error getting workflow execution: {str(e)}")
            raise
    
    async def save_task_result(self, task_result: TaskResult) -> str:
        """
        Save a task result to the database.
        
        Args:
            task_result: Task result to save
        
        Returns:
            Task result ID
        """
        try:
            # Convert to dict
            task_dict = task_result.dict()
            
            # Save to database
            await self.db.task_results.update_one(
                {"task_id": task_result.task_id},
                {"$set": task_dict},
                upsert=True
            )
            
            return task_result.task_id
        except Exception as e:
            logger.error(f"Error saving task result: {str(e)}")
            raise
    
    async def save_workflow_log(self, log: WorkflowLog) -> str:
        """
        Save a workflow log to the database.
        
        Args:
            log: Workflow log to save
        
        Returns:
            Log ID
        """
        try:
            # Convert to dict
            log_dict = log.dict()
            
            # Save to database
            await self.db.workflow_logs.update_one(
                {"id": log.id},
                {"$set": log_dict},
                upsert=True
            )
            
            return log.id
        except Exception as e:
            logger.error(f"Error saving workflow log: {str(e)}")
            raise
    
    async def get_workflow_logs(self, workflow_id: str) -> List[WorkflowLog]:
        """
        Get logs for a workflow execution.
        
        Args:
            workflow_id: Workflow execution ID
        
        Returns:
            List of workflow logs
        """
        try:
            # Get from database
            logs = await self.db.workflow_logs.find(
                {"workflow_id": workflow_id}
            ).sort("timestamp", ASCENDING).to_list(length=None)
            
            # Convert to models
            return [WorkflowLog(**log) for log in logs]
        except Exception as e:
            logger.error(f"Error getting workflow logs: {str(e)}")
            raise
    
    async def get_recent_workflow_executions(self, limit: int = 10) -> List[WorkflowExecution]:
        """
        Get recent workflow executions.
        
        Args:
            limit: Maximum number of executions to return
        
        Returns:
            List of workflow executions
        """
        try:
            # Get from database
            executions = await self.db.workflow_executions.find().sort(
                "start_time", DESCENDING
            ).limit(limit).to_list(length=limit)
            
            # Convert to models
            return [WorkflowExecution(**execution) for execution in executions]
        except Exception as e:
            logger.error(f"Error getting recent workflow executions: {str(e)}")
            raise
    
    async def get_workflow_executions_by_date(self, date: datetime) -> Dict[str, Any]:
        """
        Get workflow execution statistics for a specific date.
        
        Args:
            date: Date to get statistics for
        
        Returns:
            Dictionary with execution statistics
        """
        try:
            # Get start and end of day
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            
            # Get executions for the day
            executions = await self.db.workflow_executions.find({
                "start_time": {"$gte": start_of_day, "$lt": end_of_day}
            }).to_list(length=None)
            
            # Count executions by status
            status_counts = {}
            for execution in executions:
                status = execution["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count errors
            error_count = 0
            for execution in executions:
                if execution["status"] in ["FAILED", "PARTIALLY_COMPLETED"]:
                    error_count += 1
            
            return {
                "date": date.isoformat(),
                "total_executions": len(executions),
                "status_counts": status_counts,
                "errors": error_count,
                "executions": [execution["id"] for execution in executions]
            }
        except Exception as e:
            logger.error(f"Error getting workflow executions by date: {str(e)}")
            raise
    
    async def save_auction_item(self, item: AuctionItem) -> str:
        """
        Save an auction item to the database.
        
        Args:
            item: Auction item to save
        
        Returns:
            Item ID
        """
        try:
            # Convert to dict
            item_dict = item.dict()
            
            # Save to database
            await self.db.auction_items.update_one(
                {"id": item.id},
                {"$set": item_dict},
                upsert=True
            )
            
            return item.id
        except Exception as e:
            logger.error(f"Error saving auction item: {str(e)}")
            raise
    
    async def get_auction_item(self, item_id: str) -> Optional[AuctionItem]:
        """
        Get an auction item by ID.
        
        Args:
            item_id: Auction item ID
        
        Returns:
            Auction item or None if not found
        """
        try:
            # Get from database
            item_dict = await self.db.auction_items.find_one({"id": item_id})
            
            if not item_dict:
                return None
            
            # Convert to model
            return AuctionItem(**item_dict)
        except Exception as e:
            logger.error(f"Error getting auction item: {str(e)}")
            raise
    
    async def update_auction_item(self, item_id: str, **kwargs) -> bool:
        """
        Update an auction item.
        
        Args:
            item_id: Auction item ID
            **kwargs: Fields to update
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update in database
            result = await self.db.auction_items.update_one(
                {"id": item_id},
                {"$set": kwargs}
            )
            
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating auction item: {str(e)}")
            raise
    
    async def update_auction_item_status(self, item_id: str, status: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Update an auction item's status.
        
        Args:
            item_id: Auction item ID
            status: New status
            metadata: Additional metadata
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare update
            update = {"status": status}
            if metadata:
                update["metadata"] = metadata
            
            # Update in database
            result = await self.db.auction_items.update_one(
                {"id": item_id},
                {"$set": update}
            )
            
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating auction item status: {str(e)}")
            raise
    
    async def get_recent_auction_items(self, hours: int = 24, status: Optional[str] = None) -> List[AuctionItem]:
        """
        Get recent auction items.
        
        Args:
            hours: Number of hours to look back
            status: Filter by status
        
        Returns:
            List of auction items
        """
        try:
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Prepare query
            query = {"metadata.scrape_time": {"$gte": cutoff_time.isoformat()}}
            if status:
                query["status"] = status
            
            # Get from database
            items = await self.db.auction_items.find(query).to_list(length=None)
            
            # Convert to models
            return [AuctionItem(**item) for item in items]
        except Exception as e:
            logger.error(f"Error getting recent auction items: {str(e)}")
            raise
    
    async def check_duplicate_auction_item(self, item: AuctionItem) -> bool:
        """
        Check if an auction item is a duplicate.
        
        Args:
            item: Auction item to check
        
        Returns:
            True if duplicate, False otherwise
        """
        try:
            # Check for existing item with same title and platform
            existing = await self.db.auction_items.find_one({
                "title": item.title,
                "platform": item.platform,
                "id": {"$ne": item.id}  # Exclude the item itself
            })
            
            return existing is not None
        except Exception as e:
            logger.error(f"Error checking for duplicate auction item: {str(e)}")
            raise
    
    async def validate_auction_item(self, item: AuctionItem) -> Dict[str, Any]:
        """
        Validate an auction item.
        
        Args:
            item: Auction item to validate
        
        Returns:
            Validation result
        """
        try:
            errors = []
            
            # Check required fields
            if not item.title:
                errors.append("Missing title")
            
            if not item.url:
                errors.append("Missing URL")
            
            if not item.current_price:
                errors.append("Missing current price")
            
            if not item.end_time:
                errors.append("Missing end time")
            
            # Check end time is in the future
            if item.end_time < datetime.now():
                errors.append("End time is in the past")
            
            # Check price is positive
            if item.current_price <= 0:
                errors.append("Current price must be positive")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors
            }
        except Exception as e:
            logger.error(f"Error validating auction item: {str(e)}")
            raise
    
    async def get_auction_stats_by_date(self, date: datetime) -> Dict[str, Any]:
        """
        Get auction statistics for a specific date.
        
        Args:
            date: Date to get statistics for
        
        Returns:
            Dictionary with auction statistics
        """
        try:
            # Get start and end of day
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            
            # Get items scraped on the day
            items = await self.db.auction_items.find({
                "metadata.scrape_time": {
                    "$gte": start_of_day.isoformat(),
                    "$lt": end_of_day.isoformat()
                }
            }).to_list(length=None)
            
            # Count items by platform
            platform_counts = {}
            for item in items:
                platform = item["platform"]
                platform_counts[platform] = platform_counts.get(platform, 0) + 1
            
            # Count items by status
            status_counts = {}
            for item in items:
                status = item.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count new items (unique)
            new_items = status_counts.get("unique", 0)
            
            # Count processed items
            processed_items = sum(
                status_counts.get(status, 0)
                for status in ["validated", "profitable", "unprofitable", "high_score", "low_score"]
            )
            
            return {
                "date": date.isoformat(),
                "total_scraped": len(items),
                "new_items": new_items,
                "processed_items": processed_items,
                "platform_breakdown": platform_counts,
                "status_breakdown": status_counts
            }
        except Exception as e:
            logger.error(f"Error getting auction stats by date: {str(e)}")
            raise
    
    async def save_profit_calculation(self, calculation: ProfitCalculation) -> str:
        """
        Save a profit calculation to the database.
        
        Args:
            calculation: Profit calculation to save
        
        Returns:
            Item ID
        """
        try:
            # Convert to dict
            calc_dict = calculation.dict()
            
            # Save to database
            await self.db.profit_calculations.update_one(
                {"item_id": calculation.item_id},
                {"$set": calc_dict},
                upsert=True
            )
            
            return calculation.item_id
        except Exception as e:
            logger.error(f"Error saving profit calculation: {str(e)}")
            raise
    
    async def get_profit_calculation(self, item_id: str) -> Optional[ProfitCalculation]:
        """
        Get a profit calculation by item ID.
        
        Args:
            item_id: Item ID
        
        Returns:
            Profit calculation or None if not found
        """
        try:
            # Get from database
            calc_dict = await self.db.profit_calculations.find_one({"item_id": item_id})
            
            if not calc_dict:
                return None
            
            # Convert to model
            return ProfitCalculation(**calc_dict)
        except Exception as e:
            logger.error(f"Error getting profit calculation: {str(e)}")
            raise
    
    async def save_bid_recommendation(self, recommendation: BidRecommendation) -> str:
        """
        Save a bid recommendation to the database.
        
        Args:
            recommendation: Bid recommendation to save
        
        Returns:
            Item ID
        """
        try:
            # Convert to dict
            rec_dict = recommendation.dict()
            
            # Save to database
            await self.db.bid_recommendations.update_one(
                {"item_id": recommendation.item_id},
                {"$set": rec_dict},
                upsert=True
            )
            
            return recommendation.item_id
        except Exception as e:
            logger.error(f"Error saving bid recommendation: {str(e)}")
            raise
    
    async def get_bid_recommendation(self, item_id: str) -> Optional[BidRecommendation]:
        """
        Get a bid recommendation by item ID.
        
        Args:
            item_id: Item ID
        
        Returns:
            Bid recommendation or None if not found
        """
        try:
            # Get from database
            rec_dict = await self.db.bid_recommendations.find_one({"item_id": item_id})
            
            if not rec_dict:
                return None
            
            # Convert to model
            return BidRecommendation(**rec_dict)
        except Exception as e:
            logger.error(f"Error getting bid recommendation: {str(e)}")
            raise
    
    async def get_bid_recommendations(
        self,
        limit: int = 10,
        min_confidence: float = 0.0,
        min_profit: float = 0.0,
        require_review: Optional[bool] = None
    ) -> List[BidRecommendation]:
        """
        Get bid recommendations.
        
        Args:
            limit: Maximum number of recommendations to return
            min_confidence: Minimum confidence score
            min_profit: Minimum profit potential
            require_review: Filter by requires_review flag
        
        Returns:
            List of bid recommendations
        """
        try:
            # Prepare query
            query = {
                "confidence_score": {"$gte": min_confidence},
                "profit_potential": {"$gte": min_profit}
            }
            
            if require_review is not None:
                query["requires_review"] = require_review
            
            # Get from database
            recommendations = await self.db.bid_recommendations.find(query).sort(
                "profit_potential", DESCENDING
            ).limit(limit).to_list(length=limit)
            
            # Convert to models
            return [BidRecommendation(**rec) for rec in recommendations]
        except Exception as e:
            logger.error(f"Error getting bid recommendations: {str(e)}")
            raise
    
    async def schedule_bid(
        self,
        item_id: str,
        bid_amount: float,
        max_bid: float,
        scheduled_time: datetime,
        is_urgent: bool = False
    ) -> str:
        """
        Schedule a bid for an item.
        
        Args:
            item_id: Item ID
            bid_amount: Bid amount
            max_bid: Maximum bid amount
            scheduled_time: Scheduled time for the bid
            is_urgent: Whether the bid is urgent
        
        Returns:
            Bid ID
        """
        try:
            # Create bid record
            bid_id = f"bid_{item_id}_{scheduled_time.strftime('%Y%m%d%H%M%S')}"
            bid = {
                "id": bid_id,
                "item_id": item_id,
                "bid_amount": bid_amount,
                "max_bid": max_bid,
                "scheduled_time": scheduled_time,
                "is_urgent": is_urgent,
                "status": "scheduled",
                "created_at": datetime.now()
            }
            
            # Save to database
            await self.db.scheduled_bids.update_one(
                {"id": bid_id},
                {"$set": bid},
                upsert=True
            )
            
            return bid_id
        except Exception as e:
            logger.error(f"Error scheduling bid: {str(e)}")
            raise
    
    async def get_bid_stats_by_date(self, date: datetime) -> Dict[str, Any]:
        """
        Get bid statistics for a specific date.
        
        Args:
            date: Date to get statistics for
        
        Returns:
            Dictionary with bid statistics
        """
        try:
            # Get start and end of day
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            
            # Get recommendations generated on the day
            recommendations = await self.db.bid_recommendations.find({
                "metadata.generation_time": {
                    "$gte": start_of_day.isoformat(),
                    "$lt": end_of_day.isoformat()
                }
            }).to_list(length=None)
            
            # Get bids scheduled on the day
            scheduled_bids = await self.db.scheduled_bids.find({
                "created_at": {
                    "$gte": start_of_day,
                    "$lt": end_of_day
                }
            }).to_list(length=None)
            
            # Get bids placed on the day
            placed_bids = await self.db.scheduled_bids.find({
                "status": "placed",
                "placed_at": {
                    "$gte": start_of_day,
                    "$lt": end_of_day
                }
            }).to_list(length=None)
            
            # Get successful bids on the day
            successful_bids = await self.db.scheduled_bids.find({
                "status": "won",
                "won_at": {
                    "$gte": start_of_day,
                    "$lt": end_of_day
                }
            }).to_list(length=None)
            
            # Calculate total potential profit
            total_potential_profit = sum(
                rec["profit_potential"] for rec in recommendations
            )
            
            return {
                "date": date.isoformat(),
                "recommendations_generated": len(recommendations),
                "bids_scheduled": len(scheduled_bids),
                "bids_placed": len(placed_bids),
                "successful_bids": len(successful_bids),
                "total_potential_profit": total_potential_profit
            }
        except Exception as e:
            logger.error(f"Error getting bid stats by date: {str(e)}")
            raise
    
    async def save_notification(self, notification: Notification) -> str:
        """
        Save a notification to the database.
        
        Args:
            notification: Notification to save
        
        Returns:
            Notification ID
        """
        try:
            # Convert to dict
            notif_dict = notification.dict()
            
            # Save to database
            await self.db.notifications.update_one(
                {"id": notification.id},
                {"$set": notif_dict},
                upsert=True
            )
            
            return notification.id
        except Exception as e:
            logger.error(f"Error saving notification: {str(e)}")
            raise
    
    async def get_users_with_notification_preferences(
        self,
        notification_type: str
    ) -> List[Dict[str, Any]]:
        """
        Get users with notification preferences for a specific type.
        
        Args:
            notification_type: Notification type
        
        Returns:
            List of users with notification preferences
        """
        try:
            # Get from database
            users = await self.db.users.find({
                f"notification_preferences.{notification_type}.enabled": True
            }).to_list(length=None)
            
            # Add preferred channel
            for user in users:
                user["preferred_channel"] = user.get(
                    "notification_preferences", {}
                ).get(notification_type, {}).get(
                    "preferred_channel", "email"
                )
            
            return users
        except Exception as e:
            logger.error(f"Error getting users with notification preferences: {str(e)}")
            raise
    
    async def save_daily_summary(self, summary: DailySummary) -> datetime:
        """
        Save a daily summary to the database.
        
        Args:
            summary: Daily summary to save
        
        Returns:
            Summary date
        """
        try:
            # Convert to dict
            summary_dict = summary.dict()
            
            # Save to database
            await self.db.daily_summaries.update_one(
                {"date": summary.date},
                {"$set": summary_dict},
                upsert=True
            )
            
            return summary.date
        except Exception as e:
            logger.error(f"Error saving daily summary: {str(e)}")
            raise
    
    async def get_daily_summary(self, date: datetime) -> Optional[DailySummary]:
        """
        Get a daily summary by date.
        
        Args:
            date: Date to get summary for
        
        Returns:
            Daily summary or None if not found
        """
        try:
            # Get from database
            summary_dict = await self.db.daily_summaries.find_one({"date": date})
            
            if not summary_dict:
                return None
            
            # Convert to model
            return DailySummary(**summary_dict)
        except Exception as e:
            logger.error(f"Error getting daily summary: {str(e)}")
            raise
    
    async def get_scraper_config(self, platform: str) -> Dict[str, Any]:
        """
        Get scraper configuration for a platform.
        
        Args:
            platform: Platform name
        
        Returns:
            Scraper configuration
        """
        try:
            # Get from database
            config = await self.db.scraper_configs.find_one({"platform": platform})
            
            if not config:
                # Return default config
                return {
                    "platform": platform,
                    "enabled": True,
                    "max_pages": 5,
                    "categories": [],
                    "keywords": []
                }
            
            return config
        except Exception as e:
            logger.error(f"Error getting scraper config: {str(e)}")
            raise