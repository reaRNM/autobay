"""
Real-time bid monitoring and adjustment module.

This module provides classes for monitoring live auction data streams
and adjusting maximum bid suggestions in real-time based on competitor
behavior and market trends.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
import statistics
from collections import deque

from .utils import AuctionItem, BidData, RiskScore, KafkaSimulator


logger = logging.getLogger("bid_intelligence.real_time")


class BidHistory:
    """
    Maintains a history of bids for an auction item.
    
    This class stores and analyzes historical bid data for a specific auction item,
    providing insights into bidding patterns and trends.
    """
    
    def __init__(self, item_id: str, max_history: int = 100):
        """
        Initialize the bid history.
        
        Args:
            item_id: ID of the auction item
            max_history: Maximum number of bids to store in history
        """
        self.item_id = item_id
        self.max_history = max_history
        self.bids = deque(maxlen=max_history)
        self.bidders = set()
        self.last_update_time = 0
    
    def add_bid(self, bid: BidData) -> None:
        """
        Add a bid to the history.
        
        Args:
            bid: Bid data to add
        """
        if bid.item_id != self.item_id:
            raise ValueError(f"Bid item ID {bid.item_id} does not match history item ID {self.item_id}")
        
        self.bids.append(bid)
        self.bidders.add(bid.bidder_id)
        self.last_update_time = max(self.last_update_time, bid.timestamp)
    
    def get_current_bid(self) -> Optional[float]:
        """
        Get the current highest bid amount.
        
        Returns:
            Optional[float]: Current highest bid amount, or None if no bids
        """
        if not self.bids:
            return None
        
        return self.bids[-1].bid_amount
    
    def get_bid_velocity(self, time_window: int = 3600) -> float:
        """
        Calculate the bid velocity (bids per hour) within a time window.
        
        Args:
            time_window: Time window in seconds (default: 1 hour)
            
        Returns:
            float: Bids per hour
        """
        if not self.bids:
            return 0.0
        
        current_time = int(time.time())
        window_start = current_time - time_window
        
        # Count bids within the time window
        recent_bids = [bid for bid in self.bids if bid.timestamp >= window_start]
        
        # Calculate bids per hour
        return len(recent_bids) * (3600 / time_window)
    
    def get_price_trend(self, time_window: int = 3600) -> float:
        """
        Calculate the price trend (percentage change per hour) within a time window.
        
        Args:
            time_window: Time window in seconds (default: 1 hour)
            
        Returns:
            float: Percentage change per hour
        """
        if len(self.bids) < 2:
            return 0.0
        
        current_time = int(time.time())
        window_start = current_time - time_window
        
        # Get bids within the time window
        recent_bids = [bid for bid in self.bids if bid.timestamp >= window_start]
        
        if len(recent_bids) < 2:
            return 0.0
        
        # Calculate percentage change
        start_price = recent_bids[0].bid_amount
        end_price = recent_bids[-1].bid_amount
        
        if start_price <= 0:
            return 0.0
        
        percentage_change = ((end_price - start_price) / start_price) * 100
        
        # Convert to percentage change per hour
        time_span = (recent_bids[-1].timestamp - recent_bids[0].timestamp) / 3600
        if time_span <= 0:
            return 0.0
        
        return percentage_change / time_span
    
    def get_bid_increments(self) -> List[float]:
        """
        Get a list of bid increments (absolute changes between consecutive bids).
        
        Returns:
            List[float]: List of bid increments
        """
        if len(self.bids) < 2:
            return []
        
        increments = []
        for i in range(1, len(self.bids)):
            prev_bid = self.bids[i-1].bid_amount
            curr_bid = self.bids[i].bid_amount
            increments.append(curr_bid - prev_bid)
        
        return increments
    
    def get_average_increment(self) -> Optional[float]:
        """
        Get the average bid increment.
        
        Returns:
            Optional[float]: Average bid increment, or None if insufficient data
        """
        increments = self.get_bid_increments()
        if not increments:
            return None
        
        return sum(increments) / len(increments)
    
    def get_bidder_activity(self) -> Dict[str, int]:
        """
        Get the number of bids placed by each bidder.
        
        Returns:
            Dict[str, int]: Dictionary mapping bidder IDs to bid counts
        """
        activity = {}
        for bid in self.bids:
            activity[bid.bidder_id] = activity.get(bid.bidder_id, 0) + 1
        
        return activity
    
    def get_most_active_bidder(self) -> Optional[str]:
        """
        Get the ID of the most active bidder.
        
        Returns:
            Optional[str]: ID of the most active bidder, or None if no bids
        """
        activity = self.get_bidder_activity()
        if not activity:
            return None
        
        return max(activity.items(), key=lambda x: x[1])[0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the bid history.
        
        Returns:
            Dict[str, Any]: Dictionary of bid statistics
        """
        if not self.bids:
            return {
                "item_id": self.item_id,
                "bid_count": 0,
                "bidder_count": 0,
                "current_bid": None,
                "bid_velocity": 0.0,
                "price_trend": 0.0,
                "average_increment": None
            }
        
        return {
            "item_id": self.item_id,
            "bid_count": len(self.bids),
            "bidder_count": len(self.bidders),
            "current_bid": self.get_current_bid(),
            "bid_velocity": self.get_bid_velocity(),
            "price_trend": self.get_price_trend(),
            "average_increment": self.get_average_increment()
        }


class BidMonitor:
    """
    Monitors real-time auction data streams.
    
    This class subscribes to auction data streams (e.g., from Kafka)
    and processes incoming bid data in real-time.
    """
    
    def __init__(self, kafka_client=None):
        """
        Initialize the bid monitor.
        
        Args:
            kafka_client: Kafka client for consuming auction data streams
                         (if None, a simulator will be created)
        """
        self.kafka_client = kafka_client or KafkaSimulator()
        self.bid_histories = {}  # item_id -> BidHistory
        self.item_data = {}  # item_id -> AuctionItem
        self.callbacks = []
        self.running = False
        self._lock = asyncio.Lock()
    
    async def start(self) -> None:
        """Start monitoring auction data streams."""
        if self.running:
            return
        
        logger.info("Starting bid monitor")
        self.running = True
        
        # Start Kafka client
        if isinstance(self.kafka_client, KafkaSimulator):
            await self.kafka_client.start()
        
        # Subscribe to bid data topic
        await self.kafka_client.consume("bid_data", self._process_bid_data)
        
        # Subscribe to item data topic
        await self.kafka_client.consume("item_data", self._process_item_data)
        
        logger.info("Bid monitor started")
    
    async def stop(self) -> None:
        """Stop monitoring auction data streams."""
        if not self.running:
            return
        
        logger.info("Stopping bid monitor")
        self.running = False
        
        # Stop Kafka client
        if isinstance(self.kafka_client, KafkaSimulator):
            await self.kafka_client.stop()
        
        logger.info("Bid monitor stopped")
    
    async def _process_bid_data(self, message: Dict[str, Any]) -> None:
        """
        Process a bid data message from Kafka.
        
        Args:
            message: Bid data message
        """
        try:
            bid_data = BidData.from_dict(message)
            
            async with self._lock:
                # Create bid history if it doesn't exist
                if bid_data.item_id not in self.bid_histories:
                    self.bid_histories[bid_data.item_id] = BidHistory(bid_data.item_id)
                
                # Add bid to history
                self.bid_histories[bid_data.item_id].add_bid(bid_data)
                
                # Update current bid in item data
                if bid_data.item_id in self.item_data:
                    self.item_data[bid_data.item_id].current_bid = bid_data.bid_amount
            
            # Notify callbacks
            for callback in self.callbacks:
                asyncio.create_task(callback("bid_data", bid_data))
            
            logger.debug(f"Processed bid data: {bid_data.to_dict()}")
        
        except Exception as e:
            logger.error(f"Error processing bid data: {e}")
    
    async def _process_item_data(self, message: Dict[str, Any]) -> None:
        """
        Process an item data message from Kafka.
        
        Args:
            message: Item data message
        """
        try:
            item_data = AuctionItem.from_dict(message)
            
            async with self._lock:
                # Store item data
                self.item_data[item_data.item_id] = item_data
            
            # Notify callbacks
            for callback in self.callbacks:
                asyncio.create_task(callback("item_data", item_data))
            
            logger.debug(f"Processed item data: {item_data.to_dict()}")
        
        except Exception as e:
            logger.error(f"Error processing item data: {e}")
    
    def register_callback(self, callback: Callable) -> None:
        """
        Register a callback function to be called when new data is received.
        
        Args:
            callback: Callback function (async function taking data_type and data)
        """
        self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable) -> None:
        """
        Unregister a callback function.
        
        Args:
            callback: Callback function to unregister
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    async def get_bid_history(self, item_id: str) -> Optional[BidHistory]:
        """
        Get the bid history for an item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            Optional[BidHistory]: Bid history for the item, or None if not found
        """
        async with self._lock:
            return self.bid_histories.get(item_id)
    
    async def get_item_data(self, item_id: str) -> Optional[AuctionItem]:
        """
        Get the data for an item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            Optional[AuctionItem]: Item data, or None if not found
        """
        async with self._lock:
            return self.item_data.get(item_id)
    
    async def get_active_items(self) -> List[AuctionItem]:
        """
        Get a list of active auction items.
        
        Returns:
            List[AuctionItem]: List of active auction items
        """
        async with self._lock:
            return [item for item in self.item_data.values() if item.is_active]
    
    async def get_bid_statistics(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics about the bid history for an item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            Optional[Dict[str, Any]]: Bid statistics, or None if not found
        """
        history = await self.get_bid_history(item_id)
        if history is None:
            return None
        
        return history.get_statistics()


class BidAdjuster:
    """
    Adjusts maximum bid suggestions in real-time.
    
    This class analyzes bid data and market trends to adjust maximum bid
    suggestions for auction items in real-time.
    """
    
    def __init__(self, bid_monitor: BidMonitor, config: Dict[str, Any] = None):
        """
        Initialize the bid adjuster.
        
        Args:
            bid_monitor: Bid monitor for accessing real-time auction data
            config: Configuration dictionary
        """
        self.bid_monitor = bid_monitor
        self.config = config or {}
        self.default_markup = self.config.get("default_markup", 0.2)  # 20% markup
        self.max_markup = self.config.get("max_markup", 0.5)  # 50% markup
        self.min_markup = self.config.get("min_markup", 0.05)  # 5% markup
        self.trend_weight = self.config.get("trend_weight", 0.3)
        self.velocity_weight = self.config.get("velocity_weight", 0.2)
        self.competition_weight = self.config.get("competition_weight", 0.3)
        self.risk_weight = self.config.get("risk_weight", 0.2)
        
        # Register callback with bid monitor
        self.bid_monitor.register_callback(self._on_data_update)
    
    async def _on_data_update(self, data_type: str, data: Any) -> None:
        """
        Handle data updates from the bid monitor.
        
        Args:
            data_type: Type of data ("bid_data" or "item_data")
            data: Data object (BidData or AuctionItem)
        """
        # This method can be extended to trigger real-time adjustments
        # when new data is received
        pass
    
    async def calculate_max_bid(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Calculate the maximum bid for an item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            Optional[Dict[str, Any]]: Maximum bid information, or None if insufficient data
        """
        # Get item data
        item = await self.bid_monitor.get_item_data(item_id)
        if item is None:
            logger.warning(f"Item {item_id} not found")
            return None
        
        # Get bid statistics
        stats = await self.bid_monitor.get_bid_statistics(item_id)
        if stats is None:
            logger.warning(f"No bid statistics available for item {item_id}")
            return None
        
        # Get base bid from estimated value
        base_bid = item.estimated_value
        
        # Calculate markup based on various factors
        markup = self.default_markup
        
        # Adjust markup based on price trend
        price_trend = stats.get("price_trend", 0.0)
        if price_trend > 0:
            # Positive trend (prices rising) - increase markup
            trend_adjustment = min(price_trend / 100, 0.2)  # Cap at 20%
            markup += trend_adjustment * self.trend_weight
        else:
            # Negative trend (prices falling) - decrease markup
            trend_adjustment = min(abs(price_trend) / 100, 0.2)  # Cap at 20%
            markup -= trend_adjustment * self.trend_weight
        
        # Adjust markup based on bid velocity
        bid_velocity = stats.get("bid_velocity", 0.0)
        velocity_factor = min(bid_velocity / 10, 0.3)  # Cap at 30%
        markup += velocity_factor * self.velocity_weight
        
        # Adjust markup based on competition (number of bidders)
        bidder_count = stats.get("bidder_count", 0)
        competition_factor = min(bidder_count / 5, 0.3)  # Cap at 30%
        markup += competition_factor * self.competition_weight
        
        # Adjust markup based on risk score
        risk_score = item.risk_score.score
        markup -= risk_score * self.risk_weight
        
        # Ensure markup is within bounds
        markup = max(self.min_markup, min(self.max_markup, markup))
        
        # Calculate maximum bid
        max_bid = base_bid * (1 - markup)
        
        # Ensure max bid is above current bid
        current_bid = stats.get("current_bid", 0.0)
        if current_bid is not None:
            min_increment = stats.get("average_increment", 1.0) or 1.0
            max_bid = max(max_bid, current_bid + min_increment)
        
        # Return max bid information
        return {
            "item_id": item_id,
            "max_bid": max_bid,
            "current_bid": current_bid,
            "estimated_value": item.estimated_value,
            "markup": markup,
            "markup_percentage": markup * 100,
            "factors": {
                "price_trend": price_trend,
                "bid_velocity": bid_velocity,
                "competition": bidder_count,
                "risk_score": risk_score
            }
        }
    
    async def get_bid_recommendations(self, item_ids: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get bid recommendations for multiple items.
        
        Args:
            item_ids: List of item IDs to get recommendations for
                     (if None, get recommendations for all active items)
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping item IDs to bid recommendations
        """
        if item_ids is None:
            # Get all active items
            active_items = await self.bid_monitor.get_active_items()
            item_ids = [item.item_id for item in active_items]
        
        recommendations = {}
        for item_id in item_ids:
            recommendation = await self.calculate_max_bid(item_id)
            if recommendation is not None:
                recommendations[item_id] = recommendation
        
        return recommendations