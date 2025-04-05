"""
Fraud detection and alerting module.

This module provides classes and functions for detecting and alerting on
suspicious bidding patterns that may indicate fraud or manipulation.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
import statistics
from collections import deque

from .utils import BidData, AuctionItem, BidHistory
from .real_time import BidMonitor


logger = logging.getLogger("bid_intelligence.fraud")


@dataclass
class FraudAlert:
    """
    Alert for suspicious bidding activity.
    
    Attributes:
        item_id: ID of the auction item
        alert_type: Type of alert
        severity: Severity level (1-5, with 5 being most severe)
        description: Description of the suspicious activity
        bid_data: Related bid data
        timestamp: Time the alert was generated
        additional_info: Additional information about the alert
    """
    item_id: str
    alert_type: str
    severity: int
    description: str
    bid_data: Optional[BidData] = None
    timestamp: int = field(default_factory=lambda: int(time.time()))
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the fraud alert to a dictionary.
        
        Returns:
            Dict[str, Any]: Fraud alert as a dictionary
        """
        result = {
            "item_id": self.item_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "description": self.description,
            "timestamp": self.timestamp,
            "additional_info": self.additional_info
        }
        
        if self.bid_data:
            result["bid_data"] = self.bid_data.to_dict()
        
        return result


class FraudDetector:
    """
    Detects suspicious bidding patterns.
    
    This class analyzes bid data to identify and flag suspicious bidding patterns
    that may indicate fraud or manipulation.
    """
    
    def __init__(self, bid_monitor: BidMonitor, config: Dict[str, Any] = None):
        """
        Initialize the fraud detector.
        
        Args:
            bid_monitor: Bid monitor for accessing real-time auction data
            config: Configuration dictionary
        """
        self.bid_monitor = bid_monitor
        self.config = config or {}
        
        # Configure detection thresholds
        self.rapid_bid_threshold = self.config.get("rapid_bid_threshold", 5)  # seconds
        self.large_jump_threshold = self.config.get("large_jump_threshold", 50)  # percentage
        self.shill_bid_count_threshold = self.config.get("shill_bid_count_threshold", 3)
        self.shill_bid_time_window = self.config.get("shill_bid_time_window", 300)  # seconds
        self.outlier_std_dev_threshold = self.config.get("outlier_std_dev_threshold", 2.5)
        
        # Store alerts
        self.alerts = []
        self.alert_callbacks = []
        
        # Track items with active alerts to avoid duplicate alerts
        self.active_alerts = set()
        
        # Register callback with bid monitor
        self.bid_monitor.register_callback(self._on_data_update)
    
    async def _on_data_update(self, data_type: str, data: Any) -> None:
        """
        Handle data updates from the bid monitor.
        
        Args:
            data_type: Type of data ("bid_data" or "item_data")
            data: Data object (BidData or AuctionItem)
        """
        if data_type == "bid_data":
            # Check for suspicious bidding patterns
            await self._check_for_fraud(data)
    
    async def _check_for_fraud(self, bid_data: BidData) -> None:
        """
        Check for suspicious bidding patterns.
        
        Args:
            bid_data: Bid data to check
        """
        item_id = bid_data.item_id
        
        # Get bid history
        history = await self.bid_monitor.get_bid_history(item_id)
        if history is None or len(history.bids) < 2:
            return
        
        # Run fraud detection checks
        await asyncio.gather(
            self._check_rapid_bidding(item_id, history, bid_data),
            self._check_large_jumps(item_id, history, bid_data),
            self._check_shill_bidding(item_id, history, bid_data),
            self._check_bid_outliers(item_id, history, bid_data)
        )
    
    async def _check_rapid_bidding(self, item_id: str, history: BidHistory, bid_data: BidData) -> None:
        """
        Check for rapid bidding (multiple bids in quick succession).
        
        Args:
            item_id: ID of the auction item
            history: Bid history for the item
            bid_data: Current bid data
        """
        if len(history.bids) < 2:
            return
        
        # Get the previous bid
        prev_bid = history.bids[-2]
        
        # Check if the bids are from the same bidder
        if prev_bid.bidder_id == bid_data.bidder_id:
            # Check if the bids were placed in rapid succession
            time_diff = bid_data.timestamp - prev_bid.timestamp
            
            if time_diff <= self.rapid_bid_threshold:
                # Create a fraud alert
                alert = FraudAlert(
                    item_id=item_id,
                    alert_type="rapid_bidding",
                    severity=2,
                    description=f"Rapid bidding detected: {time_diff} seconds between bids",
                    bid_data=bid_data,
                    additional_info={
                        "time_difference": time_diff,
                        "threshold": self.rapid_bid_threshold,
                        "previous_bid": prev_bid.to_dict()
                    }
                )
                
                await self._add_alert(alert)
    
    async def _check_large_jumps(self, item_id: str, history: BidHistory, bid_data: BidData) -> None:
        """
        Check for unusually large jumps in bid amounts.
        
        Args:
            item_id: ID of the auction item
            history: Bid history for the item
            bid_data: Current bid data
        """
        if len(history.bids) < 2:
            return
        
        # Get the previous bid
        prev_bid = history.bids[-2]
        
        # Calculate the percentage increase
        if prev_bid.bid_amount > 0:
            percentage_increase = ((bid_data.bid_amount - prev_bid.bid_amount) / prev_bid.bid_amount) * 100
            
            if percentage_increase >= self.large_jump_threshold:
                # Create a fraud alert
                alert = FraudAlert(
                    item_id=item_id,
                    alert_type="large_jump",
                    severity=3,
                    description=f"Large bid jump detected: {percentage_increase:.1f}% increase",
                    bid_data=bid_data,
                    additional_info={
                        "percentage_increase": percentage_increase,
                        "threshold": self.large_jump_threshold,
                        "previous_bid": prev_bid.to_dict()
                    }
                )
                
                await self._add_alert(alert)
    
    async def _check_shill_bidding(self, item_id: str, history: BidHistory, bid_data: BidData) -> None:
        """
        Check for shill bidding (same bidder repeatedly bidding to drive up price).
        
        Args:
            item_id: ID of the auction item
            history: Bid history for the item
            bid_data: Current bid data
        """
        # Get recent bids within the time window
        current_time = int(time.time())
        window_start = current_time - self.shill_bid_time_window
        
        recent_bids = [bid for bid in history.bids if bid.timestamp >= window_start]
        
        # Count bids by bidder
        bidder_counts = {}
        for bid in recent_bids:
            bidder_counts[bid.bidder_id] = bidder_counts.get(bid.bidder_id, 0) + 1
        
        # Check if any bidder exceeds the threshold
        for bidder_id, count in bidder_counts.items():
            if count >= self.shill_bid_count_threshold:
                # Create a fraud alert
                alert = FraudAlert(
                    item_id=item_id,
                    alert_type="shill_bidding",
                    severity=4,
                    description=f"Possible shill bidding detected: {count} bids in {self.shill_bid_time_window} seconds",
                    bid_data=bid_data,
                    additional_info={
                        "bidder_id": bidder_id,
                        "bid_count": count,
                        "time_window": self.shill_bid_time_window,
                        "threshold": self.shill_bid_count_threshold
                    }
                )
                
                await self._add_alert(alert)
    
    async def _check_bid_outliers(self, item_id: str, history: BidHistory, bid_data: BidData) -> None:
        """
        Check for statistical outliers in bid amounts.
        
        Args:
            item_id: ID of the auction item
            history: Bid history for the item
            bid_data: Current bid data
        """
        if len(history.bids) < 5:  # Need enough data for statistical analysis
            return
        
        # Get bid amounts
        bid_amounts = [bid.bid_amount for bid in history.bids[:-1]]  # Exclude current bid
        
        try:
            # Calculate mean and standard deviation
            mean = statistics.mean(bid_amounts)
            stdev = statistics.stdev(bid_amounts)
            
            # Check if current bid is an outlier
            if stdev > 0:
                z_score = abs(bid_data.bid_amount - mean) / stdev
                
                if z_score > self.outlier_std_dev_threshold:
                    # Create a fraud alert
                    alert = FraudAlert(
                        item_id=item_id,
                        alert_type="bid_outlier",
                        severity=3,
                        description=f"Bid amount is a statistical outlier: {z_score:.2f} standard deviations from mean",
                        bid_data=bid_data,
                        additional_info={
                            "z_score": z_score,
                            "threshold": self.outlier_std_dev_threshold,
                            "mean": mean,
                            "stdev": stdev
                        }
                    )
                    
                    await self._add_alert(alert)
        
        except statistics.StatisticsError:
            # Not enough data or other statistical error
            pass
    
    async def _add_alert(self, alert: FraudAlert) -> None:
        """
        Add a fraud alert and notify callbacks.
        
        Args:
            alert: Fraud alert to add
        """
        # Check if there's already an active alert of the same type for this item
        alert_key = f"{alert.item_id}:{alert.alert_type}"
        
        if alert_key in self.active_alerts:
            # Update existing alert instead of creating a new one
            return
        
        # Add to active alerts
        self.active_alerts.add(alert_key)
        
        # Add to alerts list
        self.alerts.append(alert)
        
        # Log the alert
        logger.warning(f"Fraud alert: {alert.description} (Item: {alert.item_id}, Type: {alert.alert_type}, Severity: {alert.severity})")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            asyncio.create_task(callback(alert))
        
        # Schedule removal from active alerts after a delay
        asyncio.create_task(self._remove_active_alert(alert_key))
    
    async def _remove_active_alert(self, alert_key: str, delay: int = 300) -> None:
        """
        Remove an alert from the active alerts set after a delay.
        
        Args:
            alert_key: Alert key to remove
            delay: Delay in seconds
        """
        await asyncio.sleep(delay)
        self.active_alerts.discard(alert_key)
    
    def register_alert_callback(self, callback: Callable) -> None:
        """
        Register a callback function to be called when a fraud alert is generated.
        
        Args:
            callback: Callback function (async function taking a FraudAlert)
        """
        self.alert_callbacks.append(callback)
    
    def unregister_alert_callback(self, callback: Callable) -> None:
        """
        Unregister a callback function.
        
        Args:
            callback: Callback function to unregister
        """
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    async def get_alerts(self, item_id: Optional[str] = None, 
                       alert_type: Optional[str] = None,
                       min_severity: Optional[int] = None,
                       max_alerts: Optional[int] = None) -> List[FraudAlert]:
        """
        Get fraud alerts matching the specified criteria.
        
        Args:
            item_id: Filter by item ID
            alert_type: Filter by alert type
            min_severity: Filter by minimum severity
            max_alerts: Maximum number of alerts to return
            
        Returns:
            List[FraudAlert]: List of fraud alerts
        """
        filtered_alerts = self.alerts
        
        # Apply filters
        if item_id is not None:
            filtered_alerts = [alert for alert in filtered_alerts if alert.item_id == item_id]
        
        if alert_type is not None:
            filtered_alerts = [alert for alert in filtered_alerts if alert.alert_type == alert_type]
        
        if min_severity is not None:
            filtered_alerts = [alert for alert in filtered_alerts if alert.severity >= min_severity]
        
        # Sort by timestamp (newest first)
        filtered_alerts.sort(key=lambda alert: alert.timestamp, reverse=True)
        
        # Limit the number of alerts
        if max_alerts is not None:
            filtered_alerts = filtered_alerts[:max_alerts]
        
        return filtered_alerts
    
    async def clear_alerts(self, item_id: Optional[str] = None) -> None:
        """
        Clear fraud alerts.
        
        Args:
            item_id: Clear alerts for a specific item (if None, clear all alerts)
        """
        if item_id is None:
            self.alerts = []
            self.active_alerts = set()
        else:
            self.alerts = [alert for alert in self.alerts if alert.item_id != item_id]
            self.active_alerts = {key for key in self.active_alerts if not key.startswith(f"{item_id}:")}