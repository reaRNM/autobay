"""
Unit tests for the Bid Intelligence Core.
"""

import unittest
import asyncio
import time
from typing import Dict, List, Any

from bid_intelligence import (
    BidMonitor,
    BidAdjuster,
    KnapsackOptimizer,
    FraudDetector,
    AuctionItem,
    BidData,
    RiskScore
)
from bid_intelligence.utils import KafkaSimulator, calculate_risk_score


class TestBidIntelligence(unittest.TestCase):
    """Test cases for the Bid Intelligence Core."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Create Kafka simulator
        self.kafka_client = KafkaSimulator()
        
        # Create bid monitor
        self.bid_monitor = BidMonitor(kafka_client=self.kafka_client)
        
        # Create test items
        self.test_items = self._create_test_items()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.loop.run_until_complete(self.bid_monitor.stop())
        self.loop.close()
    
    def _create_test_items(self):
        """Create test auction items."""
        items = []
        for i in range(1, 6):
            risk_score = calculate_risk_score(
                price_volatility=0.2,
                historical_data_points=10 + i * 5,
                category_risk=0.3,
                seller_rating=0.8,
                time_pressure=0.4
            )
            
            item = AuctionItem(
                item_id=f"test-item-{i}",
                title=f"Test Item {i}",
                category=f"Category {(i % 3) + 1}",
                current_bid=50.0 * i,
                estimated_value=100.0 * i,
                estimated_profit=30.0 * i,
                risk_score=risk_score,
                end_time=int(time.time()) + 3600  # 1 hour from now
            )
            
            items.append(item)
        
        return items
    
    def test_risk_score_calculation(self):
        """Test risk score calculation."""
        risk_score = calculate_risk_score(
            price_volatility=0.5,
            historical_data_points=20,
            category_risk=0.3,
            seller_rating=0.7,
            time_pressure=0.4
        )
        
        self.assertIsInstance(risk_score, RiskScore)
        self.assertTrue(0 <= risk_score.score <= 1)
        self.assertEqual(len(risk_score.factors), 5)
    
    def test_knapsack_optimizer(self):
        """Test knapsack optimizer."""
        optimizer = KnapsackOptimizer()
        
        # Set budget
        budget = sum(item.current_bid for item in self.test_items) * 0.7  # 70% of total cost
        
        # Run optimization
        combinations = optimizer.optimize(self.test_items, budget)
        
        self.assertGreater(len(combinations), 0)
        
        # Check first combination
        combo = combinations[0]
        self.assertIsNotNone(combo.items)
        self.assertLessEqual(combo.total_cost, budget)
        self.assertGreater(combo.total_profit, 0)
    
    def test_bid_data_processing(self):
        """Test bid data processing."""
        async def _test():
            # Start bid monitor
            await self.bid_monitor.start()
            
            # Create a bid
            bid = BidData(
                item_id="test-item-1",
                bidder_id="test-bidder",
                bid_amount=100.0,
                timestamp=int(time.time())
            )
            
            # Produce bid data
            await self.kafka_client.produce("bid_data", bid.to_dict())
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Get bid history
            history = await self.bid_monitor.get_bid_history("test-item-1")
            
            self.assertIsNotNone(history)
            self.assertEqual(len(history.bids), 1)
            self.assertEqual(history.bids[0].bid_amount, 100.0)
        
        self.loop.run_until_complete(_test())
    
    def test_fraud_detection(self):
        """Test fraud detection."""
        async def _test():
            # Create fraud detector
            fraud_detector = FraudDetector(bid_monitor=self.bid_monitor)
            
            # Start bid monitor
            await self.bid_monitor.start()
            
            # Create a sequence of suspicious bids
            item_id = "test-item-1"
            bidder_id = "suspicious-bidder"
            
            # Initial bid
            bid1 = BidData(
                item_id=item_id,
                bidder_id=bidder_id,
                bid_amount=100.0,
                timestamp=int(time.time())
            )
            
            # Rapid follow-up bid
            bid2 = BidData(
                item_id=item_id,
                bidder_id=bidder_id,
                bid_amount=110.0,
                timestamp=int(time.time()) + 1,
                previous_bid=100.0
            )
            
            # Large jump bid
            bid3 = BidData(
                item_id=item_id,
                bidder_id="another-bidder",
                bid_amount=200.0,
                timestamp=int(time.time()) + 2,
                previous_bid=110.0
            )
            
            # Produce bid data
            await self.kafka_client.produce("bid_data", bid1.to_dict())
            await asyncio.sleep(0.1)
            await self.kafka_client.produce("bid_data", bid2.to_dict())
            await asyncio.sleep(0.1)
            await self.kafka_client.produce("bid_data", bid3.to_dict())
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            # Get alerts
            alerts = await fraud_detector.get_alerts()
            
            # We should have at least one alert (large jump)
            self.assertGreater(len(alerts), 0)
        
        self.loop.run_until_complete(_test())


if __name__ == "__main__":
    unittest.main()