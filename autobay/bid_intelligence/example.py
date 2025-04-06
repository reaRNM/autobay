"""
Example usage of the Bid Intelligence Core.
"""

import asyncio
"""
Example usage of the Bid Intelligence Core.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any

from bid_intelligence import (
    BidMonitor,
    BidAdjuster,
    KnapsackOptimizer,
    FraudDetector,
    AuctionItem,
    BidData,
    RiskScore,
    setup_logging
)
from bid_intelligence.utils import KafkaSimulator, calculate_risk_score


# Set up logging
logger = setup_logging(log_level="INFO")


async def on_fraud_alert(alert):
    """Handle fraud alerts."""
    print(f"\n⚠️ FRAUD ALERT: {alert.description}")
    print(f"   Item: {alert.item_id}, Type: {alert.alert_type}, Severity: {alert.severity}")
    print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}")


async def simulate_auction_data(kafka_client, num_items=5, num_bids=20):
    """Simulate auction data for testing."""
    print("Simulating auction data...")
    
    # Create sample items
    items = []
    for i in range(1, num_items + 1):
        # Calculate risk score
        risk_score = calculate_risk_score(
            price_volatility=0.2 + (i % 3) * 0.1,
            historical_data_points=10 + i * 5,
            category_risk=0.3 if i % 2 == 0 else 0.5,
            seller_rating=0.8 - (i % 3) * 0.1,
            time_pressure=0.4
        )
        
        # Create item
        item = AuctionItem(
            item_id=f"item-{i}",
            title=f"Test Item {i}",
            category=f"Category {(i % 3) + 1}",
            current_bid=50.0 * i,
            estimated_value=100.0 * i,
            estimated_profit=30.0 * i,
            risk_score=risk_score,
            end_time=int(time.time()) + 3600  # 1 hour from now
        )
        
        items.append(item)
        
        # Produce item data
        await kafka_client.produce("item_data", item.to_dict())
    
    # Simulate bids
    for _ in range(num_bids):
        # Choose a random item
        item_index = int(time.time() * 1000) % len(items)
        item = items[item_index]
        
        # Create a bid
        bid_amount = item.current_bid + (5.0 + (int(time.time() * 1000) % 10))
        
        bid = BidData(
            item_id=item.item_id,
            bidder_id=f"bidder-{1 + (int(time.time() * 1000) % 3)}",
            bid_amount=bid_amount,
            timestamp=int(time.time()),
            previous_bid=item.current_bid
        )
        
        # Update item's current bid
        item.current_bid = bid_amount
        
        # Produce bid data
        await kafka_client.produce("bid_data", bid.to_dict())
        
        # Produce updated item data
        await kafka_client.produce("item_data", item.to_dict())
        
        # Wait a bit
        await asyncio.sleep(0.5)
    
    # Simulate a suspicious bid pattern for fraud detection
    suspicious_item = items[0]
    
    # Rapid bidding
    for i in range(3):
        bid = BidData(
            item_id=suspicious_item.item_id,
            bidder_id="suspicious-bidder",
            bid_amount=suspicious_item.current_bid + 5.0,
            timestamp=int(time.time()),
            previous_bid=suspicious_item.current_bid
        )
        
        suspicious_item.current_bid = bid.bid_amount
        
        await kafka_client.produce("bid_data", bid.to_dict())
        await asyncio.sleep(0.1)  # Very short delay between bids
    
    # Large jump
    bid = BidData(
        item_id=suspicious_item.item_id,
        bidder_id="suspicious-bidder-2",
        bid_amount=suspicious_item.current_bid * 2,  # Double the current bid
        timestamp=int(time.time()),
        previous_bid=suspicious_item.current_bid
    )
    
    suspicious_item.current_bid = bid.bid_amount
    
    await kafka_client.produce("bid_data", bid.to_dict())
    
    print("Auction data simulation complete")


async def demonstrate_bid_adjustments(bid_adjuster):
    """Demonstrate real-time bid adjustments."""
    print("\n=== Real-Time Bid Adjustments ===")
    
    # Get bid recommendations
    recommendations = await bid_adjuster.get_bid_recommendations()
    
    if not recommendations:
        print("No bid recommendations available yet")
        return
    
    print(f"Generated {len(recommendations)} bid recommendations:")
    
    for item_id, recommendation in recommendations.items():
        print(f"\nItem: {item_id}")
        print(f"  Current Bid: ${recommendation['current_bid']:.2f}")
        print(f"  Maximum Bid: ${recommendation['max_bid']:.2f}")
        print(f"  Estimated Value: ${recommendation['estimated_value']:.2f}")
        print(f"  Markup: {recommendation['markup_percentage']:.1f}%")
        
        print("  Factors:")
        for factor, value in recommendation['factors'].items():
            print(f"    {factor}: {value}")


async def demonstrate_knapsack_optimization(optimizer, bid_monitor):
    """Demonstrate knapsack optimization."""
    print("\n=== Budget Optimization (Knapsack Algorithm) ===")
    
    # Get active items
    items = await bid_monitor.get_active_items()
    
    if not items:
        print("No active items available for optimization")
        return
    
    # Set budget
    budget = sum(item.current_bid for item in items) * 0.7  # 70% of total cost
    
    print(f"Optimizing {len(items)} items within budget ${budget:.2f}")
    
    # Run optimization
    combinations = optimizer.optimize(items, budget)
    
    if not combinations:
        print("No valid combinations found")
        return
    
    print(f"Generated {len(combinations)} optimized combinations:")
    
    for i, combo in enumerate(combinations[:3]):  # Show top 3
        print(f"\nCombination {i+1}:")
        print(f"  Items: {len(combo.items)}")
        print(f"  Total Cost: ${combo.total_cost:.2f}")
        print(f"  Total Profit: ${combo.total_profit:.2f}")
        print(f"  ROI: {(combo.total_profit / combo.total_cost) * 100:.1f}%")
        print(f"  Average Risk: {combo.average_risk:.2f}")
        print(f"  Budget Utilization: {combo.budget_utilization:.1f}%")
        
        print("  Items:")
        for item in combo.items:
            print(f"    - {item.title}: ${item.current_bid:.2f} (Profit: ${item.estimated_profit:.2f})")


async def demonstrate_fraud_detection(fraud_detector):
    """Demonstrate fraud detection."""
    print("\n=== Fraud Detection ===")
    
    # Get alerts
    alerts = await fraud_detector.get_alerts()
    
    if not alerts:
        print("No fraud alerts detected")
        return
    
    print(f"Detected {len(alerts)} fraud alerts:")
    
    for alert in alerts:
        print(f"\nAlert: {alert.description}")
        print(f"  Item: {alert.item_id}")
        print(f"  Type: {alert.alert_type}")
        print(f"  Severity: {alert.severity}")
        print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}")
        
        if alert.additional_info:
            print("  Additional Info:")
            for key, value in alert.additional_info.items():
                if isinstance(value, dict):
                    continue  # Skip nested dictionaries for brevity
                print(f"    {key}: {value}")


async def main():
    """Main function."""
    print("Bid Intelligence Core Example")
    
    # Create Kafka simulator
    kafka_client = KafkaSimulator()
    
    # Create bid monitor
    bid_monitor = BidMonitor(kafka_client=kafka_client)
    
    # Create bid adjuster
    bid_adjuster = BidAdjuster(bid_monitor=bid_monitor)
    
    # Create knapsack optimizer
    optimizer = KnapsackOptimizer()
    
    # Create fraud detector
    fraud_detector = FraudDetector(bid_monitor=bid_monitor)
    
    # Register fraud alert callback
    fraud_detector.register_alert_callback(on_fraud_alert)
    
    # Start bid monitor
    await bid_monitor.start()
    
    # Simulate auction data
    await simulate_auction_data(kafka_client)
    
    # Wait for data to be processed
    await asyncio.sleep(2)
    
    # Demonstrate bid adjustments
    await demonstrate_bid_adjustments(bid_adjuster)
    
    # Demonstrate knapsack optimization
    await demonstrate_knapsack_optimization(optimizer, bid_monitor)
    
    # Demonstrate fraud detection
    await demonstrate_fraud_detection(fraud_detector)
    
    # Stop bid monitor
    await bid_monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())