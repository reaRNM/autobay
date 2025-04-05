"""
Example usage of the AI Scoring & NLP Interface.

This script demonstrates how to use the AI Scoring & NLP Interface
to score auction items and process natural language queries.
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

from ai_auction import (
    AIScoringEngine,
    NLPInterface,
    ItemData,
    ScoringResult,
    QueryResult,
    setup_logging
)


# Configure logging
logger = setup_logging(log_level="INFO")


def create_sample_items() -> List[ItemData]:
    """Create sample auction items for demonstration."""
    logger.info("Creating sample auction items...")
    
    items = []
    categories = ["Electronics", "Collectibles", "Furniture", "Jewelry", "Art"]
    conditions = ["New", "Like New", "Good", "Fair", "Poor"]
    
    for i in range(1, 21):
        category = categories[i % len(categories)]
        condition = conditions[i % len(conditions)]
        
        # Calculate base values
        base_price = 50.0 + (i * 10)
        estimated_value = base_price * (1.5 + (i % 3) * 0.2)
        current_bid = base_price * (1.1 + (i % 5) * 0.05)
        estimated_profit = estimated_value - current_bid - (base_price * 0.1)  # Subtract fees
        profit_margin = (estimated_profit / current_bid) * 100
        
        # Create dimensions and weight based on category
        dimensions = None
        weight = None
        
        if category == "Electronics":
            dimensions = {"length": 30, "width": 20, "height": 10}
            weight = 2.0 + (i % 3)
        elif category == "Collectibles":
            dimensions = {"length": 15, "width": 10, "height": 5}
            weight = 0.5 + (i % 2)
        elif category == "Furniture":
            dimensions = {"length": 100, "width": 50, "height": 50}
            weight = 15.0 + (i % 5) * 2
        elif category == "Jewelry":
            dimensions = {"length": 5, "width": 5, "height": 2}
            weight = 0.1
        elif category == "Art":
            dimensions = {"length": 40, "width": 30, "height": 5}
            weight = 1.0 + (i % 3)
        
        # Calculate shipping cost based on weight and dimensions
        volume = dimensions["length"] * dimensions["width"] * dimensions["height"] / 1000  # in liters
        shipping_base = 5.0 + (weight * 2.0) + (volume * 0.5)
        estimated_shipping_cost = round(shipping_base, 2)
        
        # Create price history
        price_history = []
        num_bids = 3 + (i % 5)
        current_price = base_price
        
        for j in range(num_bids):
            bid_amount = current_price * (1.0 + (0.05 + (j * 0.02)))
            timestamp = datetime.now() - timedelta(hours=(num_bids - j) * 2)
            
            price_history.append({
                "amount": bid_amount,
                "timestamp": timestamp.isoformat(),
                "bidder_id": f"bidder-{j % 5 + 1}"
            })
            
            current_price = bid_amount
        
        # Create similar items sold
        similar_items_sold = []
        for j in range(3):
            sale_price = estimated_value * (0.9 + (j * 0.1))
            sale_date = datetime.now() - timedelta(days=j * 10 + 5)
            
            similar_items_sold.append({
                "item_id": f"similar-{i}-{j}",
                "sale_price": sale_price,
                "sale_date": sale_date.isoformat()
            })
        
        # Create existing scores for demonstration
        existing_scores = {
            "risk": 0.3 + (i % 5) * 0.1,
            "shipping_ease": 0.8 - (weight / 20),  # Normalize by max weight
            "trend": 0.5 + (i % 3) * 0.1
        }
        
        # Create item
        item = ItemData(
            item_id=f"item-{i}",
            title=f"Sample {category} Item {i}",
            description=f"This is a sample {category.lower()} item in {condition.lower()} condition.",
            category=category,
            condition=condition,
            auction_id=f"auction-{i}",
            auction_end_time=datetime.now() + timedelta(days=1 + (i % 7)),
            starting_bid=base_price,
            current_bid=current_bid,
            bid_count=num_bids,
            estimated_value=estimated_value,
            estimated_profit=estimated_profit,
            profit_margin=profit_margin,
            weight=weight,
            dimensions=dimensions,
            estimated_shipping_cost=estimated_shipping_cost,
            seller_id=f"seller-{i % 5 + 1}",
            seller_rating=0.7 + (i % 4) * 0.1,
            seller_feedback_count=50 + (i * 10),
            similar_items_sold=similar_items_sold,
            price_history=price_history,
            tags=[category.lower(), condition.lower(), f"tag-{i % 5 + 1}"],
            existing_scores=existing_scores
        )
        
        items.append(item)
    
    logger.info(f"Created {len(items)} sample items")
    return items


def demonstrate_scoring(scoring_engine: AIScoringEngine, items: List[ItemData]) -> None:
    """Demonstrate AI scoring functionality."""
    logger.info("\n=== AI Scoring Demonstration ===")
    
    # Score all items
    logger.info("Scoring all items...")
    results = scoring_engine.score_items(items)
    
    # Display top 5 items by priority score
    logger.info("\nTop 5 items by priority score:")
    for i, result in enumerate(results[:5]):
        item = next((item for item in items if item.item_id == result.item_id), None)
        if item:
            logger.info(f"\n{i+1}. {item.title} (ID: {item.item_id})")
            logger.info(f"   Current Bid: ${item.current_bid:.2f}")
            logger.info(f"   Estimated Value: ${item.estimated_value:.2f}")
            logger.info(f"   Estimated Profit: ${item.estimated_profit:.2f} ({item.profit_margin:.1f}%)")
            logger.info(f"   Priority Score: {result.priority_score:.4f}")
            
            # Display component scores
            logger.info("   Component Scores:")
            for component in result.components:
                logger.info(f"     - {component.name}: {component.score:.4f} (weight: {component.weight:.2f})")
                logger.info(f"       {component.explanation}")
    
    # Demonstrate custom weights
    logger.info("\nScoring with custom weights (emphasizing profit):")
    custom_weights = {
        "profit_potential": 0.6,
        "risk_assessment": 0.2,
        "shipping_ease": 0.1,
        "trend_prediction": 0.1
    }
    
    custom_results = scoring_engine.score_items(items, weights=custom_weights)
    
    # Display top 3 items with custom weights
    logger.info("\nTop 3 items with custom weights:")
    for i, result in enumerate(custom_results[:3]):
        item = next((item for item in items if item.item_id == result.item_id), None)
        if item:
            logger.info(f"\n{i+1}. {item.title} (ID: {item.item_id})")
            logger.info(f"   Priority Score: {result.priority_score:.4f}")
            logger.info(f"   Profit Potential: {next((c.score for c in result.components if c.name == 'profit_potential'), 0):.4f}")


def demonstrate_nlp_interface(nlp_interface: NLPInterface, items: List[ItemData]) -> None:
    """Demonstrate NLP interface functionality."""
    logger.info("\n=== NLP Interface Demonstration ===")
    
    # Define example queries
    example_queries = [
        "Show me the top 5 high-margin items under $200",
        "Find items with low risk and high shipping ease",
        "What are the current top-ranked auctions?",
        "Show me electronics with profit over $50",
        "Calculate profit for collectibles",
        "Estimate shipping for furniture items",
        "Analyze risk for jewelry items",
        "Compare the top 3 art items",
        "What are the trending categories?"
    ]
    
    # Process each query
    for query in example_queries:
        logger.info(f"\nQuery: \"{query}\"")
        
        result = nlp_interface.process_query(query, items)
        
        logger.info(f"Intent: {result.parsed_intent}")
        logger.info(f"Entities: {json.dumps(result.entities, default=str)}")
        logger.info(f"Found {result.total_items} matching items")
        logger.info(f"Execution time: {result.execution_time:.4f} seconds")
        
        # Display first few items
        if result.items:
            logger.info("\nTop results:")
            for i, item in enumerate(result.items[:3]):
                logger.info(f"{i+1}. {item.get('title', 'Unknown')} - ${item.get('current_bid', 0):.2f}")
                
                # Display scores if available
                if 'priority_score' in item:
                    logger.info(f"   Priority Score: {item['priority_score']:.4f}")
                
                # Display components if available
                if 'components' in item:
                    for comp in item['components']:
                        logger.info(f"   {comp['name']}: {comp['score']:.4f} - {comp['explanation']}")


def main():
    """Main function."""
    logger.info("Starting AI Scoring & NLP Interface example")
    
    # Create sample items
    items = create_sample_items()
    
    # Create AI Scoring Engine
    scoring_engine = AIScoringEngine(model_dir="models")
    
    # Create NLP Interface
    nlp_interface = NLPInterface(scoring_engine=scoring_engine)
    
    # Demonstrate scoring
    demonstrate_scoring(scoring_engine, items)
    
    # Demonstrate NLP interface
    demonstrate_nlp_interface(nlp_interface, items)
    
    logger.info("\nExample completed")


if __name__ == "__main__":
    main()