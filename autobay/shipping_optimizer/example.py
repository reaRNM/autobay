"""
Example usage of the Shipping Optimization Module.

This script demonstrates how to use the Shipping Optimization Module
to retrieve shipping rates, select the best carrier, and optimize shipping costs.
"""

import os
import json
import logging
from datetime import datetime

from shipping_optimizer import (
    ShippingOptimizer,
    Package,
    Address,
    ShippingPreference
)
from shipping_optimizer.utils import setup_logging


# Set up logging
logger = setup_logging(log_level="INFO")


def main():
    """Main function."""
    logger.info("Starting Shipping Optimization example")
    
    # Initialize shipping optimizer
    optimizer = ShippingOptimizer()
    
    # Create a sample package
    package = Package(
        weight_oz=32.0,  # 2 pounds
        length_in=12.0,
        width_in=8.0,
        height_in=6.0,
        value=99.99,
        description="Sample product",
        is_fragile=False,
        requires_signature=False,
        is_hazardous=False
    )
    
    # Create origin and destination addresses
    origin = Address(
        street1="123 Main St",
        city="San Francisco",
        state="CA",
        postal_code="94105",
        country="US",
        residential=False
    )
    
    destination = Address(
        street1="456 Market St",
        city="New York",
        state="NY",
        postal_code="10001",
        country="US",
        residential=True
    )
    
    # Get shipping rates
    logger.info("Getting shipping rates...")
    rates = optimizer.get_shipping_rates(
        package=package,
        origin=origin,
        destination=destination
    )
    
    logger.info(f"Retrieved {len(rates)} shipping rates")
    
    # Display rates
    logger.info("\nShipping Rates:")
    for i, rate in enumerate(rates):
        logger.info(f"{i+1}. {rate.carrier} {rate.service}: ${rate.total_cost:.2f}")
        logger.info(f"   Delivery Days: {rate.delivery_days}")
        logger.info(f"   Guaranteed: {rate.guaranteed}")
    
    # Create shipping preferences
    preferences = ShippingPreference(
        user_id="example_user",
        preferred_carriers=["USPS", "FedEx"],
        excluded_carriers=[],
        cost_importance=0.6,
        speed_importance=0.3,
        reliability_importance=0.1
    )
    
    # Select best shipping option
    logger.info("\nSelecting best shipping option...")
    options = optimizer.select_best_shipping(
        rates=rates,
        preferences=preferences,
        package=package
    )
    
    logger.info(f"Selected {len(options)} shipping options")
    
    # Display options
    logger.info("\nShipping Options:")
    for i, option in enumerate(options[:3]):  # Show top 3
        logger.info(f"{i+1}. {option.rate.carrier} {option.rate.service}: ${option.rate.total_cost:.2f}")
        logger.info(f"   Overall Score: {option.overall_score:.4f}")
        logger.info(f"   Cost Score: {option.cost_score:.4f}")
        logger.info(f"   Speed Score: {option.speed_score:.4f}")
        logger.info(f"   Reliability Score: {option.reliability_score:.4f}")
        logger.info(f"   Recommended: {option.is_recommended}")
    
    # Get recommended option
    recommended = next((o for o in options if o.is_recommended), None)
    
    if recommended:
        # Adjust price for shipping
        logger.info("\nAdjusting price for shipping...")
        item_price = 50.0
        shipping_cost = recommended.rate.total_cost
        
        adjustment = optimizer.adjust_price_for_shipping(
            item_price=item_price,
            shipping_cost=shipping_cost,
            handling_cost=2.0,
            target_margin=0.2
        )
        
        logger.info(f"Original Price: ${item_price:.2f}")
        logger.info(f"Shipping Cost: ${shipping_cost:.2f}")
        logger.info(f"Handling Cost: ${adjustment.handling_cost:.2f}")
        logger.info(f"Packaging Cost: ${adjustment.packaging_cost:.2f}")
        logger.info(f"Return Rate: {adjustment.return_rate:.1%}")
        logger.info(f"Target Margin: {adjustment.profit_margin:.1%}")
        logger.info(f"Adjusted Price: ${adjustment.adjusted_price:.2f}")
    
    # Predict shipping cost
    logger.info("\nPredicting shipping cost...")
    
    # Calculate distance and zone
    from shipping_optimizer.utils import calculate_distance, calculate_zone
    
    distance = calculate_distance(origin, destination)
    zone = calculate_zone(origin.postal_code, destination.postal_code)
    
    predicted_cost = optimizer.predict_shipping_cost(
        package=package,
        distance=distance,
        zone=zone,
        carrier="USPS",
        service="Priority"
    )
    
    logger.info(f"Predicted Cost for USPS Priority: ${predicted_cost:.2f}")
    
    # Analyze shipping history
    logger.info("\nAnalyzing shipping history...")
    
    analysis = optimizer.analyze_shipping_history(
        user_id="example_user",
        limit=100
    )
    
    logger.info(f"Total Shipments: {analysis.get('total_shipments', 0)}")
    logger.info(f"Total Cost: ${analysis.get('total_cost', 0):.2f}")
    logger.info(f"Average Cost: ${analysis.get('avg_cost', 0):.2f}")
    
    # Display optimization opportunities
    opportunities = analysis.get('optimization_opportunities', [])
    logger.info(f"\