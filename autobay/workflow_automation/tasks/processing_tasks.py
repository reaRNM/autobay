"""
Processing tasks for the workflow automation module.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List

from workflow_automation.db import Database
from workflow_automation.models import AuctionItem, ProfitCalculation
from workflow_automation.utils import setup_logging

# Import from existing modules
from profit_calculator import ProfitCalculator
from ai_scoring import ItemScorer


logger = setup_logging()


async def calculate_profit(db: Database, execution_id: str) -> Dict[str, Any]:
    """
    Calculate profit for validated auction items.
    
    Args:
        db: Database instance
        execution_id: Workflow execution ID
    
    Returns:
        Dictionary with profit calculation results
    """
    logger.info("Starting profit calculation task")
    
    try:
        # Get validated items
        validated_items = await db.get_recent_auction_items(
            hours=24,
            status="validated"
        )
        
        logger.info(f"Found {len(validated_items)} validated items for profit calculation")
        
        # Initialize profit calculator
        calculator = ProfitCalculator()
        
        # Track calculation results
        calculations = 0
        profitable_items = 0
        total_potential_profit = 0.0
        
        # Process each item
        for item in validated_items:
            # Calculate profit
            profit_data = await calculator.calculate_profit(item)
            
            # Create profit calculation record
            calculation = ProfitCalculation(
                item_id=item.id,
                estimated_buy_price=profit_data["estimated_buy_price"],
                estimated_sell_price=profit_data["estimated_sell_price"],
                estimated_fees=profit_data["estimated_fees"],
                estimated_shipping=profit_data["estimated_shipping"],
                estimated_profit=profit_data["estimated_profit"],
                estimated_roi=profit_data["estimated_roi"],
                confidence_score=profit_data["confidence_score"],
                metadata={
                    "calculation_time": datetime.now().isoformat(),
                    "market_data": profit_data.get("market_data", {})
                }
            )
            
            # Save calculation
            await db.save_profit_calculation(calculation)
            calculations += 1
            
            # Update item status
            if profit_data["estimated_profit"] > 0:
                await db.update_auction_item_status(item.id, "profitable")
                profitable_items += 1
                total_potential_profit += profit_data["estimated_profit"]
            else:
                await db.update_auction_item_status(item.id, "unprofitable")
        
        return {
            "total_items": len(validated_items),
            "calculations": calculations,
            "profitable_items": profitable_items,
            "total_potential_profit": total_potential_profit,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error calculating profit: {str(e)}")
        raise


async def score_items(db: Database, execution_id: str) -> Dict[str, Any]:
    """
    Score profitable auction items.
    
    Args:
        db: Database instance
        execution_id: Workflow execution ID
    
    Returns:
        Dictionary with scoring results
    """
    logger.info("Starting item scoring task")
    
    try:
        # Get profitable items
        profitable_items = await db.get_recent_auction_items(
            hours=24,
            status="profitable"
        )
        
        logger.info(f"Found {len(profitable_items)} profitable items for scoring")
        
        # Initialize item scorer
        scorer = ItemScorer()
        
        # Track scoring results
        scored_items = 0
        high_score_items = 0
        
        # Process each item
        for item in profitable_items:
            # Get profit calculation
            profit_calc = await db.get_profit_calculation(item.id)
            
            if not profit_calc:
                logger.warning(f"No profit calculation found for item {item.id}")
                continue
            
            # Score item
            score_data = await scorer.score_item(item, profit_calc)
            
            # Update item with score
            await db.update_auction_item(
                item.id,
                metadata={
                    "ai_score": score_data["overall_score"],
                    "score_breakdown": score_data["score_breakdown"],
                    "score_time": datetime.now().isoformat()
                }
            )
            scored_items += 1
            
            # Track high score items
            if score_data["overall_score"] >= 0.7:  # 70% threshold for high score
                await db.update_auction_item_status(item.id, "high_score")
                high_score_items += 1
            else:
                await db.update_auction_item_status(item.id, "low_score")
        
        return {
            "total_items": len(profitable_items),
            "scored_items": scored_items,
            "high_score_items": high_score_items,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error scoring items: {str(e)}")
        raise


def get_tasks():
    """Get all processing tasks."""
    return {
        "calculate_profit": calculate_profit,
        "score_items": score_items
    }