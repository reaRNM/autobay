"""
Example usage of the Daily Workflow Automation Module.

This script demonstrates how to use the Daily Workflow Automation Module
to automate the daily workflow of an auction research and resale tool.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from workflow_automation import WorkflowManager
from workflow_automation.utils import setup_logging


# Set up logging
logger = setup_logging(log_level="INFO")


async def main():
    """Main function."""
    logger.info("Starting Daily Workflow Automation example")
    
    # Initialize workflow manager
    workflow_manager = WorkflowManager()
    
    # Example 1: Run daily full workflow
    logger.info("\nExample 1: Run daily full workflow")
    
    # Run workflow
    execution = await workflow_manager.run_workflow("daily_full")
    
    logger.info(f"Workflow execution ID: {execution.id}")
    logger.info(f"Workflow status: {execution.status}")
    logger.info(f"Workflow start time: {execution.start_time}")
    logger.info(f"Workflow end time: {execution.end_time}")
    
    # Example 2: Get workflow status
    logger.info("\nExample 2: Get workflow status")
    
    # Get workflow status
    status = await workflow_manager.get_workflow_status(execution.id)
    
    logger.info(f"Workflow status: {status.status}")
    logger.info(f"Tasks completed: {len([t for t in status.tasks if t.status == 'COMPLETED'])}")
    logger.info(f"Tasks failed: {len([t for t in status.tasks if t.status == 'FAILED'])}")
    
    # Example 3: Get workflow logs
    logger.info("\nExample 3: Get workflow logs")
    
    # Get workflow logs
    logs = await workflow_manager.get_workflow_logs(execution.id)
    
    logger.info(f"Workflow logs: {len(logs)}")
    for i, log in enumerate(logs[:5]):  # Show first 5 logs
        logger.info(f"Log {i+1}: {log.level} - {log.message}")
    
    # Example 4: Get bid recommendations
    logger.info("\nExample 4: Get bid recommendations")
    
    # Get bid recommendations
    recommendations = await workflow_manager.get_bid_recommendations(
        limit=5,
        min_confidence=0.7,
        min_profit=20.0
    )
    
    logger.info(f"Bid recommendations: {len(recommendations)}")
    for i, rec in enumerate(recommendations):
        logger.info(f"Recommendation {i+1}: {rec.item_id} - ${rec.recommended_bid:.2f} (profit: ${rec.profit_potential:.2f})")
    
    # Example 5: Get daily summary
    logger.info("\nExample 5: Get daily summary")
    
    # Get daily summary
    summary = await workflow_manager.get_daily_summary()
    
    if summary:
        logger.info(f"Daily summary: {summary.date}")
        logger.info(f"Total items scraped: {summary.total_items_scraped}")
        logger.info(f"New items found: {summary.new_items_found}")
        logger.info(f"Bid recommendations generated: {summary.bid_recommendations_generated}")
        logger.info(f"Total potential profit: ${summary.total_potential_profit:.2f}")
    else:
        logger.info("No daily summary available")
    
    # Example 6: Schedule workflow
    logger.info("\nExample 6: Schedule workflow")
    
    # Schedule workflow to run daily at 1:00 AM
    job_id = workflow_manager.schedule_workflow("daily_full", "0 1 * * *")
    
    logger.info(f"Scheduled workflow job ID: {job_id}")
    
    # Example 7: Start API server
    logger.info("\nExample 7: Start API server")
    logger.info("To start the API server, run:")
    logger.info("from workflow_automation.api import create_app")
    logger.info("import uvicorn")
    logger.info("app = create_app()")
    logger.info("uvicorn.run(app, host='0.0.0.0', port=8000)")
    
    # Shutdown workflow manager
    workflow_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())