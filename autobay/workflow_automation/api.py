"""
API module for the workflow automation system.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from workflow_automation.core import WorkflowManager
from workflow_automation.models import (
    WorkflowStatus,
    TaskStatus,
    WorkflowExecution,
    TaskResult,
    WorkflowLog,
    BidRecommendation,
    DailySummary
)
from workflow_automation.utils import setup_logging


logger = setup_logging()


# API models
class WorkflowRequest(BaseModel):
    """Workflow request model."""
    workflow_id: str


class ScheduleWorkflowRequest(BaseModel):
    """Schedule workflow request model."""
    workflow_id: str
    cron_expression: str


class ApiResponse(BaseModel):
    """API response model."""
    success: bool
    message: str
    data: Optional[Any] = None


# Create FastAPI app
app = FastAPI(
    title="Workflow Automation API",
    description="API for the workflow automation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Workflow manager instance
workflow_manager = None


def get_workflow_manager():
    """Get the workflow manager instance."""
    global workflow_manager
    if workflow_manager is None:
        workflow_manager = WorkflowManager()
    return workflow_manager


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    global workflow_manager
    workflow_manager = WorkflowManager()
    logger.info("API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    global workflow_manager
    if workflow_manager:
        workflow_manager.shutdown()
    logger.info("API shutdown")


@app.post("/run_daily_workflow", response_model=ApiResponse)
async def run_daily_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    wm: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Run a workflow.
    
    Args:
        request: Workflow request
        background_tasks: Background tasks
        wm: Workflow manager
    
    Returns:
        API response
    """
    try:
        # Check if workflow exists
        if request.workflow_id not in wm.workflows:
            raise HTTPException(status_code=404, detail=f"Workflow {request.workflow_id} not found")
        
        # Run workflow in background
        background_tasks.add_task(wm.run_workflow, request.workflow_id)
        
        return ApiResponse(
            success=True,
            message=f"Workflow {request.workflow_id} started",
            data={"workflow_id": request.workflow_id}
        )
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/schedule_workflow", response_model=ApiResponse)
async def schedule_workflow(
    request: ScheduleWorkflowRequest,
    wm: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Schedule a workflow.
    
    Args:
        request: Schedule workflow request
        wm: Workflow manager
    
    Returns:
        API response
    """
    try:
        # Check if workflow exists
        if request.workflow_id not in wm.workflows:
            raise HTTPException(status_code=404, detail=f"Workflow {request.workflow_id} not found")
        
        # Schedule workflow
        job_id = wm.schedule_workflow(request.workflow_id, request.cron_expression)
        
        return ApiResponse(
            success=True,
            message=f"Workflow {request.workflow_id} scheduled",
            data={"job_id": job_id}
        )
    except Exception as e:
        logger.error(f"Error scheduling workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_workflow_status/{execution_id}", response_model=ApiResponse)
async def get_workflow_status(
    execution_id: str,
    wm: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Get workflow status.
    
    Args:
        execution_id: Workflow execution ID
        wm: Workflow manager
    
    Returns:
        API response
    """
    try:
        # Get workflow status
        execution = await wm.get_workflow_status(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail=f"Workflow execution {execution_id} not found")
        
        return ApiResponse(
            success=True,
            message=f"Workflow execution {execution_id} status",
            data=execution.dict()
        )
    except Exception as e:
        logger.error(f"Error getting workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch_logs/{execution_id}", response_model=ApiResponse)
async def fetch_logs(
    execution_id: str,
    wm: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Fetch workflow logs.
    
    Args:
        execution_id: Workflow execution ID
        wm: Workflow manager
    
    Returns:
        API response
    """
    try:
        # Get workflow logs
        logs = await wm.get_workflow_logs(execution_id)
        
        return ApiResponse(
            success=True,
            message=f"Workflow execution {execution_id} logs",
            data=[log.dict() for log in logs]
        )
    except Exception as e:
        logger.error(f"Error fetching workflow logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch_recommendations", response_model=ApiResponse)
async def fetch_recommendations(
    limit: int = Query(10, ge=1, le=100),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    min_profit: float = Query(0.0, ge=0.0),
    require_review: Optional[bool] = Query(None),
    wm: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Fetch bid recommendations.
    
    Args:
        limit: Maximum number of recommendations to return
        min_confidence: Minimum confidence score
        min_profit: Minimum profit potential
        require_review: Filter by requires_review flag
        wm: Workflow manager
    
    Returns:
        API response
    """
    try:
        # Get bid recommendations
        recommendations = await wm.get_bid_recommendations(
            limit=limit,
            min_confidence=min_confidence,
            min_profit=min_profit,
            require_review=require_review
        )
        
        return ApiResponse(
            success=True,
            message=f"Fetched {len(recommendations)} bid recommendations",
            data=[rec.dict() for rec in recommendations]
        )
    except Exception as e:
        logger.error(f"Error fetching bid recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_daily_summary", response_model=ApiResponse)
async def get_daily_summary(
    date: Optional[str] = None,
    wm: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Get daily summary.
    
    Args:
        date: Date to get summary for (YYYY-MM-DD)
        wm: Workflow manager
    
    Returns:
        API response
    """
    try:
        # Parse date
        if date:
            summary_date = datetime.fromisoformat(date)
        else:
            summary_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get daily summary
        summary = await wm.get_daily_summary(summary_date)
        
        if not summary:
            return ApiResponse(
                success=True,
                message=f"No daily summary found for {summary_date.strftime('%Y-%m-%d')}",
                data=None
            )
        
        return ApiResponse(
            success=True,
            message=f"Daily summary for {summary_date.strftime('%Y-%m-%d')}",
            data=summary.dict()
        )
    except Exception as e:
        logger.error(f"Error getting daily summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def create_app():
    """Create the FastAPI app."""
    return app