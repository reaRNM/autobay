"""
Core workflow management functionality.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore

from workflow_automation.models import (
    WorkflowStatus,
    TaskStatus,
    WorkflowExecution,
    TaskResult,
    WorkflowLog,
    BidRecommendation,
    DailySummary
)
from workflow_automation.db import Database
from workflow_automation.tasks import (
    data_collection_tasks,
    processing_tasks,
    bid_management_tasks,
    notification_tasks
)
from workflow_automation.utils import setup_logging


logger = setup_logging()


class WorkflowManager:
    """
    Core workflow manager for automating daily tasks.
    
    This class manages the scheduling and execution of workflow tasks,
    handles dependencies between tasks, and provides status tracking.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the workflow manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.db = Database(self.config.get("database", {}))
        self.scheduler = AsyncIOScheduler()
        
        # Add job stores
        self.scheduler.add_jobstore(MemoryJobStore(), 'default')
        
        # Initialize task registry
        self.tasks = {}
        self._register_tasks()
        
        # Initialize workflow registry
        self.workflows = {}
        self._register_workflows()
        
        # Track current workflow executions
        self.current_executions = {}
        
        # Start the scheduler
        self.scheduler.start()
        logger.info("Workflow manager initialized")
    
    def _register_tasks(self):
        """Register all available tasks."""
        # Data collection tasks
        self.tasks.update(data_collection_tasks.get_tasks())
        
        # Processing tasks
        self.tasks.update(processing_tasks.get_tasks())
        
        # Bid management tasks
        self.tasks.update(bid_management_tasks.get_tasks())
        
        # Notification tasks
        self.tasks.update(notification_tasks.get_tasks())
        
        logger.info(f"Registered {len(self.tasks)} tasks")
    
    def _register_workflows(self):
        """Register all available workflows."""
        # Daily full workflow
        self.workflows["daily_full"] = {
            "name": "Daily Full Workflow",
            "description": "Complete daily workflow including data collection, processing, and bid management",
            "tasks": [
                {"task": "scrape_hibid", "dependencies": []},
                {"task": "scrape_amazon", "dependencies": []},
                {"task": "scrape_ebay", "dependencies": []},
                {"task": "deduplicate_data", "dependencies": ["scrape_hibid", "scrape_amazon", "scrape_ebay"]},
                {"task": "validate_data", "dependencies": ["deduplicate_data"]},
                {"task": "calculate_profit", "dependencies": ["validate_data"]},
                {"task": "score_items", "dependencies": ["calculate_profit"]},
                {"task": "generate_bid_recommendations", "dependencies": ["score_items"]},
                {"task": "schedule_bids", "dependencies": ["generate_bid_recommendations"]},
                {"task": "send_daily_summary", "dependencies": ["schedule_bids"]}
            ]
        }
        
        # Data collection only workflow
        self.workflows["data_collection"] = {
            "name": "Data Collection Workflow",
            "description": "Collect data from all sources",
            "tasks": [
                {"task": "scrape_hibid", "dependencies": []},
                {"task": "scrape_amazon", "dependencies": []},
                {"task": "scrape_ebay", "dependencies": []},
                {"task": "deduplicate_data", "dependencies": ["scrape_hibid", "scrape_amazon", "scrape_ebay"]},
                {"task": "validate_data", "dependencies": ["deduplicate_data"]}
            ]
        }
        
        # Bid management only workflow
        self.workflows["bid_management"] = {
            "name": "Bid Management Workflow",
            "description": "Generate and schedule bids",
            "tasks": [
                {"task": "generate_bid_recommendations", "dependencies": []},
                {"task": "schedule_bids", "dependencies": ["generate_bid_recommendations"]}
            ]
        }
        
        # Notification only workflow
        self.workflows["notification"] = {
            "name": "Notification Workflow",
            "description": "Send notifications and summaries",
            "tasks": [
                {"task": "send_daily_summary", "dependencies": []}
            ]
        }
        
        logger.info(f"Registered {len(self.workflows)} workflows")
    
    def schedule_workflow(self, workflow_id: str, cron_expression: str):
        """
        Schedule a workflow to run on a cron schedule.
        
        Args:
            workflow_id: ID of the workflow to schedule
            cron_expression: Cron expression for scheduling
        
        Returns:
            Job ID
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        job = self.scheduler.add_job(
            self.run_workflow,
            CronTrigger.from_crontab(cron_expression),
            args=[workflow_id],
            id=f"{workflow_id}_scheduled",
            replace_existing=True
        )
        
        logger.info(f"Scheduled workflow {workflow_id} with cron expression {cron_expression}")
        return job.id
    
    async def run_workflow(self, workflow_id: str) -> WorkflowExecution:
        """
        Run a workflow by ID.
        
        Args:
            workflow_id: ID of the workflow to run
        
        Returns:
            Workflow execution result
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        execution_id = str(uuid.uuid4())
        
        # Create workflow execution record
        execution = WorkflowExecution(
            id=execution_id,
            workflow_name=workflow["name"],
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now(),
            tasks=[],
            logs=[]
        )
        
        # Store execution
        self.current_executions[execution_id] = execution
        await self.db.save_workflow_execution(execution)
        
        # Log workflow start
        log_entry = WorkflowLog(
            id=str(uuid.uuid4()),
            workflow_id=execution_id,
            level="INFO",
            message=f"Starting workflow {workflow['name']}",
            component="WorkflowManager"
        )
        execution.logs.append(log_entry)
        await self.db.save_workflow_log(log_entry)
        
        logger.info(f"Starting workflow {workflow_id} (execution {execution_id})")
        
        # Build dependency graph
        task_graph = {}
        for task_info in workflow["tasks"]:
            task_id = task_info["task"]
            task_graph[task_id] = {
                "dependencies": task_info["dependencies"],
                "status": TaskStatus.PENDING,
                "result": None
            }
        
        # Execute tasks respecting dependencies
        completed_tasks = set()
        failed_tasks = set()
        
        while len(completed_tasks) + len(failed_tasks) < len(task_graph):
            # Find tasks that can be executed (all dependencies satisfied)
            executable_tasks = []
            for task_id, task_info in task_graph.items():
                if task_id in completed_tasks or task_id in failed_tasks:
                    continue
                
                if task_info["status"] == TaskStatus.PENDING:
                    dependencies_met = all(
                        dep in completed_tasks for dep in task_info["dependencies"]
                    )
                    dependencies_failed = any(
                        dep in failed_tasks for dep in task_info["dependencies"]
                    )
                    
                    if dependencies_failed:
                        # Skip tasks with failed dependencies
                        task_info["status"] = TaskStatus.SKIPPED
                        task_result = TaskResult(
                            task_id=str(uuid.uuid4()),
                            task_name=task_id,
                            status=TaskStatus.SKIPPED,
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            error="Skipped due to failed dependencies"
                        )
                        execution.tasks.append(task_result)
                        await self.db.save_task_result(task_result)
                        failed_tasks.add(task_id)
                    elif dependencies_met:
                        executable_tasks.append(task_id)
            
            if not executable_tasks:
                # No tasks can be executed, but workflow is not complete
                # This indicates a circular dependency or all remaining tasks are blocked
                log_entry = WorkflowLog(
                    id=str(uuid.uuid4()),
                    workflow_id=execution_id,
                    level="ERROR",
                    message="Workflow deadlocked, no executable tasks but workflow not complete",
                    component="WorkflowManager"
                )
                execution.logs.append(log_entry)
                await self.db.save_workflow_log(log_entry)
                execution.status = WorkflowStatus.FAILED
                break
            
            # Execute tasks in parallel
            tasks = []
            for task_id in executable_tasks:
                task_graph[task_id]["status"] = TaskStatus.RUNNING
                tasks.append(self._execute_task(execution_id, task_id))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                task_id = executable_tasks[i]
                
                if isinstance(result, Exception):
                    # Task failed
                    task_graph[task_id]["status"] = TaskStatus.FAILED
                    task_graph[task_id]["result"] = str(result)
                    failed_tasks.add(task_id)
                    
                    log_entry = WorkflowLog(
                        id=str(uuid.uuid4()),
                        workflow_id=execution_id,
                        level="ERROR",
                        message=f"Task {task_id} failed: {str(result)}",
                        component="WorkflowManager"
                    )
                    execution.logs.append(log_entry)
                    await self.db.save_workflow_log(log_entry)
                else:
                    # Task succeeded
                    task_graph[task_id]["status"] = TaskStatus.COMPLETED
                    task_graph[task_id]["result"] = result
                    completed_tasks.add(task_id)
        
        # Determine workflow status
        if len(failed_tasks) == 0:
            execution.status = WorkflowStatus.COMPLETED
        elif len(completed_tasks) == 0:
            execution.status = WorkflowStatus.FAILED
        else:
            execution.status = WorkflowStatus.PARTIALLY_COMPLETED
        
        # Set end time
        execution.end_time = datetime.now()
        
        # Log workflow completion
        log_entry = WorkflowLog(
            id=str(uuid.uuid4()),
            workflow_id=execution_id,
            level="INFO",
            message=f"Workflow {workflow['name']} completed with status {execution.status}",
            component="WorkflowManager"
        )
        execution.logs.append(log_entry)
        await self.db.save_workflow_log(log_entry)
        
        # Update execution record
        await self.db.save_workflow_execution(execution)
        
        logger.info(f"Workflow {workflow_id} (execution {execution_id}) completed with status {execution.status}")
        
        # Clean up
        if execution_id in self.current_executions:
            del self.current_executions[execution_id]
        
        return execution
    
    async def _execute_task(self, execution_id: str, task_id: str) -> Any:
        """
        Execute a single task.
        
        Args:
            execution_id: Workflow execution ID
            task_id: Task ID to execute
        
        Returns:
            Task result
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task_func = self.tasks[task_id]
        task_result_id = str(uuid.uuid4())
        
        # Create task result record
        task_result = TaskResult(
            task_id=task_result_id,
            task_name=task_id,
            status=TaskStatus.RUNNING,
            start_time=datetime.now()
        )
        
        # Get execution
        execution = self.current_executions.get(execution_id)
        if execution:
            execution.tasks.append(task_result)
        
        # Save task result
        await self.db.save_task_result(task_result)
        
        # Log task start
        log_entry = WorkflowLog(
            id=str(uuid.uuid4()),
            workflow_id=execution_id,
            level="INFO",
            message=f"Starting task {task_id}",
            component="WorkflowManager",
            task_id=task_result_id
        )
        if execution:
            execution.logs.append(log_entry)
        await self.db.save_workflow_log(log_entry)
        
        logger.info(f"Starting task {task_id} (execution {execution_id}, task {task_result_id})")
        
        try:
            # Execute task
            result = await task_func(self.db, execution_id)
            
            # Update task result
            task_result.status = TaskStatus.COMPLETED
            task_result.end_time = datetime.now()
            task_result.result = result
            await self.db.save_task_result(task_result)
            
            # Log task completion
            log_entry = WorkflowLog(
                id=str(uuid.uuid4()),
                workflow_id=execution_id,
                level="INFO",
                message=f"Task {task_id} completed successfully",
                component="WorkflowManager",
                task_id=task_result_id
            )
            if execution:
                execution.logs.append(log_entry)
            await self.db.save_workflow_log(log_entry)
            
            logger.info(f"Task {task_id} (execution {execution_id}, task {task_result_id}) completed successfully")
            
            return result
        except Exception as e:
            # Update task result
            task_result.status = TaskStatus.FAILED
            task_result.end_time = datetime.now()
            task_result.error = str(e)
            await self.db.save_task_result(task_result)
            
            # Log task failure
            log_entry = WorkflowLog(
                id=str(uuid.uuid4()),
                workflow_id=execution_id,
                level="ERROR",
                message=f"Task {task_id} failed: {str(e)}",
                component="WorkflowManager",
                task_id=task_result_id
            )
            if execution:
                execution.logs.append(log_entry)
            await self.db.save_workflow_log(log_entry)
            
            logger.error(f"Task {task_id} (execution {execution_id}, task {task_result_id}) failed: {str(e)}")
            
            raise
    
    async def get_workflow_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """
        Get the status of a workflow execution.
        
        Args:
            execution_id: Workflow execution ID
        
        Returns:
            Workflow execution or None if not found
        """
        # Check current executions
        if execution_id in self.current_executions:
            return self.current_executions[execution_id]
        
        # Check database
        return await self.db.get_workflow_execution(execution_id)
    
    async def get_workflow_logs(self, execution_id: str) -> List[WorkflowLog]:
        """
        Get logs for a workflow execution.
        
        Args:
            execution_id: Workflow execution ID
        
        Returns:
            List of workflow logs
        """
        return await self.db.get_workflow_logs(execution_id)
    
    async def get_recent_executions(self, limit: int = 10) -> List[WorkflowExecution]:
        """
        Get recent workflow executions.
        
        Args:
            limit: Maximum number of executions to return
        
        Returns:
            List of workflow executions
        """
        return await self.db.get_recent_workflow_executions(limit)
    
    async def get_daily_summary(self, date: Optional[datetime] = None) -> Optional[DailySummary]:
        """
        Get the daily summary for a specific date.
        
        Args:
            date: Date to get summary for, defaults to today
        
        Returns:
            Daily summary or None if not found
        """
        if date is None:
            date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        return await self.db.get_daily_summary(date)
    
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
        return await self.db.get_bid_recommendations(
            limit=limit,
            min_confidence=min_confidence,
            min_profit=min_profit,
            require_review=require_review
        )
    
    def shutdown(self):
        """Shutdown the workflow manager."""
        self.scheduler.shutdown()
        logger.info("Workflow manager shutdown")