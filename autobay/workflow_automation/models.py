"""
Data models for the Daily Workflow Automation Module.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


class TaskStatus(str, Enum):
    """Status of a task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


class NotificationType(str, Enum):
    """Type of notification."""
    DAILY_SUMMARY = "daily_summary"
    HIGH_PRIORITY_BID = "high_priority_bid"
    WORKFLOW_ERROR = "workflow_error"
    BID_SUCCESS = "bid_success"
    BID_FAILURE = "bid_failure"
    AUCTION_ENDING_SOON = "auction_ending_soon"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


class NotificationChannel(str, Enum):
    """Channel for sending notifications."""
    EMAIL = "email"
    SMS = "sms"
    TELEGRAM = "telegram"
    PUSH = "push"
    DASHBOARD = "dashboard"


class AuctionItem(BaseModel):
    """Auction item data model."""
    id: str
    title: str
    description: Optional[str] = None
    platform: str
    category: str
    url: str
    current_price: float
    estimated_value: Optional[float] = None
    end_time: datetime
    image_url: Optional[str] = None
    seller_id: Optional[str] = None
    seller_rating: Optional[float] = None
    condition: Optional[str] = None
    location: Optional[str] = None
    shipping_cost: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProfitCalculation(BaseModel):
    """Profit calculation data model."""
    item_id: str
    estimated_buy_price: float
    estimated_sell_price: float
    estimated_fees: float
    estimated_shipping: float
    estimated_profit: float
    estimated_roi: float
    confidence_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BidRecommendation(BaseModel):
    """Bid recommendation data model."""
    item_id: str
    auction_id: str
    recommended_bid: float
    max_bid: float
    confidence_score: float
    profit_potential: float
    roi_potential: float
    risk_score: float
    time_sensitivity: float  # 0-1 scale, 1 being most urgent
    requires_review: bool = False
    review_reason: Optional[str] = None
    bid_time: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """Result of a task execution."""
    task_id: str
    task_name: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowLog(BaseModel):
    """Log entry for workflow execution."""
    id: str
    workflow_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    level: str  # INFO, WARNING, ERROR, DEBUG
    message: str
    component: str
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowExecution(BaseModel):
    """Workflow execution data model."""
    id: str
    workflow_name: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    tasks: List[TaskResult] = Field(default_factory=list)
    logs: List[WorkflowLog] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Notification(BaseModel):
    """Notification data model."""
    id: str
    user_id: str
    type: NotificationType
    title: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    read: bool = False
    channel: NotificationChannel
    priority: int = 0  # 0-5, 5 being highest
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DailySummary(BaseModel):
    """Daily summary data model."""
    date: datetime
    total_items_scraped: int
    new_items_found: int
    items_processed: int
    bid_recommendations_generated: int
    bids_placed: int
    successful_bids: int
    total_potential_profit: float
    top_opportunities: List[BidRecommendation] = Field(default_factory=list)
    errors_encountered: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)