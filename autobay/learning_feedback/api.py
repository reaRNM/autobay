"""
API module for the Learning & Feedback Systems Module.

This module provides a REST API for interacting with the
Learning & Feedback Systems Module.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from learning_feedback.core import LearningSystem
from learning_feedback.models import (
    AuctionStatus, FeedbackType, FeedbackSentiment
)
from learning_feedback.utils import setup_logging


logger = logging.getLogger(__name__)


# Pydantic models for API
class AuctionOutcomeCreate(BaseModel):
    """Model for creating an auction outcome."""
    
    item_id: str
    user_id: str
    platform: str
    category_id: str
    listing_id: Optional[str] = None
    estimated_price: float = 0.0
    start_price: float = 0.0
    reserve_price: Optional[float] = None
    final_price: Optional[float] = None
    shipping_cost: Optional[float] = None
    fees: Optional[float] = None
    profit: Optional[float] = None
    roi: Optional[float] = None
    views: int = 0
    watchers: int = 0
    questions: int = 0
    bids: int = 0
    status: str = "active"
    time_to_sale: Optional[int] = None
    listing_quality_score: Optional[float] = None
    ai_confidence_score: Optional[float] = None
    listing_date: Optional[str] = None
    end_date: Optional[str] = None


class UserFeedbackCreate(BaseModel):
    """Model for creating user feedback."""
    
    user_id: str
    item_id: Optional[str] = None
    auction_id: Optional[str] = None
    feedback_type: str = "general"
    sentiment: Optional[str] = None
    rating: Optional[int] = None
    comment: Optional[str] = None
    ai_suggestion: Optional[str] = None
    user_correction: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AdjustmentRequest(BaseModel):
    """Model for requesting adjustments."""
    
    user_id: str
    category_id: Optional[str] = None
    item_id: Optional[str] = None


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FastAPI application
    """
    # Set up logging
    setup_logging()
    
    # Create FastAPI app
    app = FastAPI(
        title="Learning & Feedback Systems API",
        description="API for the Learning & Feedback Systems Module",
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
    
    # Create LearningSystem
    learning_system = LearningSystem(config_path)
    
    # Dependency to get LearningSystem
    async def get_learning_system() -> LearningSystem:
        return learning_system
    
    @app.post("/api/auction-outcomes", tags=["Auction Outcomes"])
    async def create_auction_outcome(
        data: AuctionOutcomeCreate,
        learning_system: LearningSystem = Depends(get_learning_system)
    ):
        """
        Create or update an auction outcome.
        
        Args:
            data: Auction outcome data
            learning_system: LearningSystem instance
            
        Returns:
            Created or updated auction outcome
        """
        try:
            # Convert data to dictionary
            auction_data = data.dict()
            
            # Convert string dates to datetime
            if auction_data.get('listing_date'):
                auction_data['listing_date'] = datetime.fromisoformat(auction_data['listing_date'])
            
            if auction_data.get('end_date'):
                auction_data['end_date'] = datetime.fromisoformat(auction_data['end_date'])
            
            # Convert status string to enum
            if auction_data.get('status'):
                auction_data['status'] = AuctionStatus(auction_data['status'])
            
            # Create or update auction outcome
            auction = await learning_system.update_auction_outcome(auction_data)
            
            return auction.to_dict()
        
        except Exception as e:
            logger.error(f"Error creating auction outcome: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/user-feedback", tags=["User Feedback"])
    async def create_user_feedback(
        data: UserFeedbackCreate,
        learning_system: LearningSystem = Depends(get_learning_system)
    ):
        """
        Create user feedback.
        
        Args:
            data: User feedback data
            learning_system: LearningSystem instance
            
        Returns:
            Created user feedback
        """
        try:
            # Convert data to dictionary
            feedback_data = data.dict()
            
            # Convert feedback_type string to enum
            if feedback_data.get('feedback_type'):
                feedback_data['feedback_type'] = FeedbackType(feedback_data['feedback_type'])
            
            # Convert sentiment string to enum if provided
            if feedback_data.get('sentiment'):
                feedback_data['sentiment'] = FeedbackSentiment(feedback_data['sentiment'])
            
            # Create user feedback
            feedback = await learning_system.track_user_feedback(feedback_data)
            
            return feedback.to_dict()
        
        except Exception as e:
            logger.error(f"Error creating user feedback: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/recommend-adjustments", tags=["Adjustments"])
    async def recommend_adjustments(
        data: AdjustmentRequest,
        learning_system: LearningSystem = Depends(get_learning_system)
    ):
        """
        Recommend adjustments based on past performance.
        
        Args:
            data: Adjustment request data
            learning_system: LearningSystem instance
            
        Returns:
            Recommended adjustments
        """
        try:
            # Get recommendations
            recommendations = await learning_system.recommend_adjustments(
                user_id=data.user_id,
                category_id=data.category_id,
                item_id=data.item_id
            )
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error recommending adjustments: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/model-performance/{model_type}", tags=["Model Performance"])
    async def get_model_performance(
        model_type: str = Path(..., description="Type of model (pricing, bidding, shipping)"),
        category_id: Optional[str] = Query(None, description="Category ID"),
        learning_system: LearningSystem = Depends(get_learning_system)
    ):
        """
        Get performance metrics for a model.
        
        Args:
            model_type: Type of model (pricing, bidding, shipping)
            category_id: Category ID (optional)
            learning_system: LearningSystem instance
            
        Returns:
            Performance metrics
        """
        try:
            # Get model performance
            performance = await learning_system.get_model_performance(
                model_type=model_type,
                category_id=category_id
            )
            
            # Convert to dictionaries
            return [p.to_dict() for p in performance]
        
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/user-preferences/{user_id}", tags=["User Preferences"])
    async def get_user_preferences(
        user_id: str = Path(..., description="User ID"),
        learning_system: LearningSystem = Depends(get_learning_system)
    ):
        """
        Get preferences for a user.
        
        Args:
            user_id: User ID
            learning_system: LearningSystem instance
            
        Returns:
            User preferences
        """
        try:
            # Get user preferences
            preferences = await learning_system.get_user_preferences(user_id)
            
            return preferences
        
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app