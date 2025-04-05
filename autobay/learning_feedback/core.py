"""
Core functionality for the Learning & Feedback Systems Module.

This module provides the main LearningSystem class that handles
feedback collection, model training, and performance monitoring.
"""

import os
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta

from learning_feedback.models import (
    AuctionOutcome, UserFeedback, ModelPerformance, 
    BidStrategy, ShippingPerformance, ItemCategory,
    FeedbackType, FeedbackSentiment, AuctionStatus
)
from learning_feedback.data_collection import (
    AuctionOutcomeCollector, UserFeedbackCollector
)
from learning_feedback.model_training import (
    PricingModelTrainer, BidModelTrainer, ShippingModelTrainer
)
from learning_feedback.reinforcement_learning import (
    BiddingEnvironment, BiddingAgent
)
from learning_feedback.feedback_processing import (
    FeedbackProcessor, UserPreferenceTracker
)
from learning_feedback.performance_monitoring import (
    PerformanceMonitor, DriftDetector
)
from learning_feedback.db import Database
from learning_feedback.utils import setup_logging


logger = logging.getLogger(__name__)


class LearningSystem:
    """
    Main class for the Learning & Feedback Systems Module.
    
    This class provides methods for collecting feedback, training models,
    and monitoring performance.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        db_connection: Optional[str] = None
    ):
        """
        Initialize the LearningSystem.
        
        Args:
            config_path: Path to configuration file
            db_connection: Database connection string
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize database connection
        self.db = Database(db_connection or self.config.get('db_connection'))
        
        # Initialize components
        self.auction_collector = AuctionOutcomeCollector(self.db)
        self.feedback_collector = UserFeedbackCollector(self.db)
        self.pricing_trainer = PricingModelTrainer(self.config.get('pricing_model', {}))
        self.bid_trainer = BidModelTrainer(self.config.get('bid_model', {}))
        self.shipping_trainer = ShippingModelTrainer(self.config.get('shipping_model', {}))
        self.bidding_env = BiddingEnvironment()
        self.bidding_agent = BiddingAgent(self.config.get('bidding_agent', {}))
        self.feedback_processor = FeedbackProcessor(self.db)
        self.preference_tracker = UserPreferenceTracker(self.db)
        self.performance_monitor = PerformanceMonitor(self.db)
        self.drift_detector = DriftDetector()
        
        # Set up retraining schedule
        self.retraining_frequency = self.config.get('retraining_frequency_days', 30)
        self.last_retraining = self.config.get('last_retraining', None)
        
        logger.info("LearningSystem initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'db_connection': os.environ.get('LEARNING_DB_CONNECTION', 'sqlite:///learning.db'),
            'pricing_model': {
                'model_type': 'gradient_boosting',
                'features': ['category', 'condition', 'brand', 'model', 'age', 'specifications'],
                'target': 'final_price',
                'test_size': 0.2,
                'random_state': 42
            },
            'bid_model': {
                'model_type': 'random_forest',
                'features': ['category', 'condition', 'price_ratio', 'time_left', 'watchers', 'bids'],
                'target': 'success',
                'test_size': 0.2,
                'random_state': 42
            },
            'shipping_model': {
                'model_type': 'gradient_boosting',
                'features': ['weight', 'dimensions', 'distance', 'service_level'],
                'target': 'shipping_cost',
                'test_size': 0.2,
                'random_state': 42
            },
            'bidding_agent': {
                'learning_rate': 0.001,
                'discount_factor': 0.99,
                'exploration_rate': 0.1,
                'batch_size': 64,
                'memory_size': 10000
            },
            'retraining_frequency_days': 30,
            'min_samples_for_training': 100,
            'performance_threshold': 0.7,
            'drift_detection_threshold': 0.05
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with default config
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict) and isinstance(config[key], dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                
                logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                return default_config
        
        logger.info("Using default configuration")
        return default_config
    
    async def update_auction_outcome(self, auction_data: Dict[str, Any]) -> AuctionOutcome:
        """
        Update an auction outcome and trigger learning processes.
        
        Args:
            auction_data: Auction outcome data
            
        Returns:
            Updated AuctionOutcome object
        """
        logger.info(f"Updating auction outcome for item {auction_data.get('item_id')}")
        
        try:
            # Save auction outcome
            auction = await self.auction_collector.save_auction_outcome(auction_data)
            
            # Process auction outcome for learning
            if auction.status == AuctionStatus.SOLD or auction.status == AuctionStatus.UNSOLD:
                await self._process_auction_for_learning(auction)
            
            # Check if retraining is needed
            await self._check_retraining_schedule()
            
            return auction
        
        except Exception as e:
            logger.error(f"Error updating auction outcome: {e}")
            raise
    
    async def track_user_feedback(self, feedback_data: Dict[str, Any]) -> UserFeedback:
        """
        Track user feedback and update user preferences.
        
        Args:
            feedback_data: User feedback data
            
        Returns:
            UserFeedback object
        """
        logger.info(f"Tracking user feedback from user {feedback_data.get('user_id')}")
        
        try:
            # Save user feedback
            feedback = await self.feedback_collector.save_user_feedback(feedback_data)
            
            # Process feedback for learning
            await self._process_feedback_for_learning(feedback)
            
            # Update user preferences based on feedback
            if feedback.user_id:
                await self.preference_tracker.update_preferences(feedback)
            
            return feedback
        
        except Exception as e:
            logger.error(f"Error tracking user feedback: {e}")
            raise
    
    async def recommend_adjustments(
        self,
        user_id: str,
        category_id: Optional[str] = None,
        item_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Recommend adjustments based on past performance.
        
        Args:
            user_id: User ID
            category_id: Category ID (optional)
            item_id: Item ID (optional)
            
        Returns:
            Dictionary of recommended adjustments
        """
        logger.info(f"Generating adjustment recommendations for user {user_id}")
        
        try:
            recommendations = {
                'pricing_adjustments': [],
                'bidding_adjustments': [],
                'listing_adjustments': [],
                'shipping_adjustments': []
            }
            
            # Get user's auction outcomes
            auctions = await self.db.get_auction_outcomes(
                user_id=user_id,
                category_id=category_id,
                item_id=item_id,
                limit=100
            )
            
            # Get user's feedback
            feedback = await self.db.get_user_feedback(
                user_id=user_id,
                item_id=item_id,
                limit=100
            )
            
            # Get user's shipping performance
            shipping = await self.db.get_shipping_performance(
                user_id=user_id,
                limit=100
            )
            
            # Generate pricing adjustments
            pricing_adjustments = await self._generate_pricing_adjustments(auctions)
            if pricing_adjustments:
                recommendations['pricing_adjustments'] = pricing_adjustments
            
            # Generate bidding adjustments
            bidding_adjustments = await self._generate_bidding_adjustments(auctions)
            if bidding_adjustments:
                recommendations['bidding_adjustments'] = bidding_adjustments
            
            # Generate listing adjustments
            listing_adjustments = await self._generate_listing_adjustments(auctions, feedback)
            if listing_adjustments:
                recommendations['listing_adjustments'] = listing_adjustments
            
            # Generate shipping adjustments
            shipping_adjustments = await self._generate_shipping_adjustments(shipping)
            if shipping_adjustments:
                recommendations['shipping_adjustments'] = shipping_adjustments
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error generating adjustment recommendations: {e}")
            raise
    
    async def get_model_performance(
        self,
        model_type: str,
        category_id: Optional[str] = None
    ) -> List[ModelPerformance]:
        """
        Get performance metrics for a model.
        
        Args:
            model_type: Type of model (pricing, bidding, shipping)
            category_id: Category ID (optional)
            
        Returns:
            List of ModelPerformance objects
        """
        logger.info(f"Getting performance metrics for {model_type} model")
        
        try:
            # Get model performance from database
            performance = await self.db.get_model_performance(
                model_type=model_type,
                category_id=category_id,
                limit=10
            )
            
            return performance
        
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            raise
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get preferences for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary of user preferences
        """
        logger.info(f"Getting preferences for user {user_id}")
        
        try:
            # Get user preferences from database
            preferences = await self.preference_tracker.get_user_preferences(user_id)
            
            if not preferences:
                logger.warning(f"No preferences found for user {user_id}")
                return {}
            
            return preferences.to_dict()
        
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            raise
    
    async def _process_auction_for_learning(self, auction: AuctionOutcome) -> None:
        """
        Process an auction outcome for learning.
        
        Args:
            auction: Auction outcome
        """
        logger.debug(f"Processing auction {auction.id} for learning")
        
        try:
            # Update pricing model
            if auction.final_price is not None and auction.estimated_price > 0:
                # Calculate price accuracy
                accuracy = auction.price_accuracy
                
                # Log accuracy
                logger.info(f"Price accuracy for auction {auction.id}: {accuracy:.2f}%")
                
                # Update pricing model with new data
                await self.pricing_trainer.update_model_with_sample(
                    item_id=auction.item_id,
                    category_id=auction.category_id,
                    estimated_price=auction.estimated_price,
                    actual_price=auction.final_price,
                    accuracy=accuracy
                )
            
            # Update bidding model
            if auction.status == AuctionStatus.SOLD or auction.status == AuctionStatus.UNSOLD:
                # Update bidding model with new data
                await self.bid_trainer.update_model_with_sample(
                    item_id=auction.item_id,
                    category_id=auction.category_id,
                    success=auction.status == AuctionStatus.SOLD,
                    bids=auction.bids,
                    watchers=auction.watchers,
                    views=auction.views,
                    final_price=auction.final_price,
                    estimated_price=auction.estimated_price
                )
                
                # Update bidding agent with reinforcement learning
                if auction.bids > 0:
                    reward = auction.profit if auction.profit is not None else 0
                    
                    # Negative reward for unsuccessful auctions
                    if auction.status == AuctionStatus.UNSOLD:
                        reward = -10
                    
                    await self.bidding_agent.update_policy(
                        state={
                            'category_id': auction.category_id,
                            'estimated_price': auction.estimated_price,
                            'bids': auction.bids,
                            'watchers': auction.watchers,
                            'views': auction.views,
                            'time_left_percentage': 0  # Auction is complete
                        },
                        action={
                            'bid_amount': auction.final_price or 0,
                            'bid_time': 1.0  # End of auction
                        },
                        reward=reward,
                        done=True
                    )
            
            # Update performance metrics
            await self.performance_monitor.update_metrics(auction)
            
            # Check for model drift
            if auction.final_price is not None and auction.estimated_price > 0:
                drift_detected = self.drift_detector.check_for_drift(
                    model_type='pricing',
                    category_id=auction.category_id,
                    estimated_value=auction.estimated_price,
                    actual_value=auction.final_price
                )
                
                if drift_detected:
                    logger.warning(f"Drift detected in pricing model for category {auction.category_id}")
                    # Trigger retraining for this category
                    await self._retrain_model('pricing', auction.category_id)
        
        except Exception as e:
            logger.error(f"Error processing auction for learning: {e}")
    
    async def _process_feedback_for_learning(self, feedback: UserFeedback) -> None:
        """
        Process user feedback for learning.
        
        Args:
            feedback: User feedback
        """
        logger.debug(f"Processing feedback {feedback.id} for learning")
        
        try:
            # Process feedback based on type
            if feedback.feedback_type == FeedbackType.PRICING:
                # Update pricing model with feedback
                if feedback.ai_suggestion and feedback.user_correction:
                    await self.pricing_trainer.incorporate_feedback(
                        feedback.item_id,
                        float(feedback.ai_suggestion) if feedback.ai_suggestion.replace('.', '', 1).isdigit() else 0,
                        float(feedback.user_correction) if feedback.user_correction.replace('.', '', 1).isdigit() else 0,
                        feedback.rating or 3
                    )
            
            elif feedback.feedback_type == FeedbackType.BIDDING:
                # Update bidding model with feedback
                if feedback.ai_suggestion and feedback.user_correction:
                    await self.bid_trainer.incorporate_feedback(
                        feedback.item_id,
                        feedback.ai_suggestion,
                        feedback.user_correction,
                        feedback.rating or 3
                    )
            
            elif feedback.feedback_type == FeedbackType.SHIPPING:
                # Update shipping model with feedback
                if feedback.ai_suggestion and feedback.user_correction:
                    await self.shipping_trainer.incorporate_feedback(
                        feedback.item_id,
                        feedback.ai_suggestion,
                        feedback.user_correction,
                        feedback.rating or 3
                    )
            
            # Update feedback processor
            await self.feedback_processor.process_feedback(feedback)
        
        except Exception as e:
            logger.error(f"Error processing feedback for learning: {e}")
    
    async def _check_retraining_schedule(self) -> None:
        """Check if models need to be retrained based on schedule."""
        try:
            # Check if retraining is due
            if self.last_retraining is None:
                # First time, set last_retraining to now
                self.last_retraining = datetime.now()
                return
            
            # Convert last_retraining to datetime if it's a string
            if isinstance(self.last_retraining, str):
                self.last_retraining = datetime.fromisoformat(self.last_retraining)
            
            # Check if retraining frequency has elapsed
            days_since_retraining = (datetime.now() - self.last_retraining).days
            
            if days_since_retraining >= self.retraining_frequency:
                logger.info(f"Retraining scheduled (last retraining: {self.last_retraining.isoformat()})")
                
                # Retrain all models
                await self._retrain_all_models()
                
                # Update last_retraining
                self.last_retraining = datetime.now()
                
                # Save to config
                self.config['last_retraining'] = self.last_retraining.isoformat()
                
                logger.info(f"Retraining complete, next retraining scheduled for {self.retraining_frequency} days from now")
        
        except Exception as e:
            logger.error(f"Error checking retraining schedule: {e}")
    
    async def _retrain_all_models(self) -> None:
        """Retrain all models."""
        logger.info("Retraining all models")
        
        try:
            # Get all categories
            categories = await self.db.get_categories()
            
            # Retrain pricing models
            for category in categories:
                await self._retrain_model('pricing', category.id)
            
            # Retrain bidding models
            for category in categories:
                await self._retrain_model('bidding', category.id)
            
            # Retrain shipping model (global)
            await self._retrain_model('shipping')
            
            # Retrain bidding agent
            await self._retrain_bidding_agent()
        
        except Exception as e:
            logger.error(f"Error retraining all models: {e}")
    
    async def _retrain_model(self, model_type: str, category_id: Optional[str] = None) -> None:
        """
        Retrain a specific model.
        
        Args:
            model_type: Type of model to retrain
            category_id: Category ID (optional)
        """
        logger.info(f"Retraining {model_type} model for category {category_id or 'all'}")
        
        try:
            # Get minimum samples required for training
            min_samples = self.config.get('min_samples_for_training', 100)
            
            if model_type == 'pricing':
                # Get auction outcomes for training
                auctions = await self.db.get_auction_outcomes(
                    category_id=category_id,
                    status=AuctionStatus.SOLD,
                    limit=10000
                )
                
                # Check if we have enough samples
                if len(auctions) < min_samples:
                    logger.warning(f"Not enough samples to retrain pricing model for category {category_id or 'all'} ({len(auctions)} < {min_samples})")
                    return
                
                # Retrain pricing model
                performance = await self.pricing_trainer.train_model(auctions, category_id)
                
                # Save model performance
                await self.db.save_model_performance(performance)
                
                logger.info(f"Pricing model retrained for category {category_id or 'all'} with accuracy {performance.accuracy:.4f}")
            
            elif model_type == 'bidding':
                # Get auction outcomes for training
                auctions = await self.db.get_auction_outcomes(
                    category_id=category_id,
                    limit=10000
                )
                
                # Check if we have enough samples
                if len(auctions) < min_samples:
                    logger.warning(f"Not enough samples to retrain bidding model for category {category_id or 'all'} ({len(auctions)} < {min_samples})")
                    return
                
                # Retrain bidding model
                performance = await self.bid_trainer.train_model(auctions, category_id)
                
                # Save model performance
                await self.db.save_model_performance(performance)
                
                logger.info(f"Bidding model retrained for category {category_id or 'all'} with accuracy {performance.accuracy:.4f}")
            
            elif model_type == 'shipping':
                # Get shipping performance for training
                shipping_data = await self.db.get_shipping_performance(limit=10000)
                
                # Check if we have enough samples
                if len(shipping_data) < min_samples:
                    logger.warning(f"Not enough samples to retrain shipping model ({len(shipping_data)} < {min_samples})")
                    return
                
                # Retrain shipping model
                performance = await self.shipping_trainer.train_model(shipping_data)
                
                # Save model performance
                await self.db.save_model_performance(performance)
                
                logger.info(f"Shipping model retrained with accuracy {performance.accuracy:.4f}")
        
        except Exception as e:
            logger.error(f"Error retraining {model_type} model: {e}")
    
    async def _retrain_bidding_agent(self) -> None:
        """Retrain the bidding agent using reinforcement learning."""
        logger.info("Retraining bidding agent")
        
        try:
            # Get auction outcomes for training
            auctions = await self.db.get_auction_outcomes(
                status=AuctionStatus.SOLD,
                limit=10000
            )
            
            # Check if we have enough samples
            min_samples = self.config.get('min_samples_for_training', 100)
            if len(auctions) < min_samples:
                logger.warning(f"Not enough samples to retrain bidding agent ({len(auctions)} < {min_samples})")
                return
            
            # Retrain bidding agent
            performance = await self.bidding_agent.train(auctions, self.bidding_env)
            
            # Save model performance
            await self.db.save_model_performance(performance)
            
            logger.info(f"Bidding agent retrained with reward {performance.metadata.get('avg_reward', 0):.4f}")
        
        except Exception as e:
            logger.error(f"Error retraining bidding agent: {e}")
    
    async def _generate_pricing_adjustments(self, auctions: List[AuctionOutcome]) -> List[Dict[str, Any]]:
        """
        Generate pricing adjustment recommendations.
        
        Args:
            auctions: List of auction outcomes
            
        Returns:
            List of pricing adjustment recommendations
        """
        adjustments = []
        
        # Skip if no auctions
        if not auctions:
            return adjustments
        
        # Group auctions by category
        category_auctions = {}
        for auction in auctions:
            if auction.category_id not in category_auctions:
                category_auctions[auction.category_id] = []
            category_auctions[auction.category_id].append(auction)
        
        # Analyze each category
        for category_id, category_auctions_list in category_auctions.items():
            # Calculate average price accuracy
            accuracies = [a.price_accuracy for a in category_auctions_list if a.price_accuracy is not None]
            
            if not accuracies:
                continue
            
            avg_accuracy = sum(accuracies) / len(accuracies)
            
            # Check if accuracy is below threshold
            if avg_accuracy < 90:
                # Calculate average price difference
                price_diffs = [(a.final_price - a.estimated_price) / a.estimated_price if a.final_price and a.estimated_price else 0 for a in category_auctions_list]
                avg_diff = sum(price_diffs) / len(price_diffs) if price_diffs else 0
                
                # Generate adjustment recommendation
                adjustment = {
                    'category_id': category_id,
                    'avg_accuracy': avg_accuracy,
                    'avg_price_difference_percentage': avg_diff * 100,
                    'recommendation': 'increase' if avg_diff > 0 else 'decrease',
                    'adjustment_percentage': abs(avg_diff) * 100,
                    'confidence': min(100, max(0, avg_accuracy)),
                    'sample_size': len(category_auctions_list)
                }
                
                adjustments.append(adjustment)
        
        return adjustments
    
    async def _generate_bidding_adjustments(self, auctions: List[AuctionOutcome]) -> List[Dict[str, Any]]:
        """
        Generate bidding adjustment recommendations.
        
        Args:
            auctions: List of auction outcomes
            
        Returns:
            List of bidding adjustment recommendations
        """
        adjustments = []
        
        # Skip if no auctions
        if not auctions:
            return adjustments
        
        # Group auctions by category
        category_auctions = {}
        for auction in auctions:
            if auction.category_id not in category_auctions:
                category_auctions[auction.category_id] = []
            category_auctions[auction.category_id].append(auction)
        
        # Analyze each category
        for category_id, category_auctions_list in category_auctions.items():
            # Calculate success rate
            successes = [a for a in category_auctions_list if a.status == AuctionStatus.SOLD]
            success_rate = len(successes) / len(category_auctions_list) if category_auctions_list else 0
            
            # Calculate average ROI
            rois = [a.roi for a in successes if a.roi is not None]
            avg_roi = sum(rois) / len(rois) if rois else 0
            
            # Generate adjustment recommendation if success rate is low
            if success_rate < 0.7:
                # Analyze unsuccessful auctions
                unsuccessful = [a for a in category_auctions_list if a.status == AuctionStatus.UNSOLD]
                
                # Calculate average bid count
                bid_counts = [a.bids for a in unsuccessful]
                avg_bids = sum(bid_counts) / len(bid_counts) if bid_counts else 0
                
                # Generate recommendation
                if avg_bids < 1:
                    # No bids, likely too high starting price
                    adjustment = {
                        'category_id': category_id,
                        'success_rate': success_rate * 100,
                        'avg_roi': avg_roi * 100 if avg_roi else None,
                        'issue': 'low_bid_count',
                        'recommendation': 'lower_starting_price',
                        'adjustment_percentage': 10,
                        'confidence': 70,
                        'sample_size': len(category_auctions_list)
                    }
                else:
                    # Some bids, but not winning
                    adjustment = {
                        'category_id': category_id,
                        'success_rate': success_rate * 100,
                        'avg_roi': avg_roi * 100 if avg_roi else None,
                        'issue': 'not_winning',
                        'recommendation': 'increase_max_bid',
                        'adjustment_percentage': 5,
                        'confidence': 70,
                        'sample_size': len(category_auctions_list)
                    }
                
                adjustments.append(adjustment)
            
            # Generate adjustment recommendation if ROI is low
            elif avg_roi < 0.15 and success_rate >= 0.7:
                adjustment = {
                    'category_id': category_id,
                    'success_rate': success_rate * 100,
                    'avg_roi': avg_roi * 100,
                    'issue': 'low_roi',
                    'recommendation': 'be_more_selective',
                    'confidence': 80,
                    'sample_size': len(category_auctions_list)
                }
                
                adjustments.append(adjustment)
        
        return adjustments
    
    async def _generate_listing_adjustments(
        self,
        auctions: List[AuctionOutcome],
        feedback: List[UserFeedback]
    ) -> List[Dict[str, Any]]:
        """
        Generate listing adjustment recommendations.
        
        Args:
            auctions: List of auction outcomes
            feedback: List of user feedback
            
        Returns:
            List of listing adjustment recommendations
        """
        adjustments = []
        
        # Skip if no auctions
        if not auctions:
            return adjustments
        
        # Group auctions by category
        category_auctions = {}
        for auction in auctions:
            if auction.category_id not in category_auctions:
                category_auctions[auction.category_id] = []
            category_auctions[auction.category_id].append(auction)
        
        # Analyze each category
        for category_id, category_auctions_list in category_auctions.items():
            # Calculate average views and watchers
            views = [a.views for a in category_auctions_list if a.views is not None]
            watchers = [a.watchers for a in category_auctions_list if a.watchers is not None]
            
            avg_views = sum(views) / len(views) if views else 0
            avg_watchers = sum(watchers) / len(watchers) if watchers else 0
            
            # Calculate watcher-to-view ratio
            watcher_view_ratio = avg_watchers / avg_views if avg_views > 0 else 0
            
            # Generate adjustment recommendation if views are low
            if avg_views < 50:
                adjustment = {
                    'category_id': category_id,
                    'avg_views': avg_views,
                    'issue': 'low_visibility',
                    'recommendation': 'improve_keywords',
                    'confidence': 75,
                    'sample_size': len(category_auctions_list)
                }
                
                adjustments.append(adjustment)
            
            # Generate adjustment recommendation if watcher ratio is low
            elif watcher_view_ratio < 0.1 and avg_views >= 50:
                adjustment = {
                    'category_id': category_id,
                    'avg_views': avg_views,
                    'avg_watchers': avg_watchers,
                    'watcher_view_ratio': watcher_view_ratio,
                    'issue': 'low_interest',
                    'recommendation': 'improve_images_and_description',
                    'confidence': 70,
                    'sample_size': len(category_auctions_list)
                }
                
                adjustments.append(adjustment)
        
        # Analyze feedback for listing issues
        listing_feedback = [f for f in feedback if f.feedback_type == FeedbackType.LISTING]
        
        if listing_feedback:
            # Count negative feedback
            negative_feedback = [f for f in listing_feedback if f.sentiment == FeedbackSentiment.NEGATIVE]
            
            if negative_feedback:
                # Analyze feedback comments for common issues
                issues = {}
                for feedback in negative_feedback:
                    if feedback.comment:
                        # Simple keyword analysis
                        comment = feedback.comment.lower()
                        
                        if 'image' in comment or 'photo' in comment or 'picture' in comment:
                            issues['image_quality'] = issues.get('image_quality', 0) + 1
                        
                        if 'description' in comment or 'details' in comment or 'information' in comment:
                            issues['description_quality'] = issues.get('description_quality', 0) + 1
                        
                        if 'title' in comment or 'headline' in comment:
                            issues['title_quality'] = issues.get('title_quality', 0) + 1
                
                # Generate recommendations for common issues
                for issue, count in issues.items():
                    if count >= 2:  # At least 2 mentions
                        if issue == 'image_quality':
                            adjustment = {
                                'issue': 'poor_image_quality',
                                'recommendation': 'improve_images',
                                'details': 'Multiple users mentioned issues with image quality',
                                'confidence': 80,
                                'sample_size': count
                            }
                            adjustments.append(adjustment)
                        
                        elif issue == 'description_quality':
                            adjustment = {
                                'issue': 'poor_description_quality',
                                'recommendation': 'enhance_descriptions',
                                'details': 'Multiple users mentioned issues with description quality',
                                'confidence': 80,
                                'sample_size': count
                            }
                            adjustments.append(adjustment)
                        
                        elif issue == 'title_quality':
                            adjustment = {
                                'issue': 'poor_title_quality',
                                'recommendation': 'optimize_titles',
                                'details': 'Multiple users mentioned issues with title quality',
                                'confidence': 80,
                                'sample_size': count
                            }
                            adjustments.append(adjustment)
        
        return adjustments
    
    async def _generate_shipping_adjustments(self, shipping: List[ShippingPerformance]) -> List[Dict[str, Any]]:
        """
        Generate shipping adjustment recommendations.
        
        Args:
            shipping: List of shipping performance data
            
        Returns:
            List of shipping adjustment recommendations
        """
        adjustments = []
        
        # Skip if no shipping data
        if not shipping:
            return adjustments
        
        # Group by carrier
        carrier_shipping = {}
        for ship in shipping:
            if ship.carrier not in carrier_shipping:
                carrier_shipping[ship.carrier] = []
            carrier_shipping[ship.carrier].append(ship)
        
        # Analyze each carrier
        for carrier, carrier_shipping_list in carrier_shipping.items():
            # Calculate average cost accuracy
            cost_accuracies = [s.cost_accuracy for s in carrier_shipping_list]
            avg_cost_accuracy = sum(cost_accuracies) / len(cost_accuracies) if cost_accuracies else 0
            
            # Calculate average delivery time accuracy
            delivery_accuracies = [s.delivery_time_accuracy for s in carrier_shipping_list if s.delivery_time_accuracy is not None]
            avg_delivery_accuracy = sum(delivery_accuracies) / len(delivery_accuracies) if delivery_accuracies else 0
            
            # Calculate issue rate
            issues = [s for s in carrier_shipping_list if s.issues]
            issue_rate = len(issues) / len(carrier_shipping_list) if carrier_shipping_list else 0
            
            # Generate adjustment recommendation if cost accuracy is low
            if avg_cost_accuracy < 90:
                adjustment = {
                    'carrier': carrier,
                    'avg_cost_accuracy': avg_cost_accuracy,
                    'issue': 'inaccurate_cost_estimates',
                    'recommendation': 'update_shipping_cost_model',
                    'confidence': 75,
                    'sample_size': len(carrier_shipping_list)
                }
                
                adjustments.append(adjustment)
            
            # Generate adjustment recommendation if delivery accuracy is low
            if avg_delivery_accuracy < 80:
                adjustment = {
                    'carrier': carrier,
                    'avg_delivery_accuracy': avg_delivery_accuracy,
                    'issue': 'inaccurate_delivery_estimates',
                    'recommendation': 'update_delivery_time_model',
                    'confidence': 75,
                    'sample_size': len(carrier_shipping_list)
                }
                
                adjustments.append(adjustment)
            
            # Generate adjustment recommendation if issue rate is high
            if issue_rate > 0.1:
                # Count issue types
                issue_types = {}
                for ship in issues:
                    for issue in ship.issues:
                        issue_types[issue] = issue_types.get(issue, 0) + 1
                
                # Find most common issue
                most_common_issue = max(issue_types.items(), key=lambda x: x[1]) if issue_types else (None, 0)
                
                if most_common_issue[0]:
                    adjustment = {
                        'carrier': carrier,
                        'issue_rate': issue_rate * 100,
                        'most_common_issue': most_common_issue[0],
                        'issue_count': most_common_issue[1],
                        'recommendation': 'consider_alternative_carrier' if issue_rate > 0.2 else 'monitor_carrier_performance',
                        'confidence': 80 if issue_rate > 0.2 else 60,
                        'sample_size': len(carrier_shipping_list)
                    }
                    
                    adjustments.append(adjustment)
        
        return adjustments