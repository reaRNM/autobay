"""
Reinforcement learning module for the Learning & Feedback Systems Module.

This module provides functionality for training and using
reinforcement learning models for bidding strategy optimization.
"""

import os
import logging
import uuid
import pickle
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from collections import deque

from learning_feedback.models import (
    AuctionOutcome, BidStrategy, ModelPerformance, AuctionStatus
)


logger = logging.getLogger(__name__)


class BiddingEnvironment:
    """
    Environment for bidding reinforcement learning.
    
    This class simulates an auction environment for
    reinforcement learning.
    """
    
    def __init__(self):
        """Initialize the BiddingEnvironment."""
        logger.info("BiddingEnvironment initialized")
    
    def reset(self, auction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reset the environment with new auction data.
        
        Args:
            auction_data: Auction data
            
        Returns:
            Initial state
        """
        self.auction_data = auction_data
        self.current_time = 0.0  # 0.0 to 1.0 (start to end of auction)
        self.current_price = auction_data.get('start_price', 0.0)
        self.estimated_price = auction_data.get('estimated_price', 0.0)
        self.bids_placed = 0
        self.max_bids = auction_data.get('max_bids', 2)
        self.done = False
        self.won = False
        
        # Initial state
        state = {
            'category_id': auction_data.get('category_id'),
            'estimated_price': self.estimated_price,
            'current_price': self.current_price,
            'bids': self.bids_placed,
            'watchers': auction_data.get('watchers', 0),
            'views': auction_data.get('views', 0),
            'time_left_percentage': 1.0 - self.current_time
        }
        
        return state
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (bid amount and time)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Extract action
        bid_amount = action.get('bid_amount', 0.0)
        bid_time = action.get('bid_time', 0.0)  # 0.0 to 1.0 (when to place bid)
        
        # Advance time to bid time
        self.current_time = bid_time
        
        # Check if auction is already done
        if self.done:
            return self._get_state(), 0.0, True, {'message': 'Auction already ended'}
        
        # Check if we've reached max bids
        if self.bids_placed >= self.max_bids:
            self.done = True
            return self._get_state(), 0.0, True, {'message': 'Max bids reached'}
        
        # Place bid
        self.bids_placed += 1
        
        # Simulate auction outcome
        if bid_amount > self.current_price:
            self.current_price = bid_amount
            
            # Check if this is the final bid (auction end)
            if bid_time >= 0.99:
                self.done = True
                self.won = True
                
                # Calculate reward based on profit
                if self.estimated_price > 0:
                    # Reward is higher if bid is lower relative to estimated price
                    reward = self.estimated_price - bid_amount
                else:
                    reward = 10.0  # Default reward for winning
                
                return self._get_state(), reward, True, {'message': 'Auction won', 'final_price': bid_amount}
            
            # Not the final bid, continue auction
            return self._get_state(), 0.1, False, {'message': 'Bid placed'}
        else:
            # Bid too low
            return self._get_state(), -1.0, False, {'message': 'Bid too low'}
    
    def _get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.
        
        Returns:
            Current state
        """
        return {
            'category_id': self.auction_data.get('category_id'),
            'estimated_price': self.estimated_price,
            'current_price': self.current_price,
            'bids': self.bids_placed,
            'watchers': self.auction_data.get('watchers', 0),
            'views': self.auction_data.get('views', 0),
            'time_left_percentage': 1.0 - self.current_time
        }


class BiddingAgent:
    """
    Agent for bidding reinforcement learning.
    
    This class implements a reinforcement learning agent
    for bidding strategy optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BiddingAgent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.learning_rate = config.get('learning_rate', 0.001)
        self.discount_factor = config.get('discount_factor', 0.99)
        self.exploration_rate = config.get('exploration_rate', 0.1)
        self.exploration_decay = config.get('exploration_decay', 0.995)
        self.batch_size = config.get('batch_size', 64)
        self.memory_size = config.get('memory_size', 10000)
        
        # Initialize memory
        self.memory = deque(maxlen=self.memory_size)
        
        # Initialize model
        self.model = None
        
        # Create models directory if it doesn't exist
        os.makedirs('models/bidding_agent', exist_ok=True)
        
        # Try to load existing model
        model_path = 'models/bidding_agent/latest.pkl'
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Loaded existing bidding agent model")
            except Exception as e:
                logger.error(f"Error loading bidding agent model: {e}")
        
        logger.info("BiddingAgent initialized")
    
    async def train(
        self,
        auctions: List[AuctionOutcome],
        env: BiddingEnvironment
    ) -> ModelPerformance:
        """
        Train the bidding agent.
        
        Args:
            auctions: List of auction outcomes
            env: Bidding environment
            
        Returns:
            ModelPerformance object
        """
        logger.info("Training bidding agent")
        
        try:
            # Filter out auctions without status
            auctions = [a for a in auctions if a.status in [AuctionStatus.SOLD, AuctionStatus.UNSOLD]]
            
            if not auctions:
                raise ValueError("No auction data available for training")
            
            # Initialize metrics
            total_reward = 0.0
            episodes = 0
            
            # Train on each auction
            start_time = datetime.now()
            
            for auction in auctions:
                # Convert auction to environment format
                auction_data = {
                    'category_id': auction.category_id,
                    'start_price': auction.start_price,
                    'estimated_price': auction.estimated_price,
                    'watchers': auction.watchers,
                    'views': auction.views,
                    'max_bids': 2
                }
                
                # Reset environment
                state = env.reset(auction_data)
                
                # Run episode
                episode_reward = 0.0
                done = False
                
                while not done:
                    # Choose action
                    action = self._choose_action(state)
                    
                    # Take action
                    next_state, reward, done, info = env.step(action)
                    
                    # Store in memory
                    self._remember(state, action, reward, next_state, done)
                    
                    # Update state
                    state = next_state
                    
                    # Update reward
                    episode_reward += reward
                    
                    # Train on batch
                    if len(self.memory) >= self.batch_size:
                        self._replay()
                
                # Update metrics
                total_reward += episode_reward
                episodes += 1
                
                # Decay exploration rate
                self.exploration_rate *= self.exploration_decay
            
            # Calculate training duration
            training_duration = (datetime.now() - start_time).total_seconds()
            
            # Save model
            model_id = f"bidding_agent_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            model_path = f"models/bidding_agent/{model_id}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Also save as latest
            with open('models/bidding_agent/latest.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            # Calculate average reward
            avg_reward = total_reward / episodes if episodes > 0 else 0.0
            
            # Create model performance record
            performance = ModelPerformance(
                id=str(uuid.uuid4()),
                model_name=model_id,
                model_version="1.0.0",
                model_type="bidding_agent",
                category_id=None,
                accuracy=0.0,  # Not applicable for RL
                precision=0.0,  # Not applicable for RL
                recall=0.0,  # Not applicable for RL
                f1_score=0.0,  # Not applicable for RL
                mean_absolute_error=None,  # Not applicable for RL
                mean_squared_error=None,  # Not applicable for RL
                r_squared=None,  # Not applicable for RL
                sample_count=len(auctions),
                training_duration=training_duration,
                inference_latency=0.0,  # Will be updated during inference
                metadata={
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'exploration_rate': self.exploration_rate,
                    'batch_size': self.batch_size,
                    'memory_size': self.memory_size,
                    'episodes': episodes,
                    'avg_reward': avg_reward
                }
            )
            
            logger.info(f"Trained bidding agent with average reward {avg_reward:.4f}")
            return performance
        
        except Exception as e:
            logger.error(f"Error training bidding agent: {e}")
            raise
    
    async def update_policy(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        done: bool
    ) -> None:
        """
        Update policy with a new experience.
        
        Args:
            state: State
            action: Action
            reward: Reward
            done: Whether the episode is done
        """
        logger.debug("Updating bidding agent policy")
        
        try:
            # Store in memory
            self._remember(state, action, reward, state, done)  # Use same state as next_state for simplicity
            
            # Train on batch if enough samples
            if len(self.memory) >= self.batch_size:
                self._replay()
        
        except Exception as e:
            logger.error(f"Error updating bidding agent policy: {e}")
    
    async def get_bid_strategy(
        self,
        auction_data: Dict[str, Any],
        user_id: str,
        category_id: Optional[str] = None
    ) -> BidStrategy:
        """
        Get a bidding strategy for an auction.
        
        Args:
            auction_data: Auction data
            user_id: User ID
            category_id: Category ID (optional)
            
        Returns:
            BidStrategy object
        """
        logger.debug(f"Getting bid strategy for auction")
        
        try:
            # Create bid strategy
            strategy = BidStrategy(
                id=str(uuid.uuid4()),
                user_id=user_id,
                category_id=category_id,
                name="AI-Generated Strategy",
                description="Automatically generated bidding strategy based on historical data",
                max_bid_percentage=0.8,  # Default
                early_bid_threshold=0.3,  # Default
                late_bid_threshold=0.9,  # Default
                bid_increment_factor=1.05,  # Default
                max_bid_count=2,  # Default
                risk_tolerance=0.5  # Default
            )
            
            # If we have a trained model, use it to optimize strategy
            if self.model is not None:
                # Extract features from auction data
                state = {
                    'category_id': category_id,
                    'estimated_price': auction_data.get('estimated_price', 0.0),
                    'current_price': auction_data.get('start_price', 0.0),
                    'bids': 0,
                    'watchers': auction_data.get('watchers', 0),
                    'views': auction_data.get('views', 0),
                    'time_left_percentage': 1.0
                }
                
                # Get action from model
                action = self._predict_action(state)
                
                # Update strategy based on action
                if 'bid_amount' in action and auction_data.get('estimated_price', 0.0) > 0:
                    strategy.max_bid_percentage = action['bid_amount'] / auction_data['estimated_price']
                
                if 'bid_time' in action:
                    strategy.late_bid_threshold = action['bid_time']
            
            logger.info(f"Generated bid strategy for user {user_id}")
            return strategy
        
        except Exception as e:
            logger.error(f"Error getting bid strategy: {e}")
            # Return default strategy
            return BidStrategy(
                id=str(uuid.uuid4()),
                user_id=user_id,
                category_id=category_id,
                name="Default Strategy",
                description="Default bidding strategy",
                max_bid_percentage=0.8,
                early_bid_threshold=0.3,
                late_bid_threshold=0.9,
                bid_increment_factor=1.05,
                max_bid_count=2,
                risk_tolerance=0.5
            )
    
    def _choose_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Choose an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            Action to take
        """
        # Exploration: random action
        if np.random.rand() < self.exploration_rate:
            return {
                'bid_amount': state.get('estimated_price', 100.0) * np.random.uniform(0.7, 0.9),
                'bid_time': np.random.uniform(0.8, 1.0)  # Prefer late bidding
            }
        
        # Exploitation: use model
        return self._predict_action(state)
    
    def _predict_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            Predicted action
        """
        # If no model, return default action
        if self.model is None:
            return {
                'bid_amount': state.get('estimated_price', 100.0) * 0.8,
                'bid_time': 0.9
            }
        
        # In a real implementation, this would use a neural network
        # For simplicity, we'll use a rule-based approach
        estimated_price = state.get('estimated_price', 100.0)
        current_price = state.get('current_price', 0.0)
        time_left = state.get('time_left_percentage', 1.0)
        
        # Calculate bid amount
        if current_price < estimated_price * 0.5:
            # Price is low, bid conservatively
            bid_amount = current_price * 1.1
        elif current_price < estimated_price * 0.8:
            # Price is moderate, bid more aggressively
            bid_amount = current_price * 1.05
        else:
            # Price is high, bid very conservatively
            bid_amount = current_price * 1.02
        
        # Cap bid amount
        bid_amount = min(bid_amount, estimated_price * 0.9)
        
        # Calculate bid time
        if time_left > 0.5:
            # Early in auction, wait
            bid_time = 0.9
        else:
            # Late in auction, bid soon
            bid_time = max(0.0, time_left - 0.1)
        
        return {
            'bid_amount': bid_amount,
            'bid_time': bid_time
        }
    
    def _remember(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Dict[str, Any],
        done: bool
    ) -> None:
        """
        Store experience in memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def _replay(self) -> None:
        """Train the model on a batch of experiences."""
        # In a real implementation, this would update a neural network
        # For simplicity, we'll just log that training occurred
        logger.debug(f"Training bidding agent on batch of {self.batch_size} experiences")