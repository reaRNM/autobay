"""
AI Scoring Engine for Auction Items.

This module provides functionality to score auction items based on multiple factors
using machine learning models and weighted scoring algorithms.
"""

import os
import json
import logging
import time
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from ai_auction.models import ItemData, ScoreComponent, ScoringResult
from ai_auction.utils import setup_logging


logger = logging.getLogger(__name__)


class AIScoringEngine:
    """
    AI Scoring Engine for auction items.
    
    This class provides functionality to score auction items based on multiple factors
    using machine learning models and weighted scoring algorithms.
    """
    
    def __init__(
        self,
        model_dir: str = "models",
        config_path: str = None,
        default_weights: Dict[str, float] = None
    ):
        """
        Initialize the AI Scoring Engine.
        
        Args:
            model_dir: Directory to store trained models
            config_path: Path to configuration file
            default_weights: Default weights for score components
        """
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set default weights if not provided
        self.default_weights = default_weights or {
            "profit_potential": 0.4,
            "risk_assessment": 0.3,
            "shipping_ease": 0.2,
            "trend_prediction": 0.1
        }
        
        # Initialize models
        self.models = {
            "profit_potential": None,
            "risk_assessment": None,
            "shipping_ease": None,
            "trend_prediction": None
        }
        
        # Load pre-trained models if available
        self._load_models()
        
        logger.info("AI Scoring Engine initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "model_params": {
                "profit_potential": {
                    "type": "gradient_boosting",
                    "params": {
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "max_depth": 5
                    }
                },
                "risk_assessment": {
                    "type": "random_forest",
                    "params": {
                        "n_estimators": 100,
                        "max_depth": 10
                    }
                },
                "shipping_ease": {
                    "type": "linear_regression",
                    "params": {}
                },
                "trend_prediction": {
                    "type": "gradient_boosting",
                    "params": {
                        "n_estimators": 100,
                        "learning_rate": 0.05,
                        "max_depth": 3
                    }
                }
            },
            "feature_importance_threshold": 0.01,
            "min_training_samples": 50,
            "retraining_frequency_days": 7
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with default config
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                return default_config
        
        logger.info("Using default configuration")
        return default_config
    
    def _load_models(self) -> None:
        """Load pre-trained models if available."""
        for model_name in self.models.keys():
            model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    logger.info(f"Loaded pre-trained model for {model_name}")
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {e}")
    
    def _save_model(self, model_name: str, model: Any) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model
            model: Trained model object
        """
        model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model {model_name} to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
    
    def _create_model(self, model_name: str) -> Pipeline:
        """
        Create a new model pipeline based on configuration.
        
        Args:
            model_name: Name of the model to create
            
        Returns:
            Scikit-learn pipeline
        """
        model_config = self.config["model_params"].get(model_name, {})
        model_type = model_config.get("type", "gradient_boosting")
        model_params = model_config.get("params", {})
        
        if model_type == "gradient_boosting":
            regressor = GradientBoostingRegressor(**model_params)
        elif model_type == "random_forest":
            regressor = RandomForestRegressor(**model_params)
        elif model_type == "linear_regression":
            regressor = LinearRegression(**model_params)
        else:
            logger.warning(f"Unknown model type {model_type}, using GradientBoostingRegressor")
            regressor = GradientBoostingRegressor()
        
        # Create a pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])
        
        return pipeline
    
    def _extract_features(self, item: ItemData, component: str) -> Dict[str, float]:
        """
        Extract features for a specific scoring component.
        
        Args:
            item: Item data
            component: Scoring component name
            
        Returns:
            Dictionary of feature names and values
        """
        features = {}
        
        # Common features for all components
        features["current_bid"] = item.current_bid
        features["estimated_value"] = item.estimated_value
        features["bid_count"] = item.bid_count
        features["seller_rating"] = item.seller_rating
        
        # Time remaining until auction end (in hours)
        if item.auction_end_time:
            time_remaining = (item.auction_end_time - datetime.now()).total_seconds() / 3600
            features["time_remaining_hours"] = max(0, time_remaining)
        else:
            features["time_remaining_hours"] = 0
        
        # Component-specific features
        if component == "profit_potential":
            features["estimated_profit"] = item.estimated_profit
            features["profit_margin"] = item.profit_margin
            features["price_to_value_ratio"] = item.current_bid / max(1, item.estimated_value)
            
            # Historical price data
            if item.similar_items_sold:
                avg_price = sum(i.get("sale_price", 0) for i in item.similar_items_sold) / len(item.similar_items_sold)
                features["avg_similar_price"] = avg_price
                features["price_to_avg_ratio"] = item.current_bid / max(1, avg_price)
            
        elif component == "risk_assessment":
            features["bid_to_start_ratio"] = item.current_bid / max(1, item.starting_bid)
            features["seller_feedback_count"] = item.seller_feedback_count
            
            # Bid velocity
            if item.price_history and len(item.price_history) > 1:
                # Calculate average time between bids
                timestamps = [datetime.fromisoformat(bid.get("timestamp")) for bid in item.price_history 
                             if "timestamp" in bid]
                if len(timestamps) > 1:
                    time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() / 60 
                                 for i in range(1, len(timestamps))]
                    features["avg_minutes_between_bids"] = sum(time_diffs) / len(time_diffs)
                    features["bid_velocity"] = len(timestamps) / max(1, (timestamps[-1] - timestamps[0]).total_seconds() / 3600)
            
        elif component == "shipping_ease":
            if item.weight:
                features["weight"] = item.weight
            
            if item.dimensions:
                volume = item.dimensions.get("length", 0) * item.dimensions.get("width", 0) * item.dimensions.get("height", 0)
                features["volume"] = volume
            
            features["estimated_shipping_cost"] = item.estimated_shipping_cost
            
        elif component == "trend_prediction":
            # Category popularity
            features["category"] = hash(item.category) % 1000 / 1000  # Simple hash normalization
            
            # Seasonal factors
            current_month = datetime.now().month
            features["month_sin"] = np.sin(2 * np.pi * current_month / 12)
            features["month_cos"] = np.cos(2 * np.pi * current_month / 12)
            
            # Price trend
            if item.price_history and len(item.price_history) > 1:
                prices = [bid.get("amount", 0) for bid in item.price_history if "amount" in bid]
                if len(prices) > 1:
                    first_half = prices[:len(prices)//2]
                    second_half = prices[len(prices)//2:]
                    first_half_avg = sum(first_half) / len(first_half)
                    second_half_avg = sum(second_half) / len(second_half)
                    features["price_trend"] = (second_half_avg - first_half_avg) / max(1, first_half_avg)
        
        return features
    
    def _prepare_training_data(
        self, 
        items: List[ItemData], 
        component: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for a specific component.
        
        Args:
            items: List of item data
            component: Scoring component name
            
        Returns:
            Features DataFrame and target Series
        """
        features_list = []
        targets = []
        
        for item in items:
            # Extract features
            features = self._extract_features(item, component)
            
            # Get target value
            if component == "profit_potential":
                target = item.profit_margin
            elif component == "risk_assessment":
                # Invert risk (higher is better)
                target = 1.0 - item.existing_scores.get("risk", 0.5)
            elif component == "shipping_ease":
                target = item.existing_scores.get("shipping_ease", 0.5)
            elif component == "trend_prediction":
                target = item.existing_scores.get("trend", 0.5)
            else:
                logger.warning(f"Unknown component {component}, skipping item")
                continue
            
            features_list.append(features)
            targets.append(target)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        targets_series = pd.Series(targets)
        
        return features_df, targets_series
    
    def train_models(self, training_data: List[ItemData]) -> Dict[str, Dict[str, Any]]:
        """
        Train all scoring models using the provided training data.
        
        Args:
            training_data: List of item data with known outcomes
            
        Returns:
            Dictionary with training results for each model
        """
        if len(training_data) < self.config.get("min_training_samples", 50):
            logger.warning(f"Insufficient training data: {len(training_data)} items")
            return {}
        
        results = {}
        
        for component in self.models.keys():
            logger.info(f"Training model for {component}")
            
            try:
                # Prepare training data
                features, targets = self._prepare_training_data(training_data, component)
                
                if len(features) < self.config.get("min_training_samples", 50):
                    logger.warning(f"Insufficient training data for {component}: {len(features)} items")
                    continue
                
                # Split into training and validation sets
                X_train, X_val, y_train, y_val = train_test_split(
                    features, targets, test_size=0.2, random_state=42
                )
                
                # Create and train model
                model = self._create_model(component)
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                # Save model
                self.models[component] = model
                self._save_model(component, model)
                
                # Get feature importances if available
                feature_importances = {}
                if hasattr(model[-1], 'feature_importances_'):
                    importances = model[-1].feature_importances_
                    for i, feature in enumerate(features.columns):
                        if i < len(importances):
                            feature_importances[feature] = float(importances[i])
                
                # Store results
                results[component] = {
                    "mse": mse,
                    "r2": r2,
                    "n_samples": len(features),
                    "feature_importances": feature_importances
                }
                
                logger.info(f"Trained model for {component}: MSE={mse:.4f}, R²={r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training model for {component}: {e}")
        
        return results
    
    def _predict_component_score(
        self, 
        item: ItemData, 
        component: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Predict score for a specific component.
        
        Args:
            item: Item data
            component: Scoring component name
            
        Returns:
            Predicted score and factors dictionary
        """
        # Extract features
        features = self._extract_features(item, component)
        features_df = pd.DataFrame([features])
        
        # Get model
        model = self.models.get(component)
        
        # If model is not available, use heuristic scoring
        if model is None:
            logger.warning(f"Model for {component} not available, using heuristic scoring")
            return self._heuristic_score(item, component, features)
        
        # Predict score
        try:
            score = float(model.predict(features_df)[0])
            
            # Ensure score is in [0, 1] range
            score = max(0.0, min(1.0, score))
            
            # Get feature importances if available
            factors = {}
            if hasattr(model[-1], 'feature_importances_'):
                importances = model[-1].feature_importances_
                for i, (feature, value) in enumerate(features.items()):
                    if i < len(importances) and importances[i] > self.config.get("feature_importance_threshold", 0.01):
                        factors[feature] = {
                            "value": value,
                            "importance": float(importances[i])
                        }
            
            return score, factors
            
        except Exception as e:
            logger.error(f"Error predicting {component} score: {e}")
            return self._heuristic_score(item, component, features)
    
    def _heuristic_score(
        self, 
        item: ItemData, 
        component: str, 
        features: Dict[str, float]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate a heuristic score when model is not available.
        
        Args:
            item: Item data
            component: Scoring component name
            features: Extracted features
            
        Returns:
            Heuristic score and factors dictionary
        """
        score = 0.5  # Default score
        factors = {}
        
        if component == "profit_potential":
            # Simple profit margin based score
            profit_margin = item.profit_margin
            score = min(1.0, max(0.0, profit_margin / 100))
            factors = {
                "profit_margin": {
                    "value": profit_margin,
                    "importance": 0.8
                },
                "estimated_profit": {
                    "value": item.estimated_profit,
                    "importance": 0.2
                }
            }
            
        elif component == "risk_assessment":
            # Lower risk is better (higher score)
            seller_factor = min(1.0, item.seller_rating / 5.0) * 0.4
            price_factor = min(1.0, item.estimated_value / max(1, item.current_bid)) * 0.3
            time_factor = min(1.0, features.get("time_remaining_hours", 0) / 24) * 0.3
            
            score = seller_factor + price_factor + time_factor
            factors = {
                "seller_rating": {
                    "value": item.seller_rating,
                    "importance": 0.4
                },
                "price_to_value": {
                    "value": item.current_bid / max(1, item.estimated_value),
                    "importance": 0.3
                },
                "time_remaining": {
                    "value": features.get("time_remaining_hours", 0),
                    "importance": 0.3
                }
            }
            
        elif component == "shipping_ease":
            # Lower weight and dimensions are better
            weight_factor = 0.5
            if item.weight:
                # Normalize weight (assume 20kg is max)
                weight_factor = 1.0 - min(1.0, item.weight / 20.0)
            
            volume_factor = 0.5
            if item.dimensions:
                # Normalize volume (assume 1m³ is max)
                volume = item.dimensions.get("length", 0) * item.dimensions.get("width", 0) * item.dimensions.get("height", 0)
                volume_factor = 1.0 - min(1.0, volume / 1000000)
            
            score = 0.7 * weight_factor + 0.3 * volume_factor
            factors = {
                "weight": {
                    "value": item.weight,
                    "importance": 0.7
                },
                "volume": {
                    "value": features.get("volume", 0),
                    "importance": 0.3
                }
            }
            
        elif component == "trend_prediction":
            # Simple trend score based on category and time
            category_factor = 0.5
            seasonal_factor = 0.5 + 0.5 * features.get("month_sin", 0)
            
            score = 0.6 * category_factor + 0.4 * seasonal_factor
            factors = {
                "category": {
                    "value": item.category,
                    "importance": 0.6
                },
                "seasonality": {
                    "value": seasonal_factor,
                    "importance": 0.4
                }
            }
        
        return score, factors
    
    def _generate_explanation(self, component: str, score: float, factors: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation for a component score.
        
        Args:
            component: Scoring component name
            score: Component score
            factors: Score factors
            
        Returns:
            Explanation string
        """
        if component == "profit_potential":
            if score > 0.8:
                return "Excellent profit potential with high margin."
            elif score > 0.6:
                return "Good profit potential with above-average margin."
            elif score > 0.4:
                return "Moderate profit potential with average margin."
            elif score > 0.2:
                return "Limited profit potential with below-average margin."
            else:
                return "Poor profit potential with low margin."
                
        elif component == "risk_assessment":
            if score > 0.8:
                return "Very low risk with high seller rating and good price-to-value ratio."
            elif score > 0.6:
                return "Low risk with above-average seller rating."
            elif score > 0.4:
                return "Moderate risk with average seller rating."
            elif score > 0.2:
                return "High risk with below-average seller rating or poor price-to-value ratio."
            else:
                return "Very high risk with low seller rating and poor price-to-value ratio."
                
        elif component == "shipping_ease":
            if score > 0.8:
                return "Very easy to ship with low weight and small dimensions."
            elif score > 0.6:
                return "Easy to ship with manageable weight and dimensions."
            elif score > 0.4:
                return "Average shipping difficulty."
            elif score > 0.2:
                return "Difficult to ship due to weight or dimensions."
            else:
                return "Very difficult to ship with high weight and large dimensions."
                
        elif component == "trend_prediction":
            if score > 0.8:
                return "Strong upward trend with high seasonal demand."
            elif score > 0.6:
                return "Positive trend with good seasonal demand."
            elif score > 0.4:
                return "Stable trend with average seasonal demand."
            elif score > 0.2:
                return "Negative trend with below-average seasonal demand."
            else:
                return "Strong downward trend with low seasonal demand."
        
        return "No explanation available."
    
    def score_item(
        self, 
        item: ItemData, 
        weights: Optional[Dict[str, float]] = None
    ) -> ScoringResult:
        """
        Score an auction item using the trained models.
        
        Args:
            item: Item data
            weights: Optional custom weights for score components
            
        Returns:
            ScoringResult object with overall score and components
        """
        # Use provided weights or default weights
        component_weights = weights or self.default_weights
        
        # Normalize weights to sum to 1
        weight_sum = sum(component_weights.values())
        if weight_sum != 1.0:
            component_weights = {k: v / weight_sum for k, v in component_weights.items()}
        
        # Calculate scores for each component
        components = []
        for component_name in self.models.keys():
            score, factors = self._predict_component_score(item, component_name)
            weight = component_weights.get(component_name, 0.0)
            explanation = self._generate_explanation(component_name, score, factors)
            
            component = ScoreComponent(
                name=component_name,
                score=score,
                weight=weight,
                factors=factors,
                explanation=explanation
            )
            components.append(component)
        
        # Calculate overall priority score
        priority_score = sum(comp.weighted_score for comp in components)
        
        # Create and return scoring result
        result = ScoringResult(
            item_id=item.item_id,
            priority_score=priority_score,
            components=components
        )
        
        logger.info(f"Scored item {item.item_id}: {priority_score:.4f}")
        return result
    
    def score_items(
        self, 
        items: List[ItemData], 
        weights: Optional[Dict[str, float]] = None
    ) -> List[ScoringResult]:
        """
        Score multiple auction items.
        
        Args:
            items: List of item data
            weights: Optional custom weights for score components
            
        Returns:
            List of ScoringResult objects
        """
        results = []
        for item in items:
            try:
                result = self.score_item(item, weights)
                results.append(result)
            except Exception as e:
                logger.error(f"Error scoring item {item.item_id}: {e}")
        
        # Sort by priority score (descending)
        results.sort(key=lambda x: x.priority_score, reverse=True)
        
        return results
    
    def check_retraining_needed(self) -> bool:
        """
        Check if model retraining is needed based on last training date.
        
        Returns:
            True if retraining is needed, False otherwise
        """
        # Check if models exist
        if not all(self.models.values()):
            logger.info("Retraining needed: Some models are missing")
            return True
        
        # Check last modified time of model files
        retraining_frequency = self.config.get("retraining_frequency_days", 7)
        for model_name in self.models.keys():
            model_path = os.path.join(self.model_dir, f"{model_name}_model.pkl")
            if os.path.exists(model_path):
                last_modified = datetime.fromtimestamp(os.path.getmtime(model_path))
                days_since_training = (datetime.now() - last_modified).days
                if days_since_training >= retraining_frequency:
                    logger.info(f"Retraining needed: {model_name} model is {days_since_training} days old")
                    return True
            else:
                logger.info(f"Retraining needed: {model_name} model file not found")
                return True
        
        return False