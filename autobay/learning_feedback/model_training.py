"""
Model training module for the Learning & Feedback Systems Module.

This module provides functionality for training and updating
machine learning models for pricing, bidding, and shipping.
"""

import os
import logging
import uuid
import pickle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from learning_feedback.models import (
    AuctionOutcome, ShippingPerformance, ModelPerformance, AuctionStatus
)


logger = logging.getLogger(__name__)


class PricingModelTrainer:
    """
    Trainer for pricing models.
    
    This class provides methods for training and updating
    machine learning models for price prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PricingModelTrainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_type = config.get('model_type', 'gradient_boosting')
        self.features = config.get('features', ['category', 'condition', 'brand', 'model', 'age', 'specifications'])
        self.target = config.get('target', 'final_price')
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        
        # Initialize model storage
        self.models = {}  # category_id -> model
        self.scalers = {}  # category_id -> scaler
        self.encoders = {}  # category_id -> encoder
        
        # Create models directory if it doesn't exist
        os.makedirs('models/pricing', exist_ok=True)
        
        logger.info("PricingModelTrainer initialized")
    
    async def train_model(
        self,
        auctions: List[AuctionOutcome],
        category_id: Optional[str] = None
    ) -> ModelPerformance:
        """
        Train a pricing model.
        
        Args:
            auctions: List of auction outcomes
            category_id: Category ID (optional)
            
        Returns:
            ModelPerformance object
        """
        logger.info(f"Training pricing model for category {category_id or 'all'}")
        
        try:
            # Filter auctions by category if specified
            if category_id:
                auctions = [a for a in auctions if a.category_id == category_id]
            
            # Filter out auctions without final price
            auctions = [a for a in auctions if a.final_price is not None]
            
            if not auctions:
                raise ValueError("No auction data available for training")
            
            # Convert auctions to DataFrame
            df = self._convert_auctions_to_dataframe(auctions)
            
            # Prepare features and target
            X, y = self._prepare_data(df)
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Create and train model
            start_time = datetime.now()
            
            if self.model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=self.random_state
                )
            else:
                # Default to gradient boosting
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=self.random_state
                )
            
            # Create preprocessing pipeline
            numeric_features = [f for f in X.columns if X[f].dtype in ['int64', 'float64']]
            categorical_features = [f for f in X.columns if X[f].dtype == 'object']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Calculate training duration
            training_duration = (datetime.now() - start_time).total_seconds()
            
            # Evaluate model
            y_pred = pipeline.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate accuracy as 1 - (MAE / mean(y_test))
            accuracy = max(0, min(1, 1 - (mae / np.mean(y_test))))
            
            # Save model
            model_id = f"pricing_{category_id or 'all'}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            model_path = f"models/pricing/{model_id}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)
            
            # Store model in memory
            key = category_id or 'all'
            self.models[key] = pipeline
            
            # Create model performance record
            performance = ModelPerformance(
                id=str(uuid.uuid4()),
                model_name=model_id,
                model_version="1.0.0",
                model_type="pricing",
                category_id=category_id,
                accuracy=accuracy,
                precision=0.0,  # Not applicable for regression
                recall=0.0,  # Not applicable for regression
                f1_score=0.0,  # Not applicable for regression
                mean_absolute_error=mae,
                mean_squared_error=mse,
                r_squared=r2,
                sample_count=len(auctions),
                training_duration=training_duration,
                inference_latency=0.0,  # Will be updated during inference
                metadata={
                    'model_type': self.model_type,
                    'features': self.features,
                    'test_size': self.test_size,
                    'random_state': self.random_state
                }
            )
            
            logger.info(f"Trained pricing model with accuracy {accuracy:.4f}, MAE {mae:.2f}, R² {r2:.4f}")
            return performance
        
        except Exception as e:
            logger.error(f"Error training pricing model: {e}")
            raise
    
    async def update_model_with_sample(
        self,
        item_id: str,
        category_id: str,
        estimated_price: float,
        actual_price: float,
        accuracy: float
    ) -> None:
        """
        Update model with a new sample.
        
        Args:
            item_id: Item ID
            category_id: Category ID
            estimated_price: Estimated price
            actual_price: Actual price
            accuracy: Price accuracy
        """
        logger.debug(f"Updating pricing model with sample for item {item_id}")
        
        try:
            # For now, just log the sample
            # In a real implementation, this would update the model incrementally
            # or store the sample for batch retraining
            logger.info(f"New pricing sample: item={item_id}, category={category_id}, estimated={estimated_price:.2f}, actual={actual_price:.2f}, accuracy={accuracy:.2f}%")
        
        except Exception as e:
            logger.error(f"Error updating pricing model with sample: {e}")
    
    async def incorporate_feedback(
        self,
        item_id: str,
        ai_suggestion: float,
        user_correction: float,
        rating: int
    ) -> None:
        """
        Incorporate user feedback into the model.
        
        Args:
            item_id: Item ID
            ai_suggestion: AI-suggested price
            user_correction: User-corrected price
            rating: User rating (1-5)
        """
        logger.debug(f"Incorporating feedback for item {item_id}")
        
        try:
            # For now, just log the feedback
            # In a real implementation, this would update the model
            # or store the feedback for batch retraining
            logger.info(f"Pricing feedback: item={item_id}, ai={ai_suggestion:.2f}, user={user_correction:.2f}, rating={rating}")
        
        except Exception as e:
            logger.error(f"Error incorporating pricing feedback: {e}")
    
    async def predict_price(
        self,
        item_data: Dict[str, Any],
        category_id: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Predict price for an item.
        
        Args:
            item_data: Item data
            category_id: Category ID (optional)
            
        Returns:
            Tuple of (predicted_price, confidence)
        """
        logger.debug(f"Predicting price for item {item_data.get('id')}")
        
        try:
            # Get model for category
            key = category_id or 'all'
            
            if key not in self.models:
                # Try to load model from disk
                model_path = f"models/pricing/{key}_latest.pkl"
                
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[key] = pickle.load(f)
                else:
                    raise ValueError(f"No pricing model available for category {key}")
            
            # Convert item data to DataFrame
            df = pd.DataFrame([item_data])
            
            # Prepare features
            X = df[self.features]
            
            # Predict price
            start_time = datetime.now()
            predicted_price = self.models[key].predict(X)[0]
            inference_latency = (datetime.now() - start_time).total_seconds() * 1000  # in milliseconds
            
            # Calculate confidence (simple heuristic for now)
            confidence = 0.8  # Default confidence
            
            logger.info(f"Predicted price for item {item_data.get('id')}: ${predicted_price:.2f} (confidence: {confidence:.2f})")
            return predicted_price, confidence
        
        except Exception as e:
            logger.error(f"Error predicting price: {e}")
            raise
    
    def _convert_auctions_to_dataframe(self, auctions: List[AuctionOutcome]) -> pd.DataFrame:
        """
        Convert auction outcomes to DataFrame.
        
        Args:
            auctions: List of auction outcomes
            
        Returns:
            DataFrame
        """
        # Extract relevant fields from auctions
        data = []
        for auction in auctions:
            row = {
                'id': auction.id,
                'item_id': auction.item_id,
                'category_id': auction.category_id,
                'platform': auction.platform,
                'condition': auction.condition if hasattr(auction, 'condition') else 'Unknown',
                'final_price': auction.final_price,
                'shipping_cost': auction.shipping_cost,
                'views': auction.views,
                'watchers': auction.watchers,
                'bids': auction.bids
            }
            
            # Add additional fields if available
            if hasattr(auction, 'brand'):
                row['brand'] = auction.brand
            
            if hasattr(auction, 'model'):
                row['model'] = auction.model
            
            if hasattr(auction, 'age'):
                row['age'] = auction.age
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        # Select features and target
        available_features = [f for f in self.features if f in df.columns]
        X = df[available_features]
        y = df[self.target]
        
        return X, y


class BidModelTrainer:
    """
    Trainer for bidding models.
    
    This class provides methods for training and updating
    machine learning models for bid success prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BidModelTrainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_type = config.get('model_type', 'random_forest')
        self.features = config.get('features', ['category', 'condition', 'price_ratio', 'time_left', 'watchers', 'bids'])
        self.target = config.get('target', 'success')
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        
        # Initialize model storage
        self.models = {}  # category_id -> model
        self.scalers = {}  # category_id -> scaler
        self.encoders = {}  # category_id -> encoder
        
        # Create models directory if it doesn't exist
        os.makedirs('models/bidding', exist_ok=True)
        
        logger.info("BidModelTrainer initialized")
    
    async def train_model(
        self,
        auctions: List[AuctionOutcome],
        category_id: Optional[str] = None
    ) -> ModelPerformance:
        """
        Train a bidding model.
        
        Args:
            auctions: List of auction outcomes
            category_id: Category ID (optional)
            
        Returns:
            ModelPerformance object
        """
        logger.info(f"Training bidding model for category {category_id or 'all'}")
        
        try:
            # Filter auctions by category if specified
            if category_id:
                auctions = [a for a in auctions if a.category_id == category_id]
            
            # Filter out auctions without status
            auctions = [a for a in auctions if a.status in [AuctionStatus.SOLD, AuctionStatus.UNSOLD]]
            
            if not auctions:
                raise ValueError("No auction data available for training")
            
            # Convert auctions to DataFrame
            df = self._convert_auctions_to_dataframe(auctions)
            
            # Prepare features and target
            X, y = self._prepare_data(df)
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Create and train model
            start_time = datetime.now()
            
            if self.model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state
                )
            else:
                # Default to random forest
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state
                )
            
            # Create preprocessing pipeline
            numeric_features = [f for f in X.columns if X[f].dtype in ['int64', 'float64']]
            categorical_features = [f for f in X.columns if X[f].dtype == 'object']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Calculate training duration
            training_duration = (datetime.now() - start_time).total_seconds()
            
            # Evaluate model
            y_pred = pipeline.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Save model
            model_id = f"bidding_{category_id or 'all'}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            model_path = f"models/bidding/{model_id}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)
            
            # Store model in memory
            key = category_id or 'all'
            self.models[key] = pipeline
            
            # Create model performance record
            performance = ModelPerformance(
                id=str(uuid.uuid4()),
                model_name=model_id,
                model_version="1.0.0",
                model_type="bidding",
                category_id=category_id,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                mean_absolute_error=None,  # Not applicable for classification
                mean_squared_error=None,  # Not applicable for classification
                r_squared=None,  # Not applicable for classification
                sample_count=len(auctions),
                training_duration=training_duration,
                inference_latency=0.0,  # Will be updated during inference
                metadata={
                    'model_type': self.model_type,
                    'features': self.features,
                    'test_size': self.test_size,
                    'random_state': self.random_state
                }
            )
            
            logger.info(f"Trained bidding model with accuracy {accuracy:.4f}, precision {precision:.4f}, recall {recall:.4f}, F1 {f1:.4f}")
            return performance
        
        except Exception as e:
            logger.error(f"Error training bidding model: {e}")
            raise
    
    async def update_model_with_sample(
        self,
        item_id: str,
        category_id: str,
        success: bool,
        bids: int,
        watchers: int,
        views: int,
        final_price: Optional[float] = None,
        estimated_price: Optional[float] = None
    ) -> None:
        """
        Update model with a new sample.
        
        Args:
            item_id: Item ID
            category_id: Category ID
            success: Whether the bid was successful
            bids: Number of bids
            watchers: Number of watchers
            views: Number of views
            final_price: Final price (optional)
            estimated_price: Estimated price (optional)
        """
        logger.debug(f"Updating bidding model with sample for item {item_id}")
        
        try:
            # For now, just log the sample
            # In a real implementation, this would update the model incrementally
            # or store the sample for batch retraining
            price_ratio = None
            if final_price is not None and estimated_price is not None and estimated_price > 0:
                price_ratio = final_price / estimated_price
            
            logger.info(f"New bidding sample: item={item_id}, category={category_id}, success={success}, bids={bids}, watchers={watchers}, views={views}, price_ratio={price_ratio:.2f if price_ratio else 'N/A'}")
        
        except Exception as e:
            logger.error(f"Error updating bidding model with sample: {e}")
    
    async def incorporate_feedback(
        self,
        item_id: str,
        ai_suggestion: str,
        user_correction: str,
        rating: int
    ) -> None:
        """
        Incorporate user feedback into the model.
        
        Args:
            item_id: Item ID
            ai_suggestion: AI-suggested bidding strategy
            user_correction: User-corrected bidding strategy
            rating: User rating (1-5)
        """
        logger.debug(f"Incorporating feedback for item {item_id}")
        
        try:
            # For now, just log the feedback
            # In a real implementation, this would update the model
            # or store the feedback for batch retraining
            logger.info(f"Bidding feedback: item={item_id}, ai={ai_suggestion}, user={user_correction}, rating={rating}")
        
        except Exception as e:
            logger.error(f"Error incorporating bidding feedback: {e}")
    
    async def predict_bid_success(
        self,
        item_data: Dict[str, Any],
        bid_amount: float,
        category_id: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Predict bid success probability.
        
        Args:
            item_data: Item data
            bid_amount: Bid amount
            category_id: Category ID (optional)
            
        Returns:
            Tuple of (success_probability, confidence)
        """
        logger.debug(f"Predicting bid success for item {item_data.get('id')}")
        
        try:
            # Get model for category
            key = category_id or 'all'
            
            if key not in self.models:
                # Try to load model from disk
                model_path = f"models/bidding/{key}_latest.pkl"
                
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[key] = pickle.load(f)
                else:
                    raise ValueError(f"No bidding model available for category {key}")
            
            # Add bid amount to item data
            item_data_with_bid = item_data.copy()
            item_data_with_bid['bid_amount'] = bid_amount
            
            # Calculate price ratio if estimated price is available
            if 'estimated_price' in item_data and item_data['estimated_price'] > 0:
                item_data_with_bid['price_ratio'] = bid_amount / item_data['estimated_price']
            
            # Convert item data to DataFrame
            df = pd.DataFrame([item_data_with_bid])
            
            # Prepare features
            available_features = [f for f in self.features if f in df.columns]
            X = df[available_features]
            
            # Predict success probability
            start_time = datetime.now()
            success_prob = self.models[key].predict_proba(X)[0][1]  # Probability of class 1 (success)
            inference_latency = (datetime.now() - start_time).total_seconds() * 1000  # in milliseconds
            
            # Calculate confidence (simple heuristic for now)
            confidence = 0.8  # Default confidence
            
            logger.info(f"Predicted bid success for item {item_data.get('id')}: {success_prob:.2f} (confidence: {confidence:.2f})")
            return success_prob, confidence
        
        except Exception as e:
            logger.error(f"Error predicting bid success: {e}")
            raise
    
    def _convert_auctions_to_dataframe(self, auctions: List[AuctionOutcome]) -> pd.DataFrame:
        """
        Convert auction outcomes to DataFrame.
        
        Args:
            auctions: List of auction outcomes
            
        Returns:
            DataFrame 
            auctions: List of auction outcomes
            
        Returns:
            DataFrame
        """
        # Extract relevant fields from auctions
        data = []
        for auction in auctions:
            row = {
                'id': auction.id,
                'item_id': auction.item_id,
                'category_id': auction.category_id,
                'platform': auction.platform,
                'condition': auction.condition if hasattr(auction, 'condition') else 'Unknown',
                'success': auction.status == AuctionStatus.SOLD,
                'bids': auction.bids,
                'watchers': auction.watchers,
                'views': auction.views
            }
            
            # Calculate price ratio if possible
            if auction.final_price is not None and auction.estimated_price is not None and auction.estimated_price > 0:
                row['price_ratio'] = auction.final_price / auction.estimated_price
            else:
                row['price_ratio'] = None
            
            # Add additional fields if available
            if hasattr(auction, 'time_left_percentage'):
                row['time_left_percentage'] = auction.time_left_percentage
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        # Select features and target
        available_features = [f for f in self.features if f in df.columns]
        X = df[available_features]
        y = df[self.target]
        
        return X, y


class ShippingModelTrainer:
    """
    Trainer for shipping models.
    
    This class provides methods for training and updating
    machine learning models for shipping cost and time prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ShippingModelTrainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_type = config.get('model_type', 'gradient_boosting')
        self.features = config.get('features', ['weight', 'dimensions', 'distance', 'service_level'])
        self.target = config.get('target', 'shipping_cost')
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        
        # Initialize model storage
        self.cost_model = None
        self.time_model = None
        
        # Create models directory if it doesn't exist
        os.makedirs('models/shipping', exist_ok=True)
        
        logger.info("ShippingModelTrainer initialized")
    
    async def train_model(
        self,
        shipping_data: List[ShippingPerformance]
    ) -> ModelPerformance:
        """
        Train a shipping model.
        
        Args:
            shipping_data: List of shipping performance data
            
        Returns:
            ModelPerformance object
        """
        logger.info("Training shipping model")
        
        try:
            if not shipping_data:
                raise ValueError("No shipping data available for training")
            
            # Convert shipping data to DataFrame
            df = self._convert_shipping_to_dataframe(shipping_data)
            
            # Train cost model
            cost_performance = await self._train_cost_model(df)
            
            # Train delivery time model
            time_performance = await self._train_time_model(df)
            
            # Return cost model performance (primary model)
            return cost_performance
        
        except Exception as e:
            logger.error(f"Error training shipping model: {e}")
            raise
    
    async def _train_cost_model(self, df: pd.DataFrame) -> ModelPerformance:
        """
        Train shipping cost model.
        
        Args:
            df: DataFrame
            
        Returns:
            ModelPerformance object
        """
        logger.info("Training shipping cost model")
        
        try:
            # Prepare features and target for cost model
            cost_features = [f for f in self.features if f in df.columns]
            X = df[cost_features]
            y = df['actual_cost']
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Create and train model
            start_time = datetime.now()
            
            if self.model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=self.random_state
                )
            else:
                # Default to gradient boosting
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=self.random_state
                )
            
            # Create preprocessing pipeline
            numeric_features = [f for f in X.columns if X[f].dtype in ['int64', 'float64']]
            categorical_features = [f for f in X.columns if X[f].dtype == 'object']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Calculate training duration
            training_duration = (datetime.now() - start_time).total_seconds()
            
            # Evaluate model
            y_pred = pipeline.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate accuracy as 1 - (MAE / mean(y_test))
            accuracy = max(0, min(1, 1 - (mae / np.mean(y_test))))
            
            # Save model
            model_id = f"shipping_cost_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            model_path = f"models/shipping/{model_id}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)
            
            # Store model in memory
            self.cost_model = pipeline
            
            # Create model performance record
            performance = ModelPerformance(
                id=str(uuid.uuid4()),
                model_name=model_id,
                model_version="1.0.0",
                model_type="shipping_cost",
                category_id=None,
                accuracy=accuracy,
                precision=0.0,  # Not applicable for regression
                recall=0.0,  # Not applicable for regression
                f1_score=0.0,  # Not applicable for regression
                mean_absolute_error=mae,
                mean_squared_error=mse,
                r_squared=r2,
                sample_count=len(df),
                training_duration=training_duration,
                inference_latency=0.0,  # Will be updated during inference
                metadata={
                    'model_type': self.model_type,
                    'features': cost_features,
                    'test_size': self.test_size,
                    'random_state': self.random_state
                }
            )
            
            logger.info(f"Trained shipping cost model with accuracy {accuracy:.4f}, MAE {mae:.2f}, R² {r2:.4f}")
            return performance
        
        except Exception as e:
            logger.error(f"Error training shipping cost model: {e}")
            raise
    
    async def _train_time_model(self, df: pd.DataFrame) -> ModelPerformance:
        """
        Train shipping time model.
        
        Args:
            df: DataFrame
            
        Returns:
            ModelPerformance object
        """
        logger.info("Training shipping time model")
        
        try:
            # Filter out rows without actual delivery days
            df_time = df[df['actual_delivery_days'].notna()]
            
            if len(df_time) < 10:
                logger.warning("Not enough data to train shipping time model")
                return None
            
            # Prepare features and target for time model
            time_features = [f for f in self.features if f in df.columns]
            X = df_time[time_features]
            y = df_time['actual_delivery_days']
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Create and train model
            start_time = datetime.now()
            
            if self.model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=self.random_state
                )
            else:
                # Default to gradient boosting
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=self.random_state
                )
            
            # Create preprocessing pipeline
            numeric_features = [f for f in X.columns if X[f].dtype in ['int64', 'float64']]
            categorical_features = [f for f in X.columns if X[f].dtype == 'object']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Calculate training duration
            training_duration = (datetime.now() - start_time).total_seconds()
            
            # Evaluate model
            y_pred = pipeline.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate accuracy as 1 - (MAE / mean(y_test))
            accuracy = max(0, min(1, 1 - (mae / np.mean(y_test))))
            
            # Save model
            model_id = f"shipping_time_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            model_path = f"models/shipping/{model_id}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)
            
            # Store model in memory
            self.time_model = pipeline
            
            # Create model performance record
            performance = ModelPerformance(
                id=str(uuid.uuid4()),
                model_name=model_id,
                model_version="1.0.0",
                model_type="shipping_time",
                category_id=None,
                accuracy=accuracy,
                precision=0.0,  # Not applicable for regression
                recall=0.0,  # Not applicable for regression
                f1_score=0.0,  # Not applicable for regression
                mean_absolute_error=mae,
                mean_squared_error=mse,
                r_squared=r2,
                sample_count=len(df_time),
                training_duration=training_duration,
                inference_latency=0.0,  # Will be updated during inference
                metadata={
                    'model_type': self.model_type,
                    'features': time_features,
                    'test_size': self.test_size,
                    'random_state': self.random_state
                }
            )
            
            logger.info(f"Trained shipping time model with accuracy {accuracy:.4f}, MAE {mae:.2f}, R² {r2:.4f}")
            return performance
        
        except Exception as e:
            logger.error(f"Error training shipping time model: {e}")
            return None
    
    async def incorporate_feedback(
        self,
        item_id: str,
        ai_suggestion: str,
        user_correction: str,
        rating: int
    ) -> None:
        """
        Incorporate user feedback into the model.
        
        Args:
            item_id: Item ID
            ai_suggestion: AI-suggested shipping option
            user_correction: User-corrected shipping option
            rating: User rating (1-5)
        """
        logger.debug(f"Incorporating feedback for item {item_id}")
        
        try:
            # For now, just log the feedback
            # In a real implementation, this would update the model
            # or store the feedback for batch retraining
            logger.info(f"Shipping feedback: item={item_id}, ai={ai_suggestion}, user={user_correction}, rating={rating}")
        
        except Exception as e:
            logger.error(f"Error incorporating shipping feedback: {e}")
    
    async def predict_shipping(
        self,
        package_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict shipping cost and delivery time.
        
        Args:
            package_data: Package data
            
        Returns:
            Dictionary with shipping predictions
        """
        logger.debug(f"Predicting shipping for package")
        
        try:
            # Convert package data to DataFrame
            df = pd.DataFrame([package_data])
            
            # Predict shipping cost
            cost_prediction = None
            cost_confidence = 0.0
            
            if self.cost_model:
                # Prepare features
                cost_features = [f for f in self.features if f in df.columns]
                X = df[cost_features]
                
                # Predict cost
                start_time = datetime.now()
                cost_prediction = self.cost_model.predict(X)[0]
                inference_latency = (datetime.now() - start_time).total_seconds() * 1000  # in milliseconds
                
                # Calculate confidence (simple heuristic for now)
                cost_confidence = 0.8  # Default confidence
            
            # Predict delivery time
            time_prediction = None
            time_confidence = 0.0
            
            if self.time_model:
                # Prepare features
                time_features = [f for f in self.features if f in df.columns]
                X = df[time_features]
                
                # Predict time
                start_time = datetime.now()
                time_prediction = self.time_model.predict(X)[0]
                inference_latency = (datetime.now() - start_time).total_seconds() * 1000  # in milliseconds
                
                # Calculate confidence (simple heuristic for now)
                time_confidence = 0.7  # Default confidence
            
            # Return predictions
            result = {
                'cost_prediction': cost_prediction,
                'cost_confidence': cost_confidence,
                'time_prediction': time_prediction,
                'time_confidence': time_confidence
            }
            
            logger.info(f"Predicted shipping: cost=${cost_prediction:.2f} (confidence: {cost_confidence:.2f}), time={time_prediction:.1f} days (confidence: {time_confidence:.2f})")
            return result
        
        except Exception as e:
            logger.error(f"Error predicting shipping: {e}")
            raise
    
    def _convert_shipping_to_dataframe(self, shipping_data: List[ShippingPerformance]) -> pd.DataFrame:
        """
        Convert shipping performance data to DataFrame.
        
        Args:
            shipping_data: List of shipping performance data
            
        Returns:
            DataFrame
        """
        # Extract relevant fields from shipping data
        data = []
        for shipping in shipping_data:
            row = {
                'id': shipping.id,
                'auction_id': shipping.auction_id,
                'carrier': shipping.carrier,
                'service_level': shipping.service_level,
                'package_weight': shipping.package_weight,
                'origin_zip': shipping.origin_zip,
                'destination_zip': shipping.destination_zip,
                'estimated_cost': shipping.estimated_cost,
                'actual_cost': shipping.actual_cost,
                'estimated_delivery_days': shipping.estimated_delivery_days,
                'actual_delivery_days': shipping.actual_delivery_days,
                'status': shipping.status.value
            }
            
            # Add package dimensions
            if shipping.package_dimensions:
                for dim, value in shipping.package_dimensions.items():
                    row[f'dimension_{dim}'] = value
            
            # Calculate package volume
            if shipping.package_dimensions and 'length' in shipping.package_dimensions and 'width' in shipping.package_dimensions and 'height' in shipping.package_dimensions:
                row['package_volume'] = shipping.package_dimensions['length'] * shipping.package_dimensions['width'] * shipping.package_dimensions['height']
            
            data.append(row)
        
        return pd.DataFrame(data)