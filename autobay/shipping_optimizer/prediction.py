"""
Machine learning prediction module for shipping costs.

This module provides functionality to predict shipping costs
using machine learning models when real-time rates are unavailable.
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from shipping_optimizer.models import Package, ShippingRate
from shipping_optimizer.db import ShippingDatabase


logger = logging.getLogger(__name__)


class ShippingPredictor:
    """
    Machine learning predictor for shipping costs.
    
    This class provides functionality to predict shipping costs
    using machine learning models when real-time rates are unavailable.
    """
    
    def __init__(
        self,
        model_dir: str = "models",
        db: Optional[ShippingDatabase] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the ShippingPredictor.
        
        Args:
            model_dir: Directory to store trained models
            db: ShippingDatabase instance
            config_path: Path to configuration file
        """
        self.model_dir = model_dir
        self.db = db
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize models
        self.models = {}
        self.preprocessors = {}
        
        # Load pre-trained models if available
        self._load_models()
        
        logger.info("ShippingPredictor initialized")
    
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
                "algorithm": "random_forest",
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1
                },
                "gradient_boosting": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3
                },
                "linear_regression": {}
            },
            "training": {
                "test_size": 0.2,
                "random_state": 42,
                "min_samples": 50
            },
            "features": {
                "numeric": [
                    "weight_lb",
                    "length_in",
                    "width_in",
                    "height_in",
                    "volume_cubic_in",
                    "distance_miles",
                    "zone"
                ],
                "categorical": [
                    "carrier",
                    "service",
                    "origin_state",
                    "destination_state",
                    "is_residential"
                ]
            },
            "seasonal_factors": {
                "holiday_surcharge": 0.1,  # 10% surcharge during holidays
                "holiday_periods": [
                    {"start": "12-01", "end": "12-31"},  # December
                    {"start": "11-15", "end": "11-30"}   # Late November
                ],
                "day_of_week_factors": {
                    "0": 1.0,  # Monday
                    "1": 1.0,  # Tuesday
                    "2": 1.0,  # Wednesday
                    "3": 1.0,  # Thursday
                    "4": 1.05, # Friday
                    "5": 1.1,  # Saturday
                    "6": 1.1   # Sunday
                }
            }
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
    
    def _load_models(self) -> None:
        """Load pre-trained models if available."""
        carriers = ["usps", "fedex", "ups", "dhl"]
        
        for carrier in carriers:
            model_path = os.path.join(self.model_dir, f"{carrier}_model.pkl")
            preprocessor_path = os.path.join(self.model_dir, f"{carrier}_preprocessor.pkl")
            
            if os.path.exists(model_path) and os.path.exists(preprocessor_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models[carrier] = pickle.load(f)
                    
                    with open(preprocessor_path, 'rb') as f:
                        self.preprocessors[carrier] = pickle.load(f)
                    
                    logger.info(f"Loaded pre-trained model for {carrier}")
                except Exception as e:
                    logger.error(f"Error loading model for {carrier}: {e}")
    
    def _save_model(self, carrier: str, model: Any, preprocessor: Any) -> None:
        """
        Save a trained model to disk.
        
        Args:
            carrier: Carrier name
            model: Trained model
            preprocessor: Data preprocessor
        """
        model_path = os.path.join(self.model_dir, f"{carrier}_model.pkl")
        preprocessor_path = os.path.join(self.model_dir, f"{carrier}_preprocessor.pkl")
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            
            logger.info(f"Saved model for {carrier}")
        except Exception as e:
            logger.error(f"Error saving model for {carrier}: {e}")
    
    def _create_model(self) -> Any:
        """
        Create a new model based on configuration.
        
        Returns:
            Machine learning model
        """
        algorithm = self.config["model_params"]["algorithm"]
        
        if algorithm == "random_forest":
            params = self.config["model_params"]["random_forest"]
            return RandomForestRegressor(**params)
        elif algorithm == "gradient_boosting":
            params = self.config["model_params"]["gradient_boosting"]
            return GradientBoostingRegressor(**params)
        elif algorithm == "linear_regression":
            params = self.config["model_params"]["linear_regression"]
            return LinearRegression(**params)
        else:
            logger.warning(f"Unknown algorithm {algorithm}, using RandomForestRegressor")
            return RandomForestRegressor()
    
    def _create_preprocessor(self) -> Any:
        """
        Create a data preprocessor.
        
        Returns:
            Data preprocessor
        """
        numeric_features = self.config["features"]["numeric"]
        categorical_features = self.config["features"]["categorical"]
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    def train_model(
        self,
        carrier: str,
        data: Optional[pd.DataFrame] = None
    ) -> Tuple[float, float]:
        """
        Train a model for a specific carrier.
        
        Args:
            carrier: Carrier name
            data: Training data (if None, will be loaded from database)
            
        Returns:
            Tuple of (mean squared error, R² score)
        """
        # Load data if not provided
        if data is None:
            if self.db is None:
                logger.error("No database connection available")
                return (float('inf'), 0.0)
            
            data = self.db.get_shipping_rates_data(carrier=carrier)
        
        if len(data) < self.config["training"]["min_samples"]:
            logger.warning(f"Insufficient data for {carrier}: {len(data)} samples")
            return (float('inf'), 0.0)
        
        try:
            # Prepare data
            X = data.drop(['rate'], axis=1)
            y = data['rate']
            
            # Create preprocessor
            preprocessor = self._create_preprocessor()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config["training"]["test_size"],
                random_state=self.config["training"]["random_state"]
            )
            
            # Preprocess data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Create and train model
            model = self._create_model()
            model.fit(X_train_processed, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_processed)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Save model
            self.models[carrier] = model
            self.preprocessors[carrier] = preprocessor
            self._save_model(carrier, model, preprocessor)
            
            logger.info(f"Trained model for {carrier}: MSE={mse:.4f}, R²={r2:.4f}")
            return (mse, r2)
        
        except Exception as e:
            logger.error(f"Error training model for {carrier}: {e}")
            return (float('inf'), 0.0)
    
    def train_all_models(self) -> Dict[str, Tuple[float, float]]:
        """
        Train models for all carriers.
        
        Returns:
            Dictionary of carrier to (MSE, R²) tuples
        """
        if self.db is None:
            logger.error("No database connection  R²) tuples
        """
        if self.db is None:
            logger.error("No database connection available")
            return {}
        
        results = {}
        carriers = ["usps", "fedex", "ups", "dhl"]
        
        for carrier in carriers:
            logger.info(f"Training model for {carrier}")
            mse, r2 = self.train_model(carrier)
            results[carrier] = (mse, r2)
        
        return results
    
    def predict_cost(
        self,
        package: Package,
        distance: float,
        zone: int,
        carrier: str,
        service: str
    ) -> float:
        """
        Predict shipping cost using trained model.
        
        Args:
            package: Package details
            distance: Distance in miles
            zone: Shipping zone
            carrier: Carrier name
            service: Service name
            
        Returns:
            Predicted shipping cost
        """
        carrier_key = carrier.lower()
        
        # Check if model exists
        if carrier_key not in self.models or carrier_key not in self.preprocessors:
            logger.warning(f"No model available for {carrier}")
            return self._fallback_prediction(package, distance, zone, carrier, service)
        
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                'weight_lb': [package.weight_lb],
                'length_in': [package.length_in],
                'width_in': [package.width_in],
                'height_in': [package.height_in],
                'volume_cubic_in': [package.volume_cubic_in],
                'distance_miles': [distance],
                'zone': [zone],
                'carrier': [carrier_key],
                'service': [service.lower()],
                'origin_state': ['XX'],  # Placeholder
                'destination_state': ['XX'],  # Placeholder
                'is_residential': [True]  # Default to residential
            })
            
            # Preprocess data
            input_processed = self.preprocessors[carrier_key].transform(input_data)
            
            # Make prediction
            predicted_cost = self.models[carrier_key].predict(input_processed)[0]
            
            # Apply seasonal adjustments
            predicted_cost = self._apply_seasonal_adjustments(predicted_cost)
            
            # Ensure prediction is positive
            predicted_cost = max(0.01, predicted_cost)
            
            logger.info(f"Predicted cost for {carrier} {service}: ${predicted_cost:.2f}")
            return predicted_cost
        
        except Exception as e:
            logger.error(f"Error predicting cost: {e}")
            return self._fallback_prediction(package, distance, zone, carrier, service)
    
    def _fallback_prediction(
        self,
        package: Package,
        distance: float,
        zone: int,
        carrier: str,
        service: str
    ) -> float:
        """
        Fallback prediction when model is not available.
        
        Args:
            package: Package details
            distance: Distance in miles
            zone: Shipping zone
            carrier: Carrier name
            service: Service name
            
        Returns:
            Predicted shipping cost
        """
        logger.info(f"Using fallback prediction for {carrier} {service}")
        
        # Base rate per pound
        base_rates = {
            "usps": {
                "priority": 7.50,
                "first class": 4.50,
                "ground": 6.00,
                "express": 25.00
            },
            "fedex": {
                "ground": 8.50,
                "express": 30.00,
                "2day": 20.00,
                "overnight": 40.00
            },
            "ups": {
                "ground": 9.00,
                "3 day select": 15.00,
                "2nd day air": 25.00,
                "next day air": 45.00
            },
            "dhl": {
                "express": 35.00,
                "express 9:00": 50.00,
                "express 10:30": 45.00,
                "express 12:00": 40.00
            }
        }
        
        # Get base rate
        carrier_key = carrier.lower()
        service_key = service.lower()
        
        if carrier_key in base_rates:
            service_rates = base_rates[carrier_key]
            base_rate = next(
                (rate for svc, rate in service_rates.items() if svc in service_key),
                10.00  # Default if no match
            )
        else:
            base_rate = 10.00
        
        # Calculate weight-based cost
        weight_cost = base_rate + (package.billable_weight_lb - 1) * (base_rate * 0.5)
        
        # Add distance factor
        distance_factor = 1.0 + (distance / 1000.0)
        
        # Add zone factor
        zone_factor = 1.0 + ((zone - 1) * 0.1)
        
        # Calculate total cost
        predicted_cost = weight_cost * distance_factor * zone_factor
        
        # Apply seasonal adjustments
        predicted_cost = self._apply_seasonal_adjustments(predicted_cost)
        
        return predicted_cost
    
    def _apply_seasonal_adjustments(self, cost: float) -> float:
        """
        Apply seasonal adjustments to predicted cost.
        
        Args:
            cost: Base predicted cost
            
        Returns:
            Adjusted cost
        """
        today = datetime.now()
        
        # Check if current date is in a holiday period
        is_holiday = False
        for period in self.config["seasonal_factors"]["holiday_periods"]:
            start_month, start_day = map(int, period["start"].split("-"))
            end_month, end_day = map(int, period["end"].split("-"))
            
            start_date = datetime(today.year, start_month, start_day)
            end_date = datetime(today.year, end_month, end_day)
            
            if start_date <= today <= end_date:
                is_holiday = True
                break
        
        # Apply holiday surcharge if applicable
        if is_holiday:
            holiday_surcharge = self.config["seasonal_factors"]["holiday_surcharge"]
            cost *= (1 + holiday_surcharge)
        
        # Apply day of week factor
        day_of_week = str(today.weekday())
        day_factor = self.config["seasonal_factors"]["day_of_week_factors"].get(day_of_week, 1.0)
        cost *= day_factor
        
        return cost