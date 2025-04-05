"""
Core functionality for the Shipping Optimization Module.

This module provides the main ShippingOptimizer class that handles
rate retrieval, carrier selection, and shipping optimization.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import requests
from concurrent.futures import ThreadPoolExecutor

from shipping_optimizer.models import (
    Package, Address, ShippingRate, ShippingOption,
    ShippingPreference, ShippingHistory, CarrierPerformance,
    PriceAdjustment
)
from shipping_optimizer.carriers import (
    USPSClient, FedExClient, UPSClient, DHLClient
)
from shipping_optimizer.prediction import ShippingPredictor
from shipping_optimizer.cache import RateCache
from shipping_optimizer.db import ShippingDatabase
from shipping_optimizer.utils import (
    calculate_distance, calculate_zone, validate_address,
    calculate_packaging_cost, log_api_request
)


logger = logging.getLogger(__name__)


class ShippingOptimizer:
    """
    Main class for shipping optimization.
    
    This class provides methods for retrieving shipping rates,
    selecting the best carrier, and optimizing shipping costs.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        db_connection: Optional[str] = None,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,  # 1 hour
        max_workers: int = 4
    ):
        """
        Initialize the ShippingOptimizer.
        
        Args:
            config_path: Path to configuration file
            db_connection: Database connection string
            cache_enabled: Whether to enable rate caching
            cache_ttl: Cache time-to-live in seconds
            max_workers: Maximum number of concurrent API requests
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize database connection
        self.db = ShippingDatabase(db_connection or self.config.get('db_connection'))
        
        # Initialize cache
        self.cache_enabled = cache_enabled
        self.cache = RateCache(ttl=cache_ttl) if cache_enabled else None
        
        # Initialize carrier clients
        self.carriers = self._initialize_carriers()
        
        # Initialize shipping predictor
        self.predictor = ShippingPredictor(
            model_dir=self.config.get('model_dir', 'models'),
            db=self.db
        )
        
        # Initialize thread pool for concurrent API requests
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info("ShippingOptimizer initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'api_keys': {
                'usps': os.environ.get('USPS_API_KEY', ''),
                'fedex': os.environ.get('FEDEX_API_KEY', ''),
                'ups': os.environ.get('UPS_API_KEY', ''),
                'dhl': os.environ.get('DHL_API_KEY', '')
            },
            'db_connection': os.environ.get('SHIPPING_DB_CONNECTION', 'sqlite:///shipping.db'),
            'model_dir': 'models',
            'rate_request_timeout': 10,  # seconds
            'max_retries': 3,
            'retry_delay': 1,  # seconds
            'default_preferences': {
                'cost_importance': 0.5,
                'speed_importance': 0.3,
                'reliability_importance': 0.2
            },
            'carrier_settings': {
                'usps': {'enabled': True, 'timeout': 10},
                'fedex': {'enabled': True, 'timeout': 10},
                'ups': {'enabled': True, 'timeout': 10},
                'dhl': {'enabled': True, 'timeout': 10}
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
    
    def _initialize_carriers(self) -> Dict[str, Any]:
        """
        Initialize carrier API clients.
        
        Returns:
            Dictionary of carrier clients
        """
        carriers = {}
        
        # Initialize USPS client
        if self.config['carrier_settings']['usps']['enabled']:
            try:
                carriers['usps'] = USPSClient(
                    api_key=self.config['api_keys']['usps'],
                    timeout=self.config['carrier_settings']['usps']['timeout']
                )
                logger.info("USPS client initialized")
            except Exception as e:
                logger.error(f"Error initializing USPS client: {e}")
        
        # Initialize FedEx client
        if self.config['carrier_settings']['fedex']['enabled']:
            try:
                carriers['fedex'] = FedExClient(
                    api_key=self.config['api_keys']['fedex'],
                    timeout=self.config['carrier_settings']['fedex']['timeout']
                )
                logger.info("FedEx client initialized")
            except Exception as e:
                logger.error(f"Error initializing FedEx client: {e}")
        
        # Initialize UPS client
        if self.config['carrier_settings']['ups']['enabled']:
            try:
                carriers['ups'] = UPSClient(
                    api_key=self.config['api_keys']['ups'],
                    timeout=self.config['carrier_settings']['ups']['timeout']
                )
                logger.info("UPS client initialized")
            except Exception as e:
                logger.error(f"Error initializing UPS client: {e}")
        
        # Initialize DHL client
        if self.config['carrier_settings']['dhl']['enabled']:
            try:
                carriers['dhl'] = DHLClient(
                    api_key=self.config['api_keys']['dhl'],
                    timeout=self.config['carrier_settings']['dhl']['timeout']
                )
                logger.info("DHL client initialized")
            except Exception as e:
                logger.error(f"Error initializing DHL client: {e}")
        
        return carriers
    
    def get_shipping_rates(
        self,
        package: Package,
        origin: Address,
        destination: Address,
        carriers: Optional[List[str]] = None,
        services: Optional[Dict[str, List[str]]] = None,
        use_cache: bool = True
    ) -> List[ShippingRate]:
        """
        Get shipping rates from multiple carriers.
        
        Args:
            package: Package details
            origin: Origin address
            destination: Destination address
            carriers: List of carriers to query (default: all enabled carriers)
            services: Dictionary of carrier-specific services to query
            use_cache: Whether to use cached rates
            
        Returns:
            List of shipping rates
        """
        # Validate addresses
        if not validate_address(origin) or not validate_address(destination):
            logger.warning("Invalid origin or destination address")
            return []
        
        # Check cache if enabled
        cache_key = None
        if self.cache_enabled and use_cache:
            cache_key = self._generate_cache_key(package, origin, destination, carriers, services)
            cached_rates = self.cache.get(cache_key)
            if cached_rates:
                logger.info(f"Using cached rates for {cache_key}")
                return cached_rates
        
        # Determine which carriers to query
        carriers_to_query = carriers or list(self.carriers.keys())
        
        # Filter by enabled carriers
        carriers_to_query = [
            carrier for carrier in carriers_to_query
            if carrier in self.carriers
        ]
        
        if not carriers_to_query:
            logger.warning("No enabled carriers to query")
            return []
        
        # Query carriers concurrently
        futures = {}
        for carrier in carriers_to_query:
            carrier_services = services.get(carrier) if services else None
            futures[carrier] = self.executor.submit(
                self._get_carrier_rates,
                carrier,
                package,
                origin,
                destination,
                carrier_services
            )
        
        # Collect results
        all_rates = []
        for carrier, future in futures.items():
            try:
                rates = future.result()
                all_rates.extend(rates)
                logger.info(f"Retrieved {len(rates)} rates from {carrier}")
            except Exception as e:
                logger.error(f"Error retrieving rates from {carrier}: {e}")
        
        # Cache results if enabled
        if self.cache_enabled and use_cache and cache_key:
            self.cache.set(cache_key, all_rates)
        
        return all_rates
    
    def _generate_cache_key(
        self,
        package: Package,
        origin: Address,
        destination: Address,
        carriers: Optional[List[str]],
        services: Optional[Dict[str, List[str]]]
    ) -> str:
        """
        Generate a cache key for shipping rate queries.
        
        Args:
            package: Package details
            origin: Origin address
            destination: Destination address
            carriers: List of carriers to query
            services: Dictionary of carrier-specific services to query
            
        Returns:
            Cache key string
        """
        # Create a dictionary of key components
        key_dict = {
            'package': package.to_dict(),
            'origin': origin.to_dict(),
            'destination': destination.to_dict(),
            'carriers': carriers,
            'services': services
        }
        
        # Convert to JSON and hash
        key_json = json.dumps(key_dict, sort_keys=True)
        return f"rates:{hash(key_json)}"
    
    def _get_carrier_rates(
        self,
        carrier: str,
        package: Package,
        origin: Address,
        destination: Address,
        services: Optional[List[str]] = None
    ) -> List[ShippingRate]:
        """
        Get shipping rates from a specific carrier.
        
        Args:
            carrier: Carrier name
            package: Package details
            origin: Origin address
            destination: Destination address
            services: List of services to query
            
        Returns:
            List of shipping rates
        """
        carrier_client = self.carriers.get(carrier)
        if not carrier_client:
            logger.warning(f"Carrier {carrier} not initialized")
            return []
        
        try:
            # Log API request
            log_api_request(carrier, 'get_rates')
            
            # Get rates from carrier
            rates = carrier_client.get_rates(
                package=package,
                origin=origin,
                destination=destination,
                services=services
            )
            
            return rates
        except Exception as e:
            logger.error(f"Error getting rates from {carrier}: {e}")
            return []
    
    def predict_shipping_cost(
        self,
        package: Package,
        origin: Address,
        destination: Address,
        carrier: str,
        service: str
    ) -> float:
        """
        Predict shipping cost using machine learning model.
        
        Args:
            package: Package details
            origin: Origin address
            destination: Destination address
            carrier: Carrier name
            service: Service name
            
        Returns:
            Predicted shipping cost
        """
        try:
            # Calculate distance and zone
            distance = calculate_distance(origin, destination)
            zone = calculate_zone(origin.postal_code, destination.postal_code)
            
            # Predict cost
            predicted_cost = self.predictor.predict_cost(
                package=package,
                distance=distance,
                zone=zone,
                carrier=carrier,
                service=service
            )
            
            logger.info(f"Predicted shipping cost for {carrier} {service}: ${predicted_cost:.2f}")
            return predicted_cost
        except Exception as e:
            logger.error(f"Error predicting shipping cost: {e}")
            return 0.0
    
    def select_best_shipping(
        self,
        rates: List[ShippingRate],
        preferences: Optional[ShippingPreference] = None,
        package: Optional[Package] = None
    ) -> List[ShippingOption]:
        """
        Select the best shipping option based on rates and preferences.
        
        Args:
            rates: List of shipping rates
            preferences: User shipping preferences
            package: Package details (for additional context)
            
        Returns:
            List of shipping options sorted by overall score
        """
        if not rates:
            logger.warning("No rates provided for selection")
            return []
        
        # Use default preferences if not provided
        if not preferences:
            default_prefs = self.config['default_preferences']
            preferences = ShippingPreference(
                user_id="default",
                cost_importance=default_prefs['cost_importance'],
                speed_importance=default_prefs['speed_importance'],
                reliability_importance=default_prefs['reliability_importance']
            )
        
        # Filter out excluded carriers
        if preferences.excluded_carriers:
            rates = [
                rate for rate in rates
                if rate.carrier.lower() not in [c.lower() for c in preferences.excluded_carriers]
            ]
        
        if not rates:
            logger.warning("No rates available after filtering excluded carriers")
            return []
        
        # Get carrier performance metrics
        carrier_performance = self._get_carrier_performance()
        
        # Calculate scores for each rate
        options = []
        for rate in rates:
            # Calculate cost score (lower is better)
            min_cost = min(r.total_cost for r in rates)
            max_cost = max(r.total_cost for r in rates)
            cost_range = max_cost - min_cost
            
            if cost_range > 0:
                cost_score = 1.0 - ((rate.total_cost - min_cost) / cost_range)
            else:
                cost_score = 1.0
            
            # Calculate speed score (lower delivery days is better)
            if rate.delivery_days is not None:
                min_days = min((r.delivery_days for r in rates if r.delivery_days is not None), default=1)
                max_days = max((r.delivery_days for r in rates if r.delivery_days is not None), default=7)
                days_range = max_days - min_days
                
                if days_range > 0:
                    speed_score = 1.0 - ((rate.delivery_days - min_days) / days_range)
                else:
                    speed_score = 1.0
            else:
                # If delivery days not provided, use a default score
                speed_score = 0.5
            
            # Calculate reliability score based on carrier performance
            perf = next(
                (p for p in carrier_performance if p.carrier.lower() == rate.carrier.lower() and 
                 p.service.lower() == rate.service.lower()),
                None
            )
            
            if perf:
                on_time_weight = 0.5
                damage_weight = 0.3
                satisfaction_weight = 0.2
                
                reliability_score = (
                    (perf.on_time_percentage / 100.0) * on_time_weight +
                    (1.0 - (perf.damage_percentage / 100.0)) * damage_weight +
                    (perf.customer_satisfaction / 5.0) * satisfaction_weight
                )
            else:
                # Default reliability score if no performance data
                reliability_score = 0.7
            
            # Apply preference weights to calculate overall score
            overall_score = (
                cost_score * preferences.cost_importance +
                speed_score * preferences.speed_importance +
                reliability_score * preferences.reliability_importance
            )
            
            # Create shipping option
            option = ShippingOption(
                rate=rate,
                cost_score=cost_score,
                speed_score=speed_score,
                reliability_score=reliability_score,
                overall_score=overall_score
            )
            
            options.append(option)
        
        # Sort options by overall score (descending)
        options.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Mark the top option as recommended
        if options:
            options[0].is_recommended = True
        
        return options
    
    def _get_carrier_performance(self) -> List[CarrierPerformance]:
        """
        Get carrier performance metrics from historical data.
        
        Returns:
            List of carrier performance metrics
        """
        try:
            # Get performance metrics from database
            performance = self.db.get_carrier_performance()
            
            # If no data in database, use default values
            if not performance:
                logger.warning("No carrier performance data available, using defaults")
                
                # Create default performance metrics for major carriers
                performance = [
                    CarrierPerformance(
                        carrier="USPS",
                        service="Priority",
                        avg_delivery_days=2.5,
                        on_time_percentage=92.0,
                        damage_percentage=1.5,
                        loss_percentage=0.8,
                        cost_accuracy=0.95,
                        customer_satisfaction=4.2,
                        total_shipments=1000
                    ),
                    CarrierPerformance(
                        carrier="USPS",
                        service="First Class",
                        avg_delivery_days=3.0,
                        on_time_percentage=90.0,
                        damage_percentage=1.8,
                        loss_percentage=1.0,
                        cost_accuracy=0.97,
                        customer_satisfaction=4.0,
                        total_shipments=1500
                    ),
                    CarrierPerformance(
                        carrier="FedEx",
                        service="Ground",
                        avg_delivery_days=2.8,
                        on_time_percentage=94.0,
                        damage_percentage=1.2,
                        loss_percentage=0.5,
                        cost_accuracy=0.96,
                        customer_satisfaction=4.3,
                        total_shipments=1200
                    ),
                    CarrierPerformance(
                        carrier="FedEx",
                        service="Express",
                        avg_delivery_days=1.5,
                        on_time_percentage=96.0,
                        damage_percentage=1.0,
                        loss_percentage=0.3,
                        cost_accuracy=0.98,
                        customer_satisfaction=4.5,
                        total_shipments=800
                    ),
                    CarrierPerformance(
                        carrier="UPS",
                        service="Ground",
                        avg_delivery_days=2.7,
                        on_time_percentage=93.0,
                        damage_percentage=1.3,
                        loss_percentage=0.6,
                        cost_accuracy=0.95,
                        customer_satisfaction=4.2,
                        total_shipments=1100
                    ),
                    CarrierPerformance(
                        carrier="UPS",
                        service="Next Day Air",
                        avg_delivery_days=1.2,
                        on_time_percentage=97.0,
                        damage_percentage=0.9,
                        loss_percentage=0.2,
                        cost_accuracy=0.99,
                        customer_satisfaction=4.6,
                        total_shipments=600
                    ),
                    CarrierPerformance(
                        carrier="DHL",
                        service="Express",
                        avg_delivery_days=2.0,
                        on_time_percentage=95.0,
                        damage_percentage=1.1,
                        loss_percentage=0.4,
                        cost_accuracy=0.97,
                        customer_satisfaction=4.4,
                        total_shipments=700
                    )
                ]
            
            return performance
        except Exception as e:
            logger.error(f"Error getting carrier performance: {e}")
            return []
    
    def adjust_price_for_shipping(
        self,
        item_price: float,
        shipping_cost: float,
        handling_cost: float = 0.0,
        packaging_cost: Optional[float] = None,
        return_rate: float = 0.03,
        target_margin: float = 0.2
    ) -> PriceAdjustment:
        """
        Adjust item price based on shipping and handling costs.
        
        Args:
            item_price: Original item price
            shipping_cost: Shipping cost
            handling_cost: Handling cost
            packaging_cost: Packaging cost (if None, will be estimated)
            return_rate: Expected return rate (0.0 to 1.0)
            target_margin: Target profit margin (0.0 to 1.0)
            
        Returns:
            Price adjustment details
        """
        try:
            # Estimate packaging cost if not provided
            if packaging_cost is None:
                packaging_cost = calculate_packaging_cost(item_price)
            
            # Calculate total fulfillment cost
            fulfillment_cost = shipping_cost + handling_cost + packaging_cost
            
            # Factor in returns
            adjusted_fulfillment_cost = fulfillment_cost * (1.0 + return_rate)
            
            # Calculate adjusted price to maintain target margin
            total_cost = item_price + adjusted_fulfillment_cost
            adjusted_price = total_cost / (1.0 - target_margin)
            
            # Create price adjustment
            adjustment = PriceAdjustment(
                original_price=item_price,
                adjusted_price=adjusted_price,
                shipping_cost=shipping_cost,
                handling_cost=handling_cost,
                packaging_cost=packaging_cost,
                return_rate=return_rate,
                profit_margin=target_margin
            )
            
            logger.info(f"Adjusted price from ${item_price:.2f} to ${adjusted_price:.2f}")
            return adjustment
        except Exception as e:
            logger.error(f"Error adjusting price: {e}")
            return PriceAdjustment(
                original_price=item_price,
                adjusted_price=item_price,
                shipping_cost=shipping_cost,
                handling_cost=handling_cost,
                packaging_cost=packaging_cost or 0.0,
                return_rate=return_rate,
                profit_margin=0.0
            )
    
    def analyze_shipping_history(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze shipping history to identify patterns and optimization opportunities.
        
        Args:
            user_id: Filter by user ID
            start_date: Start date for analysis
            end_date: End date for analysis
            limit: Maximum number of records to analyze
            
        Returns:
            Analysis results
        """
        try:
            # Get shipping history from database
            history = self.db.get_shipping_history(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            
            if not history:
                logger.warning("No shipping history available for analysis")
                return {
                    "total_shipments": 0,
                    "total_cost": 0.0,
                    "avg_cost": 0.0,
                    "carrier_breakdown": {},
                    "service_breakdown": {},
                    "cost_accuracy": 0.0,
                    "optimization_opportunities": []
                }
            
            # Calculate basic metrics
            total_cost = sum(h.actual_cost for h in history)
            avg_cost = total_cost / len(history)
            
            # Analyze carrier and service usage
            carrier_breakdown = {}
            service_breakdown = {}
            
            for h in history:
                carrier = h.selected_rate.carrier
                service = h.selected_rate.service
                
                if carrier not in carrier_breakdown:
                    carrier_breakdown[carrier] = {
                        "count": 0,
                        "total_cost": 0.0,
                        "avg_cost": 0.0
                    }
                
                carrier_breakdown[carrier]["count"] += 1
                carrier_breakdown[carrier]["total_cost"] += h.actual_cost
                
                service_key = f"{carrier} - {service}"
                if service_key not in service_breakdown:
                    service_breakdown[service_key] = {
                        "count": 0,
                        "total_cost": 0.0,
                        "avg_cost": 0.0,
                        "avg_delivery_days": 0.0
                    }
                
                service_breakdown[service_key]["count"] += 1
                service_breakdown[service_key]["total_cost"] += h.actual_cost
                
                if h.delivery_days is not None:
                    current_total = service_breakdown[service_key]["avg_delivery_days"] * (service_breakdown[service_key]["count"] - 1)
                    service_breakdown[service_key]["avg_delivery_days"] = (current_total + h.delivery_days) / service_breakdown[service_key]["count"]
            
            # Calculate averages
            for carrier in carrier_breakdown:
                carrier_breakdown[carrier]["avg_cost"] = carrier_breakdown[carrier]["total_cost"] / carrier_breakdown[carrier]["count"]
            
            for service in service_breakdown:
                service_breakdown[service]["avg_cost"] = service_breakdown[service]["total_cost"] / service_breakdown[service]["count"]
            
            # Calculate cost accuracy
            cost_accuracy_values = [h.cost_accuracy for h in history if h.cost_accuracy is not None]
            avg_cost_accuracy = sum(cost_accuracy_values) / len(cost_accuracy_values) if cost_accuracy_values else 0.0
            
            # Identify optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(history, carrier_breakdown, service_breakdown)
            
            # Prepare results
            results = {
                "total_shipments": len(history),
                "total_cost": total_cost,
                "avg_cost": avg_cost,
                "carrier_breakdown": carrier_breakdown,
                "service_breakdown": service_breakdown,
                "cost_accuracy": avg_cost_accuracy,
                "optimization_opportunities": optimization_opportunities
            }
            
            logger.info(f"Analyzed {len(history)} shipping records")
            return results
        except Exception as e:
            logger.error(f"Error analyzing shipping history: {e}")
            return {
                "error": str(e),
                "total_shipments": 0,
                "total_cost": 0.0,
                "avg_cost": 0.0,
                "carrier_breakdown": {},
                "service_breakdown": {},
                "cost_accuracy": 0.0,
                "optimization_opportunities": []
            }
    
    def _identify_optimization_opportunities(
        self,
        history: List[ShippingHistory],
        carrier_breakdown: Dict[str, Dict[str, Any]],
        service_breakdown: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify shipping optimization opportunities.
        
        Args:
            history: Shipping history records
            carrier_breakdown: Carrier usage breakdown
            service_breakdown: Service usage breakdown
            
        Returns:
            List of optimization opportunities
        """
        opportunities = []
        
        # Check for overuse of premium services
        premium_services = ["Express", "Priority", "Next Day", "2nd Day"]
        premium_usage = {}
        
        for service, data in service_breakdown.items():
            if any(premium in service for premium in premium_services):
                premium_usage[service] = data
        
        if premium_usage:
            premium_count = sum(data["count"] for data in premium_usage.values())
            premium_percentage = premium_count / len(history) * 100
            
            if premium_percentage > 30:
                opportunities.append({
                    "type": "premium_service_overuse",
                    "description": f"Premium shipping services used for {premium_percentage:.1f}% of shipments",
                    "potential_savings": sum(data["total_cost"] * 0.3 for data in premium_usage.values()),
                    "recommendation": "Consider using standard shipping for non-urgent items"
                })
        
        # Check for consistent overpayment with specific carriers
        for carrier, data in carrier_breakdown.items():
            if data["count"] >= 10 and data["avg_cost"] > avg_cost * 1.2:
                opportunities.append({
                    "type": "carrier_overpayment",
                    "description": f"{carrier} shipments cost {(data['avg_cost'] / avg_cost - 1) * 100:.1f}% more than average",
                    "potential_savings": (data["avg_cost"] - avg_cost) * data["count"],
                    "recommendation": f"Evaluate alternative carriers to {carrier}"
                })
        
        # Check for packaging optimization
        package_sizes = {}
        for h in history:
            volume = h.package.volume_cubic_in
            weight = h.package.weight_oz
            
            size_key = f"{int(volume / 100)}_{int(weight / 16)}"
            if size_key not in package_sizes:
                package_sizes[size_key] = {
                    "count": 0,
                    "total_cost": 0.0,
                    "avg_cost": 0.0,
                    "volume": volume,
                    "weight": weight
                }
            
            package_sizes[size_key]["count"] += 1
            package_sizes[size_key]["total_cost"] += h.actual_cost
        
        # Find common package sizes
        common_sizes = {k: v for k, v in package_sizes.items() if v["count"] >= 5}
        for size_key, data in common_sizes.items():
            data["avg_cost"] = data["total_cost"] / data["count"]
            
            # Check if flat rate boxes might be cheaper
            if data["volume"] < 1000 and data["weight"] < 70:  # USPS Medium Flat Rate Box limits
                flat_rate_cost = 15.05  # USPS Medium Flat Rate Box cost
                if data["avg_cost"] > flat_rate_cost * 1.1:
                    opportunities.append({
                        "type": "packaging_optimization",
                        "description": f"Common package size ({data['volume']:.0f} cu in, {data['weight']:.1f} oz) could use flat rate boxes",
                        "potential_savings": (data["avg_cost"] - flat_rate_cost) * data["count"],
                        "recommendation": "Consider using USPS Flat Rate boxes for these items"
                    })
        
        # Check for dimensional weight issues
        dim_weight_issues = [
            h for h in history 
            if h.package.dimensional_weight_lb > h.package.weight_lb * 1.5
        ]
        
        if dim_weight_issues:
            dim_weight_percentage = len(dim_weight_issues) / len(history) * 100
            avg_excess = sum(
                h.package.dimensional_weight_lb / h.package.weight_lb 
                for h in dim_weight_issues
            ) / len(dim_weight_issues)
            
            opportunities.append({
                "type": "dimensional_weight_issues",
                "description": f"{dim_weight_percentage:.1f}% of shipments have dimensional weight {avg_excess:.1f}x actual weight",
                "potential_savings": sum(h.actual_cost * 0.2 for h in dim_weight_issues),
                "recommendation": "Use more compact packaging to reduce dimensional weight charges"
            })
        
        return opportunities