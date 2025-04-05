"""
API module for shipping optimization.

This module provides a REST API for shipping optimization.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from flask import Flask, request, jsonify

from shipping_optimizer.core import ShippingOptimizer
from shipping_optimizer.models import (
    Package, Address, ShippingRate, ShippingPreference
)
from shipping_optimizer.utils import setup_logging


logger = logging.getLogger(__name__)


def create_app(config_path: Optional[str] = None) -> Flask:
    """
    Create Flask application.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Flask application
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        app.config.from_file(config_path, load=json.load)
    
    # Initialize shipping optimizer
    optimizer = ShippingOptimizer(
        config_path=config_path,
        db_connection=app.config.get('DB_CONNECTION', 'sqlite:///shipping.db'),
        cache_enabled=app.config.get('CACHE_ENABLED', True),
        cache_ttl=app.config.get('CACHE_TTL', 3600)
    )
    
    # Set up request throttling
    request_counts = {}
    rate_limits = {
        'get_shipping_rates': 60,  # 60 requests per minute
        'predict_shipping_cost': 120,  # 120 requests per minute
        'select_best_shipping': 120,  # 120 requests per minute
        'adjust_price_for_shipping': 120  # 120 requests per minute
    }
    
    def check_rate_limit(endpoint: str) -> bool:
        """
        Check if request is within rate limit.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            True if within limit, False otherwise
        """
        now = int(time.time())
        minute_ago = now - 60
        
        # Initialize or clean up request counts
        if endpoint not in request_counts:
            request_counts[endpoint] = []
        
        request_counts[endpoint] = [ts for ts in request_counts[endpoint] if ts > minute_ago]
        
        # Check if within limit
        limit = rate_limits.get(endpoint, 60)
        if len(request_counts[endpoint]) >= limit:
            return False
        
        # Add current timestamp
        request_counts[endpoint].append(now)
        return True
    
    @app.route('/api/shipping/rates', methods=['POST'])
    def get_shipping_rates():
        """Get shipping rates."""
        # Check rate limit
        if not check_rate_limit('get_shipping_rates'):
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': 'Too many requests. Please try again later.'
            }), 429
        
        try:
            # Parse request data
            data = request.json
            
            # Validate request data
            if not data or not isinstance(data, dict):
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'Request data must be a JSON object'
                }), 400
            
            # Extract package details
            package_data = data.get('package')
            if not package_data or not isinstance(package_data, dict):
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'Package data is required'
                }), 400
            
            # Create package
            try:
                package = Package(
                    weight_oz=float(package_data.get('weight_oz', 0)),
                    length_in=float(package_data.get('length_in', 0)),
                    width_in=float(package_data.get('width_in', 0)),
                    height_in=float(package_data.get('height_in', 0)),
                    value=float(package_data.get('value', 0)),
                    description=package_data.get('description', ''),
                    is_fragile=bool(package_data.get('is_fragile', False)),
                    requires_signature=bool(package_data.get('requires_signature', False)),
                    is_hazardous=bool(package_data.get('is_hazardous', False))
                )
            except (ValueError, TypeError) as e:
                return jsonify({
                    'error': 'Invalid package data',
                    'message': str(e)
                }), 400
            
            # Extract address details
            origin_data = data.get('origin')
            destination_data = data.get('destination')
            
            if not origin_data or not isinstance(origin_data, dict):
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'Origin address is required'
                }), 400
            
            if not destination_data or not isinstance(destination_data, dict):
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'Destination address is required'
                }), 400
            
            # Create addresses
            try:
                origin = Address(
                    street1=origin_data.get('street1', ''),
                    street2=origin_data.get('street2'),
                    city=origin_data.get('city', ''),
                    state=origin_data.get('state', ''),
                    postal_code=origin_data.get('postal_code', ''),
                    country=origin_data.get('country', 'US'),
                    residential=bool(origin_data.get('residential', True))
                )
                
                destination = Address(
                    street1=destination_data.get('street1', ''),
                    street2=destination_data.get('street2'),
                    city=destination_data.get('city', ''),
                    state=destination_data.get('state', ''),
                    postal_code=destination_data.get('postal_code', ''),
                    country=destination_data.get('country', 'US'),
                    residential=bool(destination_data.get('residential', True))
                )
            except (ValueError, TypeError) as e:
                return jsonify({
                    'error': 'Invalid address data',
                    'message': str(e)
                }), 400
            
            # Extract optional parameters
            carriers = data.get('carriers')
            services = data.get('services')
            use_cache = bool(data.get('use_cache', True))
            
            # Get shipping rates
            rates = optimizer.get_shipping_rates(
                package=package,
                origin=origin,
                destination=destination,
                carriers=carriers,
                services=services,
                use_cache=use_cache
            )
            
            # Convert rates to dictionaries
            rates_dict = [rate.to_dict() for rate in rates]
            
            # Return response
            return jsonify({
                'success': True,
                'rates': rates_dict,
                'count': len(rates_dict)
            })
        
        except Exception as e:
            logger.error(f"Error in get_shipping_rates: {e}")
            return jsonify({
                'error': 'Server error',
                'message': str(e)
            }), 500
    
    @app.route('/api/shipping/predict', methods=['POST'])
    def predict_shipping_cost():
        """Predict shipping cost."""
        # Check rate limit
        if not check_rate_limit('predict_shipping_cost'):
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': 'Too many requests. Please try again later.'
            }), 429
        
        try:
            # Parse request data
            data = request.json
            
            # Validate request data
            if not data or not isinstance(data, dict):
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'Request data must be a JSON object'
                }), 400
            
            # Extract package details
            package_data = data.get('package')
            if not package_data or not isinstance(package_data, dict):
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'Package data is required'
                }), 400
            
            # Create package
            try:
                package = Package(
                    weight_oz=float(package_data.get('weight_oz', 0)),
                    length_in=float(package_data.get('length_in', 0)),
                    width_in=float(package_data.get('width_in', 0)),
                    height_in=float(package_data.get('height_in', 0)),
                    value=float(package_data.get('value', 0)),
                    description=package_data.get('description', ''),
                    is_fragile=bool(package_data.get('is_fragile', False)),
                    requires_signature=bool(package_data.get('requires_signature', False)),
                    is_hazardous=bool(package_data.get('is_hazardous', False))
                )
            except (ValueError, TypeError) as e:
                return jsonify({
                    'error': 'Invalid package data',
                    'message': str(e)
                }), 400
            
            # Extract address details
            origin_data = data.get('origin')
            destination_data = data.get('destination')
            
            if not origin_data or not isinstance(origin_data, dict):
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'Origin address is required'
                }), 400
            
            if not destination_data or not isinstance(destination_data, dict):
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'Destination address is required'
                }), 400
            
            # Create addresses
            try:
                origin = Address(
                    street1=origin_data.get('street1', ''),
                    street2=origin_data.get('street2'),
                    city=origin_data.get('city', ''),
                    state=origin_data.get('state', ''),
                    postal_code=origin_data.get('postal_code', ''),
                    country=origin_data.get('country', 'US'),
                    residential=bool(origin_data.get('residential', True))
                )
                
                destination = Address(
                    street1=destination_data.get('street1', ''),
                    street2=destination_data.get('street2'),
                    city=destination_data.get('city', ''),
                    state=destination_data.get('state', ''),
                    postal_code=destination_data.get('postal_code', ''),
                    country=destination_data.get('country', 'US'),
                    residential=bool(destination_data.get('residential', True))
                )
            except (ValueError, TypeError) as e:
                return jsonify({
                    'error': 'Invalid address data',
                    'message': str(e)
                }), 400
            
            # Extract carrier and service
            carrier = data.get('carrier')
            service = data.get('service')
            
            if not carrier or not isinstance(carrier, str):
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'Carrier is required'
                }), 400
            
            if not service or not isinstance(service, str):
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'Service is required'
                }), 400
            
            # Calculate distance and zone
            from shipping_optimizer.utils import calculate_distance, calculate_zone
            
            distance = calculate_distance(origin, destination)
            zone = calculate_zone(origin.postal_code, destination.postal_code)
            
            # Predict shipping cost
            predicted_cost = optimizer.predict_shipping_cost(
                package=package,
                distance=distance,
                zone=zone,
                carrier=carrier,
                service=service
            )
            
            # Return response
            return jsonify({
                'success': True,
                'carrier': carrier,
                'service': service,
                'predicted_cost': predicted_cost,
                'distance_miles': distance,
                'zone': zone
            })
        
        except Exception as e:
            logger.error(f"Error in predict_shipping_cost: {e}")
            return jsonify({
                'error': 'Server error',
                'message': str(e)
            }), 500
    
    @app.route('/api/shipping/select', methods=['POST'])
    def select_best_shipping():
        """Select best shipping option."""
        # Check rate limit
        if not check_rate_limit('select_best_shipping'):
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': 'Too many requests. Please try again later.'
            }), 429
        
        try:
            # Parse request data
            data = request.json
            
            # Validate request data
            if not data or not isinstance(data, dict):
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'Request data must be a JSON object'
                }), 400
            
            # Extract rates
            rates_data = data.get('rates')
            if not rates_data or not isinstance(rates_data, list):
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'Rates data is required'
                }), 400
            
            # Create shipping rates
            rates = []
            for rate_data in rates_data:
                try:
                    rate = ShippingRate(
                        carrier=rate_data.get('carrier', ''),
                        service=rate_data.get('service', ''),
                        rate=float(rate_data.get('rate', 0)),
                        delivery_days=rate_data.get('delivery_days'),
                        guaranteed=bool(rate_data.get('guaranteed', False)),
                        tracking_included=bool(rate_data.get('tracking_included', True)),
                        insurance_included=bool(rate_data.get('insurance_included', False)),
                        insurance_cost=float(rate_data.get('insurance_cost', 0)),
                        signature_cost=float(rate_data.get('signature_cost', 0)),
                        fuel_surcharge=float(rate_data.get('fuel_surcharge', 0)),
                        other_surcharges=rate_data.get('other_surcharges', {})
                    )
                    rates.append(rate)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid rate data: {e}")
                    continue
            
            if not rates:
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'No valid rates provided'
                }), 400
            
            # Extract preferences
            preferences_data = data.get('preferences')
            preferences = None
            
            if preferences_data and isinstance(preferences_data, dict):
                try:
                    preferences = ShippingPreference(
                        user_id=preferences_data.get('user_id', 'default'),
                        preferred_carriers=preferences_data.get('preferred_carriers', []),
                        excluded_carriers=preferences_data.get('excluded_carriers', []),
                        cost_importance=float(preferences_data.get('cost_importance', 0.5)),
                        speed_importance=float(preferences_data.get('speed_importance', 0.3)),
                        reliability_importance=float(preferences_data.get('reliability_importance', 0.2)),
                        default_package_type=preferences_data.get('default_package_type'),
                        auto_insurance_threshold=float(preferences_data.get('auto_insurance_threshold', 100.0)),
                        require_signature_threshold=float(preferences_data.get('require_signature_threshold', 200.0))
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid preferences data: {e}")
            
            # Extract package
            package_data = data.get('package')
            package = None
            
            if package_data and isinstance(package_data, dict):
                try:
                    package = Package(
                        weight_oz=float(package_data.get('weight_oz', 0)),
                        length_in=float(package_data.get('length_in', 0)),
                        width_in=float(package_data.get('width_in', 0)),
                        height_in=float(package_data.get('height_in', 0)),
                        value=float(package_data.get('value', 0)),
                        description=package_data.get('description', ''),
                        is_fragile=bool(package_data.get('is_fragile', False)),
                        requires_signature=bool(package_data.get('requires_signature', False)),
                        is_hazardous=bool(package_data.get('is_hazardous', False))
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid package data: {e}")
            
            # Select best shipping option
            options = optimizer.select_best_shipping(
                rates=rates,
                preferences=preferences,
                package=package
            )
            
            # Convert options to dictionaries
            options_dict = [option.to_dict() for option in options]
            
            # Return response
            return jsonify({
                'success': True,
                'options': options_dict,
                'count': len(options_dict),
                'recommended': next((o for o in options_dict if o.get('is_recommended')), None)
            })
        
        except Exception as e:
            logger.error(f"Error in select_best_shipping: {e}")
            return jsonify({
                'error': 'Server error',
                'message': str(e)
            }), 500
    
    @app.route('/api/shipping/adjust-price', methods=['POST'])
    def adjust_price_for_shipping():
        """Adjust price for shipping."""
        # Check rate limit
        if not check_rate_limit('adjust_price_for_shipping'):
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': 'Too many requests. Please try again later.'
            }), 429
        
        try:
            # Parse request data
            data = request.json
            
            # Validate request data
            if not data or not isinstance(data, dict):
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'Request data must be a JSON object'
                }), 400
            
            # Extract parameters
            try:
                item_price = float(data.get('item_price', 0))
                shipping_cost = float(data.get('shipping_cost', 0))
                handling_cost = float(data.get('handling_cost', 0))
                packaging_cost = data.get('packaging_cost')
                if packaging_cost is not None:
                    packaging_cost = float(packaging_cost)
                return_rate = float(data.get('return_rate', 0.03))
                target_margin = float(data.get('target_margin', 0.2))
            except (ValueError, TypeError) as e:
                return jsonify({
                    'error': 'Invalid parameter',
                    'message': str(e)
                }), 400
            
            # Adjust price
            adjustment = optimizer.adjust_price_for_shipping(
                item_price=item_price,
                shipping_cost=shipping_cost,
                handling_cost=handling_cost,
                packaging_cost=packaging_cost,
                return_rate=return_rate,
                target_margin=target_margin
            )
            
            # Return response
            return jsonify({
                'success': True,
                'adjustment': adjustment.to_dict()
            })
        
        except Exception as e:
            logger.error(f"Error in adjust_price_for_shipping: {e}")
            return jsonify({
                'error': 'Server error',
                'message': str(e)
            }), 500
    
    @app.route('/api/shipping/analyze-history', methods=['POST'])
    def analyze_shipping_history():
        """Analyze shipping history."""
        try:
            # Parse request data
            data = request.json
            
            # Validate request data
            if not data or not isinstance(data, dict):
                return jsonify({
                    'error': 'Invalid request',
                    'message': 'Request data must be a JSON object'
                }), 400
            
            # Extract parameters
            user_id = data.get('user_id')
            
            start_date_str = data.get('start_date')
            start_date = None
            if start_date_str:
                try:
                    start_date = datetime.fromisoformat(start_date_str)
                except ValueError:
                    return jsonify({
                        'error': 'Invalid start_date',
                        'message': 'start_date must be in ISO format (YYYY-MM-DDTHH:MM:SS)'
                    }), 400
            
            end_date_str = data.get('end_date')
            end_date = None
            if end_date_str:
                try:
                    end_date = datetime.fromisoformat(end_date_str)
                except ValueError:
                    return jsonify({
                        'error': 'Invalid end_date',
                        'message': 'end_date must be in ISO format (YYYY-MM-DDTHH:MM:SS)'
                    }), 400
            
            limit = int(data.get('limit', 100))
            
            # Analyze shipping history
            analysis = optimizer.analyze_shipping_history(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            
            # Return response
            return jsonify({
                'success': True,
                'analysis': analysis
            })
        
        except Exception as e:
            logger.error(f"Error in analyze_shipping_history: {e}")
            return jsonify({
                'error': 'Server error',
                'message': str(e)
            }), 500
    
    return app