"""
API module for the Listing Generator.

This module provides a REST API for the Listing Generator.
"""

import os
import logging
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS

from listing_generator.core import ListingGenerator
from listing_generator.models import (
    Product, Listing, ListingPerformance, Marketplace, ListingStatus
)
from listing_generator.utils import setup_logging


logger = logging.getLogger(__name__)


def create_app(config_path: Optional[str] = None) -> Flask:
    """
    Create a Flask application for the Listing Generator API.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Flask application
    """
    # Create Flask app
    app = Flask(__name__)
    CORS(app)
    
    # Initialize ListingGenerator
    generator = ListingGenerator(config_path)
    
    # Create API blueprint
    api = Blueprint('api', __name__, url_prefix='/api')
    
    @api.route('/generate_title', methods=['POST'])
    async def generate_title():
        """Generate an optimized title for a product."""
        try:
            # Get request data
            data = request.json
            
            # Validate request
            if not data or 'product' not in data or 'marketplace' not in data:
                return jsonify({
                    'error': 'Invalid request. Missing product or marketplace.'
                }), 400
            
            # Create product
            product_data = data['product']
            product = Product(
                id=product_data.get('id', str(uuid.uuid4())),
                name=product_data.get('name', ''),
                brand=product_data.get('brand'),
                model=product_data.get('model'),
                category=product_data.get('category'),
                subcategory=product_data.get('subcategory'),
                features=product_data.get('features', []),
                specifications=product_data.get('specifications', {}),
                description=product_data.get('description'),
                condition=product_data.get('condition', 'New'),
                image_urls=product_data.get('image_urls', [])
            )
            
            # Get marketplace
            marketplace = Marketplace(data['marketplace'])
            
            # Generate title
            title = await generator.title_generator.generate_title(product, marketplace)
            
            # Generate variations if requested
            variations = []
            if data.get('generate_variations', False):
                num_variations = data.get('num_variations', 3)
                variations = await generator.title_generator.generate_variations(
                    product, 
                    marketplace, 
                    num_variations
                )
            
            # Return response
            return jsonify({
                'title': title,
                'variations': variations
            })
        
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return jsonify({
                'error': f"Error generating title: {str(e)}"
            }), 500
    
    @api.route('/generate_description', methods=['POST'])
    async def generate_description():
        """Generate an optimized description for a product."""
        try:
            # Get request data
            data = request.json
            
            # Validate request
            if not data or 'product' not in data or 'marketplace' not in data:
                return jsonify({
                    'error': 'Invalid request. Missing product or marketplace.'
                }), 400
            
            # Create product
            product_data = data['product']
            product = Product(
                id=product_data.get('id', str(uuid.uuid4())),
                name=product_data.get('name', ''),
                brand=product_data.get('brand'),
                model=product_data.get('model'),
                category=product_data.get('category'),
                subcategory=product_data.get('subcategory'),
                features=product_data.get('features', []),
                specifications=product_data.get('specifications', {}),
                description=product_data.get('description'),
                condition=product_data.get('condition', 'New'),
                image_urls=product_data.get('image_urls', [])
            )
            
            # Get marketplace
            marketplace = Marketplace(data['marketplace'])
            
            # Generate description
            description = await generator.description_generator.generate_description(product, marketplace)
            
            # Return response
            return jsonify({
                'description': description
            })
        
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return jsonify({
                'error': f"Error generating description: {str(e)}"
            }), 500
    
    @api.route('/get_pricing_recommendation', methods=['POST'])
    async def get_pricing_recommendation():
        """Get a pricing recommendation for a product."""
        try:
            # Get request data
            data = request.json
            
            # Validate request
            if not data or 'product' not in data or 'marketplace' not in data:
                return jsonify({
                    'error': 'Invalid request. Missing product or marketplace.'
                }), 400
            
            # Create product
            product_data = data['product']
            product = Product(
                id=product_data.get('id', str(uuid.uuid4())),
                name=product_data.get('name', ''),
                brand=product_data.get('brand'),
                model=product_data.get('model'),
                category=product_data.get('category'),
                subcategory=product_data.get('subcategory'),
                features=product_data.get('features', []),
                specifications=product_data.get('specifications', {}),
                description=product_data.get('description'),
                condition=product_data.get('condition', 'New'),
                image_urls=product_data.get('image_urls', []),
                msrp=product_data.get('msrp'),
                cost=product_data.get('cost')
            )
            
            # Get marketplace
            marketplace = Marketplace(data['marketplace'])
            
            # Get pricing recommendation
            pricing = await generator.pricing_engine.get_pricing_recommendation(product, marketplace)
            
            # Return response
            return jsonify(pricing.to_dict())
        
        except Exception as e:
            logger.error(f"Error getting pricing recommendation: {e}")
            return jsonify({
                'error': f"Error getting pricing recommendation: {str(e)}"
            }), 500
    
    @api.route('/generate_listing', methods=['POST'])
    async def generate_listing():
        """Generate a complete listing for a product."""
        try:
            # Get request data
            data = request.json
            
            # Validate request
            if not data or 'product' not in data or 'marketplace' not in data:
                return jsonify({
                    'error': 'Invalid request. Missing product or marketplace.'
                }), 400
            
            # Create product
            product_data = data['product']
            product = Product(
                id=product_data.get('id', str(uuid.uuid4())),
                name=product_data.get('name', ''),
                brand=product_data.get('brand'),
                model=product_data.get('model'),
                category=product_data.get('category'),
                subcategory=product_data.get('subcategory'),
                features=product_data.get('features', []),
                specifications=product_data.get('specifications', {}),
                description=product_data.get('description'),
                condition=product_data.get('condition', 'New'),
                image_urls=product_data.get('image_urls', []),
                msrp=product_data.get('msrp'),
                cost=product_data.get('cost')
            )
            
            # Save product to database
            await generator.db.save_product(product)
            
            # Get marketplace
            marketplace = Marketplace(data['marketplace'])
            
            # Generate listing
            generate_variations = data.get('generate_variations', True)
            num_variations = data.get('num_variations', 3)
            
            listing = await generator.generate_listing(
                product,
                marketplace,
                generate_variations,
                num_variations
            )
            
            # Return response
            return jsonify(listing.to_dict())
        
        except Exception as e:
            logger.error(f"Error generating listing: {e}")
            return jsonify({
                'error': f"Error generating listing: {str(e)}"
            }), 500
    
    @api.route('/track_listing_performance', methods=['POST'])
    async def track_listing_performance():
        """Track performance metrics for a listing."""
        try:
            # Get request data
            data = request.json
            
            # Validate request
            if not data or 'listing_id' not in data or 'metrics' not in data:
                return jsonify({
                    'error': 'Invalid request. Missing listing_id or metrics.'
                }), 400
            
            # Get listing ID and metrics
            listing_id = data['listing_id']
            metrics = data['metrics']
            
            # Track performance
            performance = await generator.track_performance(listing_id, metrics)
            
            if not performance:
                return jsonify({
                    'error': f"Listing {listing_id} not found."
                }), 404
            
            # Return response
            return jsonify(performance.to_dict())
        
        except Exception as e:
            logger.error(f"Error tracking listing performance: {e}")
            return jsonify({
                'error': f"Error tracking listing performance: {str(e)}"
            }), 500
    
    @api.route('/optimize_listing/<listing_id>', methods=['POST'])
    async def optimize_listing(listing_id):
        """Optimize an existing listing."""
        try:
            # Get request data
            data = request.json or {}
            
            # Get optimization options
            optimize_title = data.get('optimize_title', True)
            optimize_description = data.get('optimize_description', True)
            update_pricing = data.get('update_pricing', True)
            process_images = data.get('process_images', True)
            
            # Get listing
            listing = await generator.get_listing(listing_id)
            
            if not listing:
                return jsonify({
                    'error': f"Listing {listing_id} not found."
                }), 404
            
            # Optimize listing
            if optimize_title:
                await generator.optimize_title(listing_id, use_ab_testing=data.get('use_ab_testing', True))
            
            if optimize_description:
                await generator.optimize_description(listing_id)
            
            if update_pricing:
                await generator.update_pricing(listing_id, dynamic_adjustment=data.get('dynamic_adjustment', False))
            
            if process_images:
                await generator.process_images(listing_id)
            
            # Get updated listing
            updated_listing = await generator.get_listing(listing_id)
            
            # Return response
            return jsonify(updated_listing.to_dict())
        
        except Exception as e:
            logger.error(f"Error optimizing listing: {e}")
            return jsonify({
                'error': f"Error optimizing listing: {str(e)}"
            }), 500
    
    @api.route('/listings', methods=['GET'])
    async def get_listings():
        """Get listings with optional filtering."""
        try:
            # Get query parameters
            product_id = request.args.get('product_id')
            marketplace = request.args.get('marketplace')
            status = request.args.get('status')
            limit = int(request.args.get('limit', 100))
            offset = int(request.args.get('offset', 0))
            
            # Convert marketplace and status if provided
            marketplace_enum = Marketplace(marketplace) if marketplace else None
            status_enum = ListingStatus(status) if status else None
            
            # Get listings
            listings = await generator.get_listings(
                product_id=product_id,
                marketplace=marketplace_enum,
                status=status_enum,
                limit=limit,
                offset=offset
            )
            
            # Return response
            return jsonify({
                'listings': [listing.to_dict() for listing in listings],
                'count': len(listings),
                'limit': limit,
                'offset': offset
            })
        
        except Exception as e:
            logger.error(f"Error getting listings: {e}")
            return jsonify({
                'error': f"Error getting listings: {str(e)}"
            }), 500
    
    @api.route('/listings/<listing_id>', methods=['GET'])
    async def get_listing(listing_id):
        """Get a listing by ID."""
        try:
            # Get listing
            listing = await generator.get_listing(listing_id)
            
            if not listing:
                return jsonify({
                    'error': f"Listing {listing_id} not found."
                }), 404
            
            # Return response
            return jsonify(listing.to_dict())
        
        except Exception as e:
            logger.error(f"Error getting listing: {e}")
            return jsonify({
                'error': f"Error getting listing: {str(e)}"
            }), 500
    
    @api.route('/listings/<listing_id>/performance', methods=['GET'])
    async def get_listing_performance(listing_id):
        """Get performance data for a listing."""
        try:
            # Get performance data
            performance = await generator.get_performance(listing_id)
            
            if not performance:
                return jsonify({
                    'error': f"Performance data for listing {listing_id} not found."
                }), 404
            
            # Return response
            return jsonify(performance.to_dict())
        
        except Exception as e:
            logger.error(f"Error getting listing performance: {e}")
            return jsonify({
                'error': f"Error getting listing performance: {str(e)}"
            }), 500
    
    # Register blueprint
    app.register_blueprint(api)
    
    return app