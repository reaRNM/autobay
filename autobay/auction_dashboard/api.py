"""
RESTful API for the Dashboard & Mobile Alerts module.

This module provides Flask routes for accessing dashboard data,
managing alert configurations, and other dashboard operations.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from functools import wraps
from flask import Flask, request, jsonify, Blueprint, current_app
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from .models import db, User, AuctionItem, AuctionHistory, AlertConfig
from .services import DashboardService, MetricsService
from .notifications import NotificationService

logger = logging.getLogger(__name__)

# Create blueprints
api_bp = Blueprint('api', __name__, url_prefix='/api')
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            # In a real implementation, this would validate the token
            # and get the user ID from it
            # For this example, we'll use a simple mapping
            token_mapping = current_app.config.get('TOKEN_MAPPING', {})
            user_id = token_mapping.get(token)
            
            if not user_id:
                return jsonify({'message': 'Invalid token'}), 401
            
            # Get the user
            user = User.query.get(user_id)
            if not user:
                return jsonify({'message': 'User not found'}), 401
            
            # Add user to kwargs
            kwargs['user'] = user
            
            return f(*args, **kwargs)
        
        except Exception as e:
            logger.exception(f"Error validating token: {e}")
            return jsonify({'message': 'Token validation failed'}), 401
    
    return decorated


# Error handler
@api_bp.errorhandler(Exception)
def handle_error(e):
    """Handle API errors."""
    code = 500
    message = str(e)
    
    if isinstance(e, HTTPException):
        code = e.code
    
    logger.exception(f"API error: {e}")
    
    return jsonify({
        'error': message
    }), code


# Authentication routes
@auth_bp.route('/login', methods=['POST'])
def login():
    """Login and get an authentication token."""
    data = request.json
    
    if not data:
        return jsonify({'message': 'No input data provided'}), 400
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'message': 'Username and password are required'}), 400
    
    # Get the user
    user = User.query.filter_by(username=username).first()
    
    if not user or not user.check_password(password):
        return jsonify({'message': 'Invalid username or password'}), 401
    
    # In a real implementation, this would generate a JWT token
    # For this example, we'll use a simple token
    token = f"token_{user.id}"
    
    # Update last login
    user.last_login = db.func.now()
    db.session.commit()
    
    return jsonify({
        'token': token,
        'user': user.to_dict()
    })


@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user."""
    data = request.json
    
    if not data:
        return jsonify({'message': 'No input data provided'}), 400
    
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not username or not email or not password:
        return jsonify({'message': 'Username, email, and password are required'}), 400
    
    # Check if username or email already exists
    if User.query.filter_by(username=username).first():
        return jsonify({'message': 'Username already exists'}), 400
    
    if User.query.filter_by(email=email).first():
        return jsonify({'message': 'Email already exists'}), 400
    
    # Create the user
    user = User(
        username=username,
        email=email
    )
    user.set_password(password)
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({
        'message': 'User registered successfully',
        'user': user.to_dict()
    })


# Dashboard routes
@api_bp.route('/dashboard', methods=['GET'])
@token_required
def get_dashboard(user):
    """Get dashboard data."""
    time_range = request.args.get('time_range', 'all')
    
    # Get dashboard data
    dashboard_service = DashboardService()
    data = dashboard_service.get_dashboard_data(user.id, time_range)
    
    return jsonify(data)


@api_bp.route('/metrics/profit', methods=['GET'])
@token_required
def get_profit_metrics(user):
    """Get profit metrics."""
    time_range = request.args.get('time_range', 'all')
    
    # Get profit metrics
    metrics_service = MetricsService()
    data = metrics_service.get_profit_metrics(user.id, time_range)
    
    return jsonify(data)


@api_bp.route('/metrics/auction', methods=['GET'])
@token_required
def get_auction_metrics(user):
    """Get auction metrics."""
    time_range = request.args.get('time_range', 'all')
    
    # Get auction metrics
    metrics_service = MetricsService()
    data = metrics_service.get_auction_metrics(user.id, time_range)
    
    return jsonify(data)


@api_bp.route('/metrics/risk', methods=['GET'])
@token_required
def get_risk_metrics(user):
    """Get risk metrics."""
    # Get risk metrics
    metrics_service = MetricsService()
    data = metrics_service.get_risk_metrics(user.id)
    
    return jsonify(data)


@api_bp.route('/ranking', methods=['GET'])
@token_required
def get_ranking(user):
    """Get grand ranking of auction items."""
    # Parse filters from query parameters
    filters = {}
    
    if 'category' in request.args:
        filters['category'] = request.args.get('category')
    
    if 'min_profit' in request.args:
        try  = request.args.get('category')
    
    if 'min_profit' in request.args:
        try:
            filters['min_profit'] = float(request.args.get('min_profit'))
        except ValueError:
            pass
    
    if 'max_risk' in request.args:
        try:
            filters['max_risk'] = float(request.args.get('max_risk'))
        except ValueError:
            pass
    
    if 'status' in request.args:
        filters['status'] = request.args.get('status')
    
    # Get limit parameter
    try:
        limit = int(request.args.get('limit', 100))
    except ValueError:
        limit = 100
    
    # Get grand ranking
    metrics_service = MetricsService()
    data = metrics_service.get_grand_ranking(filters, limit)
    
    return jsonify(data)


@api_bp.route('/fees/calculate', methods=['POST'])
@token_required
def calculate_fees(user):
    """Calculate fees for an auction item."""
    data = request.json
    
    if not data:
        return jsonify({'message': 'No input data provided'}), 400
    
    # Get parameters
    bid_amount = data.get('bid_amount')
    buyer_premium_rate = data.get('buyer_premium_rate', 0.15)
    sales_tax_rate = data.get('sales_tax_rate', 0.07)
    shipping_cost = data.get('shipping_cost', 0.0)
    additional_fees = data.get('additional_fees', {})
    
    if bid_amount is None:
        return jsonify({'message': 'Bid amount is required'}), 400
    
    try:
        bid_amount = float(bid_amount)
        buyer_premium_rate = float(buyer_premium_rate)
        sales_tax_rate = float(sales_tax_rate)
        shipping_cost = float(shipping_cost)
    except ValueError:
        return jsonify({'message': 'Invalid numeric value'}), 400
    
    # Calculate fees
    dashboard_service = DashboardService()
    result = dashboard_service.calculate_fees(
        bid_amount=bid_amount,
        buyer_premium_rate=buyer_premium_rate,
        sales_tax_rate=sales_tax_rate,
        shipping_cost=shipping_cost,
        additional_fees=additional_fees
    )
    
    return jsonify(result)


@api_bp.route('/profit/calculate', methods=['POST'])
@token_required
def calculate_profit(user):
    """Calculate profit for an auction item."""
    data = request.json
    
    if not data:
        return jsonify({'message': 'No input data provided'}), 400
    
    # Get parameters
    purchase_price = data.get('purchase_price')
    sale_price = data.get('sale_price')
    platform_fee_rate = data.get('platform_fee_rate', 0.1275)
    shipping_cost = data.get('shipping_cost', 0.0)
    additional_costs = data.get('additional_costs', {})
    
    if purchase_price is None or sale_price is None:
        return jsonify({'message': 'Purchase price and sale price are required'}), 400
    
    try:
        purchase_price = float(purchase_price)
        sale_price = float(sale_price)
        platform_fee_rate = float(platform_fee_rate)
        shipping_cost = float(shipping_cost)
    except ValueError:
        return jsonify({'message': 'Invalid numeric value'}), 400
    
    # Calculate profit
    dashboard_service = DashboardService()
    result = dashboard_service.calculate_profit(
        purchase_price=purchase_price,
        sale_price=sale_price,
        platform_fee_rate=platform_fee_rate,
        shipping_cost=shipping_cost,
        additional_costs=additional_costs
    )
    
    return jsonify(result)


# Auction history routes
@api_bp.route('/history', methods=['GET'])
@token_required
def get_auction_history(user):
    """Get auction history."""
    # Parse query parameters
    status = request.args.get('status')
    limit = request.args.get('limit', 100)
    offset = request.args.get('offset', 0)
    
    try:
        limit = int(limit)
        offset = int(offset)
    except ValueError:
        return jsonify({'message': 'Invalid limit or offset'}), 400
    
    # Build the query
    query = AuctionHistory.query.filter_by(user_id=user.id)
    
    if status:
        query = query.filter_by(status=status)
    
    # Get total count
    total_count = query.count()
    
    # Apply pagination
    query = query.order_by(AuctionHistory.created_at.desc())
    query = query.limit(limit).offset(offset)
    
    # Get the items
    items = query.all()
    
    return jsonify({
        'items': [item.to_dict() for item in items],
        'total_count': total_count,
        'limit': limit,
        'offset': offset
    })


@api_bp.route('/history/<int:history_id>', methods=['GET'])
@token_required
def get_auction_history_item(user, history_id):
    """Get a specific auction history item."""
    # Get the item
    item = AuctionHistory.query.filter_by(id=history_id, user_id=user.id).first()
    
    if not item:
        return jsonify({'message': 'Item not found'}), 404
    
    return jsonify(item.to_dict())


# Alert configuration routes
@api_bp.route('/alerts/config', methods=['GET'])
@token_required
def get_alert_configs(user):
    """Get alert configurations."""
    # Get alert configs
    configs = AlertConfig.query.filter_by(user_id=user.id).all()
    
    return jsonify([config.to_dict() for config in configs])


@api_bp.route('/alerts/config/<int:config_id>', methods=['GET'])
@token_required
def get_alert_config(user, config_id):
    """Get a specific alert configuration."""
    # Get the config
    config = AlertConfig.query.filter_by(id=config_id, user_id=user.id).first()
    
    if not config:
        return jsonify({'message': 'Alert configuration not found'}), 404
    
    return jsonify(config.to_dict())


@api_bp.route('/alerts/config', methods=['POST'])
@token_required
def create_alert_config(user):
    """Create a new alert configuration."""
    data = request.json
    
    if not data:
        return jsonify({'message': 'No input data provided'}), 400
    
    # Validate required fields
    required_fields = ['name', 'alert_type', 'conditions', 'notification_channels']
    for field in required_fields:
        if field not in data:
            return jsonify({'message': f'Field {field} is required'}), 400
    
    # Create the config
    config = AlertConfig(
        user_id=user.id,
        name=data['name'],
        description=data.get('description'),
        alert_type=data['alert_type'],
        conditions=data['conditions'],
        notification_channels=data['notification_channels'],
        is_active=data.get('is_active', True)
    )
    
    db.session.add(config)
    db.session.commit()
    
    return jsonify(config.to_dict())


@api_bp.route('/alerts/config/<int:config_id>', methods=['PUT'])
@token_required
def update_alert_config(user, config_id):
    """Update an alert configuration."""
    data = request.json
    
    if not data:
        return jsonify({'message': 'No input data provided'}), 400
    
    # Get the config
    config = AlertConfig.query.filter_by(id=config_id, user_id=user.id).first()
    
    if not config:
        return jsonify({'message': 'Alert configuration not found'}), 404
    
    # Update fields
    if 'name' in data:
        config.name = data['name']
    
    if 'description' in data:
        config.description = data['description']
    
    if 'alert_type' in data:
        config.alert_type = data['alert_type']
    
    if 'conditions' in data:
        config.conditions = data['conditions']
    
    if 'notification_channels' in data:
        config.notification_channels = data['notification_channels']
    
    if 'is_active' in data:
        config.is_active = data['is_active']
    
    db.session.commit()
    
    return jsonify(config.to_dict())


@api_bp.route('/alerts/config/<int:config_id>', methods=['DELETE'])
@token_required
def delete_alert_config(user, config_id):
    """Delete an alert configuration."""
    # Get the config
    config = AlertConfig.query.filter_by(id=config_id, user_id=user.id).first()
    
    if not config:
        return jsonify({'message': 'Alert configuration not found'}), 404
    
    db.session.delete(config)
    db.session.commit()
    
    return jsonify({'message': 'Alert configuration deleted'})


@api_bp.route('/alerts/test', methods=['POST'])
@token_required
def test_alert(user):
    """Test an alert notification."""
    data = request.json
    
    if not data:
        return jsonify({'message': 'No input data provided'}), 400
    
    # Get parameters
    alert_type = data.get('alert_type', 'test_alert')
    message = data.get('message', 'This is a test alert')
    channels = data.get('channels', ['telegram'])
    test_data = data.get('data', {})
    
    # Initialize notification service
    notification_service = NotificationService(current_app.config.get('NOTIFICATION_CONFIG', {}))
    
    # Send test notification
    result = notification_service.send_notification(
        user_id=user.id,
        alert_type=alert_type,
        message=message,
        data=test_data,
        channels=channels
    )
    
    return jsonify({
        'message': 'Test alert sent',
        'result': result
    })


def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Load default configuration
    app.config.from_mapping(
        SECRET_KEY='dev',
        SQLALCHEMY_DATABASE_URI='sqlite:///auction_dashboard.db',
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        TOKEN_MAPPING={},  # For demo purposes
        NOTIFICATION_CONFIG={}
    )
    
    # Load configuration from parameter
    if config:
        app.config.update(config)
    
    # Initialize extensions
    db.init_app(app)
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    app.register_blueprint(auth_bp)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app