"""
Database models for the Dashboard & Mobile Alerts module.

This module defines the SQLAlchemy models for storing auction data,
user preferences, and alert configurations.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSONB
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
    """User model for authentication and preferences."""
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    
    # User preferences
    preferences = db.Column(JSONB, default={})
    
    # Relationships
    alert_configs = db.relationship('AlertConfig', backref='user', lazy=True)
    
    def set_password(self, password):
        """Set the user's password hash."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if the provided password matches the hash."""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert user to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'preferences': self.preferences
        }


class AuctionItem(db.Model):
    """Model for auction items."""
    
    __tablename__ = 'auction_items'
    
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.String(100), unique=True, nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    category = db.Column(db.String(100), nullable=True)
    condition = db.Column(db.String(50), nullable=True)
    
    # Auction details
    auction_id = db.Column(db.String(100), nullable=True)
    auction_end_time = db.Column(db.DateTime, nullable=True)
    starting_bid = db.Column(db.Float, nullable=True)
    current_bid = db.Column(db.Float, nullable=True)
    estimated_value = db.Column(db.Float, nullable=True)
    
    # Profit calculation
    estimated_profit = db.Column(db.Float, nullable=True)
    profit_margin = db.Column(db.Float, nullable=True)
    
    # Risk assessment
    risk_score = db.Column(db.Float, nullable=True)
    risk_factors = db.Column(JSONB, default={})
    
    # Shipping and logistics
    shipping_ease_score = db.Column(db.Float, nullable=True)
    estimated_shipping_cost = db.Column(db.Float, nullable=True)
    
    # Grand ranking
    grand_ranking_score = db.Column(db.Float, nullable=True)
    
    # Status and timestamps
    status = db.Column(db.String(50), default='active')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional data
    metadata = db.Column(JSONB, default={})
    
    def to_dict(self):
        """Convert auction item to dictionary."""
        return {
            'id': self.id,
            'item_id': self.item_id,
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'condition': self.condition,
            'auction_id': self.auction_id,
            'auction_end_time': self.auction_end_time.isoformat() if self.auction_end_time else None,
            'starting_bid': self.starting_bid,
            'current_bid': self.current_bid,
            'estimated_value': self.estimated_value,
            'estimated_profit': self.estimated_profit,
            'profit_margin': self.profit_margin,
            'risk_score': self.risk_score,
            'risk_factors': self.risk_factors,
            'shipping_ease_score': self.shipping_ease_score,
            'estimated_shipping_cost': self.estimated_shipping_cost,
            'grand_ranking_score': self.grand_ranking_score,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.metadata
        }


class AuctionHistory(db.Model):
    """Model for auction history (won auctions and sales)."""
    
    __tablename__ = 'auction_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    item_id = db.Column(db.String(100), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    
    # Auction details
    auction_id = db.Column(db.String(100), nullable=True)
    auction_end_time = db.Column(db.DateTime, nullable=True)
    
    # Purchase details
    purchase_price = db.Column(db.Float, nullable=True)
    buyer_premium = db.Column(db.Float, nullable=True)
    sales_tax = db.Column(db.Float, nullable=True)
    shipping_cost = db.Column(db.Float, nullable=True)
    additional_fees = db.Column(JSONB, default={})
    total_cost = db.Column(db.Float, nullable=True)
    
    # Sale details
    sale_price = db.Column(db.Float, nullable=True)
    sale_date = db.Column(db.DateTime, nullable=True)
    platform_fees = db.Column(db.Float, nullable=True)
    
    # Profit calculation
    profit = db.Column(db.Float, nullable=True)
    roi = db.Column(db.Float, nullable=True)
    
    # Status and timestamps
    status = db.Column(db.String(50), default='purchased')  # purchased, listed, sold
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional data
    metadata = db.Column(JSONB, default={})
    
    def to_dict(self):
        """Convert auction history to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'item_id': self.item_id,
            'title': self.title,
            'auction_id': self.auction_id,
            'auction_end_time': self.auction_end_time.isoformat() if self.auction_end_time else None,
            'purchase_price': self.purchase_price,
            'buyer_premium': self.buyer_premium,
            'sales_tax': self.sales_tax,
            'shipping_cost': self.shipping_cost,
            'additional_fees': self.additional_fees,
            'total_cost': self.total_cost,
            'sale_price': self.sale_price,
            'sale_date': self.sale_date.isoformat() if self.sale_date else None,
            'platform_fees': self.platform_fees,
            'profit': self.profit,
            'roi': self.roi,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.metadata
        }


class AlertConfig(db.Model):
    """Model for user alert configurations."""
    
    __tablename__ = 'alert_configs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    
    # Alert type and conditions
    alert_type = db.Column(db.String(50), nullable=False)  # auction_opportunity, bid_spike, etc.
    conditions = db.Column(JSONB, nullable=False)
    
    # Notification settings
    notification_channels = db.Column(JSONB, nullable=False)  # telegram, email, etc.
    is_active = db.Column(db.Boolean, default=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert alert config to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'alert_type': self.alert_type,
            'conditions': self.conditions,
            'notification_channels': self.notification_channels,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class AlertHistory(db.Model):
    """Model for alert history."""
    
    __tablename__ = 'alert_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    alert_config_id = db.Column(db.Integer, db.ForeignKey('alert_configs.id'), nullable=True)
    
    # Alert details
    alert_type = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    data = db.Column(JSONB, default={})
    
    # Notification details
    notification_channels = db.Column(JSONB, default=[])
    delivery_status = db.Column(JSONB, default={})
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert alert history to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'alert_config_id': self.alert_config_id,
            'alert_type': self.alert_type,
            'message': self.message,
            'data': self.data,
            'notification_channels': self.notification_channels,
            'delivery_status': self.delivery_status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }