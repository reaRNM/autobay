"""
Unit tests for the Dashboard & Mobile Alerts module.
"""

import unittest
import json
from datetime import datetime, timedelta

from auction_dashboard import create_app
from auction_dashboard.models import db, User, AuctionItem, AuctionHistory, AlertConfig
from auction_dashboard.services import DashboardService, MetricsService
from auction_dashboard.utils import calculate_risk_score


class TestDashboardModule(unittest.TestCase):
    """Test cases for the Dashboard & Mobile Alerts module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create Flask app with test configuration
        self.app = create_app({
            'TESTING': True,
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'SECRET_KEY': 'test_key'
        })
        
        # Create application context
        self.app_context = self.app.app_context()
        self.app_context.push()
        
        # Create database tables
        db.create_all()
        
        # Create test user
        self.user = User(
            username='testuser',
            email='test@example.com'
        )
        self.user.set_password('password')
        db.session.add(self.user)
        db.session.commit()
        
        # Create test data
        self._create_test_data()
    
    def tearDown(self):
        """Tear down test fixtures."""
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
    
    def _create_test_data(self):
        """Create test data for unit tests."""
        # Create test auction items
        for i in range(1, 6):
            # Calculate risk score
            risk_score = calculate_risk_score(
                price_volatility=0.2,
                historical_data_points=10 + i,
                category_risk=0.3,
                seller_rating=0.8,
                time_pressure=0.4
            )
            
            # Create item
            item = AuctionItem(
                item_id=f'test-item-{i}',
                title=f'Test Item {i}',
                category='Test Category',
                condition='New',
                auction_id=f'test-auction-{i}',
                auction_end_time=datetime.utcnow() + timedelta(days=i),
                starting_bid=50.0 * i,
                current_bid=60.0 * i,
                estimated_value=100.0 * i,
                estimated_profit=25.0 * i,
                profit_margin=30.0,
                risk_score=risk_score.score,
                risk_factors=risk_score.factors,
                shipping_ease_score=0.7,
                estimated_shipping_cost=10.0,
                grand_ranking_score=0.0,  # Will be calculated later
                status='active'
            )
            
            db.session.add(item)
        
        # Create test auction history
        for i in range(1, 4):
            item = AuctionItem.query.filter_by(item_id=f'test-item-{i}').first()
            
            history = AuctionHistory(
                user_id=self.user.id,
                item_id=item.item_id,
                title=item.title,
                auction_id=item.auction_id,
                auction_end_time=datetime.utcnow() - timedelta(days=10),
                purchase_price=item.current_bid,
                buyer_premium=item.current_bid * 0.15,
                sales_tax=(item.current_bid + item.current_bid * 0.15) * 0.07,
                shipping_cost=item.estimated_shipping_cost,
                additional_fees={'handling': 5.0},
                total_cost=item.current_bid * 1.25,
                sale_price=item.estimated_value,
                sale_date=datetime.utcnow() - timedelta(days=5),
                platform_fees=item.estimated_value * 0.1275,
                profit=item.estimated_value * 0.8725 - item.current_bid * 1.25,
                roi=((item.estimated_value * 0.8725 - item.current_bid * 1.25) / (item.current_bid * 1.25)) * 100,
                status='sold'
            )
            
            db.session.add(history)
        
        # Create test alert config
        alert_config = AlertConfig(
            user_id=self.user.id,
            name='Test Alert',
            description='Test alert configuration',
            alert_type='auction_opportunity',
            conditions={
                'estimated_profit': {'operator': '>=', 'value': 50.0},
                'risk_score': {'operator': '<=', 'value': 0.5}
            },
            notification_channels=['telegram'],
            is_active=True
        )
        
        db.session.add(alert_config)
        db.session.commit()
        
        # Calculate grand ranking scores
        dashboard_service = DashboardService()
        dashboard_service.update_all_grand_rankings()
    
    def test_dashboard_service(self):
        """Test the dashboard service."""
        dashboard_service = DashboardService()
        
        # Test get_dashboard_data
        dashboard_data = dashboard_service.get_dashboard_data(self.user.id)
        
        self.assertIsNotNone(dashboard_data)
        self.assertIn('profit_metrics', dashboard_data)
        self.assertIn('auction_metrics', dashboard_data)
        self.assertIn('risk_metrics', dashboard_data)
        self.assertIn('top_ranked_items', dashboard_data)
        self.assertIn('recent_history', dashboard_data)
        
        # Test calculate_fees
        fee_calculation = dashboard_service.calculate_fees(
            bid_amount=100.0,
            buyer_premium_rate=0.15,
            sales_tax_rate=0.07,
            shipping_cost=10.0,
            additional_fees={'handling': 5.0}
        )
        
        self.assertIsNotNone(fee_calculation)
        self.assertEqual(fee_calculation['bid_amount'], 100.0)
        self.assertEqual(fee_calculation['buyer_premium'], 15.0)
        self.assertEqual(fee_calculation['sales_tax'], 8.05)
        self.assertEqual(fee_calculation['shipping_cost'], 10.0)
        self.assertEqual(fee_calculation['additional_fees']['handling'], 5.0)
        self.assertEqual(fee_calculation['total_cost'], 138.05)
        
        # Test calculate_profit
        profit_calculation = dashboard_service.calculate_profit(
            purchase_price=100.0,
            sale_price=150.0,
            platform_fee_rate=0.1275,
            shipping_cost=10.0
        )
        
        self.assertIsNotNone(profit_calculation)
        self.assertEqual(profit_calculation['purchase_price'], 100.0)
        self.assertEqual(profit_calculation['sale_price'], 150.0)
        self.assertEqual(profit_calculation['platform_fee'], 19.125)
        self.assertEqual(profit_calculation['shipping_cost'], 10.0)
        self.assertEqual(profit_calculation['net_revenue'], 130.875)
        self.assertEqual(profit_calculation['profit'], 20.875)
    
    def test_metrics_service(self):
        """Test the metrics service."""
        metrics_service = MetricsService()
        
        # Test get_profit_metrics
        profit_metrics = metrics_service.get_profit_metrics(self.user.id)
        
        self.assertIsNotNone(profit_metrics)
        self.assertIn('total_profit', profit_metrics)
        self.assertIn('average_profit', profit_metrics)
        self.assertIn('average_roi', profit_metrics)
        self.assertIn('total_sold', profit_metrics)
        
        # Test get_auction_metrics
        auction_metrics = metrics_service.get_auction_metrics(self.user.id)
        
        self.assertIsNotNone(auction_metrics)
        self.assertIn('total_won', auction_metrics)
        self.assertIn('total_purchase_value', auction_metrics)
        self.assertIn('average_purchase_price', auction_metrics)
        
        # Test get_risk_metrics
        