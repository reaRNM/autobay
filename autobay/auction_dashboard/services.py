"""
Service layer for the Dashboard & Mobile Alerts module.

This module provides business logic for dashboard metrics,
data aggregation, and integration with external services.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import func, desc, and_, or_

from .models import db, AuctionItem, AuctionHistory, User

logger = logging.getLogger(__name__)


class MetricsService:
    """Service for calculating and retrieving dashboard metrics."""
    
    @staticmethod
    def get_profit_metrics(user_id: Optional[int] = None, 
                         time_range: str = 'all') -> Dict[str, Any]:
        """
        Get profit metrics for the dashboard.
        
        Args:
            user_id: Optional user ID to filter by
            time_range: Time range to filter by ('day', 'week', 'month', 'year', 'all')
            
        Returns:
            Dict[str, Any]: Profit metrics
        """
        # Build the query
        query = AuctionHistory.query
        
        # Filter by user if specified
        if user_id is not None:
            query = query.filter(AuctionHistory.user_id == user_id)
        
        # Filter by time range
        if time_range != 'all':
            now = datetime.utcnow()
            if time_range == 'day':
                start_date = now - timedelta(days=1)
            elif time_range == 'week':
                start_date = now - timedelta(weeks=1)
            elif time_range == 'month':
                start_date = now - timedelta(days=30)
            elif time_range == 'year':
                start_date = now - timedelta(days=365)
            else:
                start_date = None
            
            if start_date:
                query = query.filter(AuctionHistory.sale_date >= start_date)
        
        # Filter by sold items
        query = query.filter(AuctionHistory.status == 'sold')
        
        # Get total profit
        total_profit = db.session.query(func.sum(AuctionHistory.profit)).scalar() or 0
        
        # Get average profit
        avg_profit = db.session.query(func.avg(AuctionHistory.profit)).scalar() or 0
        
        # Get average ROI
        avg_roi = db.session.query(func.avg(AuctionHistory.roi)).scalar() or 0
        
        # Get total items sold
        total_sold = query.count()
        
        # Get profit by category
        profit_by_category = []
        categories = db.session.query(
            AuctionItem.category,
            func.sum(AuctionHistory.profit).label('total_profit'),
            func.count().label('count')
        ).join(
            AuctionItem,
            AuctionHistory.item_id == AuctionItem.item_id
        ).filter(
            AuctionHistory.status == 'sold'
        ).group_by(
            AuctionItem.category
        ).all()
        
        for category, profit, count in categories:
            profit_by_category.append({
                'category': category or 'Uncategorized',
                'profit': float(profit) if profit else 0,
                'count': count
            })
        
        # Get profit over time
        profit_over_time = []
        time_periods = db.session.query(
            func.date_trunc('month', AuctionHistory.sale_date).label('month'),
            func.sum(AuctionHistory.profit).label('total_profit'),
            func.count().label('count')
        ).filter(
            AuctionHistory.status == 'sold'
        ).group_by(
            'month'
        ).order_by(
            'month'
        ).all()
        
        for period, profit, count in time_periods:
            if period:
                profit_over_time.append({
                    'period': period.strftime('%Y-%m'),
                    'profit': float(profit) if profit else 0,
                    'count': count
                })
        
        return {
            'total_profit': float(total_profit),
            'average_profit': float(avg_profit),
            'average_roi': float(avg_roi) if avg_roi else 0,
            'total_sold': total_sold,
            'profit_by_category': profit_by_category,
            'profit_over_time': profit_over_time
        }
    
    @staticmethod
    def get_auction_metrics(user_id: Optional[int] = None,
                          time_range: str = 'all') -> Dict[str, Any]:
        """
        Get auction metrics for the dashboard.
        
        Args:
            user_id: Optional user ID to filter by
            time_range: Time range to filter by ('day', 'week', 'month', 'year', 'all')
            
        Returns:
            Dict[str, Any]: Auction metrics
        """
        # Build the query
        query = AuctionHistory.query
        
        # Filter by user if specified
        if user_id is not None:
            query = query.filter(AuctionHistory.user_id == user_id)
        
        # Filter by time range
        if time_range != 'all':
            now = datetime.utcnow()
            if time_range == 'day':
                start_date = now - timedelta(days=1)
            elif time_range == 'week':
                start_date = now - timedelta(weeks=1)
            elif time_range == 'month':
                start_date = now - timedelta(days=30)
            elif time_range == 'year':
                start_date = now - timedelta(days=365)
            else:
                start_date = None
            
            if start_date:
                query = query.filter(AuctionHistory.created_at >= start_date)
        
        # Get total auctions won
        total_won = query.count()
        
        # Get total purchase value
        total_purchase = db.session.query(func.sum(AuctionHistory.total_cost)).scalar() or 0
        
        # Get average purchase price
        avg_purchase = db.session.query(func.avg(AuctionHistory.total_cost)).scalar() or 0
        
        # Get status breakdown
        status_breakdown = []
        statuses = db.session.query(
            AuctionHistory.status,
            func.count().label('count')
        ).group_by(
            AuctionHistory.status
        ).all()
        
        for status, count in statuses:
            status_breakdown.append({
                'status': status,
                'count': count
            })
        
        # Get auctions by category
        auctions_by_category = []
        categories = db.session.query(
            AuctionItem.category,
            func.count().label('count')
        ).join(
            AuctionHistory,
            AuctionHistory.item_id == AuctionItem.item_id
        ).group_by(
            AuctionItem.category
        ).all()
        
        for category, count in categories:
            auctions_by_category.append({
                'category': category or 'Uncategorized',
                'count': count
            })
        
        return {
            'total_won': total_won,
            'total_purchase_value': float(total_purchase),
            'average_purchase_price': float(avg_purchase),
            'status_breakdown': status_breakdown,
            'auctions_by_category': auctions_by_category
        }
    
    @staticmethod
    def get_risk_metrics(user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get risk metrics for the dashboard.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            Dict[str, Any]: Risk metrics
        """
        # Build the query
        query = AuctionItem.query
        
        # Get average risk score
        avg_risk = db.session.query(func.avg(AuctionItem.risk_score)).scalar() or 0
        
        # Get risk score distribution
        risk_distribution = []
        
        # Define risk score ranges
        ranges = [
            (0.0, 0.2, 'Very Low'),
            (0.2, 0.4, 'Low'),
            (0.4, 0.6, 'Medium'),
            (0.6, 0.8, 'High'),
            (0.8, 1.0, 'Very High')
        ]
        
        for min_risk, max_risk, label in ranges:
            count = AuctionItem.query.filter(
                AuctionItem.risk_score >= min_risk,
                AuctionItem.risk_score < max_risk
            ).count()
            
            risk_distribution.append({
                'range': label,
                'min': min_risk,
                'max': max_risk,
                'count': count
            })
        
        # Get risk factors
        risk_factors = {}
        items = AuctionItem.query.all()
        
        for item in items:
            if item.risk_factors:
                for factor, value in item.risk_factors.items():
                    if factor not in risk_factors:
                        risk_factors[factor] = []
                    
                    risk_factors[factor].append(value)
        
        # Calculate average for each risk factor
        risk_factor_averages = {}
        for factor, values in risk_factors.items():
            if values:
                risk_factor_averages[factor] = sum(values) / len(values)
        
        return {
            'average_risk_score': float(avg_risk),
            'risk_distribution': risk_distribution,
            'risk_factor_averages': risk_factor_averages
        }
    
    @staticmethod
    def get_grand_ranking(filters: Dict[str, Any] = None, 
                        limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the grand ranking of auction items.
        
        Args:
            filters: Filters to apply
            limit: Maximum number of items to return
            
        Returns:
            List[Dict[str, Any]]: Ranked auction items
        """
        # Build the query
        query = AuctionItem.query
        
        # Apply filters
        if filters:
            if 'category' in filters:
                query = query.filter(AuctionItem.category == filters['category'])
            
            if 'min_profit' in filters:
                query = query.filter(AuctionItem.estimated_profit >= filters['min_profit'])
            
            if 'max_risk' in filters:
                query = query.filter(AuctionItem.risk_score <= filters['max_risk'])
            
            if 'status' in filters:
                query = query.filter(AuctionItem.status == filters['status'])
        
        # Order by grand ranking score (descending)
        query = query.order_by(desc(AuctionItem.grand_ranking_score))
        
        # Limit the results
        query = query.limit(limit)
        
        # Get the items
        items = query.all()
        
        # Convert to dictionaries
        return [item.to_dict() for item in items]


class DashboardService:
    """Service for dashboard data and operations."""
    
    def __init__(self, metrics_service: MetricsService = None):
        """Initialize the dashboard service."""
        self.metrics_service = metrics_service or MetricsService()
    
    def get_dashboard_data(self, user_id: Optional[int] = None,
                         time_range: str = 'all') -> Dict[str, Any]:
        """
        Get all dashboard data.
        
        Args:
            user_id: Optional user ID to filter by
            time_range: Time range to filter by
            
        Returns:
            Dict[str, Any]: Dashboard data
        """
        # Get metrics
        profit_metrics = self.metrics_service.get_profit_metrics(user_id, time_range)
        auction_metrics = self.metrics_service.get_auction_metrics(user_id, time_range)
        risk_metrics = self.metrics_service.get_risk_metrics(user_id)
        
        # Get top ranked items
        top_items = self.metrics_service.get_grand_ranking(limit=10)
        
        # Get recent auction history
        recent_history = self._get_recent_auction_history(user_id, limit=10)
        
        return {
            'profit_metrics': profit_metrics,
            'auction_metrics': auction_metrics,
            'risk_metrics': risk_metrics,
            'top_ranked_items': top_items,
            'recent_history': recent_history
        }
    
    def _get_recent_auction_history(self, user_id: Optional[int] = None,
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent auction history.
        
        Args:
            user_id: Optional user ID to filter by
            limit: Maximum number of items to return
            
        Returns:
            List[Dict[str, Any]]: Recent auction history
        """
        # Build the query
        query = AuctionHistory.query
        
        # Filter by user if specified
        if user_id is not None:
            query = query.filter(AuctionHistory.user_id == user_id)
        
        # Order by created_at (descending)
        query = query.order_by(desc(AuctionHistory.created_at))
        
        # Limit the results
        query = query.limit(limit)
        
        # Get the items
        items = query.all()
        
        # Convert to dictionaries
        return [item.to_dict() for item in items]
    
    def calculate_fees(self, bid_amount: float, buyer_premium_rate: float,
                     sales_tax_rate: float, shipping_cost: float,
                     additional_fees: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Calculate fees for an auction item.
        
        Args:
            bid_amount: Bid amount
            buyer_premium_rate: Buyer premium rate (as a decimal)
            sales_tax_rate: Sales tax rate (as a decimal)
            shipping_cost: Shipping cost
            additional_fees: Additional fees
            
        Returns:
            Dict[str, Any]: Fee calculation results
        """
        # Calculate buyer premium
        buyer_premium = bid_amount * buyer_premium_rate
        
        # Calculate sales tax (on bid amount + buyer premium)
        taxable_amount = bid_amount + buyer_premium
        sales_tax = taxable_amount * sales_tax_rate
        
        # Calculate additional fees
        additional_fees = additional_fees or {}
        additional_fees_total = sum(additional_fees.values())
        
        # Calculate total cost
        total_cost = bid_amount + buyer_premium + sales_tax + shipping_cost + additional_fees_total
        
        return {
            'bid_amount': bid_amount,
            'buyer_premium': buyer_premium,
            'buyer_premium_rate': buyer_premium_rate,
            'sales_tax': sales_tax,
            'sales_tax_rate': sales_tax_rate,
            'shipping_cost': shipping_cost,
            'additional_fees': additional_fees,
            'additional_fees_total': additional_fees_total,
            'total_cost': total_cost
        }
    
    def calculate_profit(self, purchase_price: float, sale_price: float,
                       platform_fee_rate: float, shipping_cost: float,
                       additional_costs: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Calculate profit for an auction item.
        
        Args:
            purchase_price: Purchase price
            sale_price: Sale price
            platform_fee_rate: Platform fee rate (as a decimal)
            shipping_cost: Shipping cost
            additional_costs: Additional costs
            
        Returns:
            Dict[str, Any]: Profit calculation results
        """
        # Calculate platform fee
        platform_fee = sale_price * platform_fee_rate
        
        # Calculate additional costs
        additional_costs = additional_costs or {}
        additional_costs_total = sum(additional_costs.values())
        
        # Calculate total cost
        total_cost = purchase_price + shipping_cost + additional_costs_total
        
        # Calculate net revenue
        net_revenue = sale_price - platform_fee
        
        # Calculate profit
        profit = net_revenue - total_cost
        
        # Calculate ROI
        roi = (profit / total_cost) * 100 if total_cost > 0 else 0
        
        return {
            'purchase_price': purchase_price,
            'sale_price': sale_price,
            'platform_fee': platform_fee,
            'platform_fee_rate': platform_fee_rate,
            'shipping_cost': shipping_cost,
            'additional_costs': additional_costs,
            'additional_costs_total': additional_costs_total,
            'total_cost': total_cost,
            'net_revenue': net_revenue,
            'profit': profit,
            'roi': roi
        }
    
    def update_grand_ranking(self, item_id: str) -> Dict[str, Any]:
        """
        Update the grand ranking score for an auction item.
        
        Args:
            item_id: ID of the auction item
            
        Returns:
            Dict[str, Any]: Updated item data
        """
        # Get the item
        item = AuctionItem.query.filter_by(item_id=item_id).first()
        
        if not item:
            logger.error(f"Item {item_id} not found")
            return {
                'success': False,
                'error': 'Item not found'
            }
        
        # Calculate grand ranking score
        # This is a weighted combination of profit potential, risk, and shipping ease
        profit_weight = 0.5
        risk_weight = 0.3
        shipping_weight = 0.2
        
        # Normalize profit margin to 0-1 scale (assuming 100% is max)
        profit_score = min(1.0, item.profit_margin / 100) if item.profit_margin else 0
        
        # Risk score is already 0-1 (lower is better)
        risk_score = 1.0 - item.risk_score if item.risk_score is not None else 0.5
        
        # Shipping ease score is 0-1 (higher is better)
        shipping_score = item.shipping_ease_score if item.shipping_ease_score is not None else 0.5
        
        # Calculate weighted score
        grand_ranking_score = (
            profit_score * profit_weight +
            risk_score * risk_weight +
            shipping_score * shipping_weight
        )
        
        # Update the item
        item.grand_ranking_score = grand_ranking_score
        db.session.commit()
        
        logger.info(f"Updated grand ranking score for item {item_id}: {grand_ranking_score}")
        
        return {
            'success': True,
            'item': item.to_dict()
        }
    
    def update_all_grand_rankings(self) -> Dict[str, Any]:
        """
        Update grand ranking scores for all auction items.
        
        Returns:
            Dict[str, Any]: Update results
        """
        # Get all items
        items = AuctionItem.query.all()
        
        updated_count = 0
        for item in items:
            result = self.update_grand_ranking(item.item_id)
            if result.get('success'):
                updated_count += 1
        
        logger.info(f"Updated grand ranking scores for {updated_count} items")
        
        return {
            'success': True,
            'updated_count': updated_count,
            'total_count': len(items)
        }