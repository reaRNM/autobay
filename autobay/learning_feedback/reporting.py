"""
Reporting module for the Learning & Feedback Systems Module.

This module provides functionality for generating reports
and visualizations.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
import json
import csv

from learning_feedback.models import (
    AuctionOutcome, ModelPerformance, AuctionStatus
)
from learning_feedback.db import Database
from learning_feedback.utils import format_price, save_json


logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generator for reports and visualizations.
    
    This class provides methods for generating reports
    and visualizations.
    """
    
    def __init__(self, db: Database):
        """
        Initialize the ReportGenerator.
        
        Args:
            db: Database connection
        """
        self.db = db
        
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        logger.info("ReportGenerator initialized")
    
    async def generate_performance_report(
        self,
        user_id: Optional[str] = None,
        category_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        output_format: str = 'json'
    ) -> str:
        """
        Generate a performance report.
        
        Args:
            user_id: Filter by user ID (optional)
            category_id: Filter by category ID (optional)
            start_date: Start date (optional)
            end_date: End date (optional)
            output_format: Output format (json, csv)
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating performance report (user_id={user_id}, category_id={category_id})")
        
        try:
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now()
            
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Get auction outcomes
            auctions = await self.db.get_auction_outcomes(
                user_id=user_id,
                category_id=category_id,
                limit=1000
            )
            
            # Filter by date
            auctions = [a for a in auctions if a.listing_date >= start_date and a.listing_date <= end_date]
            
            # Calculate metrics
            total_auctions = len(auctions)
            sold_auctions = [a for a in auctions if a.status == AuctionStatus.SOLD]
            unsold_auctions = [a for a in auctions if a.status == AuctionStatus.UNSOLD]
            
            success_rate = len(sold_auctions) / total_auctions if total_auctions > 0 else 0
            
            total_revenue = sum(a.final_price or 0 for a in sold_auctions)
            total_profit = sum(a.profit or 0 for a in sold_auctions)
            total_roi = sum(a.roi or 0 for a in sold_auctions)
            
            avg_revenue = total_revenue / len(sold_auctions) if sold_auctions else 0
            avg_profit = total_profit / len(sold_auctions) if sold_auctions else 0
            avg_roi = total_roi / len(sold_auctions) if sold_auctions else 0
            
            # Calculate price accuracy
            accuracies = [a.price_accuracy for a in auctions if a.price_accuracy is not None]
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            
            # Calculate time to sale
            times_to_sale = [a.time_to_sale for a in sold_auctions if a.time_to_sale is not None]
            avg_time_to_sale = sum(times_to_sale) / len(times_to_sale) if times_to_sale else 0
            
            # Create report data
            report_data = {
                'report_id': str(uuid.uuid4()),
                'user_id': user_id,
                'category_id': category_id,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'generated_at': datetime.now().isoformat(),
                'metrics': {
                    'total_auctions': total_auctions,
                    'sold_auctions': len(sold_auctions),
                    'unsold_auctions': len(unsold_auctions),
                    'success_rate': success_rate * 100,
                    'total_revenue': total_revenue,
                    'total_profit': total_profit,
                    'avg_revenue': avg_revenue,
                    'avg_profit': avg_profit,
                    'avg_roi': avg_roi * 100 if avg_roi else 0,
                    'avg_price_accuracy': avg_accuracy,
                    'avg_time_to_sale': avg_time_to_sale
                },
                'auctions': [a.to_dict() for a in auctions]
            }
            
            # Generate report file
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"performance_report_{timestamp}"
            
            if output_format == 'json':
                # Save as JSON
                file_path = f"reports/{filename}.json"
                save_json(report_data, file_path)
            
            elif output_format == 'csv':
                # Save as CSV
                file_path = f"reports/{filename}.csv"
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Write metrics to CSV
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow(['Metric', 'Value'])
                    
                    # Write metrics
                    writer.writerow(['Total Auctions', total_auctions])
                    writer.writerow(['Sold Auctions', len(sold_auctions)])
                    writer.writerow(['Unsold Auctions', len(unsold_auctions)])
                    writer.writerow(['Success Rate', f"{success_rate * 100:.2f}%"])
                    writer.writerow(['Total Revenue', format_price(total_revenue)])
                    writer.writerow(['Total Profit', format_price(total_profit)])
                    writer.writerow(['Average Revenue', format_price(avg_revenue)])
                    writer.writerow(['Average Profit', format_price(avg_profit)])
                    writer.writerow(['Average ROI', f"{avg_roi * 100:.2f}%"])
                    writer.writerow(['Average Price Accuracy', f"{avg_accuracy:.2f}%"])
                    writer.writerow(['Average Time to Sale', f"{avg_time_to_sale:.2f} hours"])
            
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            logger.info(f"Generated performance report: {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            raise
    
    async def generate_model_performance_report(
        self,
        model_type: str,
        category_id: Optional[str] = None,
        output_format: str = 'json'
    ) -> str:
        """
        Generate a model performance report.
        
        Args:
            model_type: Type of model (pricing, bidding, shipping)
            category_id: Category ID (optional)
            output_format: Output format (json, csv)
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating model performance report (model_type={model_type}, category_id={category_id})")
        
        try:
            # Get model performance
            performance = await self.db.get_model_performance(
                model_type=model_type,
                category_id=category_id,
                limit=10
            )
            
            if not performance:
                raise ValueError(f"No performance data found for model type {model_type}")
            
            # Create report data
            report_data = {
                'report_id': str(uuid.uuid4()),
                'model_type': model_type,
                'category_id': category_id,
                'generated_at': datetime.now().isoformat(),
                'performance': [p.to_dict() for p in performance]
            }
            
            # Generate report file
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"model_performance_report_{model_type}_{timestamp}"
            
            if output_format == 'json':
                # Save as JSON
                file_path = f"reports/{filename}.json"
                save_json(report_data, file_path)
            
            elif output_format == 'csv':
                # Save as CSV
                file_path = f"reports/{filename}.csv"
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Write performance to CSV
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    if model_type in ['pricing', 'shipping']:
                        writer.writerow(['Model Name', 'Version', 'Accuracy', 'MAE', 'MSE', 'R²', 'Sample Count', 'Created At'])
                    else:
                        writer.writerow(['Model Name', 'Version', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Sample Count', 'Created At'])
                    
                    # Write performance data
                    for p in performance:
                        if model_type in ['pricing', 'shipping']:
                            writer.writerow([
                                p.model_name,
                                p.model_version,
                                f"{p.accuracy * 100:.2f}%",
                                p.mean_absolute_error,
                                p.mean_squared_error,
                                p.r_squared,
                                p.sample_count,
                                p.created_at.isoformat()
                            ])
                        else:
                            writer.writerow([
                                p.model_name,
                                p.model_version,
                                f"{p.accuracy * 100:.2f}%",
                                f"{p.precision * 100:.2f}%",
                                f"{p.recall * 100:.2f}%",
                                f"{p.f1_score * 100:.2f}%",
                                p.sample_count,
                                p.created_at.isoformat()
                            ])
            
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            logger.info(f"Generated model performance report: {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Error generating model performance report: {e}")
            raise
    
    async def generate_feedback_report(
        self,
        user_id: Optional[str] = None,
        feedback_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        output_format: str = 'json'
    ) -> str:
        """
        Generate a feedback report.
        
        Args:
            user_id: Filter by user ID (optional)
            feedback_type: Filter by feedback type (optional)
            start_date: Start date (optional)
            end_date: End date (optional)
            output_format: Output format (json, csv)
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating feedback report (user_id={user_id}, feedback_type={feedback_type})")
        
        try:
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now()
            
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Get user feedback
            feedback = await self.db.get_user_feedback(
                user_id=user_id,
                feedback_type=feedback_type,
                limit=1000
            )
            
            # Filter by date
            feedback = [f for f in feedback if f.created_at >= start_date and f.created_at <= end_date]
            
            # Calculate metrics
            total_feedback = len(feedback)
            
            # Count by sentiment
            positive_feedback = [f for f in feedback if f.sentiment == 'positive']
            neutral_feedback = [f for f in feedback if f.sentiment == 'neutral']
            negative_feedback = [f for f in feedback if f.sentiment == 'negative']
            
            # Calculate average rating
            ratings = [f.rating for f in feedback if f.rating is not None]
            avg_rating = sum(ratings) / len(ratings) if ratings else None
            
            # Create report data
            report_data = {
                'report_id': str(uuid.uuid4()),
                'user_id': user_id,
                'feedback_type': feedback_type,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'generated_at': datetime.now().isoformat(),
                'metrics': {
                    'total_feedback': total_feedback,
                    'positive_feedback': len(positive_feedback),
                    'neutral_feedback': len(neutral_feedback),
                    'negative_feedback': len(negative_feedback),
                    'average_rating': avg_rating
                },
                'feedback': [f.to_dict() for f in feedback]
            }
            
            # Generate report file
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"feedback_report_{timestamp}"
            
            if output_format == 'json':
                # Save as JSON
                file_path = f"reports/{filename}.json"
                save_json(report_data, file_path)
            
            elif output_format == 'csv':
                # Save as CSV
                file_path = f"reports/{filename}.csv"
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Write metrics to CSV
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow(['Metric', 'Value'])
                    
                    # Write metrics
                    writer.writerow(['Total Feedback', total_feedback])
                    writer.writerow(['Positive Feedback', len(positive_feedback)])
                    writer.writerow(['Neutral Feedback', len(neutral_feedback)])
                    writer.writerow(['Negative Feedback', len(negative_feedback)])
                    writer.writerow(['Average Rating', avg_rating])
            
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            logger.info(f"Generated feedback report: {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Error generating feedback report: {e}")
            raise