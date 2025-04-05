"""
Performance tracking module for the Listing Generator.

This module provides functionality for tracking and analyzing
the performance of product listings.
"""

import logging
import asyncio
import random
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

from listing_generator.models import ListingPerformance, ABTestResult
from listing_generator.db import ListingDatabase


logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracker for listing performance.
    
    This class provides methods for tracking and analyzing
    the performance of product listings.
    """
    
    def __init__(self, db: ListingDatabase, config: Dict[str, Any]):
        """
        Initialize the PerformanceTracker.
        
        Args:
            db: Database connection
            config: Configuration dictionary
        """
        self.db = db
        self.config = config
        self.ab_test_duration_days = config.get('ab_test_duration_days', 7)
        self.min_impressions = config.get('min_impressions', 100)
        self.confidence_threshold = config.get('confidence_threshold', 0.95)
        
        logger.info("PerformanceTracker initialized")
    
    async def update_metrics(
        self,
        listing_id: str,
        metrics: Dict[str, Any]
    ) -> Optional[ListingPerformance]:
        """
        Update performance metrics for a listing.
        
        Args:
            listing_id: ID of the listing
            metrics: Performance metrics to update
            
        Returns:
            Updated performance data
        """
        logger.info(f"Updating performance metrics for listing {listing_id}")
        
        try:
            # Get existing performance data
            performance = await self.db.get_listing_performance(listing_id)
            
            if performance:
                # Update existing performance data
                if 'impressions' in metrics:
                    performance.impressions += metrics['impressions']
                if 'clicks' in metrics:
                    performance.clicks += metrics['clicks']
                if 'add_to_carts' in metrics:
                    performance.add_to_carts += metrics['add_to_carts']
                if 'purchases' in metrics:
                    performance.purchases += metrics['purchases']
                if 'revenue' in metrics:
                    performance.revenue += metrics['revenue']
                if 'search_rank' in metrics:
                    performance.search_rank = metrics['search_rank']
                if 'search_terms' in metrics and metrics['search_terms']:
                    # Add new search terms
                    for term in metrics['search_terms']:
                        if term not in performance.search_terms:
                            performance.search_terms.append(term)
            else:
                # Create new performance data
                performance = ListingPerformance(
                    id=str(uuid.uuid4()),
                    listing_id=listing_id,
                    impressions=metrics.get('impressions', 0),
                    clicks=metrics.get('clicks', 0),
                    add_to_carts=metrics.get('add_to_carts', 0),
                    purchases=metrics.get('purchases', 0),
                    revenue=metrics.get('revenue', 0.0),
                    search_rank=metrics.get('search_rank'),
                    search_terms=metrics.get('search_terms', [])
                )
            
            # Save performance data
            await self.db.save_listing_performance(performance)
            
            # Update A/B test metrics if applicable
            await self._update_ab_test_metrics(listing_id, metrics)
            
            logger.info(f"Updated performance metrics for listing {listing_id}")
            return performance
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
            return None
    
    async def get_listing_performance(
        self,
        listing_id: str
    ) -> Optional[ListingPerformance]:
        """
        Get performance data for a listing.
        
        Args:
            listing_id: ID of the listing
            
        Returns:
            Performance data
        """
        return await self.db.get_listing_performance(listing_id)
    
    async def start_ab_test(
        self,
        listing_id: str,
        variation_a_id: str,
        variation_b_id: str
    ) -> Optional[ABTestResult]:
        """
        Start an A/B test for a listing.
        
        Args:
            listing_id: ID of the listing
            variation_a_id: ID of variation A
            variation_b_id: ID of variation B
            
        Returns:
            A/B test result
        """
        logger.info(f"Starting A/B test for listing {listing_id}")
        
        try:
            # Check if there's already an active test
            active_tests = await self.get_active_ab_tests(listing_id)
            if active_tests:
                logger.warning(f"Listing {listing_id} already has active A/B tests")
                return None
            
            # Create new A/B test
            test = ABTestResult(
                id=str(uuid.uuid4()),
                listing_id=listing_id,
                variation_a_id=variation_a_id,
                variation_b_id=variation_b_id,
                start_date=datetime.now()
            )
            
            # Save test
            await self.db.save_ab_test(test)
            
            logger.info(f"Started A/B test {test.id} for listing {listing_id}")
            return test
        
        except Exception as e:
            logger.error(f"Error starting A/B test: {e}")
            return None
    
    async def get_active_ab_tests(
        self,
        listing_id: str
    ) -> List[ABTestResult]:
        """
        Get active A/B tests for a listing.
        
        Args:
            listing_id: ID of the listing
            
        Returns:
            List of active A/B tests
        """
        try:
            # Get all A/B tests for the listing
            all_tests = await self.db.get_ab_tests(listing_id)
            
            # Filter for active tests (no end date)
            active_tests = [test for test in all_tests if test.end_date is None]
            
            return active_tests
        
        except Exception as e:
            logger.error(f"Error getting active A/B tests: {e}")
            return []
    
    async def end_ab_test(
        self,
        test_id: str,
        winner_id: Optional[str] = None
    ) -> Optional[ABTestResult]:
        """
        End an A/B test.
        
        Args:
            test_id: ID of the test
            winner_id: ID of the winning variation
            
        Returns:
            Updated A/B test result
        """
        logger.info(f"Ending A/B test {test_id}")
        
        try:
            # Get test
            test = await self.db.get_ab_test(test_id)
            if not test:
                logger.warning(f"A/B test {test_id} not found")
                return None
            
            # Update test
            test.end_date = datetime.now()
            test.winner_id = winner_id
            
            # Save test
            await self.db.save_ab_test(test)
            
            logger.info(f"Ended A/B test {test_id}")
            return test
        
        except Exception as e:
            logger.error(f"Error ending A/B test: {e}")
            return None
    
    def is_test_complete(self, test: ABTestResult) -> bool:
        """
        Check if an A/B test is complete.
        
        Args:
            test: A/B test to check
            
        Returns:
            True if complete, False otherwise
        """
        # Check if test has already ended
        if test.end_date is not None:
            return True
        
        # Check if test has run for the configured duration
        test_duration = datetime.now() - test.start_date
        if test_duration.days >= self.ab_test_duration_days:
            return True
        
        # Check if test has enough data
        metrics = test.metrics
        if metrics:
            impressions_a = metrics.get('impressions_a', 0)
            impressions_b = metrics.get('impressions_b', 0)
            
            if impressions_a >= self.min_impressions and impressions_b >= self.min_impressions:
                return True
        
        return False
    
    def should_optimize(self, performance: ListingPerformance) -> bool:
        """
        Check if a listing should be optimized based on performance.
        
        Args:
            performance: Listing performance data
            
        Returns:
            True if optimization is recommended, False otherwise
        """
        # Check if there's enough data
        if performance.impressions < 100:
            return False
        
        # Check for poor performance indicators
        if performance.ctr < 0.01:  # Very low CTR
            return True
        
        if performance.conversion_rate < 0.005:  # Very low conversion rate
            return True
        
        if performance.add_to_cart_rate < 0.02:  # Very low add-to-cart rate
            return True
        
        return False
    
    async def _update_ab_test_metrics(
        self,
        listing_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Update metrics for active A/B tests.
        
        Args:
            listing_id: ID of the listing
            metrics: Performance metrics
        """
        try:
            # Get active A/B tests
            active_tests = await self.get_active_ab_tests(listing_id)
            
            if not active_tests:
                return
            
            # Get the listing to access title variations
            listing = await self.db.get_listing(listing_id)
            if not listing:
                return
            
            # Process each active test
            for test in active_tests:
                # Check if test is complete
                if self.is_test_complete(test):
                    # Determine winner
                    winner_id = self._determine_test_winner(test)
                    
                    # End test
                    await self.end_ab_test(test.id, winner_id)
                    continue
                
                # Update test metrics
                if 'variation_id' in metrics:
                    variation_id = metrics['variation_id']
                    
                    # Initialize metrics if needed
                    if not test.metrics:
                        test.metrics = {
                            'impressions_a': 0,
                            'clicks_a': 0,
                            'impressions_b': 0,
                            'clicks_b': 0
                        }
                    
                    # Update metrics for the appropriate variation
                    if variation_id == test.variation_a_id:
                        if 'impressions' in metrics:
                            test.metrics['impressions_a'] = test.metrics.get('impressions_a', 0) + metrics['impressions']
                        if 'clicks' in metrics:
                            test.metrics['clicks_a'] = test.metrics.get('clicks_a', 0) + metrics['clicks']
                    elif variation_id == test.variation_b_id:
                        if 'impressions' in metrics:
                            test.metrics['impressions_b'] = test.metrics.get('impressions_b', 0) + metrics['impressions']
                        if 'clicks' in metrics:
                            test.metrics['clicks_b'] = test.metrics.get('clicks_b', 0) + metrics['clicks']
                
                # Save updated test
                await self.db.save_ab_test(test)
        
        except Exception as e:
            logger.error(f"Error updating A/B test metrics: {e}")
    
    def _determine_test_winner(self, test: ABTestResult) -> Optional[str]:
        """
        Determine the winner of an A/B test.
        
        Args:
            test: A/B test to evaluate
            
        Returns:
            ID of the winning variation or None if inconclusive
        """
        try:
            metrics = test.metrics
            if not metrics:
                return None
            
            # Get metrics
            impressions_a = metrics.get('impressions_a', 0)
            clicks_a = metrics.get('clicks_a', 0)
            impressions_b = metrics.get('impressions_b', 0)
            clicks_b = metrics.get('clicks_b', 0)
            
            # Check if there's enough data
            if impressions_a < self.min_impressions or impressions_b < self.min_impressions:
                return None
            
            # Calculate CTRs
            ctr_a = clicks_a / impressions_a if impressions_a > 0 else 0
            ctr_b = clicks_b / impressions_b if impressions_b > 0 else 0
            
            # Calculate confidence
            p_value = self._calculate_ab_test_significance(
                clicks_a, impressions_a,
                clicks_b, impressions_b
            )
            
            # Determine winner
            if p_value < (1 - self.confidence_threshold):
                if ctr_a > ctr_b:
                    return test.variation_a_id
                else:
                    return test.variation_b_id
            
            # If not statistically significant, return None
            return None
        
        except Exception as e:
            logger.error(f"Error determining test winner: {e}")
            return None
    
    def _calculate_ab_test_significance(
        self,
        clicks_a: int,
        impressions_a: int,
        clicks_b: int,
        impressions_b: int
    ) -> float:
        """
        Calculate statistical significance for an A/B test.
        
        Args:
            clicks_a: Clicks for variation A
            impressions_a: Impressions for variation A
            clicks_b: Clicks for variation B
            impressions_b: Impressions for variation B
            
        Returns:
            p-value (lower means more significant)
        """
        try:
            # Calculate conversion rates
            rate_a = clicks_a / impressions_a if impressions_a > 0 else 0
            rate_b = clicks_b / impressions_b if impressions_b > 0 else 0
            
            # Calculate standard error
            se_a = np.sqrt((rate_a * (1 - rate_a)) / impressions_a) if impressions_a > 0 else 0
            se_b = np.sqrt((rate_b * (1 - rate_b)) / impressions_b) if impressions_b > 0 else 0
            
            # Calculate z-score
            se_diff = np.sqrt(se_a**2 + se_b**2)
            z_score = abs(rate_a - rate_b) / se_diff if se_diff > 0 else 0
            
            # Calculate p-value
            p_value = 2 * (1 - stats.norm.cdf(z_score))
            
            return p_value
        
        except Exception as e:
            logger.error(f"Error calculating A/B test significance: {e}")
            return 1.0  # Return 1.0 (not significant) on error