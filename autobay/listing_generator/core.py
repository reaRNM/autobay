"""
Core functionality for the Listing Generator Module.

This module provides the main ListingGenerator class that handles
listing generation, optimization, and management.
"""

import os
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime

from listing_generator.models import (
    Product, Listing, ListingPerformance, TitleVariation,
    PricingRecommendation, ImageMetadata, Marketplace, ListingStatus
)
from listing_generator.nlp import TitleGenerator, DescriptionGenerator, KeywordOptimizer
from listing_generator.pricing import PricingEngine
from listing_generator.images import ImageProcessor
from listing_generator.performance import PerformanceTracker
from listing_generator.db import ListingDatabase
from listing_generator.utils import setup_logging


logger = logging.getLogger(__name__)


class ListingGenerator:
    """
    Main class for listing generation and optimization.
    
    This class provides methods for generating optimized listings,
    tracking performance, and improving listings over time.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        db_connection: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the ListingGenerator.
        
        Args:
            config_path: Path to configuration file
            db_connection: Database connection string
            openai_api_key: OpenAI API key
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set OpenAI API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif "OPENAI_API_KEY" not in os.environ and "openai_api_key" in self.config:
            os.environ["OPENAI_API_KEY"] = self.config["openai_api_key"]
        
        # Initialize database connection
        self.db = ListingDatabase(db_connection or self.config.get('db_connection'))
        
        # Initialize components
        self.title_generator = TitleGenerator(self.config.get('title_generator', {}))
        self.description_generator = DescriptionGenerator(self.config.get('description_generator', {}))
        self.keyword_optimizer = KeywordOptimizer(self.config.get('keyword_optimizer', {}))
        self.pricing_engine = PricingEngine(self.config.get('pricing_engine', {}))
        self.image_processor = ImageProcessor(self.config.get('image_processor', {}))
        self.performance_tracker = PerformanceTracker(
            self.db,
            self.config.get('performance_tracker', {})
        )
        
        logger.info("ListingGenerator initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'db_connection': os.environ.get('LISTING_DB_CONNECTION', 'sqlite:///listings.db'),
            'openai_api_key': os.environ.get('OPENAI_API_KEY', ''),
            'title_generator': {
                'max_length': 80,
                'num_variations': 3,
                'model': 'gpt-4o',
                'temperature': 0.7
            },
            'description_generator': {
                'max_length': 2000,
                'model': 'gpt-4o',
                'temperature': 0.7,
                'include_bullets': True,
                'include_specifications': True
            },
            'keyword_optimizer': {
                'max_keywords': 20,
                'min_keyword_length': 3,
                'use_marketplace_data': True
            },
            'pricing_engine': {
                'margin_target': 0.3,
                'competitor_weight': 0.6,
                'historical_weight': 0.3,
                'trend_weight': 0.1
            },
            'image_processor': {
                'generate_alt_text': True,
                'generate_captions': True,
                'suggest_enhancements': True,
                'max_tags': 10
            },
            'performance_tracker': {
                'ab_test_duration_days': 7,
                'min_impressions': 100,
                'confidence_threshold': 0.95
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
    
    async def generate_listing(
        self,
        product: Product,
        marketplace: Marketplace,
        generate_variations: bool = True,
        num_variations: int = 3
    ) -> Listing:
        """
        Generate an optimized listing for a product.
        
        Args:
            product: Product to list
            marketplace: Target marketplace
            generate_variations: Whether to generate title variations
            num_variations: Number of title variations to generate
            
        Returns:
            Generated listing
        """
        logger.info(f"Generating listing for product {product.id} on {marketplace}")
        
        try:
            # Generate optimized title
            title = await self.title_generator.generate_title(product, marketplace)
            
            # Generate title variations if requested
            title_variations = []
            if generate_variations:
                variations = await self.title_generator.generate_variations(
                    product, 
                    marketplace, 
                    num_variations
                )
                
                for variation in variations:
                    title_variations.append(TitleVariation(
                        id=str(uuid.uuid4()),
                        product_id=product.id,
                        title=variation,
                        marketplace=marketplace
                    ))
            
            # Generate optimized description
            description = await self.description_generator.generate_description(product, marketplace)
            
            # Extract and optimize keywords
            keywords = await self.keyword_optimizer.extract_keywords(
                product, 
                title, 
                description, 
                marketplace
            )
            
            # Get pricing recommendation
            pricing = await self.pricing_engine.get_pricing_recommendation(product, marketplace)
            
            # Process images
            image_metadata = []
            for image_url in product.image_urls:
                metadata = await self.image_processor.process_image(
                    product, 
                    image_url, 
                    is_primary=(image_url == product.image_urls[0])
                )
                image_metadata.append(metadata)
            
            # Create listing
            listing = Listing(
                id=str(uuid.uuid4()),
                product_id=product.id,
                marketplace=marketplace,
                title=title,
                description=description,
                price=pricing.recommended_price,
                quantity=1,  # Default quantity
                title_variations=title_variations,
                image_metadata=image_metadata,
                pricing_recommendation=pricing,
                keywords=keywords
            )
            
            # Save listing to database
            await self.db.save_listing(listing)
            
            logger.info(f"Generated listing {listing.id} for product {product.id}")
            return listing
        
        except Exception as e:
            logger.error(f"Error generating listing for product {product.id}: {e}")
            raise
    
    async def update_listing(
        self,
        listing_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Listing]:
        """
        Update an existing listing.
        
        Args:
            listing_id: ID of listing to update
            updates: Dictionary of updates to apply
            
        Returns:
            Updated listing or None if not found
        """
        logger.info(f"Updating listing {listing_id}")
        
        try:
            # Get existing listing
            listing = await self.db.get_listing(listing_id)
            if not listing:
                logger.warning(f"Listing {listing_id} not found")
                return None
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(listing, key):
                    setattr(listing, key, value)
            
            # Update timestamp
            listing.updated_at = datetime.now()
            
            # Save updated listing
            await self.db.save_listing(listing)
            
            logger.info(f"Updated listing {listing_id}")
            return listing
        
        except Exception as e:
            logger.error(f"Error updating listing {listing_id}: {e}")
            raise
    
    async def optimize_title(
        self,
        listing_id: str,
        use_ab_testing: bool = True
    ) -> Optional[Listing]:
        """
        Optimize the title of an existing listing.
        
        Args:
            listing_id: ID of listing to optimize
            use_ab_testing: Whether to use A/B testing
            
        Returns:
            Updated listing or None if not found
        """
        logger.info(f"Optimizing title for listing {listing_id}")
        
        try:
            # Get existing listing
            listing = await self.db.get_listing(listing_id)
            if not listing:
                logger.warning(f"Listing {listing_id} not found")
                return None
            
            # Get product
            product = await self.db.get_product(listing.product_id)
            if not product:
                logger.warning(f"Product {listing.product_id} not found")
                return None
            
            # Check if we have performance data
            performance = await self.performance_tracker.get_listing_performance(listing_id)
            
            if use_ab_testing and performance:
                # If we have active A/B tests, check if they're complete
                ab_tests = await self.performance_tracker.get_active_ab_tests(listing_id)
                
                if ab_tests:
                    # Check if any tests are complete
                    for test in ab_tests:
                        if self.performance_tracker.is_test_complete(test):
                            # Apply the winning variation
                            winner_id = test.winner_id
                            if winner_id:
                                winner = next((v for v in listing.title_variations if v.id == winner_id), None)
                                if winner:
                                    listing.title = winner.title
                                    logger.info(f"Applied winning title variation to listing {listing_id}")
                
                # Generate new variations for testing
                variations = await self.title_generator.generate_variations(
                    product, 
                    listing.marketplace, 
                    3,  # Generate 3 new variations
                    existing_title=listing.title
                )
                
                # Create new title variations
                new_variations = []
                for variation in variations:
                    new_variations.append(TitleVariation(
                        id=str(uuid.uuid4()),
                        product_id=product.id,
                        title=variation,
                        marketplace=listing.marketplace
                    ))
                
                # Add new variations to listing
                listing.title_variations.extend(new_variations)
                
                # Start new A/B test
                await self.performance_tracker.start_ab_test(
                    listing_id,
                    new_variations[0].id,
                    new_variations[1].id
                )
                
                logger.info(f"Started new A/B test for listing {listing_id}")
            else:
                # Generate a new optimized title
                new_title = await self.title_generator.generate_title(
                    product, 
                    listing.marketplace,
                    use_performance_data=bool(performance)
                )
                
                # Update listing title
                listing.title = new_title
                logger.info(f"Generated new optimized title for listing {listing_id}")
            
            # Save updated listing
            listing.updated_at = datetime.now()
            await self.db.save_listing(listing)
            
            return listing
        
        except Exception as e:
            logger.error(f"Error optimizing title for listing {listing_id}: {e}")
            raise
    
    async def optimize_description(
        self,
        listing_id: str
    ) -> Optional[Listing]:
        """
        Optimize the description of an existing listing.
        
        Args:
            listing_id: ID of listing to optimize
            
        Returns:
            Updated listing or None if not found
        """
        logger.info(f"Optimizing description for listing {listing_id}")
        
        try:
            # Get existing listing
            listing = await self.db.get_listing(listing_id)
            if not listing:
                logger.warning(f"Listing {listing_id} not found")
                return None
            
            # Get product
            product = await self.db.get_product(listing.product_id)
            if not product:
                logger.warning(f"Product {listing.product_id} not found")
                return None
            
            # Check if we have performance data
            performance = await self.performance_tracker.get_listing_performance(listing_id)
            
            # Generate a new optimized description
            new_description = await self.description_generator.generate_description(
                product, 
                listing.marketplace,
                use_performance_data=bool(performance)
            )
            
            # Update listing description
            listing.description = new_description
            
            # Re-extract and optimize keywords
            keywords = await self.keyword_optimizer.extract_keywords(
                product, 
                listing.title, 
                new_description, 
                listing.marketplace
            )
            listing.keywords = keywords
            
            # Save updated listing
            listing.updated_at = datetime.now()
            await self.db.save_listing(listing)
            
            logger.info(f"Optimized description for listing {listing_id}")
            return listing
        
        except Exception as e:
            logger.error(f"Error optimizing description for listing {listing_id}: {e}")
            raise
    
    async def update_pricing(
        self,
        listing_id: str,
        dynamic_adjustment: bool = False
    ) -> Optional[Listing]:
        """
        Update the pricing of an existing listing.
        
        Args:
            listing_id: ID of listing to update
            dynamic_adjustment: Whether to apply dynamic pricing adjustment
            
        Returns:
            Updated listing or None if not found
        """
        logger.info(f"Updating pricing for listing {listing_id}")
        
        try:
            # Get existing listing
            listing = await self.db.get_listing(listing_id)
            if not listing:
                logger.warning(f"Listing {listing_id} not found")
                return None
            
            # Get product
            product = await self.db.get_product(listing.product_id)
            if not product:
                logger.warning(f"Product {listing.product_id} not found")
                return None
            
            # Get new pricing recommendation
            pricing = await self.pricing_engine.get_pricing_recommendation(
                product, 
                listing.marketplace
            )
            
            # Apply dynamic adjustment if requested
            if dynamic_adjustment:
                # Get performance data
                performance = await self.performance_tracker.get_listing_performance(listing_id)
                
                if performance:
                    # Apply dynamic pricing based on performance
                    adjusted_price = await self.pricing_engine.get_dynamic_price_adjustment(
                        pricing.recommended_price,
                        performance
                    )
                    
                    # Update price
                    listing.price = adjusted_price
                    logger.info(f"Applied dynamic price adjustment for listing {listing_id}")
                else:
                    # Use recommended price
                    listing.price = pricing.recommended_price
            else:
                # Use recommended price
                listing.price = pricing.recommended_price
            
            # Update pricing recommendation
            listing.pricing_recommendation = pricing
            
            # Save updated listing
            listing.updated_at = datetime.now()
            await self.db.save_listing(listing)
            
            logger.info(f"Updated pricing for listing {listing_id}")
            return listing
        
        except Exception as e:
            logger.error(f"Error updating pricing for listing {listing_id}: {e}")
            raise
    
    async def process_images(
        self,
        listing_id: str
    ) -> Optional[Listing]:
        """
        Process images for an existing listing.
        
        Args:
            listing_id: ID of listing to process
            
        Returns:
            Updated listing or None if not found
        """
        logger.info(f"Processing images for listing {listing_id}")
        
        try:
            # Get existing listing
            listing = await self.db.get_listing(listing_id)
            if not listing:
                logger.warning(f"Listing {listing_id} not found")
                return None
            
            # Get product
            product = await self.db.get_product(listing.product_id)
            if not product:
                logger.warning(f"Product {listing.product_id} not found")
                return None
            
            # Process images
            image_metadata = []
            for image_url in product.image_urls:
                metadata = await self.image_processor.process_image(
                    product, 
                    image_url, 
                    is_primary=(image_url == product.image_urls[0])
                )
                image_metadata.append(metadata)
            
            # Update listing
            listing.image_metadata = image_metadata
            
            # Save updated listing
            listing.updated_at = datetime.now()
            await self.db.save_listing(listing)
            
            logger.info(f"Processed images for listing {listing_id}")
            return listing
        
        except Exception as e:
            logger.error(f"Error processing images for listing {listing_id}: {e}")
            raise
    
    async def track_performance(
        self,
        listing_id: str,
        metrics: Dict[str, Any]
    ) -> Optional[ListingPerformance]:
        """
        Track performance metrics for a listing.
        
        Args:
            listing_id: ID of listing to track
            metrics: Performance metrics
            
        Returns:
            Updated performance data or None if listing not found
        """
        logger.info(f"Tracking performance for listing {listing_id}")
        
        try:
            # Get existing listing
            listing = await self.db.get_listing(listing_id)
            if not listing:
                logger.warning(f"Listing {listing_id} not found")
                return None
            
            # Update performance metrics
            performance = await self.performance_tracker.update_metrics(listing_id, metrics)
            
            # Check if we need to optimize the listing based on performance
            if performance and self.performance_tracker.should_optimize(performance):
                logger.info(f"Performance indicates optimization needed for listing {listing_id}")
                
                # Schedule optimization tasks
                # In a real implementation, this would use a task queue
                # For simplicity, we'll just log the recommendation
                logger.info(f"Recommended optimizations for listing {listing_id}:")
                
                if performance.ctr < 0.02:  # Low CTR
                    logger.info(f"- Title optimization (low CTR: {performance.ctr:.4f})")
                
                if performance.conversion_rate < 0.01:  # Low conversion rate
                    logger.info(f"- Description optimization (low conversion: {performance.conversion_rate:.4f})")
                
                if performance.impressions < 100:  # Low impressions
                    logger.info(f"- Keyword optimization (low impressions: {performance.impressions})")
            
            return performance
        
        except Exception as e:
            logger.error(f"Error tracking performance for listing {listing_id}: {e}")
            raise
    
    async def get_listing(self, listing_id: str) -> Optional[Listing]:
        """
        Get a listing by ID.
        
        Args:
            listing_id: ID of listing to get
            
        Returns:
            Listing or None if not found
        """
        return await self.db.get_listing(listing_id)
    
    async def get_listings(
        self,
        product_id: Optional[str] = None,
        marketplace: Optional[Marketplace] = None,
        status: Optional[ListingStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Listing]:
        """
        Get listings with optional filtering.
        
        Args:
            product_id: Filter by product ID
            marketplace: Filter by marketplace
            status: Filter by status
            limit: Maximum number of listings to return
            offset: Offset for pagination
            
        Returns:
            List of listings
        """
        return await self.db.get_listings(
            product_id=product_id,
            marketplace=marketplace,
            status=status,
            limit=limit,
            offset=offset
        )
    
    async def get_performance(
        self,
        listing_id: str
    ) -> Optional[ListingPerformance]:
        """
        Get performance data for a listing.
        
        Args:
            listing_id: ID of listing to get performance for
            
        Returns:
            Performance data or None if not found
        """
        return await self.performance_tracker.get_listing_performance(listing_id)