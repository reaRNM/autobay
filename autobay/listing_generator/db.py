"""
Database module for the Listing Generator.

This module provides functionality to store and retrieve
listing data from a database.
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import sqlite3
import aiosqlite

from listing_generator.models import (
    Product, Listing, ListingPerformance, TitleVariation,
    PricingRecommendation, ImageMetadata, Marketplace,
    ListingStatus, ABTestResult
)


logger = logging.getLogger(__name__)


class ListingDatabase:
    """
    Database for listing data.
    
    This class provides functionality to store and retrieve
    listing data from a database.
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize the listing database.
        
        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        
        # Initialize database
        self._initialize_db()
        
        logger.info(f"ListingDatabase initialized with {connection_string}")
    
    async def _initialize_db(self) -> None:
        """Initialize database tables if they don't exist."""
        try:
            # Connect to database
            if self.connection_string.startswith("sqlite:///"):
                db_path = self.connection_string[10:]
                async with aiosqlite.connect(db_path) as conn:
                    # Create products table
                    await conn.execute('''
                    CREATE TABLE IF NOT EXISTS products (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        brand TEXT,
                        model TEXT,
                        category TEXT,
                        subcategory TEXT,
                        upc TEXT,
                        ean TEXT,
                        isbn TEXT,
                        asin TEXT,
                        mpn TEXT,
                        features TEXT,
                        specifications TEXT,
                        description TEXT,
                        condition TEXT,
                        weight_oz REAL,
                        dimensions TEXT,
                        image_urls TEXT,
                        msrp REAL,
                        cost REAL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    ''')
                    
                    # Create listings table
                    await conn.execute('''
                    CREATE TABLE IF NOT EXISTS listings (
                        id TEXT PRIMARY KEY,
                        product_id TEXT NOT NULL,
                        marketplace TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL,
                        price REAL NOT NULL,
                        quantity INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        marketplace_id TEXT,
                        marketplace_url TEXT,
                        title_variations TEXT,
                        image_metadata TEXT,
                        pricing_recommendation TEXT,
                        keywords TEXT,
                        category_id TEXT,
                        shipping_options TEXT,
                        return_policy TEXT,
                        item_specifics TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (product_id) REFERENCES products (id)
                    )
                    ''')
                    
                    # Create listing_performance table
                    await conn.execute('''
                    CREATE TABLE IF NOT EXISTS listing_performance (
                        id TEXT PRIMARY KEY,
                        listing_id TEXT NOT NULL,
                        impressions INTEGER NOT NULL,
                        clicks INTEGER NOT NULL,
                        add_to_carts INTEGER NOT NULL,
                        purchases INTEGER NOT NULL,
                        revenue REAL NOT NULL,
                        search_rank INTEGER,
                        search_terms TEXT,
                        start_date TEXT NOT NULL,
                        end_date TEXT,
                        FOREIGN KEY (listing_id) REFERENCES listings (id)
                    )
                    ''')
                    
                    # Create ab_tests table
                    await conn.execute('''
                    CREATE TABLE IF NOT EXISTS ab_tests (
                        id TEXT PRIMARY KEY,
                        listing_id TEXT NOT NULL,
                        variation_a_id TEXT NOT NULL,
                        variation_b_id TEXT NOT NULL,
                        winner_id TEXT,
                        confidence_level REAL NOT NULL,
                        metrics TEXT,
                        start_date TEXT NOT NULL,
                        end_date TEXT,
                        FOREIGN KEY (listing_id) REFERENCES listings (id)
                    )
                    ''')
                    
                    await conn.commit()
            
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    async def save_product(self, product: Product) -> bool:
        """
        Save a product to the database.
        
        Args:
            product: Product to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update timestamp
            product.updated_at = datetime.now()
            
            # Connect to database
            if self.connection_string.startswith("sqlite:///"):
                db_path = self.connection_string[10:]
                async with aiosqlite.connect(db_path) as conn:
                    # Convert complex types to JSON
                    features = json.dumps(product.features)
                    specifications = json.dumps(product.specifications)
                    dimensions = json.dumps(product.dimensions) if product.dimensions else None
                    image_urls = json.dumps(product.image_urls)
                    
                    # Check if product exists
                    cursor = await conn.execute(
                        "SELECT id FROM products WHERE id = ?",
                        (product.id,)
                    )
                    existing = await cursor.fetchone()
                    
                    if existing:
                        # Update existing product
                        await conn.execute('''
                        UPDATE products SET
                            name = ?,
                            brand = ?,
                            model = ?,
                            category = ?,
                            subcategory = ?,
                            upc = ?,
                            ean = ?,
                            isbn = ?,
                            asin = ?,
                            mpn = ?,
                            features = ?,
                            specifications = ?,
                            description = ?,
                            condition = ?,
                            weight_oz = ?,
                            dimensions = ?,
                            image_urls = ?,
                            msrp = ?,
                            cost = ?,
                            updated_at = ?
                        WHERE id = ?
                        ''', (
                            product.name,
                            product.brand,
                            product.model,
                            product.category,
                            product.subcategory,
                            product.upc,
                            product.ean,
                            product.isbn,
                            product.asin,
                            product.mpn,
                            features,
                            specifications,
                            product.description,
                            product.condition,
                            product.weight_oz,
                            dimensions,
                            image_urls,
                            product.msrp,
                            product.cost,
                            product.updated_at.isoformat(),
                            product.id
                        ))
                    else:
                        # Insert new product
                        await conn.execute('''
                        INSERT INTO products (
                            id, name, brand, model, category, subcategory,
                            upc, ean, isbn, asin, mpn, features, specifications,
                            description, condition, weight_oz, dimensions,
                            image_urls, msrp, cost, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            product.id,
                            product.name,
                            product.brand,
                            product.model,
                            product.category,
                            product.subcategory,
                            product.upc,
                            product.ean,
                            product.isbn,
                            product.asin,
                            product.mpn,
                            features,
                            specifications,
                            product.description,
                            product.condition,
                            product.weight_oz,
                            dimensions,
                            image_urls,
                            product.msrp,
                            product.cost,
                            product.created_at.isoformat(),
                            product.updated_at.isoformat()
                        ))
                    
                    await conn.commit()
            
            logger.info(f"Saved product {product.id}")
            return True
        except Exception as e:
            logger.error(f"Error saving product: {e}")
            return False
    
    async def get_product(self, product_id: str) -> Optional[Product]:
        """
        Get a product from the database.
        
        Args:
            product_id: ID of the product to get
            
        Returns:
            Product or None if not found
        """
        try:
            # Connect to database
            if self.connection_string.startswith("sqlite:///"):
                db_path = self.connection_string[10:]
                async with aiosqlite.connect(db_path) as conn:
                    conn.row_factory = aiosqlite.Row
                    
                    # Get product
                    cursor = await conn.execute(
                        "SELECT * FROM products WHERE id = ?",
                        (product_id,)
                    )
                    row = await cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    # Convert row to dict
                    row_dict = dict(row)
                    
                    # Parse complex types
                    features = json.loads(row_dict["features"]) if row_dict["features"] else []
                    specifications = json.loads(row_dict["specifications"]) if row_dict["specifications"] else {}
                    dimensions = json.loads(row_dict["dimensions"]) if row_dict["dimensions"] else None
                    image_urls = json.loads(row_dict["image_urls"]) if row_dict["image_urls"] else []
                    
                    # Create product
                    product = Product(
                        id=row_dict["id"],
                        name=row_dict["name"],
                        brand=row_dict["brand"],
                        model=row_dict["model"],
                        category=row_dict["category"],
                        subcategory=row_dict["subcategory"],
                        upc=row_dict["upc"],
                        ean=row_dict["ean"],
                        isbn=row_dict["isbn"],
                        asin=row_dict["asin"],
                        mpn=row_dict["mpn"],
                        features=features,
                        specifications=specifications,
                        description=row_dict["description"],
                        condition=row_dict["condition"],
                        weight_oz=row_dict["weight_oz"],
                        dimensions=dimensions,
                        image_urls=image_urls,
                        msrp=row_dict["msrp"],
                        cost=row_dict["cost"],
                        created_at=datetime.fromisoformat(row_dict["created_at"]),
                        updated_at=datetime.fromisoformat(row_dict["updated_at"])
                    )
                    
                    return product
        except Exception as e:
            logger.error(f"Error getting product: {e}")
            return None
    
    async def save_listing(self, listing: Listing) -> bool:
        """
        Save a listing to the database.
        
        Args:
            listing: Listing to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update timestamp
            listing.updated_at = datetime.now()
            
            # Connect to database
            if self.connection_string.startswith("sqlite:///"):
                db_path = self.connection_string[10:]
                async with aiosqlite.connect(db_path) as conn:
                    # Convert complex types to JSON
                    title_variations = json.dumps([v.to_dict() for v in listing.title_variations]) if listing.title_variations else None
                    image_metadata = json.dumps([m.to_dict() for m in listing.image_metadata]) if listing.image_metadata else None
                    pricing_recommendation = json.dumps(listing.pricing_recommendation.to_dict()) if listing.pricing_recommendation else None
                    keywords = json.dumps(listing.keywords)
                    shipping_options = json.dumps(listing.shipping_options)
                    return_policy = json.dumps(listing.return_policy)
                    item_specifics = json.dumps(listing.item_specifics)
                    
                    # Check if listing exists
                    cursor = await conn.execute(
                        "SELECT id FROM listings WHERE id = ?",
                        (listing.id,)
                    )
                    existing = await cursor.fetchone()
                    
                    if existing:
                        # Update existing listing
                        await conn.execute('''
                        UPDATE listings SET
                            product_id = ?,
                            marketplace = ?,
                            title = ?,
                            description = ?,
                            price = ?,
                            quantity = ?,
                            status = ?,
                            marketplace_id = ?,
                            marketplace_url = ?,
                            title_variations = ?,
                            image_metadata = ?,
                            pricing_recommendation = ?,
                            keywords = ?,
                            category_id = ?,
                            shipping_options = ?,
                            return_policy = ?,
                            item_specifics = ?,
                            updated_at = ?
                        WHERE id = ?
                        ''', (
                            listing.product_id,
                            listing.marketplace.value,
                            listing.title,
                            listing.description,
                            listing.price,
                            listing.quantity,
                            listing.status.value,
                            listing.marketplace_id,
                            listing.marketplace_url,
                            title_variations,
                            image_metadata,
                            pricing_recommendation,
                            keywords,
                            listing.category_id,
                            shipping_options,
                            return_policy,
                            item_specifics,
                            listing.updated_at.isoformat(),
                            listing.id
                        ))
                    else:
                        # Insert new listing
                        await conn.execute('''
                        INSERT INTO listings (
                            id, product_id, marketplace, title, description,
                            price, quantity, status, marketplace_id, marketplace_url,
                            title_variations, image_metadata, pricing_recommendation,
                            keywords, category_id, shipping_options, return_policy,
                            item_specifics, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            listing.id,
                            listing.product_id,
                            listing.marketplace.value,
                            listing.title,
                            listing.description,
                            listing.price,
                            listing.quantity,
                            listing.status.value,
                            listing.marketplace_id,
                            listing.marketplace_url,
                            title_variations,
                            image_metadata,
                            pricing_recommendation,
                            keywords,
                            listing.category_id,
                            shipping_options,
                            return_policy,
                            item_specifics,
                            listing.created_at.isoformat(),
                            listing.updated_at.isoformat()
                        ))
                    
                    await conn.commit()
            
            logger.info(f"Saved listing {listing.id}")
            return True
        except Exception as e:
            logger.error(f"Error saving listing: {e}")
            return False
    
    async def get_listing(self, listing_id: str) -> Optional[Listing]:
        """
        Get a listing from the database.
        
        Args:
            listing_id: ID of the listing to get
            
        Returns:
            Listing or None if not found
        """
        try:
            # Connect to database
            if self.connection_string.startswith("sqlite:///"):
                db_path = self.connection_string[10:]
                async with aiosqlite.connect(db_path) as conn:
                    conn.row_factory = aiosqlite.Row
                    
                    # Get listing
                    cursor = await conn.execute(
                        "SELECT * FROM listings WHERE id = ?",
                        (listing_id,)
                    )
                    row = await cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    # Convert row to dict
                    row_dict = dict(row)
                    
                    # Parse complex types
                    title_variations_json = row_dict["title_variations"]
                    title_variations = []
                    if title_variations_json:
                        title_variations_data = json.loads(title_variations_json)
                        for data in title_variations_data:
                            variation = TitleVariation(
                                id=data["id"],
                                product_id=data["product_id"],
                                title=data["title"],
                                marketplace=Marketplace(data["marketplace"]),
                                score=data.get("score", 0.0),
                                impressions=data.get("impressions", 0),
                                clicks=data.get("clicks", 0),
                                is_active=data.get("is_active", True),
                                created_at=datetime.fromisoformat(data["created_at"])
                            )
                            title_variations.append(variation)
                    
                    image_metadata_json = row_dict["image_metadata"]
                    image_metadata = []
                    if image_metadata_json:
                        image_metadata_data = json.loads(image_metadata_json)
                        for data in image_metadata_data:
                            metadata = ImageMetadata(
                                id=data["id"],
                                product_id=data["product_id"],
                                image_url=data["image_url"],
                                alt_text=data["alt_text"],
                                caption=data.get("caption"),
                                tags=data.get("tags", []),
                                enhancement_suggestions=data.get("enhancement_suggestions", {}),
                                is_primary=data.get("is_primary", False),
                                created_at=datetime.fromisoformat(data["created_at"])
                            )
                            image_metadata.append(metadata)
                    
                    pricing_recommendation_json = row_dict["pricing_recommendation"]
                    pricing_recommendation = None
                    if pricing_recommendation_json:
                        data = json.loads(pricing_recommendation_json)
                        pricing_recommendation = PricingRecommendation(
                            id=data["id"],
                            product_id=data["product_id"],
                            marketplace=Marketplace(data["marketplace"]),
                            min_price=data["min_price"],
                            max_price=data["max_price"],
                            recommended_price=data["recommended_price"],
                            competitor_prices=data.get("competitor_prices", []),
                            historical_prices=data.get("historical_prices", []),
                            confidence_score=data.get("confidence_score", 0.0),
                            factors=data.get("factors", {}),
                            created_at=datetime.fromisoformat(data["created_at"])
                        )
                    
                    keywords = json.loads(row_dict["keywords"]) if row_dict["keywords"] else []
                    shipping_options = json.loads(row_dict["shipping_options"]) if row_dict["shipping_options"] else {}
                    return_policy = json.loads(row_dict["return_policy"]) if row_dict["return_policy"] else {}
                    item_specifics = json.loads(row_dict["item_specifics"]) if row_dict["item_specifics"] else {}
                    
                    # Create listing
                    listing = Listing(
                        id=row_dict["id"],
                        product_id=row_dict["product_id"],
                        marketplace=Marketplace(row_dict["marketplace"]),
                        title=row_dict["title"],
                        description=row_dict["description"],
                        price=row_dict["price"],
                        quantity=row_dict["quantity"],
                        status=ListingStatus(row_dict["status"]),
                        marketplace_id=row_dict["marketplace_id"],
                        marketplace_url=row_dict["marketplace_url"],
                        title_variations=title_variations,
                        image_metadata=image_metadata,
                        pricing_recommendation=pricing_recommendation,
                        keywords=keywords,
                        category_id=row_dict["category_id"],
                        shipping_options=shipping_options,
                        return_policy=return_policy,
                        item_specifics=item_specifics,
                        created_at=datetime.fromisoformat(row_dict["created_at"]),
                        updated_at=datetime.fromisoformat(row_dict["updated_at"])
                    )
                    
                    return listing
        except Exception as e:
            logger.error(f"Error getting listing: {e}")
            return None
    
    async def get_listings(
        self,
        product_id: Optional[str] = None,
        marketplace: Optional[Marketplace] = None,
        status: Optional[ListingStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Listing]:
        """
        Get listings from the database.
        
        Args:
            product_id: Filter by product ID
            marketplace: Filter by marketplace
            status: Filter by status
            limit: Maximum number of listings to return
            offset: Offset for pagination
            
        Returns:
            List of listings
        """
        try:
            # Connect to database
            if self.connection_string.startswith("sqlite:///"):
                db_path = self.connection_string[10:]
                async with aiosqlite.connect(db_path) as conn:
                    conn.row_factory = aiosqlite.Row
                    
                    # Build query
                    query = "SELECT * FROM listings"
                    params = []
                    
                    where_clauses = []
                    if product_id:
                        where_clauses.append("product_id = ?")
                        params.append(product_id)
                    
                    if marketplace:
                        where_clauses.append("marketplace = ?")
                        params.append(marketplace.value)
                    
                    if status:
                        where_clauses.append("status = ?")
                        params.append(status.value)
                    
                    if where_clauses:
                        query += " WHERE " + " AND ".join(where_clauses)
                    
                    query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
                    params.append(limit)
                    params.append(offset)
                    
                    # Get listings
                    cursor = await conn.execute(query, params)
                    rows = await cursor.fetchall()
                    
                    # Convert rows to listings
                    listings = []
                    for row in rows:
                        # Convert row to dict
                        row_dict = dict(row)
                        
                        # Parse complex types
                        title_variations_json = row_dict["title_variations"]
                        title_variations = []
                        if title_variations_json:
                            title_variations_data = json.loads(title_variations_json)
                            for data in title_variations_data:
                                variation = TitleVariation(
                                    id=data["id"],
                                    product_id=data["product_id"],
                                    title=data["title"],
                                    marketplace=Marketplace(data["marketplace"]),
                                    score=data.get("score", 0.0),
                                    impressions=data.get("impressions", 0),
                                    clicks=data.get("clicks", 0),
                                    is_active=data.get("is_active", True),
                                    created_at=datetime.fromisoformat(data["created_at"])
                                )
                                title_variations.append(variation)
                        
                        image_metadata_json = row_dict["image_metadata"]
                        image_metadata = []
                        if image_metadata_json:
                            image_metadata_data = json.loads(image_metadata_json)
                            for data in image_metadata_data:
                                metadata = ImageMetadata(
                                    id=data["id"],
                                    product_id=data["product_id"],
                                    image_url=data["image_url"],
                                    alt_text=data["alt_text"],
                                    caption=data.get("caption"),
                                    tags=data.get("tags", []),
                                    enhancement_suggestions=data.get("enhancement_suggestions", {}),
                                    is_primary=data.get("is_primary", False),
                                    created_at=datetime.fromisoformat(data["created_at"])
                                )
                                image_metadata.append(metadata)
                        
                        pricing_recommendation_json = row_dict["pricing_recommendation"]
                        pricing_recommendation = None
                        if pricing_recommendation_json:
                            data = json.loads(pricing_recommendation_json)
                            pricing_recommendation = PricingRecommendation(
                                id=data["id"],
                                product_id=data["product_id"],
                                marketplace=Marketplace(data["marketplace"]),
                                min_price=data["min_price"],
                                max_price=data["max_price"],
                                recommended_price=data["recommended_price"],
                                competitor_prices=data.get("competitor_prices", []),
                                historical_prices=data.get("historical_prices", []),
                                confidence_score=data.get("confidence_score", 0.0),
                                factors=data.get("factors", {}),
                                created_at=datetime.fromisoformat(data["created_at"])
                            )
                        
                        keywords = json.loads(row_dict["keywords"]) if row_dict["keywords"] else []
                        shipping_options = json.loads(row_dict["shipping_options"]) if row_dict["shipping_options"] else {}
                        return_policy = json.loads(row_dict["return_policy"]) if row_dict["return_policy"] else {}
                        item_specifics = json.loads(row_dict["item_specifics"]) if row_dict["item_specifics"] else {}
                        
                        # Create listing
                        listing = Listing(
                            id=row_dict["id"],
                            product_id=row_dict["product_id"],
                            marketplace=Marketplace(row_dict["marketplace"]),
                            title=row_dict["title"],
                            description=row_dict["description"],
                            price=row_dict["price"],
                            quantity=row_dict["quantity"],
                            status=ListingStatus(row_dict["status"]),
                            marketplace_id=row_dict["marketplace_id"],
                            marketplace_url=row_dict["marketplace_url"],
                            title_variations=title_variations,
                            image_metadata=image_metadata,
                            pricing_recommendation=pricing_recommendation,
                            keywords=keywords,
                            category_id=row_dict["category_id"],
                            shipping_options=shipping_options,
                            return_policy=return_policy,
                            item_specifics=item_specifics,
                            created_at=datetime.fromisoformat(row_dict["created_at"]),
                            updated_at=datetime.fromisoformat(row_dict["updated_at"])
                        )
                        
                        listings.append(listing)
                    
                    return listings
        except Exception as e:
            logger.error(f"Error getting listings: {e}")
            return []
    
    async def save_listing_performance(self, performance: ListingPerformance) -> bool:
        """
        Save listing performance data to the database.
        
        Args:
            performance: Performance data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Connect to database
            if self.connection_string.startswith("sqlite:///"):
                db_path = self.connection_string[10:]
                async with aiosqlite.connect(db_path) as conn:
                    # Convert complex types to JSON
                    search_terms = json.dumps(performance.search_terms)
                    
                    # Check if performance data exists
                    cursor = await conn.execute(
                        "SELECT id FROM listing_performance WHERE id = ?",
                        (performance.id,)
                    )
                    existing = await cursor.fetchone()
                    
                    if existing:
                        # Update existing performance data
                        await conn.execute('''
                        UPDATE listing_performance SET
                            listing_id = ?,
                            impressions = ?,
                            clicks = ?,
                            add_to_carts = ?,
                            purchases = ?,
                            revenue = ?,
                            search_rank = ?,
                            search_terms = ?,
                            start_date = ?,
                            end_date = ?
                        WHERE id = ?
                        ''', (
                            performance.listing_id,
                            performance.impressions,
                            performance.clicks,
                            performance.add_to_carts,
                            performance.purchases,
                            performance.revenue,
                            performance.search_rank,
                            search_terms,
                            performance.start_date.isoformat(),
                            performance.end_date.isoformat() if performance.end_date else None,
                            performance.id
                        ))
                    else:
                        # Insert new performance data
                        await conn.execute('''
                        INSERT INTO listing_performance (
                            id, listing_id, impressions, clicks, add_to_carts,
                            purchases, revenue, search_rank, search_terms,
                            start_date, end_date
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            performance.id,
                            performance.listing_id,
                            performance.impressions,
                            performance.clicks,
                            performance.add_to_carts,
                            performance.purchases,
                            performance.revenue,
                            performance.search_rank,
                            search_terms,
                            performance.start_date.isoformat(),
                            performance.end_date.isoformat() if performance.end_date else None
                        ))
                    
                    await conn.commit()
            
            logger.info(f"Saved listing performance {performance.id}")
            return True
        except Exception as e:
            logger.error(f"Error saving listing performance: {e}")
            return False
    
    async def get_listing_performance(self, listing_id: str) -> Optional[ListingPerformance]:
        """
        Get listing performance data from the database.
        
        Args:
            listing_id: ID of the listing
            
        Returns:
            Performance data or None if not found
        """
        try:
            # Connect to database
            if self.connection_string.startswith("sqlite:///"):
                db_path = self.connection_string[10:]
                async with aiosqlite.connect(db_path) as conn:
                    conn.row_factory = aiosqlite.Row
                    
                    # Get performance data
                    cursor = await conn.execute(
                        "SELECT * FROM listing_performance WHERE listing_id = ?",
                        (listing_id,)
                    )
                    row = await cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    # Convert row to dict
                    row_dict = dict(row)
                    
                    # Parse complex types
                    search_terms = json.loads(row_dict["search_terms"]) if row_dict["search_terms"] else []
                    
                    # Create performance data
                    performance = ListingPerformance(
                        id=row_dict["id"],
                        listing_id=row_dict["listing_id"],
                        impressions=row_dict["impressions"],
                        clicks=row_dict["clicks"],
                        add_to_carts=row_dict["add_to_carts"],
                        purchases=row_dict["purchases"],
                        revenue=row_dict["revenue"],
                        search_rank=row_dict["search_rank"],
                        search_terms=search_terms,
                        start_date=datetime.fromisoformat(row_dict["start_date"]),
                        end_date=datetime.fromisoformat(row_dict["end_date"]) if row_dict["end_date"] else None
                    )
                    
                    return performance
        except Exception as e:
            logger.error(f"Error getting listing performance: {e}")
            return None
    
    async def save_ab_test(self, test: ABTestResult) -> bool:
        """
        Save A/B test result to the database.
        
        Args:
            test: A/B test result to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Connect to database
            if self.connection_string.startswith("sqlite:///"):
                db_path = self.connection_string[10:]
                async with aiosqlite.connect(db_path) as conn:
                    # Convert complex types to JSON
                    metrics = json.dumps(test.metrics) if test.metrics else None
                    
                    # Check if test exists
                    cursor = await conn.execute(
                        "SELECT id FROM ab_tests WHERE id = ?