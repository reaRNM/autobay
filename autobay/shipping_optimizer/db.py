"""
Database module for shipping optimization.

This module provides functionality to store and retrieve
shipping data from a database.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import sqlite3

from shipping_optimizer.models import (
    ShippingHistory, CarrierPerformance
)


logger = logging.getLogger(__name__)


class ShippingDatabase:
    """
    Database for shipping data.
    
    This class provides functionality to store and retrieve
    shipping data from a database.
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize the shipping database.
        
        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        
        # Initialize database
        self._initialize_db()
        
        logger.info(f"ShippingDatabase initialized with {connection_string}")
    
    def _initialize_db(self) -> None:
        """Initialize database tables if they don't exist."""
        try:
            # Connect to database
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create shipping_history table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS shipping_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                package_data TEXT NOT NULL,
                origin_address TEXT NOT NULL,
                destination_address TEXT NOT NULL,
                selected_rate TEXT NOT NULL,
                actual_cost REAL NOT NULL,
                ship_date TEXT NOT NULL,
                delivery_date TEXT,
                tracking_number TEXT,
                delivery_status TEXT,
                delivery_issues TEXT,
                customer_rating INTEGER,
                created_at TEXT NOT NULL
            )
            ''')
            
            # Create shipping_rates table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS shipping_rates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                carrier TEXT NOT NULL,
                service TEXT NOT NULL,
                weight_lb REAL NOT NULL,
                length_in REAL NOT NULL,
                width_in REAL NOT NULL,
                height_in REAL NOT NULL,
                volume_cubic_in REAL NOT NULL,
                distance_miles REAL NOT NULL,
                zone INTEGER NOT NULL,
                origin_postal_code TEXT NOT NULL,
                destination_postal_code TEXT NOT NULL,
                origin_state TEXT NOT NULL,
                destination_state TEXT NOT NULL,
                is_residential BOOLEAN NOT NULL,
                rate REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            ''')
            
            # Create carrier_performance table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS carrier_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                carrier TEXT NOT NULL,
                service TEXT NOT NULL,
                avg_delivery_days REAL NOT NULL,
                on_time_percentage REAL NOT NULL,
                damage_percentage REAL NOT NULL,
                loss_percentage REAL NOT NULL,
                cost_accuracy REAL NOT NULL,
                customer_satisfaction REAL NOT NULL,
                total_shipments INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _get_connection(self):
        """
        Get database connection.
        
        Returns:
            Database connection
        """
        if self.connection_string.startswith("sqlite:///"):
            db_path = self.connection_string[10:]
            return sqlite3.connect(db_path)
        else:
            # For other database types, implement appropriate connection logic
            raise NotImplementedError(f"Database type not supported: {self.connection_string}")
    
    def save_shipping_history(self, history: ShippingHistory) -> bool:
        """
        Save shipping history to database.
        
        Args:
            history: Shipping history to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to JSON
            import json
            package_data = json.dumps(history.package.to_dict())
            origin_address = json.dumps(history.origin_address.to_dict())
            destination_address = json.dumps(history.destination_address.to_dict())
            selected_rate = json.dumps(history.selected_rate.to_dict())
            delivery_issues = json.dumps(history.delivery_issues)
            
            # Connect to database
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Insert data
            cursor.execute('''
            INSERT INTO shipping_history (
                user_id, package_data, origin_address, destination_address,
                selected_rate, actual_cost, ship_date, delivery_date,
                tracking_number, delivery_status, delivery_issues,
                customer_rating, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                history.user_id, package_data, origin_address, destination_address,
                selected_rate, history.actual_cost, history.ship_date.isoformat(),
                history.delivery_date.isoformat() if history.delivery_date else None,
                history.tracking_number, history.delivery_status, delivery_issues,
                history.customer_rating, datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved shipping history for user {history.user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving shipping history: {e}")
            return False
    
    def get_shipping_history(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ShippingHistory]:
        """
        Get shipping history from database.
        
        Args:
            user_id: Filter by user ID
            start_date: Start date for filtering
            end_date: End date for filtering
            limit: Maximum number of records to return
            
        Returns:
            List of shipping history records
        """
        try:
            # Build query
            query = "SELECT * FROM shipping_history"
            params = []
            
            where_clauses = []
            if user_id:
                where_clauses.append("user_id = ?")
                params.append(user_id)
            
            if start_date:
                where_clauses.append("ship_date >= ?")
                params.append(start_date.isoformat())
            
            if end_date:
                where_clauses.append("ship_date <= ?")
                params.append(end_date.isoformat())
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += " ORDER BY ship_date DESC LIMIT ?"
            params.append(limit)
            
            # Connect to database
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to ShippingHistory objects
            import json
            from shipping_optimizer.models import Package, Address, ShippingRate
            
            history_list = []
            for row in rows:
                # Parse JSON data
                package_data = json.loads(row[2])
                origin_address_data = json.loads(row[3])
                destination_address_data = json.loads(row[4])
                selected_rate_data = json.loads(row[5])
                delivery_issues = json.loads(row[11])
                
                # Create objects
                package = Package(**package_data)
                origin_address = Address(**origin_address_data)
                destination_address = Address(**destination_address_data)
                selected_rate = ShippingRate(**selected_rate_data)
                
                # Create ShippingHistory
                history = ShippingHistory(
                    user_id=row[1],
                    package=package,
                    origin_address=origin_address,
                    destination_address=destination_address,
                    selected_rate=selected_rate,
                    actual_cost=row[6],
                    ship_date=datetime.fromisoformat(row[7]),
                    delivery_date=datetime.fromisoformat(row[8]) if row[8] else None,
                    tracking_number=row[9],
                    delivery_status=row[10],
                    delivery_issues=delivery_issues,
                    customer_rating=row[12]
                )
                
                history_list.append(history)
            
            conn.close()
            
            logger.info(f"Retrieved {len(history_list)} shipping history records")
            return history_list
        except Exception as e:
            logger.error(f"Error getting shipping history: {e}")
            return []
    
    def save_shipping_rate(
        self,
        carrier: str,
        service: str,
        weight_lb: float,
        length_in: float,
        width_in: float,
        height_in: float,
        distance_miles: float,
        zone: int,
        origin_postal_code: str,
        destination_postal_code: str,
        origin_state: str,
        destination_state: str,
        is_residential: bool,
        rate: float
    ) -> bool:
        """
        Save shipping rate to database.
        
        Args:
            carrier: Carrier name
            service: Service name
            weight_lb: Weight in pounds
            length_in: Length in inches
            width_in: Width in inches
            height_in: Height in inches
            distance_miles: Distance in miles
            zone: Shipping zone
            origin_postal_code: Origin postal code
            destination_postal_code: Destination postal code
            origin_state: Origin state
            destination_state: Destination state
            is_residential: Whether destination is residential
            rate: Shipping rate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate volume
            volume_cubic_in = length_in * width_in * height_in
            
            # Connect to database
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Insert data
            cursor.execute('''
            INSERT INTO shipping_rates (
                carrier, service, weight_lb, length_in, width_in, height_in,
                volume_cubic_in, distance_miles, zone, origin_postal_code,
                destination_postal_code, origin_state, destination_state,
                is_residential, rate, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                carrier, service, weight_lb, length_in, width_in, height_in,
                volume_cubic_in, distance_miles, zone, origin_postal_code,
                destination_postal_code, origin_state, destination_state,
                is_residential, rate, datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved shipping rate for {carrier} {service}")
            return True
        except Exception as e:
            logger.error(f"Error saving shipping rate: {e}")
            return False
    
    def get_shipping_rates_data(
        self,
        carrier: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Get shipping rates data from database.
        
        Args:
            carrier: Filter by carrier
            start_date: Start date for filtering
            end_date: End date for filtering
            limit: Maximum number of records to return
            
        Returns:
            DataFrame of shipping rates data
        """
        try:
            # Build query
            query = "SELECT * FROM shipping_rates"
            params = []
            
            where_clauses = []
            if carrier:
                where_clauses.append("carrier = ?")
                params.append(carrier)
            
            if start_date:
                where_clauses.append("created_at >= ?")
                params.append(start_date.isoformat())
            
            if end_date:
                where_clauses.append("created_at <= ?")
                params.append(end_date.isoformat())
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            # Connect to database
            conn = self._get_connection()
            
            # Execute query and load into DataFrame
            df = pd.read_sql_query(query, conn, params=params)
            
            conn.close()
            
            logger.info(f"Retrieved {len(df)} shipping rates records")
            return df
        except Exception as e:
            logger.error(f"Error getting shipping rates data: {e}")
            return pd.DataFrame()
    
    def get_carrier_performance(
        self,
        carrier: Optional[str] = None
    ) -> List[CarrierPerformance]:
        """
        Get carrier performance metrics from database.
        
        Args:
            carrier: Filter by carrier
            
        Returns:
            List of carrier performance metrics
        """
        try:
            # Build query
            query = "SELECT * FROM carrier_performance"
            params = []
            
            if carrier:
                query += " WHERE carrier = ?"
                params.append(carrier)
            
            # Connect to database
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to CarrierPerformance objects
            performance_list = []
            for row in rows:
                performance = CarrierPerformance(
                    carrier=row[1],
                    service=row[2],
                    avg_delivery_days=row[3],
                    on_time_percentage=row[4],
                    damage_percentage=row[5],
                    loss_percentage=row[6],
                    cost_accuracy=row[7],
                    customer_satisfaction=row[8],
                    total_shipments=row[9]
                )
                
                performance_list.append(performance)
            
            conn.close()
            
            logger.info(f"Retrieved {len(performance_list)} carrier performance records")
            return performance_list
        except Exception as e:
            logger.error(f"Error getting carrier performance: {e}")
            return []
    
    def update_carrier_performance(
        self,
        performance: CarrierPerformance
    ) -> bool:
        """
        Update carrier performance metrics in database.
        
        Args:
            performance: Carrier performance metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Connect to database
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if record exists
            cursor.execute(
                "SELECT id FROM carrier_performance WHERE carrier = ? AND service = ?",
                (performance.carrier, performance.service)
            )
            row = cursor.fetchone()
            
            if row:
                # Update existing record
                cursor.execute('''
                UPDATE carrier_performance SET
                    avg_delivery_days = ?,
                    on_time_percentage = ?,
                    damage_percentage = ?,
                    loss_percentage = ?,
                    cost_accuracy = ?,
                    customer_satisfaction = ?,
                    total_shipments = ?,
                    updated_at = ?
                WHERE carrier = ? AND service = ?
                ''', (
                    performance.avg_delivery_days,
                    performance.on_time_percentage,
                    performance.damage_percentage,
                    performance.loss_percentage,
                    performance.cost_accuracy,
                    performance.customer_satisfaction,
                    performance.total_shipments,
                    datetime.now().isoformat(),
                    performance.carrier,
                    performance.service
                ))
            else:
                # Insert new record
                cursor.execute('''
                INSERT INTO carrier_performance (
                    carrier, service, avg_delivery_days, on_time_percentage,
                    damage_percentage, loss_percentage, cost_accuracy,
                    customer_satisfaction, total_shipments, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    performance.carrier,
                    performance.service,
                    performance.avg_delivery_days,
                    performance.on_time_percentage,
                    performance.damage_percentage,
                    performance.loss_percentage,
                    performance.cost_accuracy,
                    performance.customer_satisfaction,
                    performance.total_shipments,
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated performance metrics for {performance.carrier} {performance.service}")
            return True
        except Exception as e:
            logger.error(f"Error updating carrier performance: {e}")
            return False