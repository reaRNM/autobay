#!/usr/bin/env python3
"""
Database initialization script for AutoBay.

This script creates all necessary database tables, indexes, and initial data
for the AutoBay system.
"""

import os
import sys
import logging
from datetime import datetime
import argparse
import uuid

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from dotenv import load_dotenv
    import sqlalchemy as sa
    from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON, Date
    from sqlalchemy.dialects.postgresql import UUID, JSONB
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/db_init.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Get database connection details from environment variables
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'autobay')
DB_USER = os.getenv('DB_USER', 'autobay_user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'your_password')

# Create SQLAlchemy base
Base = declarative_base()

# Define models
class User(Base):
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    settings = Column(JSONB)

class AuctionItem(Base):
    __tablename__ = 'auction_items'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform = Column(String(50), nullable=False)
    platform_id = Column(String(100), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    condition = Column(String(50))
    current_price = Column(Float)
    shipping_cost = Column(Float)
    end_time = Column(DateTime)
    url = Column(String(500))
    image_url = Column(String(500))
    seller_id = Column(String(100))
    seller_rating = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String(50), default='new')
    metadata = Column(JSONB)
    
    __table_args__ = (
        sa.UniqueConstraint('platform', 'platform_id', name='uix_platform_item'),
    )

class ProfitCalculation(Base):
    __tablename__ = 'profit_calculations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_id = Column(UUID(as_uuid=True), ForeignKey('auction_items.id'))
    estimated_buy_price = Column(Float, nullable=False)
    estimated_sell_price = Column(Float, nullable=False)
    estimated_fees = Column(Float, nullable=False)
    estimated_shipping = Column(Float, nullable=False)
    estimated_profit = Column(Float, nullable=False)
    estimated_roi = Column(Float, nullable=False)
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSONB)

class BidRecommendation(Base):
    __tablename__ = 'bid_recommendations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_id = Column(UUID(as_uuid=True), ForeignKey('auction_items.id'))
    recommended_bid = Column(Float, nullable=False)
    max_bid = Column(Float, nullable=False)
    confidence_score = Column(Float)
    profit_potential = Column(Float)
    roi_potential = Column(Float)
    risk_score = Column(Float)
    time_sensitivity = Column(Float)
    requires_review = Column(Boolean, default=False)
    review_reason = Column(String(255))
    bid_time = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSONB)

class ScheduledBid(Base):
    __tablename__ = 'scheduled_bids'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_id = Column(UUID(as_uuid=True), ForeignKey('auction_items.id'))
    bid_amount = Column(Float, nullable=False)
    max_bid = Column(Float, nullable=False)
    scheduled_time = Column(DateTime, nullable=False)
    status = Column(String(50), default='scheduled')
    is_urgent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class WorkflowExecution(Base):
    __tablename__ = 'workflow_executions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_name = Column(String(100), nullable=False)
    status = Column(String(50), default='PENDING')
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    metadata = Column(JSONB)

class TaskResult(Base):
    __tablename__ = 'task_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    execution_id = Column(UUID(as_uuid=True), ForeignKey('workflow_executions.id'))
    task_name = Column(String(100), nullable=False)
    status = Column(String(50), default='PENDING')
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    result = Column(JSONB)
    error = Column(Text)
    retries = Column(Integer, default=0)
    metadata = Column(JSONB)

class WorkflowLog(Base):
    __tablename__ = 'workflow_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    execution_id = Column(UUID(as_uuid=True), ForeignKey('workflow_executions.id'))
    task_id = Column(UUID(as_uuid=True), ForeignKey('task_results.id'), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    component = Column(String(100))
    metadata = Column(JSONB)

class Notification(Base):
    __tablename__ = 'notifications'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    type = Column(String(50), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    read = Column(Boolean, default=False)
    channel = Column(String(50))
    priority = Column(Integer, default=0)
    data = Column(JSONB)
    metadata = Column(JSONB)

class DailySummary(Base):
    __tablename__ = 'daily_summaries'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(Date, nullable=False, unique=True)
    total_items_scraped = Column(Integer, default=0)
    new_items_found = Column(Integer, default=0)
    items_processed = Column(Integer, default=0)
    bid_recommendations_generated = Column(Integer, default=0)
    bids_placed = Column(Integer, default=0)
    successful_bids = Column(Integer, default=0)
    total_potential_profit = Column(Float, default=0.0)
    errors_encountered = Column(Integer, default=0)
    metadata = Column(JSONB)

class FeedbackEntry(Base):
    __tablename__ = 'feedback_entries'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_id = Column(UUID(as_uuid=True), ForeignKey('auction_items.id'))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    feedback_type = Column(String(50), nullable=False)
    rating = Column(Integer)
    comments = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSONB)

class ModelPerformance(Base):
    __tablename__ = 'model_performance'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_date = Column(DateTime)
    sample_size = Column(Integer)
    metadata = Column(JSONB)

class UserPreference(Base):
    __tablename__ = 'user_preferences'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    category = Column(String(100))
    min_profit = Column(Float)
    max_risk = Column(Float)
    preferred_platforms = Column(JSONB)
    notification_settings = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def create_database(engine_url):
    """Create the database if it doesn't exist."""
    # Connect to the postgres database to create our application database
    postgres_engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres")
    
    with postgres_engine.connect() as conn:
        conn.execute("commit")
        
        # Check if database exists
        result = conn.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        if not result.fetchone():
            logger.info(f"Creating database {DB_NAME}")
            conn.execute(f"CREATE DATABASE {DB_NAME}")
            logger.info(f"Database {DB_NAME} created successfully")
        else:
            logger.info(f"Database {DB_NAME} already exists")

def init_db():
    """Initialize the database schema."""
    # Database connection URL
    engine_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    try:
        # Create database if it doesn't exist
        create_database(engine_url)
        
        # Create engine
        engine = create_engine(engine_url)
        
        # Create all tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Add default admin user if no users exist
        if not session.query(User).first():
            logger.info("Adding default admin user")
            from passlib.hash import bcrypt
            admin_user = User(
                username="admin",
                email="admin@autobay.example.com",
                password_hash=bcrypt.hash("admin123"),  # Change this in production!
                settings={"is_admin": True}
            )
            session.add(admin_user)
            session.commit()
            logger.info("Default admin user added successfully")
        
        # Close session
        session.close()
        
        logger.info("Database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize the AutoBay database")
    parser.add_argument("--force", action="store_true", help="Force recreation of all tables (WARNING: This will delete all existing data)")
    args = parser.parse_args()
    
    if args.force:
        # Database connection URL
        engine_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(engine_url)
        
        logger.warning("Dropping all tables due to --force flag")
        Base.metadata.drop_all(engine)
        logger.info("All tables dropped successfully")
    
    success = init_db()
    
    if success:
        print("Database initialization completed successfully")
        sys.exit(0)
    else:
        print("Database initialization failed. Check logs for details.")
        sys.exit(1)