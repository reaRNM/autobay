"""
Unit tests for the AI Scoring & NLP Interface.

This module contains unit tests for the AI Scoring & NLP Interface components.
"""

import unittest
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from ai_auction import (
    AIScoringEngine,
    NLPInterface,
    ItemData,
    ScoringResult,
    QueryResult
)
from ai_auction.utils import calculate_risk_score


class TestAIScoringEngine(unittest.TestCase):
    """Test cases for the AI Scoring Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scoring_engine = AIScoringEngine(model_dir="test_models")
        self.sample_items = self._create_sample_items()
    
    def _create_sample_items(self) -> List[ItemData]:
        """Create sample items for testing."""
        items = []
        
        # Create a few sample items
        for i in range(1, 6):
            item = ItemData(
                item_id=f"test-item-{i}",
                title=f"Test Item {i}",
                description=f"Test description {i}",
                category="Electronics" if i % 2 == 0 else "Collectibles",
                condition="New" if i % 3 == 0 else "Used",
                current_bid=100.0 * i,
                estimated_value=150.0 * i,
                estimated_profit=40.0 * i,
                profit_margin=30.0 + (i * 2),
                weight=1.0 * i,
                dimensions={"length": 10, "width": 10, "height": 5},
                estimated_shipping_cost=10.0 + (i * 2),
                seller_rating=0.8,
                seller_feedback_count=100,
                existing_scores={
                    "risk": 0.3,
                    "shipping_ease": 0.7,
                    "trend": 0.5
                }
            )
            items.append(item)
        
        return items
    
    def test_score_item(self):
        """Test scoring a single item."""
        # Score the first item
        item = self.sample_items[0]
        result = self.scoring_engine.score_item(item)
        
        # Check result
        self.assertIsInstance(result, ScoringResult)
        self.assertEqual(result.item_id, item.item_id)
        self.assertGreaterEqual(result.priority_score, 0.0)
        self.assertLessEqual(result.priority_score, 1.0)
        
        # Check components
        self.assertGreater(len(result.components), 0)
        for component in result.components:
            self.assertGreaterEqual(component.score, 0.0)
            self.assertLessEqual(component.score, 1.0)
            self.assertGreaterEqual(component.weight, 0.0)
            self.assertLessEqual(component.weight, 1.0)
            self.assertIsNotNone(component.explanation)
    
    def test_score_items(self):
        """Test scoring multiple items."""
        # Score all items
        results = self.scoring_engine.score_items(self.sample_items)
        
        # Check results
        self.assertEqual(len(results), len(self.sample_items))
        
        # Check sorting (should be sorted by priority score in descending order)
        for i in range(1, len(results)):
            self.assertGreaterEqual(results[i-1].priority_score, results[i].priority_score)
    
    def test_custom_weights(self):
        """Test scoring with custom weights."""
        # Define custom weights
        custom_weights = {
            "profit_potential": 0.7,
            "risk_assessment": 0.1,
            "shipping_ease": 0.1,
            "trend_prediction": 0.1
        }
        
        # Score with default weights
        default_results = self.scoring_engine.score_items(self.sample_items)
        
        # Score with custom weights
        custom_results = self.scoring_engine.score_items(self.sample_items, weights=custom_weights)
        
        # Check that results are different
        self.assertNotEqual(
            default_results[0].priority_score,
            custom_results[0].priority_score
        )
        
        # Check that profit component has higher weight
        profit_component = next(
            (c for c in custom_results[0].components if c.name == "profit_potential"),
            None
        )
        self.assertIsNotNone(profit_component)
        self.assertEqual(profit_component.weight, 0.7)


class TestNLPInterface(unittest.TestCase):
    """Test cases for the NLP Interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scoring_engine = AIScoringEngine(model_dir="test_models")
        self.nlp_interface = NLPInterface(scoring_engine=self.scoring_engine)
        self.sample_items = self._create_sample_items()
    
    def _create_sample_items(self) -> List[ItemData]:
        """Create sample items for testing."""
        items = []
        
        # Create a variety of sample items
        categories = ["Electronics", "Collectibles", "Furniture", "Jewelry", "Art"]
        conditions = ["New", "Like New", "Good", "Fair", "Poor"]
        
        for i in range(1, 21):
            category = categories[i % len(categories)]
            condition = conditions[i % len(conditions)]
            
            item = ItemData(
                item_id=f"test-item-{i}",
                title=f"Test {category} Item {i}",
                description=f"Test description {i}",
                category=category,
                condition=condition,
                current_bid=50.0 + (i * 10),
                estimated_value=100.0 + (i * 15),
                estimated_profit=30.0 + (i * 5),
                profit_margin=20.0 + (i * 2),
                weight=1.0 + (i % 5),
                dimensions={"length": 10, "width": 10, "height": 5},
                estimated_shipping_cost=10.0 + (i % 5) * 2,
                seller_rating=0.7 + (i % 4) * 0.1,
                seller_feedback_count=50 + (i * 10),
                existing_scores={
                    "risk": 0.3 + (i % 5) * 0.1,
                    "shipping_ease": 0.8 - (i % 5) * 0.1,
                    "trend": 0.5 + (i % 3) * 0.1
                }
            )
            items.append(item)
        
        return items
    
    def test_basic_query(self):
        """Test basic query processing."""
        # Process a simple query
        query = "Show me the top 5 items"
        result = self.nlp_interface.process_query(query, self.sample_items)
        
        # Check result
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query, query)
        self.assertEqual(result.parsed_intent, "show_items")
        self.assertLessEqual(len(result.items), 5)
    
    def test_price_filter(self):
        """Test query with price filter."""
        # Process query with price filter
        query = "Show me items under $100"
        result = self.nlp_interface.process_query(query, self.sample_items)
        
        # Check that all items are under $100
        for item in result.items:
            self.assertLessEqual(item.get("current_bid", 0), 100.0)
    
    def test_category_filter(self):
        """Test query with category filter."""
        # Process query with category filter
        query = "Show me electronics items"
        result = self.nlp_interface.process_query(query, self.sample_items)
        
        # Check that all items are electronics
        for item in result.items:
            self.assertIn("Electronics", item.get("title", ""))
    
    def test_profit_filter(self):
        """Test query with profit filter."""
        # Process query with profit filter
        query = "Show me items with profit over $50"
        result = self.nlp_interface.process_query(query, self.sample_items)
        
        # Check that all items have profit over $50
        for item in result.items:
            self.assertGreaterEqual(item.get("estimated_profit", 0), 50.0)
    
    def test_complex_query(self):
        """Test complex query with multiple filters."""
        # Process complex query
        query = "Show me the top 3 electronics items with low risk and high profit margin"
        result = self.nlp_interface.process_query(query, self.sample_items)
        
        # Check result
        self.assertEqual(result.parsed_intent, "show_items")
        self.assertLessEqual(len(result.items), 3)
        
        # Check entities
        self.assertIn("limit", result.entities)
        self.assertEqual(result.entities["limit"], 3)
    
    def test_calculate_profit_intent(self):
        """Test calculate profit intent."""
        # Process query with calculate profit intent
        query = "Calculate profit for electronics items"
        result = self.nlp_interface.process_query(query, self.sample_items)
        
        # Check intent
        self.assertEqual(result.parsed_intent, "calculate_profit")


class TestUtilities(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_risk_score_calculation(self):
        """Test risk score calculation."""
        # Calculate risk score
        risk_score = calculate_risk_score(
            price_volatility=0.5,
            historical_data_points=20,
            category_risk=0.3,
            seller_rating=0.8,
            time_pressure=0.4
        )
        
        # Check result
        self.assertGreaterEqual(risk_score.score, 0.0)
        self.assertLessEqual(risk_score.score, 1.0)
        self.assertEqual(len(risk_score.factors), 5)
        
        # Check factors
        self.assertIn("price_volatility", risk_score.factors)
        self.assertIn("data_confidence", risk_score.factors)
        self.assertIn("category_risk", risk_score.factors)
        self.assertIn("seller_rating", risk_score.factors)
        self.assertIn("time_pressure", risk_score.factors)


if __name__ == "__main__":
    unittest.main()