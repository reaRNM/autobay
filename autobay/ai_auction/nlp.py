"""
NLP Interface for Auction Research/Resale Automation Tool.

This module provides a natural language interface for querying auction items
using spaCy for entity extraction and intent recognition.
"""

import os
import re
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span, Token

from ai_auction.models import ItemData, QueryResult, NLPEntity
from ai_auction.scoring import AIScoringEngine


logger = logging.getLogger(__name__)


class NLPInterface:
    """
    NLP Interface for querying auction items using natural language.
    
    This class provides functionality to parse natural language queries,
    extract entities and intents, and return matching items.
    """
    
    def __init__(
        self,
        scoring_engine: AIScoringEngine,
        spacy_model: str = "en_core_web_sm",
        custom_patterns_path: Optional[str] = None
    ):
        """
        Initialize the NLP Interface.
        
        Args:
            scoring_engine: AI Scoring Engine instance
            spacy_model: spaCy model to use
            custom_patterns_path: Path to custom patterns file
        """
        self.scoring_engine = scoring_engine
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            logger.info("Downloading spaCy model...")
            os.system(f"python -m spacy download {spacy_model}")
            self.nlp = spacy.load(spacy_model)
        
        # Initialize matchers
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
        
        # Load custom patterns
        self.patterns = self._load_patterns(custom_patterns_path)
        
        # Add patterns to matchers
        self._add_patterns()
        
        # Define intent handlers
        self.intent_handlers = {
            "show_items": self._handle_show_items,
            "find_items": self._handle_show_items,  # Alias
            "calculate_profit": self._handle_calculate_profit,
            "estimate_shipping": self._handle_estimate_shipping,
            "analyze_risk": self._handle_analyze_risk,
            "compare_items": self._handle_compare_items,
            "get_trends": self._handle_get_trends
        }
        
        logger.info("NLP Interface initialized")
    
    def _load_patterns(self, custom_patterns_path: Optional[str]) -> Dict[str, Any]:
        """
        Load custom patterns from file or use default patterns.
        
        Args:
            custom_patterns_path: Path to custom patterns file
            
        Returns:
            Dictionary of patterns
        """
        default_patterns = {
            "intents": {
                "show_items": [
                    [{"LOWER": "show"}, {"LOWER": "me"}],
                    [{"LOWER": "find"}],
                    [{"LOWER": "get"}],
                    [{"LOWER": "list"}],
                    [{"LOWER": "display"}],
                    [{"LOWER": "what"}, {"LOWER": "are"}, {"LOWER": "the"}]
                ],
                "calculate_profit": [
                    [{"LOWER": "calculate"}, {"LOWER": "profit"}],
                    [{"LOWER": "estimate"}, {"LOWER": "profit"}],
                    [{"LOWER": "what"}, {"LOWER": "is"}, {"LOWER": "the"}, {"LOWER": "profit"}]
                ],
                "estimate_shipping": [
                    [{"LOWER": "estimate"}, {"LOWER": "shipping"}],
                    [{"LOWER": "calculate"}, {"LOWER": "shipping"}],
                    [{"LOWER": "shipping"}, {"LOWER": "cost"}]
                ],
                "analyze_risk": [
                    [{"LOWER": "analyze"}, {"LOWER": "risk"}],
                    [{"LOWER": "assess"}, {"LOWER": "risk"}],
                    [{"LOWER": "risk"}, {"LOWER": "assessment"}]
                ],
                "compare_items": [
                    [{"LOWER": "compare"}],
                    [{"LOWER": "versus"}],
                    [{"LOWER": "vs"}]
                ],
                "get_trends": [
                    [{"LOWER": "trend"}],
                    [{"LOWER": "trends"}],
                    [{"LOWER": "popular"}],
                    [{"LOWER": "popularity"}]
                ]
            },
            "entities": {
                "price_range": [
                    [{"LOWER": "under"}, {"SHAPE": "$ddd"}],
                    [{"LOWER": "less"}, {"LOWER": "than"}, {"SHAPE": "$ddd"}],
                    [{"LOWER": "below"}, {"SHAPE": "$ddd"}],
                    [{"LOWER": "over"}, {"SHAPE": "$ddd"}],
                    [{"LOWER": "more"}, {"LOWER": "than"}, {"SHAPE": "$ddd"}],
                    [{"LOWER": "above"}, {"SHAPE": "$ddd"}],
                    [{"LOWER": "between"}, {"SHAPE": "$ddd"}, {"LOWER": "and"}, {"SHAPE": "$ddd"}]
                ],
                "profit_threshold": [
                    [{"LOWER": "profit"}, {"LOWER": "over"}, {"SHAPE": "$ddd"}],
                    [{"LOWER": "profit"}, {"LOWER": "above"}, {"SHAPE": "$ddd"}],
                    [{"LOWER": "profit"}, {"LOWER": "more"}, {"LOWER": "than"}, {"SHAPE": "$ddd"}],
                    [{"LOWER": "margin"}, {"LOWER": "over"}, {"IS_DIGIT": True}, {"LOWER": "%"}],
                    [{"LOWER": "margin"}, {"LOWER": "above"}, {"IS_DIGIT": True}, {"LOWER": "%"}]
                ],
                "risk_level": [
                    [{"LOWER": "low"}, {"LOWER": "risk"}],
                    [{"LOWER": "high"}, {"LOWER": "risk"}],
                    [{"LOWER": "medium"}, {"LOWER": "risk"}],
                    [{"LOWER": "moderate"}, {"LOWER": "risk"}]
                ],
                "shipping_ease": [
                    [{"LOWER": "easy"}, {"LOWER": "to"}, {"LOWER": "ship"}],
                    [{"LOWER": "difficult"}, {"LOWER": "to"}, {"LOWER": "ship"}],
                    [{"LOWER": "hard"}, {"LOWER": "to"}, {"LOWER": "ship"}],
                    [{"LOWER": "shipping"}, {"LOWER": "ease"}]
                ],
                "item_condition": [
                    [{"LOWER": "new"}],
                    [{"LOWER": "used"}],
                    [{"LOWER": "like"}, {"LOWER": "new"}],
                    [{"LOWER": "good"}, {"LOWER": "condition"}],
                    [{"LOWER": "fair"}, {"LOWER": "condition"}],
                    [{"LOWER": "poor"}, {"LOWER": "condition"}]
                ],
                "category": [
                    [{"LOWER": "electronics"}],
                    [{"LOWER": "clothing"}],
                    [{"LOWER": "furniture"}],
                    [{"LOWER": "collectibles"}],
                    [{"LOWER": "jewelry"}],
                    [{"LOWER": "art"}],
                    [{"LOWER": "toys"}],
                    [{"LOWER": "books"}]
                ],
                "limit": [
                    [{"LOWER": "top"}, {"IS_DIGIT": True}],
                    [{"LOWER": "first"}, {"IS_DIGIT": True}],
                    [{"LOWER": "limit"}, {"IS_DIGIT": True}]
                ],
                "sort_by": [
                    [{"LOWER": "sort"}, {"LOWER": "by"}, {"LOWER": "profit"}],
                    [{"LOWER": "sort"}, {"LOWER": "by"}, {"LOWER": "price"}],
                    [{"LOWER": "sort"}, {"LOWER": "by"}, {"LOWER": "risk"}],
                    [{"LOWER": "sort"}, {"LOWER": "by"}, {"LOWER": "score"}],
                    [{"LOWER": "order"}, {"LOWER": "by"}, {"LOWER": "profit"}],
                    [{"LOWER": "order"}, {"LOWER": "by"}, {"LOWER": "price"}],
                    [{"LOWER": "order"}, {"LOWER": "by"}, {"LOWER": "risk"}],
                    [{"LOWER": "order"}, {"LOWER": "by"}, {"LOWER": "score"}]
                ]
            },
            "phrases": {
                "high_margin": ["high margin", "high profit margin", "good margin", "profitable"],
                "low_risk": ["low risk", "safe", "secure", "reliable"],
                "high_shipping_ease": ["easy to ship", "lightweight", "small", "compact"],
                "trending": ["trending", "popular", "in demand", "hot items"]
            }
        }
        
        if custom_patterns_path and os.path.exists(custom_patterns_path):
            try:
                with open(custom_patterns_path, 'r') as f:
                    custom_patterns = json.load(f)
                    
                # Merge with default patterns
                for category, patterns in custom_patterns.items():
                    if category in default_patterns:
                        for key, value in patterns.items():
                            if key in default_patterns[category]:
                                default_patterns[category][key].extend(value)
                            else:
                                default_patterns[category][key] = value
                    else:
                        default_patterns[category] = patterns
                
                logger.info(f"Loaded custom patterns from {custom_patterns_path}")
                
            except Exception as e:
                logger.error(f"Error loading custom patterns: {e}")
        
        return default_patterns
    
    def _add_patterns(self) -> None:
        """Add patterns to matchers."""
        # Add intent patterns
        for intent, patterns in self.patterns["intents"].items():
            for pattern in patterns:
                self.matcher.add(f"INTENT_{intent}", [pattern])
        
        # Add entity patterns
        for entity, patterns in self.patterns["entities"].items():
            for pattern in patterns:
                self.matcher.add(f"ENTITY_{entity}", [pattern])
        
        # Add phrase patterns
        for category, phrases in self.patterns["phrases"].items():
            phrase_patterns = [self.nlp(text) for text in phrases]
            self.phrase_matcher.add(f"PHRASE_{category}", phrase_patterns)
    
    def _extract_entities(self, doc: Doc) -> Dict[str, List[NLPEntity]]:
        """
        Extract entities from a spaCy Doc.
        
        Args:
            doc: spaCy Doc
            
        Returns:
            Dictionary of entity types and values
        """
        entities = {}
        
        # Use matcher to find entities
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            match_type = self.nlp.vocab.strings[match_id]
            if match_type.startswith("ENTITY_"):
                entity_type = match_type[7:]  # Remove "ENTITY_" prefix
                span = doc[start:end]
                
                # Extract value based on entity type
                value = self._extract_entity_value(entity_type, span)
                
                if value is not None:
                    entity = NLPEntity(name=entity_type, value=value)
                    if entity_type not in entities:
                        entities[entity_type] = []
                    entities[entity_type].append(entity)
        
        # Use phrase matcher to find phrases
        phrase_matches = self.phrase_matcher(doc)
        for match_id, start, end in phrase_matches:
            match_type = self.nlp.vocab.strings[match_id]
            if match_type.startswith("PHRASE_"):
                phrase_type = match_type[7:]  # Remove "PHRASE_" prefix
                span = doc[start:end]
                
                entity = NLPEntity(name=phrase_type, value=span.text)
                if phrase_type not in entities:
                    entities[phrase_type] = []
                entities[phrase_type].append(entity)
        
        # Extract numeric entities
        self._extract_numeric_entities(doc, entities)
        
        # Extract named entities
        self._extract_named_entities(doc, entities)
        
        return entities
    
    def _extract_entity_value(self, entity_type: str, span: Span) -> Any:
        """
        Extract value from an entity span based on entity type.
        
        Args:
            entity_type: Entity type
            span: Entity span
            
        Returns:
            Extracted value
        """
        text = span.text.lower()
        
        if entity_type == "price_range":
            # Extract price range
            numbers = re.findall(r'\$?(\d+(?:\.\d+)?)', text)
            if "under" in text or "less than" in text or "below" in text:
                if numbers:
                    return {"max": float(numbers[0])}
            elif "over" in text or "more than" in text or "above" in text:
                if numbers:
                    return {"min": float(numbers[0])}
            elif "between" in text and len(numbers) >= 2:
                return {"min": float(numbers[0]), "max": float(numbers[1])}
        
        elif entity_type == "profit_threshold":
            # Extract profit threshold
            numbers = re.findall(r'\$?(\d+(?:\.\d+)?)', text)
            if "%" in text:
                if numbers:
                    return {"margin": float(numbers[0])}
            else:
                if numbers:
                    return {"amount": float(numbers[0])}
        
        elif entity_type == "risk_level":
            # Extract risk level
            if "low" in text:
                return "low"
            elif "high" in text:
                return "high"
            elif "medium" in text or "moderate" in text:
                return "medium"
        
        elif entity_type == "shipping_ease":
            # Extract shipping ease
            if "easy" in text:
                return "high"
            elif "difficult" in text or "hard" in text:
                return "low"
            else:
                return "medium"
        
        elif entity_type == "item_condition":
            # Extract item condition
            if "new" in text:
                return "new"
            elif "like new" in text:
                return "like_new"
            elif "good" in text:
                return "good"
            elif "fair" in text:
                return "fair"
            elif "poor" in text:
                return "poor"
            elif "used" in text:
                return "used"
        
        elif entity_type == "category":
            # Extract category
            for token in span:
                if token.text.lower() in [
                    "electronics", "clothing", "furniture", "collectibles",
                    "jewelry", "art", "toys", "books"
                ]:
                    return token.text.lower()
        
        elif entity_type == "limit":
            # Extract limit
            numbers = re.findall(r'(\d+)', text)
            if numbers:
                return int(numbers[0])
        
        elif entity_type == "sort_by":
            # Extract sort by
            if "profit" in text:
                return "profit"
            elif "price" in text:
                return "price"
            elif "risk" in text:
                return "risk"
            elif "score" in text:
                return "score"
        
        return None
    
    def _extract_numeric_entities(self, doc: Doc, entities: Dict[str, List[NLPEntity]]) -> None:
        """
        Extract numeric entities from a spaCy Doc.
        
        Args:
            doc: spaCy Doc
            entities: Entities dictionary to update
        """
        # Extract money entities
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                value = re.findall(r'\$?(\d+(?:\.\d+)?)', ent.text)
                if value:
                    entity = NLPEntity(name="price", value=float(value[0]))
                    if "price" not in entities:
                        entities["price"] = []
                    entities["price"].append(entity)
            
            elif ent.label_ == "CARDINAL" or ent.label_ == "QUANTITY":
                # Check if it's a percentage
                if "%" in ent.text:
                    value = re.findall(r'(\d+(?:\.\d+)?)', ent.text)
                    if value:
                        entity = NLPEntity(name="percentage", value=float(value[0]))
                        if "percentage" not in entities:
                            entities["percentage"] = []
                        entities["percentage"].append(entity)
                
                # Check if it's a count
                elif re.match(r'^\d+$', ent.text):
                    entity = NLPEntity(name="count", value=int(ent.text))
                    if "count" not in entities:
                        entities["count"] = []
                    entities["count"].append(entity)
    
    def _extract_named_entities(self, doc: Doc, entities: Dict[str, List[NLPEntity]]) -> None:
        """
        Extract named entities from a spaCy Doc.
        
        Args:
            doc: spaCy Doc
            entities: Entities dictionary to update
        """
        for ent in doc.ents:
            if ent.label_ == "ORG":
                entity = NLPEntity(name="organization", value=ent.text)
                if "organization" not in entities:
                    entities["organization"] = []
                entities["organization"].append(entity)
            
            elif ent.label_ == "PRODUCT":
                entity = NLPEntity(name="product", value=ent.text)
                if "product" not in entities:
                    entities["product"] = []
                entities["product"].append(entity)
    
    def _identify_intent(self, doc: Doc, entities: Dict[str, List[NLPEntity]]) -> str:
        """
        Identify the intent of a query.
        
        Args:
            doc: spaCy Doc
            entities: Extracted entities
            
        Returns:
            Intent string
        """
        # Use matcher to find intents
        matches = self.matcher(doc)
        intent_matches = []
        
        for match_id, start, end in matches:
            match_type = self.nlp.vocab.strings[match_id]
            if match_type.startswith("INTENT_"):
                intent = match_type[7:]  # Remove "INTENT_" prefix
                intent_matches.append(intent)
        
        # Count intent matches
        intent_counts = {}
        for intent in intent_matches:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Default intent
        default_intent = "show_items"
        
        # Return the most frequent intent or default
        if intent_counts:
            return max(intent_counts.items(), key=lambda x: x[1])[0]
        
        return default_intent
    
    def _normalize_entities(self, entities: Dict[str, List[NLPEntity]]) -> Dict[str, Any]:
        """
        Normalize extracted entities into a structured format.
        
        Args:
            entities: Extracted entities
            
        Returns:
            Normalized entities dictionary
        """
        normalized = {}
        
        # Process price range
        if "price_range" in entities:
            price_range = {}
            for entity in entities["price_range"]:
                if isinstance(entity.value, dict):
                    price_range.update(entity.value)
            
            if price_range:
                normalized["price_range"] = price_range
        
        # Process individual price
        if "price" in entities and "price_range" not in normalized:
            prices = [entity.value for entity in entities["price"]]
            if len(prices) == 1:
                normalized["price_range"] = {"max": prices[0]}
            elif len(prices) >= 2:
                normalized["price_range"] = {"min": min(prices), "max": max(prices)}
        
        # Process profit threshold
        if "profit_threshold" in entities:
            for entity in entities["profit_threshold"]:
                if isinstance(entity.value, dict):
                    if "margin" in entity.value:
                        normalized["min_profit_margin"] = entity.value["margin"]
                    if "amount" in entity.value:
                        normalized["min_profit_amount"] = entity.value["amount"]
        
        # Process risk level
        if "risk_level" in entities:
            risk_levels = [entity.value for entity in entities["risk_level"]]
            if risk_levels:
                normalized["risk_level"] = risk_levels[0]
        
        # Process shipping ease
        if "shipping_ease" in entities:
            shipping_ease = [entity.value for entity in entities["shipping_ease"]]
            if shipping_ease:
                normalized["shipping_ease"] = shipping_ease[0]
        
        # Process item condition
        if "item_condition" in entities:
            conditions = [entity.value for entity in entities["item_condition"]]
            if conditions:
                normalized["item_condition"] = conditions
        
        # Process category
        if "category" in entities:
            categories = [entity.value for entity in entities["category"]]
            if categories:
                normalized["category"] = categories
        
        # Process limit
        if "limit" in entities:
            limits = [entity.value for entity in entities["limit"]]
            if limits:
                normalized["limit"] = max(limits)
        elif "count" in entities:
            counts = [entity.value for entity in entities["count"]]
            if counts:
                normalized["limit"] = max(counts)
        else:
            # Default limit
            normalized["limit"] = 10
        
        # Process sort by
        if "sort_by" in entities:
            sort_by = [entity.value for entity in entities["sort_by"]]
            if sort_by:
                normalized["sort_by"] = sort_by[0]
        
        # Process phrases
        for phrase_type in ["high_margin", "low_risk", "high_shipping_ease", "trending"]:
            if phrase_type in entities:
                if phrase_type == "high_margin":
                    normalized["min_profit_margin"] = 30.0
                elif phrase_type == "low_risk":
                    normalized["risk_level"] = "low"
                elif phrase_type == "high_shipping_ease":
                    normalized["shipping_ease"] = "high"
                elif phrase_type == "trending":
                    normalized["trending"] = True
        
        return normalized
    
    def _handle_show_items(
        self, 
        normalized_entities: Dict[str, Any], 
        items: List[ItemData]
    ) -> Dict[str, Any]:
        """
        Handle 'show_items' intent.
        
        Args:
            normalized_entities: Normalized entities
            items: List of items
            
        Returns:
            Response dictionary
        """
        # Apply filters
        filtered_items = self._filter_items(items, normalized_entities)
        
        # Score items
        scored_items = self.scoring_engine.score_items(filtered_items)
        
        # Sort items
        sort_by = normalized_entities.get("sort_by", "score")
        reverse = True  # Default to descending
        
        if sort_by == "price":
            scored_items.sort(key=lambda x: next((item.current_bid for item in filtered_items if item.item_id == x.item_id), 0), reverse=reverse)
        elif sort_by == "profit":
            scored_items.sort(key=lambda x: next((item.estimated_profit for item in filtered_items if item.item_id == x.item_id), 0), reverse=reverse)
        elif sort_by == "risk":
            # For risk, lower is better
            scored_items.sort(key=lambda x: next((comp.score for comp in x.components if comp.name == "risk_assessment"), 0), reverse=not reverse)
        # Default is already sorted by priority_score
        
        # Apply limit
        limit = normalized_entities.get("limit", 10)
        limited_items = scored_items[:limit]
        
        # Prepare response
        response = {
            "total_items": len(filtered_items),
            "shown_items": len(limited_items),
            "items": []
        }
        
        # Add items to response
        for score_result in limited_items:
            item = next((item for item in filtered_items if item.item_id == score_result.item_id), None)
            if item:
                response["items"].append({
                    "item_id": item.item_id,
                    "title": item.title,
                    "current_bid": item.current_bid,
                    "estimated_value": item.estimated_value,
                    "estimated_profit": item.estimated_profit,
                    "profit_margin": item.profit_margin,
                    "priority_score": score_result.priority_score,
                    "components": [
                        {
                            "name": comp.name,
                            "score": comp.score,
                            "explanation": comp.explanation
                        } for comp in score_result.components
                    ]
                })
        
        return response
    
    def _handle_calculate_profit(
        self, 
        normalized_entities: Dict[str, Any], 
        items: List[ItemData]
    ) -> Dict[str, Any]:
        """
        Handle 'calculate_profit' intent.
        
        Args:
            normalized_entities: Normalized entities
            items: List of items
            
        Returns:
            Response dictionary
        """
        # Apply filters
        filtered_items = self._filter_items(items, normalized_entities)
        
        # Calculate profit statistics
        total_profit = sum(item.estimated_profit for item in filtered_items)
        avg_profit = total_profit / len(filtered_items) if filtered_items else 0
        avg_margin = sum(item.profit_margin for item in filtered_items) / len(filtered_items) if filtered_items else 0
        
        # Find highest profit item
        highest_profit_item = max(filtered_items, key=lambda x: x.estimated_profit) if filtered_items else None
        
        # Prepare response
        response = {
            "total_items": len(filtered_items),
            "total_profit": total_profit,
            "average_profit": avg_profit,
            "average_margin": avg_margin,
            "highest_profit_item": None
        }
        
        if highest_profit_item:
            response["highest_profit_item"] = {
                "item_id": highest_profit_item.item_id,
                "title": highest_profit_item.title,
                "current_bid": highest_profit_item.current_bid,
                "estimated_value": highest_profit_item.estimated_value,
                "estimated_profit": highest_profit_item.estimated_profit,
                "profit_margin": highest_profit_item.profit_margin
            }
        
        return response
    
    def _handle_estimate_shipping(
        self, 
        normalized_entities: Dict[str, Any], 
        items: List[ItemData]
    ) -> Dict[str, Any]:
        """
        Handle 'estimate_shipping' intent.
        
        Args:
            normalized_entities: Normalized entities
            items: List of items
            
        Returns:
            Response dictionary
        """
        # Apply filters
        filtered_items = self._filter_items(items, normalized_entities)
        
        # Calculate shipping statistics
        total_shipping = sum(item.estimated_shipping_cost for item in filtered_items)
        avg_shipping = total_shipping / len(filtered_items) if filtered_items else 0
        
        # Score items for shipping ease
        scored_items = self.scoring_engine.score_items(filtered_items)
        
        # Find items with best shipping ease
        shipping_scores = []
        for score_result in scored_items:
            shipping_component = next((comp for comp in score_result.components if comp.name == "shipping_ease"), None)
            if shipping_component:
                item = next((item for item in filtered_items if item.item_id == score_result.item_id), None)
                if item:
                    shipping_scores.append({
                        "item_id": item.item_id,
                        "title": item.title,
                        "shipping_cost": item.estimated_shipping_cost,
                        "shipping_ease_score": shipping_component.score,
                        "explanation": shipping_component.explanation
                    })
        
        # Sort by shipping ease score (descending)
        shipping_scores.sort(key=lambda x: x["shipping_ease_score"], reverse=True)
        
        # Apply limit
        limit = normalized_entities.get("limit", 5)
        limited_items = shipping_scores[:limit]
        
        # Prepare response
        response = {
            "total_items": len(filtered_items),
            "total_shipping_cost": total_shipping,
            "average_shipping_cost": avg_shipping,
            "best_shipping_items": limited_items
        }
        
        return response
    
    def _handle_analyze_risk(
        self, 
        normalized_entities: Dict[str, Any], 
        items: List[ItemData]
    ) -> Dict[str, Any]:
        """
        Handle 'analyze_risk' intent.
        
        Args:
            normalized_entities: Normalized entities
            items: List of items
            
        Returns:
            Response dictionary
        """
        # Apply filters
        filtered_items = self._filter_items(items, normalized_entities)
        
        # Score items
        scored_items = self.scoring_engine.score_items(filtered_items)
        
        # Extract risk assessments
        risk_assessments = []
        for score_result in scored_items:
            risk_component = next((comp for comp in score_result.components if comp.name == "risk_assessment"), None)
            if risk_component:
                item = next((item for item in filtered_items if item.item_id == score_result.item_id), None)
                if item:
                    risk_level = "Low" if risk_component.score > 0.7 else "Medium" if risk_component.score > 0.4 else "High"
                    risk_assessments.append({
                        "item_id": item.item_id,
                        "title": item.title,
                        "current_bid": item.current_bid,
                        "risk_score": risk_component.score,
                        "risk_level": risk_level,
                        "explanation": risk_component.explanation,
                        "factors": risk_component.factors
                    })
        
        # Sort by risk score (descending, as higher score means lower risk)
        risk_assessments.sort(key=lambda x: x["risk_score"], reverse=True)
        
        # Apply limit
        limit = normalized_entities.get("limit", 10)
        limited_items = risk_assessments[:limit]
        
        # Calculate risk statistics
        avg_risk_score = sum(item["risk_score"] for item in risk_assessments) / len(risk_assessments) if risk_assessments else 0
        risk_distribution = {
            "low": sum(1 for item in risk_assessments if item["risk_level"] == "Low"),
            "medium": sum(1 for item in risk_assessments if item["risk_level"] == "Medium"),
            "high": sum(1 for item in risk_assessments if item["risk_level"] == "High")
        }
        
        # Prepare response
        response = {
            "total_items": len(filtered_items),
            "average_risk_score": avg_risk_score,
            "risk_distribution": risk_distribution,
            "items": limited_items
        }
        
        return response
    
    def _handle_compare_items(
        self, 
        normalized_entities: Dict[str, Any], 
        items: List[ItemData]
    ) -> Dict[str, Any]:
        """
        Handle 'compare_items' intent.
        
        Args:
            normalized_entities: Normalized entities
            items: List of items
            
        Returns:
            Response dictionary
        """
        # Apply filters
        filtered_items = self._filter_items(items, normalized_entities)
        
        # Score items
        scored_items = self.scoring_engine.score_items(filtered_items)
        
        # Apply limit
        limit = normalized_entities.get("limit", 5)
        limited_items = scored_items[:limit]
        
        # Prepare comparison data
        comparison = []
        for score_result in limited_items:
            item = next((item for item in filtered_items if item.item_id == score_result.item_id), None)
            if item:
                item_data = {
                    "item_id": item.item_id,
                    "title": item.title,
                    "current_bid": item.current_bid,
                    "estimated_value": item.estimated_value,
                    "estimated_profit": item.estimated_profit,
                    "profit_margin": item.profit_margin,
                    "priority_score": score_result.priority_score,
                    "components": {}
                }
                
                # Add component scores
                for comp in score_result.components:
                    item_data["components"][comp.name] = {
                        "score": comp.score,
                        "explanation": comp.explanation
                    }
                
                comparison.append(item_data)
        
        # Prepare response
        response = {
            "total_items": len(filtered_items),
            "compared_items": len(comparison),
            "comparison": comparison
        }
        
        return response
    
    def _handle_get_trends(
        self, 
        normalized_entities: Dict[str, Any], 
        items: List[ItemData]
    ) -> Dict[str, Any]:
        """
        Handle 'get_trends' intent.
        
        Args:
            normalized_entities: Normalized entities
            items: List of items
            
        Returns:
            Response dictionary
        """
        # Apply filters
        filtered_items = self._filter_items(items, normalized_entities)
        
        # Score items
        scored_items = self.scoring_engine.score_items(filtered_items)
        
        # Extract trend predictions
        trend_predictions = []
        for score_result in scored_items:
            trend_component = next((comp for comp in score_result.components if comp.name == "trend_prediction"), None)
            if trend_component:
                item = next((item for item in filtered_items if item.item_id == score_result.item_id), None)
                if item:
                    trend_predictions.append({
                        "item_id": item.item_id,
                        "title": item.title,
                        "category": item.category,
                        "trend_score": trend_component.score,
                        "explanation": trend_component.explanation
                    })
        
        # Sort by trend score (descending)
        trend_predictions.sort(key=lambda x: x["trend_score"], reverse=True)
        
        # Apply limit
        limit = normalized_entities.get("limit", 10)
        limited_items = trend_predictions[:limit]
        
        # Calculate category trends
        category_trends = {}
        for item in trend_predictions:
            category = item["category"]
            if category not in category_trends:
                category_trends[category] = {
                    "count": 0,
                    "total_score": 0.0,
                    "average_score": 0.0
                }
            
            category_trends[category]["count"] += 1
            category_trends[category]["total_score"] += item["trend_score"]
        
        # Calculate average scores
        for category in category_trends:
            category_trends[category]["average_score"] = (
                category_trends[category]["total_score"] / category_trends[category]["count"]
            )
        
        # Sort categories by average score
        sorted_categories = sorted(
            category_trends.items(),
            key=lambda x: x[1]["average_score"],
            reverse=True
        )
        
        # Prepare response
        response = {
            "total_items": len(filtered_items),
            "trending_items": limited_items,
            "category_trends": [
                {
                    "category": category,
                    "count": data["count"],
                    "average_trend_score": data["average_score"]
                }
                for category, data in sorted_categories
            ]
        }
        
        return response
    
    def _filter_items(
        self, 
        items: List[ItemData], 
        filters: Dict[str, Any]
    ) -> List[ItemData]:
        """
        Filter items based on extracted entities.
        
        Args:
            items: List of items
            filters: Filter criteria
            
        Returns:
            Filtered list of items
        """
        filtered_items = items
        
        # Filter by price range
        if "price_range" in filters:
            price_range = filters["price_range"]
            if "min" in price_range:
                filtered_items = [item for item in filtered_items if item.current_bid >= price_range["min"]]
            if "max" in price_range:
                filtered_items = [item for item in filtered_items if item.current_bid <= price_range["max"]]
        
        # Filter by profit margin
        if "min_profit_margin" in filters:
            min_margin = filters["min_profit_margin"]
            filtered_items = [item for item in filtered_items if item.profit_margin >= min_margin]
        
        # Filter by profit amount
        if "min_profit_amount" in filters:
            min_profit = filters["min_profit_amount"]
            filtered_items = [item for item in filtered_items if item.estimated_profit >= min_profit]
        
        # Filter by risk level
        if "risk_level" in filters:
            risk_level = filters["risk_level"]
            if risk_level == "low":
                # Low risk means high risk score (inverted)
                filtered_items = [item for item in filtered_items if item.existing_scores.get("risk", 0.5) <= 0.3]
            elif risk_level == "medium":
                filtered_items = [item for item in filtered_items if 0.3 < item.existing_scores.get("risk", 0.5) <= 0.7]
            elif risk_level == "high":
                filtered_items = [item for item in filtered_items if item.existing_scores.get("risk", 0.5) > 0.7]
        
        # Filter by shipping ease
        if "shipping_ease" in filters:
            shipping_ease = filters["shipping_ease"]
            if shipping_ease == "high":
                filtered_items = [item for item in filtered_items if item.existing_scores.get("shipping_ease", 0.5) >= 0.7]
            elif shipping_ease == "medium":
                filtered_items = [item for item in filtered_items if 0.3 <= item.existing_scores.get("shipping_ease", 0.5) < 0.7]
            elif shipping_ease == "low":
                filtered_items = [item for item in filtered_items if item.existing_scores.get("shipping_ease", 0.5) < 0.3]
        
        # Filter by item condition
        if "item_condition" in filters:
            conditions = filters["item_condition"]
            filtered_items = [item for item in filtered_items if item.condition.lower() in conditions]
        
        # Filter by category
        if "category" in filters:
            categories = filters["category"]
            filtered_items = [item for item in filtered_items if item.category.lower() in categories]
        
        return filtered_items
    
    def process_query(
        self, 
        query: str, 
        items: List[ItemData]
    ) -> QueryResult:
        """
        Process a natural language query and return matching items.
        
        Args:
            query: Natural language query
            items: List of items to search
            
        Returns:
            QueryResult object with matching items
        """
        start_time = time.time()
        
        # Parse query with spaCy
        doc = self.nlp(query)
        
        # Extract entities
        entities = self._extract_entities(doc)
        
        # Identify intent
        intent = self._identify_intent(doc, entities)
        
        # Normalize entities
        normalized_entities = self._normalize_entities(entities)
        
        # Handle intent
        handler = self.intent_handlers.get(intent, self._handle_show_items)
        response = handler(normalized_entities, items)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Create and return query result
        result = QueryResult(
            query=query,
            parsed_intent=intent,
            entities=normalized_entities,
            items=response.get("items", []),
            total_items=response.get("total_items", 0),
            execution_time=execution_time
        )
        
        logger.info(f"Processed query: '{query}' (intent: {intent}, execution time: {execution_time:.2f}s)")
        return result