"""
Budget optimization using knapsack algorithm.

This module provides classes and functions for optimizing auction item selection
within a budget using a knapsack algorithm.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .utils import AuctionItem, RiskScore


logger = logging.getLogger("bid_intelligence.optimization")


@dataclass
class ItemCombination:
    """
    Combination of auction items with aggregated metrics.
    
    Attributes:
        items: List of auction items in the combination
        total_cost: Total cost of the items
        total_profit: Total estimated profit
        average_risk: Average risk score
        budget: Budget constraint
        budget_utilization: Percentage of budget utilized
    """
    items: List[AuctionItem]
    total_cost: float
    total_profit: float
    average_risk: float
    budget: float
    budget_utilization: float
    
    @classmethod
    def from_items(cls, items: List[AuctionItem], budget: float) -> 'ItemCombination':
        """
        Create an item combination from a list of items.
        
        Args:
            items: List of auction items
            budget: Budget constraint
            
        Returns:
            ItemCombination: Item combination with aggregated metrics
        """
        total_cost = sum(item.current_bid for item in items)
        total_profit = sum(item.estimated_profit for item in items)
        
        # Calculate average risk score
        if items:
            average_risk = sum(item.risk_score.score for item in items) / len(items)
        else:
            average_risk = 0.0
        
        # Calculate budget utilization
        if budget > 0:
            budget_utilization = (total_cost / budget) * 100
        else:
            budget_utilization = 0.0
        
        return cls(
            items=items,
            total_cost=total_cost,
            total_profit=total_profit,
            average_risk=average_risk,
            budget=budget,
            budget_utilization=budget_utilization
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the item combination to a dictionary.
        
        Returns:
            Dict[str, Any]: Item combination as a dictionary
        """
        return {
            "items": [item.to_dict() for item in self.items],
            "item_count": len(self.items),
            "total_cost": self.total_cost,
            "total_profit": self.total_profit,
            "average_risk": self.average_risk,
            "budget": self.budget,
            "budget_utilization": self.budget_utilization,
            "roi": (self.total_profit / self.total_cost) * 100 if self.total_cost > 0 else 0.0
        }


class KnapsackOptimizer:
    """
    Optimizes auction item selection using a knapsack algorithm.
    
    This class implements a knapsack algorithm to generate budget-optimized
    combinations of auction items with associated risk scores.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the knapsack optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.max_combinations = self.config.get("max_combinations", 10)
        self.risk_penalty = self.config.get("risk_penalty", 0.5)
    
    def optimize(self, items: List[AuctionItem], budget: float) -> List[ItemCombination]:
        """
        Optimize item selection within a budget.
        
        Args:
            items: List of auction items to choose from
            budget: Budget constraint
            
        Returns:
            List[ItemCombination]: List of optimized item combinations
        """
        if not items:
            logger.warning("No items provided for optimization")
            return []
        
        if budget <= 0:
            logger.warning("Budget must be positive")
            return []
        
        logger.info(f"Optimizing {len(items)} items within budget ${budget:.2f}")
        
        # Sort items by profit-to-cost ratio (descending)
        sorted_items = sorted(
            items,
            key=lambda item: (item.estimated_profit / item.current_bid) if item.current_bid > 0 else 0,
            reverse=True
        )
        
        # Run knapsack algorithm
        start_time = time.time()
        combinations = self._knapsack_algorithm(sorted_items, budget)
        end_time = time.time()
        
        logger.info(f"Optimization completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Generated {len(combinations)} combinations")
        
        return combinations
    
    def _knapsack_algorithm(self, items: List[AuctionItem], budget: float) -> List[ItemCombination]:
        """
        Implement the knapsack algorithm.
        
        Args:
            items: List of auction items (sorted by profit-to-cost ratio)
            budget: Budget constraint
            
        Returns:
            List[ItemCombination]: List of optimized item combinations
        """
        n = len(items)
        
        # Create a 2D table for dynamic programming
        # dp[i][j] = maximum profit achievable with first i items and budget j
        dp = [[0.0 for _ in range(int(budget) + 1)] for _ in range(n + 1)]
        
        # Fill the dp table
        for i in range(1, n + 1):
            for j in range(int(budget) + 1):
                item = items[i - 1]
                item_cost = int(item.current_bid)
                
                if item_cost <= j:
                    # Calculate profit with risk penalty
                    risk_adjusted_profit = item.estimated_profit * (1 - item.risk_score.score * self.risk_penalty)
                    
                    # Maximum of including or excluding the item
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - item_cost] + risk_adjusted_profit)
                else:
                    # Item cost exceeds current budget, exclude it
                    dp[i][j] = dp[i - 1][j]
        
        # Backtrack to find the items in the optimal solution
        optimal_items = self._backtrack(dp, items, budget)
        optimal_combination = ItemCombination.from_items(optimal_items, budget)
        
        # Generate alternative combinations
        alternative_combinations = self._generate_alternatives(items, budget, optimal_items)
        
        # Combine and sort all combinations by total profit (descending)
        all_combinations = [optimal_combination] + alternative_combinations
        all_combinations.sort(key=lambda combo: combo.total_profit, reverse=True)
        
        # Limit the number of combinations
        return all_combinations[:self.max_combinations]
    
    def _backtrack(self, dp: List[List[float]], items: List[AuctionItem], budget: float) -> List[AuctionItem]:
        """
        Backtrack through the dynamic programming table to find the optimal items.
        
        Args:
            dp: Dynamic programming table
            items: List of auction items
            budget: Budget constraint
            
        Returns:
            List[AuctionItem]: List of items in the optimal solution
        """
        n = len(items)
        j = int(budget)
        optimal_items = []
        
        for i in range(n, 0, -1):
            if dp[i][j] != dp[i - 1][j]:
                # Item i is included in the optimal solution
                item = items[i - 1]
                optimal_items.append(item)
                j -= int(item.current_bid)
        
        return optimal_items
    
    def _generate_alternatives(self, items: List[AuctionItem], budget: float, 
                              excluded_items: List[AuctionItem]) -> List[ItemCombination]:
        """
        Generate alternative item combinations.
        
        Args:
            items: List of auction items
            budget: Budget constraint
            excluded_items: Items to exclude from one combination
            
        Returns:
            List[ItemCombination]: List of alternative item combinations
        """
        alternatives = []
        
        # Generate a combination excluding each item in the optimal solution
        for excluded_item in excluded_items:
            alternative_items = [item for item in items if item.item_id != excluded_item.item_id]
            
            # Run knapsack on the alternative items
            dp = [[0.0 for _ in range(int(budget) + 1)] for _ in range(len(alternative_items) + 1)]
            
            for i in range(1, len(alternative_items) + 1):
                for j in range(int(budget) + 1):
                    item = alternative_items[i - 1]
                    item_cost = int(item.current_bid)
                    
                    if item_cost <= j:
                        risk_adjusted_profit = item.estimated_profit * (1 - item.risk_score.score * self.risk_penalty)
                        dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - item_cost] + risk_adjusted_profit)
                    else:
                        dp[i][j] = dp[i - 1][j]
            
            # Backtrack to find the items
            alternative_optimal_items = self._backtrack(dp, alternative_items, budget)
            alternative = ItemCombination.from_items(alternative_optimal_items, budget)
            
            alternatives.append(alternative)
        
        # Generate a greedy combination
        greedy_items = []
        remaining_budget = budget
        
        for item in items:
            if item.current_bid <= remaining_budget:
                greedy_items.append(item)
                remaining_budget -= item.current_bid
        
        greedy_combination = ItemCombination.from_items(greedy_items, budget)
        alternatives.append(greedy_combination)
        
        # Generate a low-risk combination
        low_risk_items = sorted(items, key=lambda item: item.risk_score.score)
        low_risk_combination_items = []
        remaining_budget = budget
        
        for item in low_risk_items:
            if item.current_bid <= remaining_budget:
                low_risk_combination_items.append(item)
                remaining_budget -= item.current_bid
        
        low_risk_combination = ItemCombination.from_items(low_risk_combination_items, budget)
        alternatives.append(low_risk_combination)
        
        return alternatives
    
    def optimize_with_constraints(self, items: List[AuctionItem], budget: float,
                                 max_risk: float = 0.7,
                                 min_profit: float = 0.0,
                                 category_limits: Dict[str, int] = None) -> List[ItemCombination]:
        """
        Optimize item selection with additional constraints.
        
        Args:
            items: List of auction items to choose from
            budget: Budget constraint
            max_risk: Maximum average risk score
            min_profit: Minimum total profit
            category_limits: Maximum number of items per category
            
        Returns:
            List[ItemCombination]: List of optimized item combinations
        """
        if not items:
            logger.warning("No items provided for optimization")
            return []
        
        if budget <= 0:
            logger.warning("Budget must be positive")
            return []
        
        logger.info(f"Optimizing {len(items)} items with constraints")
        
        # Filter items by risk score
        filtered_items = [item for item in items if item.risk_score.score <= max_risk]
        
        if not filtered_items:
            logger.warning("No items meet the risk constraint")
            return []
        
        # Run basic optimization
        combinations = self.optimize(filtered_items, budget)
        
        # Filter combinations by minimum profit
        combinations = [combo for combo in combinations if combo.total_profit >= min_profit]
        
        # Apply category limits if specified
        if category_limits:
            valid_combinations = []
            
            for combo in combinations:
                # Count items per category
                category_counts = {}
                for item in combo.items:
                    category = item.category
                    category_counts[category] = category_counts.get(category, 0) + 1
                
                # Check if any category exceeds its limit
                valid = True
                for category, count in category_counts.items():
                    if category in category_limits and count > category_limits[category]:
                        valid = False
                        break
                
                if valid:
                    valid_combinations.append(combo)
            
            combinations = valid_combinations
        
        return combinations