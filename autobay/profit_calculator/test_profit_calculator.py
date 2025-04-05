"""
Unit tests for the Profit Calculator.
"""

import unittest
from profit_calculator import FeeConfig, ProfitCalculator


class TestFeeConfig(unittest.TestCase):
    """Test cases for the FeeConfig class."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = FeeConfig()
        self.assertEqual(config.buyer_premium_rate, 0.15)
        self.assertEqual(config.sales_tax_rate, 0.07)
        self.assertEqual(config.platform_fee_rate, 0.1275)
        self.assertEqual(config.shipping_cost, 10.0)
    
    def test_custom_values(self):
        """Test that custom values are set correctly."""
        config = FeeConfig(
            buyer_premium_rate=0.20,
            sales_tax_rate=0.08,
            platform_fee_rate=0.15,
            shipping_cost=15.0
        )
        self.assertEqual(config.buyer_premium_rate, 0.20)
        self.assertEqual(config.sales_tax_rate, 0.08)
        self.assertEqual(config.platform_fee_rate, 0.15)
        self.assertEqual(config.shipping_cost, 15.0)
    
    def test_validation(self):
        """Test that validation works correctly."""
        # Test valid values
        config = FeeConfig(
            buyer_premium_rate=0.5,
            sales_tax_rate=0.1,
            platform_fee_rate=0.2
        )
        self.assertEqual(config.buyer_premium_rate, 0.5)
        
        # Test invalid values
        with self.assertRaises(ValueError):
            FeeConfig(buyer_premium_rate=1.5)
        
        with self.assertRaises(ValueError):
            FeeConfig(sales_tax_rate=-0.1)
    
    def test_from_dict(self):
        """Test creating a config from a dictionary."""
        config_dict = {
            'buyer_premium_rate': 0.25,
            'sales_tax_rate': 0.06,
            'platform_fee_rate': 0.10,
            'shipping_cost': 12.0,
            'additional_fees': {'handling': 5.0}
        }
        
        config = FeeConfig.from_dict(config_dict)
        self.assertEqual(config.buyer_premium_rate, 0.25)
        self.assertEqual(config.sales_tax_rate, 0.06)
        self.assertEqual(config.platform_fee_rate, 0.10)
        self.assertEqual(config.shipping_cost, 12.0)
        self.assertEqual(config.additional_fees, {'handling': 5.0})
    
    def test_tiered_platform_fees(self):
        """Test tiered platform fees."""
        config = FeeConfig(
            platform_fee_rate=0.15,
            use_tiered_platform_fees=True,
            tiered_platform_fees={
                0.0: 0.15,
                100.0: 0.12,
                500.0: 0.10,
                1000.0: 0.08
            }
        )
        
        self.assertEqual(config.get_platform_fee_rate(50.0), 0.15)
        self.assertEqual(config.get_platform_fee_rate(200.0), 0.12)
        self.assertEqual(config.get_platform_fee_rate(750.0), 0.10)
        self.assertEqual(config.get_platform_fee_rate(1500.0), 0.08)


class TestProfitCalculator(unittest.TestCase):
    """Test cases for the ProfitCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = ProfitCalculator()
    
    def test_calculate_buyer_premium(self):
        """Test calculating buyer premium."""
        self.assertEqual(self.calculator.calculate_buyer_premium(100.0), 15.0)
        self.assertEqual(self.calculator.calculate_buyer_premium(200.0), 30.0)
    
    def test_calculate_sales_tax(self):
        """Test calculating sales tax."""
        self.assertEqual(self.calculator.calculate_sales_tax(100.0, 15.0), 8.05)
        self.assertEqual(self.calculator.calculate_sales_tax(200.0, 30.0), 16.1)
    
    def test_calculate_platform_fee(self):
        """Test calculating platform fee."""
        self.assertEqual(self.calculator.calculate_platform_fee(100.0), 12.75)
        self.assertEqual(self.calculator.calculate_platform_fee(200.0), 25.5)
    
    def test_calculate_total_cost(self):
        """Test calculating total cost."""
        cost_breakdown = self.calculator.calculate_total_cost(100.0)
        self.assertEqual(cost_breakdown.bid_amount, 100.0)
        self.assertEqual(cost_breakdown.buyer_premium, 15.0)
        self.assertEqual(cost_breakdown.sales_tax, 8.05)
        self.assertEqual(cost_breakdown.shipping_cost, 10.0)
        self.assertEqual(cost_breakdown.total_cost, 133.05)
    
    def test_calculate_profit(self):
        """Test calculating profit."""
        result = self.calculator.calculate_profit(
            bid_amount=100.0,
            min_sale_price=150.0,
            avg_sale_price=175.0,
            max_sale_price=200.0
        )
        
        # Check cost breakdown
        self.assertEqual(result.cost_breakdown.total_cost, 133.05)
        
        # Check platform fees
        self.assertAlmostEqual(result.platform_fees['min'], 19.125)
        self.assertAlmostEqual(result.platform_fees['avg'], 22.3125)
        self.assertAlmostEqual(result.platform_fees['max'], 25.5)
        
        # Check profits
        self.assertAlmostEqual(result.min_profit, -2.175)
        self.assertAlmostEqual(result.avg_profit, 19.6375)
        self.assertAlmostEqual(result.max_profit, 41.45)
        
        # Check ROI
        self.assertAlmostEqual(result.min_roi, -1.635, places=2)
        self.assertAlmostEqual(result.avg_roi, 14.76, places=2)
        self.assertAlmostEqual(result.max_roi, 31.15, places=2)
    
    def test_calculate_break_even_price(self):
        """Test calculating break-even price."""
        break_even = self.calculator.calculate_break_even_price(100.0)
        self.assertAlmostEqual(break_even, 152.49, places=2)
        
        # Verify that profit is close to zero at break-even price
        result = self.calculator.calculate_profit(
            bid_amount=100.0,
            avg_sale_price=break_even
        )
        self.assertAlmostEqual(result.avg_profit, 0.0, places=2)
    
    def test_calculate_target_bid(self):
        """Test calculating target bid."""
        target_bid = self.calculator.calculate_target_bid(
            target_sale_price=200.0,
            target_profit=40.0
        )
        
        # Verify that the target bid achieves the target profit
        result = self.calculator.calculate_profit(
            bid_amount=target_bid,
            avg_sale_price=200.0
        )
        self.assertAlmostEqual(result.avg_profit, 40.0, places=2)


if __name__ == '__main__':
    unittest.main()