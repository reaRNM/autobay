"""
Example usage of the Profit Calculator.
"""

import json
from profit_calculator import FeeConfig, ProfitCalculator


def example_basic_profit_calculation():
    """Example of basic profit calculation."""
    print("\n=== Basic Profit Calculation ===")
    
    # Create a profit calculator with default fee configuration
    calculator = ProfitCalculator()
    
    # Calculate profit for a simple auction purchase
    result = calculator.calculate_profit(
        bid_amount=100.0,
        min_sale_price=150.0,
        avg_sale_price=175.0,
        max_sale_price=200.0
    )
    
    # Print the results
    print("Cost Breakdown:")
    for key, value in result.cost_breakdown.format().items():
        if key == 'additional_fees':
            continue
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\nProfit Estimates:")
    print(f"  Minimum: {result.format()['profits']['min']} (ROI: {result.format()['roi']['min']})")
    print(f"  Average: {result.format()['profits']['avg']} (ROI: {result.format()['roi']['avg']})")
    print(f"  Maximum: {result.format()['profits']['max']} (ROI: {result.format()['roi']['max']})")
    
    print("\nBreak-Even Price:")
    break_even = calculator.calculate_break_even_price(bid_amount=100.0)
    print(f"  ${break_even:.2f}")


def example_custom_fee_config():
    """Example of using a custom fee configuration."""
    print("\n=== Custom Fee Configuration ===")
    
    # Create a custom fee configuration
    fee_config = FeeConfig(
        buyer_premium_rate=0.20,  # 20% buyer premium
        sales_tax_rate=0.08,      # 8% sales tax
        platform_fee_rate=0.15,   # 15% platform fee
        shipping_cost=15.0,       # $15 shipping cost
        additional_fees={
            'listing_fee': 0.35,  # $0.35 listing fee
            'payment_processing': 0.30  # $0.30 payment processing fee
        }
    )
    
    # Create a profit calculator with the custom fee configuration
    calculator = ProfitCalculator(fee_config=fee_config)
    
    # Calculate profit for an auction purchase
    result = calculator.calculate_profit(
        bid_amount=200.0,
        min_sale_price=300.0,
        avg_sale_price=350.0,
        max_sale_price=400.0,
        additional_fees={
            'insurance': 5.0  # $5 insurance fee
        }
    )
    
    # Print the results as JSON
    print("Profit Calculation Result:")
    print(json.dumps(result.to_dict(), indent=2))


def example_tiered_platform_fees():
    """Example of using tiered platform fees."""
    print("\n=== Tiered Platform Fees ===")
    
    # Create a fee configuration with tiered platform fees
    fee_config = FeeConfig(
        buyer_premium_rate=0.15,  # 15% buyer premium
        sales_tax_rate=0.07,      # 7% sales tax
        platform_fee_rate=0.15,   # Default 15% platform fee
        use_tiered_platform_fees=True,
        tiered_platform_fees={
            0.0: 0.15,      # 15% for sales up to $100
            100.0: 0.12,    # 12% for sales $100-$500
            500.0: 0.10,    # 10% for sales $500-$1000
            1000.0: 0.08    # 8% for sales over $1000
        }
    )
    
    # Create a profit calculator with the tiered fee configuration
    calculator = ProfitCalculator(fee_config=fee_config)
    
    # Calculate profit for different sale prices
    sale_prices = [50.0, 200.0, 750.0, 1500.0]
    
    for price in sale_prices:
        result = calculator.calculate_profit(
            bid_amount=price * 0.6,  # Bid at 60% of sale price
            avg_sale_price=price
        )
        
        fee_rate = fee_config.get_platform_fee_rate(price)
        platform_fee = result.platform_fees['avg']
        
        print(f"\nSale Price: ${price:.2f}")
        print(f"  Platform Fee Rate: {fee_rate * 100:.1f}%")
        print(f"  Platform Fee: ${platform_fee:.2f}")
        print(f"  Profit: {result.format()['profits']['avg']}")
        print(f"  ROI: {result.format()['roi']['avg']}")


def example_target_bid_calculation():
    """Example of calculating a target bid amount."""
    print("\n=== Target Bid Calculation ===")
    
    # Create a profit calculator
    calculator = ProfitCalculator()
    
    # Calculate target bid for different scenarios
    target_sale_price = 200.0
    target_profits = [20.0, 40.0, 60.0]
    
    print(f"Target Sale Price: ${target_sale_price:.2f}")
    
    for target_profit in target_profits:
        target_bid = calculator.calculate_target_bid(
            target_sale_price=target_sale_price,
            target_profit=target_profit
        )
        
        # Verify the calculation
        result = calculator.calculate_profit(
            bid_amount=target_bid,
            avg_sale_price=target_sale_price
        )
        
        print(f"\nTarget Profit: ${target_profit:.2f}")
        print(f"  Maximum Bid: ${target_bid:.2f}")
        print(f"  Actual Profit: ${result.avg_profit:.2f}")
        print(f"  Total Cost: ${result.cost_breakdown.total_cost:.2f}")


def example_break_even_analysis():
    """Example of break-even analysis."""
    print("\n=== Break-Even Analysis ===")
    
    # Create a profit calculator
    calculator = ProfitCalculator()
    
    # Calculate break-even prices for different bid amounts
    bid_amounts = [50.0, 100.0, 150.0, 200.0]
    
    for bid_amount in bid_amounts:
        break_even = calculator.calculate_break_even_price(bid_amount=bid_amount)
        
        # Calculate the profit at the break-even price
        result = calculator.calculate_profit(
            bid_amount=bid_amount,
            avg_sale_price=break_even
        )
        
        print(f"\nBid Amount: ${bid_amount:.2f}")
        print(f"  Break-Even Price: ${break_even:.2f}")
        print(f"  Total Cost: ${result.cost_breakdown.total_cost:.2f}")
        print(f"  Platform Fee: ${result.platform_fees['avg']:.2f}")
        print(f"  Profit: ${result.avg_profit:.2f}")


def main():
    """Main function."""
    print("Profit Calculator Examples")
    
    # Run examples
    example_basic_profit_calculation()
    example_custom_fee_config()
    example_tiered_platform_fees()
    example_target_bid_calculation()
    example_break_even_analysis()


if __name__ == "__main__":
    main()