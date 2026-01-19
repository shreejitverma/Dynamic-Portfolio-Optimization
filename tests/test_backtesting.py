import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from portfolio.backtesting import FHLongOnlyWeights

class TestBacktesting(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        # Generate random walk prices for 3 assets
        returns = np.random.normal(0.0005, 0.01, (100, 3))
        prices = 100 * (1 + returns).cumprod(axis=0)
        self.ts = pd.DataFrame(prices, index=dates, columns=['A', 'B', 'C'])

    def test_backtest_with_costs(self):
        print("\nTesting Backtest with Costs...")
        
        # 1. Run without costs
        # Create backtest obj. We need minimal params to avoid errors.
        # rebalance='W' so we have some trades.
        bt_no_cost = FHLongOnlyWeights(self.ts, DTINI='2020-01-01', DTEND='2020-04-10', 
                                       rebalance='W', weighting_scheme='EW', static=False)
        res_no_cost = bt_no_cost.run_backtest(backtest_name='NoCost', holdings_costs_bps_pa=0, rebalance_costs_bps=0)
        final_no_cost = res_no_cost.iloc[-1].item()
        
        # 2. Run with costs
        bt_with_cost = FHLongOnlyWeights(self.ts, DTINI='2020-01-01', DTEND='2020-04-10', 
                                         rebalance='W', weighting_scheme='EW', static=False)
        # 10 bps rebalance cost
        res_with_cost = bt_with_cost.run_backtest(backtest_name='WithCost', holdings_costs_bps_pa=0, rebalance_costs_bps=10)
        final_with_cost = res_with_cost.iloc[-1].item()
        
        print(f"Final Value (No Cost): {final_no_cost:.4f}")
        print(f"Final Value (With Cost): {final_with_cost:.4f}")
        
        self.assertLess(final_with_cost, final_no_cost)

    def test_attribution(self):
        print("\nTesting Performance Attribution...")
        bt = FHLongOnlyWeights(self.ts, DTINI='2020-01-01', DTEND='2020-04-10', 
                               rebalance='M', weighting_scheme='EW', static=False)
        bt.run_backtest()
        
        attrib = bt.get_performance_attribution()
        self.assertIsInstance(attrib, pd.DataFrame)
        self.assertEqual(attrib.shape[1], 3) # 3 assets
        print("Attribution tail:\n", attrib.tail())

    def test_significance(self):
        print("\nTesting Statistical Significance...")
        bt = FHLongOnlyWeights(self.ts, DTINI='2020-01-01', DTEND='2020-04-10', 
                               rebalance='M', weighting_scheme='EW', static=False)
        bt.run_backtest()
        
        # Create a dummy benchmark (market returns)
        benchmark = self.ts.mean(axis=1) # Market is average of assets
        
        stats = bt.calculate_significance(benchmark)
        print("Significance Stats:\n", stats)
        
        self.assertIn('Alpha (Ann.)', stats)
        self.assertIn('Beta', stats)
        self.assertIn('Alpha P-Value', stats)

if __name__ == '__main__':
    unittest.main()
