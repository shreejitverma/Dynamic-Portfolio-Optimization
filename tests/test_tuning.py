import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from portfolio.backtesting import LongOnlyBacktester
from portfolio.optimization_tuning import BacktestGridSearch

class TestTuning(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=150, freq='D')
        returns = np.random.normal(0.0005, 0.01, (150, 3))
        prices = 100 * (1 + returns).cumprod(axis=0)
        self.ts = pd.DataFrame(prices, index=dates, columns=['A', 'B', 'C'])

    def test_grid_search(self):
        print("\nTesting Backtest Grid Search...")
        
        param_grid = {
            'weighting_scheme': ['IVP', 'EW'],
            'rebalance': ['M', 'W']
        }
        
        fixed_params = {
            'DTINI': '2020-01-01',
            'DTEND': '2020-05-30',
            'static': False
        }
        
        gs = BacktestGridSearch(LongOnlyBacktester, self.ts, param_grid, metric='sharpe')
        gs.run(**fixed_params)
        
        self.assertIsNotNone(gs.best_params)
        self.assertGreater(len(gs.results), 0)
        
        results_df = gs.get_results_df()
        print("\nGrid Search Results:\n", results_df)
        
        self.assertEqual(len(results_df), 4) # 2x2 grid

if __name__ == '__main__':
    unittest.main()
