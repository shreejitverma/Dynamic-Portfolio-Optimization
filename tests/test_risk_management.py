import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from portfolio.performance import calculate_var, calculate_es, stress_test_portfolio, get_perf_table_single

class TestRiskManagement(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Generate 1000 daily returns (normal dist) with DatetimeIndex
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        self.returns = pd.Series(np.random.normal(0.0005, 0.02, 1000), index=dates)
        
        # Create a cumulative series for performance table
        self.prices = (1 + self.returns).cumprod()
        
    def test_var_calculation(self):
        print("\nTesting VaR Calculation...")
        var_hist = calculate_var(self.returns, 0.95, method='historical')
        var_param = calculate_var(self.returns, 0.95, method='parametric')
        
        print(f"VaR (Hist): {var_hist:.4f}")
        print(f"VaR (Param): {var_param:.4f}")
        
        self.assertGreater(var_hist, 0)
        self.assertGreater(var_param, 0)
        # Check if they are somewhat close (since data is normal)
        self.assertAlmostEqual(var_hist, var_param, delta=0.005)

    def test_es_calculation(self):
        print("\nTesting ES Calculation...")
        es_hist = calculate_es(self.returns, 0.95, method='historical')
        es_param = calculate_es(self.returns, 0.95, method='parametric')
        
        print(f"ES (Hist): {es_hist:.4f}")
        print(f"ES (Param): {es_param:.4f}")
        
        self.assertGreater(es_hist, 0)
        self.assertGreater(es_param, 0)
        self.assertGreater(es_hist, calculate_var(self.returns, 0.95, method='historical'))

    def test_stress_test(self):
        print("\nTesting Stress Test...")
        weights = pd.Series({'A': 0.6, 'B': 0.4})
        # prices not needed
        scenarios = {
            'Crash': {'A': -0.2, 'B': -0.1},
            'Boom': {'A': 0.1, 'B': 0.05}
        }
        
        results = stress_test_portfolio(weights, scenarios)
        self.assertIn('Crash', results.index)
        self.assertIn('Boom', results.index)
        
        # Expected crash return: 
        # A contributes 0.6 * -0.2 = -0.12
        # B contributes 0.4 * -0.1 = -0.04
        # Total = -0.16
        self.assertAlmostEqual(results.loc['Crash', 'Portfolio Return'], -0.16)

    def test_perf_table_integration(self):
        print("\nTesting Performance Table Integration...")
        df = get_perf_table_single(self.prices)
        self.assertIn('VaR_95', df.index)
        self.assertIn('ES_95', df.index)
        print("Performance Table:\n", df.loc[['sharpe', 'maxDD', 'VaR_95', 'ES_95']])

if __name__ == '__main__':
    unittest.main()
