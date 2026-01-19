import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from portfolio.construction import MeanCVaR

class TestModelSelection(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100)
        data = np.random.normal(0.001, 0.02, size=(100, 5)) # 5 assets
        self.returns = pd.DataFrame(data, index=dates, columns=['A', 'B', 'C', 'D', 'E'])

    def test_mean_cvar_optimization(self):
        print("\nTesting Mean-CVaR Optimization...")
        optimizer = MeanCVaR(self.returns, alpha=0.95)
        weights = optimizer.weights
        
        self.assertIsInstance(weights, pd.Series)
        print("Weights:\n", weights)
        
        # Check constraints
        self.assertAlmostEqual(weights.sum(), 1.0, places=4)
        self.assertTrue((weights >= -1e-5).all()) # Allow small numerical error
        
        # Check if weights are not just 1/N (unless random data implies that)
        # With random data, they should vary.

if __name__ == '__main__':
    unittest.main()

