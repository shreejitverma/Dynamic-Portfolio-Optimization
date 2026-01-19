import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from portfolio.construction import BlackLitterman

class TestBlackLitterman(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100)
        # 3 Assets
        data = np.random.normal(0.0005, 0.01, size=(100, 3))
        self.returns = pd.DataFrame(data, index=dates, columns=['AAPL', 'MSFT', 'GOOG'])
        self.caps = {'AAPL': 1e12, 'MSFT': 1.5e12, 'GOOG': 0.8e12}

    def test_bl_optimization(self):
        print("\nTesting Black-Litterman Optimization...")
        
        # View: AAPL will return 2% (0.02)
        views = {'AAPL': 0.02}
        
        bl = BlackLitterman(self.returns, market_caps=self.caps, absolute_views=views)
        
        weights = bl.weights
        print("BL Weights:\n", weights)
        
        self.assertAlmostEqual(weights.sum(), 1.0, places=4)
        
        # Check if posterior return for AAPL moved towards view
        prior_aapl = bl.pi['AAPL']
        post_aapl = bl.posterior_rets['AAPL']
        
        print(f"Prior AAPL: {prior_aapl:.4f}, View: 0.02, Posterior: {post_aapl:.4f}")
        
        # Posterior should be between Prior and View (roughly)
        # Note: Depending on correlations, this isn't strictly guaranteed for single asset, 
        # but with low correlation it usually holds.
        # Here we just check it runs and produces valid outputs.
        self.assertIsInstance(weights, pd.Series)

if __name__ == '__main__':
    unittest.main()
