import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.regime import MarketRegimeDetector

class TestRegime(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
        
        # Generate 2 regimes:
        # Regime 0: Low Volatility (Bull)
        # Regime 1: High Volatility (Bear)
        
        ret_low = np.random.normal(0.001, 0.005, 100)
        ret_high = np.random.normal(-0.001, 0.02, 100)
        
        returns = np.concatenate([ret_low, ret_high])
        self.returns = pd.Series(returns, index=dates)

    def test_regime_detection(self):
        print("\nTesting Regime Detection...")
        detector = MarketRegimeDetector(self.returns, n_regimes=2)
        detector.fit()
        
        probs = detector.get_regime_probabilities()
        self.assertIsInstance(probs, pd.DataFrame)
        self.assertEqual(probs.shape[0], 200)
        
        # Check current regime
        current = detector.get_current_regime()
        print(f"Current Regime: {current}")
        self.assertIsNotNone(current)

if __name__ == '__main__':
    unittest.main()
