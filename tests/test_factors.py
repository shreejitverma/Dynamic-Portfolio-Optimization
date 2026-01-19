import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.factors import FactorModel

class TestFactors(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Synthetic Asset Returns
        self.returns = pd.Series(np.random.normal(0.0005, 0.012, 100), index=dates, name='Asset')
        
        # Synthetic Factors
        data = {
            'Mkt-RF': np.random.normal(0.0005, 0.01, 100),
            'SMB': np.random.normal(0.0001, 0.005, 100),
            'HML': np.random.normal(0.0001, 0.005, 100),
            'RF': np.random.normal(0.00005, 0.0001, 100)
        }
        self.factors = pd.DataFrame(data, index=dates)

    def test_factor_model_fit(self):
        print("\nTesting Factor Model Fit...")
        fm = FactorModel(self.returns, self.factors)
        fm.fit()
        
        loadings = fm.get_loadings()
        print("Factor Loadings:\n", loadings)
        
        self.assertIn('const', loadings)
        self.assertIn('Mkt-RF', loadings)
        
    def test_attribution(self):
        print("\nTesting Factor Attribution...")
        fm = FactorModel(self.returns, self.factors)
        fm.fit()
        
        attrib = fm.get_attribution()
        self.assertIsInstance(attrib, pd.DataFrame)
        self.assertIn('Specific', attrib.columns)
        self.assertEqual(len(attrib), 100)

if __name__ == '__main__':
    unittest.main()
