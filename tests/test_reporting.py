import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reporting.visualizer import ReportGenerator

class TestReporting(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=50, freq='D')
        
        # Backtest Result
        returns = np.random.normal(0.0005, 0.01, 50)
        prices = 100 * (1 + returns).cumprod()
        self.backtest = pd.DataFrame(prices, index=dates, columns=['Portfolio'])
        
        # Weights
        w_data = np.random.dirichlet(alpha=[1,1,1], size=50)
        self.weights = pd.DataFrame(w_data, index=dates, columns=['Asset A', 'Asset B', 'Asset C'])

    def test_generate_report(self):
        print("\nTesting Report Generation...")
        report_file = 'test_report.html'
        
        gen = ReportGenerator(self.backtest, self.weights)
        gen.generate_html_report(report_file)
        
        self.assertTrue(os.path.exists(report_file))
        self.assertGreater(os.path.getsize(report_file), 0)
        
        # Clean up
        if os.path.exists(report_file):
            os.remove(report_file)

if __name__ == '__main__':
    unittest.main()
