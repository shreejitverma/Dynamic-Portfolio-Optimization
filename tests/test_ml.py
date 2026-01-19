import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.ml_models import ReturnPredictor

class TestML(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        
        # Synthetic Random Walk Prices
        returns = np.random.normal(0.0005, 0.01, (500, 3))
        prices = 100 * (1 + returns).cumprod(axis=0)
        self.prices = pd.DataFrame(prices, index=dates, columns=['Asset A', 'Asset B', 'Asset C'])

    def test_evaluation(self):
        print("\nTesting ML Model Evaluation...")
        predictor = ReturnPredictor(n_estimators=10)
        metrics = predictor.evaluate(self.prices, test_size=0.2)
        print("Evaluation Metrics:\n", metrics)
        
        self.assertIn('MSE', metrics)
        self.assertIn('R2', metrics)
        # R2 will likely be negative for random walk

    def test_prediction(self):
        print("\nTesting Prediction...")
        predictor = ReturnPredictor(n_estimators=10)
        predictor.fit(self.prices)
        
        preds = predictor.predict(self.prices)
        print("Predictions:\n", preds)
        
        self.assertEqual(len(preds), 3)
        self.assertFalse(preds.isnull().any())

if __name__ == '__main__':
    unittest.main()
