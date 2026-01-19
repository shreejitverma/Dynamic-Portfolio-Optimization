import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.volatility import GARCHModel

class TestVolatility(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Generate GARCH(1,1) process
        n = 1000
        omega = 0.00001 * 10000 # Scaled
        alpha = 0.1
        beta = 0.8
        
        returns = np.zeros(n)
        sigma2 = np.zeros(n)
        sigma2[0] = omega / (1 - alpha - beta)
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            returns[t] = np.random.normal(0, np.sqrt(sigma2[t]))
            
        self.returns = returns / 100 # Scale down to raw returns

    def test_garch_fit(self):
        print("\nTesting GARCH Model Fit...")
        model = GARCHModel(self.returns)
        params = model.fit()
        
        print("Estimated Params:", params)
        
        # Check basic constraints
        self.assertGreater(params['alpha'], 0)
        self.assertGreater(params['beta'], 0)
        self.assertLess(params['alpha'] + params['beta'], 1.0) # Stationary

    def test_prediction(self):
        print("\nTesting Volatility Prediction...")
        model = GARCHModel(self.returns)
        model.fit()
        
        vol = model.predict_next_volatility()
        print(f"Predicted Annualized Volatility: {vol:.4f}")
        
        self.assertGreater(vol, 0)

if __name__ == '__main__':
    unittest.main()
