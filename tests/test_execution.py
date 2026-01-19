import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from portfolio.execution import ExecutionSimulator

class TestExecution(unittest.TestCase):
    def test_fixed_slippage(self):
        print("\nTesting Fixed Slippage...")
        sim = ExecutionSimulator(fixed_slippage_bps=10, impact_model='linear')
        
        # Buy 100 shares @ $100
        # Slippage = 10 bps = 0.1% = $0.10
        # Exec Price = 100.10
        res = sim.execute_order('AAPL', 'buy', 100, 100.0)
        print("Buy Result:", res)
        
        self.assertAlmostEqual(res['exec_price'], 100.10)
        
        # Sell
        res_sell = sim.execute_order('AAPL', 'sell', 100, 100.0)
        self.assertAlmostEqual(res_sell['exec_price'], 99.90)

    def test_impact_slippage(self):
        print("\nTesting Market Impact (Sqrt)...")
        sim = ExecutionSimulator(fixed_slippage_bps=0, impact_model='sqrt')
        
        # Large order: 10% of volume
        # Impact ~= Price * 0.1 * sqrt(0.1) = 100 * 0.1 * 0.316 = 3.16
        res = sim.execute_order('AAPL', 'buy', 1000, 100.0, volume=10000)
        print("Impact Result:", res)
        
        self.assertGreater(res['exec_price'], 100.0)
        self.assertAlmostEqual(res['slippage_per_share'], 3.162, places=2)

if __name__ == '__main__':
    unittest.main()
