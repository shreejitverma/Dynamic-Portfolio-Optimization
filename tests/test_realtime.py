import unittest
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.stock_data_fetcher import MarketDataFetcher

class TestRealtimeData(unittest.TestCase):
    def setUp(self):
        self.fetcher = MarketDataFetcher()

    def test_fetch_latest_price(self):
        print("\nTesting fetch_latest_price...")
        price = self.fetcher.fetch_latest_price('AAPL')
        print(f"Latest AAPL Price: {price}")
        if price is not None:
            self.assertIsInstance(price, float)
            self.assertGreater(price, 0)
        else:
            print("Warning: Could not fetch price (Market might be closed or API issue).")

    def test_stream_sim(self):
        print("\nTesting stream_latest_prices (2 iterations)...")
        stream = self.fetcher.stream_latest_prices(['AAPL', 'MSFT'], interval_seconds=1)
        
        count = 0
        for data in stream:
            print(f"Stream data: {data}")
            self.assertIn('timestamp', data)
            self.assertIn('prices', data)
            count += 1
            if count >= 2:
                break

if __name__ == '__main__':
    unittest.main()

