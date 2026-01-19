import unittest
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.stock_data_fetcher import MarketDataFetcher, fetch_stock_data

class TestMarketDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = MarketDataFetcher()
        self.tickers = ['AAPL', 'MSFT']
        self.start_date = '2023-01-01'
        self.end_date = '2023-01-31'

    def test_fetch_prices_yahoo(self):
        print("\nTesting fetch_prices (Yahoo)...")
        df = self.fetcher.fetch_prices(self.tickers, self.start_date, self.end_date)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        # Check if columns are correct (tickers)
        for ticker in self.tickers:
            self.assertIn(ticker, df.columns)
        print("Fetched data shape:", df.shape)

    def test_validate_and_clean(self):
        print("\nTesting validate_and_clean...")
        # Create a dummy DF with NaNs
        dates = pd.date_range(start='2023-01-01', periods=5)
        data = {'A': [1.0, 2.0, float('nan'), 4.0, 5.0], 'B': [1.0, float('nan'), 3.0, 4.0, 5.0]}
        df = pd.DataFrame(data, index=dates)
        
        cleaned_df = self.fetcher.validate_and_clean(df)
        self.assertFalse(cleaned_df.isnull().values.any())
        self.assertEqual(cleaned_df.shape[0], 5) # Should fill NaNs, not drop all of them if ffill works
        # If the first row was NaN, it might be dropped or not depending on method. 
        # Here we have data at index 0, so ffill should work for index 1 and 2.
    
    def test_legacy_wrapper(self):
        print("\nTesting legacy fetch_stock_data wrapper...")
        df = fetch_stock_data(['GOOG'], '2023-01-01', '2023-01-10')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

if __name__ == '__main__':
    unittest.main()
