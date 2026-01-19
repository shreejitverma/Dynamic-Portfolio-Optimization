import os
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataFetcher:
    """
    A robust data fetcher for financial market data and economic indicators.
    Supports:
    - Yahoo Finance (via yfinance)
    - FRED (Federal Reserve Economic Data) (via fredapi)
    - Alpha Vantage (placeholder/structure)
    """

    def __init__(self, fred_api_key: Optional[str] = None, av_api_key: Optional[str] = None):
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.av_api_key = av_api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        
        # Initialize FRED client if key is available
        self.fred = None
        if self.fred_api_key:
            try:
                from fredapi import Fred
                self.fred = Fred(api_key=self.fred_api_key)
            except ImportError:
                logger.warning("fredapi not installed. FRED data fetching will be limited.")

    def fetch_prices(self, tickers: List[str], start_date: str, end_date: str, source: str = 'yahoo', max_retries: int = 3) -> pd.DataFrame:
        """
        Fetches historical price data with retries and validation.
        """
        for attempt in range(max_retries):
            try:
                if source == 'yahoo':
                    return self._fetch_yahoo(tickers, start_date, end_date)
                elif source == 'alpha_vantage':
                    return self._fetch_alpha_vantage(tickers, start_date, end_date)
                else:
                    raise ValueError(f"Unknown source: {source}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed for {source}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to fetch data from {source} after {max_retries} attempts.")
        return pd.DataFrame()

    def _fetch_yahoo(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Internal method to fetch from Yahoo Finance."""
        logger.info(f"Fetching {len(tickers)} tickers from Yahoo Finance...")
        # yf.download handles multi-threading automatically
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            logger.warning("No data returned from Yahoo Finance.")
            return data

        # Handle MultiIndex columns (Price Type, Ticker) -> just return Adj Close
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.get_level_values(0):
                 data = data['Adj Close']
            elif 'Close' in data.columns.get_level_values(0):
                 data = data['Close'] # Fallback if Adj Close is missing (rare)

        # Ensure we have a DataFrame with tickers as columns
        if isinstance(data, pd.Series):
             data = data.to_frame(name=tickers[0])
             
        return data

    def _fetch_alpha_vantage(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Internal method for Alpha Vantage (Placeholder)."""
        if not self.av_api_key:
            raise ValueError("Alpha Vantage API key not provided.")
        logger.info("Alpha Vantage fetching not fully implemented. Returning empty DF.")
        # Implementation would go here using requests or alpha_vantage library
        return pd.DataFrame()

    def fetch_economic_data(self, series_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.Series:
        """
        Fetches economic data from FRED.
        """
        if self.fred:
            try:
                logger.info(f"Fetching {series_id} from FRED...")
                data = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                return data
            except Exception as e:
                logger.error(f"Error fetching from FRED: {e}")
                return pd.Series()
        else:
            logger.warning("FRED API key not found. Cannot fetch economic data via API.")
            return pd.Series()

    def fetch_latest_price(self, ticker: str) -> Optional[float]:
        """
        Fetches the latest available price for a ticker.
        """
        try:
            stock = yf.Ticker(ticker)
            # Fetch 1-day, 1-minute interval data for the most recent price
            data = stock.history(period='1d', interval='1m')
            if not data.empty:
                return data['Close'].iloc[-1]
            return None
        except Exception as e:
            logger.error(f"Error fetching latest price for {ticker}: {e}")
            return None

    def stream_latest_prices(self, tickers: List[str], interval_seconds: int = 60):
        """
        A generator that yields latest prices at specified intervals.
        (Simulated streaming)
        """
        logger.info(f"Starting real-time price stream for {tickers}...")
        while True:
            prices = {}
            for ticker in tickers:
                price = self.fetch_latest_price(ticker)
                if price:
                    prices[ticker] = price
            
            yield {
                'timestamp': datetime.now(),
                'prices': prices
            }
            time.sleep(interval_seconds)

    def validate_and_clean(self, df: pd.DataFrame, fill_method: str = 'ffill') -> pd.DataFrame:
        """
        Validates and cleans the data:
        1. Checks for missing values.
        2. Fills missing values (forward fill by default).
        3. Drops remaining NaNs.
        4. Checks for potential outliers (e.g., 0 prices).
        """
        if df.empty:
            return df

        initial_shape = df.shape
        
        # Replace zeros with NaN (assuming price cannot be 0)
        df = df.replace(0, np.nan)

        # Fill missing values
        if fill_method == 'ffill':
            df = df.ffill()
        elif fill_method == 'bfill':
            df = df.bfill()
        elif fill_method == 'interpolate':
            df = df.interpolate(method='time')

        # Drop rows that are still NaN (e.g., at the start)
        df = df.dropna()

        # Check for outliers (simple check: return > 50% in a day?)
        # For now, just logging
        pct_change = df.pct_change().dropna()
        if (pct_change.abs() > 0.5).any().any():
             logger.warning("Potential outliers detected (daily return > 50%).")

        final_shape = df.shape
        logger.info(f"Data cleaned. Shape changed from {initial_shape} to {final_shape}")
        
        return df

# --- Wrapper functions for backward compatibility ---

def fetch_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Wrapper for MarketDataFetcher.fetch_prices"""
    fetcher = MarketDataFetcher()
    df = fetcher.fetch_prices(tickers, start_date, end_date)
    return fetcher.validate_and_clean(df)

def save_stock_data(data: pd.DataFrame, filepath: str):
    """Saves the stock data to a CSV file."""
    try:
        data.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")

def load_stock_data(filepath: str) -> pd.DataFrame:
    """Loads stock data from a CSV file."""
    try:
        if os.path.exists(filepath):
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return data
        else:
            logger.error(f"File not found: {filepath}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()