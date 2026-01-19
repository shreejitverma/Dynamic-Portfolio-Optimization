import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

class MarketRegimeDetector:
    def __init__(self, returns, n_regimes=2):
        """
        Initializes the Regime Detector.
        
        Args:
            returns (pd.Series): Market returns (e.g., SPY daily returns).
            n_regimes (int): Number of regimes to detect (default 2: Bull/Bear or Low/High Vol).
        """
        self.returns = returns
        self.n_regimes = n_regimes
        self.model = None
        self.results = None

    def fit(self, trend='c', switching_variance=True):
        """
        Fits the Markov Switching Model.
        
        Args:
            trend (str): Trend parameter ('c' for constant, 'nc' for no constant).
            switching_variance (bool): Whether variance switches between regimes.
        """
        # Markov Regression
        # We model returns based on regimes.
        # Often meaningful to model *Volatility* or *Returns*.
        # Simple approach: Model returns with switching mean and variance.
        
        try:
            self.model = MarkovRegression(self.returns, k_regimes=self.n_regimes, 
                                          trend=trend, switching_variance=switching_variance)
            self.results = self.model.fit()
            print(self.results.summary())
        except Exception as e:
            print(f"Error fitting Markov Model: {e}")

    def get_regime_probabilities(self):
        """
        Returns the smoothed probabilities of being in each regime.
        """
        if self.results is None:
            return None
        return self.results.smoothed_marginal_probabilities

    def get_current_regime(self):
        """
        Returns the most likely regime for the last data point.
        """
        probs = self.get_regime_probabilities()
        if probs is None:
            return None
        return probs.iloc[-1].idxmax()
