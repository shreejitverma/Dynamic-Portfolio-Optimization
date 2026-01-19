import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List, Optional

class FactorModel:
    """
    Implements Factor Models (e.g., CAPM, Fama-French) for risk analysis and performance attribution.
    """

    def __init__(self, asset_returns: pd.Series, factor_data: Optional[pd.DataFrame] = None):
        """
        Initializes the Factor Model.

        Args:
            asset_returns (pd.Series): Returns of the asset/portfolio to analyze.
            factor_data (pd.DataFrame): DataFrame containing factor returns (e.g., Mkt-RF, SMB, HML).
                                        If None, tries to fetch Fama-French 3 Factors (requires internet).
        """
        self.asset_returns = asset_returns
        self.factor_data = factor_data
        self.results = None
        
        if self.factor_data is None:
             self.factor_data = self._fetch_fama_french()

    def _fetch_fama_french(self):
        """
        Fetches Fama-French 3 Factors using pandas-datareader.
        Falls back to synthetic data if fetch fails (for robustness in environments without full access).
        """
        try:
            import pandas_datareader.data as web
            # Fetch Fama-French 3 Factors
            ff_data = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', 
                                     start=self.asset_returns.index.min(), 
                                     end=self.asset_returns.index.max())[0]
            # Convert to decimal
            return ff_data / 100.0
        except Exception as e:
            print(f"Could not fetch Fama-French data: {e}. Using synthetic factors for demonstration.")
            # Synthetic Factors
            dates = self.asset_returns.index
            n = len(dates)
            data = {
                'Mkt-RF': np.random.normal(0.0005, 0.01, n),
                'SMB': np.random.normal(0.0001, 0.005, n),
                'HML': np.random.normal(0.0001, 0.005, n),
                'RF': np.random.normal(0.00005, 0.0001, n)
            }
            return pd.DataFrame(data, index=dates)

    def fit(self):
        """
        Fits the factor model using OLS regression.
        """
        # Align data
        common_index = self.asset_returns.index.intersection(self.factor_data.index)
        y = self.asset_returns.loc[common_index]
        
        # Subtract RF from asset returns if available to get Excess Returns
        if 'RF' in self.factor_data.columns:
            y = y - self.factor_data.loc[common_index, 'RF']
            X = self.factor_data.loc[common_index].drop(columns=['RF'])
        else:
            X = self.factor_data.loc[common_index]
            
        # Add constant for Alpha
        X = sm.add_constant(X)
        
        self.model = sm.OLS(y, X)
        self.results = self.model.fit()
        
        print(self.results.summary())

    def get_loadings(self):
        """Returns factor loadings (betas) and alpha."""
        if self.results:
            return self.results.params
        return None

    def get_attribution(self):
        """
        Returns attribution: Factor Contribution vs Idiosyncratic Return.
        """
        if self.results is None:
            return None
        
        loadings = self.results.params
        factors = self.results.model.exog  # Includes constant
        factor_names = self.results.model.exog_names
        
        # Contribution = Beta * Factor Return
        attribution = pd.DataFrame(factors * loadings.values, 
                                   index=self.asset_returns.index.intersection(self.factor_data.index), 
                                   columns=factor_names)
        
        # Specific Return (Residuals)
        attribution['Specific'] = self.results.resid
        
        return attribution
