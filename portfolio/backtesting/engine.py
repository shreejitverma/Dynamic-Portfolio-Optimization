import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class BacktestEngine(ABC):
    """
    Abstract Base Class for Backtesting Engines.
    """

    def __init__(self, ts_data: pd.DataFrame):
        self.ts = ts_data
        self.weights = None
        self.backtest = None
        self.pnl = None
        self.holdings = None
        self.traded_notional = None

    @abstractmethod
    def run_backtest(self, backtest_name='backtest', holdings_costs_bps_pa=0, rebalance_costs_bps=0):
        """
        Executes the backtest simulation.
        """
        pass

    def get_performance_attribution(self):
        """
        Calculates simple performance attribution.
        """
        asset_returns = self.ts.pct_change()
        daily_weights = self.weights.reindex(self.ts.index).ffill()
        contribution = daily_weights.shift(1) * asset_returns
        cum_contribution = (1 + contribution).cumprod() - 1
        return cum_contribution

    def calculate_significance(self, benchmark_series):
        """
        Calculates Alpha, Beta, etc.
        """
        if isinstance(benchmark_series, pd.DataFrame):
            benchmark_series = benchmark_series.iloc[:, 0]
            
        strategy_rets = self.backtest.pct_change().dropna()
        
        if benchmark_series.iloc[0] > 2: 
            benchmark_rets = benchmark_series.pct_change().dropna()
        else:
            benchmark_rets = benchmark_series.dropna()
            
        common_index = strategy_rets.index.intersection(benchmark_rets.index)
        y = strategy_rets.loc[common_index].iloc[:, 0]
        x = benchmark_rets.loc[common_index]
        
        # Add constant
        import statsmodels.api as sm
        X = pd.DataFrame({'Benchmark': x, 'Alpha': 1})
        
        try:
            model = sm.OLS(y, X).fit()
            return pd.Series({
                'Alpha (Ann.)': model.params['Alpha'] * 252,
                'Beta': model.params['Benchmark'],
                'Alpha T-Stat': model.tvalues['Alpha'],
                'Alpha P-Value': model.pvalues['Alpha'],
                'R-Squared': model.rsquared
            })
        except Exception as e:
            print(f"Error in significance calc: {e}")
            return None
