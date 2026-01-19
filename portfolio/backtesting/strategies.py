import pandas as pd
import numpy as np
from .engine import BacktestEngine
from .utils import BacktestUtils

class LongOnlyBacktester(BacktestEngine):
    """
    Implements long-only portfolio strategies.
    """

    def __init__(self, ts, DTINI='1997-12-31', DTEND='today', static=True,
                 weighting_scheme='IVP', rebalance='M', rescale_weights=False, vol_target=0.1,
                 cov_type='rolling', cov_period=21, cov_window=756, halflife=60):
        super().__init__(ts)
        
        # fill na's and store time series data
        ts = ts.copy().ffill().dropna(how='all')
        ts.index = pd.DatetimeIndex(pd.to_datetime(ts.index))
        relevant_time_period = pd.DatetimeIndex([t for t in ts.index if
                                                 pd.to_datetime(DTINI) <= t <= pd.to_datetime(DTEND)])
        self.ts = ts.loc[relevant_time_period]

        self.rebalance_dates = BacktestUtils.resample_dates(relevant_time_period, rebalance)

        if static:
            try:
                cov = BacktestUtils.get_cov_matrix_on_date(ts.dropna().index[-1], ts, h=cov_period,
                                    cov_type='expanding', cov_window=cov_window, halflife=halflife)
                static_weights = BacktestUtils.static_weights(weighting_scheme, cov, vol_target=vol_target)
            except:
                print(f'{weighting_scheme} not recognized/failed, defaulting to static equal weights')
                static_weights = BacktestUtils.static_weights('EW', cov)
            
            self.weights = BacktestUtils.expand_static_weights(self.rebalance_dates, static_weights)
        else:
            dynamic_weights = pd.DataFrame(index=self.rebalance_dates, columns=ts.columns)
            for r in dynamic_weights.index:
                cov = BacktestUtils.get_cov_matrix_on_date(r, ts, cov_type=cov_type, h=cov_period,
                                                 cov_window=cov_window, halflife=halflife)
                static_weights = BacktestUtils.static_weights(weighting_scheme, cov, vol_target=vol_target)
                dynamic_weights.loc[r] = static_weights.values
            self.weights = dynamic_weights.copy()

        # Rescaling logic could be added here if needed (omitted for brevity in this step)

    def run_backtest(self, backtest_name='backtest', holdings_costs_bps_pa=0, rebalance_costs_bps=0):
        self.backtest = pd.Series(index=self.ts.index, dtype=float)
        self.backtest.iloc[0] = 1.0
        self.pnl = pd.Series(index=self.ts.index, dtype=float)
        self.pnl.iloc[0] = 0.0

        if min(self.weights.index) > min(self.ts.index):
            w0 = pd.DataFrame(columns=[min(self.ts.index)], index=self.weights.columns, data=self.weights.iloc[0].values)
            self.weights = pd.concat([self.weights, w0.T]).sort_index()

        self.holdings = pd.DataFrame(index=self.ts.index, columns=self.ts.columns, dtype=float)
        self.holdings.iloc[0] = self.weights.iloc[0] / self.ts.iloc[0]
        
        self.traded_notional = pd.DataFrame(index=self.ts.index, columns=self.ts.columns, data=0.0)

        tc = rebalance_costs_bps / 10000.0
        hc = holdings_costs_bps_pa / 10000.0
        reb_costs = 0.0

        for t, tm1 in zip(self.backtest.index[1:], self.backtest.index[:-1]):
            prices_t = self.ts.loc[:t].iloc[-1]
            previous_prices = self.ts.loc[:tm1].iloc[-1]
            
            self.pnl[t] = (self.holdings.loc[tm1].copy() * (prices_t - previous_prices)).sum()
            
            # Holding costs
            holdings_costs = (self.holdings.loc[tm1] * previous_prices * hc * (t - tm1).days/365.25).sum()
            
            self.backtest[t] = self.backtest[tm1] + self.pnl[t] - reb_costs - holdings_costs
            reb_costs = 0.0

            if t in self.weights.index:
                # Rebalance
                target_value = self.backtest.loc[tm1] * self.weights.loc[t]
                self.holdings.loc[t] = target_value / self.ts.loc[t]
                
                # Transaction Costs
                traded_val = (np.abs(self.holdings.loc[t] - self.holdings.loc[tm1]) * prices_t).values
                self.traded_notional.loc[t] = traded_val
                reb_costs = np.sum(traded_val * tc)
            else:
                self.holdings.loc[t] = self.holdings.loc[tm1].copy()

        self.backtest = self.backtest.to_frame(backtest_name)
        return self.backtest


class SignalBacktester(BacktestEngine):
    """
    Implements signal-based long-short strategies.
    """
    # (Implementation similar to FHSignalBasedWeights, using BacktestUtils)
    # Keeping it brief for this iteration, but would follow same pattern.
    pass
