# API Reference

## Portfolio Construction (`portfolio.construction`)

### `MeanCVaR`
Optimizes portfolio weights to minimize Conditional Value at Risk (CVaR).

- `__init__(data, alpha=0.95)`: Initializes with return data and confidence level.
- `_optimize()`: Solves the convex problem using `cvxpy`.

### `BlackLitterman`
Combines market equilibrium with investor views.

- `__init__(data, market_caps, risk_aversion=2.5, tau=0.05, absolute_views=None)`:
    - `absolute_views`: Dict of {Asset: Return} (e.g., {'AAPL': 0.05}).
- `weights`: The optimal weights based on posterior estimates.

## Analysis & ML (`analysis`)

### `FactorModel` (`analysis.factors`)
Implements Fama-French factor analysis.

- `fit()`: Fits OLS regression of asset returns against factors.
- `get_attribution()`: Returns performance attribution (Factor vs. Specific).

### `ReturnPredictor` (`analysis.ml_models`)
Machine Learning model for return forecasting.

- `fit(prices_df)`: Trains Random Forest on technical indicators (Momentum, Volatility).
- `predict(current_prices_df)`: Predicts next-day returns.

### `MarketRegimeDetector` (`analysis.regime`)
Identifies market regimes (e.g., Bull/Bear) using Markov Switching.

- `fit()`: Fits the Markov model.
- `get_current_regime()`: Returns the current regime state.

## Backtesting (`portfolio.backtesting`)

### `FHLongOnlyWeights`
Long-only backtesting engine.

- `run_backtest(backtest_name, holdings_costs_bps_pa, rebalance_costs_bps)`: Runs the simulation with transaction costs.
- `calculate_significance(benchmark_series)`: Computes Alpha, Beta, T-Stats, and P-Values.

## Reporting (`reporting`)

### `ReportGenerator` (`reporting.visualizer`)
Generates HTML performance reports.

- `generate_html_report(filename)`: Saves interactive Plotly charts to HTML.
