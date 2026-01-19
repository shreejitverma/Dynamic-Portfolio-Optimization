# API Reference

## Portfolio Optimization (`portfolio.optimization`)

The optimization module follows the **Strategy Pattern**, where all strategies inherit from `OptimizationStrategy`.

### Base Class
- **`OptimizationStrategy`** (`portfolio.optimization.base`): Abstract base class requiring `calculate_weights()`.

### Strategies

#### Mean-Variance (`portfolio.optimization.mean_variance`)
- **`MinVar`**: Minimizes portfolio variance.
- **`IVP`**: Inverse Variance Portfolio (1/Volatility).

#### Risk Parity (`portfolio.optimization.risk_parity`)
- **`HRP`**: Hierarchical Risk Parity using clustering and recursive bisection.
- **`ERC`**: Equal Risk Contribution optimization.

#### Advanced (`portfolio.optimization.advanced`)
- **`MeanCVaR`**: Convex optimization minimizing Conditional Value at Risk (Expected Shortfall).
- **`BlackLitterman`**: Bayesian model combining market priors with investor views.

## Backtesting (`portfolio.backtesting`)

### Engine (`portfolio.backtesting.engine`)
- **`BacktestEngine`**: Abstract base class for simulation logic.
    - `run_backtest()`: Executes the simulation.
    - `get_performance_attribution()`: Calculates return contribution.
    - `calculate_significance()`: Computes alpha/beta statistics.

### Strategies (`portfolio.backtesting.strategies`)
- **`LongOnlyBacktester`**: Implementation for long-only strategies with rebalancing.
- **`SignalBacktester`**: Implementation for signal-based (long/short) strategies.

### Utilities (`portfolio.backtesting.utils`)
- **`BacktestUtils`**: Helper functions for date resampling and weight expansion.

## Analysis (`analysis`)

### `GARCHModel` (`analysis.volatility`)
- `fit()`: Estimates GARCH(1,1) parameters via MLE.
- `predict_next_volatility()`: Forecasts next-period annualized volatility.

### `FactorModel` (`analysis.factors`)
- `fit()`: Fits Fama-French factor model.
- `get_attribution()`: Returns factor vs. specific return decomposition.

### `ReturnPredictor` (`analysis.ml_models`)
- `fit(prices_df)`: Trains Random Forest on technical features.
- `predict(current_prices_df)`: Generates return forecasts.

### `MarketRegimeDetector` (`analysis.regime`)
- `fit()`: Identifies market regimes (e.g., Bull/Bear) using Markov Switching.

## Reporting (`reporting`)

### `dashboard.py`
Streamlit application for interactive analysis.

### `ReportGenerator` (`reporting.visualizer`)
Generates static HTML reports with Plotly charts.