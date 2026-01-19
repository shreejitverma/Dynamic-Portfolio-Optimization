# Dynamic Portfolio Optimization

**A WorldQuant University Masters Capstone Project**

## Overview

This project develops a sophisticated **dynamic portfolio optimization model** that integrates advanced derivatives, stochastic calculus, and machine learning to manage financial risk in real-time. The system responds dynamically to market fluctuations, interest rate changes (SOFR/SONIA), and currency movements while incorporating hedging strategies for multinational portfolios.

It has been **significantly enhanced and refactored** to follow modern software engineering standards, utilizing modular design patterns (Strategy), robust typing, and the latest Python libraries (Pandas 2.0+, Scikit-Learn 1.3+).

### Authors
- **Farbod Tabatabai** (farbodt2000@gmail.com)
- **Shreejit Verma** (shreejitverma@gmail.com)  
- **Hillary Lulu** (lulumusilu@gmail.com)

### Institution
**WorldQuant University** - Masters in Financial Engineering (MScFE)  
**Course**: MScFE690 - Capstone Project

---

## Key Features

### 1. **Modular Optimization Engine (`portfolio.optimization`)**
- **Strategy Pattern Implementation**: Easily extensible framework for adding new strategies.
- **Hierarchical Risk Parity (HRP)**: Robust allocation using graph theory.
- **Mean-CVaR Optimization**: Minimizes Conditional Value at Risk for tail risk management.
- **Black-Litterman Model**: Combines market equilibrium with investor views.
- **Markowitz Mean-Variance**: Classic efficient frontier analysis (MinVar, Max Sharpe).

### 2. **Advanced Backtesting Framework (`portfolio.backtesting`)**
- **Event-Driven Engine**: Simulates realistic trading environments.
- **Transaction Costs**: Models rebalancing and holding costs.
- **Statistical Significance**: Alpha/Beta analysis with t-stats and p-values.
- **Performance Attribution**: Breakdown of returns by asset contribution.

### 3. **Machine Learning & Analytics (`analysis`)**
- **Volatility Forecasting**: GARCH(1,1) model for dynamic volatility prediction.
- **Return Prediction**: Random Forest models to predict future asset returns.
- **Factor Models**: Fama-French 3-factor analysis for risk attribution.
- **Regime Detection**: Markov Switching models to identify high/low volatility regimes.

### 4. **Data & Visualization**
- **Robust Data Ingestion**: `MarketDataFetcher` supporting Yahoo Finance, FRED, and Alpha Vantage.
- **Interactive Dashboard**: Streamlit-based UI for real-time analysis and backtesting.
- **Real-time Streaming**: Simulates live data feeds.

### 5. **DevOps & Quality**
- **CI/CD Pipeline**: GitHub Actions for automated testing.
- **Dockerized**: Containerized application for easy deployment.
- **Type Hinting**: Extensive use of Python type hints for code clarity.

---

## Project Architecture

```
Dynamic-Portfolio-Optimization/
├── analysis/                      # Analytics & ML
│   ├── factors.py                 # Fama-French Factor Models
│   ├── ml_models.py               # ML Return Predictor
│   ├── regime.py                  # Market Regime Detection
│   └── volatility.py              # GARCH Volatility Model
├── data/                          # Data fetching
│   ├── stock_data_fetcher.py      # Robust Market Data (YF/FRED)
│   └── ...
├── portfolio/                     # Core Logic
│   ├── optimization/              # Optimization Strategies
│   │   ├── base.py                # Abstract Base Class
│   │   ├── mean_variance.py       # MinVar, IVP
│   │   ├── risk_parity.py         # HRP, ERC
│   │   └── advanced.py            # MeanCVaR, BlackLitterman
│   ├── backtesting/               # Backtesting Framework
│   │   ├── engine.py              # Base Engine
│   │   ├── strategies.py          # LongOnly, SignalBased
│   │   └── utils.py               # Utilities
│   ├── performance.py             # Risk Metrics (VaR, ES)
│   ├── execution.py               # Trade Execution Simulator
│   └── optimization_tuning.py     # Grid Search Optimization
├── reporting/                     # Visualization
│   ├── dashboard.py               # Streamlit App
│   └── visualizer.py              # HTML Report Generator
├── tests/                         # Unit Tests
│   └── ...
├── docs/                          # Documentation
└── requirements.txt               # Dependencies
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/shreejitverma/Dynamic-Portfolio-Optimization.git
cd Dynamic-Portfolio-Optimization

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage Examples

#### 1. Interactive Dashboard
Launch the full-featured dashboard to explore strategies:
```bash
streamlit run reporting/dashboard.py
```

#### 2. Optimization (New Modular API)
```python
from portfolio.optimization import BlackLitterman
import pandas as pd

# Load data
returns = pd.read_csv('data/returns.csv', index_col=0, parse_dates=True)
market_caps = {'AAPL': 1e12, 'MSFT': 1.2e12}

# Investor Views: AAPL will return 2%
views = {'AAPL': 0.02}

# Optimize
bl = BlackLitterman(returns, market_caps=market_caps, absolute_views=views)
print("Optimal Weights:", bl.weights)
```

#### 3. Backtesting
```python
from portfolio.backtesting import LongOnlyBacktester

# Initialize Backtester
bt = LongOnlyBacktester(prices_df, weighting_scheme='HRP', rebalance='M')

# Run Simulation
results = bt.run_backtest(backtest_name="HRP_Strategy")
print(results.head())
```

---

## Documentation

Detailed documentation is available in the `docs/` directory:

- [**API Reference**](docs/API_REFERENCE.md): Detailed class and method descriptions for the new modular structure.
- [**Architecture**](docs/ARCHITECTURE.md): System design, patterns, and data flow.
- [**Key Concepts**](docs/KEY_CONCEPTS.md): Explanations of financial models used.
- [**Results**](docs/RESULTS.md): Sample performance metrics and benchmarks.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
