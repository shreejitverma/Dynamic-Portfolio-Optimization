# Dynamic Portfolio Optimization

**A WorldQuant University Masters Capstone Project**

## Overview

This project develops a sophisticated **dynamic portfolio optimization model** that integrates advanced derivatives, stochastic calculus, and machine learning to manage financial risk in real-time. The system responds dynamically to market fluctuations, interest rate changes (SOFR/SONIA), and currency movements while incorporating hedging strategies for multinational portfolios.

It has been significantly enhanced to include state-of-the-art features like Mean-CVaR optimization, Black-Litterman models, Machine Learning-based alpha generation, and a comprehensive interactive dashboard.

### Authors
- **Farbod Tabatabai** (farbodt2000@gmail.com)
- **Shreejit Verma** (shreejitverma@gmail.com)  
- **Hillary Lulu** (lulumusilu@gmail.com)

### Institution
**WorldQuant University** - Masters in Financial Engineering (MScFE)  
**Course**: MScFE690 - Capstone Project

---

## Key Features

### 1. **Advanced Optimization Engine**
- **Hierarchical Risk Parity (HRP)**: Robust allocation using graph theory.
- **Mean-CVaR Optimization**: Minimizes Conditional Value at Risk for tail risk management.
- **Black-Litterman Model**: Combines market equilibrium with investor views for posterior return estimation.
- **Markowitz Mean-Variance**: Classic efficient frontier analysis.

### 2. **Comprehensive Risk Management**
- **VaR & Expected Shortfall**: Historical and parametric calculations.
- **Stress Testing**: Scenario analysis for market crashes and booms.
- **Factor Models**: Fama-French 3-factor analysis for risk attribution.
- **Regime Detection**: Markov Switching models to identify high/low volatility regimes.

### 3. **Machine Learning & Analytics**
- **Return Prediction**: Random Forest models to predict future asset returns using technical indicators.
- **Execution Simulation**: Realistic trade execution modeling slippage and market impact.
- **Statistical Significance**: Alpha/Beta analysis with t-stats and p-values.

### 4. **Data & Technology**
- **Real-time Data**: Streaming prices via Yahoo Finance.
- **Parallel Processing**: Multi-core optimization for hyperparameter tuning.
- **Interactive Dashboard**: Streamlit-based UI for backtesting and visualization.

---

## Project Architecture

```
Dynamic-Portfolio-Optimization/
├── analysis/                      # Analytics & ML
│   ├── factors.py                 # Fama-French Factor Models
│   ├── ml_models.py               # ML Return Predictor
│   └── regime.py                  # Market Regime Detection
├── data/                          # Data fetching
│   ├── stock_data_fetcher.py      # Robust Market Data (YF/FRED)
│   └── ...
├── portfolio/                     # Core Logic
│   ├── construction.py            # HRP, Mean-CVaR, Black-Litterman
│   ├── backtesting.py             # Backtest Engine with Costs
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

#### 2. Black-Litterman Optimization
```python
from portfolio.construction import BlackLitterman
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

#### 3. Machine Learning Prediction
```python
from analysis.ml_models import ReturnPredictor

# Train model
predictor = ReturnPredictor(n_estimators=100)
predictor.fit(prices_df)

# Predict next period returns
forecast = predictor.predict(prices_df)
print(forecast)
```

---

## Documentation

Detailed documentation is available in the `docs/` directory:

- [**API Reference**](docs/API_REFERENCE.md): Detailed class and method descriptions.
- [**Architecture**](docs/ARCHITECTURE.md): System design and data flow.
- [**Key Concepts**](docs/KEY_CONCEPTS.md): Explanations of financial models used.
- [**Results**](docs/RESULTS.md): Sample performance metrics and benchmarks.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.