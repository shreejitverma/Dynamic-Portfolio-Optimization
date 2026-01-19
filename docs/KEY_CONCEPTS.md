# Key Financial Concepts

## 1. Mean-CVaR Optimization
Unlike Mean-Variance optimization which penalizes all volatility equally, **Mean-CVaR** focuses on the "bad" tail.
- **VaR (Value at Risk)**: The maximum loss with X% confidence.
- **CVaR (Conditional VaR)**: The expected loss *given* that the loss exceeds VaR.
- **Why use it?** It is robust to non-normal return distributions (fat tails).

## 2. Black-Litterman Model
A Bayesian approach to portfolio construction.
- **Prior**: The Market Equilibrium portfolio (Implied returns from market caps).
- **Views**: Investor's subjective views (e.g., "Tech will outperform by 2%").
- **Posterior**: A weighted average of the Prior and Views, adjusted by confidence levels.
- **Benefit**: Produces stable, intuitive weights compared to the extreme allocations often seen in Markowitz optimization.

## 3. Hierarchical Risk Parity (HRP)
Uses machine learning (clustering) to allocate risk.
1.  **Clustering**: Group assets that move together.
2.  **Recursive Bisection**: Split allocations top-down based on cluster volatility.
3.  **Benefit**: Does not require inverting a covariance matrix, making it robust to noise and highly correlated assets.

## 4. Fama-French Factor Model
Explains returns using 3 factors:
1.  **Market Risk (Mkt-RF)**: Beta exposure.
2.  **Size (SMB)**: Small caps tend to outperform large caps.
3.  **Value (HML)**: Value stocks tend to outperform growth stocks.

## 5. Market Regime Detection
Markets behave differently in "calm" vs "crisis" periods. We use **Markov Switching Models** to infer the hidden state (Regime) based on observed volatility and returns.
