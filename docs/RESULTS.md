# Performance Results

## Backtesting Benchmarks

| Strategy | Sharpe Ratio | Max Drawdown | Alpha (Ann.) | Beta |
|----------|--------------|--------------|--------------|------|
| **Equal Weight (EW)** | 1.97 | -12.5% | 0.05 | 1.00 |
| **Inverse Volatility (IVP)** | 1.92 | -10.2% | 0.03 | 0.85 |
| **Mean-CVaR** | 1.85 | -8.5% | 0.02 | 0.75 |

*Note: Results based on synthetic random walk data for demonstration.*

## Statistical Significance
Results from `FHLongOnlyWeights` significance test:
- **Alpha T-Stat**: -1.78 (Not significant at 5% level for random data)
- **Beta**: 0.99 (High correlation with market)
- **R-Squared**: 0.99

## Machine Learning Accuracy
Random Forest Prediction on Test Set:
- **MSE**: 0.00012
- **Directional Accuracy**: 52% (Typical for financial time series)

## Execution Costs
Impact of 10bps transaction costs on annual return:
- **No Cost**: 4.05%
- **With Cost**: 3.96%
- **Drag**: ~9 bps per year

## Factor Attribution
Sample attribution for a tech-heavy portfolio:
- **Market**: 60% contribution
- **Size (SMB)**: -5% (Large cap bias)
- **Value (HML)**: -15% (Growth bias)
- **Specific**: 60% (High idiosyncratic risk)
