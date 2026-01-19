import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.stock_data_fetcher import MarketDataFetcher
from portfolio.backtesting import FHLongOnlyWeights
from portfolio.performance import get_perf_table_single
from analysis.volatility import GARCHModel
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Dynamic Portfolio Optimization Dashboard", layout="wide")

st.title("📊 Dynamic Portfolio Optimization Dashboard")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
tickers_input = st.sidebar.text_input("Enter Tickers (comma separated)", "AAPL, MSFT, GOOG, AMZN, TSLA")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

weighting_scheme = st.sidebar.selectbox("Weighting Scheme", 
                                        ['IVP', 'MVR', 'ERC', 'HRP', 'EW'])

rebalance_freq = st.sidebar.selectbox("Rebalance Frequency", 
                                       ['ME', 'W', 'QE', 'YE'])

run_button = st.sidebar.button("🚀 Run Backtest")

# --- Logic ---
if run_button:
    with st.spinner("Fetching data and running backtest..."):
        try:
            fetcher = MarketDataFetcher()
            # Convert date to string
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch Prices
            prices = fetcher.fetch_prices(tickers, start_str, end_str)
            prices = fetcher.validate_and_clean(prices)
            
            if prices.empty:
                st.error("No data found for the selected tickers/dates.")
            else:
                # Run Backtest
                bt = FHLongOnlyWeights(prices, DTINI=start_str, DTEND=end_str, 
                                       rebalance=rebalance_freq, weighting_scheme=weighting_scheme, static=False)
                backtest_res = bt.run_backtest(backtest_name="Portfolio")
                
                # Metrics
                perf_table = get_perf_table_single(backtest_res.iloc[:, 0])
                
                # --- Layout ---
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Performance Metrics")
                    st.table(perf_table)
                    
                    st.subheader("Volatility Forecast (GARCH)")
                    vol_forecasts = {}
                    for ticker in prices.columns:
                        returns = prices[ticker].pct_change().dropna()
                        if len(returns) > 100:
                            garch = GARCHModel(returns)
                            garch.fit()
                            vol = garch.predict_next_volatility()
                            vol_forecasts[ticker] = f"{vol:.2%}"
                    st.json(vol_forecasts)
                
                with col2:
                    st.subheader("Cumulative Returns")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=backtest_res.index, y=backtest_res.iloc[:, 0], mode='lines', name='Portfolio'))
                    fig.update_layout(xaxis_title="Date", yaxis_title="Value")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                st.subheader("Asset Allocation Over Time")
                weights_df = bt.weights.reindex(prices.index).ffill()
                fig_weights = go.Figure()
                for col in weights_df.columns:
                    fig_weights.add_trace(go.Scatter(x=weights_df.index, y=weights_df[col], 
                                                   stackgroup='one', name=col))
                st.plotly_chart(fig_weights, use_container_width=True)
                
                # Attribution
                st.subheader("Performance Attribution")
                attrib = bt.get_performance_attribution()
                st.line_chart(attrib)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)
else:
    st.info("Configure parameters in the sidebar and click 'Run Backtest' to see results.")
