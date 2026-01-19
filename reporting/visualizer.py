import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

class ReportGenerator:
    def __init__(self, backtest_result, weights=None, benchmark=None):
        """
        Initializes the report generator.
        
        Args:
            backtest_result (pd.DataFrame): Series/DataFrame of portfolio value/returns.
            weights (pd.DataFrame): Asset allocation weights over time.
            benchmark (pd.Series): Benchmark returns/value for comparison.
        """
        self.backtest = backtest_result
        self.weights = weights
        self.benchmark = benchmark

    def plot_cumulative_returns(self):
        fig = go.Figure()
        
        # Portfolio
        fig.add_trace(go.Scatter(x=self.backtest.index, y=self.backtest.iloc[:, 0], 
                                 mode='lines', name='Portfolio'))
        
        # Benchmark
        if self.benchmark is not None:
            # Normalize benchmark to start at same level as portfolio
            norm_bench = (self.benchmark / self.benchmark.iloc[0]) * self.backtest.iloc[0, 0]
            fig.add_trace(go.Scatter(x=self.benchmark.index, y=norm_bench, 
                                     mode='lines', name='Benchmark', line=dict(dash='dash')))
            
        fig.update_layout(title='Cumulative Returns',
                          xaxis_title='Date',
                          yaxis_title='Portfolio Value')
        return fig

    def plot_drawdown(self):
        # Calculate Drawdown
        wealth_index = self.backtest.iloc[:, 0]
        previous_peaks = wealth_index.cummax()
        drawdown = (wealth_index - previous_peaks) / previous_peaks
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, 
                                 mode='lines', name='Drawdown', fill='tozeroy', line=dict(color='red')))
        
        fig.update_layout(title='Underwater Plot (Drawdown)',
                          xaxis_title='Date',
                          yaxis_title='Drawdown')
        return fig

    def plot_asset_allocation(self):
        if self.weights is None:
            return go.Figure()
            
        fig = px.area(self.weights, facet_col=None)
        fig.update_layout(title='Asset Allocation over Time',
                          xaxis_title='Date',
                          yaxis_title='Weight')
        return fig

    def generate_html_report(self, filename='portfolio_report.html'):
        fig_ret = self.plot_cumulative_returns()
        fig_dd = self.plot_drawdown()
        fig_weights = self.plot_asset_allocation()
        
        with open(filename, 'w') as f:
            f.write("<html><head><title>Portfolio Performance Report</title></head><body>")
            f.write("<h1>Portfolio Performance Report</h1>")
            
            f.write("<h2>Cumulative Returns</h2>")
            f.write(fig_ret.to_html(full_html=False, include_plotlyjs='cdn'))
            
            f.write("<h2>Drawdown</h2>")
            f.write(fig_dd.to_html(full_html=False, include_plotlyjs='cdn'))
            
            if self.weights is not None:
                f.write("<h2>Asset Allocation</h2>")
                f.write(fig_weights.to_html(full_html=False, include_plotlyjs='cdn'))
                
            f.write("</body></html>")
        
        print(f"Report generated: {os.path.abspath(filename)}")
