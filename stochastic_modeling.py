"""
Stochastic Modeling Module
==========================

Implements geometric Brownian motion, Monte Carlo simulations, and risk metrics
for portfolio analysis and option pricing.

Author: Dynamic Portfolio Optimization Team
Institution: WorldQuant University
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion (GBM) asset price simulator.
    
    Models continuous-time stochastic process:
    dS = μS dt + σS dW
    
    where:
        S: Asset price
        μ: Drift (expected return)
        σ: Volatility (annualized)
        dW: Wiener process increment
    """
    
    def __init__(self, S0: float, mu: float, sigma: float, 
                 T: float, steps: int = 252):
        """
        Initialize GBM.
        
        Args:
            S0: Initial asset price
            mu: Drift coefficient (annual return)
            sigma: Volatility coefficient (annual)
            T: Time horizon (years)
            steps: Number of steps per year
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.steps = int(steps * T)
        self.dt = T / self.steps
        
        logger.info(f"GBM initialized: S0=${S0:.2f}, μ={mu:.3f}, σ={sigma:.3f}, T={T}yr")
    
    def simulate_single_path(self, random_state: Optional[int] = None) -> np.ndarray:
        """Simulate a single GBM path."""
        if random_state is not None:
            np.random.seed(random_state)
        
        path = np.zeros(self.steps + 1)
        path[0] = self.S0
        
        for i in range(1, self.steps + 1):
            dW = np.random.normal(0, np.sqrt(self.dt))
            path[i] = path[i-1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * self.dt + 
                self.sigma * dW
            )
        
        return path
    
    def simulate(self, n_simulations: int = 10000, 
                 random_state: Optional[int] = None) -> np.ndarray:
        """
        Simulate multiple GBM paths.
        
        Args:
            n_simulations: Number of paths to simulate
            random_state: Random seed for reproducibility
            
        Returns:
            Array of shape (n_simulations, steps+1) with simulated paths
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        paths = np.zeros((n_simulations, self.steps + 1))
        paths[:, 0] = self.S0
        
        dW = np.random.normal(0, np.sqrt(self.dt), (n_simulations, self.steps))
        
        for i in range(1, self.steps + 1):
            paths[:, i] = paths[:, i-1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * self.dt + 
                self.sigma * dW[:, i-1]
            )
        
        return paths
    
    def calculate_statistics(self, paths: np.ndarray) -> Dict:
        """Calculate statistics of final prices."""
        final_prices = paths[:, -1]
        
        stats_dict = {
            'mean': np.mean(final_prices),
            'std': np.std(final_prices),
            'min': np.min(final_prices),
            'max': np.max(final_prices),
            'median': np.median(final_prices),
            'q25': np.percentile(final_prices, 25),
            'q75': np.percentile(final_prices, 75),
            'skewness': stats.skew(final_prices),
            'kurtosis': stats.kurtosis(final_prices)
        }
        
        return stats_dict
    
    def calculate_var(self, paths: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        VaR is the maximum expected loss at a given confidence level.
        """
        final_prices = paths[:, -1]
        pnl = final_prices - self.S0
        
        var = np.percentile(pnl, (1 - confidence) * 100)
        return var
    
    def calculate_cvar(self, paths: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        CVaR is the expected loss given that loss exceeds VaR.
        """
        final_prices = paths[:, -1]
        pnl = final_prices - self.S0
        var = np.percentile(pnl, (1 - confidence) * 100)
        
        cvar = np.mean(pnl[pnl <= var])
        return cvar
    
    def calculate_return_distribution(self, paths: np.ndarray) -> Dict:
        """Calculate log-return distribution statistics."""
        log_returns = np.log(paths[:, -1] / paths[:, 0])
        
        return {
            'mean_return': np.mean(log_returns),
            'std_return': np.std(log_returns),
            'annualized_return': np.mean(log_returns) / self.T,
            'annualized_vol': np.std(log_returns) / np.sqrt(self.T)
        }
    
    def print_summary(self, paths: np.ndarray):
        """Print detailed summary of simulation results."""
        stats_dict = self.calculate_statistics(paths)
        var_95 = self.calculate_var(paths, 0.95)
        cvar_95 = self.calculate_cvar(paths, 0.95)
        
        print("\n" + "="*60)
        print("GEOMETRIC BROWNIAN MOTION SIMULATION SUMMARY")
        print("="*60)
        print(f"Number of Simulations:  {paths.shape[0]:,}")
        print(f"Time Horizon:           {self.T} years")
        print(f"Time Steps:             {self.steps:,}")
        print(f"Initial Price:          ${self.S0:.2f}")
        print(f"Drift (μ):              {self.mu*100:.2f}%")
        print(f"Volatility (σ):         {self.sigma*100:.2f}%")
        print("-"*60)
        print(f"\nFINAL PRICE DISTRIBUTION:")
        print(f"Mean:                   ${stats_dict['mean']:.2f}")
        print(f"Std Dev:                ${stats_dict['std']:.2f}")
        print(f"Min:                    ${stats_dict['min']:.2f}")
        print(f"25th Percentile:        ${stats_dict['q25']:.2f}")
        print(f"Median:                 ${stats_dict['median']:.2f}")
        print(f"75th Percentile:        ${stats_dict['q75']:.2f}")
        print(f"Max:                    ${stats_dict['max']:.2f}")
        print(f"\nRISK METRICS:")
        print(f"VaR (95%):              ${var_95:.2f}")
        print(f"CVaR (95%):             ${cvar_95:.2f}")
        print(f"Skewness:               {stats_dict['skewness']:.4f}")
        print(f"Kurtosis:               {stats_dict['kurtosis']:.4f}")
        print("="*60 + "\n")


class PortfolioSimulation:
    """
    Simulate correlated asset paths for portfolio analysis.
    """
    
    def __init__(self, assets: List[str], initial_prices: Dict[str, float],
                 mu: Dict[str, float], sigma: Dict[str, float],
                 correlation_matrix: np.ndarray, T: float, steps: int = 252):
        """
        Initialize Portfolio Simulation.
        
        Args:
            assets: List of asset names
            initial_prices: Dict of initial prices
            mu: Dict of expected returns
            sigma: Dict of volatilities
            correlation_matrix: Correlation matrix between assets
            T: Time horizon (years)
            steps: Number of steps per year
        """
        self.assets = assets
        self.initial_prices = initial_prices
        self.mu = mu
        self.sigma = sigma
        self.correlation_matrix = correlation_matrix
        self.T = T
        self.steps = int(steps * T)
        self.dt = T / self.steps
        
        # Cholesky decomposition for correlation
        self.L = np.linalg.cholesky(correlation_matrix)
        
        logger.info(f"Portfolio simulation initialized for {len(assets)} assets")
    
    def simulate_correlated_paths(self, n_simulations: int = 10000) -> Dict:
        """
        Simulate correlated asset paths.
        
        Returns:
            Dict with simulated paths for each asset
        """
        paths = {asset: np.zeros((n_simulations, self.steps + 1)) 
                for asset in self.assets}
        
        # Initialize first prices
        for i, asset in enumerate(self.assets):
            paths[asset][:, 0] = self.initial_prices[asset]
        
        # Generate correlated random numbers
        n_assets = len(self.assets)
        Z = np.random.normal(0, 1, (n_simulations, self.steps, n_assets))
        
        # Apply Cholesky decomposition for correlation
        correlated_Z = np.zeros_like(Z)
        for t in range(self.steps):
            for s in range(n_simulations):
                correlated_Z[s, t, :] = self.L @ Z[s, t, :]
        
        # Generate paths
        for i, asset in enumerate(self.assets):
            for t in range(1, self.steps + 1):
                for s in range(n_simulations):
                    dW = correlated_Z[s, t-1, i] * np.sqrt(self.dt)
                    paths[asset][s, t] = paths[asset][s, t-1] * np.exp(
                        (self.mu[asset] - 0.5 * self.sigma[asset]**2) * self.dt + 
                        self.sigma[asset] * dW
                    )
        
        return paths
    
    def calculate_portfolio_statistics(self, paths: Dict, 
                                      weights: Dict[str, float]) -> Dict:
        """Calculate portfolio statistics."""
        portfolio_value = np.zeros(paths[self.assets[0]].shape[0])
        
        for asset in self.assets:
            portfolio_value += weights[asset] * paths[asset][:, -1]
        
        initial_portfolio = sum(weights[asset] * self.initial_prices[asset] 
                              for asset in self.assets)
        
        returns = (portfolio_value - initial_portfolio) / initial_portfolio
        
        stats_dict = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'var_95': np.percentile(returns, 5),
            'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_drawdown': np.min(returns)
        }
        
        return stats_dict


if __name__ == "__main__":
    # Example: GBM Simulation
    gbm = GeometricBrownianMotion(S0=100, mu=0.08, sigma=0.15, T=1, steps=252)
    paths = gbm.simulate(n_simulations=10000)
    gbm.print_summary(paths)
