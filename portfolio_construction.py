"""
Portfolio Construction Module
==============================

Implements Hierarchical Risk Parity (HRP), mean-variance optimization,
and portfolio rebalancing strategies.

Author: Dynamic Portfolio Optimization Team
Institution: WorldQuant University
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioConstruction:
    """
    Portfolio construction using multiple optimization algorithms.
    
    Algorithms:
        - Hierarchical Risk Parity (HRP)
        - Inverse Volatility Weighting
        - Markowitz Mean-Variance Optimization
        - Risk Parity
    """
    
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize Portfolio Construction.
        
        Args:
            returns: DataFrame of historical returns (assets in columns)
        """
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        
        # Calculate statistics
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.corr_matrix = returns.corr()
        self.volatilities = returns.std()
        
        logger.info(f"Portfolio initialized with {self.n_assets} assets")
    
    def calculate_hrp_weights(self) -> np.ndarray:
        """
        Calculate Hierarchical Risk Parity weights.
        
        HRP uses hierarchical clustering on correlations to build a robust
        allocation that's less sensitive to estimation error than traditional
        mean-variance optimization.
        
        Returns:
            Array of weights (one per asset)
        """
        # Step 1: Calculate correlation distance matrix
        corr_dist = np.sqrt(0.5 * (1 - self.corr_matrix))
        
        # Step 2: Hierarchical clustering
        link = linkage(squareform(corr_dist), method='single')
        
        # Step 3: Recursive bisection
        weights = np.ones(self.n_assets)
        nodes = self._tree_clustering(link, corr_dist)
        
        # Step 4: Assign weights recursively
        weights = self._recursive_bisection(link, weights, nodes)
        
        return weights
    
    def _tree_clustering(self, link: np.ndarray, corr_dist: np.ndarray) -> Dict:
        """Build tree structure from hierarchical clustering."""
        nodes = {}
        for i in range(self.n_assets):
            nodes[i] = [i]
        
        for i, row in enumerate(link):
            node_id = self.n_assets + i
            nodes[node_id] = nodes[int(row[0])] + nodes[int(row[1])]
        
        return nodes
    
    def _recursive_bisection(self, link: np.ndarray, weights: np.ndarray,
                            nodes: Dict) -> np.ndarray:
        """Recursively bisect tree and allocate capital."""
        def _bisect(indices: List[int], weights: np.ndarray, link: np.ndarray) -> np.ndarray:
            if len(indices) <= 1:
                return weights
            
            # Find split point
            for i, row in enumerate(link):
                left = set(nodes[int(row[0])])
                right = set(nodes[int(row[1])])
                
                if left.intersection(set(indices)) and right.intersection(set(indices)):
                    left_indices = list(left.intersection(set(indices)))
                    right_indices = list(right.intersection(set(indices)))
                    break
            else:
                left_indices = indices[:len(indices)//2]
                right_indices = indices[len(indices)//2:]
            
            # Allocate inversely proportional to risk
            left_vol = self.volatilities[left_indices].sum()
            right_vol = self.volatilities[right_indices].sum()
            total_vol = left_vol + right_vol
            
            left_weight = right_vol / total_vol if total_vol > 0 else 0.5
            right_weight = left_vol / total_vol if total_vol > 0 else 0.5
            
            for idx in left_indices:
                weights[idx] *= left_weight
            for idx in right_indices:
                weights[idx] *= right_weight
            
            # Recurse
            _bisect(left_indices, weights, link)
            _bisect(right_indices, weights, link)
            
            return weights
        
        indices = list(range(self.n_assets))
        weights = _recursive_bisection(indices, weights, link)
        
        # Normalize
        weights = weights / weights.sum()
        return weights
    
    def calculate_inverse_volatility_weights(self) -> np.ndarray:
        """
        Calculate inverse volatility weights.
        
        Allocate inversely proportional to volatility.
        """
        inv_vol = 1.0 / self.volatilities
        weights = inv_vol / inv_vol.sum()
        return weights.values
    
    def calculate_markowitz_weights(self, target_return: Optional[float] = None,
                                  risk_aversion: float = 2.0,
                                  constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Calculate mean-variance optimal weights.
        
        Solves: min_w (w'Σw - λ w'μ)
        where Σ is covariance matrix and μ is mean returns
        """
        if target_return is None:
            # Use portfolio with risk_aversion parameter
            def portfolio_var(w):
                return w @ self.cov_matrix @ w
            
            def portfolio_return(w):
                return w @ self.mean_returns
            
            def objective(w):
                return portfolio_var(w) - (2 / risk_aversion) * portfolio_return(w)
        else:
            def objective(w):
                return w @ self.cov_matrix @ w
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: w.sum() - 1}  # weights sum to 1
        ]
        
        if target_return is not None:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda w: w @ self.mean_returns - target_return
            })
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints_list)
        
        return result.x if result.success else x0
    
    def calculate_risk_parity_weights(self) -> np.ndarray:
        """
        Calculate risk parity weights.
        
        Each asset contributes equally to portfolio risk (volatility).
        """
        def risk_parity_objective(w):
            portfolio_var = w @ self.cov_matrix @ w
            portfolio_vol = np.sqrt(portfolio_var)
            
            marginal_contrib = self.cov_matrix @ w / portfolio_vol
            risk_contrib = w * marginal_contrib
            target_contrib = portfolio_vol / self.n_assets
            
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(risk_parity_objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def calculate_efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Calculate efficient frontier.
        
        Returns array of (return, volatility) pairs for optimal portfolios.
        """
        frontier = []
        
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max()
        returns = np.linspace(min_return, max_return, n_points)
        
        for target_return in returns:
            try:
                weights = self.calculate_markowitz_weights(target_return)
                port_return = weights @ self.mean_returns
                port_vol = np.sqrt(weights @ self.cov_matrix @ weights)
                sharpe = (port_return - 0.02) / port_vol if port_vol > 0 else 0
                
                frontier.append({
                    'Return': port_return,
                    'Volatility': port_vol,
                    'Sharpe': sharpe
                })
            except:
                pass
        
        return pd.DataFrame(frontier)
    
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict:
        """Calculate portfolio performance metrics."""
        port_return = weights @ self.mean_returns
        port_vol = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe = (port_return - 0.02) / port_vol if port_vol > 0 else 0
        
        # Contribution to risk
        marginal_contrib = self.cov_matrix @ weights / port_vol
        risk_contrib = weights * marginal_contrib
        
        return {
            'return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe,
            'risk_contributions': risk_contrib
        }
    
    def print_portfolio_summary(self, weights: np.ndarray, method: str = "Custom"):
        """Print portfolio summary."""
        metrics = self.calculate_portfolio_metrics(weights)
        
        print("\n" + "="*60)
        print(f"PORTFOLIO SUMMARY ({method})")
        print("="*60)
        print(f"Expected Return:        {metrics['return']*100:.2f}%")
        print(f"Volatility:             {metrics['volatility']*100:.2f}%")
        print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.4f}")
        print("-"*60)
        print(f"\nWEIGHT ALLOCATION:")
        
        for i, asset in enumerate(self.assets):
            print(f"{asset:10s}  {weights[i]*100:6.2f}%  (Risk Contrib: {metrics['risk_contributions'][i]*100:.2f}%)")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # Generate sample returns
    np.random.seed(42)
    dates = pd.date_range('2019-01-01', periods=252, freq='D')
    returns = pd.DataFrame(
        np.random.normal(0.0005, 0.02, (252, 5)),
        columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
        index=dates
    )
    
    # Create portfolio
    portfolio = PortfolioConstruction(returns)
    
    # Calculate weights
    hrp_weights = portfolio.calculate_hrp_weights()
    portfolio.print_portfolio_summary(hrp_weights, "HRP")
