import numpy as np
import pandas as pd
import cvxpy as cp
from .base import OptimizationStrategy

class MeanCVaR(OptimizationStrategy):
    """
    Implements Mean-CVaR (Conditional Value at Risk) Optimization.
    """

    def __init__(self, data: pd.DataFrame, alpha: float = 0.95):
        super().__init__(data)
        self.alpha = alpha
        self.calculate_weights()

    def calculate_weights(self) -> pd.Series:
        returns = self.data.values
        assets = self.data.columns
        n_assets = len(assets)
        n_samples = returns.shape[0]

        # Variables
        w = cp.Variable(n_assets)
        VaR = cp.Variable()
        z = cp.Variable(n_samples)

        # Objective: Minimize CVaR
        objective = cp.Minimize(VaR + (1 / ((1 - self.alpha) * n_samples)) * cp.sum(z))

        # Constraints
        constraints = [
            z >= 0,
            z >= -returns @ w - VaR,
            cp.sum(w) == 1,
            w >= 0  # Long only
        ]

        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve()
            if prob.status == cp.OPTIMAL:
                self.weights = pd.Series(data=w.value, index=assets, name='Mean-CVaR')
                return self.weights
            else:
                print(f"Optimization failed: {prob.status}")
                return None
        except Exception as e:
            print(f"Error in CVaR optimization: {e}")
            return None


class BlackLitterman(OptimizationStrategy):
    """
    Implements the Black-Litterman Model.
    """

    def __init__(self, data: pd.DataFrame, market_caps=None, risk_aversion=2.5, tau=0.05, 
                 absolute_views=None, view_confidences=None):
        super().__init__(data)
        self.market_caps = market_caps
        self.delta = risk_aversion
        self.tau = tau
        self.views = absolute_views
        self.confidences = view_confidences
        self.calculate_weights()

    def calculate_weights(self) -> pd.Series:
        assets = self.data.columns
        n_assets = len(assets)
        cov = self.data.cov()
        
        # 1. Market Equilibrium (Prior)
        if self.market_caps is not None:
            if isinstance(self.market_caps, dict):
                caps = pd.Series(self.market_caps)
            else:
                caps = self.market_caps
            market_weights = caps / caps.sum()
        else:
            market_weights = pd.Series(1/n_assets, index=assets)
            
        market_weights = market_weights.reindex(assets).fillna(0)
        
        # Equilibrium Returns (Pi)
        self.pi = self.delta * cov.dot(market_weights)
        
        # 2. Views
        P, Q, Omega = self._process_views(n_assets, assets, cov)
        
        # 3. Posterior
        self.posterior_rets, self.posterior_cov = self._calculate_posterior(self.pi, cov, P, Q, Omega)
        
        # 4. Optimal Weights (Max Sharpe on Posterior)
        sigma_inv = np.linalg.inv(self.posterior_cov)
        raw_weights = sigma_inv @ self.posterior_rets
        weights_array = raw_weights / raw_weights.sum()
        
        self.weights = pd.Series(weights_array, index=assets, name='Black-Litterman')
        
        return self.weights

    def _process_views(self, n_assets, assets, cov):
        if not self.views:
            return None, None, None
            
        k = len(self.views)
        P = np.zeros((k, n_assets))
        Q = np.zeros(k)
        Omega = np.zeros((k, k))
        
        for i, (asset, ret) in enumerate(self.views.items()):
            if asset in assets:
                col_idx = assets.get_loc(asset)
                P[i, col_idx] = 1
                Q[i] = ret
                
                if self.confidences and asset in self.confidences:
                    Omega[i, i] = self.confidences[asset]
                else:
                    p_vec = P[i, :].reshape(1, -1)
                    Omega[i, i] = self.tau * (p_vec @ cov.values @ p_vec.T).item()
                    
        return P, Q, Omega

    def _calculate_posterior(self, pi, cov, P, Q, Omega):
        if P is None:
            return pi, cov + self.tau * cov
            
        tau_sigma = self.tau * cov
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)
        
        M = tau_sigma_inv + P.T @ omega_inv @ P
        M_inv = np.linalg.inv(M)
        
        rhs = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
        
        posterior_rets = M_inv @ rhs
        posterior_cov = cov + M_inv
        
        return pd.Series(posterior_rets, index=self.data.columns), pd.DataFrame(posterior_cov, index=self.data.columns, columns=self.data.columns)
