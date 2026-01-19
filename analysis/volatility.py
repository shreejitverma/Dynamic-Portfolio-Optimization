import numpy as np
import pandas as pd
from scipy.optimize import minimize

class GARCHModel:
    """
    Implements a GARCH(1,1) model for volatility forecasting using Maximum Likelihood Estimation.
    GARCH(1,1) Variance Equation:
    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
    """

    def __init__(self, returns):
        """
        Args:
            returns (pd.Series or np.array): Series of asset returns (zero mean assumed or residuals).
        """
        self.returns = np.array(returns) * 100 # Scale up for numerical stability
        self.params = None
        self.conditional_variance = None

    def _garch_filter(self, params, returns):
        """
        Calculates the conditional variance path given parameters.
        """
        omega, alpha, beta = params
        n = len(returns)
        sigma2 = np.zeros(n)
        
        # Initialize with sample variance
        sigma2[0] = np.var(returns)
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            
        return sigma2

    def _log_likelihood(self, params, returns):
        """
        Negative Log-Likelihood function for GARCH(1,1).
        Assumes Normal distribution of errors.
        """
        omega, alpha, beta = params
        
        # Constraints enforcement (penalty)
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1:
            return 1e10
            
        sigma2 = self._garch_filter(params, returns)
        
        # Log-Likelihood: -0.5 * sum(log(sigma2) + returns^2 / sigma2)
        # We return Negative for minimization
        ll = 0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
        return ll

    def fit(self):
        """
        Estimates GARCH(1,1) parameters (omega, alpha, beta).
        """
        # Initial guesses: omega ~ var * (1-alpha-beta), alpha=0.05, beta=0.9
        var = np.var(self.returns)
        initial_params = [var * 0.05, 0.05, 0.90]
        
        # Bounds: omega > 0, 0 <= alpha, beta < 1
        bounds = ((1e-6, None), (1e-6, 1), (1e-6, 1))
        
        res = minimize(self._log_likelihood, initial_params, args=(self.returns,),
                       method='L-BFGS-B', bounds=bounds)
        
        self.params = res.x
        self.conditional_variance = self._garch_filter(self.params, self.returns) / 10000 # Scale back down
        
        return {
            'omega': self.params[0],
            'alpha': self.params[1],
            'beta': self.params[2]
        }

    def predict_next_volatility(self):
        """
        Forecasts the next period's annualized volatility.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before prediction.")
            
        omega, alpha, beta = self.params
        last_return = self.returns[-1]
        last_variance = self.conditional_variance[-1] * 10000 # Rescale up
        
        # One-step forecast
        next_variance = omega + alpha * last_return**2 + beta * last_variance
        
        # Convert to daily vol then annualized
        daily_vol = np.sqrt(next_variance) / 100
        annualized_vol = daily_vol * np.sqrt(252)
        
        return annualized_vol
