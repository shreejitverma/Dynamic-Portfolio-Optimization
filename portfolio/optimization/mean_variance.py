import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .base import OptimizationStrategy

class MinVar(OptimizationStrategy):
    """
    Implements Minimal Variance Portfolio.
    """

    def calculate_weights(self) -> pd.Series:
        cov = self.data.cov()
        n_assets = cov.shape[0]

        def _port_var(w):
            return w.dot(cov).dot(w)

        eq_cons = {'type': 'eq',
                   'fun': lambda w: w.sum() - 1}
        
        # Bounds: Long only (0, 1)
        bounds = tuple((0, 1) for _ in range(n_assets))

        w0 = np.ones(n_assets) / n_assets

        res = minimize(_port_var, w0, method='SLSQP', constraints=eq_cons, bounds=bounds,
                       options={'ftol': 1e-9, 'disp': False})

        if not res.success:
            raise ArithmeticError('Convergence Failed')

        self.weights = pd.Series(data=res.x, index=cov.columns, name='Min Var')
        return self.weights

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.calculate_weights()


class IVP(OptimizationStrategy):
    """
    Implements Inverse Variance Portfolio.
    """

    def __init__(self, data: pd.DataFrame, use_std: bool = False):
        super().__init__(data)
        self.use_std = use_std
        self.calculate_weights()

    def calculate_weights(self) -> pd.Series:
        cov = self.data.cov()
        w = np.diag(cov)

        if self.use_std:
            w = np.sqrt(w)

        w = 1 / w
        w = w / w.sum()

        self.weights = pd.Series(data=w, index=cov.columns, name='IVP')
        return self.weights
