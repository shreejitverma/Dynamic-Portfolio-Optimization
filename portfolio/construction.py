import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.cluster.hierarchy as sch
import cvxpy as cp


class HRP(object):
    """
    Implements Hierarchical Risk Parity
    """

    def __init__(self, data, method='single', metric='euclidean'):
        """
        Combines the assets in `data` using HRP
        returns an object with the following attributes:
            - 'cov': covariance matrix of the returns
            - 'corr': correlation matrix of the returns
            - 'sort_ix': list of sorted column names according to cluster
            - 'link': linkage matrix of size (N-1)x4 with structure Y=[{y_m,1  y_m,2  y_m,3  y_m,4}_m=1,N-1].
                      At the i-th iteration, clusters with indices link[i, 0] and link[i, 1] are combined to form
                      cluster n+1. A cluster with an index less than n corresponds to one of the original observations.
                      The distance between clusters link[i, 0] and link[i, 1] is given by link[i, 2]. The fourth value
                      link[i, 3] represents the number of original observations in the newly formed cluster.
            - 'weights': final weights for each asset

        :param data: pandas DataFrame where each column is a series of returns
        :param method: any method available in scipy.cluster.hierarchy.linkage
        :param metric: any metric available in scipy.cluster.hierarchy.linkage
        """

        assert isinstance(data, pd.DataFrame), "input 'data' must be a pandas DataFrame"

        self.cov = data.cov()
        self.corr = data.corr()
        self.method = method
        self.metric = metric

        self.link = self._tree_clustering(self.corr, self.method, self.metric)
        self.sort_ix = self._get_quasi_diag(self.link)
        self.sort_ix = self.corr.index[self.sort_ix].tolist()  # recover labels
        self.sorted_corr = self.corr.loc[self.sort_ix, self.sort_ix]  # reorder correlation matrix
        self.weights = self._get_recursive_bisection(self.cov, self.sort_ix)
        # TODO self.cluster_nember = sch.fcluster(self.link, t=5, criterion='maxclust')

    @staticmethod
    def _tree_clustering(corr, method, metric):
        dist = np.sqrt(((1 - corr)/2))
        link = sch.linkage(dist, method, metric)
        return link

    @staticmethod
    def _get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0]*2, 2)  # make space
            df0 = sort_ix[sort_ix >= num_items]  # find clusters
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i+1)
            sort_ix = sort_ix.append(df0)  # item 2
            sort_ix = sort_ix.sort_index()  # re-sort
            sort_ix.index = range(sort_ix.shape[0])  # re-index
        return sort_ix.tolist()

    def _get_recursive_bisection(self, cov, sort_ix):
        w = pd.Series(1, index=sort_ix, name='HRP')
        c_items = [sort_ix]  # initialize all items in one cluster
        # c_items = sort_ix

        while len(c_items) > 0:

            # bi-section
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]

            for i in range(0, len(c_items), 2):  # parse in pairs
                c_items0 = c_items[i]  # cluster 1
                c_items1 = c_items[i + 1]  # cluster 2
                c_var0 = self._get_cluster_var(cov, c_items0)
                c_var1 = self._get_cluster_var(cov, c_items1)
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                w[c_items0] *= alpha  # weight 1
                w[c_items1] *= 1 - alpha  # weight 2
        return w

    def _get_cluster_var(self, cov, c_items):
        cov_ = cov.loc[c_items, c_items]  # matrix slice
        w_ = self._get_ivp(cov_).reshape(-1, 1)
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return c_var

    @staticmethod
    def _get_ivp(cov):
        ivp = 1 / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def plot_corr_matrix(self, save_path=None, show_chart=True, cmap='vlag', linewidth=0, figsize=(10, 10)):
        """
        Plots the correlation matrix
        :param save_path: local directory to save file. If provided, saves a png of the image to the address.
        :param show_chart: If True, shows the chart.
        :param cmap: matplotlib colormap.
        :param linewidth: witdth of the grid lines of the correlation matrix.
        :param figsize: tuple with figsize dimensions.
        """

        sns.clustermap(self.corr, method=self.method, metric=self.metric, cmap=cmap,
                       figsize=figsize, linewidths=linewidth,
                       col_linkage=self.link, row_linkage=self.link)

        plt.tight_layout()

        if not (save_path is None):
            plt.savefig(save_path,
                        pad_inches=1,
                        dpi=400)

        if show_chart:
            plt.show()

        plt.close()

    def plot_dendrogram(self, show_chart=True, save_path=None, figsize=(8, 8),
                        threshold=None):
        """
        Plots the dendrogram using scipy's own method.
        :param show_chart: If True, shows the chart.
        :param save_path: local directory to save file.
        :param figsize: tuple with figsize dimensions.
        :param threshold: height of the dendrogram to color the nodes. If None, the colors of the nodes follow scipy's
                           standard behaviour, which cuts the dendrogram on 70% of its height (0.7*max(self.link[:,2]).
        """

        plt.figure(figsize=figsize)
        dn = sch.dendrogram(self.link, orientation='left', labels=self.sort_ix, color_threshold=threshold)

        plt.tight_layout()

        if not (save_path is None):
            plt.savefig(save_path,
                        pad_inches=1,
                        dpi=400)

        if show_chart:
            plt.show()

        plt.close()


class MinVar(object):
    """
    Implements Minimal Variance Portfolio
    """

    def __init__(self, data):
        """
        Combines the assets in 'data' by finding the minimal variance portfolio
        returns an object with the following atributes:
            - 'cov': covariance matrix of the returns
            - 'weights': final weights for each asset

        :param data: pandas DataFrame where each column is a series of returns
        """

        assert isinstance(data, pd.DataFrame), "input 'data' must be a pandas DataFrame"

        self.cov = data.cov()

        eq_cons = {'type': 'eq',
                   'fun': lambda w: w.sum() - 1}

        w0 = np.zeros(self.cov.shape[0])

        res = minimize(self._port_var, w0, method='SLSQP', constraints=eq_cons,
                       options={'ftol': 1e-9, 'disp': False})

        if not res.success:
            raise ArithmeticError('Convergence Failed')

        self.weights = pd.Series(data=res.x, index=self.cov.columns, name='Min Var')

    def _port_var(self, w):
        return w.dot(self.cov).dot(w)


class IVP(object):
    """
    Implements Inverse Variance Portfolio
    """

    def __init__(self, data, use_std=False):
        """
        Combines the assets in 'data' by their inverse variances
        returns an object with the following atributes:
            - 'cov': covariance matrix of the returns
            - 'weights': final weights for each asset

        :param data: pandas DataFrame where each column is a series of returns
        :param use_std: if True, uses the inverse standard deviation. If False, uses the inverse variance.
        """

        assert isinstance(data, pd.DataFrame), "input 'data' must be a pandas DataFrame"
        assert isinstance(use_std, bool), "input 'use_variance' must be boolean"

        self.cov = data.cov()
        w = np.diag(self.cov)

        if use_std:
            w = np.sqrt(w)

        w = 1 / w
        w = w / w.sum()

        self.weights = pd.Series(data=w, index=self.cov.columns, name='IVP')


class ERC(object):
    """
    Implements Equal Risk Contribution portfolio
    """

    def __init__(self, data, vol_target=0.10):
        """
        Combines the assets in 'data' so that all of them have equal contributions to the overall risk of the portfolio.
        Returns an object with the following atributes:
            - 'cov': covariance matrix of the returns
            - 'weights': final weights for each asset

        :param data: pandas DataFrame where each column is a series of returns
        """
        self.cov = data.cov()
        self.vol_target = vol_target
        self.n_assets = self.cov.shape[0]

        cons = ({'type': 'ineq',
                 'fun': lambda w: vol_target - self._port_vol(w)},  # <= 0
                {'type': 'eq',
                 'fun': lambda w: 1 - w.sum()})
        w0 = np.zeros(self.n_assets)
        res = minimize(self._dist_to_target, w0, method='SLSQP', constraints=cons)
        self.weights = pd.Series(index=self.cov.columns, data=res.x, name='ERC')

    def _port_vol(self, w):
        return np.sqrt(w.dot(self.cov).dot(w))

    def _risk_contribution(self, w):
        return w * ((w @ self.cov) / (self._port_vol(w)**2))

    def _dist_to_target(self, w):
        return np.abs(self._risk_contribution(w) - np.ones(self.n_assets)/self.n_assets).sum()


class MeanCVaR(object):
    """
    Implements Mean-CVaR (Conditional Value at Risk) Optimization.
    Minimizes CVaR at a given confidence level.
    """

    def __init__(self, data, alpha=0.95):
        """
        Initializes the Mean-CVaR portfolio optimization.

        :param data: pandas DataFrame where each column is a series of returns (rows are time periods).
        :param alpha: Confidence level for CVaR (e.g., 0.95 means worst 5%).
        """
        assert isinstance(data, pd.DataFrame), "input 'data' must be a pandas DataFrame"
        
        self.returns = data.values
        self.assets = data.columns
        self.n_assets = len(self.assets)
        self.n_samples = self.returns.shape[0]
        self.alpha = alpha

        self.weights = self._optimize()

    def _optimize(self):
        # Variables
        w = cp.Variable(self.n_assets)
        VaR = cp.Variable()
        z = cp.Variable(self.n_samples)

        # Objective: Minimize CVaR
        # CVaR = VaR + (1 / ((1 - alpha) * T)) * sum(z)
        objective = cp.Minimize(VaR + (1 / ((1 - self.alpha) * self.n_samples)) * cp.sum(z))

        # Constraints
        constraints = [
            z >= 0,
            z >= -self.returns @ w - VaR,  # Loss exceeds VaR
            cp.sum(w) == 1,
            w >= 0  # No short selling constraint
        ]

        # Problem
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve()
            if prob.status == cp.OPTIMAL:
                return pd.Series(data=w.value, index=self.assets, name='Mean-CVaR')
            else:
                print(f"Optimization failed: {prob.status}")
                return None
        except Exception as e:
            print(f"Error in CVaR optimization: {e}")
            return None


class BlackLitterman(object):
    """
    Implements the Black-Litterman Model for portfolio optimization.
    Combines market equilibrium returns (prior) with investor views to produce
    posterior expected returns and covariance.
    """

    def __init__(self, data, market_caps=None, risk_aversion=2.5, tau=0.05, 
                 absolute_views=None, view_confidences=None):
        """
        Initializes the Black-Litterman model.

        :param data: pandas DataFrame of asset returns.
        :param market_caps: pandas Series or dict of market capitalizations (for equilibrium weights).
                            If None, assumes equal weights (1/N) as the market prior (simplified).
        :param risk_aversion: float, risk aversion coefficient (delta).
        :param tau: float, uncertainty scaling factor for the prior.
        :param absolute_views: dict, investor views {Asset: Expected_Return}. e.g. {'AAPL': 0.10}
        :param view_confidences: dict, confidence in views (diagonal elements of Omega). 
                                 If None, heuristics are used.
        """
        self.data = data
        self.assets = data.columns
        self.n_assets = len(self.assets)
        self.cov = data.cov()
        self.delta = risk_aversion
        self.tau = tau
        
        # 1. Market Equilibrium (Prior)
        if market_caps is not None:
            if isinstance(market_caps, dict):
                market_caps = pd.Series(market_caps)
            market_weights = market_caps / market_caps.sum()
        else:
            # Fallback to Equal Weights if no caps provided
            market_weights = pd.Series(1/self.n_assets, index=self.assets)
            
        self.market_weights = market_weights.reindex(self.assets).fillna(0)
        
        # Calculate Equilibrium Returns (Pi)
        # Pi = delta * Sigma * w_market
        self.pi = self.delta * self.cov.dot(self.market_weights)
        
        # 2. Views (P and Q matrices)
        self.P, self.Q, self.Omega = self._process_views(absolute_views, view_confidences)
        
        # 3. Posterior Calculations
        self.posterior_rets, self.posterior_cov = self._calculate_posterior()
        
        # 4. Optimal Weights based on Posterior
        # w* = (lambda * Sigma)^-1 * mu
        # Using Max Sharpe (approx) or unconstrained mean-variance
        # Here we return weights for Max Sharpe given posterior params
        self.weights = self._optimize_weights()

    def _process_views(self, views, confidences):
        """
        Constructs P (picking matrix), Q (view vector), and Omega (uncertainty matrix).
        """
        if not views:
            return None, None, None
            
        k = len(views)
        P = np.zeros((k, self.n_assets))
        Q = np.zeros(k)
        Omega = np.zeros((k, k))
        
        for i, (asset, ret) in enumerate(views.items()):
            if asset in self.assets:
                col_idx = self.assets.get_loc(asset)
                P[i, col_idx] = 1
                Q[i] = ret
                
                # Confidence / Uncertainty (Omega)
                # If not provided, Heuristic: tau * p * Sigma * p.T
                if confidences and asset in confidences:
                    Omega[i, i] = confidences[asset]
                else:
                    # Heuristic
                    p_vec = P[i, :].reshape(1, -1)
                    Omega[i, i] = self.tau * (p_vec @ self.cov.values @ p_vec.T).item()
                    
        return P, Q, Omega

    def _calculate_posterior(self):
        """
        Calculates posterior expected returns and covariance.
        """
        if self.P is None:
            # No views -> Posterior = Prior
            return self.pi, self.cov + self.tau * self.cov
            
        # BL Formulas
        # mu_bl = [(tau*Sigma)^-1 + P.T * Omega^-1 * P]^-1 * [(tau*Sigma)^-1 * pi + P.T * Omega^-1 * Q]
        
        tau_sigma = self.tau * self.cov
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(self.Omega)
        
        # M = (tau*Sigma)^-1 + P.T * Omega^-1 * P
        M = tau_sigma_inv + self.P.T @ omega_inv @ self.P
        M_inv = np.linalg.inv(M)
        
        # RHS = (tau*Sigma)^-1 * pi + P.T * Omega^-1 * Q
        rhs = tau_sigma_inv @ self.pi + self.P.T @ omega_inv @ self.Q
        
        posterior_rets = M_inv @ rhs
        
        # Posterior Covariance
        # Sigma_bl = Sigma + M^-1
        posterior_cov = self.cov + M_inv
        
        return pd.Series(posterior_rets, index=self.assets), pd.DataFrame(posterior_cov, index=self.assets, columns=self.assets)

    def _optimize_weights(self):
        """
        Computes optimal weights using posterior estimates.
        Unconstrained Mean-Variance: w = (delta * Sigma)^-1 * mu
        Normalized to sum to 1.
        """
        # w = (delta * Sigma_post)^-1 * mu_post
        sigma_inv = np.linalg.inv(self.posterior_cov)
        raw_weights = sigma_inv @ self.posterior_rets
        
        # Normalize
        normalized_weights = raw_weights / raw_weights.sum()
        
        return pd.Series(normalized_weights, index=self.assets, name='Black-Litterman')
