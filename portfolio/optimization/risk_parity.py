import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from .base import OptimizationStrategy

class HRP(OptimizationStrategy):
    """
    Implements Hierarchical Risk Parity.
    """

    def __init__(self, data: pd.DataFrame, method: str = 'single', metric: str = 'euclidean'):
        super().__init__(data)
        self.method = method
        self.metric = metric
        self.cov = data.cov()
        self.corr = data.corr()
        self.sort_ix = None
        self.link = None
        self.calculate_weights()

    def calculate_weights(self) -> pd.Series:
        self.link = self._tree_clustering(self.corr, self.method, self.metric)
        self.sort_ix = self._get_quasi_diag(self.link)
        self.sort_ix = self.corr.index[self.sort_ix].tolist()  # recover labels
        self.weights = self._get_recursive_bisection(self.cov, self.sort_ix)
        return self.weights

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
            sort_ix = pd.concat([sort_ix, df0])  # item 2
            sort_ix = sort_ix.sort_index()  # re-sort
            sort_ix.index = range(sort_ix.shape[0])  # re-index
        return sort_ix.tolist()

    def _get_recursive_bisection(self, cov, sort_ix):
        w = pd.Series(1, index=sort_ix, name='HRP')
        c_items = [sort_ix]  # initialize all items in one cluster

        while len(c_items) > 0:
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
        sns.clustermap(self.corr, method=self.method, metric=self.metric, cmap=cmap,
                       figsize=figsize, linewidths=linewidth,
                       col_linkage=self.link, row_linkage=self.link)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, pad_inches=1, dpi=400)
        if show_chart:
            plt.show()
        plt.close()


class ERC(OptimizationStrategy):
    """
    Implements Equal Risk Contribution portfolio.
    """

    def __init__(self, data: pd.DataFrame, vol_target: float = 0.10):
        super().__init__(data)
        self.vol_target = vol_target
        self.cov = data.cov()
        self.calculate_weights()

    def calculate_weights(self) -> pd.Series:
        n_assets = self.cov.shape[0]

        def _port_vol(w):
            return np.sqrt(w.dot(self.cov).dot(w))

        def _risk_contribution(w):
            return w * ((w @ self.cov) / (_port_vol(w)**2))

        def _dist_to_target(w):
            return np.abs(_risk_contribution(w) - np.ones(n_assets)/n_assets).sum()

        cons = ({'type': 'ineq',
                 'fun': lambda w: self.vol_target - _port_vol(w)},
                {'type': 'eq',
                 'fun': lambda w: 1 - w.sum()})
        
        w0 = np.zeros(n_assets)
        res = minimize(_dist_to_target, w0, method='SLSQP', constraints=cons)
        self.weights = pd.Series(index=self.cov.columns, data=res.x, name='ERC')
        return self.weights
