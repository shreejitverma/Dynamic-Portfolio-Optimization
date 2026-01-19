import pandas as pd
import numpy as np
import itertools
from typing import Dict, List, Any, Type
from concurrent.futures import ProcessPoolExecutor, as_completed
from portfolio.performance import get_perf_table_single

def _run_single_backtest(strategy_class, ts_data, params, metric):
    """
    Helper function for parallel execution.
    Must be top-level to be picklable.
    """
    try:
        strat = strategy_class(ts_data, **params)
        backtest_res = strat.run_backtest()
        perf_table = get_perf_table_single(backtest_res.iloc[:, 0])
        score = perf_table.loc[metric].item()
        return {**params, 'score': score}
    except Exception as e:
        print(f"Error with params {params}: {e}")
        return None

class BacktestGridSearch:
    """
    Automates hyperparameter tuning for portfolio backtesting strategies.
    Runs backtests over a grid of parameters and identifies the best combination.
    """

    def __init__(self, strategy_class: Type, ts_data: pd.DataFrame, 
                 param_grid: Dict[str, List[Any]], metric: str = 'sharpe'):
        """
        Initializes the grid search.

        Args:
            strategy_class (Type): The class to instantiate (e.g., LongOnlyBacktester).
            ts_data (pd.DataFrame): Time series data for the backtest.
            param_grid (Dict): Parameters to iterate over. 
                               e.g., {'weighting_scheme': ['IVP', 'ERC'], 'rebalance': ['M', 'W']}
            metric (str): Performance metric to optimize for (from get_perf_table_single).
        """
        self.strategy_class = strategy_class
        self.ts_data = ts_data
        self.param_grid = param_grid
        self.metric = metric
        self.results = []
        self.best_params = None
        self.best_score = -np.inf

    def run(self, max_workers=None, **fixed_params):
        """
        Executes the grid search (parallelized).

        Args:
            max_workers (int): Number of parallel processes.
            **fixed_params: Parameters that are kept constant for all runs.
        """
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"Starting Grid Search over {len(combinations)} combinations...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for params in combinations:
                all_params = {**fixed_params, **params}
                futures.append(
                    executor.submit(_run_single_backtest, self.strategy_class, self.ts_data, all_params, self.metric)
                )
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    self.results.append(result)
                    score = result['score']
                    
                    # Log minimal info (don't print full params to reduce noise in parallel)
                    # print(f"Finished run -> {self.metric}: {score:.4f}")

                    if score > self.best_score:
                        self.best_score = score
                        # Filter out score from params
                        self.best_params = {k: v for k, v in result.items() if k != 'score'}

        print(f"\nOptimization Complete.")
        print(f"Best Params: {self.best_params}")
        print(f"Best {self.metric}: {self.best_score:.4f}")

    def get_results_df(self) -> pd.DataFrame:
        """Returns the grid search results as a DataFrame."""
        return pd.DataFrame(self.results).sort_values(by='score', ascending=False)
