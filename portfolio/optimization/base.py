from abc import ABC, abstractmethod
import pandas as pd

class OptimizationStrategy(ABC):
    """
    Abstract Base Class for all portfolio optimization strategies.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the strategy with historical data.
        
        Args:
            data (pd.DataFrame): Historical returns or prices.
        """
        self.data = data
        self.weights = None

    @abstractmethod
    def calculate_weights(self) -> pd.Series:
        """
        Calculate and return optimal portfolio weights.
        
        Returns:
            pd.Series: Asset weights indexed by asset names.
        """
        pass
