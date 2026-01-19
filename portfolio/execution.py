import numpy as np
import pandas as pd

class ExecutionSimulator:
    """
    Simulates trade execution with slippage and transaction costs.
    """

    def __init__(self, fixed_slippage_bps=5.0, impact_model='linear'):
        """
        Args:
            fixed_slippage_bps (float): Fixed slippage in basis points (1 bp = 0.01%).
            impact_model (str): 'linear' or 'sqrt' (market impact model).
        """
        self.fixed_slippage = fixed_slippage_bps / 10000.0
        self.impact_model = impact_model

    def calculate_slippage(self, quantity, price, volume=None, volatility=None):
        """
        Calculates price impact/slippage.

        Args:
            quantity (float): Number of shares/units.
            price (float): Current market price.
            volume (float): Daily/Interval volume (required for impact models).
            volatility (float): Asset volatility (optional, for advanced models).

        Returns:
            float: Slippage cost per share (absolute value).
        """
        # Base fixed slippage
        slippage = price * self.fixed_slippage
        
        # Market Impact
        if self.impact_model == 'sqrt' and volume is not None and volume > 0:
            # Square Root Law of Market Impact
            # Impact ~= sigma * (Q / V)^0.5
            # Simplified constant coefficient for now
            participation_rate = abs(quantity) / volume
            impact = price * 0.1 * np.sqrt(participation_rate)
            slippage += impact
            
        return slippage

    def execute_order(self, ticker, side, quantity, price, **kwargs):
        """
        Simulates execution of an order.

        Args:
            ticker (str): Asset symbol.
            side (str): 'buy' or 'sell'.
            quantity (float): Quantity to trade.
            price (float): Reference market price (e.g., mid or last).
            **kwargs: Additional data like 'volume', 'volatility'.

        Returns:
            dict: Execution details {avg_price, cost, slippage}
        """
        slippage_per_share = self.calculate_slippage(quantity, price, **kwargs)
        
        if side.lower() == 'buy':
            exec_price = price + slippage_per_share
        else: # sell
            exec_price = price - slippage_per_share
            
        total_consideration = exec_price * quantity
        
        return {
            'ticker': ticker,
            'side': side,
            'quantity': quantity,
            'ref_price': price,
            'exec_price': exec_price,
            'slippage_per_share': slippage_per_share,
            'total_value': total_consideration
        }
