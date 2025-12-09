"""
Advanced Derivatives Module
===========================

Implements sophisticated derivative pricing and hedging strategies for 
Interest Rate Swaps, Forward Swaps, and Currency Hedging.

Author: Dynamic Portfolio Optimization Team
Institution: WorldQuant University
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterestRateSwap:
    """
    Interest Rate Swap (IRS) pricer and analytics.
    
    An IRS is an agreement to exchange fixed rate payments for floating rate
    payments on a notional amount over a specified period.
    
    Attributes:
        notional: Principal amount
        fixed_rate: Fixed leg rate (annual)
        floating_rate: Initial floating leg rate
        maturity_years: Maturity in years
        payment_frequency: Payment frequency ('annual', 'semi-annual', 'quarterly', 'monthly')
        day_count: Day count convention ('actual/360', 'actual/365', '30/360')
    """
    
    def __init__(self, notional: float, fixed_rate: float, 
                 floating_rate: float, maturity_years: float,
                 payment_frequency: str = 'quarterly',
                 day_count: str = 'actual/360'):
        """Initialize Interest Rate Swap."""
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.floating_rate = floating_rate
        self.maturity_years = maturity_years
        self.payment_frequency = payment_frequency
        self.day_count = day_count
        self.start_date = datetime.now()
        self.end_date = self.start_date + timedelta(days=365*maturity_years)
        
        # Calculate payment dates
        self.payment_dates = self._calculate_payment_dates()
        self.cash_flows = None
        
        logger.info(f"IRS created: ${notional:,.0f} at {fixed_rate*100:.2f}% fixed")
    
    def _calculate_payment_dates(self) -> List[datetime]:
        """Calculate payment dates based on frequency."""
        dates = []
        if self.payment_frequency == 'annual':
            freq_days = 365
        elif self.payment_frequency == 'semi-annual':
            freq_days = 182
        elif self.payment_frequency == 'quarterly':
            freq_days = 91
        elif self.payment_frequency == 'monthly':
            freq_days = 30
        else:
            freq_days = 365
            
        current_date = self.start_date
        while current_date <= self.end_date:
            dates.append(current_date)
            current_date += timedelta(days=freq_days)
            
        return dates[1:]  # Exclude start date
    
    def calculate_dv01(self) -> Dict[str, float]:
        """
        Calculate DV01 (Dollar Value of 1 basis point).
        
        DV01 measures the change in swap value for a 1 basis point change in rates.
        
        Returns:
            Dictionary with DV01 for fixed and floating legs
        """
        base_value = self.calculate_swap_value()
        
        # Recalculate with rates bumped by 1bp (0.01%)
        original_fixed = self.fixed_rate
        self.fixed_rate += 0.0001
        up_value = self.calculate_swap_value()
        self.fixed_rate = original_fixed
        
        dv01_fixed = abs(up_value - base_value)
        
        return {
            'dv01_fixed': dv01_fixed,
            'dv01_floating': dv01_fixed,  # Approximately equal
            'total_dv01': dv01_fixed,
            'duration_years': dv01_fixed / self.notional * 10000
        }
    
    def calculate_fixed_leg_pv(self, discount_rate: Optional[float] = None) -> float:
        """Calculate present value of fixed leg."""
        if discount_rate is None:
            discount_rate = self.fixed_rate
            
        pv = 0.0
        n_periods = len(self.payment_dates)
        period_rate = discount_rate / (12 / (12 / len(self.payment_dates)) if self.payment_frequency != 'annual' else 1)
        
        for i, date in enumerate(self.payment_dates):
            time_fraction = (i + 1) / (12 / (12 / len(self.payment_dates)) if self.payment_frequency != 'annual' else 1)
            cf = self.notional * self.fixed_rate * time_fraction
            discount_factor = np.exp(-discount_rate * time_fraction)
            pv += cf * discount_factor
            
        return pv
    
    def calculate_floating_leg_pv(self, discount_rate: Optional[float] = None) -> float:
        """Calculate present value of floating leg."""
        if discount_rate is None:
            discount_rate = self.floating_rate
            
        pv = 0.0
        for i, date in enumerate(self.payment_dates):
            time_fraction = (i + 1) / len(self.payment_dates)
            cf = self.notional * self.floating_rate * time_fraction
            discount_factor = np.exp(-discount_rate * time_fraction)
            pv += cf * discount_factor
            
        return pv
    
    def calculate_swap_value(self) -> float:
        """Calculate net swap value (floating PV - fixed PV)."""
        floating_pv = self.calculate_floating_leg_pv()
        fixed_pv = self.calculate_fixed_leg_pv()
        return floating_pv - fixed_pv
    
    def calculate_cash_flows(self) -> pd.DataFrame:
        """Generate detailed cash flow schedule."""
        cash_flows = []
        
        for i, date in enumerate(self.payment_dates):
            time_fraction = (i + 1) / len(self.payment_dates)
            
            fixed_cf = self.notional * self.fixed_rate * time_fraction
            floating_cf = self.notional * self.floating_rate * time_fraction
            net_cf = floating_cf - fixed_cf
            
            discount_factor = np.exp(-self.fixed_rate * time_fraction)
            pv = net_cf * discount_factor
            
            cash_flows.append({
                'Date': date,
                'Period': i + 1,
                'Days': (date - self.start_date).days,
                'Fixed_CF': fixed_cf,
                'Floating_CF': floating_cf,
                'Net_CF': net_cf,
                'Discount_Factor': discount_factor,
                'PV': pv
            })
        
        self.cash_flows = pd.DataFrame(cash_flows)
        return self.cash_flows
    
    def print_summary(self):
        """Print IRS valuation summary."""
        print("\n" + "="*60)
        print("INTEREST RATE SWAP SUMMARY")
        print("="*60)
        print(f"Notional Amount:     ${self.notional:,.2f}")
        print(f"Fixed Rate:          {self.fixed_rate*100:.3f}%")
        print(f"Floating Rate:       {self.floating_rate*100:.3f}%")
        print(f"Maturity:            {self.maturity_years} years")
        print(f"Payment Frequency:   {self.payment_frequency}")
        print(f"Start Date:          {self.start_date.strftime('%Y-%m-%d')}")
        print(f"End Date:            {self.end_date.strftime('%Y-%m-%d')}")
        print(f"Number of Payments:  {len(self.payment_dates)}")
        print("-"*60)
        
        fixed_pv = self.calculate_fixed_leg_pv()
        floating_pv = self.calculate_floating_leg_pv()
        swap_value = self.calculate_swap_value()
        dv01 = self.calculate_dv01()
        
        print(f"\nVALUATION:")
        print(f"Fixed Leg PV:        ${fixed_pv:,.2f}")
        print(f"Floating Leg PV:     ${floating_pv:,.2f}")
        print(f"Swap Value:          ${swap_value:,.2f}")
        
        print(f"\nRISK METRICS (DV01):")
        print(f"DV01:                ${dv01['total_dv01']:,.2f}")
        print(f"Duration:            {dv01['duration_years']:.4f} years")
        print("="*60 + "\n")


class ForwardSwap:
    """
    Forward Swap pricer using cost-of-carry model.
    
    A forward swap is an agreement to enter into a swap at a future date,
    with terms fixed today.
    """
    
    def __init__(self, forward_rate: float, fixed_rate: float, 
                 time_to_expiry: float, carrying_cost_rate: float):
        """
        Initialize Forward Swap.
        
        Args:
            forward_rate: Current underlying swap rate
            fixed_rate: Fixed rate for the forward swap
            time_to_expiry: Time to forward swap start (years)
            carrying_cost_rate: Carrying cost rate (repo/financing rate)
        """
        self.forward_rate = forward_rate
        self.fixed_rate = fixed_rate
        self.time_to_expiry = time_to_expiry
        self.carrying_cost_rate = carrying_cost_rate
    
    def calculate_forward_price(self) -> float:
        """
        Calculate forward swap price using cost-of-carry model.
        
        F = S * e^((r - q)T)
        where r is carrying cost and q is convenience yield
        """
        forward_price = self.forward_rate * np.exp(
            self.carrying_cost_rate * self.time_to_expiry
        )
        return forward_price
    
    def calculate_forward_premium(self) -> float:
        """Calculate premium over spot rate."""
        forward_price = self.calculate_forward_price()
        premium = forward_price - self.forward_rate
        premium_pct = (premium / self.forward_rate) * 100
        return premium, premium_pct
    
    def calculate_value_at_expiry(self) -> float:
        """Calculate forward value at time of expiry."""
        forward_price = self.calculate_forward_price()
        value = (forward_price - self.fixed_rate) * 100  # Assume 100 notional for display
        return value


class HedgingStrategy:
    """
    Implement hedging strategies using derivatives.
    """
    
    def __init__(self, portfolio_value: float, interest_rate_exposure: float,
                 fx_exposure: Dict[str, float], credit_exposure: float):
        """
        Initialize Hedging Strategy.
        
        Args:
            portfolio_value: Total portfolio value
            interest_rate_exposure: Exposure to interest rates (duration)
            fx_exposure: Dict of currency exposures {currency: exposure}
            credit_exposure: Credit risk exposure
        """
        self.portfolio_value = portfolio_value
        self.interest_rate_exposure = interest_rate_exposure
        self.fx_exposure = fx_exposure
        self.credit_exposure = credit_exposure
        self.hedges = {}
    
    def calculate_interest_rate_hedge(self, target_duration: float = 0.0,
                                    hedge_ratio: float = 0.5) -> Dict:
        """
        Calculate interest rate hedge using IRS.
        
        Args:
            target_duration: Target portfolio duration
            hedge_ratio: Fraction to hedge (0-1)
        """
        notional_to_hedge = self.portfolio_value * hedge_ratio
        
        hedge_info = {
            'notional': notional_to_hedge,
            'hedge_ratio': hedge_ratio,
            'current_duration': self.interest_rate_exposure,
            'target_duration': target_duration,
            'duration_reduction': self.interest_rate_exposure - target_duration,
            'estimated_cost': notional_to_hedge * 0.002  # 2bps cost
        }
        
        self.hedges['interest_rate'] = hedge_info
        return hedge_info
    
    def calculate_fx_hedge_benefit(self) -> Dict:
        """Calculate benefit of currency hedging."""
        total_fx_exposure = sum(self.fx_exposure.values())
        
        benefits = {
            'total_fx_exposure': total_fx_exposure,
            'unhedged_volatility': np.std(list(self.fx_exposure.values())),
            'hedged_volatility': np.std(list(self.fx_exposure.values())) * 0.3,
            'volatility_reduction': '70%',
            'hedge_cost': total_fx_exposure * 0.001  # 1bp cost
        }
        
        return benefits


if __name__ == "__main__":
    # Example: Create and price an IRS
    irs = InterestRateSwap(
        notional=1_000_000,
        fixed_rate=0.03,
        floating_rate=0.025,
        maturity_years=5,
        payment_frequency='quarterly'
    )
    
    irs.print_summary()
    print("\nCash Flows:")
    print(irs.calculate_cash_flows())
