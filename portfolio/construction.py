import warnings
from portfolio.optimization import MinVar, IVP, HRP, ERC, MeanCVaR, BlackLitterman

warnings.warn("This module is deprecated. Please import from 'portfolio.optimization' instead.", DeprecationWarning, stacklevel=2)