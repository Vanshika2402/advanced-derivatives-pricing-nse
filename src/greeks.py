import numpy as np
from scipy.stats import norm
from src.black_scholes import d1_d2


def greeks(S, K, T, r, sigma, option_type="CE"):
    """
    Calculate Black-Scholes Greeks.

    Delta: sensitivity to underlying price
    Gamma: sensitivity of Delta to underlying price
    Vega: sensitivity to volatility
    Theta: sensitivity to time decay
    Rho: sensitivity to interest rate
    """

    d1, d2 = d1_d2(S, K, T, r, sigma)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    if option_type == "CE":
        delta = norm.cdf(d1)

        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365

        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100

    else:
        delta = norm.cdf(d1) - 1

        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / 365

        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta": delta,
        "gamma": gamma,
        "vega_per_1pct": vega,
        "theta_per_day": theta,
        "rho_per_1pct": rho
    }
