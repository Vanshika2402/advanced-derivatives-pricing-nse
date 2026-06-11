import numpy as np
from scipy.stats import norm


def d1_d2(S, K, T, r, sigma):
    """
    Calculate d1 and d2 used in the Black-Scholes model.
    """
    d1 = (
        np.log(S / K) + (r + 0.5 * sigma ** 2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2


def black_scholes_price(S, K, T, r, sigma, option_type="CE"):
    """
    Black-Scholes price for European call or put options.

    CE = Call option
    PE = Put option
    """

    if T <= 0:
        if option_type == "CE":
            return max(S - K, 0)
        return max(K - S, 0)

    if sigma <= 0:
        raise ValueError("Volatility must be positive.")

    d1, d2 = d1_d2(S, K, T, r, sigma)

    if option_type == "CE":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price
