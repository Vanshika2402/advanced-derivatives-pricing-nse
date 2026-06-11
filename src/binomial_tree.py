import numpy as np


def binomial_tree_price(S, K, T, r, sigma, steps=500, option_type="CE"):
    """
    Cox-Ross-Rubinstein binomial tree option pricing model.

    CE = Call option
    PE = Put option
    """

    dt = T / steps

    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    p = (np.exp(r * dt) - d) / (u - d)

    prices = np.array([
        S * (u ** j) * (d ** (steps - j))
        for j in range(steps + 1)
    ])

    if option_type == "CE":
        option_values = np.maximum(prices - K, 0)
    else:
        option_values = np.maximum(K - prices, 0)

    discount = np.exp(-r * dt)

    for i in range(steps - 1, -1, -1):
        option_values = discount * (
            p * option_values[1:i + 2]
            + (1 - p) * option_values[0:i + 1]
        )

    return option_values[0]
