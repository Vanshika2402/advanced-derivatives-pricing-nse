import numpy as np


def monte_carlo_price(
    S,
    K,
    T,
    r,
    sigma,
    paths=50000,
    steps=252,
    option_type="CE",
    seed=42
):
    """
    Monte Carlo option pricing using risk-neutral Geometric Brownian Motion.

    CE = Call option
    PE = Put option
    """

    np.random.seed(seed)

    dt = T / steps

    random_numbers = np.random.standard_normal((paths, steps))

    price_increments = (
        (r - 0.5 * sigma ** 2) * dt
        + sigma * np.sqrt(dt) * random_numbers
    )

    log_price_paths = np.cumsum(price_increments, axis=1)

    terminal_prices = S * np.exp(log_price_paths[:, -1])

    if option_type == "CE":
        payoff = np.maximum(terminal_prices - K, 0)
    else:
        payoff = np.maximum(K - terminal_prices, 0)

    discounted_payoff = np.exp(-r * T) * payoff

    option_price = np.mean(discounted_payoff)
    standard_error = np.std(discounted_payoff) / np.sqrt(paths)

    return option_price, standard_error
