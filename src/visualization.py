import os
import numpy as np
import matplotlib.pyplot as plt
from src.greeks import greeks


def ensure_dir(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def plot_payoff(S, K, premium, option_type, title, path):
    """
    Plot option payoff diagram.
    """

    spot_range = np.linspace(S * 0.6, S * 1.4, 200)

    if option_type == "CE":
        payoff = np.maximum(spot_range - K, 0) - premium
    else:
        payoff = np.maximum(K - spot_range, 0) - premium

    plt.figure(figsize=(10, 5))
    plt.plot(spot_range, payoff)
    plt.axhline(0, linestyle="--")
    plt.axvline(K, linestyle="--")
    plt.title(title)
    plt.xlabel("Underlying Price")
    plt.ylabel("Profit / Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_pricing_comparison(df, path):
    """
    Compare Black-Scholes, Binomial Tree, and Monte Carlo prices.
    """

    plt.figure(figsize=(10, 5))

    x = np.arange(len(df))
    width = 0.25

    plt.bar(x - width, df["black_scholes"], width, label="Black-Scholes")
    plt.bar(x, df["binomial"], width, label="Binomial Tree")
    plt.bar(x + width, df["monte_carlo"], width, label="Monte Carlo")

    plt.xticks(x, df["instrument"])
    plt.ylabel("Option Price")
    plt.title("Model Pricing Comparison")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_greek_sensitivity(S, K, T, r, sigma, option_type, path):
    """
    Plot Greeks as spot price changes.
    """

    spots = np.linspace(S * 0.75, S * 1.25, 100)

    delta_values = []
    gamma_values = []
    vega_values = []
    theta_values = []
    rho_values = []

    for spot in spots:
        greek_values = greeks(spot, K, T, r, sigma, option_type)

        delta_values.append(greek_values["delta"])
        gamma_values.append(greek_values["gamma"])
        vega_values.append(greek_values["vega_per_1pct"])
        theta_values.append(greek_values["theta_per_day"])
        rho_values.append(greek_values["rho_per_1pct"])

    plt.figure(figsize=(10, 5))
    plt.plot(spots, delta_values, label="Delta")
    plt.plot(spots, gamma_values, label="Gamma")
    plt.plot(spots, vega_values, label="Vega")
    plt.plot(spots, theta_values, label="Theta")
    plt.plot(spots, rho_values, label="Rho")

    plt.title("Greeks Sensitivity to Spot Price")
    plt.xlabel("Spot Price")
    plt.ylabel("Greek Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_volatility_surface(option_chain_df, path):
    """
    Plot implied volatility against strike price.
    This is a volatility smile / single-expiry volatility surface slice.
    """

    df = option_chain_df.copy()
    df = df.dropna(subset=["strike", "implied_volatility"])

    plt.figure(figsize=(10, 5))
    plt.scatter(df["strike"], df["implied_volatility"])
    plt.title("Implied Volatility Smile / Surface Slice")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_scenarios(df, title, path):
    """
    Plot option prices under different market scenarios.
    """

    plt.figure(figsize=(10, 5))
    plt.bar(df["scenario"], df["option_price"])
    plt.title(title)
    plt.xlabel("Scenario")
    plt.ylabel("Option Price")
    plt.xticks(rotation=25)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_hedging(df, title, path):
    """
    Plot hedging error over time.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["hedging_error"])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Hedging Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
