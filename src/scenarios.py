import pandas as pd
from src.black_scholes import black_scholes_price


def scenario_analysis(S, K, T, r, sigma, option_type="CE"):
    """
    Scenario analysis for option pricing under different market conditions.

    Scenarios:
    - Low volatility
    - High volatility
    - Bull market
    - Bear market
    - Market crash
    """

    scenarios = [
        {
            "scenario": "Low Volatility",
            "spot_multiplier": 1.00,
            "vol_multiplier": 0.70,
            "rate_shift": 0.00
        },
        {
            "scenario": "High Volatility",
            "spot_multiplier": 1.00,
            "vol_multiplier": 1.50,
            "rate_shift": 0.00
        },
        {
            "scenario": "Bull Market",
            "spot_multiplier": 1.10,
            "vol_multiplier": 1.00,
            "rate_shift": 0.00
        },
        {
            "scenario": "Bear Market",
            "spot_multiplier": 0.90,
            "vol_multiplier": 1.20,
            "rate_shift": 0.00
        },
        {
            "scenario": "Market Crash",
            "spot_multiplier": 0.75,
            "vol_multiplier": 2.00,
            "rate_shift": -0.005
        }
    ]

    rows = []

    for scenario in scenarios:
        scenario_spot = S * scenario["spot_multiplier"]
        scenario_volatility = sigma * scenario["vol_multiplier"]
        scenario_rate = max(r + scenario["rate_shift"], 0)

        option_price = black_scholes_price(
            scenario_spot,
            K,
            T,
            scenario_rate,
            scenario_volatility,
            option_type
        )

        rows.append({
            "scenario": scenario["scenario"],
            "spot": scenario_spot,
            "volatility": scenario_volatility,
            "risk_free_rate": scenario_rate,
            "option_price": option_price
        })

    return pd.DataFrame(rows)
