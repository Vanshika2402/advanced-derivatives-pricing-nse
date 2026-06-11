import numpy as np
import pandas as pd
from src.black_scholes import black_scholes_price
from src.greeks import greeks


def delta_hedging_backtest(price_df, K, r, sigma, T_initial, option_type="CE"):
    """
    Dynamic delta hedging backtest using historical underlying prices.

    The strategy:
    1. Sell/price one option.
    2. Hedge using Black-Scholes delta.
    3. Rebalance daily.
    4. Track hedging error.

    CE = Call option
    PE = Put option
    """

    df = price_df.copy().reset_index(drop=True)

    # Use latest 120 observations for hedging test
    df = df.tail(min(len(df), 120)).reset_index(drop=True)

    if len(df) < 30:
        raise ValueError("Need at least 30 observations for hedging backtest.")

    initial_spot = df.loc[0, "close"]

    initial_option_value = black_scholes_price(
        initial_spot,
        K,
        T_initial,
        r,
        sigma,
        option_type
    )

    initial_delta = greeks(
        initial_spot,
        K,
        T_initial,
        r,
        sigma,
        option_type
    )["delta"]

    cash_account = initial_option_value - initial_delta * initial_spot
    stock_position = initial_delta

    hedge_rows = []

    for i in range(1, len(df)):
        spot = df.loc[i, "close"]

        remaining_time = max(T_initial - i / 252, 1 / 252)

        option_value = black_scholes_price(
            spot,
            K,
            remaining_time,
            r,
            sigma,
            option_type
        )

        new_delta = greeks(
            spot,
            K,
            remaining_time,
            r,
            sigma,
            option_type
        )["delta"]

        # cash account earns risk-free interest daily
        cash_account = cash_account * np.exp(r / 252)

        # rebalance hedge
        delta_change = new_delta - stock_position
        cash_account = cash_account - delta_change * spot
        stock_position = new_delta

        hedge_portfolio_value = stock_position * spot + cash_account
        hedging_error = hedge_portfolio_value - option_value

        hedge_rows.append({
            "date": df.loc[i, "date"],
            "spot": spot,
            "option_value": option_value,
            "delta": new_delta,
            "hedge_portfolio": hedge_portfolio_value,
            "hedging_error": hedging_error
        })

    return pd.DataFrame(hedge_rows)
