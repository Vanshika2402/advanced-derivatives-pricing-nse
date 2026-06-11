import numpy as np


def calculate_log_returns(price_df):
    """
    Calculate daily log returns from closing prices.
    """
    df = price_df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df.dropna()


def historical_volatility(price_df, trading_days=252):
    """
    Annualized historical volatility.
    """
    df = calculate_log_returns(price_df)
    daily_volatility = df["log_return"].std()
    annualized_volatility = daily_volatility * np.sqrt(trading_days)
    return annualized_volatility


def return_statistics(price_df):
    """
    Return distribution statistics.
    """
    df = calculate_log_returns(price_df)

    return {
        "mean_daily_return": df["log_return"].mean(),
        "annualized_return": df["log_return"].mean() * 252,
        "daily_volatility": df["log_return"].std(),
        "annualized_volatility": df["log_return"].std() * np.sqrt(252),
        "skewness": df["log_return"].skew(),
        "kurtosis": df["log_return"].kurtosis()
    }
