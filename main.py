import os
import zipfile
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from io import BytesIO
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq


INSTRUMENTS = [
    {"name": "NIFTY", "symbol": "NIFTY", "yahoo": "^NSEI", "option_type": "CE"},
    {"name": "BANKNIFTY", "symbol": "BANKNIFTY", "yahoo": "^NSEBANK", "option_type": "CE"},
    {"name": "RELIANCE", "symbol": "RELIANCE", "yahoo": "RELIANCE.NS", "option_type": "CE"},
]

START_DATE = "2024-01-01"
RISK_FREE_RATE = 0.068
TRADING_DAYS = 252

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
PLOT_DIR = "outputs/plots"

MC_PATHS = 30000
MC_STEPS = 252


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def nse_headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/all-reports-derivatives",
    }


def bhavcopy_urls(date_obj):
    y = date_obj.strftime("%Y")
    m = date_obj.strftime("%b").upper()
    dmy = date_obj.strftime("%d%b%Y").upper()
    ymd = date_obj.strftime("%Y%m%d")

    return [
        f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip",
        f"https://archives.nseindia.com/content/historical/DERIVATIVES/{y}/{m}/fo{dmy}bhav.csv.zip",
        f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{y}/{m}/fo{dmy}bhav.csv.zip",
        f"https://www.nseindia.com/content/historical/DERIVATIVES/{y}/{m}/fo{dmy}bhav.csv.zip",
    ]


def download_latest_fo_bhavcopy(max_back_days=45):
    session = requests.Session()
    session.headers.update(nse_headers())

    try:
        session.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass

    today = datetime.today()

    errors = []

    for i in range(max_back_days):
        date_obj = today - timedelta(days=i)

        if date_obj.weekday() >= 5:
            continue

        for url in bhavcopy_urls(date_obj):
            try:
                response = session.get(url, timeout=25)

                if response.status_code != 200 or len(response.content) < 1000:
                    errors.append(f"{date_obj.date()} failed: {response.status_code}")
                    continue

                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f)

                raw_path = f"{DATA_DIR}/nse_fo_bhavcopy_raw_{date_obj.date()}.csv"
                df.to_csv(raw_path, index=False)

                print(f"Downloaded NSE F&O bhavcopy for {date_obj.date()}")
                print(f"Source URL: {url}")

                return df, date_obj.date(), url

            except Exception as exc:
                errors.append(str(exc))

    raise RuntimeError("Could not download NSE F&O bhavcopy. Try again later. Last errors: " + " | ".join(errors[-5:]))


def clean_col_name(x):
    return "".join(ch.lower() for ch in str(x) if ch.isalnum())


def pick_col(df, possible_names):
    col_map = {clean_col_name(c): c for c in df.columns}

    for name in possible_names:
        key = clean_col_name(name)
        if key in col_map:
            return col_map[key]

    return None


def to_number(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )


def normalize_fo_bhavcopy(df, trade_date):
    symbol_col = pick_col(df, ["SYMBOL", "TckrSymb", "FinInstrmNm"])
    expiry_col = pick_col(df, ["EXPIRY_DT", "FinInstrmActlXpryDt", "XpryDt"])
    strike_col = pick_col(df, ["STRIKE_PR", "StrkPric"])
    option_col = pick_col(df, ["OPTION_TYP", "OptnTp"])
    close_col = pick_col(df, ["CLOSE", "ClsPric", "SETTLE_PR", "SttlmPric"])
    oi_col = pick_col(df, ["OPEN_INT", "OpnIntrst"])
    volume_col = pick_col(df, ["CONTRACTS", "TtlTradgVol"])
    underlying_col = pick_col(df, ["UndrlygPric", "UNDERLYING"])

    required = [symbol_col, expiry_col, strike_col, option_col, close_col]

    if any(c is None for c in required):
        raise ValueError(
            "Could not identify required columns in NSE bhavcopy. "
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["symbol"] = df[symbol_col].astype(str).str.upper().str.strip()
    out["expiry"] = pd.to_datetime(df[expiry_col], errors="coerce")
    out["strike"] = to_number(df[strike_col])
    out["option_type"] = df[option_col].astype(str).str.upper().str.strip()
    out["market_price"] = to_number(df[close_col])

    if oi_col:
        out["open_interest"] = to_number(df[oi_col])
    else:
        out["open_interest"] = np.nan

    if volume_col:
        out["volume"] = to_number(df[volume_col])
    else:
        out["volume"] = np.nan

    if underlying_col:
        out["underlying_from_bhavcopy"] = to_number(df[underlying_col])
    else:
        out["underlying_from_bhavcopy"] = np.nan

    out["trade_date"] = pd.to_datetime(trade_date)

    out = out.dropna(subset=["expiry", "strike", "market_price"])
    out = out[out["option_type"].isin(["CE", "PE"])]
    out = out[out["strike"] > 0]
    out = out[out["market_price"] > 0]

    normalized_path = f"{DATA_DIR}/nse_fo_bhavcopy_normalized_{trade_date}.csv"
    out.to_csv(normalized_path, index=False)

    return out


def fetch_historical_prices(yahoo_symbol):
    df = yf.download(yahoo_symbol, start=START_DATE, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No Yahoo historical data found for {yahoo_symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df.rename(columns={"Date": "date", "Close": "close"})
    df = df[["date", "close"]].dropna()

    return df


def log_return_stats(price_df):
    df = price_df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna()

    return {
        "historical_vol": df["log_return"].std() * np.sqrt(TRADING_DAYS),
        "annualized_return": df["log_return"].mean() * TRADING_DAYS,
        "skewness": df["log_return"].skew(),
        "kurtosis": df["log_return"].kurtosis(),
    }


def d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(S, K, T, r, sigma, option_type):
    if T <= 0:
        return max(S - K, 0) if option_type == "CE" else max(K - S, 0)

    d1, d2 = d1_d2(S, K, T, r, sigma)

    if option_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_volatility(market_price, S, K, T, r, option_type):
    try:
        intrinsic = max(S - K, 0) if option_type == "CE" else max(K - S, 0)

        if market_price < intrinsic:
            return np.nan

        def objective(sigma):
            return black_scholes_price(S, K, T, r, sigma, option_type) - market_price

        return brentq(objective, 0.0001, 5.0)

    except Exception:
        return np.nan


def binomial_tree_price(S, K, T, r, sigma, steps, option_type):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    prices = np.array([S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])

    if option_type == "CE":
        values = np.maximum(prices - K, 0)
    else:
        values = np.maximum(K - prices, 0)

    discount = np.exp(-r * dt)

    for i in range(steps - 1, -1, -1):
        values = discount * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])

    return values[0]


def monte_carlo_price(S, K, T, r, sigma, option_type):
    np.random.seed(42)

    dt = T / MC_STEPS
    Z = np.random.standard_normal((MC_PATHS, MC_STEPS))

    log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    terminal_prices = S * np.exp(np.cumsum(log_returns, axis=1)[:, -1])

    if option_type == "CE":
        payoff = np.maximum(terminal_prices - K, 0)
    else:
        payoff = np.maximum(K - terminal_prices, 0)

    discounted = np.exp(-r * T) * payoff

    return np.mean(discounted), np.std(discounted) / np.sqrt(MC_PATHS)


def calculate_greeks(S, K, T, r, sigma, option_type):
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

    return delta, gamma, vega, theta, rho


def scenario_analysis(S, K, T, r, sigma, option_type):
    scenarios = [
        ["Low Volatility", 1.00, 0.70, 0.000],
        ["High Volatility", 1.00, 1.50, 0.000],
        ["Bull Market", 1.10, 1.00, 0.000],
        ["Bear Market", 0.90, 1.20, 0.000],
        ["Market Crash", 0.75, 2.00, -0.005],
    ]

    rows = []

    for name, spot_mult, vol_mult, rate_shift in scenarios:
        new_S = S * spot_mult
        new_sigma = sigma * vol_mult
        new_r = max(r + rate_shift, 0)

        rows.append({
            "scenario": name,
            "spot": new_S,
            "volatility": new_sigma,
            "risk_free_rate": new_r,
            "option_price": black_scholes_price(new_S, K, T, new_r, new_sigma, option_type),
        })

    return pd.DataFrame(rows)


def delta_hedging_backtest(price_df, K, r, sigma, T_initial, option_type):
    df = price_df.tail(120).reset_index(drop=True)

    if len(df) < 30:
        return pd.DataFrame()

    S0 = df.loc[0, "close"]
    option_value = black_scholes_price(S0, K, T_initial, r, sigma, option_type)
    delta, _, _, _, _ = calculate_greeks(S0, K, T_initial, r, sigma, option_type)

    stock_position = delta
    cash = option_value - delta * S0

    rows = []

    for i in range(1, len(df)):
        S = df.loc[i, "close"]
        T = max(T_initial - i / 252, 1 / 252)

        option_value = black_scholes_price(S, K, T, r, sigma, option_type)
        new_delta, _, _, _, _ = calculate_greeks(S, K, T, r, sigma, option_type)

        cash *= np.exp(r / 252)
        cash -= (new_delta - stock_position) * S
        stock_position = new_delta

        hedge_value = stock_position * S + cash
        error = hedge_value - option_value

        rows.append({
            "date": df.loc[i, "date"],
            "spot": S,
            "option_value": option_value,
            "delta": new_delta,
            "hedge_portfolio": hedge_value,
            "hedging_error": error,
        })

    return pd.DataFrame(rows)


def plot_payoff(S, K, premium, option_type, path, title):
    spots = np.linspace(S * 0.6, S * 1.4, 200)

    if option_type == "CE":
        payoff = np.maximum(spots - K, 0) - premium
    else:
        payoff = np.maximum(K - spots, 0) - premium

    plt.figure(figsize=(10, 5))
    plt.plot(spots, payoff)
    plt.axhline(0, linestyle="--")
    plt.axvline(K, linestyle="--")
    plt.title(title)
    plt.xlabel("Underlying Price")
    plt.ylabel("Profit / Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_greeks(S, K, T, r, sigma, option_type, path, title):
    spots = np.linspace(S * 0.75, S * 1.25, 120)

    delta_values = []
    gamma_values = []
    vega_values = []
    theta_values = []
    rho_values = []

    for spot in spots:
        d, g, v, t, rh = calculate_greeks(spot, K, T, r, sigma, option_type)
        delta_values.append(d)
        gamma_values.append(g)
        vega_values.append(v)
        theta_values.append(t)
        rho_values.append(rh)

    plt.figure(figsize=(10, 5))
    plt.plot(spots, delta_values, label="Delta")
    plt.plot(spots, gamma_values, label="Gamma")
    plt.plot(spots, vega_values, label="Vega")
    plt.plot(spots, theta_values, label="Theta")
    plt.plot(spots, rho_values, label="Rho")
    plt.title(title)
    plt.xlabel("Spot Price")
    plt.ylabel("Greek Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_vol_surface(surface_df, path, title):
    df = surface_df.dropna(subset=["strike", "implied_volatility"])

    plt.figure(figsize=(10, 5))
    plt.scatter(df["strike"], df["implied_volatility"])
    plt.title(title)
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_scenarios(df, path, title):
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


def plot_hedging(df, path, title):
    if df.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["hedging_error"])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Hedging Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def select_atm_option(fo_df, instrument, price_df, bhavcopy_date):
    symbol = instrument["symbol"]
    option_type = instrument["option_type"]

    chain = fo_df[fo_df["symbol"] == symbol].copy()

    if chain.empty:
        raise ValueError(f"No option data found for {symbol} in NSE F&O bhavcopy.")

    chain = chain[chain["expiry"] >= pd.to_datetime(bhavcopy_date)]

    if chain.empty:
        raise ValueError(f"No valid future expiry found for {symbol}.")

    nearest_expiry = chain["expiry"].min()
    chain = chain[chain["expiry"] == nearest_expiry]

    chain = chain[chain["option_type"] == option_type]

    if chain.empty:
        raise ValueError(f"No {option_type} option found for {symbol}.")

    spot_from_file = chain["underlying_from_bhavcopy"].dropna()

    if len(spot_from_file) > 0 and spot_from_file.iloc[0] > 0:
        spot = float(spot_from_file.iloc[0])
    else:
        spot = float(price_df["close"].iloc[-1])

    chain["atm_distance"] = (chain["strike"] - spot).abs()
    atm = chain.sort_values("atm_distance").iloc[0].to_dict()

    return atm, chain, spot


def generate_report(summary_df, scenario_tables, bhavcopy_date, source_url):
    report = []

    report.append("# Advanced Derivatives Pricing and Risk Modeling Framework")
    report.append("")
    report.append("## Market")
    report.append("Indian NSE derivatives market.")
    report.append("")
    report.append("## Real Data Used")
    report.append(f"- NSE F&O bhavcopy date: {bhavcopy_date}")
    report.append(f"- NSE source URL: {source_url}")
    report.append("- Yahoo Finance historical underlying price data through yfinance.")
    report.append("")
    report.append("## Instruments")
    report.append("- NIFTY option")
    report.append("- BANKNIFTY option")
    report.append("- RELIANCE option")
    report.append("")
    report.append("## Mathematical Framework")
    report.append("")
    report.append("The underlying price is modeled as Geometric Brownian Motion:")
    report.append("")
    report.append("```text")
    report.append("dS_t = mu S_t dt + sigma S_t dW_t")
    report.append("```")
    report.append("")
    report.append("Under risk-neutral valuation:")
    report.append("")
    report.append("```text")
    report.append("dS_t = r S_t dt + sigma S_t dW_t")
    report.append("```")
    report.append("")
    report.append("Black-Scholes call price:")
    report.append("")
    report.append("```text")
    report.append("C = S N(d1) - K e^(-rT) N(d2)")
    report.append("```")
    report.append("")
    report.append("Black-Scholes put price:")
    report.append("")
    report.append("```text")
    report.append("P = K e^(-rT) N(-d2) - S N(-d1)")
    report.append("```")
    report.append("")
    report.append("where:")
    report.append("")
    report.append("```text")
    report.append("d1 = [ln(S/K) + (r + sigma^2/2)T] / [sigma sqrt(T)]")
    report.append("d2 = d1 - sigma sqrt(T)")
    report.append("```")
    report.append("")
    report.append("## Black-Scholes Assumptions")
    report.append("- Underlying follows GBM.")
    report.append("- Volatility is constant.")
    report.append("- Risk-free rate is constant.")
    report.append("- No arbitrage.")
    report.append("- No transaction costs.")
    report.append("- Continuous trading is possible.")
    report.append("- European exercise.")
    report.append("")
    report.append("## Pricing Summary")
    report.append("")
    report.append(summary_df.to_markdown(index=False))
    report.append("")
    report.append("## Scenario Analysis")
    report.append("")

    for name, df in scenario_tables.items():
        report.append(f"### {name}")
        report.append("")
        report.append(df.to_markdown(index=False))
        report.append("")

    report.append("## Interpretation")
    report.append("")
    report.append(
        "Black-Scholes provides a closed-form benchmark price. "
        "The binomial model approximates risk-neutral pricing through a discrete tree. "
        "Monte Carlo pricing simulates terminal stock prices and discounts expected payoff."
    )
    report.append("")
    report.append(
        "Market price may differ from theoretical prices because of volatility smile, liquidity, "
        "bid-ask spread, discrete dividends, jump risk, and model assumptions."
    )
    report.append("")
    report.append("## Hedging Interpretation")
    report.append("")
    report.append(
        "Delta hedging reduces first-order exposure to the underlying asset. "
        "Hedging error remains due to discrete rebalancing, changing volatility, transaction costs, and market jumps."
    )
    report.append("")
    report.append("## Limitations")
    report.append("- NSE public files can occasionally be unavailable or delayed.")
    report.append("- Bhavcopy data is end-of-day, not live intraday data.")
    report.append("- Black-Scholes assumes constant volatility.")
    report.append("- Transaction costs, taxes, and slippage are excluded.")
    report.append("- Dividend effects are not explicitly modeled.")
    report.append("")
    report.append("## Future Improvements")
    report.append("- Add vendor-level historical option-chain data.")
    report.append("- Add dividend-adjusted Black-Scholes.")
    report.append("- Add Heston stochastic volatility.")
    report.append("- Add jump-diffusion pricing.")
    report.append("- Add transaction-cost-aware hedging.")
    report.append("- Add portfolio VaR and Expected Shortfall.")

    with open(f"{OUTPUT_DIR}/research_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))


def main():
    ensure_dirs()

    raw_bhavcopy, bhavcopy_date, source_url = download_latest_fo_bhavcopy()
    fo_df = normalize_fo_bhavcopy(raw_bhavcopy, bhavcopy_date)

    summary_rows = []
    scenario_tables = {}

    for instrument in INSTRUMENTS:
        name = instrument["name"]

        print(f"\nProcessing {name}...")

        price_df = fetch_historical_prices(instrument["yahoo"])
        price_df.to_csv(f"{DATA_DIR}/{name}_historical_prices.csv", index=False)

        stats = log_return_stats(price_df)

        atm, chain, spot = select_atm_option(fo_df, instrument, price_df, bhavcopy_date)

        K = float(atm["strike"])
        market_price = float(atm["market_price"])
        option_type = atm["option_type"]
        expiry = pd.to_datetime(atm["expiry"])
        trade_date = pd.to_datetime(bhavcopy_date)

        T = max((expiry - trade_date).days / 365, 1 / 365)
        r = RISK_FREE_RATE

        iv = implied_volatility(market_price, spot, K, T, r, option_type)

        if np.isnan(iv) or iv <= 0:
            sigma = stats["historical_vol"]
        else:
            sigma = iv

        chain_out = chain.copy()
        chain_out["spot_used"] = spot
        chain_out["time_to_expiry"] = T
        chain_out["implied_volatility"] = chain_out.apply(
            lambda row: implied_volatility(
                row["market_price"],
                spot,
                row["strike"],
                T,
                r,
                row["option_type"]
            ),
            axis=1
        )

        chain_out.to_csv(f"{DATA_DIR}/{name}_option_chain_from_nse_bhavcopy.csv", index=False)

        bs = black_scholes_price(spot, K, T, r, sigma, option_type)
        bt = binomial_tree_price(spot, K, T, r, sigma, 500, option_type)
        mc, mc_err = monte_carlo_price(spot, K, T, r, sigma, option_type)

        delta, gamma, vega, theta, rho = calculate_greeks(spot, K, T, r, sigma, option_type)

        scenario_df = scenario_analysis(spot, K, T, r, sigma, option_type)
        scenario_tables[name] = scenario_df
        scenario_df.to_csv(f"{DATA_DIR}/{name}_scenario_analysis.csv", index=False)

        hedge_df = delta_hedging_backtest(price_df, K, r, sigma, min(T + 120 / 252, 1), option_type)
        hedge_df.to_csv(f"{DATA_DIR}/{name}_hedging_backtest.csv", index=False)

        plot_payoff(
            spot,
            K,
            market_price,
            option_type,
            f"{PLOT_DIR}/{name}_payoff.png",
            f"{name} Option Payoff"
        )

        plot_greeks(
            spot,
            K,
            T,
            r,
            sigma,
            option_type,
            f"{PLOT_DIR}/{name}_greeks.png",
            f"{name} Greeks Sensitivity"
        )

        plot_vol_surface(
            chain_out,
            f"{PLOT_DIR}/{name}_volatility_surface.png",
            f"{name} Implied Volatility Smile"
        )

        plot_scenarios(
            scenario_df,
            f"{PLOT_DIR}/{name}_scenario_analysis.png",
            f"{name} Scenario Analysis"
        )

        plot_hedging(
            hedge_df,
            f"{PLOT_DIR}/{name}_hedging_error.png",
            f"{name} Delta Hedging Error"
        )

        summary_rows.append({
            "instrument": name,
            "bhavcopy_date": str(bhavcopy_date),
            "expiry": expiry.date(),
            "option_type": option_type,
            "spot": round(spot, 2),
            "strike": round(K, 2),
            "market_price": round(market_price, 2),
            "historical_vol": round(stats["historical_vol"], 4),
            "implied_vol": round(sigma, 4),
            "black_scholes": round(bs, 2),
            "binomial": round(bt, 2),
            "monte_carlo": round(mc, 2),
            "mc_stderr": round(mc_err, 4),
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "vega": round(vega, 4),
            "theta": round(theta, 4),
            "rho": round(rho, 4),
            "annualized_return": round(stats["annualized_return"], 4),
            "skewness": round(stats["skewness"], 4),
            "kurtosis": round(stats["kurtosis"], 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{OUTPUT_DIR}/pricing_summary.csv", index=False)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary_df))
    width = 0.25
    plt.bar(x - width, summary_df["black_scholes"], width, label="Black-Scholes")
    plt.bar(x, summary_df["binomial"], width, label="Binomial Tree")
    plt.bar(x + width, summary_df["monte_carlo"], width, label="Monte Carlo")
    plt.xticks(x, summary_df["instrument"])
    plt.ylabel("Option Price")
    plt.title("Pricing Model Comparison")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/pricing_comparison.png")
    plt.close()

    generate_report(summary_df, scenario_tables, bhavcopy_date, source_url)

    print("\nProject run completed.")
    print(f"Summary saved to: {OUTPUT_DIR}/pricing_summary.csv")
    print(f"Report saved to: {OUTPUT_DIR}/research_report.md")
    print(f"Plots saved to: {PLOT_DIR}")
    print(f"Data saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
