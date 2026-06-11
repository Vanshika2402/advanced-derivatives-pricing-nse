import os
import pandas as pd
from datetime import datetime

from config import (
    INSTRUMENTS,
    START_DATE,
    END_DATE,
    RISK_FREE_RATE,
    TRADING_DAYS,
    MONTE_CARLO_PATHS,
    MONTE_CARLO_STEPS,
    OUTPUT_DIR,
    PLOT_DIR,
    DATA_DIR
)

from src.data_loader import NSEDataLoader, fetch_historical_prices
from src.volatility import historical_volatility, return_statistics
from src.black_scholes import black_scholes_price
from src.binomial_tree import binomial_tree_price
from src.monte_carlo import monte_carlo_price
from src.greeks import greeks
from src.scenarios import scenario_analysis
from src.hedging import delta_hedging_backtest
from src.visualization import (
    ensure_dir,
    plot_payoff,
    plot_pricing_comparison,
    plot_greek_sensitivity,
    plot_volatility_surface,
    plot_scenarios,
    plot_hedging
)
from src.report_generator import generate_markdown_report


def days_to_expiry(expiry_string):
    expiry_date = datetime.strptime(expiry_string, "%d-%b-%Y")
    today = datetime.now()
    days = max((expiry_date - today).days, 1)
    return days / 365


def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PLOT_DIR)
    ensure_dir(DATA_DIR)

    nse = NSEDataLoader()

    summary_rows = []
    scenario_tables = {}

    for instrument in INSTRUMENTS:
        name = instrument["name"]
        print(f"\nProcessing {name}...")

        raw_chain = nse.fetch_option_chain(
            instrument["nse_symbol"],
            instrument["type"]
        )

        atm_option, chain_df = nse.parse_nearest_atm_option(
            raw_chain,
            option_type=instrument["option_type"]
        )

        chain_path = f"{DATA_DIR}/{name}_option_chain.csv"
        chain_df.to_csv(chain_path, index=False)

        price_df = fetch_historical_prices(
            instrument["yahoo_symbol"],
            START_DATE,
            END_DATE
        )

        price_path = f"{DATA_DIR}/{name}_historical_prices.csv"
        price_df.to_csv(price_path, index=False)

        S = float(atm_option["underlying"])
        K = float(atm_option["strike"])
        market_price = float(atm_option["last_price"])
        option_type = atm_option["option_type"]
        T = days_to_expiry(atm_option["expiry"])
        r = RISK_FREE_RATE

        hist_vol = historical_volatility(price_df, TRADING_DAYS)

        iv_raw = atm_option.get("implied_volatility")
        if iv_raw is not None and iv_raw > 0:
            sigma = float(iv_raw) / 100
        else:
            sigma = hist_vol

        ret_stats = return_statistics(price_df)

        bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
        bin_price = binomial_tree_price(S, K, T, r, sigma, 500, option_type)
        mc_price, mc_stderr = monte_carlo_price(
            S,
            K,
            T,
            r,
            sigma,
            MONTE_CARLO_PATHS,
            MONTE_CARLO_STEPS,
            option_type
        )

        g = greeks(S, K, T, r, sigma, option_type)

        summary_rows.append({
            "instrument": name,
            "expiry": atm_option["expiry"],
            "option_type": option_type,
            "spot": round(S, 2),
            "strike": round(K, 2),
            "market_price": round(market_price, 2),
            "historical_vol": round(hist_vol, 4),
            "implied_vol": round(sigma, 4),
            "black_scholes": round(bs_price, 2),
            "binomial": round(bin_price, 2),
            "monte_carlo": round(mc_price, 2),
            "mc_stderr": round(mc_stderr, 4),
            "delta": round(g["delta"], 4),
            "gamma": round(g["gamma"], 6),
            "vega": round(g["vega_per_1pct"], 4),
            "theta": round(g["theta_per_day"], 4),
            "rho": round(g["rho_per_1pct"], 4),
            "annualized_return": round(ret_stats["annualized_return"], 4),
            "skewness": round(ret_stats["skewness"], 4),
            "kurtosis": round(ret_stats["kurtosis"], 4)
        })

        plot_payoff(
            S,
            K,
            market_price,
            option_type,
            f"{name} Option Payoff",
            f"{PLOT_DIR}/{name}_payoff.png"
        )

        plot_greek_sensitivity(
            S,
            K,
            T,
            r,
            sigma,
            option_type,
            f"{PLOT_DIR}/{name}_greeks.png"
        )

        plot_volatility_surface(
            chain_df,
            f"{PLOT_DIR}/{name}_volatility_surface.png"
        )

        sc_df = scenario_analysis(S, K, T, r, sigma, option_type)
        scenario_tables[name] = sc_df
        sc_df.to_csv(f"{DATA_DIR}/{name}_scenario_analysis.csv", index=False)

        plot_scenarios(
            sc_df,
            f"{name} Scenario Analysis",
            f"{PLOT_DIR}/{name}_scenarios.png"
        )

        try:
            hedge_df = delta_hedging_backtest(
                price_df,
                K,
                r,
                sigma,
                min(T + 120 / 252, 1),
                option_type
            )
            hedge_df.to_csv(f"{DATA_DIR}/{name}_hedging_backtest.csv", index=False)

            plot_hedging(
                hedge_df,
                f"{name} Delta Hedging Error",
                f"{PLOT_DIR}/{name}_hedging_error.png"
            )
        except Exception as exc:
            print(f"Hedging backtest skipped for {name}: {exc}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{OUTPUT_DIR}/pricing_summary.csv", index=False)

    plot_pricing_comparison(
        summary_df,
        f"{PLOT_DIR}/pricing_comparison.png"
    )

    generate_markdown_report(
        summary_df,
        scenario_tables,
        f"{OUTPUT_DIR}/research_report.md"
    )

    print("\nProject run completed.")
    print(f"Summary saved to: {OUTPUT_DIR}/pricing_summary.csv")
    print(f"Report saved to: {OUTPUT_DIR}/research_report.md")
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
