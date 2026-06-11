import os
from tabulate import tabulate


def generate_markdown_report(summary_df, scenario_tables, output_path):
    """
    Generate a Markdown research report after model execution.
    """

    lines = []

    lines.append("# Advanced Derivatives Pricing and Risk Modeling Framework")
    lines.append("")
    lines.append("## Market")
    lines.append("")
    lines.append("Indian NSE derivatives market.")
    lines.append("")
    lines.append("## Instruments")
    lines.append("")
    lines.append("The framework analyzes liquid NSE derivatives on NIFTY, BANKNIFTY, and RELIANCE.")
    lines.append("")
    lines.append("## Data Sources")
    lines.append("")
    lines.append("- NSE option-chain data")
    lines.append("- Yahoo Finance historical price data through yfinance")
    lines.append("")
    lines.append("## Mathematical Framework")
    lines.append("")
    lines.append("### Geometric Brownian Motion")
    lines.append("")
    lines.append("The underlying asset price is assumed to follow:")
    lines.append("")
    lines.append("```text")
    lines.append("dS_t = mu S_t dt + sigma S_t dW_t")
    lines.append("```")
    lines.append("")
    lines.append("Under the risk-neutral measure:")
    lines.append("")
    lines.append("```text")
    lines.append("dS_t = r S_t dt + sigma S_t dW_t")
    lines.append("```")
    lines.append("")
    lines.append("### Black-Scholes Pricing")
    lines.append("")
    lines.append("For a European call option:")
    lines.append("")
    lines.append("```text")
    lines.append("C = S N(d1) - K e^(-rT) N(d2)")
    lines.append("```")
    lines.append("")
    lines.append("For a European put option:")
    lines.append("")
    lines.append("```text")
    lines.append("P = K e^(-rT) N(-d2) - S N(-d1)")
    lines.append("```")
    lines.append("")
    lines.append("where:")
    lines.append("")
    lines.append("```text")
    lines.append("d1 = [ln(S/K) + (r + sigma^2 / 2)T] / [sigma sqrt(T)]")
    lines.append("d2 = d1 - sigma sqrt(T)")
    lines.append("```")
    lines.append("")
    lines.append("## Black-Scholes Assumptions")
    lines.append("")
    lines.append("- The underlying follows geometric Brownian motion.")
    lines.append("- Volatility is constant.")
    lines.append("- Risk-free rate is constant.")
    lines.append("- Markets are frictionless.")
    lines.append("- No arbitrage opportunity exists.")
    lines.append("- Continuous trading is possible.")
    lines.append("- No transaction costs or taxes.")
    lines.append("- Options are European-style.")
    lines.append("")
    lines.append("## Greeks Interpretation")
    lines.append("")
    lines.append("- Delta measures sensitivity to changes in the underlying price.")
    lines.append("- Gamma measures sensitivity of Delta to changes in the underlying price.")
    lines.append("- Vega measures sensitivity to changes in volatility.")
    lines.append("- Theta measures time decay.")
    lines.append("- Rho measures sensitivity to changes in the risk-free rate.")
    lines.append("")
    lines.append("## Pricing Model Comparison")
    lines.append("")
    lines.append(tabulate(summary_df, headers="keys", tablefmt="github", showindex=False))
    lines.append("")
    lines.append("## Scenario Analysis")
    lines.append("")

    for instrument_name, scenario_df in scenario_tables.items():
        lines.append(f"### {instrument_name}")
        lines.append("")
        lines.append(tabulate(scenario_df, headers="keys", tablefmt="github", showindex=False))
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Black-Scholes provides a closed-form theoretical price under strict assumptions. "
        "The binomial tree model is more flexible and can be extended to American-style options. "
        "Monte Carlo pricing is useful for complex payoff structures but has simulation error."
    )
    lines.append("")
    lines.append(
        "Differences between market price and model price may arise due to volatility smile effects, "
        "liquidity conditions, bid-ask spread, supply-demand imbalance, discrete dividends, and model assumptions."
    )
    lines.append("")
    lines.append("## Hedging Framework")
    lines.append("")
    lines.append(
        "The project implements dynamic delta hedging. The hedge position is rebalanced daily using the Black-Scholes Delta. "
        "The hedging error measures the difference between the hedging portfolio value and theoretical option value."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- NSE public endpoints may block automated requests depending on cookies, headers, or IP behavior.")
    lines.append("- NSE option-chain data is a live snapshot, not a complete historical options database.")
    lines.append("- Black-Scholes assumes constant volatility and European exercise.")
    lines.append("- Transaction costs, slippage, taxes, liquidity effects, and bid-ask spread are excluded.")
    lines.append("- Risk-free rate is manually configured.")
    lines.append("- Dividend effects are not explicitly modeled.")
    lines.append("")
    lines.append("## Future Improvements")
    lines.append("")
    lines.append("- Add paid historical options data vendor support.")
    lines.append("- Add dividend-adjusted pricing.")
    lines.append("- Implement Heston stochastic volatility model.")
    lines.append("- Implement jump-diffusion pricing.")
    lines.append("- Add transaction-cost-aware delta hedging.")
    lines.append("- Add portfolio-level VaR and Expected Shortfall.")
    lines.append("- Build a Streamlit dashboard.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
