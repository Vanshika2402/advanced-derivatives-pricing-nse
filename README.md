# Advanced Derivatives Pricing and Risk Modeling Framework

## Project Objective

This project develops a quantitative research framework for pricing NSE derivative instruments, analyzing risk exposures, calculating Greeks, evaluating volatility behavior, and testing delta hedging effectiveness under different market conditions.

## Market

Indian NSE derivatives market.

## Instruments Used

The project uses real market data for:

1. NIFTY option
2. BANKNIFTY option
3. RELIANCE option

## Models Implemented

- Black-Scholes option pricing model
- Cox-Ross-Rubinstein binomial tree model
- Monte Carlo option pricing model
- Greeks calculation
- Volatility and return analysis
- Scenario analysis
- Delta hedging backtest

## Data Sources

- NSE option-chain data
- Yahoo Finance historical price data through yfinance

## Project Structure

```text
advanced-derivatives-pricing-nse/
│
├── README.md
├── requirements.txt
├── config.py
├── main.py
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── volatility.py
│   ├── black_scholes.py
│   ├── binomial_tree.py
│   ├── monte_carlo.py
│   ├── greeks.py
│   ├── scenarios.py
│   ├── hedging.py
│   ├── visualization.py
│   └── report_generator.py
│
├── data/
│
└── outputs/
    └── plots/
