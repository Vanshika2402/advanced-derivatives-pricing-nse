INSTRUMENTS = [
    {
        "name": "NIFTY",
        "type": "index",
        "nse_symbol": "NIFTY",
        "yahoo_symbol": "^NSEI",
        "option_type": "CE"
    },
    {
        "name": "BANKNIFTY",
        "type": "index",
        "nse_symbol": "BANKNIFTY",
        "yahoo_symbol": "^NSEBANK",
        "option_type": "CE"
    },
    {
        "name": "RELIANCE",
        "type": "equity",
        "nse_symbol": "RELIANCE",
        "yahoo_symbol": "RELIANCE.NS",
        "option_type": "CE"
    }
]

START_DATE = "2024-01-01"
END_DATE = None

RISK_FREE_RATE = 0.068
TRADING_DAYS = 252

MONTE_CARLO_PATHS = 50000
MONTE_CARLO_STEPS = 252

OUTPUT_DIR = "outputs"
PLOT_DIR = "outputs/plots"
DATA_DIR = "data"
