import time
import requests
import pandas as pd
import yfinance as yf


try:
    from nsepythonserver import option_chain as nse_option_chain
except Exception:
    nse_option_chain = None


class NSEDataLoader:
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/option-chain",
            "Connection": "keep-alive"
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self._set_cookies()

    def _set_cookies(self):
        try:
            self.session.get(self.base_url, timeout=10)
            time.sleep(1)
        except Exception as exc:
            print(f"NSE cookie setup warning: {exc}")

    def fetch_option_chain(self, symbol, instrument_type="index"):
        """
        Fetch live NSE option-chain data.

        Primary method:
        - nsepythonserver, better for Google Colab/server environments.

        Backup method:
        - Direct NSE endpoint.
        """

        # Method 1: nsepythonserver
        if nse_option_chain is not None:
            try:
                payload = nse_option_chain(symbol)
                if payload and "records" in payload:
                    return payload
            except Exception as exc:
                print(f"nsepythonserver failed for {symbol}: {exc}")

        # Method 2: direct NSE endpoint fallback
        if instrument_type == "index":
            url = f"{self.base_url}/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"{self.base_url}/api/option-chain-equities?symbol={symbol}"

        response = self.session.get(url, timeout=20)

        if response.status_code != 200:
            self._set_cookies()
            response = self.session.get(url, timeout=20)

        response.raise_for_status()
        return response.json()

    def parse_nearest_atm_option(self, raw_json, option_type="CE"):
        """
        Select nearest-expiry ATM option.

        CE = Call option
        PE = Put option
        """

        records = raw_json["records"]["data"]

        if "underlyingValue" in raw_json["records"]:
            underlying = raw_json["records"]["underlyingValue"]
        else:
            valid_underlying_values = []

            for row in records:
                if option_type in row and "underlyingValue" in row[option_type]:
                    valid_underlying_values.append(row[option_type]["underlyingValue"])

            if not valid_underlying_values:
                raise ValueError("Underlying value not found in NSE option-chain data.")

            underlying = valid_underlying_values[0]

        expiries = raw_json["records"]["expiryDates"]

        if not expiries:
            raise ValueError("No expiry dates found in NSE option chain.")

        nearest_expiry = expiries[0]

        rows = []

        for row in records:
            if row.get("expiryDate") != nearest_expiry:
                continue

            strike = row.get("strikePrice")
            option_data = row.get(option_type)

            if option_data:
                rows.append({
                    "expiry": nearest_expiry,
                    "strike": strike,
                    "option_type": option_type,
                    "underlying": underlying,
                    "last_price": option_data.get("lastPrice"),
                    "bid": option_data.get("bidprice"),
                    "ask": option_data.get("askPrice"),
                    "implied_volatility": option_data.get("impliedVolatility"),
                    "open_interest": option_data.get("openInterest"),
                    "change_oi": option_data.get("changeinOpenInterest"),
                    "volume": option_data.get("totalTradedVolume")
                })

        df = pd.DataFrame(rows)

        if df.empty:
            raise ValueError("No valid option-chain data found.")

        df["atm_distance"] = (df["strike"] - underlying).abs()

        atm_option = df.sort_values("atm_distance").iloc[0].to_dict()

        return atm_option, df


def fetch_historical_prices(yahoo_symbol, start_date, end_date=None):
    """
    Fetch historical price data using Yahoo Finance.
    """

    df = yf.download(
        yahoo_symbol,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No historical data found for {yahoo_symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df.rename(columns={"Date": "date", "Close": "close"})

    df = df[["date", "close"]].dropna()

    return df
