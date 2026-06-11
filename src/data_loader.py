import time
import requests
import pandas as pd
import yfinance as yf


try:
    import nsepythonserver as nse
except Exception:
    nse = None


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

        Method 1: nsepythonserver option-chain scraper
        Method 2: nsepythonserver oi_chain_builder
        Method 3: direct NSE endpoint fallback
        """

        errors = []

        if nse is not None:
            scraper = getattr(nse, "nse_optionchain_scrapper", None)

            if scraper is not None:
                try:
                    payload = scraper(symbol)
                    if payload and isinstance(payload, dict) and "records" in payload:
                        return {
                            "source": "raw_json",
                            "payload": payload
                        }
                except Exception as exc:
                    errors.append(f"nse_optionchain_scrapper failed: {exc}")

            oi_chain_builder = getattr(nse, "oi_chain_builder", None)
            expiry_list = getattr(nse, "expiry_list", None)

            if oi_chain_builder is not None:
                try:
                    oi_df, ltp, crontime = oi_chain_builder(symbol, "latest", "full")

                    expiry = None
                    if expiry_list is not None:
                        try:
                            exp = expiry_list(symbol, "list")
                            if isinstance(exp, list) and len(exp) > 0:
                                expiry = exp[0]
                        except Exception:
                            expiry = None

                    if expiry is None:
                        expiry = "latest"

                    if isinstance(oi_df, pd.DataFrame) and not oi_df.empty:
                        return {
                            "source": "oi_chain_builder",
                            "payload": oi_df,
                            "underlying": float(ltp),
                            "expiry": expiry
                        }
                except Exception as exc:
                    errors.append(f"oi_chain_builder failed: {exc}")

        urls = []

        if instrument_type == "index":
            urls.append(f"{self.base_url}/api/option-chain-indices?symbol={symbol}")
            if symbol == "NIFTY":
                urls.append(f"{self.base_url}/api/option-chain-indices?symbol=NIFTY%2050")
            if symbol == "BANKNIFTY":
                urls.append(f"{self.base_url}/api/option-chain-indices?symbol=NIFTY%20BANK")
        else:
            urls.append(f"{self.base_url}/api/option-chain-equities?symbol={symbol}")

        for url in urls:
            try:
                response = self.session.get(url, timeout=20)

                if response.status_code != 200:
                    self._set_cookies()
                    response = self.session.get(url, timeout=20)

                response.raise_for_status()
                payload = response.json()

                if payload and "records" in payload:
                    return {
                        "source": "raw_json",
                        "payload": payload
                    }

            except Exception as exc:
                errors.append(f"direct NSE failed for {url}: {exc}")

        raise RuntimeError(
            "Unable to fetch NSE option-chain data. "
            "Try running again after some time. Errors: " + " | ".join(errors)
        )

    def _find_column(self, df, keywords):
        for col in df.columns:
            col_lower = str(col).lower()
            if all(keyword.lower() in col_lower for keyword in keywords):
                return col
        return None

    def _to_number(self, value):
        try:
            if pd.isna(value):
                return None
            return float(str(value).replace(",", "").strip())
        except Exception:
            return None

    def _parse_oi_chain_builder(self, raw_json, option_type="CE"):
        oi_df = raw_json["payload"].copy()
        underlying = raw_json["underlying"]
        expiry = raw_json["expiry"]

        side = "CALLS" if option_type == "CE" else "PUTS"

        strike_col = self._find_column(oi_df, ["strike"])
        ltp_col = self._find_column(oi_df, [side, "ltp"])
        iv_col = self._find_column(oi_df, [side, "iv"])
        volume_col = self._find_column(oi_df, [side, "volume"])
        bid_col = self._find_column(oi_df, [side, "bid", "price"])
        ask_col = self._find_column(oi_df, [side, "ask", "price"])

        oi_col = self._find_column(oi_df, [side, "oi"])
        change_oi_col = (
            self._find_column(oi_df, [side, "chng", "oi"])
            or self._find_column(oi_df, [side, "change", "oi"])
        )

        if strike_col is None:
            raise ValueError("Strike price column not found in oi_chain_builder data.")

        rows = []

        for _, row in oi_df.iterrows():
            strike = self._to_number(row.get(strike_col))

            if strike is None:
                continue

            rows.append({
                "expiry": expiry,
                "strike": strike,
                "option_type": option_type,
                "underlying": underlying,
                "last_price": self._to_number(row.get(ltp_col)) if ltp_col else None,
                "bid": self._to_number(row.get(bid_col)) if bid_col else None,
                "ask": self._to_number(row.get(ask_col)) if ask_col else None,
                "implied_volatility": self._to_number(row.get(iv_col)) if iv_col else None,
                "open_interest": self._to_number(row.get(oi_col)) if oi_col else None,
                "change_oi": self._to_number(row.get(change_oi_col)) if change_oi_col else None,
                "volume": self._to_number(row.get(volume_col)) if volume_col else None
            })

        df = pd.DataFrame(rows)

        if df.empty:
            raise ValueError("No valid data found from oi_chain_builder.")

        df["atm_distance"] = (df["strike"] - underlying).abs()
        atm_option = df.sort_values("atm_distance").iloc[0].to_dict()

        return atm_option, df

    def _parse_raw_json(self, raw_json, option_type="CE"):
        payload = raw_json["payload"]

        records = payload["records"]["data"]

        if "underlyingValue" in payload["records"]:
            underlying = payload["records"]["underlyingValue"]
        else:
            values = []
            for row in records:
                if option_type in row and "underlyingValue" in row[option_type]:
                    values.append(row[option_type]["underlyingValue"])

            if not values:
                raise ValueError("Underlying value not found in NSE data.")

            underlying = values[0]

        expiries = payload["records"]["expiryDates"]

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

    def parse_nearest_atm_option(self, raw_json, option_type="CE"):
        if raw_json["source"] == "raw_json":
            return self._parse_raw_json(raw_json, option_type)

        if raw_json["source"] == "oi_chain_builder":
            return self._parse_oi_chain_builder(raw_json, option_type)

        raise ValueError("Unknown NSE data source format.")


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
