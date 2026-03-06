import yfinance as yf
import pandas as pd

def RSI(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * series.diff().clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def Bollinger_Bands(series, period=20, std=2):
    ma = series.rolling(period).mean()
    rolling_std = series.rolling(period).std()
    upper = ma + std * rolling_std
    lower = ma - std * rolling_std
    return upper, lower


def ATR(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


TICKERS = {
    "gold": "GC=F",
    "sp500": "^GSPC",
    "usd": "DX-Y.NYB",
    "oil": "CL=F",
    "vix": "^VIX",
    "bond_yield": "^TNX",
}


def load_gold_data(start_date="2000-01-01"):
    data = pd.DataFrame()
    gold_high = None
    gold_low = None

    for name, ticker in TICKERS.items():
        df = yf.download(ticker, start=start_date)
        data[name] = df["Close"]
        if name == "gold":
            gold_high = df["High"]
            gold_low = df["Low"]

    if gold_high is None or gold_low is None:
        gold_df = yf.download(TICKERS["gold"], start=start_date)
        gold_high = gold_df["High"]
        gold_low = gold_df["Low"]

    data = data.dropna()
    high = gold_high.reindex(data.index)
    low = gold_low.reindex(data.index)
    return data, high, low


def build_dataset(start_date="2000-01-01"):
    data, high, low = load_gold_data(start_date)

    # Base returns
    returns = data.pct_change()

    # Lag features for gold
    for lag in range(1, 21):
        returns[f"gold_lag_{lag}"] = returns["gold"].shift(lag)

    # Momentum
    returns["momentum_5"] = data["gold"].pct_change(5)
    returns["momentum_20"] = data["gold"].pct_change(20)
    returns["momentum_50"] = data["gold"].pct_change(50)

    # Moving averages
    data["ma20"] = data["gold"].rolling(20).mean()
    data["ma50"] = data["gold"].rolling(50).mean()
    data["ma100"] = data["gold"].rolling(100).mean()
    returns["ma20"] = data["ma20"].pct_change()
    returns["ma50"] = data["ma50"].pct_change()
    returns["ma100"] = data["ma100"].pct_change()

    # Volatility
    returns["volatility"] = returns["gold"].rolling(20).std()

    # Technical indicators on price
    returns["RSI"] = RSI(data["gold"], 14)
    returns["MACD"], returns["MACD_signal"] = MACD(data["gold"])
    returns["BB_upper"], returns["BB_lower"] = Bollinger_Bands(data["gold"])
    returns["ATR"] = ATR(high, low, data["gold"], 14)

    # Targets
    returns["target_1d"] = (data["gold"].shift(-1) > data["gold"]).astype(int)
    returns["target_10d"] = (data["gold"].shift(-10) > data["gold"]).astype(int)
    returns["target_30d"] = (data["gold"].shift(-30) > data["gold"]).astype(int)

    dataset = returns.dropna()
    return dataset

