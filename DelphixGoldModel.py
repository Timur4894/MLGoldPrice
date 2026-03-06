import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# --------- Technical indicators helpers ---------
#Relative Strength Index
def RSI(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    RS = roll_up / roll_down
    return 100 - (100 / (1 + RS))

#Moving Average Convergence Divergence
def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

#Bollinger Bands
def Bollinger_Bands(series, period=20, std=2):
    ma = series.rolling(period).mean()
    upper = ma + std*series.rolling(period).std()
    lower = ma - std*series.rolling(period).std()
    return upper, lower

#Average True Range
def ATR(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# --------- Download data ---------
tickers = {
    "gold": "GC=F",
    "sp500": "^GSPC",
    "usd": "DX-Y.NYB",
    "oil": "CL=F",
    "vix": "^VIX",
    "bond_yield": "^TNX"
}

data = pd.DataFrame()

for name, ticker in tickers.items():
    df = yf.download(ticker, start="2000-01-01")
    data[name] = df['Close']

# Optionally save high/low for ATR
high = yf.download("GC=F", start="2000-01-01")['High']
low = yf.download("GC=F", start="2000-01-01")['Low']

data = data.dropna()

# --------- Features ---------
returns = data.pct_change().dropna()

# Lag features for gold
for lag in range(1,6):
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

# RSI
returns["RSI"] = RSI(data["gold"], 14)

# MACD
returns["MACD"], returns["MACD_signal"] = MACD(data["gold"])

# Bollinger Bands
returns["BB_upper"], returns["BB_lower"] = Bollinger_Bands(data["gold"])

# ATR
returns["ATR"] = ATR(high, low, data["gold"], 14)

# Target: next day direction
returns["target"] = (returns["gold"].shift(-1) > 0).astype(int)

dataset = returns.dropna()

# --------- Features & Target ---------
X = dataset.drop("target", axis=1)
y = dataset["target"]

# --------- Walk-forward validation ---------
initial_train_size = int(len(X)*0.8)
test_size = 15
predictions = []
y_tests = []

# ----------------- Choose model -----------------
# Uncomment one of these:

# 1️⃣ XGBoost
# import xgboost as xgb
# model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, n_jobs=-1)

# 2️⃣ LightGBM
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.05, n_jobs=-1)

# 3️⃣ CatBoost
# from catboost import CatBoostClassifier
# model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, verbose=0)

for end_train in range(initial_train_size, len(X)-test_size+1, test_size):
    X_train = X.iloc[:end_train]
    y_train = y.iloc[:end_train]
    X_test = X.iloc[end_train:end_train+test_size]
    y_test = y.iloc[end_train:end_train+test_size]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    predictions.extend(pred)
    y_tests.extend(y_test)

accuracy = accuracy_score(y_tests, predictions)
print("Walk-forward accuracy:", accuracy)


#xgboost - Walk-forward accuracy: 0.5354581673306773
#LightGBM - Walk-forward accuracy: 0.5450199203187251
#CatBoost - Walk-forward accuracy: 0.5370517928286852

