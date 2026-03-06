import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

# Модели
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ------------------------- Технические индикаторы -------------------------
def RSI(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    RS = roll_up / roll_down
    return 100 - (100 / (1 + RS))

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def Bollinger_Bands(series, period=20, std=2):
    ma = series.rolling(period).mean()
    upper = ma + std*series.rolling(period).std()
    lower = ma - std*series.rolling(period).std()
    return upper, lower

def ATR(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ------------------------- Загрузка данных -------------------------
tickers = {
    "gold": "GC=F",
    "sp500": "^GSPC",
    "usd": "DX-Y.NYB",
    "oil": "CL=F",
    "vix": "^VIX",
    "bond_yield": "^TNX"
}

data = pd.DataFrame()
high_low = pd.DataFrame()  # для ATR

for name, ticker in tickers.items():
    df = yf.download(ticker, start="2000-01-01")
    data[name] = df['Close']
    if name == 'gold':
        high_low['high'] = df['High']
        high_low['low'] = df['Low']

data = data.dropna()

# ------------------------- Feature Engineering -------------------------
returns = data.pct_change().dropna()

# Lag features золота
for lag in range(1, 6):
    returns[f"gold_lag_{lag}"] = returns["gold"].shift(lag)

# Lag features других активов
for ticker_name in ['sp500','usd','oil','vix','bond_yield']:
    for lag in range(1,4):
        returns[f"{ticker_name}_lag_{lag}"] = returns[ticker_name].shift(lag)

# Momentum
for period in [5, 20, 50]:
    returns[f"momentum_{period}"] = data['gold'].pct_change(period)

# Скользящие средние
for ma in [20, 50, 100]:
    data[f"ma{ma}"] = data['gold'].rolling(ma).mean()
    returns[f"ma{ma}"] = data[f"ma{ma}"].pct_change()

# Volatility
returns['volatility'] = returns['gold'].rolling(20).std()

# RSI
returns['RSI'] = RSI(data['gold'], 14)

# MACD
returns['MACD'], returns['MACD_signal'] = MACD(data['gold'])

# Bollinger Bands
returns['BB_upper'], returns['BB_lower'] = Bollinger_Bands(data['gold'])

# ATR
returns['ATR'] = ATR(high_low['high'], high_low['low'], data['gold'], 14)

# Target
returns['target'] = (returns['gold'].shift(-1) > 0).astype(int)
dataset = returns.dropna()

# ------------------------- Features & Target -------------------------
X = dataset.drop('target', axis=1)
y = dataset['target']

# ------------------------- Walk-forward параметры -------------------------
initial_train_size = int(len(X) * 0.8)
test_size = 5
predictions = []
y_tests = []

# ------------------------- GridSearchCV для XGBoost -------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

# Берем первый блок для подбора параметров
X_grid = X.iloc[:initial_train_size]
y_grid = y.iloc[:initial_train_size]

grid = GridSearchCV(XGBClassifier(n_jobs=-1), param_grid, cv=3, scoring='accuracy', verbose=1)
grid.fit(X_grid, y_grid)
best_params = grid.best_params_

# ------------------------- Инициализация моделей -------------------------
xgb_model = XGBClassifier(**best_params, n_jobs=-1)
lgb_model = LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.05, n_jobs=-1)
cat_model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, verbose=0)

ensemble = VotingClassifier(
    estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('cat', cat_model)],
    voting='soft'
)

# ------------------------- Walk-forward -------------------------
for end_train in range(initial_train_size, len(X)-test_size+1, test_size):
    X_train = X.iloc[:end_train]
    y_train = y.iloc[:end_train]
    X_test = X.iloc[end_train:end_train+test_size]
    y_test = y.iloc[end_train:end_train+test_size]

    ensemble.fit(X_train, y_train)
    pred = ensemble.predict(X_test)

    predictions.extend(pred)
    y_tests.extend(y_test)

# ------------------------- Accuracy -------------------------
accuracy = accuracy_score(y_tests, predictions)
print("Walk-forward accuracy (Ensemble XGB+LGB+CatBoost):", accuracy)