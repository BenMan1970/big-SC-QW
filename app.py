import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import json
import time

# Configuration
API_KEY = "YOUR_TWELVE_DATA_API_KEY"  # À remplacer au déploiement
BASE_URL = "https://api.twelvedata.com" 

# Liste des paires forex + XAU/USD
PAIRS = {
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
    "USD/JPY": "USD/JPY",
    "AUD/USD": "AUD/USD",
    "USD/CAD": "USD/CAD",
    "USD/CHF": "USD/CHF",
    "NZD/USD": "NZD/USD",
    "XAU/USD": "XAU/USD"
}

TIMEFRAMES = ["1min", "5min", "15min", "30min", "60min"]

INDICATORS = {
    "EMA": "ema",
    "RSI": "rsi",
    "MACD": "macd",
    "Bollinger Bands": "bbands"
}

# === Fonctions utilitaires ===
def fetch_data(symbol, interval="60min", outputsize=50):
    url = f"{BASE_URL}/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "type": "price",
        "apikey": API_KEY,
        "outputsize": outputsize,
        "format": "json"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "meta" in data and "values" in data:
            df = pd.DataFrame(data["values"])
            df.columns = ["datetime", "open", "high", "low", "close", "volume"]
            df["close"] = df["close"].astype(float)
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            return df
    return None

def calculate_indicators(df):
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=10).std() * np.sqrt(10)

    # EMA
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    df['macd_line'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['signal_line']

    return df

def is_volatility_in_direction(df):
    last_row = df.iloc[-1]
    second_last = df.iloc[-2]

    trend_bullish = last_row['ema_20'] > last_row['ema_50']
    rsi_ok = last_row['rsi'] < 70 if trend_bullish else last_row['rsi'] > 30

    price_up = last_row['close'] > second_last['close']
    macd_ok = (df['macd_line'].iloc[-1] > df['signal_line'].iloc[-1]) if trend_bullish \
        else (df['macd_line'].iloc[-1] < df['signal_line'].iloc[-1])

    volatility_high = last_row['volatility'] > df['volatility'].mean() * 1.2

    return {
        'valid': trend_bullish and rsi_ok and price_up and macd_ok and volatility_high,
        'direction': "Long" if trend_bullish else "Short",
        'score': sum([trend_bullish, rsi_ok, price_up, macd_ok, volatility_high])
    }

# === Interface Streamlit ===
st.set_page_config(page_title="Scanner Pro Volatilité", layout="wide")
st.title("📈 Scanner Intraday Pro – Volatilité dans le Bon Sens")

col1, col2 = st.columns(2)
with col1:
    selected_pairs = st.multiselect("Paires à analyser", options=list(PAIRS.keys()), default=["EUR/USD", "XAU/USD"])

with col2:
    selected_timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=4)

if st.button("🔍 Lancer le scan"):
    with st.spinner("Analyse en cours..."):

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, pair in enumerate(selected_pairs):
            symbol = PAIRS[pair]
            status_text.text(f"Analyse de {symbol}...")
            df = fetch_data(symbol, interval=selected_timeframe)
            if df is not None and not df.empty:
                df = calculate_indicators(df)
                signal = is_volatility_in_direction(df)

                if signal['valid']:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Candles'
                    ))
                    fig.add_trace(go.Scatter(x=df.index, y=df['ema_20'], mode='lines', name='EMA 20'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['ema_50'], mode='lines', name='EMA 50'))

                    fig.update_layout(title=f"{symbol} | Signal: {signal['direction']} | Score: {signal['score']}/5",
                                      xaxis_rangeslider_visible=False, height=300)

                    results.append({
                        "pair": symbol,
                        "signal": signal['direction'],
                        "score": signal['score'],
                        "chart": fig
                    })

            progress_bar.progress((i + 1) / len(selected_pairs))

        status_text.text("Scan terminé.")

        if results:
            st.success("✅ Setup(s) trouvé(s) :")
            for res in results:
                st.plotly_chart(res["chart"], use_container_width=True)
        else:
            st.info("❌ Aucun setup valide détecté.")

else:
    st.info("👉 Cliquez sur 'Lancer le scan' pour commencer.")
