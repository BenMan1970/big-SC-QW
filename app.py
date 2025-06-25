# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import pytz

# === Configuration ===
API_KEY = "YOUR_TWELVE_DATA_API_KEY"  # Remplacer par la clé réelle ou via .env
BASE_URL = "https://api.twelvedata.com" 

# Liste des paires à scanner
PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CAD", "USD/CHF", "NZD/USD", "XAU/USD"
]

# Timeframe à utiliser
TIMEFRAME = "1h"

# === Fonctions ===
def fetch_data(pair: str, interval: str = "1h", outputsize: int = 50):
    """Récupère les données OHLC via Twelve Data API"""
    url = f"{BASE_URL}/time_series"
    params = {
        "symbol": pair,
        "interval": interval,
        "type": "price",
        "apikey": API_KEY,
        "outputsize": outputsize,
        "format": "json"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "values" in data:
            df = pd.DataFrame(data["values"])
            df.columns = ["datetime", "open", "high", "low", "close", "volume"]
            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            return df
    return None


def calculate_indicators(df):
    """Calcule les indicateurs techniques nécessaires"""
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


def is_valid_setup(df):
    """Vérifie si on a un setup de volatilité dans le bon sens"""
    if len(df) < 2:
        return {"valid": False}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    trend_bullish = last['ema_20'] > last['ema_50']
    price_up = last['close'] > prev['close']
    rsi_ok = last['rsi'] < 70 if trend_bullish else last['rsi'] > 30
    macd_ok = last['macd_line'] > last['signal_line'] if trend_bullish else last['macd_line'] < last['signal_line']
    volatility_high = last['volatility'] > df['volatility'].mean() * 1.2

    score = sum([trend_bullish, price_up, rsi_ok, macd_ok, volatility_high])

    return {
        "valid": all([trend_bullish, price_up, rsi_ok, macd_ok, volatility_high]),
        "direction": "Long" if trend_bullish else "Short",
        "score": score
    }


def plot_chart(df, pair):
    """Affiche le graphique interactif avec Plotly"""
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

    fig.update_layout(
        title=f"{pair} | Signal: {is_valid_setup(df)['direction']} | Score: {is_valid_setup(df)['score']}/5",
        xaxis_title="Date",
        yaxis_title="Prix",
        xaxis_rangeslider_visible=False,
        height=400
    )
    return fig


# === Interface Streamlit ===
st.set_page_config(page_title="Scanner Forex Pro - Twelve Data", layout="wide")
st.title("🔍 Scanner Intraday Pro – Volatilité dans le Bon Sens")

st.markdown("Application développée avec [Twelve Data](https://twelvedata.com/)  pour repérer les setups de volatilité directionnels.")

selected_pairs = st.multiselect("Paires à analyser :", options=PAIRS, default=["EUR/USD", "XAU/USD"])

if st.button("🚀 Lancer le Scan"):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, pair in enumerate(selected_pairs):
        status_text.text(f"Analyse en cours : {pair}")
        df = fetch_data(pair, TIMEFRAME)
        if df is not None and not df.empty:
            df = calculate_indicators(df)
            signal = is_valid_setup(df)
            if signal["valid"]:
                fig = plot_chart(df, pair)
                results.append({"pair": pair, "signal": signal, "chart": fig})
        progress_bar.progress((i + 1) / len(selected_pairs))

    status_text.text("Scan terminé.")
    if results:
        st.success("✅ Setup(s) trouvé(s) :")
        for res in results:
            st.plotly_chart(res["chart"], use_container_width=True)
    else:
        st.info("❌ Aucun setup valide détecté.")

else:
    st.info("👉 Cliquez sur 'Lancer le Scan' pour commencer.")
