import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

# Liste des paires
PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CAD", "USD/CHF", "NZD/USD", "XAU/USD"
]

# Configuration Streamlit
st.set_page_config(page_title="Scanner Intraday A+", layout="wide")
st.title("🚀 Scanner Intraday A+ – Momentum & Volatilité")
st.markdown("Scan automatique de toutes les paires en 1H pour détecter les **meilleures opportunités (4/4)**.")

# Récupération de la clé API
def get_api_key():
    try:
        return st.secrets["TWELVE_DATA_API_KEY"]
    except:
        st.error("❌ Clé API manquante. Ajoutez-la dans les secrets.")
        return st.text_input("Entrez votre clé API Twelve Data :", type="password")

api_key = get_api_key()
if not api_key:
    st.stop()

# Récupération des données OHLC
def fetch_data(pair, api_key, interval="1h", outputsize=50):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": pair,
        "interval": interval,
        "apikey": api_key,
        "outputsize": outputsize,
        "format": "json"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "values" not in data:
            return None
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        return df
    except:
        return None

# Calcul des indicateurs
def calculate_indicators(df):
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std() * 100

    # ADX simplifié (approximatif pour détection de force de trend)
    df["adx"] = df["volatility"].rolling(14).mean()

    return df

# Analyse du signal
def analyze(df):
    if df is None or len(df) < 25:
        return None

    last = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]

    ema_up = last["ema_20"] > last["ema_50"]
    ema_slope = last["ema_20"] > prev1["ema_20"] > prev2["ema_20"]

    rsi_valid = (55 < last["rsi"] < 70) if ema_up else (30 < last["rsi"] < 45)
    adx_valid = last["adx"] > 20
    macd_valid = (last["macd"] > last["macd_signal"]) and (last["macd"] > prev1["macd"])

    score = sum([ema_up and ema_slope, rsi_valid, adx_valid, macd_valid])

    return {
        "valid": score == 4,
        "score": score,
        "direction": "LONG" if ema_up else "SHORT",
        "rsi": round(last["rsi"], 1),
        "adx": round(last["adx"], 1),
        "volatility": round(last["volatility"], 2),
        "price": round(last["close"], 5)
    }

# --- Scan automatique ---
st.markdown("### 🧠 Résultats du scan en 1H (dernières 50 bougies)")

results = []
progress = st.progress(0)
for i, pair in enumerate(PAIRS):
    df = fetch_data(pair, api_key, "1h", 50)
    if df is None or df.empty:
        progress.progress((i + 1) / len(PAIRS))
        continue

    df = calculate_indicators(df)
    signal = analyze(df)

    if signal and signal["valid"]:
        results.append({
            "Paire": pair,
            "Direction": signal["direction"],
            "Score": f"{signal['score']}/4",
            "RSI": signal["rsi"],
            "ADX": signal["adx"],
            "Volatilité": f"{signal['volatility']}%",
            "Prix": signal["price"]
        })

    progress.progress((i + 1) / len(PAIRS))
    time.sleep(0.3)

if results:
    st.success(f"✅ {len(results)} signal(s) A+ détecté(s) sur {len(PAIRS)}")
    st.dataframe(results, use_container_width=True)
else:
    st.info("ℹ️ Aucun signal A+ détecté pour le moment.")

# --- Instructions ---
with st.expander("📘 Critères utilisés pour un signal A+ (Score 4/4)"):
    st.markdown("""
    - 📈 **Tendance claire** : EMA 20 > EMA 50 + pente ascendante
    - 🔍 **Momentum valide** : RSI dans une zone directionnelle (55–70 ou 30–45)
    - 🔥 **Volatilité réelle** : ADX > 20 basé sur std dev (approché)
    - 💡 **Confirmation** : MACD > Signal + MACD croissant

    > ⚠️ Ces signaux sont filtrés pour l'intraday (1h), pour éviter les faux signaux.
    """)


