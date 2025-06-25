import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
import time

# Configuration de la page
st.set_page_config(page_title="Scanner Forex Pro", layout="wide")

# Liste des paires à analyser
PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CAD", "USD/CHF", "NZD/USD", "XAU/USD"
]

# === Clé API ===
def get_api_key():
    try:
        return st.secrets["TWELVE_DATA_API_KEY"]
    except:
        st.error("❌ Clé API manquante. Configurez TWELVE_DATA_API_KEY dans les secrets.")
        st.info("Ou saisissez-la ici pour un test temporaire :")
        return st.text_input("Clé API Twelve Data :", type="password")

# === Téléchargement des données ===
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
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if "code" in data and data["code"] != 200:
            return None, f"Erreur API: {data.get('message', 'Erreur inconnue')}"
        if "values" not in data:
            return None, "Aucune donnée disponible"

        df = pd.DataFrame(data["values"])

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df["datetime"] = pd.to_datetime(df["datetime"])
        df.dropna(inplace=True)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

        return df, None

    except Exception as e:
        return None, f"Erreur: {str(e)}"

# === Indicateurs ===
def calculate_indicators(df):
    if len(df) < 20:
        return df

    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=min(50, len(df))).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9).mean()

    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(10).std() * np.sqrt(10)

    return df

# === Analyse du signal ===
def analyze_signal(df):
    if len(df) < 20:
        return {"valid": False, "direction": "N/A", "score": 0}

    last = df.iloc[-1]
    trend_up = last['ema_20'] > last['ema_50']
    rsi_ok = 30 < last['rsi'] < 70
    macd_ok = last['macd'] > last['signal']
    vol_high = last['volatility'] > df['volatility'].mean()

    score = sum([trend_up, rsi_ok, macd_ok, vol_high])

    return {
        "valid": score >= 3,
        "direction": "LONG" if trend_up else "SHORT",
        "score": score,
        "rsi": last['rsi'],
        "price": last['close']
    }

# === Graphique Plotly ===
def create_chart(df, pair, signal):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='Prix'
    ))

    if 'ema_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ema_20'], name='EMA 20', line=dict(color='blue')))
    if 'ema_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ema_50'], name='EMA 50', line=dict(color='red')))

    fig.update_layout(
        title=f"{pair} – {signal['direction']} – Score: {signal['score']}/4",
        height=400,
        xaxis_rangeslider_visible=False
    )
    return fig

# === Application principale ===
def main():
    st.title("📊 Scanner Intraday Pro – Forex & Or (XAU)")
    st.markdown("Analyse technique automatisée via [Twelve Data](https://twelvedata.com/) pour identifier les meilleures opportunités.")

    api_key = get_api_key()
    if not api_key:
        st.stop()

    st.subheader("⚙️ Paramètres")
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_pairs = st.multiselect("Sélectionnez les paires à analyser :", PAIRS, default=["EUR/USD", "GBP/USD", "XAU/USD"])
    with col2:
        outputsize = st.selectbox("Nombre de bougies :", [30, 50, 100], index=1)

    if st.button("🚀 LANCER LE SCAN", type="primary", use_container_width=True):
        if not selected_pairs:
            st.warning("⚠️ Veuillez sélectionner au moins une paire.")
            return

        results, errors = [], []
        progress = st.progress(0)
        status = st.empty()

        for i, pair in enumerate(selected_pairs):
            status.text(f"🔄 Analyse de {pair}...")
            df, error = fetch_data(pair, api_key, "1h", outputsize)

            if error:
                errors.append(f"{pair} : {error}")
            elif df is not None:
                df = calculate_indicators(df)
                signal = analyze_signal(df)

                if signal["valid"]:
                    chart = create_chart(df, pair, signal)
                    results.append({
                        "pair": pair,
                        "signal": signal,
                        "chart": chart
                    })

            progress.progress((i + 1) / len(selected_pairs))
            time.sleep(0.5)

        status.text("✅ Scan terminé !")

        if errors:
            st.error("Erreurs rencontrées :")
            for e in errors:
                st.text("❌ " + e)

        if results:
            st.success(f"📈 {len(results)} signal(s) détecté(s) !")

            table_data = [{
                "Paire": r["pair"],
                "Direction": r["signal"]["direction"],
                "Score": f"{r['signal']['score']}/4",
                "RSI": f"{r['signal']['rsi']:.1f}",
                "Prix": f"{r['signal']['price']:.5f}"
            } for r in results]

            st.dataframe(table_data, use_container_width=True)

            st.subheader("📉 Graphiques")
            for r in results:
                st.plotly_chart(r["chart"], use_container_width=True)
        else:
            st.info("ℹ️ Aucun signal valide trouvé pour les paires sélectionnées.")

    else:
        st.info("🟢 Cliquez sur 'LANCER LE SCAN' après avoir configuré les paramètres.")

    with st.expander("📘 Aide & critères de signal"):
        st.markdown("""
        ### Critères utilisés :
        - **Trend haussier** : EMA 20 > EMA 50
        - **RSI entre 30 et 70** : pas de surachat/survente
        - **MACD haussier** : MACD > Signal
        - **Volatilité > moyenne** : filtre de dynamique

        ➕ Un signal est considéré **valide** s’il obtient au moins **3/4** de ces critères.
        """)

if __name__ == "__main__":
    main()
