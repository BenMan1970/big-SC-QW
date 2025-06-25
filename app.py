import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time

# --- Page config ---
st.set_page_config(page_title="Scanner Forex Pro", layout="wide")

# --- Liste des paires ---
PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CAD", "USD/CHF", "NZD/USD", "XAU/USD"
]

# --- API Key ---
def get_api_key():
    try:
        return st.secrets["TWELVE_DATA_API_KEY"]
    except:
        st.error("❌ Clé API manquante. Ajoutez-la dans les secrets.")
        return st.text_input("Saisissez votre clé API Twelve Data :", type="password")

# --- Récupération des données ---
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

        if "status" in data and data["status"] == "error":
            return None, f"Erreur API : {data.get('message', 'Erreur inconnue')}"
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
        return None, f"Erreur de téléchargement : {str(e)}"

# --- Indicateurs techniques ---
def calculate_indicators(df):
    if df.empty or len(df) < 25:
        return df

    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()

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

# --- Analyse du signal ---
def analyze_signal(df):
    if df is None or len(df) < 25:
        return {"valid": False, "direction": "N/A", "score": 0}

    last = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]

    # Critères professionnels
    trend_up = last['ema_20'] > last['ema_50']
    ema_slope_ok = df['ema_20'].iloc[-1] > df['ema_20'].iloc[-2] > df['ema_20'].iloc[-3]
    rsi_ok = (50 < last['rsi'] < 70) if trend_up else (30 < last['rsi'] < 50)
    macd_ok = (last['macd'] > last['signal']) and ((last['macd'] - last['signal']) > (prev1['macd'] - prev1['signal']))
    vol_ok = last['volatility'] > df['volatility'].rolling(20).mean().iloc[-1]
    return_ok = last['returns'] > 0 if trend_up else last['returns'] < 0

    score = sum([trend_up and ema_slope_ok, rsi_ok, macd_ok, vol_ok and return_ok])

    return {
        "valid": score == 4,
        "direction": "LONG" if trend_up else "SHORT",
        "score": score,
        "rsi": last['rsi'],
        "price": last['close']
    }

# --- Graphique ---
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
        title=f"{pair} – {signal['direction']} – Score {signal['score']}/4",
        height=400,
        xaxis_rangeslider_visible=False
    )
    return fig

# --- Application principale ---
def main():
    st.title("🔍 Scanner Intraday A+ – Forex & Or")
    st.markdown("Ce scanner détecte les **meilleurs setups de confluence** sur 1H grâce aux indicateurs professionnels.")

    api_key = get_api_key()
    if not api_key:
        st.stop()

    st.subheader("⚙️ Configuration")
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

            if error or df is None or df.empty or len(df) < 25:
                errors.append(f"{pair} : {error or 'Pas assez de données'}")
                progress.progress((i + 1) / len(selected_pairs))
                continue

            try:
                df = calculate_indicators(df)
                signal = analyze_signal(df)

                if signal["valid"]:
                    chart = create_chart(df, pair, signal)
                    results.append({
                        "pair": pair,
                        "signal": signal,
                        "chart": chart
                    })

            except Exception as e:
                errors.append(f"{pair} : erreur de traitement – {str(e)}")

            progress.progress((i + 1) / len(selected_pairs))
            time.sleep(0.5)

        status.text("✅ Scan terminé !")

        if errors:
            st.error("Erreurs rencontrées :")
            for e in errors:
                st.text("❌ " + e)

        if results:
            st.success(f"📈 {len(results)} signal(s) A+ détecté(s) !")
            table = [{
                "Paire": r["pair"],
                "Direction": r["signal"]["direction"],
                "Score": f"{r['signal']['score']}/4",
                "RSI": f"{r['signal']['rsi']:.1f}",
                "Prix": f"{r['signal']['price']:.5f}"
            } for r in results]

            st.dataframe(table, use_container_width=True)

            st.subheader("📉 Graphiques")
            for r in results:
                st.plotly_chart(r["chart"], use_container_width=True)
        else:
            st.info("ℹ️ Aucun signal A+ détecté pour les paires sélectionnées.")

    else:
        st.info("🟢 Cliquez sur 'LANCER LE SCAN' pour commencer.")

    with st.expander("📘 Aide & critères"):
        st.markdown("""
        ### Ce que détecte le scanner :
        - **Tendance forte** : EMA 20 au-dessus de la 50 + pente haussière
        - **Momentum clair** : RSI dans zone active et MACD confirmé
        - **Volatilité directionnelle** : pas de range
        - **Score requis** : 4/4 pour valider un signal A+

        ➕ Utilisez ce scanner en complément de vos analyses manuelles.
        """)

if __name__ == "__main__":
    main()
