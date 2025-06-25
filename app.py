# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import time

# === Configuration ===
st.set_page_config(page_title="Scanner Forex Pro", layout="wide")

# Configuration de l'API
PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CAD", "USD/CHF", "NZD/USD", "XAU/USD"
]

# === Gestion des secrets ===
def get_api_key():
    """Récupère la clé API depuis les secrets Streamlit"""
    try:
        return st.secrets["TWELVE_DATA_API_KEY"]
    except:
        st.error("❌ Clé API manquante ! Configurez TWELVE_DATA_API_KEY dans les secrets.")
        st.info("Pour tester, vous pouvez temporairement entrer votre clé API ci-dessous :")
        api_key = st.text_input("Clé API Twelve Data :", type="password")
        return api_key if api_key else None

# === Fonctions ===
def fetch_data(pair: str, api_key: str, interval: str = "1h", outputsize: int = 50):
    """Récupère les données OHLC via Twelve Data API"""
    if not api_key:
        return None, "Clé API manquante"
    
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
        
        # Création du DataFrame
        df = pd.DataFrame(data["values"])
        df.columns = ["datetime", "open", "high", "low", "close", "volume"]
        
        # Conversion des types
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.dropna()
        df.set_index("datetime", inplace=True)
        df = df.sort_index()
        
        return df, None
        
    except Exception as e:
        return None, f"Erreur: {str(e)}"

def calculate_indicators(df):
    """Calcule les indicateurs techniques"""
    if len(df) < 20:
        return df
    
    # EMA
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=min(50, len(df))).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9).mean()
    
    # Volatilité
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(10).std() * np.sqrt(10)
    
    return df

def analyze_signal(df):
    """Analyse le signal de trading"""
    if len(df) < 20:
        return {"valid": False, "direction": "N/A", "score": 0}
    
    last = df.iloc[-1]
    
    # Conditions
    trend_up = last['ema_20'] > last['ema_50']
    rsi_ok = 30 < last['rsi'] < 70
    macd_bullish = last['macd'] > last['signal']
    vol_high = last['volatility'] > df['volatility'].mean()
    
    score = sum([trend_up, rsi_ok, macd_bullish, vol_high])
    
    return {
        "valid": score >= 3,
        "direction": "LONG" if trend_up else "SHORT",
        "score": score,
        "rsi": last['rsi'],
        "price": last['close']
    }

def create_chart(df, pair, signal):
    """Crée le graphique"""
    fig = go.Figure()
    
    # Chandelier
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Prix'
    ))
    
    # EMA
    if 'ema_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ema_20'], name='EMA 20', line=dict(color='blue')))
    if 'ema_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ema_50'], name='EMA 50', line=dict(color='red')))
    
    fig.update_layout(
        title=f"{pair} - {signal['direction']} - Score: {signal['score']}/4",
        height=400,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# === Interface Streamlit ===
def main():
    st.title("🔍 Scanner Intraday Pro – Volatilité dans le Bon Sens")
    st.markdown("Application développée avec [Twelve Data](https://twelvedata.com/) pour repérer les setups de volatilité directionnels.")
    
    # Récupération de la clé API
    api_key = get_api_key()
    
    if not api_key:
        st.stop()
    
    st.markdown("---")
    
    # Configuration simple
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_pairs = st.multiselect(
            "Sélectionnez les paires à analyser :",
            options=PAIRS,
            default=["EUR/USD", "GBP/USD", "XAU/USD"]
        )
    
    with col2:
        outputsize = st.selectbox("Nombre de bougies :", [30, 50, 100], index=1)
    
    st.markdown("---")
    
    # Bouton de scan
    if st.button("🚀 LANCER LE SCAN", type="primary", use_container_width=True):
        if not selected_pairs:
            st.warning("⚠️ Veuillez sélectionner au moins une paire")
            return
        
        results = []
        errors = []
        
        # Progress bar
        progress = st.progress(0)
        status = st.empty()
        
        for i, pair in enumerate(selected_pairs):
            status.text(f"📡 Analyse de {pair}...")
            
            # Récupération des données
            df, error = fetch_data(pair, api_key, "1h", outputsize)
            
            if error:
                errors.append(f"❌ {pair}: {error}")
            elif df is not None and len(df) > 20:
                # Calcul des indicateurs
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
            time.sleep(0.5)  # Éviter le rate limiting
        
        status.text("✅ Scan terminé !")
        
        # Affichage des erreurs
        if errors:
            st.error("Erreurs rencontrées :")
            for error in errors:
                st.text(error)
        
        # Affichage des résultats
        if results:
            st.success(f"🎯 {len(results)} signal(s) détecté(s) !")
            
            # Tableau récapitulatif
            summary_data = []
            for res in results:
                summary_data.append({
                    "Paire": res["pair"],
                    "Direction": res["signal"]["direction"],
                    "Score": f"{res['signal']['score']}/4",
                    "RSI": f"{res['signal']['rsi']:.1f}",
                    "Prix": f"{res['signal']['price']:.5f}"
                })
            
            st.dataframe(summary_data, use_container_width=True)
            
            # Graphiques
            st.subheader("📈 Graphiques détaillés")
            for res in results:
                st.plotly_chart(res["chart"], use_container_width=True)
        else:
            st.info("ℹ️ Aucun signal valide trouvé pour les paires sélectionnées")
    
    else:
        st.info("👆 Configurez vos paramètres et cliquez sur 'LANCER LE SCAN'")
    
    # Instructions
    with st.expander("📋 Comment utiliser l'application"):
        st.markdown("""
        ### Étapes à suivre :
        
        1. **Configurez votre clé API** Twelve Data dans les secrets Streamlit
        2. **Sélectionnez les paires** que vous voulez analyser  
        3. **Choisissez le nombre de bougies** à analyser
        4. **Cliquez sur 'LANCER LE SCAN'**
        
        ### Critères de signal :
        - **Trend** : EMA 20 > EMA 50 (haussier) ou EMA 20 < EMA 50 (baissier)
        - **RSI** : Entre 30 et 70 (éviter les extrêmes)
        - **MACD** : MACD > Signal (haussier) ou MACD < Signal (baissier)
        - **Volatilité** : Supérieure à la moyenne
        
        **Score minimum requis : 3/4**
        """)

if __name__ == "__main__":
    main()
