# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import pytz
import time

# === Configuration ===
st.set_page_config(page_title="Scanner Forex Pro - Twelve Data", layout="wide")

# Configuration de l'API
@st.cache_data
def get_api_config():
    return {
        "base_url": "https://api.twelvedata.com",
        "pairs": [
            "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
            "USD/CAD", "USD/CHF", "NZD/USD", "XAU/USD"
        ],
        "timeframe": "1h"
    }

# === Gestion des secrets ===
def get_api_key():
    """Récupère la clé API depuis les secrets Streamlit"""
    try:
        return st.secrets["TWELVE_DATA_API_KEY"]
    except KeyError:
        st.error("❌ Clé API manquante ! Ajoutez TWELVE_DATA_API_KEY dans les secrets Streamlit.")
        st.stop()

# === Fonctions améliorées ===
@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def fetch_data(pair: str, interval: str = "1h", outputsize: int = 50, api_key: str = None):
    """Récupère les données OHLC via Twelve Data API avec gestion d'erreurs"""
    if not api_key:
        return None, "Clé API manquante"
    
    config = get_api_config()
    url = f"{config['base_url']}/time_series"
    params = {
        "symbol": pair,
        "interval": interval,
        "apikey": api_key,
        "outputsize": outputsize,
        "format": "json"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Gestion des erreurs API
        if "code" in data and data["code"] != 200:
            return None, f"Erreur API: {data.get('message', 'Erreur inconnue')}"
        
        if "values" not in data or not data["values"]:
            return None, "Aucune donnée disponible"
        
        # Création du DataFrame
        df = pd.DataFrame(data["values"])
        df.columns = ["datetime", "open", "high", "low", "close", "volume"]
        
        # Conversion des types
        numeric_cols = ["open", "high", "low", "close"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce')
        df = df.dropna()  # Supprime les lignes avec des valeurs manquantes
        
        if df.empty:
            return None, "Données invalides après nettoyage"
        
        df.set_index("datetime", inplace=True)
        df = df.sort_index()  # Tri par date
        
        return df, None
        
    except requests.exceptions.Timeout:
        return None, "Timeout de la requête API"
    except requests.exceptions.RequestException as e:
        return None, f"Erreur de connexion: {str(e)}"
    except Exception as e:
        return None, f"Erreur inattendue: {str(e)}"


def calculate_indicators(df):
    """Calcule les indicateurs techniques avec gestion des erreurs"""
    try:
        df = df.copy()
        
        # Rendements et volatilité
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=min(10, len(df)-1)).std() * np.sqrt(10)

        # EMA avec gestion des périodes courtes
        ema_20_period = min(20, len(df)-1)
        ema_50_period = min(50, len(df)-1)
        
        if ema_20_period > 0:
            df['ema_20'] = df['close'].ewm(span=ema_20_period, adjust=False).mean()
        if ema_50_period > 0:
            df['ema_50'] = df['close'].ewm(span=ema_50_period, adjust=False).mean()

        # RSI avec gestion des périodes courtes
        rsi_period = min(14, len(df)-1)
        if rsi_period > 0:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            
            # Éviter la division par zéro
            rs = avg_gain / (avg_loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        if len(df) >= 26:
            df['macd_line'] = (df['close'].ewm(span=12, adjust=False).mean() - 
                              df['close'].ewm(span=26, adjust=False).mean())
            df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd_line'] - df['signal_line']

        return df
        
    except Exception as e:
        st.error(f"Erreur lors du calcul des indicateurs: {str(e)}")
        return df


def is_valid_setup(df):
    """Vérifie si on a un setup de volatilité valide"""
    if len(df) < 5:  # Minimum de données requis
        return {"valid": False, "direction": "N/A", "score": 0, "reason": "Données insuffisantes"}

    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Vérifications avec gestion des valeurs manquantes
        conditions = {
            "trend_bullish": False,
            "price_up": False,
            "rsi_ok": False,
            "macd_ok": False,
            "volatility_high": False
        }

        # Trend (EMA)
        if 'ema_20' in df.columns and 'ema_50' in df.columns:
            if not (pd.isna(last['ema_20']) or pd.isna(last['ema_50'])):
                conditions["trend_bullish"] = last['ema_20'] > last['ema_50']

        # Prix
        if not (pd.isna(last['close']) or pd.isna(prev['close'])):
            conditions["price_up"] = last['close'] > prev['close']

        # RSI
        if 'rsi' in df.columns and not pd.isna(last['rsi']):
            if conditions["trend_bullish"]:
                conditions["rsi_ok"] = last['rsi'] < 70
            else:
                conditions["rsi_ok"] = last['rsi'] > 30

        # MACD
        if 'macd_line' in df.columns and 'signal_line' in df.columns:
            if not (pd.isna(last['macd_line']) or pd.isna(last['signal_line'])):
                if conditions["trend_bullish"]:
                    conditions["macd_ok"] = last['macd_line'] > last['signal_line']
                else:
                    conditions["macd_ok"] = last['macd_line'] < last['signal_line']

        # Volatilité
        if 'volatility' in df.columns and not pd.isna(last['volatility']):
            vol_mean = df['volatility'].mean()
            if not pd.isna(vol_mean):
                conditions["volatility_high"] = last['volatility'] > vol_mean * 1.2

        score = sum(conditions.values())
        all_valid = all(conditions.values())

        return {
            "valid": all_valid,
            "direction": "Long" if conditions["trend_bullish"] else "Short",
            "score": score,
            "conditions": conditions
        }
        
    except Exception as e:
        return {"valid": False, "direction": "Error", "score": 0, "reason": str(e)}


def plot_chart(df, pair, signal_info):
    """Affiche le graphique interactif avec Plotly"""
    try:
        fig = go.Figure()
        
        # Chandelier
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Prix',
            showlegend=False
        ))
        
        # EMA si disponibles
        if 'ema_20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['ema_20'], 
                mode='lines', 
                name='EMA 20',
                line=dict(color='blue', width=1)
            ))
        
        if 'ema_50' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['ema_50'], 
                mode='lines', 
                name='EMA 50',
                line=dict(color='red', width=1)
            ))

        # Configuration du layout
        direction_color = "green" if signal_info['direction'] == "Long" else "red"
        fig.update_layout(
            title=f"📊 {pair} | Signal: {signal_info['direction']} | Score: {signal_info['score']}/5",
            xaxis_title="Date",
            yaxis_title="Prix",
            xaxis_rangeslider_visible=False,
            height=500,
            title_font_color=direction_color,
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique pour {pair}: {str(e)}")
        return None


# === Interface Streamlit ===
def main():
    st.title("🔍 Scanner Intraday Pro – Volatilité dans le Bon Sens")
    st.markdown("Application développée avec [Twelve Data](https://twelvedata.com/) pour repérer les setups de volatilité directionnels.")
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = get_api_key()
        
        config = get_api_config()
        selected_pairs = st.multiselect(
            "Paires à analyser :",
            options=config["pairs"],
            default=["EUR/USD", "XAU/USD"]
        )
        
        outputsize = st.slider("Nombre de bougies:", 30, 100, 50)
        
        st.info("ℹ️ Le cache est actif pendant 5 minutes pour éviter les appels API excessifs.")

    # Interface principale
    col1, col2 = st.columns([3, 1])
    
    with col1:
        scan_button = st.button("🚀 Lancer le Scan", type="primary")
    
    with col2:
        if st.button("🔄 Vider le Cache"):
            st.cache_data.clear()
            st.success("Cache vidé !")

    if scan_button and selected_pairs:
        results = []
        errors = []
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, pair in enumerate(selected_pairs):
            status_text.text(f"📡 Analyse en cours : {pair}")
            
            # Récupération des données
            df, error = fetch_data(pair, config["timeframe"], outputsize, api_key)
            
            if error:
                errors.append(f"{pair}: {error}")
                progress_bar.progress((i + 1) / len(selected_pairs))
                continue
            
            if df is not None and not df.empty:
                # Calcul des indicateurs
                df = calculate_indicators(df)
                signal = is_valid_setup(df)
                
                if signal["valid"]:
                    chart = plot_chart(df, pair, signal)
                    if chart:
                        results.append({
                            "pair": pair,
                            "signal": signal,
                            "chart": chart,
                            "last_price": df['close'].iloc[-1]
                        })
                
            progress_bar.progress((i + 1) / len(selected_pairs))
            
            # Pause pour éviter le rate limiting
            time.sleep(0.5)

        # Affichage des résultats
        status_text.text("✅ Scan terminé.")
        
        # Erreurs
        if errors:
            with st.expander("⚠️ Erreurs rencontrées"):
                for error in errors:
                    st.warning(error)
        
        # Résultats valides
        if results:
            st.success(f"✅ {len(results)} setup(s) trouvé(s) :")
            
            # Tableau récapitulatif
            summary_data = []
            for res in results:
                summary_data.append({
                    "Paire": res["pair"],
                    "Direction": res["signal"]["direction"],
                    "Score": f"{res['signal']['score']}/5",
                    "Prix Actuel": f"{res['last_price']:.5f}"
                })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            # Graphiques
            st.header("📈 Graphiques détaillés")
            for res in results:
                st.plotly_chart(res["chart"], use_container_width=True)
                
        else:
            st.info("❌ Aucun setup valide détecté pour les paires sélectionnées.")
    
    elif scan_button and not selected_pairs:
        st.warning("⚠️ Veuillez sélectionner au moins une paire à analyser.")
    
    else:
        st.info("👉 Sélectionnez vos paires et cliquez sur 'Lancer le Scan' pour commencer.")
        
        # Instructions pour la configuration
        with st.expander("📋 Instructions de configuration"):
            st.markdown("""
            ### Configuration de l'API Twelve Data
            
            1. **Clé API** : Ajoutez votre clé API dans les secrets Streamlit :
               - Allez dans Settings > Secrets
               - Ajoutez : `TWELVE_DATA_API_KEY = "votre_cle_api"`
            
            2. **Limits API** : 
               - Version gratuite : 800 requêtes/jour
               - L'application utilise un cache de 5 minutes pour optimiser l'usage
            
            3. **Paires supportées** : Forex majeures et métaux précieux
            """)

if __name__ == "__main__":
    main()
