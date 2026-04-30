import streamlit as st
import pandas as pd
import xgboost as xgb
import pandas_ta_classic as ta
import plotly.graph_objects as go
from datetime import timedelta
import psycopg2

# --- 1. SPRACH-EINSTELLUNGEN ---
# Hier definieren wir alle Texte für die App
languages = {
    "Deutsch": {
        "title": "🏛️ Professional AI Procurement Radar (Quant Edition)",
        "subtitle": "**KI-Modell:** Robustes XGBoost | **Daten:** Makroökonomie, Marktpsychologie & Alternative Data",
        "sidebar_status": "🟢 Verbunden mit Supabase Cloud-DB",
        "sidebar_error": "❌ Datenbank-Fehler",
        "spinner": "Lade Daten & berechne KI-Prognose...",
        "metric_live": "Live Kupferpreis",
        "metric_pred": "KI-Prognose (7 Tage)",
        "status_buy": "🚨 KAUF-ALARM (Makro-gestützt)",
        "status_wait": "💎 WARTEN (Preisverfall erwartet)",
        "status_stable": "⚖️ MARKT STABIL",
        "chart_title": "90-Tage Historie & KI-Vorhersage",
        "chart_real": "Realer Preis",
        "chart_pred": "Quant-Prognose"
    },
    "English": {
        "title": "🏛️ Professional AI Procurement Radar (Quant Edition)",
        "subtitle": "**AI Model:** Robust XGBoost | **Data:** Macroeconomics, Market Sentiment & Alternative Data",
        "sidebar_status": "🟢 Connected to Supabase Cloud-DB",
        "sidebar_error": "❌ Database Error",
        "spinner": "Loading data & calculating AI forecast...",
        "metric_live": "Live Copper Price",
        "metric_pred": "AI Forecast (7 Days)",
        "status_buy": "🚨 BUY ALARM (Macro-supported)",
        "status_wait": "💎 WAIT (Price drop expected)",
        "status_stable": "⚖️ MARKET STABLE",
        "chart_title": "90-Day History & AI Prediction",
        "chart_real": "Real Price",
        "chart_pred": "Quant Forecast"
    }
}

# --- 2. UI SETUP & SPRACHWAHL ---
st.set_page_config(page_title="AI Procurement Radar", layout="wide")

# Sprachauswahl in der Sidebar ganz oben
selected_lang = st.sidebar.selectbox("Language / Sprache", ["Deutsch", "English"])
t = languages[selected_lang] # 't' ist jetzt unser Kurz-Weg für 'translations'

st.title(t["title"])
st.markdown(t["subtitle"])

# --- 3. DATEN LADEFUNKTION ---
@st.cache_data(ttl=3600) 
def lade_daten_aus_supabase():
    conn = psycopg2.connect(st.secrets["DB_URI"])
    query = "SELECT * FROM kupfer_historie ORDER BY datum ASC;"
    df = pd.read_sql_query(query, conn, index_col="datum", parse_dates=["datum"])
    conn.close()
    return df

@st.cache_data(ttl=3600)
def load_and_train_quant_model():
    historie_df = lade_daten_aus_supabase()
    
    df = pd.DataFrame({
        'Preis': historie_df['kupfer_preis'], 'SP500': historie_df['sp500'], 'Oel': historie_df['oel'], 
        'DXY': historie_df['dxy'], 'CNY': historie_df['cny'], 
        'Minen_Aktien': historie_df['copx'], 'Zinsen': historie_df['tnx']
    }).ffill()
    
    df['Preis_in_7_Tagen'] = df['Preis'].shift(-7)
    df['Target_Delta'] = df['Preis_in_7_Tagen'] - df['Preis']
    df['Preis_Gestern'] = df['Preis'].shift(1)
    df['SMA_14'] = df['Preis'].rolling(window=14).mean()
    
    # Trends
    for col in ['SP500', 'Oel', 'DXY', 'Minen_Aktien', 'Zinsen']:
        df[f'{col}_Trend'] = df[col] - df[col].shift(7)
    
    df.ta.rsi(close='Preis', length=14, append=True)
    df.ta.macd(close='Preis', append=True)
    
    features = [
        'Preis', 'Preis_Gestern', 'SMA_14', 'SP500', 'Oel', 'SP500_Trend', 'Oel_Trend',
        'DXY', 'CNY', 'DXY_Trend', 'RSI_14', 'MACD_12_26_9',
        'Minen_Aktien', 'Zinsen', 'Minen_Trend', 'Zins_Trend'
    ]
    
    heute_features = df[features].iloc[-1:]
    aktueller_preis = df['Preis'].iloc[-1]
    
    train_df = df.dropna()
    model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.01, max_depth=5, random_state=42)
    model.fit(train_df[features], train_df['Target_Delta'])
    
    predicted_delta = model.predict(heute_features)[0]
    return aktueller_preis, aktueller_preis + predicted_delta, predicted_delta, df

try:
    with st.spinner(t["spinner"]):
        aktueller_preis, vorhersage_preis, delta, df = load_and_train_quant_model()
    st.sidebar.success(t["sidebar_status"])

except Exception as e:
    st.sidebar.error(f"{t['sidebar_error']}: {e}")
    st.stop() 

# --- 4. UI DASHBOARD ---
st.write("---")
col1, col2, col3 = st.columns(3)
prozent = (delta / aktueller_preis) * 100

with col1:
    st.metric(t["metric_live"], f"${aktueller_preis:.4f}")
with col2:
    st.metric(t["metric_pred"], f"${vorhersage_preis:.4f}", f"{prozent:.2f}%")
with col3:
    if prozent > 0.5:
        st.error(t["status_buy"])
    elif prozent < -0.5:
        st.success(t["status_wait"])
    else:
        st.info(t["status_stable"])

# --- 5. CHART ---
st.write("---")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-90:], y=df['Preis'].iloc[-90:], name=t["chart_real"], line=dict(color='royalblue')))
fig.add_trace(go.Scatter(x=[df.index[-1], df.index[-1] + timedelta(days=7)], 
                         y=[aktueller_preis, vorhersage_preis], 
                         name=t["chart_pred"], line=dict(color='red', width=3, dash='dash')))
fig.update_layout(height=500, title=t["chart_title"], xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig, use_container_width=True)
