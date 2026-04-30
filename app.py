import streamlit as st
import pandas as pd
import xgboost as xgb
import pandas_ta_classic as ta
import plotly.graph_objects as go
from datetime import timedelta
import psycopg2

# --- 1. UI SETUP ---
st.set_page_config(page_title="AI Procurement Radar | Quant Edition", layout="wide")
st.title("🏛️ Professional AI Procurement Radar (Quant Edition)")
st.markdown("**KI-Modell:** Robustes XGBoost | **Daten:** Makroökonomie, Marktpsychologie & Alternative Data (Minen-ETFs, Zinsen)")

# --- 2. DATEN LADEFUNKTION (AUS CLOUD DATENBANK) ---
@st.cache_data(ttl=3600) 
def lade_daten_aus_supabase():
    conn = psycopg2.connect(st.secrets["DB_URI"])
    query = "SELECT * FROM kupfer_historie ORDER BY datum ASC;"
    df = pd.read_sql_query(query, conn, index_col="datum", parse_dates=["datum"])
    conn.close()
    return df

# --- 3. MODELL TRAINING ---
@st.cache_data(ttl=3600)
def load_and_train_quant_model():
    # Daten aus unserer neuen Supabase-Datenbank ziehen
    historie_df = lade_daten_aus_supabase()
    
    copper = historie_df['kupfer_preis']
    sp500 = historie_df['sp500']
    oil = historie_df['oel']
    dxy = historie_df['dxy']
    cny = historie_df['cny']
    copx = historie_df['copx']
    tnx = historie_df['tnx']

    df = pd.DataFrame({
        'Preis': copper, 'SP500': sp500, 'Oel': oil, 
        'DXY': dxy, 'CNY': cny, 'Minen_Aktien': copx, 'Zinsen': tnx
    }).ffill()
    
    # Feature Engineering (Ziel)
    df['Preis_in_7_Tagen'] = df['Preis'].shift(-7)
    df['Target_Delta'] = df['Preis_in_7_Tagen'] - df['Preis']
    df['Preis_Gestern'] = df['Preis'].shift(1)
    df['SMA_14'] = df['Preis'].rolling(window=14).mean()
    
    # Trends (Makro & Alternative)
    df['SP500_Trend'] = df['SP500'] - df['SP500'].shift(7)
    df['Oel_Trend'] = df['Oel'] - df['Oel'].shift(7)
    df['DXY_Trend'] = df['DXY'] - df['DXY'].shift(7)
    df['Minen_Trend'] = df['Minen_Aktien'] - df['Minen_Aktien'].shift(7)
    df['Zins_Trend'] = df['Zinsen'] - df['Zinsen'].shift(7)
    
    # Technische Indikatoren
    df.ta.rsi(close='Preis', length=14, append=True)
    df.ta.macd(close='Preis', append=True)
    
    # Alle Features exakt wie im Notebook
    features = [
        'Preis', 'Preis_Gestern', 'SMA_14', 'SP500', 'Oel', 'SP500_Trend', 'Oel_Trend',
        'DXY', 'CNY', 'DXY_Trend', 'RSI_14', 'MACD_12_26_9',
        'Minen_Aktien', 'Zinsen', 'Minen_Trend', 'Zins_Trend'
    ]
    
    heute_features = df[features].iloc[-1:]
    aktueller_preis = df['Preis'].iloc[-1]
    
    train_df = df.dropna()
    X = train_df[features]
    y = train_df['Target_Delta']
    
    # Das optimierte Modell
    model = xgb.XGBRegressor(
        n_estimators=50, learning_rate=0.01, max_depth=5, 
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X, y)
    
    predicted_delta = model.predict(heute_features)[0]
    return aktueller_preis, aktueller_preis + predicted_delta, predicted_delta, df

try:
    with st.spinner("Lade Daten aus Supabase Cloud & berechne KI-Prognose..."):
        aktueller_preis, vorhersage_preis, delta, df = load_and_train_quant_model()
    st.sidebar.success("🟢 Verbunden mit Supabase Cloud-DB")

except Exception as e:
    st.sidebar.error(f"❌ Datenbank-Fehler: {e}")
    st.stop() 

# --- 4. UI DASHBOARD ---
st.write("---")
col1, col2, col3 = st.columns(3)
prozent = (delta / aktueller_preis) * 100

with col1:
    st.metric("Live Kupferpreis", f"${aktueller_preis:.4f}")
with col2:
    st.metric("KI-Prognose (7 Tage)", f"${vorhersage_preis:.4f}", f"{prozent:.2f}%")
with col3:
    if prozent > 0.5:
        st.error("🚨 KAUF-ALARM (Gestützt durch Makro-Daten)")
    elif prozent < -0.5:
        st.success("💎 WARTEN (Preisverfall erwartet)")
    else:
        st.info("⚖️ MARKT STABIL")

# --- 5. CHART ---
st.write("---")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-90:], y=df['Preis'].iloc[-90:], name="Realer Preis", line=dict(color='royalblue')))
fig.add_trace(go.Scatter(x=[df.index[-1], df.index[-1] + timedelta(days=7)], y=[aktueller_preis, vorhersage_preis], name="Quant-Prognose", line=dict(color='red', width=3, dash='dash')))
fig.update_layout(height=500, title="90-Tage Historie & KI-Vorhersage", xaxis_title="Datum", yaxis_title="Preis (USD)")
st.plotly_chart(fig, use_container_width=True)
