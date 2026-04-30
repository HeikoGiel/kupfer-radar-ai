import streamlit as st
import yfinance as yf
import pandas as pd
import xgboost as xgb
import pandas_ta_classic as ta
import plotly.graph_objects as go
from datetime import timedelta

# --- 1. UI SETUP ---
st.set_page_config(page_title="AI Procurement Radar | Quant Edition", layout="wide")
st.title("🏛️ Professional AI Procurement Radar (Quant Edition)")
st.markdown("**KI-Modell:** Robustes XGBoost | **Daten:** Makroökonomie, Marktpsychologie & Alternative Data (Minen-ETFs, Zinsen)")

# --- 2. DATEN & MODELL ---
@st.cache_data(ttl=3600)
def load_and_train_quant_model():
    # Alle 7 globalen Datenströme ziehen
    copper = yf.Ticker("HG=F").history(period="5y")['Close']
    sp500 = yf.Ticker("^GSPC").history(period="5y")['Close']
    oil = yf.Ticker("CL=F").history(period="5y")['Close']
    dxy = yf.Ticker("DX-Y.NYB").history(period="5y")['Close']
    cny = yf.Ticker("CNY=X").history(period="5y")['Close']
    copx = yf.Ticker("COPX").history(period="5y")['Close']
    tnx = yf.Ticker("^TNX").history(period="5y")['Close']

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

with st.spinner("Scanne globale Märkte, Zinsen und Logistikdaten..."):
    aktueller_preis, vorhersage_preis, delta, df = load_and_train_quant_model()

# --- 3. UI DASHBOARD ---
st.write("---")
col1, col2, col3 = st.columns(3)
prozent = (delta / aktueller_preis) * 100

import streamlit as st
import yfinance as yf
import pandas as pd
import xgboost as xgb
import pandas_ta_classic as ta
import plotly.graph_objects as go
from datetime import timedelta

# --- 1. UI SETUP ---
st.set_page_config(page_title="AI Procurement Radar | Quant Edition", layout="wide")
st.title("🏛️ Professional AI Procurement Radar (Quant Edition)")
st.markdown("**KI-Modell:** Robustes XGBoost | **Daten:** Makroökonomie, Marktpsychologie & Alternative Data (Minen-ETFs, Zinsen)")

# --- 2. DATEN & MODELL ---
@st.cache_data(ttl=3600)
def load_and_train_quant_model():
    # Alle 7 globalen Datenströme ziehen
    copper = yf.download("HG=F", period="5y")['Close'].squeeze()
    sp500 = yf.download("^GSPC", period="5y")['Close'].squeeze()
    oil = yf.download("CL=F", period="5y")['Close'].squeeze()
    dxy = yf.download("DX-Y.NYB", period="5y")['Close'].squeeze()
    cny = yf.download("CNY=X", period="5y")['Close'].squeeze()
    copx = yf.download("COPX", period="5y")['Close'].squeeze() # Minen-Aktien
    tnx = yf.download("^TNX", period="5y")['Close'].squeeze()  # Zinsen

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

with st.spinner("Scanne globale Märkte, Zinsen und Logistikdaten..."):
    aktueller_preis, vorhersage_preis, delta, df = load_and_train_quant_model()

# --- 3. UI DASHBOARD ---
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

# --- 4. CHART ---
st.write("---")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-90:], y=df['Preis'].iloc[-90:], name="Realer Preis", line=dict(color='royalblue')))
fig.add_trace(go.Scatter(x=[df.index[-1], df.index[-1] + timedelta(days=7)], y=[aktueller_preis, vorhersage_preis], name="Quant-Prognose", line=dict(color='red', width=3, dash='dash')))
fig.update_layout(height=500, title="90-Tage Historie & KI-Vorhersage", xaxis_title="Datum", yaxis_title="Preis (USD)")
st.plotly_chart(fig, use_container_width=True)
