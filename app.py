import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go

# --- SETUP ---
st.set_page_config(page_title="Kupfer Radar AI", layout="wide")
st.title("📈 Kupfer Radar AI - Dashboard")

# --- DATENBANK VERBINDUNG ---
# Streamlit nutzt "st.secrets" statt "os.environ" für Passwörter
@st.cache_resource
def init_connection():
    return psycopg2.connect(st.secrets["DB_URI"])

try:
    conn = init_connection()
except Exception as e:
    st.error("❌ Datenbankverbindung fehlgeschlagen. Sind die Secrets in Streamlit hinterlegt?")
    st.stop()

# --- DATEN LADEN ---
@st.cache_data(ttl=3600) # Speichert die Daten für 1 Stunde im Zwischenspeicher (macht die App pfeilschnell)
def load_data():
    # Wir holen beide Tabellen
    df_hist = pd.read_sql("SELECT * FROM kupfer_historie ORDER BY datum ASC;", conn)
    df_track = pd.read_sql("SELECT * FROM modell_tracking ORDER BY datum ASC;", conn)
    return df_hist, df_track

df_hist, df_track = load_data()

# --- LAYOUT & METRIKEN ---
st.subheader("Aktuelle KI-Vorhersage (1 Woche)")

if not df_hist.empty and not df_track.empty:
    # Neuesten echten Preis holen
    letzter_echter_preis = df_hist.iloc[-1]['kupfer_preis']
    letztes_datum = df_hist.iloc[-1]['datum']

    # Die aktuellste Vorhersage aus der Tracking-Tabelle holen
    zukuenftige_vorhersagen = df_track[df_track['vorhersage'].notna()]
    
    if not zukuenftige_vorhersagen.empty:
        neueste_vorhersage_row = zukuenftige_vorhersagen.iloc[-1]
        vorhersage_wert = neueste_vorhersage_row['vorhersage']
        vorhersage_datum = neueste_vorhersage_row['datum']
        
        # Rendite berechnen
        differenz = vorhersage_wert - letzter_echter_preis
        prozent = (differenz / letzter_echter_preis) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric(f"Letzter Preis ({letztes_datum})", f"{letzter_echter_preis:.4f} USD")
        col2.metric(f"Prognose für {vorhersage_datum}", f"{vorhersage_wert:.4f} USD", f"{prozent:.2f} %")
    else:
        st.warning("Noch keine Vorhersagen in der Datenbank gefunden.")

# --- CHART 1: MODELL-TÜV (Vorhersage vs. Realität) ---
st.markdown("---")
st.subheader("🔍 Modell-TÜV: Vorhersage vs. Echter Preis")
st.markdown("Hier siehst du, wie gut das Modell in der Vergangenheit lag. Die rote Linie zeigt, was das Modell **eine Woche im Voraus** für diesen Tag prophezeit hat.")

fig_tuev = go.Figure()

# Echter Preis
fig_tuev.add_trace(go.Scatter(
    x=df_track['datum'], y=df_track['tatsaechlich'], 
    mode='lines', name='Echter Preis', line=dict(color='#1f77b4', width=2)
))

# Vorhersage
fig_tuev.add_trace(go.Scatter(
    x=df_track['datum'], y=df_track['vorhersage'], 
    mode='lines', name='Vorhersage (1 Woche alt)', line=dict(color='#ff7f0e', width=2, dash='dash')
))

fig_tuev.update_layout(height=400, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig_tuev, use_container_width=True)

# --- CHART 2: HISTORIE & TRENDS ---
st.markdown("---")
st.subheader("📊 Kupfer-Historie & Technische Trends")

fig_hist = go.Figure()

# Kupferpreis
fig_hist.add_trace(go.Scatter(
    x=df_hist['datum'], y=df_hist['kupfer_preis'], 
    mode='lines', name='Kupferpreis', line=dict(color='white')
))
# Gleitende Durchschnitte
fig_hist.add_trace(go.Scatter(
    x=df_hist['datum'], y=df_hist['sma_50'], 
    mode='lines', name='SMA 50 (Kurzfristig)', line=dict(color='yellow', width=1)
))
fig_hist.add_trace(go.Scatter(
    x=df_hist['datum'], y=df_hist['sma_200'], 
    mode='lines', name='SMA 200 (Langfristig)', line=dict(color='purple', width=1)
))

fig_hist.update_layout(height=500, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig_hist, use_container_width=True)
