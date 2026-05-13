import os
import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta # NEU: Unser Tool für RSI, Gleitende Durchschnitte etc.
import psycopg2
import psycopg2.extras
from datetime import timedelta
from xgboost import XGBRegressor # NEU: Die KI

# --- 1. VERBINDUNG ZUR DATENBANK ---
DB_URI = os.environ.get("DB_URI")

if not DB_URI:
    raise ValueError("❌ FEHLER: Die Umgebungsvariable 'DB_URI' fehlt.")

print("Starte Daten-Arbeiter & KI-Modell...")

try:
    conn = psycopg2.connect(DB_URI)
    cursor = conn.cursor()
    
    # --- 2. DATEN HOLEN & FEATURE ENGINEERING ---
    print("Hole Daten und berechne technische Indikatoren...")
    
    # Kupfer separat holen (wir brauchen High und Low für die Volatilität)
    copper_raw = yf.Ticker("HG=F").history(period="5y")
    copper_raw.index = copper_raw.index.tz_localize(None).normalize()

    def lade_sauber(ticker_symbol):
        daten = yf.Ticker(ticker_symbol).history(period="5y")['Close']
        daten.index = daten.index.tz_localize(None).normalize()
        return daten

    # Basistabelle bauen
    df = pd.DataFrame({
        'kupfer_preis': copper_raw['Close'],
        'kupfer_high': copper_raw['High'],
        'kupfer_low': copper_raw['Low'],
        'sp500': lade_sauber("^GSPC"),
        'oel': lade_sauber("CL=F"),
        'dxy': lade_sauber("DX-Y.NYB"),
        'cny': lade_sauber("CNY=X"),
        'copx': lade_sauber("COPX"),
        'tnx': lade_sauber("^TNX")
    })
    
    df = df.ffill().dropna()

    # KI schlau machen: Indikatoren hinzufügen
    df['sma_50'] = ta.sma(df['kupfer_preis'], length=50)
    df['sma_200'] = ta.sma(df['kupfer_preis'], length=200)
    df['rsi_14'] = ta.rsi(df['kupfer_preis'], length=14)
    df['atr_14'] = ta.atr(df['kupfer_high'], df['kupfer_low'], df['kupfer_preis'], length=14)
    
    df = df.dropna() # Leere Zeilen vom Indikatoren-Berechnen entfernen

    # --- 3. HISTORIE IN SUPABASE SPEICHERN ---
    print("Speichere angereicherte Historie in Supabase...")
    cursor.execute("DELETE FROM kupfer_historie;")
    
    daten_liste = []
    for index, row in df.iterrows():
        daten_liste.append((
            index.strftime('%Y-%m-%d'), 
            float(row['kupfer_preis']), float(row['sp500']), float(row['oel']), 
            float(row['dxy']), float(row['cny']), float(row['copx']), float(row['tnx']),
            float(row['sma_50']), float(row['sma_200']), float(row['rsi_14']), float(row['atr_14'])
        ))

    insert_query = """
        INSERT INTO kupfer_historie (datum, kupfer_preis, sp500, oel, dxy, cny, copx, tnx, sma_50, sma_200, rsi_14, atr_14)
        VALUES %s
    """
    psycopg2.extras.execute_values(cursor, insert_query, daten_liste)

    # --- 4. KI-TRAINING & VORHERSAGE (1 WOCHE) ---
    print("Trainiere KI für 1-Wochen-Prognose...")
    
    # Target: Der Preis in 5 Handelstagen (entspricht 1 Kalenderwoche)
    df['target'] = df['kupfer_preis'].shift(-5)
    
    # Die letzten 5 Tage fliegen fürs Training raus, weil wir deren Zukunft noch nicht kennen
    df_train = df.dropna() 
    features = ['kupfer_preis', 'sp500', 'oel', 'dxy', 'cny', 'copx', 'tnx', 'sma_50', 'sma_200', 'rsi_14', 'atr_14']
    
    X = df_train[features]
    y = df_train['target']

    # Modell trainieren
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    # Vorhersage für in 1 Woche berechnen (basierend auf Daten von heute)
    heute_datum = df.index[-1]
    ziel_datum = heute_datum + timedelta(days=7) # +7 Kalendertage
    
    letzter_tag_daten = df[features].iloc[-1:]
    vorhersage_woche = float(model.predict(letzter_tag_daten)[0])
    echter_preis_heute = float(df['kupfer_preis'].iloc[-1])

    # --- 5. TRACKING TABELLE UPDATEN ---
    print(f"Update Tracking: Wahrer Preis für {heute_datum.date()} / Vorhersage für {ziel_datum.date()}")

    # Echten Preis von heute eintragen (löst eine alte Vorhersage auf)
    upsert_tatsaechlich = """
        INSERT INTO modell_tracking (datum, tatsaechlich) VALUES (%s, %s)
        ON CONFLICT (datum) DO UPDATE SET tatsaechlich = EXCLUDED.tatsaechlich;
    """
    cursor.execute(upsert_tatsaechlich, (heute_datum.strftime('%Y-%m-%d'), echter_preis_heute))

    # Neue Vorhersage für in 7 Tagen eintragen
    upsert_vorhersage = """
        INSERT INTO modell_tracking (datum, vorhersage) VALUES (%s, %s)
        ON CONFLICT (datum) DO UPDATE SET vorhersage = EXCLUDED.vorhersage;
    """
    cursor.execute(upsert_vorhersage, (ziel_datum.strftime('%Y-%m-%d'), vorhersage_woche))

    conn.commit()
    print("✅ Alles erfolgreich abgeschlossen! Historie und Tracking sind aktuell.")

except Exception as e:
    print(f"❌ Fehler: {e}")

finally:
    if 'cursor' in locals() and cursor:
        cursor.close()
    if 'conn' in locals() and conn:
        conn.close()
        print("Datenbank-Verbindung sicher geschlossen.")
