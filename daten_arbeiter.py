import yfinance as yf
import pandas as pd
import psycopg2
from datetime import datetime
import os # NEU: Wird für die versteckten Passwörter gebraucht
import psycopg2.extras

# --- 1. VERBINDUNG ZUR DATENBANK AUFBAUEN ---
# NEU: Das Passwort wird jetzt sicher aus den GitHub Secrets geladen!
DB_URI = os.environ.get("DB_URI")

# Sicherheits-Check: Fehlt das Passwort in GitHub?
if not DB_URI:
    raise ValueError("❌ FEHLER: Die Umgebungsvariable 'DB_URI' wurde nicht gefunden. Hast du das Secret in GitHub angelegt?")

print("Starte Daten-Arbeiter...")
print("Verbinde mit Supabase-Datenbank...")

try:
    conn = psycopg2.connect(DB_URI)
    cursor = conn.cursor()
    print("✅ Verbindung erfolgreich!")

    # --- 2. ALTE DATEN LÖSCHEN (FULL LOAD) ---
    print("Lösche alte Datenstämme (Full Load)...")
    cursor.execute("DELETE FROM kupfer_historie;")
    conn.commit() 
    print("✅ Alte Daten gelöscht!")

    # --- 3. FRISCHE DATEN VON YAHOO HOLEN ---
    print("Hole frische Daten von Yahoo Finance...")
    
    # Kleine Hilfsfunktion, um die nervigen Zeitzonen abzuschneiden
    def lade_sauber(ticker_symbol):
        daten = yf.Ticker(ticker_symbol).history(period="5y")['Close']
        # Schneidet die Uhrzeit ab und macht alle zu reinen, neutralen Daten
        daten.index = daten.index.tz_localize(None).normalize()
        return daten

    copper = lade_sauber("HG=F")
    sp500 = lade_sauber("^GSPC")
    oil = lade_sauber("CL=F")
    dxy = lade_sauber("DX-Y.NYB")
    cny = lade_sauber("CNY=X")
    copx = lade_sauber("COPX")
    tnx = lade_sauber("^TNX")

    # Wir bauen ein sauberes "DataFrame" (Tabelle)
    df = pd.DataFrame({
        'kupfer_preis': copper,
        'sp500': sp500,
        'oel': oil,
        'dxy': dxy,
        'cny': cny,
        'copx': copx,
        'tnx': tnx
    })
    
    # Letzte Lücken füllen (Feiertage) und Reste löschen
    df = df.ffill().dropna()

    # --- 4. NEUE DATEN IN DIE DATENBANK SCHIEBEN ---
    print(f"Schreibe {len(df)} neue Zeilen in die Datenbank (Bulk Insert)...")
    
    # Wir verpacken alle Zeilen in eine einzige große Liste
    daten_liste = []
    for index, row in df.iterrows():
        datum_str = index.strftime('%Y-%m-%d')
        werte = (
            datum_str, 
            float(row['kupfer_preis']), 
            float(row['sp500']), 
            float(row['oel']), 
            float(row['dxy']), 
            float(row['cny']), 
            float(row['copx']), 
            float(row['tnx'])
        )
        daten_liste.append(werte)

    # Der SQL-Befehl erwartet jetzt ein riesiges Paket
    insert_query = """
        INSERT INTO kupfer_historie (datum, kupfer_preis, sp500, oel, dxy, cny, copx, tnx)
        VALUES %s
    """
    
    # Der magische Turbo-Befehl, der alles auf einmal schickt
    psycopg2.extras.execute_values(cursor, insert_query, daten_liste)

    conn.commit()
    print("✅ Alle neuen Daten erfolgreich in Supabase gespeichert!")
    
except Exception as e:
    print(f"❌ Es gab einen Fehler: {e}")

finally:
    if 'cursor' in locals() and cursor:
        cursor.close()
    if 'conn' in locals() and conn:
        conn.close()
        print("Datenbank-Verbindung sicher geschlossen.")
