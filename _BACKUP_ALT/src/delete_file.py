import pandas as pd
import os

DB_FILE = "messdaten_db.parquet"
# Der exakte Name der Problem-Datei
TARGET_FILE = "2025_12_10_1402-Celsa-ALO_12070-3000A-10R8-L2_0_sortiert.csv"

def run():
    if not os.path.exists(DB_FILE):
        print("Datenbank nicht gefunden.")
        return

    df = pd.read_parquet(DB_FILE)
    count_before = len(df)
    
    # Filter: Alles behalten, was NICHT unsere Datei ist
    df_clean = df[df["raw_file"] != TARGET_FILE]
    count_after = len(df_clean)
    
    diff = count_before - count_after
    
    if diff > 0:
        print(f"🗑️  Habe {diff} Zeilen für '{TARGET_FILE}' gelöscht.")
        df_clean.to_parquet(DB_FILE)
        print("✅ Datenbank gespeichert.")
    else:
        print(f"⚠️  Datei '{TARGET_FILE}' wurde in der DB gar nicht gefunden.")

if __name__ == "__main__":
    run()