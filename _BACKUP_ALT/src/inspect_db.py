import pandas as pd
import os

DB_FILE = "messdaten_db.parquet"

if not os.path.exists(DB_FILE):
    print("❌ Keine Datenbank gefunden.")
    exit()

df = pd.read_parquet(DB_FILE)

print(f"📊 Datenbank Status: {len(df)} Zeilen gesamt")
print("-" * 60)

# 1. Welche Dateien sind drin?
unique_files = sorted(df["raw_file"].unique())
print(f"📂 Enthaltene Dateien ({len(unique_files)} Stück):")
for f in unique_files:
    # Wir zählen kurz, wie viele Zeilen pro Datei da sind
    count = len(df[df["raw_file"] == f])
    print(f"  • {f} ({count} Einträge)")

print("-" * 60)

# 2. Welche Geräte-Namen gibt es jetzt?
unique_duts = sorted(df["dut_name"].unique())
print("🔧 Verfügbare Geräte-Namen (DUTs):")
for d in unique_duts:
    print(f"  • {d}")
