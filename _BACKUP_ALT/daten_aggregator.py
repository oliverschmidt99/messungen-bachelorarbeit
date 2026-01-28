import pandas as pd
import numpy as np
import re
import os
import glob

# --- KONFIGURATION ---
OUTPUT_FILE = "messdaten_db.parquet"
SEARCH_DIR = "messungen_sortiert"
TARGET_LEVELS = [5, 20, 50, 80, 90, 100, 120]
PHASES = ["L1", "L2", "L3"]

# Priorität für Referenz
REF_KEYWORDS = ["pac1", "einspeisung", "ref", "source", "norm"]

# Spalten, die wir unbedingt behalten wollen (Metadaten aus Tab 3)
META_COLS_KEEP = [
    "Preis (€)",
    "Nennbürde (VA)",
    "T (mm)",
    "B (mm)",
    "H (mm)",
    "Kommentar",
    "Hersteller",  # Falls manuell korrigiert
    "Modell",  # Falls manuell korrigiert
    "Geometrie",  # Falls manuell korrigiert
]


def load_existing_metadata():
    """Lädt die existierende DB und speichert Metadaten in einem Dictionary"""
    if not os.path.exists(OUTPUT_FILE):
        return {}

    try:
        df_old = pd.read_parquet(OUTPUT_FILE)
        # Wir brauchen einen eindeutigen Schlüssel pro Datei. 'raw_file' ist perfekt.
        # Wir erstellen ein Dict: { 'Dateiname.csv': { 'Preis (€)': 50, ... } }
        meta_dict = {}

        # Iteriere über eindeutige Dateien
        for file_name in df_old["raw_file"].unique():
            # Nimm die erste Zeile, die zu dieser Datei gehört (Metadaten sind ja pro Datei gleich)
            row = df_old[df_old["raw_file"] == file_name].iloc[0]

            file_meta = {}
            for col in META_COLS_KEEP:
                if col in df_old.columns:
                    val = row[col]
                    # Speichere nur, wenn es nicht leer/0 ist (um keine leeren Werte zu übernehmen)
                    if pd.notna(val) and val != 0 and val != "" and val != "Unbekannt":
                        file_meta[col] = val

            if file_meta:
                meta_dict[file_name] = file_meta

        print(f"♻️  Metadaten für {len(meta_dict)} Dateien aus alter DB gesichert.")
        return meta_dict
    except Exception as e:
        print(f"⚠️ Konnte alte DB nicht lesen (Metadaten gehen verloren): {e}")
        return {}


def extract_metadata_from_filename(filepath):
    filename = os.path.basename(filepath)
    original_name = filename.replace("_sortiert.csv", "")
    folder_name = os.path.basename(os.path.dirname(filepath))

    match_amp = re.search(r"[-_](\d+)A[-_]", original_name)
    nennstrom = float(match_amp.group(1)) if match_amp else 0.0

    lower_name = original_name.lower()

    if "messstrecke" in lower_name:
        manufacturer = "Messstrecke"
    elif "mbs" in lower_name:
        manufacturer = "MBS"
    elif "celsa" in lower_name:
        manufacturer = "Celsa"
    elif "redur" in lower_name:
        manufacturer = "Redur"
    else:
        manufacturer = "Andere"

    # Modellnamen raten (einfach)
    model_str = original_name

    wandler_key = f"{manufacturer} {model_str}"

    return {
        "filepath": filepath,
        "folder": folder_name,
        "hersteller_auto": manufacturer,  # "_auto" damit wir es nicht mit manuellen Daten verwechseln
        "nennstrom": nennstrom,
        "wandler_key": wandler_key,
        "dateiname": filename,
        "original_name_clean": original_name,
    }


def analyze_sorted_file(filepath, meta):
    try:
        df = pd.read_csv(filepath, sep=";")
        if len(df.columns) < 2:  # Fallback
            df = pd.read_csv(filepath, sep=",")
    except:
        return [], "Lesefehler"

    df.columns = [c.strip() for c in df.columns]
    value_cols = [c for c in df.columns if "_I" in c]

    if not value_cols:
        return [], "Keine Strom-Daten"

    results = []

    for level in TARGET_LEVELS:
        lvl_str = f"{level:02d}"
        nominal_amp = meta["nennstrom"] * (level / 100.0)

        for phase in PHASES:
            # Suche Spalten für dieses Level und Phase
            relevant_cols = [
                c for c in value_cols if c.startswith(f"{lvl_str}_{phase}")
            ]
            if not relevant_cols:
                continue

            devices_map = {}
            for col in relevant_cols:
                # Format: 05_L1_I_Gerät (Neu) oder 05_L1_Gerät_I (Alt)
                match_new = re.search(rf"{lvl_str}_{phase}_I_(.+)$", col)
                match_old = re.search(rf"{lvl_str}_{phase}_(.+)_I$", col)

                if match_new:
                    devices_map[match_new.group(1)] = col
                elif match_old:
                    devices_map[match_old.group(1)] = col

            if not devices_map:
                continue

            # Referenz finden
            phys_ref_device = None
            for kw in REF_KEYWORDS:
                for dev in devices_map.keys():
                    if kw in dev.lower():
                        phys_ref_device = dev
                        break
                if phys_ref_device:
                    break

            if not phys_ref_device:
                phys_ref_device = sorted(list(devices_map.keys()))[0]

            col_phys_ref = devices_map[phys_ref_device]
            vals_phys_ref = pd.to_numeric(df[col_phys_ref], errors="coerce").dropna()

            phys_ref_mean = vals_phys_ref.mean() if not vals_phys_ref.empty else 0
            phys_ref_std = vals_phys_ref.std() if not vals_phys_ref.empty else 0

            # DUTs berechnen
            for dev, col_dut in devices_map.items():
                vals_dut = pd.to_numeric(df[col_dut], errors="coerce").dropna()
                if vals_dut.empty:
                    continue

                dut_mean = vals_dut.mean()
                dut_std = vals_dut.std()

                # Basis-Eintrag erstellen
                base_entry = {
                    "wandler_key": meta["wandler_key"],
                    "folder": meta["folder"],
                    "phase": phase,
                    "target_load": level,
                    "nennstrom": meta["nennstrom"],
                    "val_dut_mean": dut_mean,
                    "val_dut_std": dut_std,
                    "dut_name": dev,
                    "raw_file": meta["dateiname"],
                    # Initial-Werte für Metadaten (werden später überschrieben, falls vorhanden)
                    "Hersteller": meta["hersteller_auto"],
                    "Modell": meta["original_name_clean"],
                }

                # A: Relativ
                if dev != phys_ref_device and phys_ref_mean > 0:
                    entry = base_entry.copy()
                    entry.update(
                        {
                            "val_ref_mean": phys_ref_mean,
                            "val_ref_std": phys_ref_std,
                            "ref_name": phys_ref_device,
                            "comparison_mode": "device_ref",
                        }
                    )
                    results.append(entry)

                # B: Absolut
                if nominal_amp > 0:
                    entry = base_entry.copy()
                    entry.update(
                        {
                            "val_ref_mean": nominal_amp,
                            "val_ref_std": 0.0,
                            "ref_name": "Nennwert",
                            "comparison_mode": "nominal_ref",
                        }
                    )
                    results.append(entry)

    return results, "OK"


def main():
    print("--- Start: DB-Update (Smart Merge) ---")
    print(f"Suche in: {SEARCH_DIR}")

    # 1. Alte Metadaten retten
    saved_metadata = load_existing_metadata()

    # 2. Dateien neu verarbeiten
    files = glob.glob(os.path.join(SEARCH_DIR, "**", "*_sortiert.csv"), recursive=True)

    if not files:
        print("❌ Keine Dateien gefunden!")
        return

    all_data = []
    print(f"Verarbeite {len(files)} Dateien...")

    for f in files:
        meta = extract_metadata_from_filename(f)
        stats, status = analyze_sorted_file(f, meta)

        if stats:
            # 3. Metadaten injizieren
            # Wir schauen, ob wir für diesen Dateinamen gespeicherte Infos haben
            filename_key = meta["dateiname"]

            if filename_key in saved_metadata:
                saved_info = saved_metadata[filename_key]
                # Für jeden berechneten Punkt die gespeicherten Infos hinzufügen
                for stat_entry in stats:
                    stat_entry.update(saved_info)  # Überschreibt Defaults mit DB-Werten

            all_data.extend(stats)
            print(f"  ✅ {filename_key} ({len(stats)} Pts)")
        else:
            print(f"  ⚠️ {os.path.basename(f)}: {status}")

    if not all_data:
        print("❌ Keine Daten extrahiert.")
        return

    # 4. Speichern
    df_all = pd.DataFrame(all_data)

    # Sicherstellen, dass alle Spalten existieren (auch wenn keine Metadaten da waren)
    for col in META_COLS_KEEP:
        if col not in df_all.columns:
            df_all[col] = "" if col == "Kommentar" else 0.0

    # Deduplizieren
    df_clean = df_all.drop_duplicates(
        subset=["raw_file", "dut_name", "phase", "target_load", "comparison_mode"],
        keep="last",
    )

    df_clean.to_parquet(OUTPUT_FILE)
    print("-" * 40)
    print(f"✅ Datenbank aktualisiert: {OUTPUT_FILE}")
    print(f"✅ Einträge: {len(df_clean)}")
    print("✅ Alte Metadaten (Preise, Geometrien) wurden übernommen.")


if __name__ == "__main__":
    main()
