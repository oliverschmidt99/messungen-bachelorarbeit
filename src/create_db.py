import pandas as pd
import numpy as np
import re
import os
import glob

# --- KONFIGURATION ---
OUTPUT_FILE = "messdaten_db.parquet"
SEARCH_DIR = "messungen/messungen"
TARGET_LEVELS = [5, 20, 50, 80, 90, 100, 120]
PHASES = ["L1", "L2", "L3"]
REF_KEYWORDS = ["pac1", "einspeisung", "ref", "source", "norm", "powermeter"]

# Diese Spalten wollen wir aus der alten DB retten, falls sie existieren
META_COLS_EDIT = [
    "Preis (€)",
    "Nennbürde (VA)",
    "T (mm)",
    "B (mm)",
    "H (mm)",
    "Kommentar",
]


def parse_folder_geometry(folder_name):
    fn = folder_name.lower()
    if "parallel" in fn:
        return "Parallel"
    elif "dreieck" in fn:
        return "Dreieck"
    elif "messstrecke" in fn:
        return "Messstrecke"
    return "Unbekannt"


def extract_metadata_strict(filepath):
    filename = os.path.basename(filepath)
    folder_path = os.path.dirname(filepath)
    folder_name = os.path.basename(folder_path)

    name_no_ext = filename.replace("_sortiert.csv", "")
    parts = name_no_ext.split("-")

    hersteller = "Unbekannt"
    modell = "Unbekannt"
    nennstrom = 0.0
    mess_burde = "Unbekannt"

    if len(parts) >= 5:
        hersteller = parts[1].replace("_", " ")
        modell = parts[2].replace("_", " ")
        try:
            nennstrom = float(parts[3].upper().replace("A", ""))
        except:
            nennstrom = 0.0
        mess_burde = parts[4]
    else:
        if "Messstrecke" in name_no_ext:
            hersteller = "Referenz"
            modell = "Messstrecke"
            match = re.search(r"(\d+)A", name_no_ext)
            if match:
                nennstrom = float(match.group(1))

    geometrie = parse_folder_geometry(folder_name)
    wandler_key = f"{hersteller} {modell} {mess_burde}".strip()

    return {
        "filepath": filepath,
        "folder": folder_name,
        "raw_file": filename,
        "Hersteller": hersteller,
        "Modell": modell,
        "nennstrom": nennstrom,
        "Mess-Bürde": mess_burde,
        "Geometrie": geometrie,
        "wandler_key": wandler_key,
    }


def analyze_csv_file(filepath, meta):
    try:
        df = pd.read_csv(filepath, sep=";")
    except:
        return []

    df.columns = [c.strip() for c in df.columns]
    value_cols = [c for c in df.columns if "_I" in c]
    if not value_cols:
        return []

    results = []
    for level in TARGET_LEVELS:
        lvl_str = f"{level:02d}"
        nominal_amp = meta["nennstrom"] * (level / 100.0)

        for phase in PHASES:
            relevant_cols = [
                c for c in value_cols if c.startswith(f"{lvl_str}_{phase}")
            ]
            if not relevant_cols:
                continue

            devices_map = {}
            for col in relevant_cols:
                match_new = re.search(rf"{lvl_str}_{phase}_(.+)_I$", col)
                match_old = re.search(rf"{lvl_str}_{phase}_I_(.+)$", col)
                if match_new:
                    devices_map[match_new.group(1)] = col
                elif match_old:
                    devices_map[match_old.group(1)] = col

            if not devices_map:
                continue

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

            for dev, col_dut in devices_map.items():
                vals_dut = pd.to_numeric(df[col_dut], errors="coerce").dropna()
                if vals_dut.empty:
                    continue

                dut_mean = vals_dut.mean()
                dut_std = vals_dut.std()

                base_entry = {
                    "wandler_key": meta["wandler_key"],
                    "folder": meta["folder"],
                    "phase": phase,
                    "target_load": level,
                    "nennstrom": meta["nennstrom"],
                    "val_dut_mean": dut_mean,
                    "val_dut_std": dut_std,
                    "dut_name": dev,
                    "raw_file": meta["raw_file"],
                    "Hersteller": meta["Hersteller"],
                    "Modell": meta["Modell"],
                    "Geometrie": meta["Geometrie"],
                    "Mess-Bürde": meta["Mess-Bürde"],
                }

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
    return results


def main():
    print("🚀 Start: DB-Update (mit Datenerhalt)")

    # 1. Bestehende Stammdaten retten
    existing_meta_map = {}
    if os.path.exists(OUTPUT_FILE):
        print("💾 Lese existierende Datenbank und rette Preise/Maße...")
        try:
            old_df = pd.read_parquet(OUTPUT_FILE)
            # Wir brauchen nur die Spalten, die wir retten wollen + den Schlüssel (raw_file)
            cols_to_save = ["raw_file"] + [
                c for c in META_COLS_EDIT if c in old_df.columns
            ]

            # Duplikate entfernen (wir brauchen nur 1 Eintrag pro Datei für die Metadaten)
            meta_df = old_df[cols_to_save].drop_duplicates(subset=["raw_file"])

            # Als Dictionary speichern: {'dateiname.csv': {'Preis': 50, 'T (mm)': 10...}}
            existing_meta_map = meta_df.set_index("raw_file").to_dict(orient="index")
            print(f"✅ Daten von {len(existing_meta_map)} Dateien gesichert.")
        except Exception as e:
            print(f"⚠️ Warnung: Konnte alte Daten nicht lesen ({e}). Starte leer.")

    # 2. Neu scannen
    files = glob.glob(os.path.join(".", "**", "*_sortiert.csv"), recursive=True)
    if not files:
        print("❌ Keine CSV-Dateien gefunden!")
        return

    print(f"📂 Scanne {len(files)} Dateien neu...")
    all_data = []
    for f in files:
        meta = extract_metadata_strict(f)
        stats = analyze_csv_file(f, meta)
        if stats:
            all_data.extend(stats)

    if not all_data:
        print("❌ Keine Daten extrahiert.")
        return

    df_new = pd.DataFrame(all_data)

    # 3. Gerettete Stammdaten wiederherstellen
    print("🔄 Führe alte Daten (Preise etc.) mit neuen Messungen zusammen...")

    # Erstmal leere Spalten anlegen
    for col in META_COLS_EDIT:
        if col not in df_new.columns:
            df_new[col] = 0.0 if ("Preis" in col or "mm" in col or "VA" in col) else ""

    # Jetzt Zeile für Zeile (basierend auf Dateiname) die alten Werte eintragen
    # Das ist sehr schnell mit .map()
    for col in META_COLS_EDIT:
        # Wir bauen eine Map für diese Spalte: {Dateiname: Alter_Wert}
        # Falls eine Datei neu ist, gibt es keinen Eintrag -> Default Wert nutzen
        val_map = {
            k: v.get(col, 0.0 if ("Preis" in col or "mm" in col or "VA" in col) else "")
            for k, v in existing_meta_map.items()
        }

        # Anwenden: Wenn Dateiname in Map -> Alter Wert, sonst -> Jetziger Wert (0/leer)
        df_new[col] = df_new["raw_file"].map(val_map).fillna(df_new[col])

        # Sicherstellen, dass Kommentar String ist
        if col == "Kommentar":
            df_new[col] = df_new[col].astype(str).replace("nan", "")

    # Speichern
    df_new.to_parquet(OUTPUT_FILE)
    print(
        f"🎉 Fertig! Datenbank aktualisiert ({len(df_new)} Zeilen). Preise wurden behalten."
    )


if __name__ == "__main__":
    main()
