import json
import os
import shutil

# Pfade
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "daten", "saved_configs.json")
BACKUP_FILE = os.path.join(BASE_DIR, "daten", "saved_configs_backup.json")

def repair_from_keys():
    if not os.path.exists(CONFIG_FILE):
        print("❌ Keine Config-Datei gefunden.")
        return

    # Backup erstellen
    shutil.copy(CONFIG_FILE, BACKUP_FILE)
    print(f"📦 Backup erstellt: {BACKUP_FILE}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Wir schauen in den 'configurations' Block
    configs = data.get("configurations", {})
    
    # Zähler für Statistik
    repaired_count = 0

    for name, conf in configs.items():
        # Wir suchen nach Schlüsseln, die die Unique-IDs enthalten
        # Bevorzugt custom_colors, sonst custom_legends
        source_map = conf.get("custom_colors", {})
        if not source_map:
            source_map = conf.get("custom_legends", {})
        
        if not source_map:
            print(f"⏭️  Überspringe '{name}': Keine Farben/Legenden gefunden.")
            continue

        # Sets zum Sammeln der gefundenen Werte
        found_files = set()
        found_duts = set()

        # Format ist: "Dateiname.csv | DUT"
        for unique_id in source_map.keys():
            if " | " in unique_id:
                parts = unique_id.split(" | ")
                # Der erste Teil ist der Dateiname (Raw File)
                file_part = parts[0].strip()
                # Der zweite Teil ist das Gerät (DUT)
                dut_part = parts[1].strip()
                
                found_files.add(file_part)
                found_duts.add(dut_part)
        
        if found_files and found_duts:
            # Wir aktualisieren die Listen in der Config
            old_wandler_count = len(conf.get("wandlers", []))
            
            # WICHTIG: Wir schreiben hier die Dateinamen rein. 
            # Dein Dashboard (mit dem Smart-Match Code von vorhin) wird diese 
            # Dateinamen dann den echten Wandler-Keys zuordnen.
            conf["wandlers"] = sorted(list(found_files))
            conf["duts"] = sorted(list(found_duts))
            
            print(f"✅ Repariert '{name}':")
            print(f"   - Wandler: {old_wandler_count} -> {len(found_files)} Einträge")
            print(f"   - DUTs:    {len(conf.get('duts', []))} -> {len(found_duts)} Einträge")
            repaired_count += 1

    # Speichern
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("-" * 30)
    print(f"🚀 Fertig! {repaired_count} Konfigurationen basierend auf Farbcodes repariert.")

if __name__ == "__main__":
    repair_from_keys()