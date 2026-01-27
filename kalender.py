import uuid
from datetime import datetime, timedelta

# Datenbasis aus den Dienstplänen 2026 (1. und 2. Halbjahr)
# Format: (Datum dd.mm.yyyy, Titel, Beschreibung/Fahrzeuge)
events_data = [
    # --- 1. Halbjahr [cite: 9] ---
    ("06.01.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("10.01.2026", "Weihnachtsbaum-Sammelaktion", "Mit Jugendfeuerwehr. Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    ("13.01.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("17.01.2026", "Winterfest", "Keine Fahrzeugzuordnung angegeben"),
    ("20.01.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    ("03.02.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("10.02.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    ("17.02.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"), # Laut OCR
    ("03.03.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("10.03.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    ("13.03.2026", "Jahreshauptversammlung", ""),
    ("17.03.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"), # Laut OCR (Reihenfolge)
    ("07.04.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    ("14.04.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("18.04.2026", "Bollerwagentour", ""),
    ("21.04.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    ("05.05.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("12.05.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    ("19.05.26", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("09.06.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    ("16.06.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("23.06.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    
    # --- 2. Halbjahr [cite: 16] ---
    ("07.07.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("14.07.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    ("21.07.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("11.08.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    ("15.08.2026", "Fahrradtour", ""),
    ("18.08.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("25.08.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    ("08.09.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("15.09.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    ("22.09.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("06.10.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    ("13.10.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("20.10.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"), # Zweiter Eintrag im Block
    ("03.11.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("10.11.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
    ("15.11.2026", "Kranzniederlegung in Arle", ""),
    ("17.11.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("08.12.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"), # Zweiter Eintrag im Block
    ("12.12.2026", "Weihnachtsfeier", ""),
    ("15.12.2026", "Gruppendienst", "Gruppe 1: HLF | Gruppe 2: TLF & TSF"),
    ("22.12.2026", "Gruppendienst", "Gruppe 1: TLF & TSF | Gruppe 2: HLF"),
]

def create_ics(events, filename="Feuerwehr_Dienstplan_2026.ics"):
    # Standard Startzeit für Dienste (kann hier angepasst werden)
    start_hour = 19
    start_minute = 30
    duration_hours = 2

    ics_content = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Feuerwehr Grossheide//Dienstplan 2026//DE",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:Feuerwehr Dienstplan 2026",
        "X-WR-TIMEZONE:Europe/Berlin",
    ]

    for date_str, summary, description in events:
        try:
            # Datum parsen
            # Format im Array ist manchmal dd.mm.yyyy oder dd.mm.yy
            if len(date_str.split(".")[-1]) == 2:
                 date_obj = datetime.strptime(date_str, "%d.%m.%y")
            else:
                 date_obj = datetime.strptime(date_str, "%d.%m.%20y" if len(date_str.split(".")[-1])==2 else "%d.%m.%Y")
            
            # Entscheidung: Ganztägig für Sonderveranstaltungen, Abendtermin für "Dienst"
            is_special = any(x in summary for x in ["Fest", "Tour", "Aktion", "Versammlung", "Feier", "Kranzniederlegung"])
            
            if is_special:
                # Ganztägiges Ereignis (sicherer, da Zeit unbekannt)
                dtstart = date_obj.strftime("%Y%m%d")
                # DTEND bei Ganztägig ist exklusiv (+1 Tag)
                dtend = (date_obj + timedelta(days=1)).strftime("%Y%m%d")
                event_block = [
                    "BEGIN:VEVENT",
                    f"UID:{uuid.uuid4()}@feuerwehr-grossheide",
                    f"DTSTAMP:{datetime.now().strftime('%Y%m%dT%H%M%SZ')}",
                    f"DTSTART;VALUE=DATE:{dtstart}",
                    f"DTEND;VALUE=DATE:{dtend}",
                    f"SUMMARY:{summary}",
                    f"DESCRIPTION:{description}",
                    "END:VEVENT"
                ]
            else:
                # Normaler Dienstabend (z.B. 19:30 - 21:30)
                start_dt = date_obj.replace(hour=start_hour, minute=start_minute)
                end_dt = start_dt + timedelta(hours=duration_hours)
                
                dtstart = start_dt.strftime("%Y%m%dT%H%M%S")
                dtend = end_dt.strftime("%Y%m%dT%H%M%S")
                
                event_block = [
                    "BEGIN:VEVENT",
                    f"UID:{uuid.uuid4()}@feuerwehr-grossheide",
                    f"DTSTAMP:{datetime.now().strftime('%Y%m%dT%H%M%SZ')}",
                    f"DTSTART;TZID=Europe/Berlin:{dtstart}",
                    f"DTEND;TZID=Europe/Berlin:{dtend}",
                    f"SUMMARY:{summary}",
                    f"DESCRIPTION:{description}",
                    "END:VEVENT"
                ]
            
            ics_content.extend(event_block)
            
        except ValueError as e:
            print(f"Fehler bei Datum {date_str}: {e}")

    ics_content.append("END:VCALENDAR")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(ics_content))
    
    print(f"Datei '{filename}' wurde erfolgreich erstellt.")

if __name__ == "__main__":
    create_ics(events_data)