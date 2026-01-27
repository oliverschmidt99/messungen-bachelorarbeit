import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import re
import json
import numpy as np
from pathlib import Path

# --- KONFIGURATION ---
TRACKING_CSV = "manuelle_ergebnisse.csv"
CONFIG_JSON = "saved_configs.json"
TARGET_LEVELS = [5, 20, 50, 80, 90, 100, 120]
PHASES = ["L1", "L2", "L3"]
OUTPUT_ROOT_DIR = "messungen_sortiert"

st.set_page_config(layout="wide", page_title="Manueller Rohdaten-Export (Multi-Device)")

# --- INITIALISIERUNG SESSION STATE ---
for level in TARGET_LEVELS:
    if f"s_{level}" not in st.session_state:
        st.session_state[f"s_{level}"] = 0
    if f"e_{level}" not in st.session_state:
        st.session_state[f"e_{level}"] = 0


# --- CONFIG JSON HANDLING ---
def load_config(filename):
    if os.path.exists(CONFIG_JSON):
        try:
            with open(CONFIG_JSON, "r") as f:
                data = json.load(f)
                return data.get(filename, {})
        except:
            return {}
    return {}


def save_config(filename, ranges_map):
    data = {}
    if os.path.exists(CONFIG_JSON):
        try:
            with open(CONFIG_JSON, "r") as f:
                data = json.load(f)
        except:
            data = {}

    data[filename] = ranges_map
    with open(CONFIG_JSON, "w") as f:
        json.dump(data, f, indent=4)


# --- SMART RECOVERY ---
def find_subsequence(full_series, sub_series, tolerance=1e-2):
    if len(sub_series) > len(full_series) or len(sub_series) == 0:
        return None
    signature_len = min(50, len(sub_series))
    signature = sub_series[:signature_len]
    full_arr = full_series.to_numpy()
    sig_arr = signature.to_numpy()
    first_val = sig_arr[0]
    candidates = np.where(
        np.isclose(full_arr[: -len(sub_series) + 1], first_val, atol=tolerance)
    )[0]
    for idx in candidates:
        check_slice = full_arr[idx : idx + signature_len]
        if np.allclose(check_slice, sig_arr, atol=tolerance):
            return idx
    return None


def try_recover_from_sorted_file(original_path, full_data_l1, dev_name):
    orig_p = Path(original_path)
    target_dir = Path(OUTPUT_ROOT_DIR) / orig_p.parent
    sorted_filename = f"{orig_p.stem}_sortiert.csv"
    sorted_path = target_dir / sorted_filename

    if not sorted_path.exists():
        return False, "Keine sortierte Datei gefunden."

    try:
        df_sorted = pd.read_csv(sorted_path, sep=";", decimal=".", engine="python")
        if df_sorted.shape[1] < 2:
            df_sorted = pd.read_csv(sorted_path, sep=";", decimal=",", engine="python")

        recovered_count = 0
        for level in TARGET_LEVELS:
            col_name_new = f"{level:02d}_L1_I_{dev_name}"
            col_name_old = f"{level:02d}_L1_{dev_name}_I"
            col_name = (
                col_name_new if col_name_new in df_sorted.columns else col_name_old
            )

            if col_name in df_sorted.columns:
                sub_data = df_sorted[col_name].dropna()
                if len(sub_data) > 0:
                    start_idx = find_subsequence(full_data_l1, sub_data)
                    if start_idx is not None:
                        end_idx = start_idx + len(sub_data)
                        st.session_state[f"s_{level}"] = int(start_idx)
                        st.session_state[f"e_{level}"] = int(end_idx)
                        recovered_count += 1

        if recovered_count > 0:
            return True, f"{recovered_count} Bereiche wiederhergestellt!"
        else:
            return False, "Datenformat passt, aber keine Positionen gefunden."
    except Exception as e:
        return False, f"Fehler beim Lesen: {e}"


# --- CALLBACKS ---
def update_start_callback(lvl):
    end_key = f"e_{lvl}"
    start_key = f"s_{lvl}"
    current_end = st.session_state[end_key]
    if st.session_state[start_key] == 0:
        # Hier ist der Default 600, wie du wolltest
        new_start = max(0, current_end - 600)
        st.session_state[start_key] = new_start


def on_select_change():
    for level in TARGET_LEVELS:
        st.session_state[f"s_{level}"] = 0
        st.session_state[f"e_{level}"] = 0


# --- DATA LOADING ---
def identify_devices(df):
    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]
    val_cols = [c for c in df.columns if "ValueY" in c]
    devices = set()
    for col in val_cols:
        name = col.replace("ValueY", "").strip()
        name = re.sub(r"[_ ]?L[123][_ ]?", "", name, flags=re.IGNORECASE)
        name = name.strip("_ ")
        if name:
            devices.add(name)
    return sorted(list(devices))


def get_files():
    files = []
    for root, _, filenames in os.walk("."):
        if any(
            x in root for x in ["venv", ".git", ".idea", "__pycache__", OUTPUT_ROOT_DIR]
        ):
            continue
        for f in filenames:
            if (
                f.lower().endswith(".csv")
                and "manuelle_ergebnisse" not in f
                and "_sortiert" not in f
                and "plot_data" not in f
            ):
                files.append(os.path.join(root, f))
    return sorted(files)


def extract_metadata(filepath):
    filename = os.path.basename(filepath)
    match_amp = re.search(r"[-_](\d+)A[-_]", filename)
    nennstrom = float(match_amp.group(1)) if match_amp else 0.0
    clean_name = filename.replace(".csv", "")
    return clean_name, nennstrom


@st.cache_data
def load_file_preview(filepath):
    df = None
    encodings = ["utf-16", "utf-8", "cp1252", "latin1"]
    for enc in encodings:
        try:
            temp = pd.read_csv(
                filepath, sep=";", decimal=",", encoding=enc, engine="python", nrows=5
            )
            if len(temp.columns) < 2:
                temp = pd.read_csv(
                    filepath,
                    sep=",",
                    decimal=".",
                    encoding=enc,
                    engine="python",
                    nrows=5,
                )
            if len(temp.columns) > 1:
                df = temp
                break
        except:
            continue
    if df is None:
        return []
    return identify_devices(df)


@st.cache_data
def load_all_data(filepath, all_devices):
    df = None
    encodings = ["utf-16", "utf-8", "cp1252", "latin1"]
    for enc in encodings:
        try:
            temp = pd.read_csv(
                filepath, sep=";", decimal=",", encoding=enc, engine="python"
            )
            if len(temp.columns) < 2:
                temp = pd.read_csv(
                    filepath, sep=",", decimal=".", encoding=enc, engine="python"
                )
            if len(temp.columns) > 1:
                df = temp
                break
        except:
            continue
    if df is None:
        return None, None
    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]
    val_cols = [c for c in df.columns if "ValueY" in c]
    full_data = {dev: {} for dev in all_devices}
    t = []
    for device in all_devices:
        for phase in PHASES:
            target_col = None
            for col in val_cols:
                clean_name = col.replace("ValueY", "").strip()
                dev_name_check = re.sub(
                    r"[_ ]?L[123][_ ]?", "", clean_name, flags=re.IGNORECASE
                ).strip("_ ")
                if dev_name_check == device and phase in col:
                    target_col = col
                    break
            if target_col:
                vals = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
                full_data[device][phase] = vals
                if not t and len(vals) > 0:
                    t = list(range(len(vals)))
            else:
                full_data[device][phase] = None
    return t, full_data


def load_status_tracking():
    if os.path.exists(TRACKING_CSV):
        return pd.read_csv(TRACKING_CSV, index_col=0)
    return pd.DataFrame(columns=["Status"])


def save_tracking_status(base_name, status):
    df_track = load_status_tracking()
    df_track.loc[base_name, "Status"] = status
    df_track.to_csv(TRACKING_CSV)


def save_sorted_raw_data(
    original_filepath,
    full_data,
    start_end_map,
    export_devices,
    ref_device,
    clean_filename,
    skip_n_start=0,
):
    orig_path = Path(original_filepath)
    target_dir = Path(OUTPUT_ROOT_DIR) / orig_path.parent
    base_name = orig_path.stem
    new_filename = f"{base_name}_sortiert.csv"
    output_path = target_dir / new_filename

    remaining_devices = sorted([d for d in export_devices if d != ref_device])
    sorted_devices = (
        [ref_device] + remaining_devices
        if ref_device in export_devices
        else remaining_devices
    )

    df_export = pd.DataFrame()

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        for level in TARGET_LEVELS:
            s, e = start_end_map.get(level, (0, 0))
            adjusted_s = s + skip_n_start

            if adjusted_s > 0 and e > adjusted_s:
                for phase in PHASES:
                    for device in sorted_devices:
                        vals = full_data[device][phase]
                        if vals is not None and e <= len(vals):
                            slice_data = vals.iloc[adjusted_s:e].reset_index(drop=True)
                            col_t = f"{level:02d}_{phase}_t_{device}"
                            col_I = f"{level:02d}_{phase}_I_{device}"
                            df_export[col_t] = pd.Series(range(len(slice_data)))
                            df_export[col_I] = slice_data

        df_export.to_csv(output_path, index=False, sep=";")
        save_config(clean_filename, start_end_map)
        return str(output_path)
    except Exception as e:
        st.error(f"Save Error: {e}")
        return None


# --- SIDEBAR ---
st.sidebar.title("🛠️ Rohdaten Export")
all_files = get_files()
df_status = load_status_tracking()

files_options = []
file_map = {}
for f in all_files:
    name, _ = extract_metadata(f)
    icon = "❌"
    if name in df_status.index:
        s = df_status.loc[name, "Status"]
        if s == "OK":
            icon = "✅"
        elif s == "WARNING":
            icon = "⚠️"
    display_str = f"{icon} {name}"
    files_options.append(display_str)
    file_map[display_str] = f

if not files_options:
    st.warning("Keine CSV-Dateien gefunden.")
    st.stop()

if "selected_file_idx" not in st.session_state:
    st.session_state.selected_file_idx = 0

# --- NAVIGATION LOGIK (NEU: VOR/ZURÜCK + NÄCHSTE OFFENE) ---

# 1. Bestimme aktuellen Index
current_key = st.session_state.get(
    "file_selector", files_options[0] if files_options else None
)
current_idx = 0
if current_key in files_options:
    current_idx = files_options.index(current_key)

col_prev, col_next = st.sidebar.columns(2)

# VOR / ZURÜCK BUTTONS
if col_prev.button("⬅️ Zurück", use_container_width=True):
    new_idx = (current_idx - 1) % len(files_options)
    st.session_state["file_selector"] = files_options[new_idx]
    on_select_change()
    st.rerun()

if col_next.button("Weiter ➡️", use_container_width=True):
    new_idx = (current_idx + 1) % len(files_options)
    st.session_state["file_selector"] = files_options[new_idx]
    on_select_change()
    st.rerun()

# NÄCHSTE OFFENE BUTTON
if st.sidebar.button(
    "⏩ Nächste offene Messung", type="primary", use_container_width=True
):
    # Suche ab nächstem Index
    start_idx = (current_idx + 1) % len(files_options)
    search_list = files_options[start_idx:] + files_options[:start_idx]

    found_next = False
    for display_str in search_list:
        f_path = file_map[display_str]
        name, _ = extract_metadata(f_path)

        current_status = "NONE"
        if name in df_status.index:
            current_status = df_status.loc[name, "Status"]

        if current_status != "OK":
            st.session_state["file_selector"] = display_str
            on_select_change()
            found_next = True
            st.rerun()
            break

    if not found_next:
        st.sidebar.success("🎉 Alles erledigt! Keine offenen Messungen gefunden.")

# DATEI AUSWAHL BOX
selected_display = st.sidebar.selectbox(
    "Datei wählen:", files_options, key="file_selector", on_change=on_select_change
)
selected_file_path = file_map[selected_display]

# --- HAUPTBEREICH ---
if selected_file_path:
    wandler_name, nennstrom = extract_metadata(selected_file_path)

    saved_ranges = load_config(wandler_name)
    all_zero = all(st.session_state[f"s_{l}"] == 0 for l in TARGET_LEVELS)
    if all_zero and saved_ranges:
        for lvl_str, (s, e) in saved_ranges.items():
            lvl = int(lvl_str)
            if lvl in TARGET_LEVELS:
                st.session_state[f"s_{lvl}"] = s
                st.session_state[f"e_{lvl}"] = e

    st.header(f"Export: {wandler_name}")

    detected_devices = load_file_preview(selected_file_path)
    if not detected_devices:
        st.error("Keine Geräte erkannt.")
        st.stop()

    default_ref_index = 0
    for i, dev in enumerate(detected_devices):
        if "pac1" in dev.lower() or "einspeisung" in dev.lower():
            default_ref_index = i
            break

    st.sidebar.markdown("### 📊 Anzeige-Einstellungen")
    ref_device_for_plot = st.sidebar.selectbox(
        "Referenz für Diagramm (L1):", detected_devices, index=default_ref_index
    )

    col_info, col_status = st.columns([3, 1])
    with col_info:
        if nennstrom > 0:
            st.info(f"Nennstrom: **{int(nennstrom)} A**")
        else:
            nennstrom = st.number_input("Nennstrom manuell setzen [A]:", value=2000.0)

    t, full_data = load_all_data(selected_file_path, detected_devices)
    if not t:
        st.error("Fehler beim Laden der Daten.")
        st.stop()

    # Recovery Button
    has_active_ranges = any(st.session_state[f"e_{l}"] > 0 for l in TARGET_LEVELS)
    if not has_active_ranges:
        st.markdown("---")
        col_rec1, col_rec2 = st.columns([1, 4])
        with col_rec1:
            if st.button("♻️ Bereiche aus sortierter Datei suchen"):
                with st.spinner("Durchsuche Originaldaten nach Mustern..."):
                    l1_data = full_data[ref_device_for_plot]["L1"]
                    success, msg = try_recover_from_sorted_file(
                        selected_file_path, l1_data, ref_device_for_plot
                    )
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        with col_rec2:
            st.caption(
                "Versucht, bereits sortierte Daten (falls vorhanden) im Original wiederzufinden."
            )

    # Plot
    ref_l1 = full_data[ref_device_for_plot]["L1"]
    if ref_l1 is not None:
        ref_pct = (ref_l1 / nennstrom) * 100
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=t,
                y=ref_pct,
                mode="lines",
                name=f"{ref_device_for_plot} L1 [%]",
                line=dict(color="orange", width=2),
            )
        )

        for level in TARGET_LEVELS:
            fig.add_hline(y=level, line_dash="dot", line_color="rgba(150,150,150,0.5)")
            s_val = st.session_state.get(f"s_{level}", 0)
            e_val = st.session_state.get(f"e_{level}", 0)
            if s_val >= 0 and e_val > s_val:
                fig.add_vrect(
                    x0=s_val, x1=e_val, fillcolor="rgba(0, 200, 100, 0.2)", line_width=0
                )
                fig.add_annotation(
                    x=(s_val + e_val) / 2,
                    y=level + 2,
                    text=f"<b>{level}%</b>",
                    showarrow=False,
                    font=dict(color="green"),
                )

        fig.update_layout(
            title=f"Verlauf {ref_device_for_plot} - L1",
            xaxis_title="Zeit [Index]",
            yaxis_title="Last [%]",
            height=400,
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, width="stretch")

    # Inputs
    st.markdown("### ✂️ Bereiche definieren")
    current_status = "OK"
    if wandler_name in df_status.index:
        current_status = df_status.loc[wandler_name, "Status"]
    status_idx = 0 if current_status == "OK" else 1
    status_selection = st.radio(
        "Status:", ["OK", "Problem (⚠️)"], index=status_idx, horizontal=True
    )
    st.markdown("---")

    def chunked(iterable, n):
        return [iterable[i : i + n] for i in range(0, len(iterable), n)]

    batches = chunked(TARGET_LEVELS, 4)
    for batch in batches:
        cols_head = st.columns(4)
        for i, level in enumerate(batch):
            cols_head[i].markdown(f"**{level}%**")
        cols_ende = st.columns(4)
        for i, level in enumerate(batch):
            cols_ende[i].number_input(
                f"Ende",
                min_value=0,
                step=1,
                key=f"e_{level}",
                on_change=update_start_callback,
                args=(level,),
                label_visibility="collapsed",
            )
        cols_start = st.columns(4)
        for i, level in enumerate(batch):
            cols_start[i].number_input(
                f"Start",
                min_value=0,
                step=1,
                key=f"s_{level}",
                label_visibility="collapsed",
            )
        st.markdown("")

    st.markdown("---")

    # --- EXPORT EINSTELLUNGEN ---
    col_sel, col_skip = st.columns([2, 1])

    with col_sel:
        # Multiselect für Geräte
        selected_export_devices = st.multiselect(
            "Geräte für Export auswählen:", detected_devices, default=detected_devices
        )
        if ref_device_for_plot in selected_export_devices:
            st.caption(f"ℹ️ '{ref_device_for_plot}' wird als erstes exportiert.")
        else:
            st.warning(f"⚠️ Referenz '{ref_device_for_plot}' ist nicht ausgewählt!")

    with col_skip:
        skip_n = st.number_input(
            "🛡️ Einschwingen ignorieren (Werte):",
            min_value=0,
            value=0,
            help="Anzahl der Messwerte, die am Anfang des gewählten Bereichs abgeschnitten werden.",
        )

    st.markdown("")

    if st.button("🚀 Rohdaten Exportieren (Gewählte Geräte)", type="primary"):
        if not selected_export_devices:
            st.error("Bitte mindestens ein Gerät auswählen.")
        else:
            start_end_map = {}
            for level in TARGET_LEVELS:
                s_val = st.session_state[f"s_{level}"]
                e_val = st.session_state[f"e_{level}"]
                start_end_map[level] = (s_val, e_val)

            new_file_path = save_sorted_raw_data(
                selected_file_path,
                full_data,
                start_end_map,
                selected_export_devices,
                ref_device_for_plot,
                wandler_name,
                skip_n_start=skip_n,
            )

            if new_file_path:
                status_code = "WARNING" if "Problem" in status_selection else "OK"
                save_tracking_status(wandler_name, status_code)
                st.success(
                    f"Gespeichert!\nGeräte: {', '.join(selected_export_devices)}\nPfad: `{new_file_path}`"
                )
