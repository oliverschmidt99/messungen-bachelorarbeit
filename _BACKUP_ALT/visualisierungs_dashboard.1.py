import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import io
import zipfile
import subprocess
import re
import json

# --- KONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "daten")  # Verweis auf Unterordner

DATA_FILE = os.path.join(DATA_DIR, "messdaten_db.parquet")

# NEU: Intelligente Suche nach der Config-Datei
# Prüft Reihenfolge: daten/saved_config.json -> root/saved_config.json -> daten/saved_configs.json
possible_configs = [
    os.path.join(DATA_DIR, "saved_config.json"),
    os.path.join(BASE_DIR, "saved_config.json"),
    os.path.join(DATA_DIR, "saved_configs.json")
]

CONFIG_FILE = os.path.join(DATA_DIR, "saved_configs.json") # Standard-Fallback
for f in possible_configs:
    if os.path.exists(f):
        CONFIG_FILE = f
        break

WORK_DIR = os.path.join(BASE_DIR, "matlab_working_dir")

PHASES = ["L1", "L2", "L3"]
ZONES = {
    "Niederstrom (5-50%)": [5, 20, 50],
    "Nennstrom (80-100%)": [80, 90, 100],
    "Überlast (≥120%)": [120, 150, 200],
}

# --- DEFINITION DER SPALTEN (WICHTIG FÜR TAB 3) ---
META_COLS_EDIT = [
    "Preis (€)",
    "Nennbürde (VA)",
    "T (mm)",
    "B (mm)",
    "H (mm)",
    "Kommentar",
]
META_COLS_FIX = [
    "Hersteller",
    "Modell",
    "nennstrom",
    "Mess-Bürde",
    "Geometrie",
    "raw_file",
]
META_COLS = META_COLS_FIX + META_COLS_EDIT

# --- STYLES ---
BLUES = [
    "#1f4e8c",
    "#2c6fb2",
    "#4a8fd1",
    "#6aa9e3",
    "#8fc0ee",
    "#b3d5f7",
    "#d6e8fb",
    "#5f7fd9",
    "#6fa3ff",
    "#8bb8ff",
    "#a6ccff",
    "#cfe2ff",
]
ORANGES = [
    "#8c4a2f",
    "#a65a2a",
    "#c96a2a",
    "#e07b39",
    "#f28e4b",
    "#f6a25e",
    "#f9b872",
    "#f4a261",
    "#f7b267",
    "#ff9f4a",
    "#ffb347",
    "#ffd199",
]
OTHERS = [
    "#4caf50",
    "#6bd36b",
    "#b0b0b0",
    "#b39ddb",
    "#bc8f6f",
    "#f2a7d6",
    "#d4d65a",
    "#6fd6e5",
]

# Plotly Optionen
LINE_STYLES = ["solid", "dash", "dot", "dashdot", "longdash"]
MARKER_SYMBOLS = ["circle", "square", "diamond", "cross", "x", "triangle-up", "star"]


# --- HELPER: CONFIG MANAGEMENT ---
def load_all_configs():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_new_config(name, data):
    configs = load_all_configs()
    configs[name] = data
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(configs, f, ensure_ascii=False, indent=2)


def delete_config(name):
    configs = load_all_configs()
    if name in configs:
        del configs[name]
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(configs, f, ensure_ascii=False, indent=2)
        return True
    return False


# --- HELPER: FILE NAMING ---
def sanitize_filename(name):
    """Entfernt ungültige Zeichen für Dateinamen."""
    if not name:
        return "Unbenannt"
    # Ersetze ungültige Zeichen durch Unterstrich oder leer
    clean = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    return clean.replace(" ", "_")


# --- HELPER: PLOTTING & DATA ---
def get_trumpet_limits(class_val):
    x = [1, 5, 20, 100, 120]
    if class_val == 0.2:
        y = [0.75, 0.35, 0.2, 0.2, 0.2]
    elif class_val == 0.5:
        y = [1.5, 1.5, 0.75, 0.5, 0.5]
    elif class_val == 1.0:
        y = [3.0, 1.5, 1.0, 1.0, 1.0]
    elif class_val == 3.0:
        y = [None, 3.0, 3.0, 3.0, 3.0]
    else:
        y = [1.5, 1.5, 0.75, 0.5, 0.5]
    y_neg = [-v if v is not None else None for v in y]
    return x, y, y_neg


def auto_format_name(row):
    raw_key = str(row["wandler_key"])
    folder_lower = str(row["folder"]).lower()
    tokens = re.split(r"[_\s]+", raw_key)
    name_parts = []
    burden_part = ""
    for t in tokens:
        if not t:
            continue
        if re.match(r"^\d+R\d*$", t):
            burden_part = f"{t.replace('R', ',')} $\Omega$"
        else:
            name_parts.append(t)
    base_name = " ".join(name_parts)
    dut = str(row["dut_name"])
    if dut.lower() not in base_name.lower():
        base_name = f"{base_name} | {dut}"
    if burden_part:
        base_name = f"{base_name} | {burden_part}"
    if "parallel" in folder_lower:
        base_name += " | Parallel"
    elif "dreieck" in folder_lower:
        base_name += " | Dreieck"
    if "Kommentar" in row and row["Kommentar"]:
        clean_comment = str(row["Kommentar"]).strip()
        if clean_comment and clean_comment != "0.0":
            base_name += f" | {clean_comment}"
    if "nennstrom" in row and row["nennstrom"] > 0:
        base_name += f" ({int(row['nennstrom'])}A)"
    return base_name


def extract_base_type(wandler_key):
    s = str(wandler_key)
    tokens = re.split(r"[_\s\-]+", s)
    clean_tokens = []
    for t in tokens:
        if re.match(r"^\d+R\d*$", t, re.IGNORECASE):
            continue
        if t.lower() in ["parallel", "dreieck", "messstrecke", "l1", "l2", "l3"]:
            continue
        if not t:
            continue
        clean_tokens.append(t)
    return " ".join(clean_tokens)


def parse_filename_info(filename_str):
    if not isinstance(filename_str, str):
        return pd.Series(["Unbekannt", "Unbekannt", 0.0, "Unbekannt"])
    base = os.path.basename(filename_str)
    name_only = os.path.splitext(base)[0]
    parts = name_only.split("-")
    if len(parts) < 5:
        return pd.Series(["Unbekannt", "Unbekannt", 0.0, "Unbekannt"])
    try:
        hersteller = parts[1].replace("_", " ")
        modell = parts[2].replace("_", " ")
        strom_str = parts[3].upper().replace("A", "")
        nennstrom = float(strom_str) if strom_str.replace(".", "", 1).isdigit() else 0.0
        burde_part = parts[4]
        mess_burde = burde_part
        return pd.Series([hersteller, modell, nennstrom, mess_burde])
    except:
        return pd.Series(["Fehler", "Fehler", 0.0, "Fehler"])


def create_single_phase_figure(
    df_sub,
    phase,
    acc_class,
    y_limit,
    bottom_mode,
    show_err_bars,
    title_prefix="",
    nticks_x=20,
    nticks_y=15,  # <--- NEUE PARAMETER HIER
):
    """
    bottom_mode: 'Standardabweichung', 'Messwert (Absolut)', 'Ausblenden'
    """
    is_single_row = bottom_mode == "Ausblenden"

    if is_single_row:
        # Nur 1 Zeile -> Der Plot nutzt 100% der Höhe
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=[f"Fehlerverlauf {phase}"],
        )
    else:
        # 2 Zeilen mit Aufteilung 70% / 30%
        sec_title = (
            "Standardabweichung"
            if bottom_mode == "Standardabweichung"
            else "Messwert (Absolut)"
        )
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"Fehlerverlauf {phase}", f"{sec_title} {phase}"),
        )

    # --- Plotting Main Error (Row 1) ---
    lim_x, lim_y_p, lim_y_n = get_trumpet_limits(acc_class)
    fig.add_trace(
        go.Scatter(
            x=lim_x,
            y=lim_y_p,
            mode="lines",
            line=dict(color="black", width=1.5, dash="dash"),
            name="Klassengrenze",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=lim_x,
            y=lim_y_n,
            mode="lines",
            line=dict(color="black", width=1.5, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    phase_data = df_sub[df_sub["phase"] == phase]
    for uid, group in phase_data.groupby("unique_id"):
        group = group.sort_values("target_load")
        row_first = group.iloc[0]

        # NEU: Check Visibility
        if not row_first["final_visible"]:
            continue

        leg_name = row_first["final_legend"]
        color = row_first["final_color"]
        style = row_first["final_style"]
        symbol = row_first["final_symbol"]
        width = row_first["final_width"]
        size_val = row_first["final_size"]

        # 1. Main Plot
        fig.add_trace(
            go.Scatter(
                x=group["target_load"],
                y=group["err_ratio"],
                mode="lines+markers",
                name=leg_name,
                line=dict(color=color, width=width, dash=style),
                marker=dict(size=size_val, symbol=symbol),
                legendgroup=leg_name,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # 2. Secondary Plot (nur wenn NICHT ausgeblendet)
        if not is_single_row:
            if bottom_mode == "Standardabweichung":
                if show_err_bars:  # Berücksichtigt Checkbox, falls relevant
                    fig.add_trace(
                        go.Bar(
                            x=group["target_load"],
                            y=group["err_std"],
                            marker_color=color,
                            legendgroup=leg_name,
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )
            elif bottom_mode == "Messwert (Absolut)":
                fig.add_trace(
                    go.Scatter(
                        x=group["target_load"],
                        y=group["val_dut_mean"],
                        mode="lines+markers",
                        line=dict(color=color, width=1.5, dash="dot"),
                        marker=dict(symbol="x", size=size_val),
                        legendgroup=leg_name,
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

    fig.update_layout(
        title=dict(text=f"{title_prefix} - Phase {phase}", font=dict(size=18)),
        template="plotly_white",
        width=1123,
        height=794,
        font=dict(family="Serif", size=14, color="black"),
        legend=dict(
            orientation="h",
            y=-0.15,  # Legende unter das Diagramm
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=16),
        ),
        margin=dict(l=60, r=30, t=80, b=120),
    )

    fig.update_yaxes(range=[-y_limit, y_limit], title_text="Fehler [%]", row=1, col=1)

    if not is_single_row:
        if bottom_mode == "Standardabweichung":
            fig.update_yaxes(title_text="StdAbw [%]", row=2, col=1)
        else:
            fig.update_yaxes(title_text="Strom [A]", row=2, col=1)
        fig.update_xaxes(title_text="Strom [% In]", row=2, col=1)
    else:
        # X-Achsen-Label an Row 1, da Row 2 nicht existiert
        fig.update_xaxes(title_text="Strom [% In]", row=1, col=1)

    fig.update_xaxes(nticks=nticks_x)
    fig.update_yaxes(nticks=nticks_y)

    return fig


# --- MATLAB TEMPLATE ---
MATLAB_SCRIPT_TEMPLATE = r"""
%% Automatische Diagrammerstellung
clear; clc; close all;
try
    filename = 'plot_data.csv';
    phases = {'L1', 'L2', 'L3'};
    limits_class = ACC_CLASS_PH; 
    nennstrom = NOMINAL_CURRENT_PH;
    if ~isfile(filename); error('Datei plot_data.csv nicht gefunden!'); end
    data = readtable(filename, 'Delimiter', ',');
    hex2rgb = @(hex) sscanf(hex(2:end),'%2x%2x%2x',[1 3])/255;
    x_lims = [1, 5, 20, 100, 120];
    if limits_class == 0.2; y_lims = [0.75, 0.35, 0.2, 0.2, 0.2];
    elseif limits_class == 0.5; y_lims = [1.5, 1.5, 0.75, 0.5, 0.5];
    elseif limits_class == 1.0; y_lims = [3.0, 1.5, 1.0, 1.0, 1.0];
    else; y_lims = [1.5, 1.5, 0.75, 0.5, 0.5]; end

    for i = 1:length(phases)
        p = phases{i};
        rows = strcmp(data.phase, p);
        sub_data = data(rows, :);
        if isempty(sub_data); continue; end
        
        f = figure('Visible', 'off', 'PaperType', 'A4', 'PaperOrientation', 'landscape');
        set(f, 'Color', 'w', 'Units', 'centimeters', 'Position', [0 0 29.7 21]);
        t = tiledlayout(f, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        ax1 = nexttile; hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
        plot(ax1, x_lims, y_lims, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        plot(ax1, x_lims, -y_lims, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        
        [unique_traces, ~, ~] = unique(sub_data.trace_id, 'stable');
        for k = 1:length(unique_traces)
            trace = unique_traces{k};
            d = sub_data(strcmp(sub_data.trace_id, trace), :);
            [~, sort_idx] = sort(d.target_load); d = d(sort_idx, :);
            col_rgb = hex2rgb(d.color_hex{1});
            leg_lbl = strrep(d.legend_name{1}, '_', '\_');
            plot(ax1, d.target_load, d.err_ratio, '-o', 'Color', col_rgb, 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', col_rgb, 'DisplayName', leg_lbl);
        end
        title(ax1, sprintf('Fehlerverlauf - Phase %s', p));
        ylabel(ax1, 'Fehler [%]'); ylim(ax1, [- Y_LIMIT_PH, Y_LIMIT_PH]); xlim(ax1, [0, 125]);
        
        legend(ax1, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Interpreter', 'tex', 'Box', 'off');
        
        ax2 = nexttile; hold(ax2, 'on'); grid(ax2, 'on'); box(ax2, 'on');
        num_groups = length(unique_traces);
        single_bar_width = min(1.5, 5 / num_groups);
        for k = 1:length(unique_traces)
            trace = unique_traces{k};
            d = sub_data(strcmp(sub_data.trace_id, trace), :);
            [~, sort_idx] = sort(d.target_load); d = d(sort_idx, :);
            col_rgb = hex2rgb(d.color_hex{1});
            x_pos = d.target_load + (k - 1 - (num_groups - 1) / 2) * single_bar_width;
            bar(ax2, x_pos, d.err_std, 'FaceColor', col_rgb, 'EdgeColor', 'none', 'FaceAlpha', 0.8, 'BarWidth', 0.1);
        end
        title(ax2, sprintf('Standardabweichung - Phase %s', p)); ylabel(ax2, 'StdAbw [%]'); xlabel(ax2, 'Last [% In]'); xlim(ax2, [0, 125]);
        exportgraphics(f, sprintf('Detail_%s_MultiCurrent.pdf', p), 'ContentType', 'vector');
        close(f);
    end
catch ME; disp(ME.message); exit(1); end
exit(0);
"""

# ==========================================
# --- HELPER FÜR TAB 4 (ROHDATEN SELECTOR) ---
# ==========================================
SELECTOR_TRACKING_CSV = os.path.join(DATA_DIR, "manuelle_ergebnisse.csv")


def extract_metadata(filepath):
    filename = os.path.basename(filepath)
    # Sucht nach Muster wie "-2000A-" oder "_2000A_"
    match_amp = re.search(r"[-_](\d+)A[-_]", filename)
    nennstrom = float(match_amp.group(1)) if match_amp else 0.0
    clean_name = filename.replace(".csv", "")
    return clean_name, nennstrom


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
    # Pfad-Logik angepasst an Haupt-App Struktur
    orig_p = Path(original_path)
    # Wir suchen im "messungen_sortiert" Ordner im Root oder relativ
    sorted_root = os.path.join(BASE_DIR, "messungen_sortiert")
    # Versuche den relativen Pfad abzubilden
    try:
        rel_path = orig_p.relative_to(os.getcwd())
        target_dir = Path(sorted_root) / rel_path.parent
    except:
        target_dir = Path(sorted_root) / orig_p.parent.name

    sorted_filename = f"{orig_p.stem}_sortiert.csv"
    sorted_path = target_dir / sorted_filename

    if not sorted_path.exists():
        return False, f"Keine Datei gefunden: {sorted_path}"

    try:
        df_sorted = pd.read_csv(sorted_path, sep=";", decimal=".", engine="python")
        if df_sorted.shape[1] < 2:
            df_sorted = pd.read_csv(sorted_path, sep=";", decimal=",", engine="python")

        recovered_count = 0
        levels = [5, 20, 50, 80, 90, 100, 120]
        for level in levels:
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
        return False, "Keine Positionen gefunden."
    except Exception as e:
        return False, f"Fehler: {e}"


def identify_devices_tab4(df):
    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]
    val_cols = [c for c in df.columns if "ValueY" in c]
    devices = set()
    for col in val_cols:
        name = col.replace("ValueY", "").strip()
        name = re.sub(r"[_ ]?L[123][_ ]?", "", name, flags=re.IGNORECASE).strip("_ ")
        if name:
            devices.add(name)
    return sorted(list(devices))


def get_files_tab4():
    files = []
    # Suche im aktuellen Verzeichnis rekursiv
    for root, _, filenames in os.walk(BASE_DIR):
        if any(
            x in root
            for x in [
                "venv",
                ".git",
                "__pycache__",
                "messungen_sortiert",
                "matlab_working_dir",
            ]
        ):
            continue
        for f in filenames:
            if (
                f.lower().endswith(".csv")
                and "manuelle" not in f
                and "_sortiert" not in f
                and "plot_data" not in f
            ):
                files.append(os.path.join(root, f))
    return sorted(files)


@st.cache_data
def load_file_preview_tab4(filepath):
    df = None
    for enc in ["utf-16", "utf-8", "cp1252", "latin1"]:
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
    return identify_devices_tab4(df) if df is not None else []


@st.cache_data
def load_all_data_tab4(filepath, all_devices):
    df = None
    for enc in ["utf-16", "utf-8", "cp1252", "latin1"]:
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
    full_data = {dev: {} for dev in all_devices}
    t = []
    val_cols = [c for c in df.columns if "ValueY" in c]

    for device in all_devices:
        for phase in PHASES:
            target_col = None
            for col in val_cols:
                clean = col.replace("ValueY", "").strip()
                check = re.sub(
                    r"[_ ]?L[123][_ ]?", "", clean, flags=re.IGNORECASE
                ).strip("_ ")
                if check == device and phase in col:
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
    if os.path.exists(SELECTOR_TRACKING_CSV):
        return pd.read_csv(SELECTOR_TRACKING_CSV, index_col=0)
    return pd.DataFrame(columns=["Status"])


def save_tracking_status(base_name, status):
    df_track = load_status_tracking()
    df_track.loc[base_name, "Status"] = status
    df_track.to_csv(SELECTOR_TRACKING_CSV)


def update_start_callback(lvl):
    if st.session_state[f"s_{lvl}"] == 0:
        st.session_state[f"s_{lvl}"] = max(0, st.session_state[f"e_{lvl}"] - 600)


def save_sorted_raw_data_tab4(
    original_filepath,
    full_data,
    start_end_map,
    export_devices,
    ref_device,
    clean_filename,
    skip_n_start=0,
):
    orig_path = Path(original_filepath)
    # Output Ordner fixiert auf "messungen_sortiert" im Base Dir
    target_dir = Path(BASE_DIR) / "messungen_sortiert" / orig_path.parent.name
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / f"{orig_path.stem}_sortiert.csv"

    sorted_devs = [ref_device] + sorted([d for d in export_devices if d != ref_device])
    df_export = pd.DataFrame()

    try:
        levels = [5, 20, 50, 80, 90, 100, 120]
        for level in levels:
            s, e = start_end_map.get(level, (0, 0))
            adj_s = s + skip_n_start
            if adj_s > 0 and e > adj_s:
                for phase in PHASES:
                    for device in sorted_devs:
                        vals = full_data[device][phase]
                        if vals is not None and e <= len(vals):
                            slice_data = vals.iloc[adj_s:e].reset_index(drop=True)
                            df_export[f"{level:02d}_{phase}_t_{device}"] = pd.Series(
                                range(len(slice_data))
                            )
                            df_export[f"{level:02d}_{phase}_I_{device}"] = slice_data

        df_export.to_csv(output_path, index=False, sep=";")

        # Config speichern in die zentrale JSON
        save_new_config(clean_filename, start_end_map)
        return str(output_path)
    except Exception as e:
        return None


@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_parquet(DATA_FILE)
    if "raw_file" in df.columns:
        df["raw_file"] = (
            df["raw_file"]
            .astype(str)
            .apply(
                lambda x: (
                    x.replace("['", "").replace("']", "") if x.startswith("['") else x
                )
            )
        )
    if "dut_name" in df.columns:
        df["trace_id"] = df["folder"] + " | " + df["dut_name"].astype(str)
    else:
        df["trace_id"] = df["folder"]
    if "target_load" in df.columns:
        df["target_load"] = pd.to_numeric(df["target_load"], errors="coerce")
    if "base_type" not in df.columns:
        df["base_type"] = df["wandler_key"].apply(extract_base_type)
    for col in ["Hersteller", "Modell", "nennstrom", "Mess-Bürde", "Geometrie"]:
        if col not in df.columns:
            df[col] = 0.0 if col == "nennstrom" else "Unbekannt"
    for col in META_COLS_EDIT:
        if col not in df.columns:
            df[col] = "" if col == "Kommentar" else 0.0
    if "Kommentar" in df.columns:
        df["Kommentar"] = (
            df["Kommentar"].astype(str).replace("nan", "").replace("None", "")
        )
    if "val_dut_mean" not in df.columns:
        df["val_dut_mean"] = 0.0
    return df


def save_db(df_to_save):
    try:
        df_to_save.to_parquet(DATA_FILE)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(str(e))
        return False


def ensure_working_dir():
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)
    return os.path.abspath(WORK_DIR)


# --- APP START ---
st.set_page_config(page_title="Wandler Dashboard", layout="wide", page_icon="📈")

# --- HIER EINFÜGEN: CSS FÜR BREITERE SIDEBAR ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 600px;
        max-width: 900px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# -----------------------------------------------

df = load_data()
if df is None:
    st.error(f"⚠️ Datei '{DATA_FILE}' fehlt.")
    st.stop()

# --- SIDEBAR: CONFIG ---
st.sidebar.header("💾 Konfiguration")
all_configs = load_all_configs()
config_names = sorted(list(all_configs.keys()))

with st.sidebar.expander("Laden & Speichern", expanded=True):
    sel_config_load = st.selectbox(
        "Gespeicherte Config wählen:", ["-- Neu / Leer --"] + config_names
    )
    col_load, col_del = st.columns([1, 1])
    with col_load:
        if st.button("📂 Laden", use_container_width=True):
            if sel_config_load != "-- Neu / Leer --":
                data = all_configs[sel_config_load]
                # Filter Values
                st.session_state["k_current"] = data.get("current", [])
                st.session_state["k_geos"] = data.get("geos", [])
                st.session_state["k_wandlers"] = data.get("wandlers", [])
                st.session_state["k_duts"] = data.get("duts", [])
                st.session_state["k_comp"] = data.get(
                    "comp_mode", "Messgerät (z.B. PAC1)"
                )
                # Design
                st.session_state["k_sync"] = data.get("sync_axes", True)
                st.session_state["k_ylim"] = data.get("y_limit", 1.5)
                st.session_state["k_class"] = data.get("acc_class", 0.2)

                saved_err_bool = data.get("show_err_bars", True)

                # Bottom Plot logic
                saved_mode = data.get("bottom_plot_mode", "Standardabweichung")
                if not saved_err_bool and saved_mode == "Standardabweichung":
                    # Fallback für alte Configs wo es nur checkbox gab
                    pass

                st.session_state["k_bottom_mode"] = saved_mode
                st.session_state["k_errbars"] = saved_err_bool

                # Eco
                st.session_state["k_eco_x"] = data.get("eco_x", "Preis (€)")
                st.session_state["k_eco_y"] = data.get("eco_y", ["Fehler Nennstrom"])
                st.session_state["k_eco_type"] = data.get("eco_type", "Scatter")
                # Custom
                st.session_state["loaded_colors"] = data.get("custom_colors", {})
                st.session_state["loaded_legends"] = data.get("custom_legends", {})
                st.session_state["loaded_titles"] = data.get("custom_titles", {})
                st.session_state["loaded_styles"] = data.get("custom_styles", {})
                st.session_state["loaded_symbols"] = data.get("custom_symbols", {})
                st.session_state["loaded_widths"] = data.get("custom_widths", {})
                st.session_state["loaded_visible"] = data.get("custom_visible", {})
                st.session_state["loaded_sizes"] = data.get("custom_sizes", {})
                st.success(f"'{sel_config_load}' geladen!")
                st.rerun()
    with col_del:
        if st.button("🗑️ Löschen", use_container_width=True):
            if delete_config(sel_config_load):
                st.success("Gelöscht!")
                st.rerun()
    st.divider()
    new_config_name = st.text_input("Name für aktuelle Ansicht:")
    if st.button("💾 Speichern", use_container_width=True):
        if new_config_name:
            st.session_state["trigger_save"] = True
            st.session_state["save_name"] = new_config_name
        else:
            st.warning("Bitte Namen eingeben.")

# --- SIDEBAR: FILTER ---
st.sidebar.header("🎛️ Globale Filter")

available_currents = sorted(df["nennstrom"].unique())
default_curr = st.session_state.get(
    "k_current", [available_currents[0]] if len(available_currents) > 0 else []
)
if not isinstance(default_curr, list):
    default_curr = [default_curr]
default_curr = [c for c in default_curr if c in available_currents]

if "k_current" not in st.session_state:
    st.session_state["k_current"] = default_curr

sel_currents = st.sidebar.multiselect(
    "1. Nennstrom (Mehrfachauswahl):",
    available_currents,
    format_func=lambda x: f"{int(x)} A",
    key="k_current",
)

if not sel_currents:
    st.warning("Bitte mindestens einen Nennstrom auswählen.")
    st.stop()

# 1. Schritt: Daten nach Strom filtern
df_curr = df[df["nennstrom"].isin(sel_currents)]
available_geos = sorted(df_curr["Geometrie"].astype(str).unique())

# Sanitize Geos
# Sanitize Geos
saved_geos = st.session_state.get("k_geos", available_geos)
valid_geos = [g for g in saved_geos if g in available_geos]

# Fallback: Falls die Auswahl leer ist oder ungültig wurde, nimm alles
if not valid_geos and available_geos:
    valid_geos = available_geos

# KORREKTUR 1: Den bereinigten Wert explizit in den Session State schreiben
st.session_state["k_geos"] = valid_geos

# KORREKTUR 2: Das 'default'-Argument entfernen (Streamlit nimmt jetzt den Wert aus dem Session State)
sel_geos = st.sidebar.multiselect("2. Geometrie:", available_geos, key="k_geos")

if not sel_geos:
    st.stop()

# 2. Schritt: Daten nach Geometrie filtern
df_geo_filtered = df_curr[df_curr["Geometrie"].isin(sel_geos)]
available_wandlers = sorted(df_geo_filtered["wandler_key"].unique())

# Sanitize Wandler
saved_wandlers = st.session_state.get("k_wandlers", available_wandlers)
valid_wandlers = [w for w in saved_wandlers if w in available_wandlers]
if not valid_wandlers and available_wandlers:
    valid_wandlers = available_wandlers

st.session_state["k_wandlers"] = valid_wandlers

sel_wandlers = st.sidebar.multiselect(
    "3. Wandler / Messung:",
    available_wandlers,
    key="k_wandlers",
)

if not sel_wandlers:
    st.stop()

# 3. Schritt: Daten nach Wandler filtern
df_wandler_subset = df_geo_filtered[df_geo_filtered["wandler_key"].isin(sel_wandlers)]
available_duts = sorted(df_wandler_subset["dut_name"].unique())

# Sanitize DUTs
saved_duts = st.session_state.get("k_duts", available_duts)
valid_duts = [d for d in saved_duts if d in available_duts]
if not valid_duts and available_duts:
    valid_duts = available_duts

st.session_state["k_duts"] = valid_duts

sel_duts = st.sidebar.multiselect(
    "4. Geräte (DUTs) auswählen:", available_duts, key="k_duts"
)

comp_options = ["Messgerät (z.B. PAC1)", "Nennwert (Ideal)"]
try:
    c_idx = comp_options.index(st.session_state.get("k_comp", comp_options[0]))
except:
    c_idx = 0
comp_mode_disp = st.sidebar.radio(
    "Vergleichsgrundlage:", comp_options, index=c_idx, key="k_comp"
)
comp_mode_val = "device_ref" if "Messgerät" in comp_mode_disp else "nominal_ref"

# Finaler Filter
mask = (
    (df["nennstrom"].isin(sel_currents))
    & (df["Geometrie"].isin(sel_geos))
    & (df["wandler_key"].isin(sel_wandlers))
    & (df["dut_name"].isin(sel_duts))
)
if "comparison_mode" in df.columns:
    mask = mask & (df["comparison_mode"] == comp_mode_val)

df_sub = df[mask].copy()
if df_sub.empty:
    st.warning("⚠️ Keine Daten.")
    st.stop()

if comp_mode_val == "device_ref" and "ref_name" in df_sub.columns:
    df_sub = df_sub[df_sub["dut_name"] != df_sub["ref_name"]]
df_sub["unique_id"] = df_sub["raw_file"] + " | " + df_sub["dut_name"].astype(str)
df_sub["err_ratio"] = (
    (df_sub["val_dut_mean"] - df_sub["val_ref_mean"]) / df_sub["val_ref_mean"]
) * 100
df_sub["err_std"] = (df_sub["val_dut_std"] / df_sub["val_ref_mean"]) * 100

current_title_str = ", ".join([str(int(c)) for c in sorted(sel_currents)]) + " A"

# --- SIDEBAR: DESIGN ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Design & Settings")
sync_axes = st.sidebar.checkbox(
    "🔗 Phasen synchronisieren",
    value=st.session_state.get("k_sync", True),
    key="k_sync",
)
y_limit = st.sidebar.slider(
    "Y-Achse Zoom (+/- %)",
    0.2,
    10.0,
    float(st.session_state.get("k_ylim", 1.5)),
    0.1,
    key="k_ylim",
)
acc_class = st.sidebar.selectbox(
    "Norm-Klasse",
    [0.2, 0.5, 1.0, 3.0],
    index=[0.2, 0.5, 1.0, 3.0].index(st.session_state.get("k_class", 0.2)),
    key="k_class",
)


# --- NEU EINFÜGEN START ---
st.sidebar.markdown("#### 📏 Achsen-Auflösung")
col_ticks_x, col_ticks_y = st.sidebar.columns(2)
with col_ticks_x:
    nticks_x = st.slider("Ticks X-Achse", 5, 50, 20, key="k_nticks_x")
with col_ticks_y:
    nticks_y = st.slider("Ticks Y-Achse", 5, 50, 15, key="k_nticks_y")
# --- NEU EINFÜGEN ENDE ---

# NEUE AUSWAHL FÜR UNTERES DIAGRAMM
bottom_plot_options = ["Standardabweichung", "Messwert (Absolut)", "Ausblenden"]
try:
    saved_mode = st.session_state.get("k_bottom_mode", "Standardabweichung")
    b_idx = bottom_plot_options.index(saved_mode)
except:
    b_idx = 0

bottom_plot_mode = st.sidebar.selectbox(
    "Unteres Diagramm:", bottom_plot_options, index=b_idx, key="k_bottom_mode"
)

# WICHTIG: DIESE VARIABLE STEUERT DAS LAYOUT
use_single_row = bottom_plot_mode == "Ausblenden"


show_err_bars = st.sidebar.checkbox(
    "Fehlerbalken (StdAbw)",
    value=st.session_state.get("k_errbars", True),
    key="k_errbars",
)


# --- CUSTOMIZATION: FARBEN, NAMEN, STILE, SICHTBARKEIT, GRÖSSE ---
# --- CUSTOMIZATION: FARBEN, NAMEN, STILE, SICHTBARKEIT, GRÖSSE ---
with st.sidebar.expander("Farben & Namen bearbeiten", expanded=False):
    # 1. "Geometrie" zur Liste hinzufügen
    unique_curves = df_sub[
        [
            "unique_id",
            "wandler_key",
            "folder",
            "dut_name",
            "Kommentar",
            "nennstrom",
            "Geometrie",
        ]
    ].drop_duplicates()

    loaded_colors = st.session_state.get("loaded_colors", {})
    loaded_legends = st.session_state.get("loaded_legends", {})
    loaded_styles = st.session_state.get("loaded_styles", {})
    loaded_symbols = st.session_state.get("loaded_symbols", {})
    loaded_widths = st.session_state.get("loaded_widths", {})
    loaded_visible = st.session_state.get("loaded_visible", {})
    loaded_sizes = st.session_state.get("loaded_sizes", {})

    config_data = []
    b_idx, o_idx, x_idx = 0, 0, 0
    for idx, row in unique_curves.iterrows():
        uid = row["unique_id"]
        auto_name = auto_format_name(row)
        folder_lower = str(row["folder"]).lower()
        if "parallel" in folder_lower:
            col = BLUES[b_idx % len(BLUES)]
            b_idx += 1
        elif "dreieck" in folder_lower:
            col = ORANGES[o_idx % len(ORANGES)]
            o_idx += 1
        else:
            col = OTHERS[x_idx % len(OTHERS)]
            x_idx += 1

        cur_legend = loaded_legends.get(uid, auto_name)
        cur_color = loaded_colors.get(uid, col)
        cur_style = loaded_styles.get(uid, "solid")
        cur_symbol = loaded_symbols.get(uid, "circle")
        cur_width = loaded_widths.get(uid, 2.5)
        cur_visible = loaded_visible.get(uid, True)
        cur_size = loaded_sizes.get(uid, 8)

        # --- HIER IST DIE FEHLENDE ZEILE ---
        cur_geo = str(row["Geometrie"])
        # -----------------------------------

        config_data.append(
            {
                "ID": uid,
                "Anzeigen": cur_visible,
                "Legende": cur_legend,
                "Geometrie": cur_geo,  # Jetzt ist cur_geo bekannt
                "Farbe": cur_color,
                "Linie": cur_style,
                "Breite": cur_width,
                "Marker": cur_symbol,
                "Größe": cur_size,
            }
        )

    if hasattr(st.column_config, "ColorColumn"):
        color_col_config = st.column_config.ColorColumn("Farbe")
    else:
        color_col_config = st.column_config.TextColumn("Farbe (Hex)")

    edited_config = st.data_editor(
        pd.DataFrame(config_data),
        column_config={
            "ID": None,
            "Anzeigen": st.column_config.CheckboxColumn("Anzeigen", default=True),
            "Legende": "Legende",
            "Geometrie": st.column_config.TextColumn(
                "Geometrie", disabled=True
            ),  # Nicht editierbar
            "Farbe": color_col_config,
            "Linie": st.column_config.SelectboxColumn(
                "Linienstil", options=LINE_STYLES, required=True
            ),
            "Marker": st.column_config.SelectboxColumn(
                "Symbol", options=MARKER_SYMBOLS, required=True
            ),
            "Breite": st.column_config.NumberColumn(
                "Dicke", min_value=0.5, max_value=10.0, step=0.5
            ),
            "Größe": st.column_config.NumberColumn(
                "Sym.Größe", min_value=1, max_value=30, step=1
            ),
        },
        disabled=["ID", "Geometrie"],
        hide_index=True,
        key="design_editor",
        use_container_width=True,
    )

    # Mapping erstellen
    map_legend = dict(zip(edited_config["ID"], edited_config["Legende"]))
    map_color = dict(zip(edited_config["ID"], edited_config["Farbe"]))
    map_style = dict(zip(edited_config["ID"], edited_config["Linie"]))
    map_symbol = dict(zip(edited_config["ID"], edited_config["Marker"]))
    map_width = dict(zip(edited_config["ID"], edited_config["Breite"]))
    map_visible = dict(zip(edited_config["ID"], edited_config["Anzeigen"]))
    map_size = dict(zip(edited_config["ID"], edited_config["Größe"]))

    # Zuweisung an den DataFrame
    df_sub["final_legend"] = df_sub["unique_id"].map(map_legend)
    df_sub["final_color"] = df_sub["unique_id"].map(map_color)
    df_sub["final_style"] = df_sub["unique_id"].map(map_style)
    df_sub["final_symbol"] = df_sub["unique_id"].map(map_symbol)
    df_sub["final_width"] = df_sub["unique_id"].map(map_width)
    df_sub["final_visible"] = df_sub["unique_id"].map(map_visible)
    df_sub["final_size"] = df_sub["unique_id"].map(map_size)

with st.sidebar.expander("Diagramm-Titel bearbeiten", expanded=False):
    loaded_titles = st.session_state.get("loaded_titles", {})

    default_titles_data = [
        {
            "Typ": "Gesamtübersicht (Tab 1)",
            "Default": f"{current_title_str} | Fehlerkurve |  | Phasen-Vergleich",
        },
        {
            "Typ": "Scatter-Plot",
            "Default": f"{current_title_str} | Scatter-Plot | Kosten-Nutzen-Analyse",
        },
        {
            "Typ": "Performance-Index",
            "Default": f"{current_title_str} | Performance-Index | Ranking",
        },
        {
            "Typ": "Heatmap",
            "Default": f"{current_title_str} | Heatmap | Fehlerverteilung",
        },
        {"Typ": "Boxplot", "Default": f"{current_title_str} | Boxplot | Statistik"},
        {"Typ": "Pareto", "Default": f"{current_title_str} | Pareto | Fehler-Ursachen"},
        {"Typ": "Radar", "Default": f"{current_title_str} | Radar | Multi-Kriteriell"},
    ]

    merged_titles = []
    for item in default_titles_data:
        t_type = item["Typ"]
        val = loaded_titles.get(t_type, item["Default"])
        merged_titles.append({"Typ": t_type, "Titel": val})

    edited_titles_df = st.data_editor(
        pd.DataFrame(merged_titles),
        column_config={
            "Typ": st.column_config.TextColumn(disabled=True),
            "Titel": "Titel",
        },
        hide_index=True,
        key="titles_editor",
    )
    TITLES_MAP = dict(zip(edited_titles_df["Typ"], edited_titles_df["Titel"]))

if st.session_state.get("trigger_save", False):
    snapshot_data = {
        "current": sel_currents,
        "geos": sel_geos,
        "wandlers": sel_wandlers,
        "duts": sel_duts,
        "comp_mode": comp_mode_disp,
        "sync_axes": sync_axes,
        "y_limit": y_limit,
        "acc_class": acc_class,
        "show_err_bars": show_err_bars,
        "bottom_plot_mode": bottom_plot_mode,  # Save this!
        "custom_colors": map_color,
        "custom_legends": map_legend,
        "custom_styles": map_style,
        "custom_symbols": map_symbol,
        "custom_widths": map_width,
        "custom_visible": map_visible,  # NEU
        "custom_sizes": map_size,  # NEU
        "custom_titles": TITLES_MAP,
        "eco_x": st.session_state.get("k_eco_x", "Preis (€)"),
        "eco_y": st.session_state.get("k_eco_y", ["Fehler Nennstrom"]),
        "eco_type": st.session_state.get("k_eco_type", "Scatter"),
    }
    save_new_config(st.session_state["save_name"], snapshot_data)
    st.session_state["trigger_save"] = False
    st.toast(f"Konfiguration '{st.session_state['save_name']}' gespeichert!", icon="💾")

# --- EXPORT ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 PDF Export")
export_opts = [
    "Gesamtübersicht (Tab 1)",
    "Detail-Phasen (Tab 1)",
    "Ökonomie: Performance-Index",
    "Ökonomie: Scatter-Plot",
    "Ökonomie: Heatmap",
    "Ökonomie: Boxplot",
    "Ökonomie: Pareto",
    "Ökonomie: Radar",
]
export_selection = st.sidebar.multiselect(
    "Zu exportierende Diagramme:",
    export_opts,
    default=["Gesamtübersicht (Tab 1)", "Ökonomie: Performance-Index"],
)
engine_mode = st.sidebar.selectbox(
    "Render-Engine für Details:", ["Python (Direkt)", "MATLAB (Automatischer Start)"]
)
matlab_exe = (
    st.sidebar.text_input("Pfad MATLAB:", value=DEFAULT_MATLAB_PATH)
    if "MATLAB" in engine_mode
    else None
)

if st.sidebar.button("🔄 Export starten", type="primary"):
    if not export_selection:
        st.error("Bitte mindestens ein Diagramm auswählen.")
    else:
        # Ermittle Konfigurationsnamen für Dateinamen
        current_config_name = "Unbenannt"
        if sel_config_load and sel_config_load != "-- Neu / Leer --":
            current_config_name = sel_config_load

        safe_conf_name = sanitize_filename(current_config_name)

        zip_buffer = io.BytesIO()
        has_eco_request = any("Ökonomie" in s for s in export_selection)
        if has_eco_request:
            try:
                df_err_exp = (
                    df_sub.groupby("unique_id")
                    .agg(
                        wandler_key=("wandler_key", "first"),
                        legend_name=("final_legend", "first"),
                        err_nieder=(
                            "err_ratio",
                            lambda x: x[
                                df_sub.loc[x.index, "target_load"].isin(
                                    ZONES["Niederstrom (5-50%)"]
                                )
                            ]
                            .abs()
                            .mean(),
                        ),
                        err_nom=(
                            "err_ratio",
                            lambda x: x[
                                df_sub.loc[x.index, "target_load"].isin(
                                    ZONES["Nennstrom (80-100%)"]
                                )
                            ]
                            .abs()
                            .mean(),
                        ),
                        err_high=(
                            "err_ratio",
                            lambda x: x[
                                df_sub.loc[x.index, "target_load"].isin(
                                    ZONES["Überlast (≥120%)"]
                                )
                            ]
                            .abs()
                            .mean(),
                        ),
                        preis=("Preis (€)", "first"),
                        vol_t=("T (mm)", "first"),
                        vol_b=("B (mm)", "first"),
                        vol_h=("H (mm)", "first"),
                        color_hex=("final_color", "first"),
                    )
                    .reset_index()
                )
                df_err_exp["volumen"] = (
                    df_err_exp["vol_t"] * df_err_exp["vol_b"] * df_err_exp["vol_h"]
                ) / 1000.0
                mx_p = df_err_exp["preis"].max() or 1
                mx_v = df_err_exp["volumen"].max() or 1
                mx_en = df_err_exp["err_nieder"].abs().max() or 0.01
                mx_nn = df_err_exp["err_nom"].abs().max() or 0.01
                mx_eh = df_err_exp["err_high"].abs().max() or 0.01
                df_err_exp["Norm: Preis"] = (df_err_exp["preis"] / mx_p) * 100
                df_err_exp["Norm: Volumen"] = (df_err_exp["volumen"] / mx_v) * 100
                df_err_exp["Norm: Niederstrom"] = (
                    df_err_exp["err_nieder"].abs() / mx_en
                ) * 100
                df_err_exp["Norm: Nennstrom"] = (
                    df_err_exp["err_nom"].abs() / mx_nn
                ) * 100
                df_err_exp["Norm: Überstrom"] = (
                    df_err_exp["err_high"].abs() / mx_eh
                ) * 100
                df_err_exp["total_score"] = (
                    df_err_exp["Norm: Preis"]
                    + df_err_exp["Norm: Volumen"]
                    + df_err_exp["Norm: Nennstrom"]
                )
                color_map_dict = dict(
                    zip(df_err_exp["legend_name"], df_err_exp["color_hex"])
                )

                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                    if "Ökonomie: Performance-Index" in export_selection:
                        title_str = TITLES_MAP.get(
                            "Performance-Index", "Performance Index"
                        )
                        df_sorted = df_err_exp.sort_values(
                            "total_score", ascending=True
                        )
                        df_long = df_sorted.melt(
                            id_vars=["legend_name"],
                            value_vars=[
                                "Norm: Preis",
                                "Norm: Volumen",
                                "Norm: Nennstrom",
                            ],
                            var_name="Kategorie",
                            value_name="Anteil (%)",
                        )
                        fig_perf = px.bar(
                            df_long,
                            y="legend_name",
                            x="Anteil (%)",
                            color="Kategorie",
                            orientation="h",
                            title=title_str,
                            color_discrete_map={
                                "Norm: Preis": "#1f77b4",
                                "Norm: Volumen": "#aec7e8",
                                "Norm: Nennstrom": "#ff7f0e",
                            },
                        )
                        fig_perf.update_layout(
                            yaxis=dict(autorange="reversed"),
                            template="plotly_white",
                            width=1123,
                            height=794,
                            font=dict(family="Serif", size=14, color="black"),
                            legend=dict(
                                orientation="h",
                                y=-0.2,
                                x=0.5,
                                xanchor="center",
                                font=dict(size=16),
                            ),
                        )
                        zf.writestr(
                            f"{safe_conf_name}-Oekonomie_Performance_Index.pdf",
                            fig_perf.to_image(format="pdf"),
                        )
                    if "Ökonomie: Scatter-Plot" in export_selection:
                        title_str = TITLES_MAP.get(
                            "Scatter-Plot", "Kosten-Nutzen-Analyse"
                        )
                        fig_scat = px.scatter(
                            df_err_exp,
                            x="preis",
                            y="err_nom",
                            color="legend_name",
                            size=[20] * len(df_err_exp),
                            color_discrete_map=color_map_dict,
                            title=title_str,
                        )
                        fig_scat.update_layout(
                            template="plotly_white",
                            width=1123,
                            height=794,
                            font=dict(family="Serif", size=14, color="black"),
                            legend=dict(
                                orientation="h",
                                y=-0.2,
                                x=0.5,
                                xanchor="center",
                                font=dict(size=16),
                            ),
                        )
                        zf.writestr(
                            f"{safe_conf_name}-Oekonomie_Scatter.pdf",
                            fig_scat.to_image(format="pdf"),
                        )
                    if "Ökonomie: Heatmap" in export_selection:
                        title_str = TITLES_MAP.get("Heatmap", "Fehler-Heatmap")
                        df_hm = df_err_exp.melt(
                            id_vars=["legend_name"],
                            value_vars=["err_nieder", "err_nom", "err_high"],
                            var_name="Bereich",
                            value_name="Fehler",
                        )
                        fig_hm = px.density_heatmap(
                            df_hm,
                            x="legend_name",
                            y="Bereich",
                            z="Fehler",
                            color_continuous_scale="Blues",
                            title=title_str,
                        )
                        fig_hm.update_layout(
                            template="plotly_white",
                            width=1123,
                            height=794,
                            font=dict(family="Serif", size=14, color="black"),
                        )
                        zf.writestr(
                            f"{safe_conf_name}-Oekonomie_Heatmap.pdf",
                            fig_hm.to_image(format="pdf"),
                        )
                    if "Ökonomie: Boxplot" in export_selection:
                        title_str = TITLES_MAP.get(
                            "Boxplot", "Fehlerverteilung (Boxplot)"
                        )
                        df_box = df_err_exp.melt(
                            id_vars=["legend_name"],
                            value_vars=["err_nieder", "err_nom", "err_high"],
                            var_name="Bereich",
                            value_name="Fehler",
                        )
                        fig_box = px.box(
                            df_box,
                            x="legend_name",
                            y="Fehler",
                            color="Bereich",
                            title=title_str,
                        )
                        fig_box.update_layout(
                            template="plotly_white",
                            width=1123,
                            height=794,
                            font=dict(family="Serif", size=14, color="black"),
                            legend=dict(
                                orientation="h",
                                y=-0.2,
                                x=0.5,
                                xanchor="center",
                                font=dict(size=16),
                            ),
                        )
                        zf.writestr(
                            f"{safe_conf_name}-Oekonomie_Boxplot.pdf",
                            fig_box.to_image(format="pdf"),
                        )
                    if "Ökonomie: Pareto" in export_selection:
                        title_str = TITLES_MAP.get("Pareto", "Pareto-Analyse")
                        df_par = df_err_exp.sort_values(by="err_nom", ascending=False)
                        df_par["cum_pct"] = (
                            df_par["err_nom"].cumsum() / df_par["err_nom"].sum() * 100
                        )
                        fig_par = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_par.add_trace(
                            go.Bar(
                                x=df_par["legend_name"],
                                y=df_par["err_nom"],
                                name="Fehler (Nenn)",
                                marker_color=df_par["color_hex"],
                            ),
                            secondary_y=False,
                        )
                        fig_par.add_trace(
                            go.Scatter(
                                x=df_par["legend_name"],
                                y=df_par["cum_pct"],
                                name="Kumulativ %",
                                mode="lines+markers",
                                line=dict(color="red", width=3),
                            ),
                            secondary_y=True,
                        )
                        fig_par.update_layout(
                            title=title_str,
                            template="plotly_white",
                            width=1123,
                            height=794,
                            font=dict(family="Serif", size=14, color="black"),
                            legend=dict(
                                orientation="h",
                                y=-0.2,
                                x=0.5,
                                xanchor="center",
                                font=dict(size=16),
                            ),
                        )
                        zf.writestr(
                            f"{safe_conf_name}-Oekonomie_Pareto.pdf",
                            fig_par.to_image(format="pdf"),
                        )
                    if "Ökonomie: Radar" in export_selection:
                        title_str = TITLES_MAP.get("Radar", "Radar-Profil")
                        fig_rad = go.Figure()
                        cats = [
                            "Preis",
                            "Volumen",
                            "Err Nieder",
                            "Err Nenn",
                            "Err High",
                        ]
                        for i, row in df_err_exp.iterrows():
                            vals = [
                                row["Norm: Preis"] / 100,
                                row["Norm: Volumen"] / 100,
                                row["Norm: Niederstrom"] / 100,
                                row["Norm: Nennstrom"] / 100,
                                row["Norm: Überstrom"] / 100,
                                row["Norm: Preis"] / 100,
                            ]
                            fig_rad.add_trace(
                                go.Scatterpolar(
                                    r=vals,
                                    theta=cats + [cats[0]],
                                    fill="toself",
                                    name=row["legend_name"],
                                    line_color=row["color_hex"],
                                )
                            )
                        fig_rad.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            title=title_str,
                            template="plotly_white",
                            width=1123,
                            height=794,
                            font=dict(family="Serif", size=14, color="black"),
                            legend=dict(
                                orientation="h",
                                y=-0.2,
                                x=0.5,
                                xanchor="center",
                                font=dict(size=16),
                            ),
                        )
                        zf.writestr(
                            f"{safe_conf_name}-Oekonomie_Radar.pdf",
                            fig_rad.to_image(format="pdf"),
                        )
            except Exception as e:
                st.error(f"Fehler bei Ökonomie-Export: {e}")

        # GESAMT EXPORT
        if "Gesamtübersicht (Tab 1)" in export_selection:
            with st.spinner("Generiere Gesamtübersicht..."):
                main_title_export = TITLES_MAP.get(
                    "Gesamtübersicht (Tab 1)", f"Gesamtübersicht: {current_title_str}"
                )

                # Layout-Konfiguration basierend auf Modus
                if use_single_row:
                    # MODUS: NUR FEHLERKURVE (Volle Höhe)
                    fig_ex = make_subplots(
                        rows=1,
                        cols=3,
                        shared_xaxes=True,
                        subplot_titles=PHASES,
                        horizontal_spacing=0.05,
                        # WICHTIG: Keine row_heights, damit Plotly den Platz füllt
                    )
                else:
                    # MODUS: MIT UNTERDIAGRAMM (Geteilte Höhe)
                    fig_ex = make_subplots(
                        rows=2,
                        cols=3,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        horizontal_spacing=0.05,
                        row_heights=[0.7, 0.3],  # 70% oben, 30% unten
                        subplot_titles=PHASES,
                    )

                lim_x, lim_y_p, lim_y_n = get_trumpet_limits(acc_class)
                for c_idx, ph in enumerate(PHASES, 1):
                    fig_ex.add_trace(
                        go.Scatter(
                            x=lim_x,
                            y=lim_y_p,
                            mode="lines",
                            line=dict(color="black", width=1.5, dash="dash"),
                            showlegend=(c_idx == 1),
                            name="Klassengrenze",
                        ),
                        row=1,
                        col=c_idx,
                    )
                    fig_ex.add_trace(
                        go.Scatter(
                            x=lim_x,
                            y=lim_y_n,
                            mode="lines",
                            line=dict(color="black", width=1.5, dash="dash"),
                            showlegend=False,
                        ),
                        row=1,
                        col=c_idx,
                    )
                    phase_data = df_sub[df_sub["phase"] == ph]
                    for uid, group in phase_data.groupby("unique_id"):
                        group = group.sort_values("target_load")
                        row_first = group.iloc[0]

                        # NEU: Check Visibility
                        if not row_first["final_visible"]:
                            continue

                        # NEW: Access Style & Symbol & Width
                        style = row_first["final_style"]
                        symbol = row_first["final_symbol"]
                        width = row_first["final_width"]
                        size_val = row_first["final_size"]

                        fig_ex.add_trace(
                            go.Scatter(
                                x=group["target_load"],
                                y=group["err_ratio"],
                                mode="lines+markers",
                                name=row_first["final_legend"],
                                line=dict(
                                    color=row_first["final_color"],
                                    width=width,
                                    dash=style,
                                ),
                                marker=dict(size=size_val, symbol=symbol),
                                legendgroup=row_first["final_legend"],
                                showlegend=(c_idx == 1),
                            ),
                            row=1,
                            col=c_idx,
                        )

                        if not use_single_row:
                            if bottom_plot_mode == "Standardabweichung":
                                if show_err_bars:
                                    fig_ex.add_trace(
                                        go.Bar(
                                            x=group["target_load"],
                                            y=group["err_std"],
                                            marker_color=row_first["final_color"],
                                            legendgroup=row_first["final_legend"],
                                            showlegend=False,
                                        ),
                                        row=2,
                                        col=c_idx,
                                    )
                            elif bottom_plot_mode == "Messwert (Absolut)":
                                fig_ex.add_trace(
                                    go.Scatter(
                                        x=group["target_load"],
                                        y=group["val_dut_mean"],
                                        mode="lines+markers",
                                        line=dict(
                                            color=row_first["final_color"],
                                            width=1.5,
                                            dash="dot",
                                        ),
                                        marker=dict(symbol="x", size=size_val),
                                        legendgroup=row_first["final_legend"],
                                        showlegend=False,
                                    ),
                                    row=2,
                                    col=c_idx,
                                )

                fig_ex.update_layout(
                    title=dict(text=main_title_export, font=dict(size=18)),
                    template="plotly_white",
                    width=1123,
                    height=794,
                    font=dict(family="Serif", size=14, color="black"),
                    legend=dict(
                        orientation="h",
                        y=-0.15,
                        x=0.5,
                        xanchor="center",
                        font=dict(size=16),
                    ),
                )
                fig_ex.update_yaxes(range=[-y_limit, y_limit], row=1)

                if not use_single_row:
                    if bottom_plot_mode == "Standardabweichung":
                        fig_ex.update_yaxes(title_text="StdAbw [%]", row=2, col=1)
                    else:
                        fig_ex.update_yaxes(title_text="Strom [A]", row=2, col=1)

                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr(
                        f"{safe_conf_name}-Zusammenfassung_MultiCurrent.pdf",
                        fig_ex.to_image(format="pdf", width=1123, height=794),
                    )

        # DETAIL EXPORT
        if "Detail-Phasen (Tab 1)" in export_selection:
            if "MATLAB" in engine_mode:
                if not os.path.exists(matlab_exe):
                    st.error("MATLAB nicht gefunden.")
                else:
                    with st.spinner("MATLAB rendert Details..."):
                        work_dir_abs = ensure_working_dir()
                        export_df = df_sub.copy().rename(
                            columns={
                                "final_color": "color_hex",
                                "final_legend": "legend_name",
                            }
                        )
                        export_df["trace_id"] = export_df["legend_name"]
                        export_df.drop_duplicates(
                            subset=["trace_id", "target_load", "phase"], keep="last"
                        ).to_csv(
                            os.path.join(work_dir_abs, "plot_data.csv"), index=False
                        )
                        with open(
                            os.path.join(work_dir_abs, "create_plots.m"), "w"
                        ) as f:
                            nom_curr_val = int(sel_currents[0]) if sel_currents else 0
                            f.write(
                                MATLAB_SCRIPT_TEMPLATE.replace(
                                    "ACC_CLASS_PH", str(acc_class)
                                )
                                .replace("Y_LIMIT_PH", str(y_limit))
                                .replace("NOMINAL_CURRENT_PH", str(nom_curr_val))
                            )
                        try:
                            subprocess.run(
                                [matlab_exe, "-batch", "create_plots"],
                                cwd=work_dir_abs,
                                check=True,
                            )
                            # Dateien mit Prefix in die Zip packen
                            [
                                zipfile.ZipFile(
                                    zip_buffer, "a", zipfile.ZIP_DEFLATED
                                ).write(
                                    os.path.join(work_dir_abs, f),
                                    f"{safe_conf_name}-{f}",
                                )
                                for f in os.listdir(work_dir_abs)
                                if f.endswith(".pdf")
                            ]
                            st.success("✅ Details exportiert")
                        except Exception as e:
                            st.error(f"MATLAB Fehler: {e}")
            else:
                with st.spinner("Python generiert Details..."):
                    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                        for ph in PHASES:
                            fig_s = create_single_phase_figure(
                                df_sub,
                                ph,
                                acc_class,
                                y_limit,
                                bottom_plot_mode,
                                show_err_bars,
                                title_prefix=f"{current_title_str}",
                                nticks_x=nticks_x,  # <--- HIER EINFÜGEN
                                nticks_y=nticks_y,  # <--- HIER EINFÜGEN
                            )
                            zf.writestr(
                                f"{safe_conf_name}-Detail_{ph}_MultiCurrent.pdf",
                                fig_s.to_image(format="pdf", width=1123, height=794),
                            )
                    st.success("✅ Details exportiert")

        st.session_state["zip_data"] = zip_buffer.getvalue()
        st.session_state["zip_name"] = f"{safe_conf_name}.zip"

if "zip_data" in st.session_state:
    st.sidebar.download_button(
        "💾 Download ZIP",
        st.session_state["zip_data"],
        st.session_state["zip_name"],
        "application/zip",
    )

# =============================================================================
# MAIN TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Gesamtgenauigkeit",
        "💰 Ökonomische Analyse",
        "⚙️ Stammdaten-Editor",
        "✂️ Rohdaten-Selektor",
    ]
)

with tab1:
    custom_title_tab1 = TITLES_MAP.get(
        "Gesamtübersicht (Tab 1)", f"Gesamtübersicht: {current_title_str}"
    )

    # Layout-Konfiguration basierend auf Modus
    if use_single_row:
        # MODUS: NUR FEHLERKURVE (Volle Höhe)
        fig_main = make_subplots(
            rows=1,
            cols=3,
            shared_xaxes=True,
            subplot_titles=PHASES,
            horizontal_spacing=0.05,
            # WICHTIG: Keine row_heights, damit Plotly den Platz füllt
        )
    else:
        # MODUS: MIT UNTERDIAGRAMM (Geteilte Höhe)
        fig_main = make_subplots(
            rows=2,
            cols=3,
            shared_xaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            row_heights=[0.7, 0.3],  # 70% oben, 30% unten
            subplot_titles=PHASES,
        )

    lim_x, lim_y_p, lim_y_n = get_trumpet_limits(acc_class)
    for col_idx, phase in enumerate(PHASES, start=1):
        # --- Klassengrenzen ---
        fig_main.add_trace(
            go.Scatter(
                x=lim_x,
                y=lim_y_p,
                mode="lines",
                line=dict(color="black", width=1, dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )
        fig_main.add_trace(
            go.Scatter(
                x=lim_x,
                y=lim_y_n,
                mode="lines",
                line=dict(color="black", width=1, dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )

        # --- Daten-Kurven ---
        phase_data = df_sub[df_sub["phase"] == phase]
        for uid, group in phase_data.groupby("unique_id"):
            group = group.sort_values("target_load")
            row_first = group.iloc[0]

            # NEU: Check Visibility
            if not row_first["final_visible"]:
                continue

            # NEW: Access Style & Symbol & Width
            style = row_first["final_style"]
            symbol = row_first["final_symbol"]
            width = row_first["final_width"]
            size_val = row_first["final_size"]

            # HAUPT-PLOT (Immer Zeile 1)
            fig_main.add_trace(
                go.Scatter(
                    x=group["target_load"],
                    y=group["err_ratio"],
                    mode="lines+markers",
                    name=row_first["final_legend"],
                    line=dict(color=row_first["final_color"], width=width, dash=style),
                    marker=dict(size=size_val, symbol=symbol),
                    legendgroup=row_first["final_legend"],
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )

            # UNTERES DIAGRAMM (Nur wenn nicht 'Ausblenden')
            if not use_single_row:
                if bottom_plot_mode == "Standardabweichung":
                    if show_err_bars:
                        fig_main.add_trace(
                            go.Bar(
                                x=group["target_load"],
                                y=group["err_std"],
                                marker_color=row_first["final_color"],
                                legendgroup=row_first["final_legend"],
                                showlegend=False,
                            ),
                            row=2,
                            col=col_idx,
                        )
                elif bottom_plot_mode == "Messwert (Absolut)":
                    fig_main.add_trace(
                        go.Scatter(
                            x=group["target_load"],
                            y=group["val_dut_mean"],
                            mode="lines+markers",
                            line=dict(
                                color=row_first["final_color"], width=1.5, dash="dot"
                            ),
                            marker=dict(symbol="x", size=size_val),
                            legendgroup=row_first["final_legend"],
                            showlegend=False,
                        ),
                        row=2,
                        col=col_idx,
                    )

    fig_main.update_layout(
        title=custom_title_tab1,
        template="plotly_white",
        height=800,  # Fixe Gesamthöhe - bei Single Row wird diese voll genutzt
        legend=dict(
            orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=16)
        ),
        font=dict(family="Serif", size=14, color="black"),
    )

    # Y-Achsen Sync für Zeile 1
    if sync_axes:
        fig_main.update_yaxes(matches="y", row=1)

    fig_main.update_yaxes(
        range=[-y_limit, y_limit], title_text="Fehler [%]", row=1, col=1
    )

    # Achsenbeschriftungen anpassen je nach Modus
    if not use_single_row:
        if bottom_plot_mode == "Standardabweichung":
            fig_main.update_yaxes(title_text="StdAbw [%]", row=2, col=1)
        else:
            fig_main.update_yaxes(title_text="Strom [A]", row=2, col=1)
        fig_main.update_xaxes(title_text="Last [% In]", row=2, col=2)
    else:
        # Wenn nur 1 Zeile, muss das Label an die x-Achse der 1. Zeile
        fig_main.update_xaxes(title_text="Last [% In]", row=1, col=2)

    # --- NEU EINFÜGEN START ---
    # Hier wenden wir die Slider-Werte auf das Haupt-Dashboard an
    fig_main.update_xaxes(nticks=nticks_x)
    fig_main.update_yaxes(nticks=nticks_y)
    # --- NEU EINFÜGEN ENDE ---

    st.plotly_chart(fig_main, use_container_width=True)

with tab2:
    st.markdown("### 💰 Preis/Leistung & Varianten-Vergleich")
    df_err = (
        df_sub.groupby("unique_id")
        .agg(
            wandler_key=("wandler_key", "first"),
            legend_name=("final_legend", "first"),
            err_nieder=(
                "err_ratio",
                lambda x: x[
                    df_sub.loc[x.index, "target_load"].isin(
                        ZONES["Niederstrom (5-50%)"]
                    )
                ]
                .abs()
                .mean(),
            ),
            err_nom=(
                "err_ratio",
                lambda x: x[
                    df_sub.loc[x.index, "target_load"].isin(
                        ZONES["Nennstrom (80-100%)"]
                    )
                ]
                .abs()
                .mean(),
            ),
            err_high=(
                "err_ratio",
                lambda x: x[
                    df_sub.loc[x.index, "target_load"].isin(ZONES["Überlast (≥120%)"])
                ]
                .abs()
                .mean(),
            ),
            preis=("Preis (€)", "first"),
            vol_t=("T (mm)", "first"),
            vol_b=("B (mm)", "first"),
            vol_h=("H (mm)", "first"),
            color_hex=("final_color", "first"),
        )
        .reset_index()
    )
    df_err["volumen"] = (df_err["vol_t"] * df_err["vol_b"] * df_err["vol_h"]) / 1000.0

    Y_OPTIONS_MAP = {
        "Fehler Niederstrom": "err_nieder",
        "Fehler Nennstrom": "err_nom",
        "Fehler Überlast": "err_high",
        "Preis (€)": "preis",
        "Volumen (Gesamt)": "volumen",
        "Breite (B)": "vol_b",
        "Höhe (H)": "vol_h",
        "Tiefe (T)": "vol_t",
    }
    REVERSE_Y_MAP = {v: k for k, v in Y_OPTIONS_MAP.items()}

    if not df_err.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            x_sel = st.selectbox(
                "X-Achse:",
                ["Preis (€)", "Volumen (dm³)"],
                index=0 if st.session_state.get("k_eco_x") == "Preis (€)" else 1,
                key="k_eco_x",
            )
            x_col = "preis" if "Preis" in x_sel else "volumen"
        with c2:
            y_selection = st.multiselect(
                "Y-Achse:",
                options=list(Y_OPTIONS_MAP.keys()),
                default=st.session_state.get("k_eco_y", ["Fehler Nennstrom"]),
                key="k_eco_y",
            )
            y_cols_selected = [Y_OPTIONS_MAP[label] for label in y_selection]
        with c3:
            types = [
                "Scatter",
                "Performance-Index",
                "Heatmap",
                "Boxplot",
                "Pareto",
                "Radar",
            ]
            try:
                t_idx = types.index(st.session_state.get("k_eco_type", "Scatter"))
            except:
                t_idx = 0
            chart_type = st.radio("Diagramm-Typ:", types, index=t_idx, key="k_eco_type")

        color_map_dict = dict(zip(df_err["legend_name"], df_err["color_hex"]))

        if not y_cols_selected:
            st.warning("Bitte wähle mindestens einen Wert für die Y-Achse aus.")
        else:
            if chart_type == "Scatter":
                title_str = TITLES_MAP.get("Scatter-Plot", f"{x_sel} vs. Auswahl")
                df_long = df_err.melt(
                    id_vars=["unique_id", "legend_name", x_col, "color_hex"],
                    value_vars=y_cols_selected,
                    var_name="Metrik_Intern",
                    value_name="Wert",
                )
                df_long["Metrik"] = df_long["Metrik_Intern"].map(REVERSE_Y_MAP)
                fig_eco = px.scatter(
                    df_long,
                    x=x_col,
                    y="Wert",
                    color="legend_name",
                    symbol="Metrik",
                    size=[15] * len(df_long),
                    color_discrete_map=color_map_dict,
                    title=title_str,
                )
                fig_eco.update_layout(
                    legend=dict(
                        orientation="h",
                        y=-0.2,
                        x=0.5,
                        xanchor="center",
                        font=dict(size=16),
                    )
                )
                st.plotly_chart(fig_eco, use_container_width=True)
            elif chart_type == "Performance-Index":
                title_str = TITLES_MAP.get("Performance-Index", "Performance Index")
                norm_cols = []
                df_err["total_score"] = 0.0
                for label in y_selection:
                    raw_col = Y_OPTIONS_MAP[label]
                    mx_val = df_err[raw_col].abs().max()
                    if mx_val == 0:
                        mx_val = 1.0
                    df_err[label] = (df_err[raw_col].abs() / mx_val) * 100
                    df_err["total_score"] += df_err[label]
                    norm_cols.append(label)
                df_sorted = df_err.sort_values("total_score", ascending=True)
                df_long = df_sorted.melt(
                    id_vars=["legend_name"],
                    value_vars=norm_cols,
                    var_name="Kategorie",
                    value_name="Normalisierter Anteil (%)",
                )
                fig_eco = px.bar(
                    df_long,
                    y="legend_name",
                    x="Normalisierter Anteil (%)",
                    color="Kategorie",
                    orientation="h",
                    title=title_str,
                )
                fig_eco.update_layout(
                    yaxis=dict(autorange="reversed"),
                    legend=dict(
                        orientation="h",
                        y=-0.2,
                        x=0.5,
                        xanchor="center",
                        font=dict(size=16),
                    ),
                )
                st.plotly_chart(fig_eco, use_container_width=True)
            elif chart_type == "Heatmap":
                title_str = TITLES_MAP.get("Heatmap", "Heatmap der Auswahl")
                df_long = df_err.melt(
                    id_vars=["legend_name"],
                    value_vars=y_cols_selected,
                    var_name="Kategorie_Intern",
                    value_name="Wert",
                )
                df_long["Kategorie"] = df_long["Kategorie_Intern"].map(REVERSE_Y_MAP)
                fig_eco = px.density_heatmap(
                    df_long,
                    x="legend_name",
                    y="Kategorie",
                    z="Wert",
                    color_continuous_scale="Blues",
                    title=title_str,
                )
                st.plotly_chart(fig_eco, use_container_width=True)
            elif chart_type == "Boxplot":
                title_str = TITLES_MAP.get(
                    "Boxplot", f"Verteilung: {', '.join(y_selection)}"
                )
                df_long = df_err.melt(
                    id_vars=["legend_name"],
                    value_vars=y_cols_selected,
                    var_name="Kategorie_Intern",
                    value_name="Wert",
                )
                df_long["Kategorie"] = df_long["Kategorie_Intern"].map(REVERSE_Y_MAP)
                fig_eco = px.box(
                    df_long,
                    x="legend_name",
                    y="Wert",
                    color="Kategorie",
                    title=title_str,
                )
                fig_eco.update_layout(
                    legend=dict(
                        orientation="h",
                        y=-0.2,
                        x=0.5,
                        xanchor="center",
                        font=dict(size=16),
                    )
                )
                st.plotly_chart(fig_eco, use_container_width=True)
            elif chart_type == "Pareto":
                title_str = TITLES_MAP.get("Pareto", "Pareto-Analyse")
                target_y = y_cols_selected[0]
                target_label = y_selection[0]
                df_sorted = df_err.sort_values(by=target_y, ascending=False)
                df_sorted["cum_pct"] = (
                    df_sorted[target_y].cumsum() / df_sorted[target_y].sum() * 100
                )
                fig_par = make_subplots(specs=[[{"secondary_y": True}]])
                fig_par.add_trace(
                    go.Bar(
                        x=df_sorted["legend_name"],
                        y=df_sorted[target_y],
                        name=target_label,
                        marker_color=df_sorted["color_hex"],
                    ),
                    secondary_y=False,
                )
                fig_par.add_trace(
                    go.Scatter(
                        x=df_sorted["legend_name"],
                        y=df_sorted["cum_pct"],
                        name="Kumulativ %",
                        mode="lines+markers",
                        line=dict(color="red"),
                    ),
                    secondary_y=True,
                )
                fig_par.update_layout(
                    title=title_str,
                    legend=dict(
                        orientation="h",
                        y=-0.2,
                        x=0.5,
                        xanchor="center",
                        font=dict(size=16),
                    ),
                )
                st.plotly_chart(fig_par, use_container_width=True)
            elif chart_type == "Radar":
                title_str = TITLES_MAP.get("Radar", "Radar-Vergleich")
                fig_r = go.Figure()
                categories = y_selection
                max_vals = {}
                for col_name in y_cols_selected:
                    m = df_err[col_name].max()
                    max_vals[col_name] = m if m != 0 else 1
                for i, row in df_err.iterrows():
                    r_vals = []
                    for col_name in y_cols_selected:
                        val = row[col_name]
                        norm_val = val / max_vals[col_name]
                        r_vals.append(norm_val)
                    r_vals.append(r_vals[0])
                    theta_vals = categories + [categories[0]]
                    fig_r.add_trace(
                        go.Scatterpolar(
                            r=r_vals,
                            theta=theta_vals,
                            fill="toself",
                            name=row["legend_name"],
                            line_color=row["color_hex"],
                        )
                    )
                fig_r.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title=title_str,
                    legend=dict(
                        orientation="h",
                        y=-0.2,
                        x=0.5,
                        xanchor="center",
                        font=dict(size=16),
                    ),
                )
                st.plotly_chart(fig_r, use_container_width=True)

with tab3:
    st.markdown("### ⚙️ Stammdaten pro Messdatei")

    col_info, col_btn = st.columns([2, 1])
    with col_info:
        st.info("Die Datenbank ist die Hauptquelle.")
    with col_btn:
        if st.button(
            "🔄 Infos aus Dateinamen neu einlesen",
            type="secondary",
            use_container_width=True,
        ):
            src_col = "raw_file" if "raw_file" in df.columns else "wandler_key"
            new_meta = df[src_col].apply(parse_filename_info)
            new_meta.columns = ["Hersteller", "Modell", "nennstrom", "Mess-Bürde"]
            df["Hersteller"] = new_meta["Hersteller"]
            df["Modell"] = new_meta["Modell"]
            df["nennstrom"] = new_meta["nennstrom"]
            df["Mess-Bürde"] = new_meta["Mess-Bürde"]
            if save_db(df):
                st.success("Aktualisiert!")
                st.rerun()

    # --- CHECKBOX FÜR ALLE DATEIEN ---
    show_all_files = st.checkbox(
        "Alle Dateien der gewählten Nennströme anzeigen (ignoriert Filter für Geometrie/Wandler/DUT)",
        value=True,
    )

    if show_all_files:
        df_editor_source = df[df["nennstrom"].isin(sel_currents)].copy()
    else:
        df_editor_source = df_sub.copy()

    df_editor_view = (
        df_editor_source[META_COLS]
        .drop_duplicates(subset=["raw_file"])
        .set_index("raw_file")
    )

    edited_df = st.data_editor(
        df_editor_view,
        column_config={"raw_file": st.column_config.TextColumn(disabled=True)},
        hide_index=True,
        key="specs_editor",
    )

    if st.button("💾 Änderungen speichern", type="primary"):
        changes = edited_df.to_dict(orient="index")
        df_to_save = df.copy()
        count = 0

        for fname, attrs in changes.items():
            mask = df_to_save["raw_file"] == str(fname).strip()
            if mask.any():
                count += 1
                for c in META_COLS_EDIT:
                    if c in attrs:
                        df_to_save.loc[mask, c] = attrs[c]

        if count > 0:
            save_db(df_to_save)
            st.success(f"✅ {count} Dateien gespeichert!")
            st.rerun()
        else:
            st.warning("Keine Übereinstimmung gefunden.")

        if sel_file_path:
            clean_name_base, detected_nennstrom = extract_metadata(sel_file_path)

            # Auto-Load Config Logic
            saved_times = all_configs.get(clean_name_base, {})
            ranges_empty = all(
                st.session_state[f"s_{l}"] == 0 for l in TARGET_LEVELS_T4
            )

            if ranges_empty and saved_times:
                for lvl_str, vals in saved_times.items():
                    if lvl_str.isdigit():
                        lvl = int(lvl_str)
                        if (
                            lvl in TARGET_LEVELS_T4
                            and isinstance(vals, list)
                            and len(vals) == 2
                        ):
                            st.session_state[f"s_{lvl}"] = vals[0]
                            st.session_state[f"e_{lvl}"] = vals[1]

            detected = load_file_preview_tab4(sel_file_path)

            c_inf1, c_inf2 = st.columns([3, 1])
            with c_inf1:
                st.subheader(clean_name_base)
            with c_inf2:
                # Hier ist der Key, der den Fehler verursachte. Jetzt gibt es ihn nur noch 1x.
                nennstrom_t4 = st.number_input(
                    "Nennstrom:", value=float(detected_nennstrom), key="ns_t4"
                )

            t_idx, full_data_t4 = load_all_data_tab4(sel_file_path, detected)

            if t_idx:
                c_tool1, c_tool2 = st.columns([2, 1])
                with c_tool1:
                    sel_ref_dev = st.selectbox(
                        "Referenz-Gerät (L1):", detected, key="ref_dev_t4"
                    )
                with c_tool2:
                    if st.button("🔄 Config neu laden"):
                        for lvl_str, vals in saved_times.items():
                            if lvl_str.isdigit():
                                lvl = int(lvl_str)
                                if (
                                    lvl in TARGET_LEVELS_T4
                                    and isinstance(vals, list)
                                    and len(vals) == 2
                                ):
                                    st.session_state[f"s_{lvl}"] = vals[0]
                                    st.session_state[f"e_{lvl}"] = vals[1]
                        st.rerun()

                ref_l1 = full_data_t4[sel_ref_dev]["L1"]
                if ref_l1 is not None:
                    fig_sel = go.Figure()
                    pct_val = (ref_l1 / nennstrom_t4) * 100
                    fig_sel.add_trace(
                        go.Scatter(
                            x=t_idx,
                            y=pct_val,
                            name=f"{sel_ref_dev} L1 %",
                            line=dict(color="orange"),
                        )
                    )

                    for level in TARGET_LEVELS_T4:
                        fig_sel.add_hline(
                            y=level, line_dash="dot", line_color="gray", opacity=0.5
                        )
                        s = st.session_state[f"s_{level}"]
                        e = st.session_state[f"e_{level}"]
                        if s > 0 and e > s:
                            fig_sel.add_vrect(
                                x0=s,
                                x1=e,
                                fillcolor="rgba(0,200,100,0.2)",
                                line_width=0,
                            )
                            fig_sel.add_annotation(
                                x=(s + e) / 2,
                                y=level + 5,
                                text=f"<b>{level}%</b>",
                                showarrow=False,
                                font=dict(color="green"),
                            )

                    fig_sel.update_layout(
                        height=400,
                        margin=dict(t=30, b=0, l=0, r=0),
                        yaxis_title="Last %",
                    )
                    st.plotly_chart(fig_sel, use_container_width=True)

                st.markdown("##### Bereiche definieren")
                batches = [
                    TARGET_LEVELS_T4[i : i + 4]
                    for i in range(0, len(TARGET_LEVELS_T4), 4)
                ]
                for batch in batches:
                    c_h = st.columns(4)
                    for i, l in enumerate(batch):
                        c_h[i].markdown(f"**{l}%**")
                    c_e = st.columns(4)
                    for i, l in enumerate(batch):
                        c_e[i].number_input(
                            "Ende",
                            key=f"e_{l}",
                            on_change=update_start_callback,
                            args=(l,),
                            label_visibility="collapsed",
                        )
                    c_s = st.columns(4)
                    for i, l in enumerate(batch):
                        c_s[i].number_input(
                            "Start", key=f"s_{l}", label_visibility="collapsed"
                        )

                st.divider()

                c_exp1, c_exp2, c_exp3 = st.columns([2, 1, 1])
                with c_exp1:
                    exp_devs = st.multiselect(
                        "Export Geräte:", detected, default=detected
                    )
                with c_exp2:
                    skip_n = st.number_input("Einschwingen Skip:", value=0)
                with c_exp3:
                    stat_opt = st.radio("Status:", ["OK", "Problem"], horizontal=True)

                if st.button(
                    "🚀 Speichern & Exportieren",
                    type="primary",
                    use_container_width=True,
                ):
                    if not exp_devs:
                        st.error("Keine Geräte gewählt.")
                    else:
                        semap = {}
                        for l in TARGET_LEVELS_T4:
                            semap[l] = (
                                st.session_state[f"s_{l}"],
                                st.session_state[f"e_{l}"],
                            )

                        out = save_sorted_raw_data_tab4(
                            sel_file_path,
                            full_data_t4,
                            semap,
                            exp_devs,
                            sel_ref_dev,
                            clean_name_base,
                            skip_n,
                        )
                        if out:
                            s_code = "WARNING" if stat_opt == "Problem" else "OK"
                            save_tracking_status(clean_name_base, s_code)
                            st.success(f"Gespeichert: {out}")
        if sel_file_path:
            # Metadaten (Name und Nennstrom) holen
            clean_name_base, detected_nennstrom = extract_metadata(sel_file_path)

            # --- LOGIK START: ZEITEN AUS CONFIG LADEN ---
            # 1. Config Eintrag suchen
            saved_times = all_configs.get(clean_name_base, {})

            # 2. Prüfen, ob die aktuellen Felder leer sind (alles 0)
            ranges_empty = all(
                st.session_state[f"s_{l}"] == 0 for l in TARGET_LEVELS_T4
            )

            # 3. Wenn leer und Config vorhanden -> Laden
            if ranges_empty and saved_times:
                for lvl_str, vals in saved_times.items():
                    # JSON Keys sind Strings ("5"), wir brauchen Int
                    if lvl_str.isdigit():
                        lvl = int(lvl_str)
                        # Prüfen ob Level relevant ist und Datenformat [Start, Ende] passt
                        if (
                            lvl in TARGET_LEVELS_T4
                            and isinstance(vals, list)
                            and len(vals) == 2
                        ):
                            st.session_state[f"s_{lvl}"] = vals[0]
                            st.session_state[f"e_{lvl}"] = vals[1]
            # --- LOGIK ENDE ---

            detected = load_file_preview_tab4(sel_file_path)

            # Info Header
            c_inf1, c_inf2 = st.columns([3, 1])
            with c_inf1:
                st.subheader(clean_name_base)
            with c_inf2:
                # Nennstrom anzeigen/ändern
                nennstrom_t4 = st.number_input(
                    "Nennstrom:", value=float(detected_nennstrom), key="ns_t4"
                )

            # Daten laden
            t_idx, full_data_t4 = load_all_data_tab4(sel_file_path, detected)

            if t_idx:
                # Toolbar
                c_tool1, c_tool2 = st.columns([2, 1])
                with c_tool1:
                    sel_ref_dev = st.selectbox(
                        "Referenz-Gerät (L1):", detected, key="ref_dev_t4"
                    )
                with c_tool2:
                    # Manueller Reload Button, falls man sich verklickt hat
                    if st.button("🔄 Config neu laden"):
                        for lvl_str, vals in saved_times.items():
                            if lvl_str.isdigit():
                                lvl = int(lvl_str)
                                if (
                                    lvl in TARGET_LEVELS_T4
                                    and isinstance(vals, list)
                                    and len(vals) == 2
                                ):
                                    st.session_state[f"s_{lvl}"] = vals[0]
                                    st.session_state[f"e_{lvl}"] = vals[1]
                        st.rerun()

                # PLOT
                ref_l1 = full_data_t4[sel_ref_dev]["L1"]
                if ref_l1 is not None:
                    fig_sel = go.Figure()
                    pct_val = (ref_l1 / nennstrom_t4) * 100
                    fig_sel.add_trace(
                        go.Scatter(
                            x=t_idx,
                            y=pct_val,
                            name=f"{sel_ref_dev} L1 %",
                            line=dict(color="orange"),
                        )
                    )

                    # Areas (Schattierung der gewählten Bereiche)
                    for level in TARGET_LEVELS_T4:
                        fig_sel.add_hline(
                            y=level, line_dash="dot", line_color="gray", opacity=0.5
                        )
                        s = st.session_state[f"s_{level}"]
                        e = st.session_state[f"e_{level}"]
                        if s > 0 and e > s:
                            fig_sel.add_vrect(
                                x0=s,
                                x1=e,
                                fillcolor="rgba(0,200,100,0.2)",
                                line_width=0,
                            )
                            fig_sel.add_annotation(
                                x=(s + e) / 2,
                                y=level + 5,
                                text=f"<b>{level}%</b>",
                                showarrow=False,
                                font=dict(color="green"),
                            )

                    fig_sel.update_layout(
                        height=400,
                        margin=dict(t=30, b=0, l=0, r=0),
                        yaxis_title="Last %",
                    )
                    st.plotly_chart(fig_sel, use_container_width=True)

                # INPUTS (Start/Ende Eingabefelder)
                st.markdown("##### Bereiche definieren")

                batches = [
                    TARGET_LEVELS_T4[i : i + 4]
                    for i in range(0, len(TARGET_LEVELS_T4), 4)
                ]
                for batch in batches:
                    c_h = st.columns(4)
                    for i, l in enumerate(batch):
                        c_h[i].markdown(f"**{l}%**")
                    c_e = st.columns(4)
                    for i, l in enumerate(batch):
                        c_e[i].number_input(
                            "Ende",
                            key=f"e_{l}",
                            on_change=update_start_callback,
                            args=(l,),
                            label_visibility="collapsed",
                        )
                    c_s = st.columns(4)
                    for i, l in enumerate(batch):
                        c_s[i].number_input(
                            "Start", key=f"s_{l}", label_visibility="collapsed"
                        )

                st.divider()

                # EXPORT BEREICH
                c_exp1, c_exp2, c_exp3 = st.columns([2, 1, 1])
                with c_exp1:
                    exp_devs = st.multiselect(
                        "Export Geräte:", detected, default=detected
                    )
                with c_exp2:
                    skip_n = st.number_input("Einschwingen Skip:", value=0)
                with c_exp3:
                    stat_opt = st.radio("Status:", ["OK", "Problem"], horizontal=True)

                if st.button(
                    "🚀 Speichern & Exportieren",
                    type="primary",
                    use_container_width=True,
                ):
                    if not exp_devs:
                        st.error("Keine Geräte gewählt.")
                    else:
                        semap = {}
                        for l in TARGET_LEVELS_T4:
                            semap[l] = (
                                st.session_state[f"s_{l}"],
                                st.session_state[f"e_{l}"],
                            )

                        out = save_sorted_raw_data_tab4(
                            sel_file_path,
                            full_data_t4,
                            semap,
                            exp_devs,
                            sel_ref_dev,
                            clean_name_base,
                            skip_n,
                        )
                        if out:
                            s_code = "WARNING" if stat_opt == "Problem" else "OK"
                            save_tracking_status(clean_name_base, s_code)
                            st.success(f"Gespeichert: {out}")
            clean_name_base = os.path.basename(sel_file_path).replace(".csv", "")

            # Smart Loading Config
            # Prüfen ob wir Config haben (nutzt main load_all_configs)
            loaded_cfg = all_configs.get(clean_name_base, {})
            # Wenn state 0 ist, aber Config da, laden
            if (
                all(st.session_state[f"s_{l}"] == 0 for l in TARGET_LEVELS_T4)
                and loaded_cfg
            ):
                for l_str, vals in loaded_cfg.items():
                    if isinstance(vals, list) and len(vals) == 2:
                        l_int = int(l_str) if l_str.isdigit() else 0
                        if l_int in TARGET_LEVELS_T4:
                            st.session_state[f"s_{l_int}"] = vals[0]
                            st.session_state[f"e_{l_int}"] = vals[1]

            detected = load_file_preview_tab4(sel_file_path)

            # Info Header
            c_inf1, c_inf2 = st.columns([3, 1])
            with c_inf1:
                st.subheader(clean_name_base)
            with c_inf2:
                # Nennstrom aus Dateiname raten
                match = re.search(r"[-_](\d+)A[-_]", clean_name_base)
                ns_val = float(match.group(1)) if match else 2000.0
                nennstrom_t4 = st.number_input("Nennstrom:", value=ns_val, key="ns_t4")

            # Load Full Data
            t_idx, full_data_t4 = load_all_data_tab4(sel_file_path, detected)

            if t_idx:
                # Plot Settings
                sel_ref_dev = st.selectbox(
                    "Referenz-Gerät (L1):", detected, key="ref_dev_t4"
                )

                # RECOVERY BUTTON
                if st.button("♻️ Import von existierender 'Sortiert'-Datei"):
                    l1_dat = full_data_t4[sel_ref_dev]["L1"]
                    succ, txt = try_recover_from_sorted_file(
                        sel_file_path, l1_dat, sel_ref_dev
                    )
                    if succ:
                        st.success(txt)
                        st.rerun()
                    else:
                        st.error(txt)

                # PLOT
                ref_l1 = full_data_t4[sel_ref_dev]["L1"]
                if ref_l1 is not None:
                    fig_sel = go.Figure()
                    pct_val = (ref_l1 / nennstrom_t4) * 100
                    fig_sel.add_trace(
                        go.Scatter(
                            x=t_idx,
                            y=pct_val,
                            name=f"{sel_ref_dev} L1 %",
                            line=dict(color="orange"),
                        )
                    )

                    # Areas
                    for level in TARGET_LEVELS_T4:
                        fig_sel.add_hline(y=level, line_dash="dot", line_color="gray")
                        s = st.session_state[f"s_{level}"]
                        e = st.session_state[f"e_{level}"]
                        if s > 0 and e > s:
                            fig_sel.add_vrect(
                                x0=s,
                                x1=e,
                                fillcolor="rgba(0,200,100,0.2)",
                                line_width=0,
                            )
                            fig_sel.add_annotation(
                                x=(s + e) / 2,
                                y=level + 5,
                                text=f"<b>{level}%</b>",
                                showarrow=False,
                                font=dict(color="green"),
                            )

                    fig_sel.update_layout(
                        height=400,
                        margin=dict(t=30, b=0, l=0, r=0),
                        yaxis_title="Last %",
                    )
                    st.plotly_chart(fig_sel, use_container_width=True)

                # INPUTS
                st.markdown("##### Bereiche definieren")

                # Batch Inputs
                batches = [
                    TARGET_LEVELS_T4[i : i + 4]
                    for i in range(0, len(TARGET_LEVELS_T4), 4)
                ]
                for batch in batches:
                    c_h = st.columns(4)
                    for i, l in enumerate(batch):
                        c_h[i].markdown(f"**{l}%**")
                    c_e = st.columns(4)
                    for i, l in enumerate(batch):
                        c_e[i].number_input(
                            "Ende",
                            key=f"e_{l}",
                            on_change=update_start_callback,
                            args=(l,),
                            label_visibility="collapsed",
                        )
                    c_s = st.columns(4)
                    for i, l in enumerate(batch):
                        c_s[i].number_input(
                            "Start", key=f"s_{l}", label_visibility="collapsed"
                        )

                st.divider()

                # EXPORT
                c_exp1, c_exp2, c_exp3 = st.columns([2, 1, 1])
                with c_exp1:
                    exp_devs = st.multiselect(
                        "Export Geräte:", detected, default=detected
                    )
                with c_exp2:
                    skip_n = st.number_input("Einschwingen Skip:", value=0)
                with c_exp3:
                    stat_opt = st.radio("Status:", ["OK", "Problem"], horizontal=True)

                if st.button(
                    "🚀 Speichern & Exportieren",
                    type="primary",
                    use_container_width=True,
                ):
                    if not exp_devs:
                        st.error("Keine Geräte gewählt.")
                    else:
                        semap = {}
                        for l in TARGET_LEVELS_T4:
                            semap[l] = (
                                st.session_state[f"s_{l}"],
                                st.session_state[f"e_{l}"],
                            )

                        out = save_sorted_raw_data_tab4(
                            sel_file_path,
                            full_data_t4,
                            semap,
                            exp_devs,
                            sel_ref_dev,
                            clean_name_base,
                            skip_n,
                        )
                        if out:
                            s_code = "WARNING" if stat_opt == "Problem" else "OK"
                            save_tracking_status(clean_name_base, s_code)
                            st.success(f"Gespeichert: {out}")


with tab4:
    st.markdown("### ✂️ Manueller Rohdaten-Export")

    # 1. State Initialisierung für Tab 4
    TARGET_LEVELS_T4 = [5, 20, 50, 80, 90, 100, 120]
    for level in TARGET_LEVELS_T4:
        if f"s_{level}" not in st.session_state: st.session_state[f"s_{level}"] = 0
        if f"e_{level}" not in st.session_state: st.session_state[f"e_{level}"] = 0

    # 2. Layout aufteilen
    col_sel_nav, col_sel_main = st.columns([1, 3])

    # Globale Daten laden
    all_files_t4 = get_files_tab4()
    df_status_t4 = load_status_tracking()

    # --- LINKER BEREICH: NAVIGATION ---
    with col_sel_nav:
        st.markdown("#### 📂 Dateiauswahl")
        files_options = []
        file_map = {}
        
        # Dateiliste bauen
        for f in all_files_t4:
            name_only = os.path.basename(f)
            clean_name_base, _ = extract_metadata(f)

            # Status Icon ermitteln
            icon = "❌"
            if clean_name_base in df_status_t4.index:
                s = df_status_t4.loc[clean_name_base, "Status"]
                if s == "OK": icon = "✅"
                elif s == "WARNING": icon = "⚠️"

            disp = f"{icon} {name_only}"
            files_options.append(disp)
            file_map[disp] = f

        if not files_options:
            st.warning("Keine CSV Dateien.")
            st.stop()

        # --- CALLBACKS DEFINIEREN ---
        def on_change_file():
            # Reset der Bereiche bei Dateiwechsel
            for l in TARGET_LEVELS_T4:
                st.session_state[f"s_{l}"] = 0
                st.session_state[f"e_{l}"] = 0

        def nav_callback(direction):
            current_val = st.session_state.get("t4_file_sel")
            if current_val in files_options:
                curr_idx = files_options.index(current_val)
                if direction == "prev":
                    new_idx = (curr_idx - 1) % len(files_options)
                else:
                    new_idx = (curr_idx + 1) % len(files_options)
                st.session_state["t4_file_sel"] = files_options[new_idx]
                on_change_file()

        def next_open_callback():
            current_val = st.session_state.get("t4_file_sel")
            start_idx = 0
            if current_val in files_options:
                start_idx = (files_options.index(current_val) + 1) % len(files_options)
            
            search_list = files_options[start_idx:] + files_options[:start_idx]
            for disp_str in search_list:
                f_path = file_map[disp_str]
                base_n, _ = extract_metadata(f_path)
                stat = "NONE"
                if base_n in df_status_t4.index: 
                    stat = df_status_t4.loc[base_n, "Status"]
                if stat != "OK":
                    st.session_state["t4_file_sel"] = disp_str
                    on_change_file()
                    break

        # --- WIDGETS ANZEIGEN ---
        sel_file_disp = st.selectbox(
            "Datei:", 
            files_options, 
            key="t4_file_sel", 
            on_change=on_change_file
        )
        sel_file_path = file_map[sel_file_disp]

        # Navigation Buttons
        c_prev, c_next = st.columns(2)
        c_prev.button("⬅️ Zurück", key="btn_prev", on_click=nav_callback, args=("prev",))
        c_next.button("Weiter ➡️", key="btn_next", on_click=nav_callback, args=("next",))
        
        st.button(
            "⏩ Nächste Offene", 
            type="primary", 
            use_container_width=True, 
            on_click=next_open_callback
        )

    # --- RECHTER BEREICH: INHALT ---
    with col_sel_main:
        if sel_file_path:
            clean_name_base, detected_nennstrom = extract_metadata(sel_file_path)

            # --- AUTO-LOAD LOGIK ---
            current_configs = load_all_configs()
            saved_times = current_configs.get(clean_name_base, {})
            
            # Status-Prüfung für Anzeige
            config_found = False
            if saved_times:
                config_found = True

            # Nur laden, wenn alle Felder 0 sind UND wir Daten haben
            ranges_empty = all(st.session_state[f"s_{l}"] == 0 for l in TARGET_LEVELS_T4)
            
            if ranges_empty and config_found:
                for lvl_str, vals in saved_times.items():
                    if lvl_str.isdigit():
                        lvl = int(lvl_str)
                        if lvl in TARGET_LEVELS_T4 and isinstance(vals, list) and len(vals) == 2:
                            st.session_state[f"s_{lvl}"] = vals[0]
                            st.session_state[f"e_{lvl}"] = vals[1]

            # Vorschau laden
            detected = load_file_preview_tab4(sel_file_path)

            # Header Info mit Status-Anzeige
            c_inf1, c_inf2 = st.columns([3, 1])
            with c_inf1:
                st.subheader(clean_name_base)
                if config_found:
                    st.caption("✅ Zeiten aus Config geladen")
                else:
                    st.caption("❌ Keine gespeicherten Zeiten für diese Datei gefunden")
            
            with c_inf2:
                nennstrom_t4 = st.number_input(
                    "Nennstrom:", 
                    value=float(detected_nennstrom), 
                    key=f"ns_{clean_name_base}"
                )

            # Volle Daten laden
            t_idx, full_data_t4 = load_all_data_tab4(sel_file_path, detected)

            if t_idx:
                # Plot Konfiguration
                c_tool1, c_tool2 = st.columns([2, 1])
                with c_tool1:
                    sel_ref_dev = st.selectbox("Referenz-Gerät (L1):", detected, key="ref_dev_t4")
                with c_tool2:
                    if st.button("🔄 Config neu laden"):
                        fresh_configs = load_all_configs()
                        fresh_times = fresh_configs.get(clean_name_base, {})
                        if fresh_times:
                            for lvl_str, vals in fresh_times.items():
                                if lvl_str.isdigit():
                                    lvl = int(lvl_str)
                                    if lvl in TARGET_LEVELS_T4 and isinstance(vals, list) and len(vals) == 2:
                                        st.session_state[f"s_{lvl}"] = vals[0]
                                        st.session_state[f"e_{lvl}"] = vals[1]
                            st.rerun()
                        else:
                            st.error("Nichts in Config gefunden.")

                # Plot erstellen
                ref_l1 = full_data_t4[sel_ref_dev]["L1"]
                if ref_l1 is not None:
                    fig_sel = go.Figure()
                    pct_val = (ref_l1 / nennstrom_t4) * 100
                    fig_sel.add_trace(go.Scatter(x=t_idx, y=pct_val, name=f"{sel_ref_dev} L1 %", line=dict(color="orange")))
                    
                    # Bereiche einzeichnen
                    for level in TARGET_LEVELS_T4:
                        fig_sel.add_hline(y=level, line_dash="dot", line_color="gray", opacity=0.5)
                        s = st.session_state[f"s_{level}"]
                        e = st.session_state[f"e_{level}"]
                        if s > 0 and e > s:
                            fig_sel.add_vrect(x0=s, x1=e, fillcolor="rgba(0,200,100,0.2)", line_width=0)
                            fig_sel.add_annotation(x=(s+e)/2, y=level+5, text=f"<b>{level}%</b>", showarrow=False, font=dict(color="green"))
                    
                    fig_sel.update_layout(height=400, margin=dict(t=30,b=0,l=0,r=0), yaxis_title="Last %")
                    st.plotly_chart(fig_sel, use_container_width=True)

                # Eingabefelder
                st.markdown("##### Bereiche definieren")
                batches = [TARGET_LEVELS_T4[i:i + 4] for i in range(0, len(TARGET_LEVELS_T4), 4)]
                for batch in batches:
                    c_h = st.columns(4)
                    for i, l in enumerate(batch): c_h[i].markdown(f"**{l}%**")
                    c_e = st.columns(4)
                    for i, l in enumerate(batch): 
                        c_e[i].number_input("Ende", key=f"e_{l}", on_change=update_start_callback, args=(l,), label_visibility="collapsed")
                    c_s = st.columns(4)
                    for i, l in enumerate(batch): 
                        c_s[i].number_input("Start", key=f"s_{l}", label_visibility="collapsed")

                st.divider()
                
                # Export Controls
                c_exp1, c_exp2, c_exp3 = st.columns([2, 1, 1])
                with c_exp1:
                    exp_devs = st.multiselect("Export Geräte:", detected, default=detected)
                with c_exp2:
                    skip_n = st.number_input("Einschwingen Skip:", value=0)
                with c_exp3:
                    stat_opt = st.radio("Status:", ["OK", "Problem"], horizontal=True)
                
                if st.button("🚀 Speichern & Exportieren", type="primary", use_container_width=True):
                    if not exp_devs:
                        st.error("Keine Geräte gewählt.")
                    else:
                        semap = {}
                        for l in TARGET_LEVELS_T4:
                            semap[l] = (st.session_state[f"s_{l}"], st.session_state[f"e_{l}"])
                        
                        out = save_sorted_raw_data_tab4(sel_file_path, full_data_t4, semap, exp_devs, sel_ref_dev, clean_name_base, skip_n)
                        if out:
                            s_code = "WARNING" if stat_opt == "Problem" else "OK"
                            save_tracking_status(clean_name_base, s_code)
                            st.success(f"Gespeichert: {out}")