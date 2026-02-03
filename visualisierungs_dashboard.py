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
import numpy as np
from pathlib import Path
import glob

# --- KONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "daten")

DATA_FILE = os.path.join(DATA_DIR, "messdaten_db.parquet")

# Config Datei Pfad
CONFIG_FILE = os.path.join(DATA_DIR, "saved_configs.json")

WORK_DIR = os.path.join(BASE_DIR, "matlab_working_dir")

DEFAULT_MATLAB_PATH = r"C:\Program Files\MATLAB\R2025a\bin\matlab.exe"

PHASES = ["L1", "L2", "L3"]
ZONES = {
    "Niederstrom (5-50%)": [5, 20, 50],
    "Nennstrom (80-100%)": [80, 90, 100],
    "Überlast (≥120%)": [120, 150, 200],
}

# --- DEFINITIONEN ---
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

BLUES = ["#1f4e8c", "#2c6fb2", "#4a8fd1", "#6aa9e3", "#8fc0ee", "#b3d5f7"]
ORANGES = ["#8c4a2f", "#a65a2a", "#c96a2a", "#e07b39", "#f28e4b", "#f6a25e"]
OTHERS = ["#4caf50", "#6bd36b", "#b0b0b0", "#b39ddb", "#bc8f6f", "#f2a7d6"]

LINE_STYLES = ["solid", "dash", "dot", "dashdot", "longdash"]
MARKER_SYMBOLS = ["circle", "square", "diamond", "cross", "x", "triangle-up", "star"]


# --- HELPER: CONFIG MANAGEMENT ---
def load_full_json():
    """Lädt die JSON-Datei mit Fehleranzeige."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "configurations" not in data:
                    data["configurations"] = {}
                if "times" not in data:
                    data["times"] = {}
                return data
        except json.JSONDecodeError as e:
            st.error(
                f"⚠️ FEHLER: Die Datei 'saved_configs.json' ist beschädigt (Syntax-Fehler)!\nDetails: {e}"
            )
            return {"configurations": {}, "times": {}}
        except Exception as e:
            st.error(f"⚠️ Unbekannter Fehler beim Laden der Config: {e}")
            return {"configurations": {}, "times": {}}
    return {"configurations": {}, "times": {}}


def load_dashboard_configs():
    data = load_full_json()
    return data.get("configurations", {})


def load_time_configs():
    data = load_full_json()
    return data.get("times", {})


def save_dashboard_config(name, config_data):
    full_data = load_full_json()
    if "configurations" not in full_data:
        full_data["configurations"] = {}
    full_data["configurations"][name] = config_data
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)


def save_time_config(filename, time_data):
    full_data = load_full_json()
    if "times" not in full_data:
        full_data["times"] = {}
    full_data["times"][filename] = time_data
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)


def delete_config(name):
    full_data = load_full_json()
    if "configurations" in full_data and name in full_data["configurations"]:
        del full_data["configurations"][name]
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2)
        return True
    return False


# --- HELPER: FILE NAMING ---
def sanitize_filename(name):
    if not name:
        return "Unbenannt"
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
        return pd.Series([hersteller, modell, nennstrom, burde_part])
    except:
        return pd.Series(["Fehler", "Fehler", 0.0, "Fehler"])


def create_single_phase_figure(
    df_sub,
    phase,
    acc_class,
    y_limit,
    y_shift,
    bottom_mode,
    show_err_bars,
    title_prefix="",
    nticks_x=20,
    nticks_y=15,
):
    is_single_row = bottom_mode == "Ausblenden"
    if is_single_row:
        fig = make_subplots(rows=1, cols=1, subplot_titles=[f"Fehlerverlauf {phase}"])
    else:
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

    lim_x, lim_y_p, lim_y_n = get_trumpet_limits(acc_class)

    # Dynamischer Name für die Legende
    label_class = f"Klassengrenzen {str(acc_class).replace('.', ',')}"

    # HIER WAR DER FEHLER: fig statt fig_ex
    fig.add_trace(
        go.Scatter(
            x=lim_x,
            y=lim_y_p,
            mode="lines",
            line=dict(color="black", width=1.5, dash="dash"),
            name=label_class,
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
        if not row_first["final_visible"]:
            continue

        leg_name = row_first["final_legend"]
        color = row_first["final_color"]
        style = row_first["final_style"]
        symbol = row_first["final_symbol"]
        width = row_first["final_width"]
        size_val = row_first["final_size"]

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

        if not is_single_row:
            if bottom_mode == "Standardabweichung":
                if show_err_bars:
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
            y=-0.15,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=16),
        ),
        margin=dict(l=60, r=30, t=80, b=120),
    )

    y_min = -y_limit + y_shift
    y_max = y_limit + y_shift
    fig.update_yaxes(range=[y_min, y_max], title_text="Fehler [%]", row=1, col=1)

    if not is_single_row:
        if bottom_mode == "Standardabweichung":
            fig.update_yaxes(title_text="StdAbw [%]", row=2, col=1)
        else:
            fig.update_yaxes(title_text="Strom [A]", row=2, col=1)
        fig.update_xaxes(title_text="Strom [% In]", row=2, col=1)
    else:
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
    shift_val = Y_SHIFT_PH;
    
    if ~isfile(filename); error('Datei plot_data.csv nicht gefunden!'); end
    data = readtable(filename, 'Delimiter', ',');
    hex2rgb = @(hex) sscanf(hex(2:end),'%2x%2x%2x',[1 3])/255;
    x_lims = [1, 5, 20, 100, 120];
    if limits_class == 0.2; y_lims = [0.75, 0.35, 0.2, 0.2, 0.2];
    elif limits_class == 0.5; y_lims = [1.5, 1.5, 0.75, 0.5, 0.5];
    elif limits_class == 1.0; y_lims = [3.0, 1.5, 1.0, 1.0, 1.0];
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
        ylabel(ax1, 'Fehler [%]'); 
        
        % NEU: Verschiebung anwenden
        ylim(ax1, [- Y_LIMIT_PH + shift_val, Y_LIMIT_PH + shift_val]); 
        xlim(ax1, [0, 125]);
        
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
# --- HELPER FÜR TAB 4 & 5 (ROHDATEN & AGGREGATION) ---
# ==========================================
SELECTOR_TRACKING_CSV = os.path.join(DATA_DIR, "manuelle_ergebnisse.csv")


def extract_metadata(filepath):
    filename = os.path.basename(filepath)
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
    orig_p = Path(original_path)
    sorted_root = os.path.join(BASE_DIR, "messungen_sortiert")
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
        save_time_config(clean_filename, start_end_map)
        return str(output_path)
    except Exception as e:
        return None


# --- AGGREGATION HELPER (TAB 5) ---
def aggregator_load_metadata(db_path, keep_cols):
    if not os.path.exists(db_path):
        return {}
    try:
        df_old = pd.read_parquet(db_path)
        meta_dict = {}
        if "raw_file" not in df_old.columns:
            return {}
        for file_name in df_old["raw_file"].unique():
            row = df_old[df_old["raw_file"] == file_name].iloc[0]
            file_meta = {}
            for col in keep_cols:
                if col in df_old.columns:
                    val = row[col]
                    if pd.notna(val) and val != 0 and val != "" and val != "Unbekannt":
                        file_meta[col] = val
            if file_meta:
                meta_dict[file_name] = file_meta
        return meta_dict
    except:
        return {}


def aggregator_extract_metadata(filepath):
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
    wandler_key = f"{manufacturer} {original_name}"
    return {
        "filepath": filepath,
        "folder": folder_name,
        "hersteller_auto": manufacturer,
        "nennstrom": nennstrom,
        "wandler_key": wandler_key,
        "dateiname": filename,
        "original_name_clean": original_name,
    }


def aggregator_analyze_file(filepath, meta, target_levels, phases, ref_keywords):
    try:
        df = pd.read_csv(filepath, sep=";")
        if len(df.columns) < 2:
            df = pd.read_csv(filepath, sep=",")
    except:
        return [], "Lesefehler"
    df.columns = [c.strip() for c in df.columns]
    value_cols = [c for c in df.columns if "_I" in c]
    if not value_cols:
        return [], "Keine Strom-Daten"
    results = []
    for level in target_levels:
        lvl_str = f"{level:02d}"
        nominal_amp = meta["nennstrom"] * (level / 100.0)
        for phase in phases:
            relevant_cols = [
                c for c in value_cols if c.startswith(f"{lvl_str}_{phase}")
            ]
            if not relevant_cols:
                continue
            devices_map = {}
            for col in relevant_cols:
                m1 = re.search(rf"{lvl_str}_{phase}_I_(.+)$", col)
                m2 = re.search(rf"{lvl_str}_{phase}_(.+)_I$", col)
                if m1:
                    devices_map[m1.group(1)] = col
                elif m2:
                    devices_map[m2.group(1)] = col
            if not devices_map:
                continue
            phys_ref = None
            for kw in ref_keywords:
                for dev in devices_map.keys():
                    if kw in dev.lower():
                        phys_ref = dev
                        break
                if phys_ref:
                    break
            if not phys_ref:
                phys_ref = sorted(list(devices_map.keys()))[0]
            col_ref = devices_map[phys_ref]
            vals_ref = pd.to_numeric(df[col_ref], errors="coerce").dropna()
            mean_ref = vals_ref.mean() if not vals_ref.empty else 0
            std_ref = vals_ref.std() if not vals_ref.empty else 0
            for dev, col_dut in devices_map.items():
                vals_dut = pd.to_numeric(df[col_dut], errors="coerce").dropna()
                if vals_dut.empty:
                    continue
                mean_dut = vals_dut.mean()
                std_dut = vals_dut.std()
                base = {
                    "wandler_key": meta["wandler_key"],
                    "folder": meta["folder"],
                    "phase": phase,
                    "target_load": level,
                    "nennstrom": meta["nennstrom"],
                    "val_dut_mean": mean_dut,
                    "val_dut_std": std_dut,
                    "dut_name": dev,
                    "raw_file": meta["dateiname"],
                    "Hersteller": meta["hersteller_auto"],
                    "Modell": meta["original_name_clean"],
                    "Geometrie": "Unbekannt",
                }
                if dev != phys_ref and mean_ref > 0:
                    e = base.copy()
                    e.update(
                        {
                            "val_ref_mean": mean_ref,
                            "val_ref_std": std_ref,
                            "ref_name": phys_ref,
                            "comparison_mode": "device_ref",
                        }
                    )
                    results.append(e)
                if nominal_amp > 0:
                    e = base.copy()
                    e.update(
                        {
                            "val_ref_mean": nominal_amp,
                            "val_ref_std": 0.0,
                            "ref_name": "Nennwert",
                            "comparison_mode": "nominal_ref",
                        }
                    )
                    results.append(e)
    return results, "OK"


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

st.markdown(
    """<style>[data-testid="stSidebar"] { min-width: 600px; max-width: 900px; }</style>""",
    unsafe_allow_html=True,
)

df = load_data()
if df is None:
    st.error(f"⚠️ Datei '{DATA_FILE}' fehlt.")
    st.stop()

# --- NEU: Globale Listen für Smart-Match vorbereiten ---
ALL_WANDLERS = sorted(df["wandler_key"].unique())
ALL_DUTS = sorted(df["dut_name"].unique())

# --- SIDEBAR: CONFIG ---
st.sidebar.header("💾 Konfiguration")
dashboard_configs = load_dashboard_configs()
config_names = sorted(list(dashboard_configs.keys()))

# --- SIDEBAR: CONFIG ---
st.sidebar.header("💾 Konfiguration")
dashboard_configs = load_dashboard_configs()
config_names = sorted(list(dashboard_configs.keys()))

# Container für die Konfigurations-Verwaltung
with st.sidebar.expander("Verwaltung", expanded=True):

    # --- VORBEREITUNG: DIALOG-FUNKTION (Muss vor dem Aufruf definiert sein) ---
    @st.dialog("Wirklich löschen?")
    def open_delete_dialog(config_name):
        st.write(
            f"Möchtest du die Konfiguration **'{config_name}'** wirklich unwiderruflich löschen?"
        )
        st.warning("Dieser Vorgang kann nicht rückgängig gemacht werden.")

        col_yes, col_no = st.columns([1, 1])

        with col_yes:
            # "Ja"-Button (Führt die Löschung aus)
            if st.button("Ja, löschen", type="primary", use_container_width=True):
                if delete_config(config_name):
                    st.toast(f"Konfiguration '{config_name}' gelöscht!", icon="🗑️")
                    st.rerun()

        with col_no:
            # "Nein"-Button (Schließt das Fenster durch Rerun)
            if st.button("Nein", type="secondary", use_container_width=True):
                st.rerun()

    # --- ABSCHNITT 1: BESTEHENDE VERWALTEN ---
    st.markdown("**1. Bestehende Config wählen:**")
    sel_config_load = st.selectbox(
        "Wähle Config:",
        config_names,
        index=0 if config_names else None,
        label_visibility="collapsed",
    )

    # Drei Spalten für: Laden | Update | Löschen
    col_load, col_update, col_del = st.columns([1, 1, 1])

    # 1.1 LADEN
    with col_load:
        if st.button("📂 Laden", use_container_width=True):
            if sel_config_load:
                data = dashboard_configs[sel_config_load]

                # --- HELPER: INTELLIGENTES MATCHING (Lokal definiert für Zugriff) ---
                def smart_match(stored_list, available_options, col_name_in_df=None):
                    if not stored_list:
                        return []

                    # Performance & Typ-Konvertierung für Vergleich
                    avail_set = set(available_options)
                    # Erstelle Mapping String->Original, um Typ-Probleme (2000 vs 2000.0) zu lösen
                    avail_str_map = {str(x): x for x in available_options}

                    valid = []

                    # 1. Direkter Match (Exakt)
                    for item in stored_list:
                        if item in avail_set:
                            valid.append(item)
                        # 2. String Match (z.B. JSON int 2000 vs DataFrame float 2000.0)
                        elif str(item) in avail_str_map:
                            valid.append(avail_str_map[str(item)])

                    # 3. Falls Meta-Suche nötig (für Wandler-Keys, die sich geändert haben könnten)
                    missing = set(stored_list) - set(valid)
                    # Wir prüfen nur Items, die wir noch nicht über String-Match gefunden haben
                    # (Achtung: stored_list items könnten hier noch int/float sein)

                    if missing and col_name_in_df:
                        for m in missing:
                            m_str = str(m)
                            # A) Teilstring-Suche in den verfügbaren Optionen
                            matches_key = [
                                opt for opt in available_options if m_str in str(opt)
                            ]
                            if matches_key:
                                valid.extend(matches_key)
                                continue

                            # B) Suche über Hersteller/Modell im DataFrame
                            if col_name_in_df == "wandler_key":
                                # Wir nutzen df (global), sicherstellen dass Strings verglichen werden
                                matches_meta = df[
                                    df["Modell"]
                                    .astype(str)
                                    .str.contains(m_str, regex=False, case=False)
                                    | df["Hersteller"]
                                    .astype(str)
                                    .str.contains(m_str, regex=False, case=False)
                                ]["wandler_key"].unique()
                                valid.extend(matches_meta)

                    return sorted(list(set(valid)))

                # Session State befüllen
                st.session_state["k_current"] = data.get("current", [])
                st.session_state["k_geos"] = data.get("geos", [])

                raw_wandlers = data.get("wandlers", [])
                st.session_state["k_wandlers"] = smart_match(
                    raw_wandlers, ALL_WANDLERS, "wandler_key"
                )

                raw_duts = data.get("duts", [])
                dut_aliases = {"Einspeisung": "PAC1", "Pruefling": "PAC2"}
                mapped_duts = [dut_aliases.get(d, d) for d in raw_duts]
                st.session_state["k_duts"] = smart_match(mapped_duts, ALL_DUTS)

                st.session_state["k_comp"] = data.get(
                    "comp_mode", "Messgerät (z.B. PAC1)"
                )
                st.session_state["k_sync"] = data.get("sync_axes", True)
                st.session_state["k_ylim"] = data.get("y_limit", 1.5)
                st.session_state["k_yshift"] = data.get("y_shift", 0.0)
                st.session_state["k_class"] = data.get("acc_class", 0.2)
                st.session_state["k_bottom_mode"] = data.get(
                    "bottom_plot_mode", "Standardabweichung"
                )
                st.session_state["k_errbars"] = data.get("show_err_bars", True)
                st.session_state["k_eco_x"] = data.get("eco_x", "Preis (€)")
                st.session_state["k_eco_y"] = data.get("eco_y", ["Fehler Nennstrom"])
                st.session_state["k_eco_type"] = data.get("eco_type", "Scatter")

                st.session_state["loaded_colors"] = data.get("custom_colors", {})
                st.session_state["loaded_legends"] = data.get("custom_legends", {})
                st.session_state["loaded_titles"] = data.get("custom_titles", {})
                st.session_state["loaded_styles"] = data.get("custom_styles", {})
                st.session_state["loaded_symbols"] = data.get("custom_symbols", {})
                st.session_state["loaded_widths"] = data.get("custom_widths", {})
                st.session_state["loaded_visible"] = data.get("custom_visible", {})
                st.session_state["loaded_sizes"] = data.get("custom_sizes", {})

                if "design_editor" in st.session_state:
                    del st.session_state["design_editor"]
                if "titles_editor" in st.session_state:
                    del st.session_state["titles_editor"]

                st.success(f"'{sel_config_load}' geladen!")
                st.rerun()

    # 1.2 UPDATE (Überschreiben)
    with col_update:
        if st.button("💾 Speichern", use_container_width=True):
            if sel_config_load:
                st.session_state["trigger_save"] = True
                st.session_state["save_name"] = sel_config_load
            else:
                st.warning("Keine Config ausgewählt.")

    # 1.3 LÖSCHEN (Mit Pop-up Aufruf)
    with col_del:
        if st.button("🗑️ Löschen", use_container_width=True):
            if sel_config_load:
                # Hier rufen wir das Pop-up auf, statt direkt zu löschen
                open_delete_dialog(sel_config_load)
            else:
                st.warning("Keine Config ausgewählt.")

    st.divider()

    # --- ABSCHNITT 2: NEUE ANSICHT ---
    st.markdown("**2. Neue Ansicht erstellen:**")
    new_config_name = st.text_input(
        "Name eingeben:", placeholder="z.B. Test_Neu", label_visibility="collapsed"
    )

    if st.button("➕ Hinzufügen / Als Neu speichern", use_container_width=True):
        if new_config_name:
            if new_config_name in config_names:
                st.warning(
                    "Name existiert bereits! Bitte oben 'Speichern' nutzen oder umbenennen."
                )
            else:
                st.session_state["trigger_save"] = True
                st.session_state["save_name"] = new_config_name
        else:
            st.warning("Bitte Namen eingeben.")


# --- SIDEBAR: FILTER ---
st.sidebar.header("🎛️ Globale Filter")
available_currents = sorted(df["nennstrom"].unique())
default_curr = st.session_state.get(
    "k_current", [available_currents[0]] if available_currents else []
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

df_curr = df[df["nennstrom"].isin(sel_currents)]
available_geos = sorted(df_curr["Geometrie"].astype(str).unique())
saved_geos = st.session_state.get("k_geos", available_geos)
valid_geos = [g for g in saved_geos if g in available_geos]
if not valid_geos and available_geos:
    valid_geos = available_geos
st.session_state["k_geos"] = valid_geos

sel_geos = st.sidebar.multiselect("2. Geometrie:", available_geos, key="k_geos")
if not sel_geos:
    st.stop()

df_geo_filtered = df_curr[df_curr["Geometrie"].isin(sel_geos)]
available_wandlers = sorted(df_geo_filtered["wandler_key"].unique())
saved_wandlers = st.session_state.get("k_wandlers", available_wandlers)
valid_wandlers = [w for w in saved_wandlers if w in available_wandlers]
st.session_state["k_wandlers"] = valid_wandlers

sel_wandlers = st.sidebar.multiselect(
    "3. Wandler / Messung:", available_wandlers, key="k_wandlers"
)
if not sel_wandlers:
    st.stop()

df_wandler_subset = df_geo_filtered[df_geo_filtered["wandler_key"].isin(sel_wandlers)]
available_duts = sorted(df_wandler_subset["dut_name"].unique())
saved_duts = st.session_state.get("k_duts", available_duts)
valid_duts = [d for d in saved_duts if d in available_duts]
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
    0.5,
    20.0,
    float(st.session_state.get("k_ylim", 1.5)),
    0.5,
    key="k_ylim",
)

# NEU: Y-SHIFT SLIDER
y_shift = st.sidebar.slider(
    "Y-Achse Verschiebung (+/- %)",
    -20.0,
    20.0,
    float(st.session_state.get("k_yshift", 0.0)),
    0.5,
    key="k_yshift",
)

acc_class = st.sidebar.selectbox(
    "Norm-Klasse",
    [0.2, 0.5, 1.0, 3.0],
    index=[0.2, 0.5, 1.0, 3.0].index(st.session_state.get("k_class", 0.2)),
    key="k_class",
)

st.sidebar.markdown("#### 📏 Achsen-Auflösung")
col_ticks_x, col_ticks_y = st.sidebar.columns(2)
with col_ticks_x:
    nticks_x = st.slider("Ticks X-Achse", 5, 50, 20, key="k_nticks_x")
with col_ticks_y:
    nticks_y = st.slider("Ticks Y-Achse", 5, 50, 15, key="k_nticks_y")

bottom_plot_options = ["Standardabweichung", "Messwert (Absolut)", "Ausblenden"]
try:
    b_idx = bottom_plot_options.index(
        st.session_state.get("k_bottom_mode", "Standardabweichung")
    )
except:
    b_idx = 0
bottom_plot_mode = st.sidebar.selectbox(
    "Unteres Diagramm:", bottom_plot_options, index=b_idx, key="k_bottom_mode"
)
use_single_row = bottom_plot_mode == "Ausblenden"
show_err_bars = st.sidebar.checkbox(
    "Fehlerbalken (StdAbw)",
    value=st.session_state.get("k_errbars", True),
    key="k_errbars",
)

# --- CUSTOMIZATION ---
with st.sidebar.expander("Farben & Namen bearbeiten", expanded=False):
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
        cur_geo = str(row["Geometrie"])

        config_data.append(
            {
                "ID": uid,
                "Anzeigen": cur_visible,
                "Legende": cur_legend,
                "Geometrie": cur_geo,
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
            "Geometrie": st.column_config.TextColumn("Geometrie", disabled=True),
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

    map_legend = dict(zip(edited_config["ID"], edited_config["Legende"]))
    map_color = dict(zip(edited_config["ID"], edited_config["Farbe"]))
    map_style = dict(zip(edited_config["ID"], edited_config["Linie"]))
    map_symbol = dict(zip(edited_config["ID"], edited_config["Marker"]))
    map_width = dict(zip(edited_config["ID"], edited_config["Breite"]))
    map_visible = dict(zip(edited_config["ID"], edited_config["Anzeigen"]))
    map_size = dict(zip(edited_config["ID"], edited_config["Größe"]))

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
        "y_shift": y_shift,  # <--- HIER MIT SPEICHERN
        "acc_class": acc_class,
        "show_err_bars": show_err_bars,
        "bottom_plot_mode": bottom_plot_mode,
        "custom_colors": map_color,
        "custom_legends": map_legend,
        "custom_styles": map_style,
        "custom_symbols": map_symbol,
        "custom_widths": map_width,
        "custom_visible": map_visible,
        "custom_sizes": map_size,
        "custom_titles": TITLES_MAP,
        "eco_x": st.session_state.get("k_eco_x", "Preis (€)"),
        "eco_y": st.session_state.get("k_eco_y", ["Fehler Nennstrom"]),
        "eco_type": st.session_state.get("k_eco_type", "Scatter"),
    }
    save_dashboard_config(st.session_state["save_name"], snapshot_data)
    st.session_state["trigger_save"] = False
    st.toast(f"Konfiguration '{st.session_state['save_name']}' gespeichert!", icon="💾")

# --- EXPORT (IN DER SIDEBAR) ---
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

# --- HIER IST DIE ÄNDERUNG ---
# Wir weisen den Button der Variable zu, statt eine if-Abfrage zu starten.
trigger_export_btn = st.sidebar.button("🔄 Export starten", type="primary")

# --- MAIN TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Gesamtgenauigkeit",
        "💰 Ökonomische Analyse",
        "⚙️ Stammdaten-Editor",
        "✂️ Rohdaten-Selektor",
        "🔄 DB-Update",
    ]
)

with tab1:
    fig_main = create_single_phase_figure(
        df_sub,
        PHASES[0],
        acc_class,
        y_limit,
        y_shift,
        bottom_plot_mode,
        show_err_bars,
        title_prefix="",
        nticks_x=nticks_x,
        nticks_y=nticks_y,
    )

    main_title_export = TITLES_MAP.get(
        "Gesamtübersicht (Tab 1)", f"Gesamtübersicht: {current_title_str}"
    )
    if use_single_row:
        fig_main = make_subplots(
            rows=1,
            cols=3,
            shared_xaxes=True,
            subplot_titles=PHASES,
            horizontal_spacing=0.05,
        )
    else:
        fig_main = make_subplots(
            rows=2,
            cols=3,
            shared_xaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=PHASES,
        )

    lim_x, lim_y_p, lim_y_n = get_trumpet_limits(acc_class)
    for col_idx, phase in enumerate(PHASES, start=1):
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
        phase_data = df_sub[df_sub["phase"] == phase]
        for uid, group in phase_data.groupby("unique_id"):
            group = group.sort_values("target_load")
            row_first = group.iloc[0]
            if not row_first["final_visible"]:
                continue
            fig_main.add_trace(
                go.Scatter(
                    x=group["target_load"],
                    y=group["err_ratio"],
                    mode="lines+markers",
                    name=row_first["final_legend"],
                    line=dict(
                        color=row_first["final_color"],
                        width=row_first["final_width"],
                        dash=row_first["final_style"],
                    ),
                    marker=dict(
                        size=row_first["final_size"], symbol=row_first["final_symbol"]
                    ),
                    legendgroup=row_first["final_legend"],
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )
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
                            marker=dict(symbol="x", size=row_first["final_size"]),
                            legendgroup=row_first["final_legend"],
                            showlegend=False,
                        ),
                        row=2,
                        col=col_idx,
                    )

    fig_main.update_layout(
        title=main_title_export,
        template="plotly_white",
        height=800,
        legend=dict(
            orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=16)
        ),
        font=dict(family="Serif", size=14, color="black"),
    )
    if sync_axes:
        fig_main.update_yaxes(matches="y", row=1)

    # NEU: Verschiebung auch in der Übersicht
    y_min = -y_limit + y_shift
    y_max = y_limit + y_shift
    fig_main.update_yaxes(range=[y_min, y_max], title_text="Fehler [%]", row=1, col=1)

    if not use_single_row:
        if bottom_plot_mode == "Standardabweichung":
            fig_main.update_yaxes(title_text="StdAbw [%]", row=2, col=1)
        else:
            fig_main.update_yaxes(title_text="Strom [A]", row=2, col=1)
        fig_main.update_xaxes(title_text="Last [% In]", row=2, col=2)
    else:
        fig_main.update_xaxes(title_text="Last [% In]", row=1, col=2)

    fig_main.update_xaxes(nticks=nticks_x)
    fig_main.update_yaxes(nticks=nticks_y)
    st.plotly_chart(fig_main, use_container_width=True)
    st.session_state["fig_snapshot_tab1"] = fig_main

# --- NEU: Detaillierte Tabelle mit korrekter Spalten-Beschriftung (Tab 1) ---
    st.markdown("---")

    # 1. Gewünschte Lastpunkte (als Integer)
    LOAD_POINTS = [5, 20, 50, 80, 90, 100, 120]

    # 2. Daten filtern und Typ erzwingen (Wichtig: int casting für saubere Spalten)
    df_points = df_sub[df_sub["target_load"].isin(LOAD_POINTS)].copy()
    
    if not df_points.empty:
        df_points["target_load"] = df_points["target_load"].astype(int)

        # Pivotieren: Spalten werden zu 5, 20, 100... (Integer)
        df_pivot = df_points.pivot_table(
            index=["unique_id", "final_legend", "phase"],
            columns="target_load",
            values="err_ratio"
        ).reset_index()

        # 3. Metadaten dazu holen
        df_meta = df_sub.groupby("unique_id").agg({
            "Preis (€)": "first",
            "T (mm)": "first",
            "B (mm)": "first",
            "H (mm)": "first"
        }).reset_index()
        df_meta["volumen"] = (df_meta["T (mm)"] * df_meta["B (mm)"] * df_meta["H (mm)"]) / 1000.0

        df_final_t1 = pd.merge(df_pivot, df_meta, on="unique_id", how="left")

        # 4. Performance Index berechnen (auf den numerischen Spalten)
        df_final_t1["total_score"] = 0.0
        # Wir prüfen, welche der LOAD_POINTS als Spalte existieren
        existing_numeric_cols = [c for c in LOAD_POINTS if c in df_final_t1.columns]
        
        for col in existing_numeric_cols:
            mx = df_final_t1[col].abs().max()
            if mx == 0: mx = 1.0
            df_final_t1["total_score"] += (df_final_t1[col].abs() / mx) * 100

        # 5. UMBENENNUNG DER SPALTEN (Fix für die falsche Anzeige)
        # Wir benennen die Zahl-Spalten (100) in Strings ("100% In") um.
        rename_map = {c: f"{c}% In" for c in existing_numeric_cols}
        df_final_t1.rename(columns=rename_map, inplace=True)
        
        # Liste der neuen String-Spalten in korrekter Reihenfolge
        display_cols_points = [rename_map[c] for c in existing_numeric_cols]

        # 6. Konfiguration erstellen
        col_config = {
            "final_legend": "Variante",
            "phase": "Phase",
            "total_score": st.column_config.ProgressColumn(
                "Performance Index",
                help="Summierte Abweichung (Niedriger ist besser)",
                format="%.1f",
                min_value=0,
                max_value=float(df_final_t1["total_score"].max()) if not df_final_t1.empty else 100,
            ),
            "Preis (€)": st.column_config.NumberColumn("Preis", format="%.2f €"),
            "volumen": st.column_config.NumberColumn("Volumen", format="%.2f dm³"),
        }

        # Formatierung für die neuen String-Spalten (z.B. "100% In") hinzufügen
        for col_name in display_cols_points:
            col_config[col_name] = st.column_config.NumberColumn(
                col_name,       # Name fixieren
                format="%.3f %%" # 3 Nachkommastellen
            )

        # 7. Tabelle anzeigen
        with st.expander("🔢 Detaillierte Datentabelle (Stützstellen)", expanded=True):
            # Exakte Reihenfolge der Spalten definieren
            cols_order = ["final_legend", "phase"] + display_cols_points + ["total_score", "Preis (€)", "volumen"]
            
            # Nur Spalten nehmen, die wirklich da sind
            final_cols_to_show = [c for c in cols_order if c in df_final_t1.columns]

            st.dataframe(
                df_final_t1[final_cols_to_show],
                use_container_width=True,
                column_config=col_config,
                hide_index=True
            )
    else:
        st.info("Keine Daten an den Stützstellen gefunden.")

with tab2:
    st.markdown("### 💰 Ökonomische Analyse & Varianten-Vergleich")

    # --- 1. DATEN AGGREGATION (Berechnung der KPIs) ---
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
    # Volumen berechnen
    df_err["volumen"] = (df_err["vol_t"] * df_err["vol_b"] * df_err["vol_h"]) / 1000.0

    # Falls keine Daten da sind, abbrechen
    if df_err.empty:
        st.warning("Keine Daten für die aktuelle Auswahl verfügbar.")
        st.stop()

    # --- 2. KPI DASHBOARD ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Anzahl Varianten", len(df_err))
    with kpi2:
        avg_price = df_err["preis"].replace(0, np.nan).mean()
        st.metric("Ø Preis", f"{avg_price:.2f} €" if pd.notna(avg_price) else "-")
    with kpi3:
        best_acc = df_err["err_nom"].min()
        st.metric("Bester Fehler (Nenn)", f"{best_acc:.3f} %")
    with kpi4:
        best_var = df_err.loc[df_err["err_nom"].idxmin()]["legend_name"]
        st.metric("Genaueste Variante", best_var.split("|")[0][:15] + "...")

    st.markdown("---")

    # --- 3. KONFIGURATION ---
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

    with st.expander("⚙️ Achsen & Metriken konfigurieren", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            x_sel = st.selectbox(
                "X-Achse (für Scatter):",
                ["Preis (€)", "Volumen (dm³)"],
                index=0 if st.session_state.get("k_eco_x") == "Preis (€)" else 1,
                key="k_eco_x",
            )
            x_col = "preis" if "Preis" in x_sel else "volumen"
        with c2:
            y_selection = st.multiselect(
                "Vergleichs-Metriken (Y-Achsen / Radar / Ranking):",
                options=list(Y_OPTIONS_MAP.keys()),
                default=st.session_state.get("k_eco_y", ["Fehler Nennstrom"]),
                key="k_eco_y",
            )
            y_cols_selected = [Y_OPTIONS_MAP[label] for label in y_selection]

    if not y_cols_selected:
        st.error("Bitte mindestens eine Metrik auswählen.")
        st.stop()

    color_map_dict = dict(zip(df_err["legend_name"], df_err["color_hex"]))

    # --- STYLE DEFINITION FÜR ALLE LEGENDEN UNTEN ---
    legend_layout_bottom = dict(
        orientation="h",
        y=-0.25,  # Position unterhalb der X-Achse
        x=0.5,  # Zentriert
        xanchor="center",
        bgcolor="rgba(255,255,255,0.8)",
        font=dict(size=14),
    )

    # --- 4. DIAGRAMM TABS ---
    t_scat, t_bar, t_heat, t_box, t_par, t_rad = st.tabs(
        [
            "🔵 Scatter (Kosten/Nutzen)",
            "📊 Ranking (Index)",
            "🔥 Heatmap",
            "📦 Verteilung",
            "📉 Pareto",
            "🕸️ Radar",
        ]
    )

    # --- TAB: SCATTER ---
    with t_scat:
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
            hover_data=["legend_name", "Wert"],
        )
        fig_eco.update_layout(legend=legend_layout_bottom)  # LEGENDE UNTEN
        st.plotly_chart(fig_eco, use_container_width=True)

        st.session_state["fig_snapshot_tab2"] = fig_eco
        st.session_state["fig_snapshot_tab2_type"] = "Ökonomie: Scatter-Plot"

# --- TAB: RANKING (PERFORMANCE INDEX) ---
    with t_bar:
        title_str = TITLES_MAP.get(
            "Performance-Index", "Performance Index (Niedriger ist besser)"
        )
        
        # --- ANPASSUNG: Farben und Reihenfolge ---
        CUSTOM_COLORS = {
            "Preis (€)": "green",             # Grün
            "Volumen (Gesamt)": "brown",       # Braun
            "Fehler Niederstrom": "darkblue",  # Dunkelblau (Niederstrom)
            "Fehler Nennstrom": "orange",      # Orange (Nennstrom)
            "Fehler Überlast": "red"           # Rot (Überlast)
        }
        
        # Gewünschte Sortierung (Priorität)
        CUSTOM_ORDER = [
            "Preis (€)", 
            "Volumen (Gesamt)", 
            "Fehler Niederstrom", 
            "Fehler Nennstrom", 
            "Fehler Überlast"
        ]

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
            value_name="Anteil am Score (%)",
        )

        fig_eco = px.bar(
            df_long,
            y="legend_name",
            x="Anteil am Score (%)",
            color="Kategorie",
            orientation="h",
            title=title_str,
            # Hier werden Farben und Reihenfolge angewendet
            color_discrete_map=CUSTOM_COLORS,
            category_orders={"Kategorie": CUSTOM_ORDER}
        )
        
        fig_eco.update_layout(
            yaxis=dict(autorange="reversed"),
            legend=dict(
                orientation="h",
                y=-0.25, 
                x=0.5, 
                xanchor="center",
                bgcolor="rgba(255,255,255,0.8)",
                # Hier die Schriftanpassung (Dicker / Größer)
                font=dict(size=14, color="black", family="Arial Black")
            ),
        )
        st.plotly_chart(fig_eco, use_container_width=True)

        st.session_state["fig_snapshot_tab2"] = fig_eco
        st.session_state["fig_snapshot_tab2_type"] = "Ökonomie: Performance-Index"
    # --- TAB: HEATMAP ---
    with t_heat:
        title_str = TITLES_MAP.get("Heatmap", "Werte-Heatmap")
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
            text_auto=True,
            color_continuous_scale="Blues",
            title=title_str,
        )
        # Heatmap hat Colorbar, keine Legende. Wir lassen sie rechts,
        # passen aber die X-Achsen Labels an.
        fig_eco.update_layout(xaxis=dict(tickangle=45))
        st.plotly_chart(fig_eco, use_container_width=True)

        st.session_state["fig_snapshot_tab2"] = fig_eco
        st.session_state["fig_snapshot_tab2_type"] = "Ökonomie: Heatmap"

    # --- TAB: BOXPLOT ---
    with t_box:
        title_str = TITLES_MAP.get("Boxplot", f"Verteilung: {', '.join(y_selection)}")
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
        fig_eco.update_layout(legend=legend_layout_bottom)  # LEGENDE UNTEN
        st.plotly_chart(fig_eco, use_container_width=True)

        st.session_state["fig_snapshot_tab2"] = fig_eco
        st.session_state["fig_snapshot_tab2_type"] = "Ökonomie: Boxplot"

    # --- TAB: PARETO ---
    with t_par:
        title_str = TITLES_MAP.get("Pareto", "Pareto-Analyse (Einflussfaktoren)")
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
                line=dict(color="red", width=2),
            ),
            secondary_y=True,
        )
        fig_par.update_layout(
            title=title_str, legend=legend_layout_bottom
        )  # LEGENDE UNTEN
        st.plotly_chart(fig_par, use_container_width=True)

        st.session_state["fig_snapshot_tab2"] = fig_par
        st.session_state["fig_snapshot_tab2_type"] = "Ökonomie: Pareto"

    # --- TAB: RADAR ---
    with t_rad:
        title_str = TITLES_MAP.get("Radar", "Multi-Kriterieller Radar")
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
            legend=legend_layout_bottom,  # LEGENDE UNTEN
        )
        st.plotly_chart(fig_r, use_container_width=True)

        st.session_state["fig_snapshot_tab2"] = fig_r
        st.session_state["fig_snapshot_tab2_type"] = "Ökonomie: Radar"

    # --- UNTERER BEREICH: DATENTABELLE ---
    with st.expander("🔢 Detaillierte Datentabelle ansehen"):
        # Lokale Kopie für die Anzeige
        df_table = df_err.copy()

        # Sicherstellen, dass der Score (Performance Index) berechnet ist,
        # auch wenn man nicht auf dem "Ranking"-Tab war
        if "total_score" not in df_table.columns:
            df_table["total_score"] = 0.0
            for label in y_selection:
                raw_col = Y_OPTIONS_MAP.get(label)
                if raw_col and raw_col in df_table.columns:
                    mx_val = df_table[raw_col].abs().max()
                    if mx_val == 0:
                        mx_val = 1.0
                    df_table["total_score"] += (df_table[raw_col].abs() / mx_val) * 100

        # Unnötige Spalten entfernen
        df_display = df_table.drop(
            columns=["color_hex", "vol_t", "vol_b", "vol_h", "wandler_key"],
            errors="ignore",
        )

        # Tabelle mit Formatierung (Column Config) anzeigen
        st.dataframe(
            df_display,
            use_container_width=True,
            column_config={
                "err_nom": st.column_config.NumberColumn(
                    "Genauigkeit (Nenn) %",
                    help="Mittlerer Fehler im Nennbereich (80-100%)",
                    format="%.3f %%",  # 3 Nachkommastellen + Prozentzeichen
                ),
                "err_nieder": st.column_config.NumberColumn(
                    "Genauigkeit (Nieder) %", format="%.3f %%"
                ),
                "err_high": st.column_config.NumberColumn(
                    "Genauigkeit (Überlast) %", format="%.3f %%"
                ),
                "total_score": st.column_config.ProgressColumn(
                    "Performance Index",
                    help="Niedriger ist besser (Berechnet aus gewählten Metriken)",
                    format="%.1f",
                    min_value=0,
                    max_value=(
                        float(df_display["total_score"].max())
                        if not df_display.empty
                        else 100
                    ),
                ),
                "preis": st.column_config.NumberColumn("Preis", format="%.2f €"),
                "volumen": st.column_config.NumberColumn("Volumen", format="%.2f dm³"),
            },
        )

with tab3:
    st.markdown("### ⚙️ Stammdaten pro Messdatei")

    col_info, col_btn = st.columns([2, 1])
    with col_info:
        st.info("Die Datenbank ist die Hauptquelle für Hersteller, Modell & Geometrie.")
    with col_btn:
        if st.button(
            "🔄 Infos aus Dateinamen neu einlesen",
            type="secondary",
            use_container_width=True,
            help="Überschreibt Hersteller/Modell basierend auf dem Dateinamen",
        ):
            src_col = "raw_file" if "raw_file" in df.columns else "wandler_key"
            new_meta = df[src_col].apply(parse_filename_info)
            new_meta.columns = ["Hersteller", "Modell", "nennstrom", "Mess-Bürde"]
            df["Hersteller"] = new_meta["Hersteller"]
            df["Modell"] = new_meta["Modell"]
            df["nennstrom"] = new_meta["nennstrom"]
            df["Mess-Bürde"] = new_meta["Mess-Bürde"]
            if save_db(df):
                st.success("Datenbank aus Dateinamen aktualisiert!")
                st.rerun()

    st.markdown("---")

    # --- HIER IST DIE ÄNDERUNG ---
    # Checkbox lädt nun wirklich ALLES (df), nicht nur den ausgewählten Strom
    load_full_db = st.checkbox(
        "📋 Alle Messreihen der gesamten Datenbank laden (ignoriert alle Filter)",
        value=False,
    )

    if load_full_db:
        # Zeige ALLE Daten der Datenbank (df)
        df_editor_source = df.copy()
        st.caption(
            f"Zeige alle {len(df_editor_source['raw_file'].unique())} Dateien in der Datenbank."
        )
    else:
        # Zeige nur die aktuell gefilterten Daten (df_sub)
        df_editor_source = df_sub.copy()
        st.caption(
            f"Zeige {len(df_editor_source['raw_file'].unique())} Dateien (basierend auf Sidebar-Filter)."
        )

    # Vorbereitung der Tabelle (Duplikate entfernen, da DB pro Messpunkt speichert)
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
        num_rows="dynamic",  # Erlaubt Einfügen neuer Zeilen
        use_container_width=True,
    )

    if st.button("💾 Änderungen in DB speichern", type="primary"):
        changes = edited_df.to_dict(orient="index")
        df_to_save = df.copy()
        count = 0

        # Wir iterieren durch die Änderungen und wenden sie auf die Haupt-DB an
        for fname, attrs in changes.items():
            mask = df_to_save["raw_file"] == str(fname).strip()
            if mask.any():
                count += 1
                for c in META_COLS_EDIT:
                    if c in attrs:
                        df_to_save.loc[mask, c] = attrs[c]

        if count > 0:
            save_db(df_to_save)
            st.toast(f"{count} Dateien erfolgreich aktualisiert!", icon="✅")
            # Kurze Pause damit Toast sichtbar bleibt, dann Reload für Tabellen-Update
            import time

            time.sleep(1)
            st.rerun()
        else:
            st.warning("Keine übereinstimmenden Dateien zum Aktualisieren gefunden.")

# --- TAB 4: SELECTOR (DESIGN UPDATE) ---
with tab4:
    st.markdown("### ✂️ Manueller Rohdaten-Export")
    TARGET_LEVELS_T4 = [5, 20, 50, 80, 90, 100, 120]

    # Session State initialisieren
    for level in TARGET_LEVELS_T4:
        if f"s_{level}" not in st.session_state:
            st.session_state[f"s_{level}"] = 0
        if f"e_{level}" not in st.session_state:
            st.session_state[f"e_{level}"] = 0

    col_sel_nav, col_sel_main = st.columns([1, 3])
    all_files_t4 = get_files_tab4()
    df_status_t4 = load_status_tracking()

    # --- LINKE SPALTE: DATEI-NAVI ---
    with col_sel_nav:
        st.markdown("#### 📂 Dateiauswahl")
        files_options = []
        file_map = {}
        for f in all_files_t4:
            name_only = os.path.basename(f)
            clean_name_base, _ = extract_metadata(f)
            icon = "❌"
            if clean_name_base in df_status_t4.index:
                s = df_status_t4.loc[clean_name_base, "Status"]
                if s == "OK":
                    icon = "✅"
                elif s == "WARNING":
                    icon = "⚠️"
            disp = f"{icon} {name_only}"
            files_options.append(disp)
            file_map[disp] = f

        if not files_options:
            st.warning("Keine CSV Dateien.")
            st.stop()

        def on_change_file():
            for l in TARGET_LEVELS_T4:
                st.session_state[f"s_{l}"] = 0
                st.session_state[f"e_{l}"] = 0

        def nav_callback(direction):
            current_val = st.session_state.get("t4_file_sel")
            if current_val in files_options:
                curr_idx = files_options.index(current_val)
                new_idx = (
                    (curr_idx - 1) % len(files_options)
                    if direction == "prev"
                    else (curr_idx + 1) % len(files_options)
                )
                st.session_state["t4_file_sel"] = files_options[new_idx]
                on_change_file()

        def next_open_callback():
            current_val = st.session_state.get("t4_file_sel")
            start_idx = (
                (files_options.index(current_val) + 1) % len(files_options)
                if current_val in files_options
                else 0
            )
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

        sel_file_disp = st.selectbox(
            "Datei:", files_options, key="t4_file_sel", on_change=on_change_file
        )
        sel_file_path = file_map[sel_file_disp]

        c_prev, c_next = st.columns(2)
        c_prev.button(
            "⬅️ Zurück",
            key="btn_prev",
            on_click=nav_callback,
            args=("prev",),
            use_container_width=True,
        )
        c_next.button(
            "Weiter ➡️",
            key="btn_next",
            on_click=nav_callback,
            args=("next",),
            use_container_width=True,
        )

        st.markdown("---")
        st.button(
            "⏩ Nächste Offene suchen",
            type="secondary",
            use_container_width=True,
            on_click=next_open_callback,
        )

    # --- RECHTE SPALTE: HAUPTBEREICH ---
    with col_sel_main:
        if sel_file_path:
            clean_name_base, detected_nennstrom = extract_metadata(sel_file_path)

            # Zeiten laden
            all_times = load_time_configs()
            saved_times = all_times.get(clean_name_base, {})
            config_found = bool(saved_times)

            ranges_empty = all(
                st.session_state[f"s_{l}"] == 0 for l in TARGET_LEVELS_T4
            )
            if ranges_empty and config_found:
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

            # Header Bereich mit Datei-Infos
            with st.container(border=True):
                c_inf1, c_inf2, c_inf3 = st.columns([3, 1, 1])
                with c_inf1:
                    st.markdown(f"**Datei:** `{clean_name_base}`")
                    if config_found:
                        st.caption("✅ Zeiten aus Config geladen")
                    else:
                        st.caption("❌ Keine gespeicherten Zeiten")
                with c_inf2:
                    nennstrom_t4 = st.number_input(
                        "Nennstrom (A):",
                        value=float(detected_nennstrom),
                        key=f"ns_{clean_name_base}",
                    )
                with c_inf3:
                    if st.button("🔄 Reset Zeiten", use_container_width=True):
                        # Reload logic embedded in button
                        fresh_times = load_time_configs()
                        fresh_specific = fresh_times.get(clean_name_base, {})
                        if fresh_specific:
                            for lvl_str, vals in fresh_specific.items():
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

            t_idx, full_data_t4 = load_all_data_tab4(sel_file_path, detected)

            if t_idx:
                # Plot Bereich
                sel_ref_dev = st.selectbox(
                    "Referenz-Gerät für Vorschau (L1):", detected, key="ref_dev_t4"
                )

                ref_l1 = full_data_t4[sel_ref_dev]["L1"]
                if ref_l1 is not None:
                    fig_sel = go.Figure()
                    pct_val = (ref_l1 / nennstrom_t4) * 100
                    fig_sel.add_trace(
                        go.Scatter(
                            x=t_idx,
                            y=pct_val,
                            name=f"{sel_ref_dev} L1 %",
                            line=dict(color="#1f77b4", width=1),
                        )
                    )
                    # Markiere Bereiche
                    for level in TARGET_LEVELS_T4:
                        fig_sel.add_hline(
                            y=level, line_dash="dot", line_color="gray", opacity=0.3
                        )
                        s = st.session_state[f"s_{level}"]
                        e = st.session_state[f"e_{level}"]
                        if s > 0 and e > s:
                            fig_sel.add_vrect(
                                x0=s,
                                x1=e,
                                fillcolor="rgba(46, 204, 113, 0.2)",
                                line_width=0,
                            )
                            fig_sel.add_annotation(
                                x=(s + e) / 2,
                                y=level + 5,
                                text=f"<b>{level}%</b>",
                                showarrow=False,
                                font=dict(color="green", size=10),
                            )
                    fig_sel.update_layout(
                        height=350,
                        margin=dict(t=10, b=10, l=10, r=10),
                        yaxis_title="Last %",
                        xaxis_title="Datenpunkte",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_sel, use_container_width=True)

                st.markdown("##### 🎯 Bereiche definieren")

                # --- NEUES LAYOUT: GRID ---
                # Wir erstellen 4 Spalten für die Eingabefelder
                grid_cols = st.columns(4)

                for i, level in enumerate(TARGET_LEVELS_T4):
                    # Spalte auswählen (Index modulo 4)
                    col = grid_cols[i % 4]

                    with col:
                        # Container für bessere Optik (Rahmen um jedes Level)
                        with st.container(border=True):
                            st.markdown(f"**Stufe {level}%**")

                            # Start und Ende nebeneinander in diesem kleinen Container
                            c_start, c_end = st.columns(2)

                            with c_start:
                                st.number_input(
                                    "Start",
                                    key=f"s_{level}",
                                    min_value=0,
                                    label_visibility="collapsed",  # Label ausblenden, Platz sparen
                                )
                                st.caption("Start")  # Kleines Label drunter

                            with c_end:
                                st.number_input(
                                    "Ende",
                                    key=f"e_{level}",
                                    min_value=0,
                                    on_change=update_start_callback,
                                    args=(level,),
                                    label_visibility="collapsed",
                                )
                                st.caption("Ende")

                st.divider()

                # Footer: Export Controls
                c_exp1, c_exp2, c_exp3 = st.columns([2, 1, 1])
                with c_exp1:
                    exp_devs = st.multiselect(
                        "Geräte exportieren:", detected, default=detected
                    )
                with c_exp2:
                    skip_n = st.number_input(
                        "Einschwing-Skip:",
                        value=0,
                        help="Punkte am Anfang des Bereichs ignorieren",
                    )
                with c_exp3:
                    stat_opt = st.radio(
                        "Status setzen:", ["OK", "Problem"], horizontal=True
                    )

                if st.button(
                    "💾 Speichern & CSV Exportieren",
                    type="primary",
                    use_container_width=True,
                ):
                    if not exp_devs:
                        st.error("Bitte mindestens ein Gerät auswählen.")
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
                            st.toast(f"Gespeichert: {os.path.basename(out)}", icon="✅")
                            # Success Message kurz anzeigen
                            st.success(
                                "Daten wurden exportiert und Zeiten gespeichert."
                            )

# --- TAB 5: DB AGGREGATOR ---
with tab5:
    st.markdown("### 🔄 DB-Update (Smart Merge)")
    if st.button("🚀 Update starten", type="primary"):
        status_container = st.empty()
        progress_bar = st.progress(0)

        SEARCH_DIR_AGG = os.path.join(BASE_DIR, "messungen_sortiert")
        REF_KEYWORDS = ["pac1", "einspeisung", "ref", "source", "norm"]
        META_COLS_KEEP_AGG = [
            "Preis (€)",
            "Nennbürde (VA)",
            "T (mm)",
            "B (mm)",
            "H (mm)",
            "Kommentar",
            "Hersteller",
            "Modell",
            "Geometrie",
        ]

        status_container.info("Lese existierende Datenbank...")
        saved_metadata = aggregator_load_metadata(DATA_FILE, META_COLS_KEEP_AGG)
        files = glob.glob(
            os.path.join(SEARCH_DIR_AGG, "**", "*_sortiert.csv"), recursive=True
        )

        if not files:
            status_container.error("❌ Keine sortierten Dateien gefunden!")
        else:
            all_data = []
            for i, f in enumerate(files):
                progress_bar.progress((i + 1) / len(files))
                meta = aggregator_extract_metadata(f)
                stats, status_msg = aggregator_analyze_file(
                    f, meta, TARGET_LEVELS_T4, PHASES, REF_KEYWORDS
                )
                if stats:
                    filename_key = meta["dateiname"]
                    if filename_key in saved_metadata:
                        for stat_entry in stats:
                            stat_entry.update(saved_metadata[filename_key])
                    all_data.extend(stats)

            if all_data:
                status_container.info("Speichere Datenbank...")
                df_all = pd.DataFrame(all_data)
                for col in META_COLS_KEEP_AGG:
                    if col not in df_all.columns:
                        df_all[col] = "" if col == "Kommentar" else 0.0
                df_clean = df_all.drop_duplicates(
                    subset=[
                        "raw_file",
                        "dut_name",
                        "phase",
                        "target_load",
                        "comparison_mode",
                    ],
                    keep="last",
                )
                if save_db(df_clean):
                    status_container.success(
                        f"✅ Update fertig! ({len(df_clean)} Einträge)"
                    )
                else:
                    status_container.error("Fehler beim Speichern.")
            else:
                status_container.error("Keine gültigen Daten.")


# =======================================================
# --- NEUE HELPER-FUNKTION FÜR BATCH-FILTERUNG ---
# =======================================================
def get_df_subset_from_config(full_df, conf_data):
    """
    Erzeugt df_sub basierend auf einer gespeicherten Konfiguration,
    ohne den Session-State zu verändern.
    """
    # 1. Listen laden
    c_list = conf_data.get("current", [])
    g_list = conf_data.get("geos", [])
    w_list = conf_data.get("wandlers", [])
    d_list = conf_data.get("duts", [])
    comp_val = (
        "device_ref" if "Messgerät" in conf_data.get("comp_mode", "") else "nominal_ref"
    )

    # 2. Filtern
    mask = (
        (full_df["nennstrom"].isin(c_list))
        & (full_df["Geometrie"].isin(g_list))
        & (full_df["wandler_key"].isin(w_list))
        & (full_df["dut_name"].isin(d_list))
    )
    if "comparison_mode" in full_df.columns:
        mask = mask & (full_df["comparison_mode"] == comp_val)

    sub = full_df[mask].copy()

    if sub.empty:
        return sub

    # 3. Berechnungen / IDs (Gleiche Logik wie im Hauptteil)
    if comp_val == "device_ref" and "ref_name" in sub.columns:
        sub = sub[sub["dut_name"] != sub["ref_name"]]

    sub["unique_id"] = sub["raw_file"] + " | " + sub["dut_name"].astype(str)
    sub["err_ratio"] = (
        (sub["val_dut_mean"] - sub["val_ref_mean"]) / sub["val_ref_mean"]
    ) * 100
    sub["err_std"] = (sub["val_dut_std"] / sub["val_ref_mean"]) * 100

    # 4. Styling anwenden (Legenden, Farben etc. aus Config in DF schreiben)
    legs = conf_data.get("custom_legends", {})
    cols = conf_data.get("custom_colors", {})
    stys = conf_data.get("custom_styles", {})
    syms = conf_data.get("custom_symbols", {})
    wids = conf_data.get("custom_widths", {})
    viss = conf_data.get("custom_visible", {})
    sizs = conf_data.get("custom_sizes", {})

    # Fallback-Logik für Farben
    def get_col(uid):
        if uid in cols:
            return cols[uid]
        return "#000000"

    sub["final_legend"] = sub["unique_id"].map(lambda x: legs.get(x, x))
    sub["final_color"] = sub["unique_id"].map(get_col)
    sub["final_style"] = sub["unique_id"].map(lambda x: stys.get(x, "solid"))
    sub["final_symbol"] = sub["unique_id"].map(lambda x: syms.get(x, "circle"))
    sub["final_width"] = sub["unique_id"].map(lambda x: wids.get(x, 2.5))
    sub["final_visible"] = sub["unique_id"].map(lambda x: viss.get(x, True))
    sub["final_size"] = sub["unique_id"].map(lambda x: sizs.get(x, 8))

    return sub


# =======================================================
# --- EXPORT SIDEBAR (BATCH MODUS) ---
# =======================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Batch PDF Export")

# 1. Auswahl: Welche Configs?
all_saved_conf_names = sorted(list(dashboard_configs.keys()))
selected_batch_configs = st.sidebar.multiselect(
    "1. Konfigurationen wählen:",
    all_saved_conf_names,
    default=(
        all_saved_conf_names
        if len(all_saved_conf_names) < 3
        else all_saved_conf_names[:1]
    ),
    help="Jede gewählte Config erzeugt einen eigenen Unterordner im ZIP.",
)

# 2. Auswahl: Welche Diagramme?
export_opts = [
    "Gesamtübersicht (Tab 1)",
    "Detail-Phasen (Tab 1)",
    "Ökonomie: Performance-Index",  # Ranking Tabelle
    "Ökonomie: Scatter-Plot",
    "Ökonomie: Heatmap",
    "Ökonomie: Boxplot",
    "Ökonomie: Pareto",
    "Ökonomie: Radar",
]
export_selection = st.sidebar.multiselect(
    "2. Diagramm-Typen wählen:",
    export_opts,
    default=["Gesamtübersicht (Tab 1)", "Ökonomie: Performance-Index"],
)

trigger_export_btn = st.sidebar.button("🔄 Batch-Export starten", type="primary")

# =======================================================
# --- EXECUTE BATCH EXPORT ---
# =======================================================
if trigger_export_btn:
    if not selected_batch_configs:
        st.error("Bitte mindestens eine Konfiguration auswählen.")
    elif not export_selection:
        st.error("Bitte mindestens einen Diagramm-Typ auswählen.")
    else:
        # Puffer für das ZIP
        zip_buffer = io.BytesIO()
        progress_text = st.empty()
        prog_bar = st.progress(0)

        # --- HILFSFUNKTIONEN (Innerhalb Scope) ---
        def clean_tex_and_break(text):
            text = str(text)
            symbol_map = {"Ω": r"$\Omega$", "µ": r"$\mu$", "²": r"$^2$", "³": r"$^3$"}
            for char, repl in symbol_map.items():
                text = text.replace(char, repl)
            special_chars = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}
            for char, repl in special_chars.items():
                if char in text:
                    text = text.replace(char, repl)

            parts = text.split("|")
            clean_parts = [p.strip() for p in parts]

            # Umbruch nach dem 2. Element (meist Strom), Rest danach
            if len(clean_parts) >= 3:
                line1 = " | ".join(clean_parts[:2])
                line2 = " | ".join(clean_parts[2:])
                return f"{line1} \\newline {line2}"
            return " | ".join(clean_parts)

        with st.spinner("Erstelle Batch-Export..."):
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:

                total_steps = len(selected_batch_configs)

                # --- HAUPTSCHLEIFE ÜBER ALLE CONFIGS ---
                for i, conf_name in enumerate(selected_batch_configs):
                    prog_bar.progress((i) / total_steps)
                    progress_text.text(f"Verarbeite Config: {conf_name}...")

                    # 1. Config laden & Daten filtern
                    conf_data = dashboard_configs[conf_name]
                    df_batch = get_df_subset_from_config(df, conf_data)

                    if df_batch.empty:
                        continue

                    safe_folder = sanitize_filename(conf_name)

                    # Parameter aus Config extrahieren
                    b_acc_class = conf_data.get("acc_class", 0.2)
                    b_ylim = conf_data.get("y_limit", 1.5)
                    b_yshift = conf_data.get("y_shift", 0.0)
                    b_bottom_mode = conf_data.get(
                        "bottom_plot_mode", "Standardabweichung"
                    )
                    b_show_err = conf_data.get("show_err_bars", True)
                    b_titles = conf_data.get("custom_titles", {})

                    curr_list = sorted(conf_data.get("current", []))
                    curr_str = ", ".join([str(int(c)) for c in curr_list]) + " A"

                    # --- A) EXPORT TAB 1: ÜBERSICHT ---
                    if "Gesamtübersicht (Tab 1)" in export_selection:
                        use_single_row_b = b_bottom_mode == "Ausblenden"
                        if use_single_row_b:
                            fig_main = make_subplots(
                                rows=1, cols=3, shared_xaxes=True, subplot_titles=PHASES
                            )
                        else:
                            fig_main = make_subplots(
                                rows=2,
                                cols=3,
                                shared_xaxes=True,
                                vertical_spacing=0.08,
                                row_heights=[0.7, 0.3],
                                subplot_titles=PHASES,
                            )

                        lim_x, lim_y_p, lim_y_n = get_trumpet_limits(b_acc_class)

                        for col_idx, phase in enumerate(PHASES, start=1):
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

                            p_data = df_batch[df_batch["phase"] == phase]
                            for uid, group in p_data.groupby("unique_id"):
                                group = group.sort_values("target_load")
                                rf = group.iloc[0]
                                if not rf["final_visible"]:
                                    continue

                                fig_main.add_trace(
                                    go.Scatter(
                                        x=group["target_load"],
                                        y=group["err_ratio"],
                                        mode="lines+markers",
                                        name=rf["final_legend"],
                                        line=dict(
                                            color=rf["final_color"],
                                            width=rf["final_width"],
                                            dash=rf["final_style"],
                                        ),
                                        marker=dict(
                                            size=rf["final_size"],
                                            symbol=rf["final_symbol"],
                                        ),
                                        legendgroup=rf["final_legend"],
                                        showlegend=(col_idx == 1),
                                    ),
                                    row=1,
                                    col=col_idx,
                                )

                                if not use_single_row_b:
                                    if b_bottom_mode == "Standardabweichung":
                                        if b_show_err:
                                            fig_main.add_trace(
                                                go.Bar(
                                                    x=group["target_load"],
                                                    y=group["err_std"],
                                                    marker_color=rf["final_color"],
                                                    legendgroup=rf["final_legend"],
                                                    showlegend=False,
                                                ),
                                                row=2,
                                                col=col_idx,
                                            )
                                    elif b_bottom_mode == "Messwert (Absolut)":
                                        fig_main.add_trace(
                                            go.Scatter(
                                                x=group["target_load"],
                                                y=group["val_dut_mean"],
                                                mode="lines+markers",
                                                line=dict(
                                                    color=rf["final_color"],
                                                    width=1.5,
                                                    dash="dot",
                                                ),
                                                marker=dict(
                                                    symbol="x", size=rf["final_size"]
                                                ),
                                                legendgroup=rf["final_legend"],
                                                showlegend=False,
                                            ),
                                            row=2,
                                            col=col_idx,
                                        )

                        t_title = b_titles.get(
                            "Gesamtübersicht (Tab 1)", f"Gesamtübersicht: {curr_str}"
                        )
                        fig_main.update_layout(
                            title=t_title,
                            template="plotly_white",
                            height=800,
                            width=1123,
                            font=dict(family="Serif", size=14, color="black"),
                            legend=dict(
                                orientation="h", y=-0.15, x=0.5, xanchor="center"
                            ),
                        )

                        y_min = -b_ylim + b_yshift
                        y_max = b_ylim + b_yshift
                        fig_main.update_yaxes(range=[y_min, y_max], row=1)

                        zf.writestr(
                            f"{safe_folder}/{safe_folder}-Zusammenfassung_MultiCurrent.pdf",
                            fig_main.to_image(format="pdf"),
                        )

                    # --- B) EXPORT TAB 1: DETAILS ---
                    if "Detail-Phasen (Tab 1)" in export_selection:
                        for ph in PHASES:
                            fig_s = create_single_phase_figure(
                                df_batch,
                                ph,
                                b_acc_class,
                                b_ylim,
                                b_yshift,
                                b_bottom_mode,
                                b_show_err,
                                title_prefix=f"{curr_str}",
                            )
                            zf.writestr(
                                f"{safe_folder}/{safe_folder}-Detail_{ph}_MultiCurrent.pdf",
                                fig_s.to_image(format="pdf", width=1123, height=794),
                            )

                    # --- C) OEKONOMIE BERECHNUNG ---
                    eco_req = [x for x in export_selection if "Ökonomie" in x]

                    if eco_req and not df_batch.empty:
                        # Aggregation
                        df_agg = (
                            df_batch.groupby("unique_id")
                            .agg(
                                wandler_key=("wandler_key", "first"),
                                legend_name=("final_legend", "first"),
                                err_nieder=(
                                    "err_ratio",
                                    lambda x: x[
                                        df_batch.loc[x.index, "target_load"].isin(
                                            ZONES["Niederstrom (5-50%)"]
                                        )
                                    ]
                                    .abs()
                                    .mean(),
                                ),
                                err_nom=(
                                    "err_ratio",
                                    lambda x: x[
                                        df_batch.loc[x.index, "target_load"].isin(
                                            ZONES["Nennstrom (80-100%)"]
                                        )
                                    ]
                                    .abs()
                                    .mean(),
                                ),
                                err_high=(
                                    "err_ratio",
                                    lambda x: x[
                                        df_batch.loc[x.index, "target_load"].isin(
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
                        df_agg["volumen"] = (
                            df_agg["vol_t"] * df_agg["vol_b"] * df_agg["vol_h"]
                        ) / 1000.0

                        k_eco_x = conf_data.get("eco_x", "Preis (€)")
                        k_eco_y = conf_data.get("eco_y", ["Fehler Nennstrom"])

                        Y_OPT_EXP = {
                            "Fehler Niederstrom": "err_nieder",
                            "Fehler Nennstrom": "err_nom",
                            "Fehler Überlast": "err_high",
                            "Preis (€)": "preis",
                            "Volumen (Gesamt)": "volumen",
                            "Breite (B)": "vol_b",
                            "Höhe (H)": "vol_h",
                            "Tiefe (T)": "vol_t",
                        }
                        REV_Y_EXP = {v: k for k, v in Y_OPT_EXP.items()}

                        sel_y_cols = [Y_OPT_EXP[k] for k in k_eco_y if k in Y_OPT_EXP]
                        col_x_exp = "preis" if "Preis" in k_eco_x else "volumen"
                        col_map = dict(zip(df_agg["legend_name"], df_agg["color_hex"]))

                        # --- C1) SCATTER ---
                        if "Ökonomie: Scatter-Plot" in export_selection:
                            t_scat = b_titles.get(
                                "Scatter-Plot", f"{k_eco_x} vs. Auswahl"
                            )
                            df_long = df_agg.melt(
                                id_vars=["legend_name", col_x_exp, "color_hex"],
                                value_vars=sel_y_cols,
                                value_name="Wert",
                            )
                            df_long["Metrik"] = df_long["variable"].map(REV_Y_EXP)

                            fig_eco = px.scatter(
                                df_long,
                                x=col_x_exp,
                                y="Wert",
                                color="legend_name",
                                symbol="Metrik",
                                size=[15] * len(df_long),
                                color_discrete_map=col_map,
                                title=t_scat,
                            )
                            fig_eco.update_layout(
                                template="plotly_white",
                                width=1123,
                                height=794,
                                font=dict(family="Serif", size=14),
                                legend=dict(
                                    orientation="h", y=-0.15, x=0.5, xanchor="center"
                                ),
                            )
                            zf.writestr(
                                f"{safe_folder}/{safe_folder}-Oekonomie_Scatter.pdf",
                                fig_eco.to_image(format="pdf"),
                            )

                        # --- C2) RANKING (Performance Index) ---
                        if "Ökonomie: Performance-Index" in export_selection:
                            t_rank = b_titles.get("Performance-Index", "Ranking")
                            df_rank = df_agg.copy()
                            df_rank["total_score"] = 0.0
                            
                            # --- 1. FARBEN & REIHENFOLGE DEFINIEREN ---
                            # Anpassung: Preis auf Grün geändert für besseren Kontrast zu Braun
                            ECO_COLORS_EXPORT = {
                                "Preis (€)": "#2ca02c",             # Neu: Grün (statt Violett)
                                "Volumen (Gesamt)": "#8c564b",      # Braun
                                "Fehler Niederstrom": "#000080",    # Dunkelblau
                                "Fehler Nennstrom": "#ffbf00",      # Orange (Amber)
                                "Fehler Überlast": "#ff0000"        # Rot
                            }
                            
                            # Die Reihenfolge der Stacks im Balken
                            ECO_ORDER_EXPORT = [
                                "Preis (€)", 
                                "Volumen (Gesamt)", 
                                "Fehler Niederstrom", 
                                "Fehler Nennstrom", 
                                "Fehler Überlast"
                            ]

                            norm_cols = []
                            # Daten normalisieren und Score berechnen
                            for k in k_eco_y:
                                if k in Y_OPT_EXP:
                                    rc = Y_OPT_EXP[k]
                                    mx = df_rank[rc].abs().max()
                                    if mx == 0:
                                        mx = 1
                                    df_rank[k] = (df_rank[rc].abs() / mx) * 100
                                    df_rank["total_score"] += df_rank[k]
                                    norm_cols.append(k)

                            # Sortieren nach Gesamt-Score (Beste oben)
                            df_rank = df_rank.sort_values("total_score", ascending=True)

                            # Daten für Plotly aufbereiten (Wide -> Long Format)
                            df_l = df_rank.melt(
                                id_vars=["legend_name"],
                                value_vars=norm_cols,
                                value_name="Anteil",
                                var_name="Kategorie" # Wichtig für die Legende
                            )

                            # --- 2. PLOT ERSTELLEN (Mit Custom Colors & Style) ---
                            fig_bar = px.bar(
                                df_l,
                                y="legend_name",
                                x="Anteil",
                                color="Kategorie",
                                orientation="h",
                                title=t_rank,
                                # Hier werden deine Farben und die Reihenfolge erzwungen:
                                color_discrete_map=ECO_COLORS_EXPORT,
                                category_orders={"Kategorie": ECO_ORDER_EXPORT}
                            )
                            
                            # --- 3. LAYOUT ANPASSEN (Schriftgröße & Balkendicke) ---
                            fig_bar.update_layout(
                                yaxis=dict(
                                    autorange="reversed",
                                    # Schriftgröße der Wandler-Namen (Y-Achse) vergrößern:
                                    tickfont=dict(size=10, family="Arial"),
                                    title=None # "legend_name" Label ausblenden für mehr Platz
                                ),
                                xaxis=dict(
                                    title="Anteil am Score (%)",
                                    tickfont=dict(size=10)
                                ),
                                template="plotly_white",
                                width=1123,
                                height=794,
                                # bargap vergrößern macht die Balken dünner (0.4 = 40% Platz zwischen Balken)
                                bargap=0.4, 
                                legend=dict(
                                    orientation="h", 
                                    y=-0.15, 
                                    x=0.5, 
                                    xanchor="center",
                                    font=dict(size=12, family="Arial Black") # Legende etwas fetter
                                ),
                                margin=dict(l=200) # Mehr Platz links für lange Namen
                            )
                            
                            # PDF speichern
                            zf.writestr(
                                f"{safe_folder}/{safe_folder}-Oekonomie_Ranking.pdf",
                                fig_bar.to_image(format="pdf"),
                            )

                            # LATEX Export (unverändert, nutzt Daten)
                            ltx = []
                            ltx.append(r"\begin{table}[H]")
                            ltx.append(r"    \centering")
                            ltx.append(rf"    \caption{{{t_rank}}}")
                            ltx.append(
                                rf"    \label{{tab:{sanitize_filename(conf_name)}_ranking}}"
                            )
                            col_def = "p{6cm}" + "c" * (len(norm_cols) + 1)
                            ltx.append(rf"    \begin{{tabular}}{{{col_def}}}")
                            ltx.append(r"        \toprule")

                            h_cells = [r"\textbf{Messsystem}"]
                            for cn in norm_cols:
                                cc = (
                                    cn.replace("%", r"\%")
                                    .replace("_", r"\_")
                                    .replace("Ω", r"$\Omega$")
                                    .replace(" ", r" \\ ")
                                )
                                h_cells.append(rf"\textbf{{\shortstack[c]{{{cc}}}}}")
                            h_cells.append(
                                r"\textbf{\shortstack[c]{Fehler-Score \\ {[\%]}}}"
                            )

                            ltx.append("        " + " & ".join(h_cells) + r" \\")
                            ltx.append(r"        \midrule")

                            for _, r_row in df_rank.iterrows():
                                n_cl = clean_tex_and_break(r_row["legend_name"])
                                r_c = [n_cl]
                                for cn in norm_cols:
                                    r_c.append(f"{r_row[cn]:.2f}".replace(".", ","))
                                r_c.append(
                                    f"{r_row['total_score']:.2f}".replace(".", ",")
                                )
                                ltx.append("        " + " & ".join(r_c) + r" \\")

                            ltx.append(r"        \bottomrule")
                            ltx.append(r"    \end{tabular}")
                            ltx.append(r"\end{table}")
                            zf.writestr(
                                f"{safe_folder}/{safe_folder}-Oekonomie_Ranking.tex",
                                "\n".join(ltx),
                            )
                        # --- C3) HEATMAP ---
                        if "Ökonomie: Heatmap" in export_selection:
                            t_heat = b_titles.get("Heatmap", "Heatmap")
                            df_l = df_agg.melt(
                                id_vars=["legend_name"],
                                value_vars=sel_y_cols,
                                value_name="Wert",
                            )
                            df_l["Kat"] = df_l["variable"].map(REV_Y_EXP)
                            fh = px.density_heatmap(
                                df_l,
                                x="legend_name",
                                y="Kat",
                                z="Wert",
                                text_auto=True,
                                title=t_heat,
                                color_continuous_scale="Blues",
                            )
                            fh.update_layout(
                                template="plotly_white", width=1123, height=794
                            )
                            zf.writestr(
                                f"{safe_folder}/{safe_folder}-Oekonomie_Heatmap.pdf",
                                fh.to_image(format="pdf"),
                            )

                        # --- C4) BOXPLOT ---
                        if "Ökonomie: Boxplot" in export_selection:
                            t_box = b_titles.get("Boxplot", "Boxplot")
                            df_l = df_agg.melt(
                                id_vars=["legend_name"],
                                value_vars=sel_y_cols,
                                value_name="Wert",
                            )
                            df_l["Kat"] = df_l["variable"].map(REV_Y_EXP)
                            fb = px.box(
                                df_l,
                                x="legend_name",
                                y="Wert",
                                color="Kat",
                                title=t_box,
                            )
                            fb.update_layout(
                                template="plotly_white",
                                width=1123,
                                height=794,
                                legend=dict(orientation="h", y=-0.15),
                            )
                            zf.writestr(
                                f"{safe_folder}/{safe_folder}-Oekonomie_Boxplot.pdf",
                                fb.to_image(format="pdf"),
                            )

                        # --- C5) PARETO ---
                        if "Ökonomie: Pareto" in export_selection and sel_y_cols:
                            t_par = b_titles.get("Pareto", "Pareto")
                            ty = sel_y_cols[0]
                            tl = k_eco_y[0]
                            dfs = df_agg.sort_values(by=ty, ascending=False)
                            dfs["cum"] = dfs[ty].cumsum() / dfs[ty].sum() * 100
                            fp = make_subplots(specs=[[{"secondary_y": True}]])
                            fp.add_trace(
                                go.Bar(
                                    x=dfs["legend_name"],
                                    y=dfs[ty],
                                    name=tl,
                                    marker_color=dfs["color_hex"],
                                ),
                                secondary_y=False,
                            )
                            fp.add_trace(
                                go.Scatter(
                                    x=dfs["legend_name"],
                                    y=dfs["cum"],
                                    name="Cum %",
                                    mode="lines+markers",
                                    line=dict(color="red"),
                                ),
                                secondary_y=True,
                            )
                            fp.update_layout(
                                title=t_par,
                                template="plotly_white",
                                width=1123,
                                height=794,
                                legend=dict(orientation="h", y=-0.15),
                            )
                            zf.writestr(
                                f"{safe_folder}/{safe_folder}-Oekonomie_Pareto.pdf",
                                fp.to_image(format="pdf"),
                            )

                        # --- C6) RADAR ---
                        if "Ökonomie: Radar" in export_selection and sel_y_cols:
                            t_rad = b_titles.get("Radar", "Radar")
                            fr = go.Figure()
                            mxv = {
                                cn: (df_agg[cn].max() if df_agg[cn].max() != 0 else 1)
                                for cn in sel_y_cols
                            }
                            for _, r_row in df_agg.iterrows():
                                rvals = [(r_row[c] / mxv[c]) for c in sel_y_cols]
                                rvals.append(rvals[0])
                                fr.add_trace(
                                    go.Scatterpolar(
                                        r=rvals,
                                        theta=k_eco_y + [k_eco_y[0]],
                                        fill="toself",
                                        name=r_row["legend_name"],
                                        line_color=r_row["color_hex"],
                                    )
                                )
                            fr.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                title=t_rad,
                                template="plotly_white",
                                width=1123,
                                height=794,
                                legend=dict(orientation="h", y=-0.15),
                            )
                            zf.writestr(
                                f"{safe_folder}/{safe_folder}-Oekonomie_Radar.pdf",
                                fr.to_image(format="pdf"),
                            )

                prog_bar.progress(1.0)
                progress_text.text("Fertig!")

        st.session_state["zip_data"] = zip_buffer.getvalue()
        st.session_state["zip_name"] = "Batch_Export.zip"
        st.success("✅ Batch-Export bereit!")

if "zip_data" in st.session_state:
    st.sidebar.download_button(
        "💾 Download Batch ZIP",
        st.session_state["zip_data"],
        st.session_state["zip_name"],
        "application/zip",
    )

# =======================================================
# --- SPEICHER-LOGIK ---
# =======================================================
if st.session_state.get("trigger_save", False):
    current_colors = (
        map_color
        if "map_color" in locals()
        else st.session_state.get("loaded_colors", {})
    )
    current_legends = (
        map_legend
        if "map_legend" in locals()
        else st.session_state.get("loaded_legends", {})
    )
    current_styles = (
        map_style
        if "map_style" in locals()
        else st.session_state.get("loaded_styles", {})
    )
    current_symbols = (
        map_symbol
        if "map_symbol" in locals()
        else st.session_state.get("loaded_symbols", {})
    )
    current_widths = (
        map_width
        if "map_width" in locals()
        else st.session_state.get("loaded_widths", {})
    )
    current_visible = (
        map_visible
        if "map_visible" in locals()
        else st.session_state.get("loaded_visible", {})
    )
    current_sizes = (
        map_size if "map_size" in locals() else st.session_state.get("loaded_sizes", {})
    )
    current_titles = (
        TITLES_MAP
        if "TITLES_MAP" in locals()
        else st.session_state.get("loaded_titles", {})
    )

    snapshot_data = {
        "current": st.session_state.get("k_current", []),
        "geos": st.session_state.get("k_geos", []),
        "wandlers": st.session_state.get("k_wandlers", []),
        "duts": st.session_state.get("k_duts", []),
        "comp_mode": st.session_state.get("k_comp", "Messgerät (z.B. PAC1)"),
        "sync_axes": st.session_state.get("k_sync", True),
        "y_limit": st.session_state.get("k_ylim", 1.5),
        "y_shift": st.session_state.get("k_yshift", 0.0),
        "acc_class": st.session_state.get("k_class", 0.2),
        "show_err_bars": st.session_state.get("k_errbars", True),
        "bottom_plot_mode": st.session_state.get("k_bottom_mode", "Standardabweichung"),
        "custom_colors": current_colors,
        "custom_legends": current_legends,
        "custom_styles": current_styles,
        "custom_symbols": current_symbols,
        "custom_widths": current_widths,
        "custom_visible": current_visible,
        "custom_sizes": current_sizes,
        "custom_titles": current_titles,
        "eco_x": st.session_state.get("k_eco_x", "Preis (€)"),
        "eco_y": st.session_state.get("k_eco_y", ["Fehler Nennstrom"]),
        "eco_type": st.session_state.get("k_eco_type", "Scatter"),
    }

    save_name = st.session_state["save_name"]
    save_dashboard_config(save_name, snapshot_data)
    st.session_state["trigger_save"] = False

    st.session_state["loaded_colors"] = current_colors
    st.session_state["loaded_legends"] = current_legends

    st.toast(f"Konfiguration '{save_name}' erfolgreich gespeichert!", icon="💾")
    st.rerun()
