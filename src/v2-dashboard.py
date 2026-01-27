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

# --- KONFIGURATION ---
DATA_FILE = "messdaten_db.parquet"
SPECS_FILE = "wandler_specs.csv"
PHASES = ["L1", "L2", "L3"]
WORK_DIR = "matlab_working_dir"
DEFAULT_MATLAB_PATH = r"C:\Program Files\MATLAB\R2025a\bin\matlab.exe"

# Fehler-Bereiche Definition für Ökonomie
ZONES = {
    "Niederstrom (5-50%)": [5, 20, 50],
    "Nennstrom (80-100%)": [80, 90, 100],
    "Überlast (≥120%)": [120, 150, 200],
}

# --- FARBPALETTEN ---
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

# --- MATLAB SKRIPT TEMPLATE ---
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
        
        % Plot 1: Fehler
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
        title(ax1, sprintf('Fehlerverlauf - Phase %s (%d A)', p, nennstrom));
        ylabel(ax1, 'Fehler [%]'); ylim(ax1, [- Y_LIMIT_PH, Y_LIMIT_PH]); xlim(ax1, [0, 125]);
        legend(ax1, 'Location', 'southoutside', 'Orientation', 'horizontal', 'NumColumns', 2, 'Interpreter', 'tex');
        
        % Plot 2: StdAbw
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
        exportgraphics(f, sprintf('Detail_%s_%dA.pdf', p, nennstrom), 'ContentType', 'vector');
        close(f);
    end
catch ME; disp(ME.message); exit(1); end
exit(0);
"""

st.set_page_config(page_title="Wandler Dashboard", layout="wide", page_icon="📈")


# --- HILFSFUNKTIONEN ---


@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_parquet(DATA_FILE)
    if "dut_name" in df.columns:
        df["trace_id"] = df["folder"] + " | " + df["dut_name"].astype(str)
    else:
        df["trace_id"] = df["folder"]
    if "target_load" in df.columns:
        df["target_load"] = pd.to_numeric(df["target_load"], errors="coerce")

    # BASIS-TYP GENERIEREN (für Specs)
    # Wir nehmen an: wandler_key enthält den vollen Namen. Wir strippen die Bürde.
    df["base_type"] = df["wandler_key"].apply(extract_base_type)

    return df


def extract_base_type(wandler_key):
    """
    Entfernt Bürden-Infos (z.B. 8R1, 0R5) und irrelevante Suffixe aus dem Key.
    Ziel: Identische Wandler (Hersteller + Typ) gruppieren.
    """
    s = str(wandler_key)
    # Splitte bei typischen Trennern
    tokens = re.split(r"[_\s\-]+", s)

    clean_tokens = []
    for t in tokens:
        # Filter: Bürde (Zahl + R + Zahl, z.B. 8R1, 10R)
        if re.match(r"^\d+R\d*$", t, re.IGNORECASE):
            continue
        # Filter: Parallel/Dreieck/Messstrecke Infos, falls die als "Typ" egal sind
        if t.lower() in ["parallel", "dreieck", "messstrecke", "l1", "l2", "l3"]:
            continue
        # Filter: Leere Tokens
        if not t:
            continue
        clean_tokens.append(t)

    return " ".join(clean_tokens)


def load_specs(all_base_types):
    """
    Lädt Specs basierend auf 'base_type' (Bereinigter Typ).
    """
    if os.path.exists(SPECS_FILE):
        df_specs = pd.read_csv(SPECS_FILE)
        # Migration falls nötig (Spalte umbenennen)
        if "base_type" not in df_specs.columns and "wandler_key" in df_specs.columns:
            st.toast("Migriere Specs auf Basis-Typ...", icon="🔄")
            df_specs.rename(columns={"wandler_key": "base_type"}, inplace=True)
        elif "base_type" not in df_specs.columns and "dut_name" in df_specs.columns:
            df_specs.rename(columns={"dut_name": "base_type"}, inplace=True)

        df_specs["base_type"] = df_specs["base_type"].astype(str)
    else:
        df_specs = pd.DataFrame(
            columns=[
                "base_type",
                "Preis (€)",
                "L (mm)",
                "B (mm)",
                "H (mm)",
                "Kommentar",
            ]
        )

    existing_keys = df_specs["base_type"].tolist()
    new_rows = []
    for k in all_base_types:
        if str(k) not in existing_keys:
            new_rows.append(
                {
                    "base_type": str(k),
                    "Preis (€)": 0.0,
                    "L (mm)": 0.0,
                    "B (mm)": 0.0,
                    "H (mm)": 0.0,
                    "Kommentar": "",
                }
            )

    if new_rows:
        df_specs = pd.concat([df_specs, pd.DataFrame(new_rows)], ignore_index=True)
    return df_specs


def calculate_zone_errors(df_filtered):
    """
    Berechnet Fehler pro Zone (Nieder, Nenn, Überlast) aggregiert pro WANDLER_KEY (Messung).
    Aber wir behalten 'base_type' bei, um später zu mergen.
    """
    results = []
    if "abs_error_pct" not in df_filtered.columns:
        df_filtered["abs_error_pct"] = df_filtered["err_ratio"].abs()

    # Wir gruppieren nach wandler_key (damit wir im Plot noch die Punkte sehen),
    # aber merken uns den base_type
    for w_key, group in df_filtered.groupby("wandler_key"):

        base_t = group["base_type"].iloc[0]  # Sollte identisch sein in der Gruppe

        mask_low = group["target_load"].isin(ZONES["Niederstrom (5-50%)"])
        err_low = group.loc[mask_low, "abs_error_pct"].mean()

        mask_nom = group["target_load"].isin(ZONES["Nennstrom (80-100%)"])
        err_nom = group.loc[mask_nom, "abs_error_pct"].mean()

        mask_high = group["target_load"] >= 120
        err_high = group.loc[mask_high, "abs_error_pct"].mean()

        results.append(
            {
                "wandler_key": w_key,
                "base_type": base_t,
                "Fehler Niederstrom (%)": err_low,
                "Fehler Nennstrom (%)": err_nom,
                "Fehler Überlast (%)": err_high,
            }
        )
    return pd.DataFrame(results)


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
            burden_part = f"{t.replace('R', ',')} Ω"
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
    return base_name


def create_single_phase_figure(
    df_sub, phase, acc_class, y_limit, show_err_bars, title_prefix=""
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"Fehlerverlauf {phase}", f"Standardabweichung {phase}"),
    )
    lim_x, lim_y_p, lim_y_n = get_trumpet_limits(acc_class)
    fig.add_trace(
        go.Scatter(
            x=lim_x,
            y=lim_y_p,
            mode="lines",
            line=dict(color="black", width=1, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=lim_x,
            y=lim_y_n,
            mode="lines",
            line=dict(color="black", width=1, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    phase_data = df_sub[df_sub["phase"] == phase]
    for uid, group in phase_data.groupby("unique_id"):
        group = group.sort_values("target_load")
        leg_name = group.iloc[0]["final_legend"]
        color = group.iloc[0]["final_color"]
        fig.add_trace(
            go.Scatter(
                x=group["target_load"],
                y=group["err_ratio"],
                mode="lines+markers",
                name=leg_name,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                legendgroup=leg_name,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
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

    fig.update_layout(
        title=f"{title_prefix} - Phase {phase}",
        template="plotly_white",
        height=800,
        width=1100,
        legend=dict(orientation="h", y=-0.15),
    )
    fig.update_yaxes(range=[-y_limit, y_limit], title_text="Fehler [%]", row=1, col=1)
    return fig


def ensure_working_dir():
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)
    return os.path.abspath(WORK_DIR)


def clear_cache():
    if "zip_data" in st.session_state:
        del st.session_state["zip_data"]


# --- APP START ---
df = load_data()
if df is None:
    st.error(f"⚠️ Datei '{DATA_FILE}' fehlt.")
    st.stop()

# --- SIDEBAR: GLOBAL FILTER ---
st.sidebar.header("🎛️ Globale Filter")

available_currents = sorted(df["nennstrom"].unique())
sel_current = st.sidebar.selectbox(
    "1. Nennstrom:",
    available_currents,
    format_func=lambda x: f"{int(x)} A",
    on_change=clear_cache,
)

df_curr = df[df["nennstrom"] == sel_current]
available_wandlers = sorted(df_curr["wandler_key"].unique())
sel_wandlers = st.sidebar.multiselect(
    "2. Wandler / Messung:",
    available_wandlers,
    default=available_wandlers,
    on_change=clear_cache,
)

if not sel_wandlers:
    st.info("Bitte mindestens einen Wandler auswählen.")
    st.stop()

df_wandler_subset = df_curr[df_curr["wandler_key"].isin(sel_wandlers)]
available_duts = sorted(df_wandler_subset["dut_name"].unique())
sel_duts = st.sidebar.multiselect(
    "3. Geräte (DUTs) auswählen:",
    available_duts,
    default=available_duts,
    on_change=clear_cache,
)

if not sel_duts:
    st.stop()

# --- DATEN FILTERN ---
mask = (
    (df["nennstrom"] == sel_current)
    & (df["wandler_key"].isin(sel_wandlers))
    & (df["dut_name"].isin(sel_duts))
)
if "comparison_mode" in df.columns:
    comp_mode_disp = st.sidebar.radio(
        "Vergleichsgrundlage:",
        ["Messgerät (z.B. PAC1)", "Nennwert (Ideal)"],
        on_change=clear_cache,
    )
    comp_mode_val = "device_ref" if "Messgerät" in comp_mode_disp else "nominal_ref"
    mask = mask & (df["comparison_mode"] == comp_mode_val)

df_sub = df[mask].copy()
if comp_mode_val == "device_ref" and "ref_name" in df_sub.columns:
    df_sub = df_sub[df_sub["dut_name"] != df_sub["ref_name"]]

# Global Calcs
df_sub["err_ratio"] = (
    (df_sub["val_dut_mean"] - df_sub["val_ref_mean"]) / df_sub["val_ref_mean"]
) * 100
df_sub["err_std"] = (df_sub["val_dut_std"] / df_sub["val_ref_mean"]) * 100
df_sub["unique_id"] = df_sub["wandler_key"] + " - " + df_sub["trace_id"]

# --- DESIGN EDITOR ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Design & Settings")
sync_axes = st.sidebar.checkbox("🔗 Phasen synchronisieren", value=True)
y_limit = st.sidebar.slider("Y-Achse Zoom (+/- %)", 0.2, 10.0, 1.5, 0.1)
acc_class = st.sidebar.selectbox("Norm-Klasse", [0.2, 0.5, 1.0, 3.0], index=1)
show_err_bars = st.sidebar.checkbox("Fehlerbalken (StdAbw)", value=True)

with st.sidebar.expander("Farben & Namen bearbeiten", expanded=False):
    unique_curves = df_sub[
        ["unique_id", "wandler_key", "folder", "dut_name", "trace_id"]
    ].drop_duplicates()
    config_data = []
    b_idx, o_idx, x_idx = 0, 0, 0
    for idx, row in unique_curves.iterrows():
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
        config_data.append({"ID": row["unique_id"], "Legende": auto_name, "Farbe": col})

    df_config_default = pd.DataFrame(config_data)
    try:
        color_col_config = st.column_config.ColorColumn("Farbe")
    except AttributeError:
        color_col_config = st.column_config.TextColumn("Farbe (Hex)")
    edited_config = st.data_editor(
        df_config_default,
        column_config={"ID": None, "Legende": "Legende", "Farbe": color_col_config},
        disabled=["ID"],
        hide_index=True,
        use_container_width=True,
        key="design_editor",
    )

map_legend = dict(zip(edited_config["ID"], edited_config["Legende"]))
map_color = dict(zip(edited_config["ID"], edited_config["Farbe"]))
df_sub["final_legend"] = df_sub["unique_id"].map(map_legend)
df_sub["final_color"] = df_sub["unique_id"].map(map_color)

# --- EXPORT ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 PDF Export")
engine_mode = st.sidebar.selectbox(
    "Modus:", ["Python (Direkt)", "MATLAB (Automatischer Start)"]
)
matlab_exe = (
    st.sidebar.text_input("Pfad MATLAB:", value=DEFAULT_MATLAB_PATH)
    if "MATLAB" in engine_mode
    else None
)

if st.sidebar.button("🔄 Export starten", type="primary"):
    zip_buffer = io.BytesIO()
    if "MATLAB" in engine_mode:
        if not os.path.exists(matlab_exe):
            st.error("MATLAB nicht gefunden.")
        else:
            with st.spinner("MATLAB rendert..."):
                work_dir_abs = ensure_working_dir()
                export_df = df_sub.copy().rename(
                    columns={"final_color": "color_hex", "final_legend": "legend_name"}
                )
                export_df["trace_id"] = export_df["legend_name"]
                export_df.drop_duplicates(
                    subset=["trace_id", "target_load", "phase"], keep="last"
                ).to_csv(os.path.join(work_dir_abs, "plot_data.csv"), index=False)
                with open(os.path.join(work_dir_abs, "create_plots.m"), "w") as f:
                    f.write(
                        MATLAB_SCRIPT_TEMPLATE.replace("ACC_CLASS_PH", str(acc_class))
                        .replace("Y_LIMIT_PH", str(y_limit))
                        .replace("NOMINAL_CURRENT_PH", str(int(sel_current)))
                    )
                try:
                    subprocess.run(
                        [matlab_exe, "-batch", "create_plots"],
                        cwd=work_dir_abs,
                        check=True,
                    )
                    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                        for f in os.listdir(work_dir_abs):
                            if f.endswith(".pdf"):
                                zf.write(os.path.join(work_dir_abs, f), f)
                    st.success("✅ MATLAB Export fertig!")
                except Exception as e:
                    st.error(f"Fehler: {e}")
    else:
        with st.spinner("Python generiert PDFs..."):
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                ref_name_disp = (
                    df_sub.iloc[0]["ref_name"]
                    if comp_mode_val == "device_ref" and "ref_name" in df_sub.columns
                    else "Nennwert"
                )
                main_title_export = f"{int(sel_current)} A | Ref: {ref_name_disp}"
                fig_ex = make_subplots(
                    rows=2,
                    cols=3,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    row_heights=[0.7, 0.3],
                    subplot_titles=PHASES,
                )
                lim_x, lim_y_p, lim_y_n = get_trumpet_limits(acc_class)
                for c_idx, ph in enumerate(PHASES, 1):
                    fig_ex.add_trace(
                        go.Scatter(
                            x=lim_x,
                            y=lim_y_p,
                            mode="lines",
                            line=dict(color="black", width=1, dash="dash"),
                            showlegend=False,
                        ),
                        row=1,
                        col=c_idx,
                    )
                    fig_ex.add_trace(
                        go.Scatter(
                            x=lim_x,
                            y=lim_y_n,
                            mode="lines",
                            line=dict(color="black", width=1, dash="dash"),
                            showlegend=False,
                        ),
                        row=1,
                        col=c_idx,
                    )
                    p_data = df_sub[df_sub["phase"] == ph]
                    for _, grp in p_data.groupby("unique_id"):
                        grp = grp.sort_values("target_load")
                        l_nm = grp.iloc[0]["final_legend"]
                        col = grp.iloc[0]["final_color"]
                        fig_ex.add_trace(
                            go.Scatter(
                                x=grp["target_load"],
                                y=grp["err_ratio"],
                                mode="lines+markers",
                                name=l_nm,
                                line=dict(color=col, width=2),
                                legendgroup=l_nm,
                                showlegend=(c_idx == 1),
                            ),
                            row=1,
                            col=c_idx,
                        )
                        if show_err_bars:
                            fig_ex.add_trace(
                                go.Bar(
                                    x=grp["target_load"],
                                    y=grp["err_std"],
                                    marker_color=col,
                                    legendgroup=l_nm,
                                    showlegend=False,
                                ),
                                row=2,
                                col=c_idx,
                            )
                fig_ex.update_layout(
                    title=f"Gesamtübersicht: {main_title_export}",
                    template="plotly_white",
                    height=800,
                    width=1169,
                )
                fig_ex.update_yaxes(range=[-y_limit, y_limit], row=1)
                zf.writestr(
                    f"Zusammenfassung_{int(sel_current)}A.pdf",
                    fig_ex.to_image(format="pdf", width=1169, height=827),
                )
                for ph in PHASES:
                    fig_s = create_single_phase_figure(
                        df_sub,
                        ph,
                        acc_class,
                        y_limit,
                        show_err_bars,
                        title_prefix=main_title_export,
                    )
                    zf.writestr(
                        f"Detail_{ph}_{int(sel_current)}A.pdf",
                        fig_s.to_image(format="pdf", width=1169, height=827),
                    )
            st.success("✅ Python Export fertig!")
    st.session_state["zip_data"] = zip_buffer.getvalue()

if "zip_data" in st.session_state:
    st.sidebar.download_button(
        "💾 Download ZIP", st.session_state["zip_data"], "Report.zip", "application/zip"
    )

# =============================================================================
# MAIN TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs(
    ["📈 Gesamtgenauigkeit", "💰 Ökonomische Analyse", "⚙️ Stammdaten"]
)

# --- TAB 1 ---
with tab1:
    ref_name_disp = (
        df_sub.iloc[0]["ref_name"]
        if comp_mode_val == "device_ref" and "ref_name" in df_sub.columns
        else "Nennwert"
    )
    main_title = f"{int(sel_current)} A | Ref: {ref_name_disp}"
    fig = make_subplots(
        rows=2,
        cols=3,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=PHASES,
    )
    lim_x, lim_y_p, lim_y_n = get_trumpet_limits(acc_class)
    for col_idx, phase in enumerate(PHASES, start=1):
        fig.add_trace(
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
        fig.add_trace(
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
            leg_name = group.iloc[0]["final_legend"]
            color = group.iloc[0]["final_color"]
            fig.add_trace(
                go.Scatter(
                    x=group["target_load"],
                    y=group["err_ratio"],
                    mode="lines+markers",
                    name=leg_name,
                    line=dict(color=color, width=2),
                    legendgroup=leg_name,
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )
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
                    col=col_idx,
                )
    fig.update_layout(
        title=f"Gesamtübersicht: {main_title}",
        template="plotly_white",
        height=800,
        margin=dict(t=80, b=100),
        legend=dict(orientation="h", y=-0.15, x=0.5),
    )
    if sync_axes:
        fig.update_yaxes(matches="y", row=1)
    fig.update_yaxes(range=[-y_limit, y_limit], title_text="Fehler [%]", row=1, col=1)
    fig.update_yaxes(title_text="StdAbw [%]", row=2, col=1)
    fig.update_xaxes(title_text="Last [% In]", row=2, col=2)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: ÖKONOMIE (Gematcht über Base Type) ---
with tab2:
    st.markdown("### 💰 Preis, Volumen & Zonen-Genauigkeit")
    df_zones = calculate_zone_errors(df_sub)

    # Hier laden wir Specs anhand der Basis-Typen im aktuellen Filter
    unique_base_types = sorted(df_sub["base_type"].unique())
    df_specs = load_specs(unique_base_types)

    # Merge über 'base_type' statt wandler_key
    df_analysis = pd.merge(df_zones, df_specs, on="base_type", how="left")
    df_analysis["Volumen (cm³)"] = (
        df_analysis["L (mm)"] * df_analysis["B (mm)"] * df_analysis["H (mm)"]
    ) / 1000.0

    c1, c2, c3 = st.columns(3)
    with c1:
        x_axis = st.selectbox("X-Achse:", ["Preis (€)", "Volumen (cm³)"])
    with c2:
        y_axis = st.selectbox(
            "Y-Achse:",
            ["Fehler Niederstrom (%)", "Fehler Nennstrom (%)", "Fehler Überlast (%)"],
        )
    with c3:
        chart_type = st.radio("Typ:", ["Scatter", "Radar"])

    if df_analysis.empty:
        st.warning("Keine Daten.")
    elif chart_type == "Scatter":
        # Wir zeigen im Plot immer noch den detaillierten wandler_key (z.B. mit 8R1),
        # aber die X-Achse (Preis) kommt vom Basis-Typ.
        fig_eco = px.scatter(
            df_analysis,
            x=x_axis,
            y=y_axis,
            color="base_type",
            hover_name="wandler_key",  # Detailinfo beim Drüberfahren
            size=[15] * len(df_analysis),
            title=f"{x_axis} vs. {y_axis}",
            template="plotly_white",
        )
        fig_eco.update_traces(textposition="top center")
        st.plotly_chart(fig_eco, use_container_width=True)
    else:
        fig_radar = go.Figure()
        max_vals = {
            "Preis (€)": df_analysis["Preis (€)"].max() or 1,
            "Volumen (cm³)": df_analysis["Volumen (cm³)"].max() or 1,
            "Fehler Niederstrom (%)": df_analysis["Fehler Niederstrom (%)"].max()
            or 0.01,
            "Fehler Nennstrom (%)": df_analysis["Fehler Nennstrom (%)"].max() or 0.01,
            "Fehler Überlast (%)": df_analysis["Fehler Überlast (%)"].max() or 0.01,
        }
        cats = [
            "Preis (€)",
            "Volumen (cm³)",
            "Fehler Nieder",
            "Fehler Nenn",
            "Fehler Überlast",
        ]
        # Für Radar nehmen wir nur EINEN Eintrag pro Basis-Typ (mitteln falls mehrere Bürden da sind?),
        # oder wir zeigen alle. Hier zeigen wir alle Punkte als Linien.
        for i, row in df_analysis.iterrows():
            r_vals = [
                row["Preis (€)"] / max_vals["Preis (€)"],
                row["Volumen (cm³)"] / max_vals["Volumen (cm³)"],
                (row["Fehler Niederstrom (%)"] or 0)
                / max_vals["Fehler Niederstrom (%)"],
                (row["Fehler Nennstrom (%)"] or 0) / max_vals["Fehler Nennstrom (%)"],
                (row["Fehler Überlast (%)"] or 0) / max_vals["Fehler Überlast (%)"],
                row["Preis (€)"] / max_vals["Preis (€)"],
            ]
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=r_vals,
                    theta=cats + [cats[0]],
                    fill="toself",
                    name=row["wandler_key"],
                )
            )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Normalisiertes Profil (Innen=Besser)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# --- TAB 3: STAMMDATEN (Nur Basis-Typen) ---
with tab3:
    st.markdown("### ⚙️ Preise & Maße (Wandler-Typen gruppiert)")
    st.info("Bürden (z.B. 8R1) werden ignoriert. Trage Werte für den Basis-Typ ein.")

    # Alle Basis-Typen aus der DB holen
    all_base_types = sorted(df["base_type"].unique())
    current_specs = load_specs(all_base_types)

    edited_specs = st.data_editor(
        current_specs,
        column_config={
            "base_type": st.column_config.TextColumn(
                "Wandler-Typ (Ohne Bürde)", disabled=True
            ),
            "Preis (€)": st.column_config.NumberColumn("Preis (€)", format="%.2f"),
        },
        num_rows="fixed",
        hide_index=True,
        use_container_width=True,
    )
    if st.button("💾 Speichern"):
        edited_specs.to_csv(SPECS_FILE, index=False)
        st.success("Gespeichert!")
        st.rerun()
