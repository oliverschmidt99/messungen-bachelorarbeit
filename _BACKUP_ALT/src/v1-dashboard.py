import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
import zipfile
import shutil
import subprocess
import time
import re

# --- KONFIGURATION ---
DATA_FILE = "messdaten_db.parquet"
PHASES = ["L1", "L2", "L3"]
WORK_DIR = "matlab_working_dir"
DEFAULT_MATLAB_PATH = r"C:\Program Files\MATLAB\R2025a\bin\matlab.exe"

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
    "#4caf50",  # Grün
    "#6bd36b",  # Hellgrün
    "#b0b0b0",  # Grau
    "#b39ddb",  # Violett
    "#bc8f6f",  # Braun
    "#f2a7d6",  # Pink
    "#d4d65a",  # Gelbgrün
    "#6fd6e5",  # Cyan
]


# --- MATLAB SKRIPT TEMPLATE ---
MATLAB_SCRIPT_TEMPLATE = r"""
%% Automatische Diagrammerstellung
clear; clc; close all;

try
    % Konfiguration
    filename = 'plot_data.csv';
    phases = {'L1', 'L2', 'L3'};
    limits_class = ACC_CLASS_PH; 
    nennstrom = NOMINAL_CURRENT_PH;

    if ~isfile(filename)
        error('Datei plot_data.csv nicht gefunden!');
    end

    data = readtable(filename, 'Delimiter', ',');

    % Farben definieren
    hex2rgb = @(hex) sscanf(hex(2:end),'%2x%2x%2x',[1 3])/255;

    % Trompeten-Grenzwerte
    x_lims = [1, 5, 20, 100, 120];
    if limits_class == 0.2
        y_lims = [0.75, 0.35, 0.2, 0.2, 0.2];
    elseif limits_class == 0.5
        y_lims = [1.5, 1.5, 0.75, 0.5, 0.5];
    elseif limits_class == 1.0
        y_lims = [3.0, 1.5, 1.0, 1.0, 1.0];
    else
        y_lims = [1.5, 1.5, 0.75, 0.5, 0.5];
    end

    for i = 1:length(phases)
        p = phases{i};
        fprintf('Bearbeite Phase %s...\n', p);
        
        % Daten filtern
        rows = strcmp(data.phase, p);
        sub_data = data(rows, :);
        
        if isempty(sub_data)
            fprintf('Warnung: Keine Daten fuer Phase %s.\n', p);
            continue; 
        end
        
        f = figure('Visible', 'off', 'PaperType', 'A4', 'PaperOrientation', 'landscape');
        set(f, 'Color', 'w'); 
        set(f, 'Units', 'centimeters', 'Position', [0 0 29.7 21]);
        
        t = tiledlayout(f, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        % --- PLOT 1: Fehler ---
        ax1 = nexttile;
        hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
        set(ax1, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.8 0.8 0.8]);
        
        plot(ax1, x_lims, y_lims, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        plot(ax1, x_lims, -y_lims, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
        
        % Eindeutige Kurven anhand der trace_id finden
        [unique_traces, ~, ~] = unique(sub_data.trace_id, 'stable');
        
        for k = 1:length(unique_traces)
            trace = unique_traces{k};
            trace_rows = strcmp(sub_data.trace_id, trace);
            d = sub_data(trace_rows, :);
            [~, sort_idx] = sort(d.target_load);
            d = d(sort_idx, :);
            
            % HIER WAR DER FEHLER: Wir nutzen jetzt die korrekten Spaltennamen der CSV
            col_rgb = hex2rgb(d.color_hex{1});
            
            % Unterstriche escapen
            leg_lbl = strrep(d.legend_name{1}, '_', '\_');
            
            plot(ax1, d.target_load, d.err_ratio, '-o', 'Color', col_rgb, ...
                'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', col_rgb, ...
                'DisplayName', leg_lbl);
        end
        
        title(ax1, sprintf('Fehlerverlauf - Phase %s (%d A)', p, nennstrom), 'Color', 'k');
        ylabel(ax1, 'Fehler [%]', 'Color', 'k');
        ylim(ax1, [- Y_LIMIT_PH, Y_LIMIT_PH]);
        xlim(ax1, [0, 125]);
        
        lgd = legend(ax1, 'Location', 'southoutside', 'Orientation', 'horizontal');
        set(lgd, 'TextColor', 'k', 'Color', 'w', 'Interpreter', 'tex');
        lgd.NumColumns = 2;
        
        % --- PLOT 2: StdAbw (Side-by-Side) ---
        ax2 = nexttile;
        hold(ax2, 'on'); grid(ax2, 'on'); box(ax2, 'on');
        set(ax2, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.8 0.8 0.8]);
        
        num_groups = length(unique_traces);
        total_group_width = 5; 
        single_bar_width = total_group_width / num_groups;
        if single_bar_width > 1.5; single_bar_width = 1.5; end
        
        for k = 1:length(unique_traces)
            trace = unique_traces{k};
            trace_rows = strcmp(sub_data.trace_id, trace);
            d = sub_data(trace_rows, :);
            [~, sort_idx] = sort(d.target_load);
            d = d(sort_idx, :);
            
            col_rgb = hex2rgb(d.color_hex{1});
            leg_lbl = strrep(d.legend_name{1}, '_', '\_');
            
            offset = (k - 1 - (num_groups - 1) / 2) * single_bar_width;
            x_pos = d.target_load + offset;
            
            bar(ax2, x_pos, d.err_std, 'FaceColor', col_rgb, 'EdgeColor', 'none', ...
                'FaceAlpha', 0.8, 'BarWidth', 0.1, 'DisplayName', leg_lbl);
        end
        
        title(ax2, sprintf('Standardabweichung - Phase %s (%d A)', p, nennstrom), 'Color', 'k');
        ylabel(ax2, 'StdAbw [%]', 'Color', 'k');
        xlabel(ax2, 'Last [% I_{Nenn}]', 'Color', 'k');
        xlim(ax2, [0, 125]);
        
        out_name = sprintf('Detail_%s_%dA.pdf', p, nennstrom);
        exportgraphics(f, out_name, 'ContentType', 'vector', 'BackgroundColor', 'w');
        close(f);
    end
catch ME
    disp('FEHLER:');
    disp(ME.message);
    exit(1); 
end
exit(0);
"""

st.set_page_config(page_title="Wandler Dashboard", layout="wide", page_icon="📈")

# --- CSS ---
st.markdown(
    """
<style>
    @media print {
        .stSidebar, header, footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; }
    }
    .block-container { padding-top: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)


# --- FUNKTIONEN ---
@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_parquet(DATA_FILE)
    if "dut_name" in df.columns:
        df["trace_id"] = df["folder"] + " | " + df["dut_name"].astype(str)
    else:
        df["trace_id"] = df["folder"]
    return df


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


# --- INTELLIGENTE NAMENSFORMATIERUNG ---
def auto_format_name(row):
    """
    Zerlegt den Namen in Token (getrennt durch _ oder Leerzeichen).
    Erkennt Bürden (0R1, 8R1) und formatiert sie.
    Setzt den Rest ohne Unterstriche wieder zusammen.
    """
    raw_key = str(row["wandler_key"])
    folder_lower = str(row["folder"]).lower()

    # 1. Alles in Token zerlegen (Unterstrich oder Leerzeichen als Trenner)
    tokens = re.split(r"[_\s]+", raw_key)

    name_parts = []
    burden_part = ""

    for t in tokens:
        if not t:
            continue
        # Regex für Bürde (z.B. 0R1, 8R1, 10R)
        if re.match(r"^\d+R\d*$", t):
            val = t.replace("R", ",")
            burden_part = f"{val} Ω"
        else:
            name_parts.append(t)

    # 2. Basis Name wieder zusammenbauen (mit Leerzeichen)
    base_name = " ".join(name_parts)

    # 3. Prüfling hinzufügen falls noch nicht drin
    dut = str(row["dut_name"])
    if dut.lower() not in base_name.lower():
        base_name = f"{base_name} | {dut}"

    # 4. Bürde anfügen
    if burden_part:
        base_name = f"{base_name} | {burden_part}"

    # 5. Arrangement
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

        first_row = group.iloc[0]
        leg_name = first_row["final_legend"]
        color = first_row["final_color"]

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
        margin=dict(t=80, b=100),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
    )
    fig.update_yaxes(range=[-y_limit, y_limit], title_text="Fehler [%]", row=1, col=1)
    fig.update_yaxes(title_text="StdAbw [%]", row=2, col=1)
    fig.update_xaxes(title_text="Last [% I_Nenn]", row=2, col=1)

    return fig


def clear_cache():
    if "zip_data" in st.session_state:
        del st.session_state["zip_data"]


def ensure_working_dir():
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)
    return os.path.abspath(WORK_DIR)


# --- APP START ---
df = load_data()
if df is None:
    st.error(f"⚠️ Datei '{DATA_FILE}' fehlt. Bitte erst `precalc.py` ausführen.")
    st.stop()

# --- SIDEBAR: FILTER ---
st.sidebar.header("🎛️ Filter & Settings")

sync_axes = st.sidebar.checkbox(
    "🔗 Phasen synchronisieren", value=True, on_change=clear_cache
)
y_limit = st.sidebar.slider(
    "Y-Achse Zoom (+/- %)", 0.2, 10.0, 1.5, 0.1, on_change=clear_cache
)
acc_class = st.sidebar.selectbox(
    "Norm-Klasse", [0.2, 0.5, 1.0, 3.0], index=1, on_change=clear_cache
)
show_err_bars = st.sidebar.checkbox(
    "Fehlerbalken (StdAbw)", value=True, on_change=clear_cache
)

st.sidebar.markdown("---")

if "comparison_mode" in df.columns:
    comp_mode_disp = st.sidebar.radio(
        "Vergleichsgrundlage:",
        ["Messgerät (z.B. PAC1)", "Nennwert (Ideal)"],
        on_change=clear_cache,
    )
    comp_mode_val = "device_ref" if "Messgerät" in comp_mode_disp else "nominal_ref"
else:
    st.sidebar.warning("Datenbank veraltet.")
    comp_mode_val = None

st.sidebar.markdown("---")

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

# --- DATEN FILTERN & BERECHNEN ---
mask = (
    (df["nennstrom"] == sel_current)
    & (df["wandler_key"].isin(sel_wandlers))
    & (df["dut_name"].isin(sel_duts))
)
if comp_mode_val:
    mask = mask & (df["comparison_mode"] == comp_mode_val)

df_sub = df[mask].copy()

if comp_mode_val == "device_ref" and "ref_name" in df_sub.columns:
    df_sub = df_sub[df_sub["dut_name"] != df_sub["ref_name"]]

if df_sub.empty:
    st.warning("Keine Daten.")
    st.stop()

df_sub["err_ratio"] = (
    (df_sub["val_dut_mean"] - df_sub["val_ref_mean"]) / df_sub["val_ref_mean"]
) * 100
df_sub["err_std"] = (df_sub["val_dut_std"] / df_sub["val_ref_mean"]) * 100

# --- EINDEUTIGE ID ---
df_sub["unique_id"] = df_sub["wandler_key"] + " - " + df_sub["trace_id"]

# --- SIDEBAR: DESIGN-EDITOR ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Design anpassen")

unique_curves = df_sub[
    ["unique_id", "wandler_key", "folder", "dut_name", "trace_id"]
].drop_duplicates()

config_data = []
b_idx, o_idx, x_idx = 0, 0, 0

for idx, row in unique_curves.iterrows():
    # Neue Namenslogik anwenden
    auto_name = auto_format_name(row)

    # Auto-Farbe
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

# Versuche Farbwähler zu nutzen
try:
    color_col_config = st.column_config.ColorColumn("Farbe (Wähler)")
except AttributeError:
    color_col_config = st.column_config.TextColumn(
        "Farbe (Hex)",
        help="Bitte 'pip install --upgrade streamlit' ausführen für Farbrad!",
    )

with st.sidebar.expander("Namen & Farben bearbeiten", expanded=False):
    edited_config = st.data_editor(
        df_config_default,
        column_config={
            "ID": None,
            "Legende": st.column_config.TextColumn("Legende"),
            "Farbe": color_col_config,
        },
        disabled=["ID"],
        hide_index=True,
        # Alte API war use_container_width=True, neue API erlaubt das auch,
        # aber st.plotly_chart meckert. Hier ist es für data_editor:
        use_container_width=True,
        key="design_editor",
    )

# Änderungen anwenden
map_legend = dict(zip(edited_config["ID"], edited_config["Legende"]))
map_color = dict(zip(edited_config["ID"], edited_config["Farbe"]))

df_sub["final_legend"] = df_sub["unique_id"].map(map_legend)
df_sub["final_color"] = df_sub["unique_id"].map(map_color)


# --- SCREEN PLOT ---
ref_name_disp = "Einspeisung"
if not df_sub.empty and "ref_name" in df_sub.columns:
    ref_name_disp = (
        df_sub.iloc[0]["ref_name"] if comp_mode_val == "device_ref" else "Nennwert"
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
        show_leg = col_idx == 1

        fig.add_trace(
            go.Scatter(
                x=group["target_load"],
                y=group["err_ratio"],
                mode="lines+markers",
                name=leg_name,
                line=dict(color=color, width=2),
                legendgroup=leg_name,
                showlegend=show_leg,
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
if not sync_axes:
    fig.update_yaxes(range=[-y_limit, y_limit], row=1, col=2)
    fig.update_yaxes(range=[-y_limit, y_limit], row=1, col=3)
fig.update_yaxes(title_text="StdAbw [%]", row=2, col=1)
fig.update_xaxes(title_text="Last [% I_Nenn]", row=2, col=2)
# FIX: width="stretch" für plotly_chart um Warnungen zu vermeiden
st.plotly_chart(
    fig,
    width=(
        "stretch"
        if "use_container_width" not in st.plotly_chart.__code__.co_varnames
        else None
    ),
    use_container_width=True,
)

# --- EXPORT LOGIK ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 PDF Export")

engine_mode = st.sidebar.selectbox(
    "Modus:", ["Python (Direkt)", "MATLAB (Automatischer Start)"], index=0
)

if "MATLAB" in engine_mode:
    matlab_exe = st.sidebar.text_input("Pfad zu matlab.exe:", value=DEFAULT_MATLAB_PATH)
else:
    matlab_exe = None

if st.sidebar.button("🔄 Export starten", type="primary"):
    zip_buffer = io.BytesIO()

    if "MATLAB" in engine_mode:
        if not os.path.exists(matlab_exe):
            st.error(
                f"❌ matlab.exe nicht gefunden unter:\n{matlab_exe}\nBitte Pfad korrigieren."
            )
        else:
            with st.spinner("MATLAB rendert im Hintergrund..."):
                work_dir_abs = ensure_working_dir()

                export_df = df_sub.copy()
                export_df = export_df.rename(
                    columns={"final_color": "color_hex", "final_legend": "legend_name"}
                )
                export_df["trace_id"] = export_df["legend_name"]
                export_df = export_df.drop_duplicates(
                    subset=["trace_id", "target_load", "phase"], keep="last"
                )

                csv_str = export_df.to_csv(index=False)
                matlab_code = MATLAB_SCRIPT_TEMPLATE.replace(
                    "ACC_CLASS_PH", str(acc_class)
                )
                matlab_code = matlab_code.replace("Y_LIMIT_PH", str(y_limit))
                matlab_code = matlab_code.replace(
                    "NOMINAL_CURRENT_PH", str(int(sel_current))
                )

                try:
                    with open(
                        os.path.join(work_dir_abs, "plot_data.csv"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(csv_str)
                    with open(
                        os.path.join(work_dir_abs, "create_plots.m"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(matlab_code)
                except OSError as e:
                    st.error(f"❌ Dateizugriff fehlgeschlagen: {e}.")
                    st.stop()

                cmd = [matlab_exe, "-batch", "create_plots"]
                try:
                    subprocess.run(cmd, cwd=work_dir_abs, check=True)
                    with zipfile.ZipFile(
                        zip_buffer, "a", zipfile.ZIP_DEFLATED, False
                    ) as zip_file:
                        found_pdfs = False
                        for file in os.listdir(work_dir_abs):
                            if file.endswith(".pdf"):
                                found_pdfs = True
                                zip_file.write(os.path.join(work_dir_abs, file), file)
                        if found_pdfs:
                            st.success("✅ Fertig! MATLAB hat die PDFs erstellt.")
                        else:
                            st.error(
                                "❌ MATLAB lief durch, aber keine PDFs gefunden. Prüfe matlab_working_dir."
                            )
                except subprocess.CalledProcessError as e:
                    st.error(f"❌ MATLAB Fehler. Exit Code: {e.returncode}")
                except Exception as e:
                    st.error(f"❌ Allgemeiner Fehler: {e}")
    else:
        with st.spinner("Generiere PDFs..."):
            with zipfile.ZipFile(
                zip_buffer, "a", zipfile.ZIP_DEFLATED, False
            ) as zip_file:
                img_bytes = fig.to_image(format="pdf", width=1169, height=827)
                zip_file.writestr(f"Zusammenfassung_{int(sel_current)}A.pdf", img_bytes)
                for phase in PHASES:
                    fig_single = create_single_phase_figure(
                        df_sub,
                        phase,
                        acc_class,
                        y_limit,
                        show_err_bars,
                        title_prefix=main_title,
                    )
                    img_bytes_single = fig_single.to_image(
                        format="pdf", width=1169, height=827
                    )
                    zip_file.writestr(
                        f"Detail_{phase}_{int(sel_current)}A.pdf", img_bytes_single
                    )
            st.success("✅ Fertig!")

    st.session_state["zip_data"] = zip_buffer.getvalue()

if "zip_data" in st.session_state:
    suffix = "MATLAB" if "MATLAB" in engine_mode else "Python"
    st.sidebar.download_button(
        label=f"💾 ZIP herunterladen ({suffix})",
        data=st.session_state["zip_data"],
        file_name=f"Report_{int(sel_current)}A_{suffix}.zip",
        mime="application/zip",
    )
