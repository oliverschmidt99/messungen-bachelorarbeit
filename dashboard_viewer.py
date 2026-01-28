import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
import numpy as np
import re

# --- KONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "daten")
DATA_FILE = os.path.join(DATA_DIR, "messdaten_db.parquet")
CONFIG_FILE = os.path.join(DATA_DIR, "saved_configs.json")

PHASES = ["L1", "L2", "L3"]
ZONES = {
    "Niederstrom (5-50%)": [5, 20, 50],
    "Nennstrom (80-100%)": [80, 90, 100],
    "Überlast (≥120%)": [120, 150, 200],
}

# Farben & Stile
BLUES = ["#1f4e8c", "#2c6fb2", "#4a8fd1", "#6aa9e3", "#8fc0ee", "#b3d5f7"]
ORANGES = ["#8c4a2f", "#a65a2a", "#c96a2a", "#e07b39", "#f28e4b", "#f6a25e"]
OTHERS = ["#4caf50", "#6bd36b", "#b0b0b0", "#b39ddb", "#bc8f6f", "#f2a7d6"]
LINE_STYLES = ["solid", "dash", "dot", "dashdot", "longdash"]
MARKER_SYMBOLS = ["circle", "square", "diamond", "cross", "x", "triangle-up", "star"]


# --- HELPER (NUR LESEN) ---
def load_dashboard_configs():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("configurations", {})
        except:
            return {}
    return {}


@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_parquet(DATA_FILE)
    # Bereinigungen
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

    cols_check = [
        "Hersteller",
        "Modell",
        "nennstrom",
        "Mess-Bürde",
        "Geometrie",
        "Preis (€)",
        "T (mm)",
        "B (mm)",
        "H (mm)",
    ]
    for c in cols_check:
        if c not in df.columns:
            df[c] = 0.0 if "mm" in c or "Preis" in c or "strom" in c else "Unbekannt"

    if "val_dut_mean" not in df.columns:
        df["val_dut_mean"] = 0.0
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


def auto_format_name(row):
    try:
        base = str(row["wandler_key"])
        dut = str(row["dut_name"])
        if dut not in base:
            base += f" | {dut}"
        return base
    except:
        return "Unbekannt"


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
    label_class = f"Klassengrenzen {str(acc_class).replace('.', ',')}"

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

        fig.add_trace(
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
                            marker_color=row_first["final_color"],
                            legendgroup=row_first["final_legend"],
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
                        line=dict(
                            color=row_first["final_color"], width=1.5, dash="dot"
                        ),
                        marker=dict(symbol="x", size=row_first["final_size"]),
                        legendgroup=row_first["final_legend"],
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

    fig.update_layout(
        title=dict(text=f"{title_prefix} - Phase {phase}", font=dict(size=18)),
        template="plotly_white",
        height=700,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
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


# --- APP START ---
st.set_page_config(
    page_title="Bachelorarbeit: Dashboard (Viewer)", layout="wide", page_icon="👀"
)
st.markdown(
    """<style>[data-testid="stSidebar"] { min-width: 400px; max-width: 600px; }</style>""",
    unsafe_allow_html=True,
)

df = load_data()
if df is None:
    st.error("Datenbank nicht gefunden.")
    st.stop()

ALL_WANDLERS = sorted(df["wandler_key"].unique())
ALL_DUTS = sorted(df["dut_name"].unique())

# --- SIDEBAR (VIEWER MODUS) ---
st.sidebar.header("🔍 Ansicht Konfigurieren")
st.sidebar.info(
    "Dies ist die Viewer-Version. Änderungen sind nur temporär für diese Sitzung."
)

# 1. Preset Laden
configs = load_dashboard_configs()
sel_preset = st.sidebar.selectbox(
    "Voreinstellung laden:", ["-- Benutzerdefiniert --"] + sorted(list(configs.keys()))
)

if sel_preset != "-- Benutzerdefiniert --":
    data = configs[sel_preset]

    def smart_match(stored_list, available_options):
        if not stored_list:
            return []
        valid = [x for x in stored_list if x in available_options]
        return sorted(list(set(valid)))

    if st.session_state.get("last_preset") != sel_preset:
        st.session_state["k_current"] = data.get("current", [])
        st.session_state["k_geos"] = data.get("geos", [])
        st.session_state["k_wandlers"] = smart_match(
            data.get("wandlers", []), ALL_WANDLERS
        )

        dut_aliases = {"Einspeisung": "PAC1", "Pruefling": "PAC2"}
        mapped_duts = [dut_aliases.get(d, d) for d in data.get("duts", [])]
        st.session_state["k_duts"] = smart_match(mapped_duts, ALL_DUTS)

        st.session_state["k_comp"] = data.get("comp_mode", "Messgerät (z.B. PAC1)")
        st.session_state["k_ylim"] = data.get("y_limit", 1.5)
        st.session_state["k_yshift"] = data.get("y_shift", 0.0)
        st.session_state["k_class"] = data.get("acc_class", 0.2)

        # --- HIER: Alle Design-Attribute laden ---
        st.session_state["loaded_colors"] = data.get("custom_colors", {})
        st.session_state["loaded_legends"] = data.get("custom_legends", {})
        st.session_state["loaded_styles"] = data.get("custom_styles", {})
        st.session_state["loaded_symbols"] = data.get("custom_symbols", {})
        st.session_state["loaded_widths"] = data.get("custom_widths", {})
        st.session_state["loaded_visible"] = data.get("custom_visible", {})
        st.session_state["loaded_sizes"] = data.get("custom_sizes", {})

        st.session_state["last_preset"] = sel_preset
        st.rerun()

# 2. Filter
st.sidebar.markdown("### 🎛️ Filter")
avail_curr = sorted(df["nennstrom"].unique())
sel_curr = st.sidebar.multiselect(
    "Nennstrom:",
    avail_curr,
    default=st.session_state.get("k_current", [avail_curr[0]]),
    format_func=lambda x: f"{int(x)} A",
    key="k_current",
)

df_c = df[df["nennstrom"].isin(sel_curr)]
avail_geo = sorted(df_c["Geometrie"].astype(str).unique())
sel_geo = st.sidebar.multiselect(
    "Geometrie:", avail_geo, default=st.session_state.get("k_geos", []), key="k_geos"
)

df_g = df_c[df_c["Geometrie"].isin(sel_geo)]
avail_w = sorted(df_g["wandler_key"].unique())
sel_w = st.sidebar.multiselect(
    "Wandler:",
    avail_w,
    default=st.session_state.get("k_wandlers", []),
    key="k_wandlers",
)

df_w = df_g[df_g["wandler_key"].isin(sel_w)]
avail_d = sorted(df_w["dut_name"].unique())
sel_d = st.sidebar.multiselect(
    "Geräte:", avail_d, default=st.session_state.get("k_duts", []), key="k_duts"
)

# 3. Design
st.sidebar.markdown("### 🎨 Darstellung")
y_lim = st.sidebar.slider(
    "Zoom Y-Achse (+/- %)",
    0.2,
    10.0,
    float(st.session_state.get("k_ylim", 1.5)),
    key="k_ylim",
)
y_shift = st.sidebar.slider(
    "Verschiebung Y-Achse",
    -2.0,
    2.0,
    float(st.session_state.get("k_yshift", 0.0)),
    key="k_yshift",
)
acc_cls = st.sidebar.selectbox(
    "Klasse",
    [0.2, 0.5, 1.0],
    index=[0.2, 0.5, 1.0].index(st.session_state.get("k_class", 0.2)),
    key="k_class",
)
sync = st.sidebar.checkbox("Phasen synchronisieren", True, key="k_sync")

# --- DATEN FILTERN ---
comp_mode = "device_ref"
mask = (
    (df["nennstrom"].isin(sel_curr))
    & (df["Geometrie"].isin(sel_geo))
    & (df["wandler_key"].isin(sel_w))
    & (df["dut_name"].isin(sel_d))
    & (df["comparison_mode"] == comp_mode)
)
df_sub = df[mask].copy()

if df_sub.empty:
    st.info("Bitte wählen Sie Filter aus, um Daten anzuzeigen.")
    st.stop()

# Fehler berechnen
if "ref_name" in df_sub.columns:
    df_sub = df_sub[df_sub["dut_name"] != df_sub["ref_name"]]
df_sub["unique_id"] = df_sub["raw_file"] + " | " + df_sub["dut_name"].astype(str)
df_sub["err_ratio"] = (
    (df_sub["val_dut_mean"] - df_sub["val_ref_mean"]) / df_sub["val_ref_mean"]
) * 100
df_sub["err_std"] = (df_sub["val_dut_std"] / df_sub["val_ref_mean"]) * 100

# --- FARBEN & NAMEN EDITOR (FIX) ---
with st.sidebar.expander("Farben & Namen anpassen (Lokal)", expanded=False):
    u_curves = df_sub[
        ["unique_id", "wandler_key", "folder", "dut_name", "Kommentar"]
    ].drop_duplicates()

    # Laden aller Attribute aus dem State
    l_cols = st.session_state.get("loaded_colors", {})
    l_legs = st.session_state.get("loaded_legends", {})
    l_stys = st.session_state.get("loaded_styles", {})
    l_syms = st.session_state.get("loaded_symbols", {})
    l_wids = st.session_state.get("loaded_widths", {})
    l_vis = st.session_state.get("loaded_visible", {})
    l_siz = st.session_state.get("loaded_sizes", {})

    conf_data = []
    b_i, o_i, x_i = 0, 0, 0
    for idx, r in u_curves.iterrows():
        uid = r["unique_id"]
        auto = auto_format_name(r)
        # Default Farbe bestimmen
        if "parallel" in str(r["folder"]).lower():
            c = BLUES[b_i % len(BLUES)]
            b_i += 1
        elif "dreieck" in str(r["folder"]).lower():
            c = ORANGES[o_i % len(ORANGES)]
            o_i += 1
        else:
            c = OTHERS[x_i % len(OTHERS)]
            x_i += 1

        # Attribute holen (mit Fallbacks)
        conf_data.append(
            {
                "ID": uid,
                "Anzeigen": l_vis.get(uid, True),
                "Legende": l_legs.get(uid, auto),
                "Farbe": l_cols.get(uid, c),
                "Linie": l_stys.get(uid, "solid"),
                "Marker": l_syms.get(uid, "circle"),
                "Größe": l_siz.get(uid, 8),
                "Breite": l_wids.get(uid, 2.5),
            }
        )

    # Rückwärtskompatibilität für Farbe
    if hasattr(st.column_config, "ColorColumn"):
        col_config_color = st.column_config.ColorColumn("Farbe")
    else:
        col_config_color = st.column_config.TextColumn("Farbe (Hex)")

    edited = st.data_editor(
        pd.DataFrame(conf_data),
        column_config={
            "ID": None,
            "Farbe": col_config_color,
            "Linie": st.column_config.SelectboxColumn(
                "Linie", options=LINE_STYLES, required=True
            ),
            "Marker": st.column_config.SelectboxColumn(
                "Marker", options=MARKER_SYMBOLS, required=True
            ),
            "Größe": st.column_config.NumberColumn("Größe", min_value=1, max_value=20),
            "Breite": st.column_config.NumberColumn(
                "Breite", min_value=0.5, max_value=10.0
            ),
        },
        hide_index=True,
        disabled=["ID"],
        key="viewer_editor",
    )

    # Maps erstellen
    map_leg = dict(zip(edited["ID"], edited["Legende"]))
    map_col = dict(zip(edited["ID"], edited["Farbe"]))
    map_vis = dict(zip(edited["ID"], edited["Anzeigen"]))
    map_sty = dict(zip(edited["ID"], edited["Linie"]))
    map_sym = dict(zip(edited["ID"], edited["Marker"]))
    map_siz = dict(zip(edited["ID"], edited["Größe"]))
    map_wid = dict(zip(edited["ID"], edited["Breite"]))

    # Maps anwenden
    df_sub["final_legend"] = df_sub["unique_id"].map(map_leg)
    df_sub["final_color"] = df_sub["unique_id"].map(map_col)
    df_sub["final_visible"] = df_sub["unique_id"].map(map_vis)
    df_sub["final_style"] = df_sub["unique_id"].map(map_sty)
    df_sub["final_symbol"] = df_sub["unique_id"].map(map_sym)
    df_sub["final_size"] = df_sub["unique_id"].map(map_siz)
    df_sub["final_width"] = df_sub["unique_id"].map(map_wid)

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Genauigkeit", "💰 Ökonomie", "📋 Daten-Vorschau", "🔍 Rohendaten-Verlauf"]
)

with tab1:
    st.markdown("### Messgenauigkeit & Fehlerverlauf")
    curr_str = ", ".join([str(int(c)) for c in sel_curr]) + " A"

    fig_main = make_subplots(
        rows=1,
        cols=3,
        shared_xaxes=True,
        subplot_titles=PHASES,
        horizontal_spacing=0.05,
    )
    lim_x, lim_y_p, lim_y_n = get_trumpet_limits(acc_cls)
    lbl_cls = f"Klassengrenzen {str(acc_cls).replace('.', ',')}"

    for i, ph in enumerate(PHASES, 1):
        fig_main.add_trace(
            go.Scatter(
                x=lim_x,
                y=lim_y_p,
                mode="lines",
                line=dict(color="black", width=1, dash="dash"),
                name=lbl_cls,
                showlegend=(i == 1),
            ),
            row=1,
            col=i,
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
            col=i,
        )

        p_data = df_sub[df_sub["phase"] == ph]
        for uid, grp in p_data.groupby("unique_id"):
            grp = grp.sort_values("target_load")
            row1 = grp.iloc[0]
            if not row1["final_visible"]:
                continue

            fig_main.add_trace(
                go.Scatter(
                    x=grp["target_load"],
                    y=grp["err_ratio"],
                    mode="lines+markers",
                    name=row1["final_legend"],
                    line=dict(
                        color=row1["final_color"],
                        width=row1["final_width"],
                        dash=row1["final_style"],
                    ),
                    marker=dict(size=row1["final_size"], symbol=row1["final_symbol"]),
                    legendgroup=row1["final_legend"],
                    showlegend=(i == 1),
                ),
                row=1,
                col=i,
            )

    fig_main.update_layout(
        template="plotly_white",
        height=600,
        title=f"Übersicht: {curr_str}",
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    ym, yM = -y_lim + y_shift, y_lim + y_shift
    fig_main.update_yaxes(range=[ym, yM], title_text="Fehler [%]", row=1, col=1)
    if sync:
        fig_main.update_yaxes(matches="y")
    fig_main.update_xaxes(title_text="Last [% In]")

    st.plotly_chart(fig_main, use_container_width=True)

    if st.checkbox("Detail-Ansicht einzelner Phasen zeigen"):
        for ph in PHASES:
            st.plotly_chart(
                create_single_phase_figure(
                    df_sub,
                    ph,
                    acc_cls,
                    y_lim,
                    y_shift,
                    "Standardabweichung",
                    True,
                    curr_str,
                ),
                use_container_width=True,
            )

with tab2:
    st.markdown("### 💰 Kosten-Nutzen-Analyse")
    
    # 1. DATEN AGGREGATION
    df_agg = df_sub.groupby("unique_id").agg(
        legend=("final_legend", "first"),
        color=("final_color", "first"),
        # Fehlerwerte mitteln
        err_nieder=("err_ratio", lambda x: x[df_sub.loc[x.index, "target_load"].isin(ZONES["Niederstrom (5-50%)"])].abs().mean()),
        err_nom=("err_ratio", lambda x: x[df_sub.loc[x.index, "target_load"].isin(ZONES["Nennstrom (80-100%)"])].abs().mean()),
        err_high=("err_ratio", lambda x: x[df_sub.loc[x.index, "target_load"].isin(ZONES["Überlast (≥120%)"])].abs().mean()),
        # Physische Daten
        preis=("Preis (€)", "first"),
        vol_t=("T (mm)", "first"), vol_b=("B (mm)", "first"), vol_h=("H (mm)", "first")
    ).reset_index()
    
    df_agg["volumen"] = (df_agg["vol_t"] * df_agg["vol_b"] * df_agg["vol_h"]) / 1000.0
    
    # Mapping
    Y_OPTIONS_MAP = {
        "Fehler Niederstrom": "err_nieder",
        "Fehler Nennstrom": "err_nom",
        "Fehler Überlast": "err_high",
        "Preis (€)": "preis",
        "Volumen": "volumen",
    }

    # 2. KPI HEADER
    k1, k2, k3 = st.columns(3)
    k1.metric("Anzahl Varianten", len(df_agg))
    avg_p = df_agg['preis'].replace(0, np.nan).mean()
    k2.metric("Ø Preis", f"{avg_p:.2f} €" if pd.notna(avg_p) else "-")
    if not df_agg.empty:
        best_row = df_agg.loc[df_agg["err_nom"].idxmin()]
        k3.metric("Beste Genauigkeit (Nenn)", f"{best_row['err_nom']:.3f}%", best_row['legend'])
    
    st.divider()

    # 3. KONFIGURATION (Im Expander versteckt für saubere Optik)
    with st.expander("⚙️ Diagramm-Einstellungen (Ranking & Achsen)", expanded=False):
        c_conf1, c_conf2 = st.columns(2)
        
        # Links: Scatter Konfig
        with c_conf1:
            st.markdown("##### 🔵 Scatter-Plot Achsen")
            scat_x = st.selectbox("X-Achse:", ["Preis (€)", "Volumen", "Fehler Nennstrom"], index=0, key="viewer_sx")
            scat_y = st.selectbox("Y-Achse:", ["Fehler Nennstrom", "Fehler Niederstrom", "Preis (€)"], index=0, key="viewer_sy")

        # Rechts: Ranking Konfig (Multiselect)
        with c_conf2:
            st.markdown("##### 📊 Ranking Kriterien")
            sel_rank_criteria = st.multiselect(
                "Berechnungsgrundlage (Score):",
                options=list(Y_OPTIONS_MAP.keys()),
                default=["Fehler Nennstrom", "Preis (€)"],
                help="Wählen Sie die Kriterien, die aufsummiert werden sollen (niedriger ist besser).",
                key="viewer_rank"
            )

    # 4. BERECHNUNG DES RANKINGS
    df_rank = df_agg.copy()
    norm_cols = []
    
    # Falls nichts ausgewählt ist, Fallback auf Nennstrom
    criteria_to_use = sel_rank_criteria if sel_rank_criteria else ["Fehler Nennstrom"]
    
    df_rank["total_score"] = 0.0
    for label in criteria_to_use:
        col_key = Y_OPTIONS_MAP[label]
        mx = df_rank[col_key].abs().max()
        if mx == 0: mx = 1
        # Normalisieren (0-100%)
        df_rank[label] = (df_rank[col_key].abs() / mx) * 100
        df_rank["total_score"] += df_rank[label]
        norm_cols.append(label)
    
    df_rank = df_rank.sort_values("total_score", ascending=True)

    # 5. DARSTELLUNG (2-SPALTEN LAYOUT)
    c1, c2 = st.columns(2)
    
    # Links: Scatter
    with c1:
        # Mapping für Scatter Achsen
        col_x = "preis" if "Preis" in scat_x else ("volumen" if "Volumen" in scat_x else "err_nom")
        col_y_mapped = "preis" if "Preis" in scat_y else Y_OPTIONS_MAP.get(scat_y, "err_nom")
        
        fig_s = px.scatter(
            df_agg, 
            x=col_x, 
            y=col_y_mapped, 
            color="legend", 
            size=[15]*len(df_agg), 
            title=f"{scat_x} vs. {scat_y}", 
            color_discrete_sequence=df_agg["color"].tolist(),
            hover_data=["legend", col_x, col_y_mapped]
        )
        fig_s.update_layout(legend=dict(orientation="h", y=-0.25), template="plotly_white")
        st.plotly_chart(fig_s, use_container_width=True)
        
    # Rechts: Ranking (Balken)
    with c2:
        df_long = df_rank.melt(id_vars=["legend"], value_vars=norm_cols, var_name="Kategorie", value_name="Anteil (%)")
        
        fig_b = px.bar(
            df_long, 
            y="legend", 
            x="Anteil (%)", 
            orientation="h", 
            title="Performance Index (Ranking)", 
            color="Kategorie", 
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_b.update_layout(
            showlegend=True, 
            template="plotly_white", 
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_b, use_container_width=True)

    # --- UNTERER BEREICH: DATENTABELLE ---
    with st.expander("🔢 Detaillierte Datentabelle ansehen"):
        st.dataframe(
            df_agg.drop(columns=["color_hex", "vol_t", "vol_b", "vol_h", "wandler_key"], errors="ignore"), 
            use_container_width=True
        )


with tab3:
    st.markdown("### 📋 Datenbank-Vorschau (Read-Only)")
    st.info(
        "Hier sehen Sie alle Rohdaten, die in der Datenbank hinterlegt sind. Änderungen sind hier nicht möglich."
    )
    st.dataframe(df, use_container_width=True)

with tab4:
    st.markdown("### 🔍 Detaillierter Messverlauf (Aggregiert)")
    st.info(
        "Diese Ansicht zeigt die absoluten Messwerte (Kennlinie) der in der Datenbank gespeicherten Lastpunkte. Ideal zur Prüfung auf Linearität."
    )

    fig_raw = make_subplots(
        rows=1,
        cols=3,
        shared_xaxes=True,
        subplot_titles=PHASES,
        horizontal_spacing=0.05,
    )

    for i, ph in enumerate(PHASES, 1):
        p_data = df_sub[df_sub["phase"] == ph]
        for uid, grp in p_data.groupby("unique_id"):
            grp = grp.sort_values("target_load")
            row1 = grp.iloc[0]
            if not row1["final_visible"]:
                continue

            # Hier plotten wir Messwert vs Last (Absolut)
            fig_raw.add_trace(
                go.Scatter(
                    x=grp["target_load"],
                    y=grp["val_dut_mean"],
                    mode="lines+markers",
                    name=row1["final_legend"],
                    line=dict(color=row1["final_color"], width=1.5),
                    marker=dict(symbol=row1["final_symbol"], size=6),
                    legendgroup=row1["final_legend"],
                    showlegend=(i == 1),
                ),
                row=1,
                col=i,
            )

    fig_raw.update_layout(
        template="plotly_white",
        height=600,
        title="Absolute Messwerte (Kennlinie)",
        legend=dict(orientation="h", y=-0.15),
    )
    fig_raw.update_xaxes(title_text="Last [% In]")
    fig_raw.update_yaxes(title_text="Strom [A]")

    st.plotly_chart(fig_raw, use_container_width=True)
