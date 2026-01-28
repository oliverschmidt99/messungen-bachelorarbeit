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
    # Bereinigungen für saubere Anzeige
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

    # Sicherstellen, dass wichtige Spalten existieren (Fallback für Viewer)
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

# 1. Preset Laden (Nur Lesen)
configs = load_dashboard_configs()
sel_preset = st.sidebar.selectbox(
    "Voreinstellung laden:", ["-- Benutzerdefiniert --"] + sorted(list(configs.keys()))
)

if sel_preset != "-- Benutzerdefiniert --":
    data = configs[sel_preset]

    # Helper für Smart Match
    def smart_match(stored_list, available_options, col_name_in_df=None):
        if not stored_list:
            return []
        valid = [x for x in stored_list if x in available_options]
        return sorted(list(set(valid)))

    # Session State einmalig setzen bei Wechsel
    if st.session_state.get("last_preset") != sel_preset:
        st.session_state["k_current"] = data.get("current", [])
        st.session_state["k_geos"] = data.get("geos", [])
        st.session_state["k_wandlers"] = smart_match(
            data.get("wandlers", []), ALL_WANDLERS, "wandler_key"
        )

        dut_aliases = {"Einspeisung": "PAC1", "Pruefling": "PAC2"}
        mapped_duts = [dut_aliases.get(d, d) for d in data.get("duts", [])]
        st.session_state["k_duts"] = smart_match(mapped_duts, ALL_DUTS)

        st.session_state["k_comp"] = data.get("comp_mode", "Messgerät (z.B. PAC1)")
        st.session_state["k_ylim"] = data.get("y_limit", 1.5)
        st.session_state["k_yshift"] = data.get("y_shift", 0.0)
        st.session_state["k_class"] = data.get("acc_class", 0.2)

        st.session_state["loaded_colors"] = data.get("custom_colors", {})
        st.session_state["loaded_legends"] = data.get("custom_legends", {})
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
comp_mode = "device_ref"  # Immer gegen Referenz
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

# Temporäre Farben/Namen Editor (Lokal)
with st.sidebar.expander("Farben & Namen anpassen (Lokal)", expanded=False):
    u_curves = df_sub[
        ["unique_id", "wandler_key", "folder", "dut_name", "Kommentar"]
    ].drop_duplicates()
    l_cols = st.session_state.get("loaded_colors", {})
    l_legs = st.session_state.get("loaded_legends", {})

    conf_data = []
    b_i, o_i, x_i = 0, 0, 0
    for idx, r in u_curves.iterrows():
        uid = r["unique_id"]
        auto = auto_format_name(r)
        if "parallel" in str(r["folder"]).lower():
            c = BLUES[b_i % len(BLUES)]
            b_i += 1
        elif "dreieck" in str(r["folder"]).lower():
            c = ORANGES[o_i % len(ORANGES)]
            o_i += 1
        else:
            c = OTHERS[x_i % len(OTHERS)]
            x_i += 1

        conf_data.append(
            {
                "ID": uid,
                "Anzeigen": True,
                "Legende": l_legs.get(uid, auto),
                "Farbe": l_cols.get(uid, c),
                "Linie": "solid",
                "Marker": "circle",
                "Größe": 8,
                "Breite": 2.5,
            }
        )

    # --- FIX: Rückwärtskompatibilität für ältere Streamlit-Versionen ---
    if hasattr(st.column_config, "ColorColumn"):
        col_config_color = st.column_config.ColorColumn("Farbe")
    else:
        col_config_color = st.column_config.TextColumn("Farbe (Hex)")

    edited = st.data_editor(
        pd.DataFrame(conf_data),
        column_config={"ID": None, "Farbe": col_config_color},
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

    df_sub["final_legend"] = df_sub["unique_id"].map(map_leg)
    df_sub["final_color"] = df_sub["unique_id"].map(map_col)
    df_sub["final_visible"] = df_sub["unique_id"].map(map_vis)
    df_sub["final_style"] = df_sub["unique_id"].map(map_sty)
    df_sub["final_symbol"] = df_sub["unique_id"].map(map_sym)
    df_sub["final_size"] = df_sub["unique_id"].map(map_siz)
    df_sub["final_width"] = df_sub["unique_id"].map(map_wid)

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(
    ["📈 Genauigkeit (Fehlerkurven)", "💰 Ökonomische Analyse", "📋 Daten-Vorschau"]
)

with tab1:
    st.markdown("### Messgenauigkeit & Fehlerverlauf")
    # Titel generieren
    curr_str = ", ".join([str(int(c)) for c in sel_curr]) + " A"

    # 3 Spalten Layout oder 2 Zeilen
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
        # Grenzen
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

        # Daten
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

    # Detailansicht Option
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

    # Aggregation
    df_agg = (
        df_sub.groupby("unique_id")
        .agg(
            legend=("final_legend", "first"),
            color=("final_color", "first"),
            err_nom=(
                "err_ratio",
                lambda x: x[df_sub.loc[x.index, "target_load"].isin([80, 90, 100])]
                .abs()
                .mean(),
            ),
            preis=("Preis (€)", "first"),
            vol_t=("T (mm)", "first"),
            vol_b=("B (mm)", "first"),
            vol_h=("H (mm)", "first"),
        )
        .reset_index()
    )
    df_agg["volumen"] = (df_agg["vol_t"] * df_agg["vol_b"] * df_agg["vol_h"]) / 1000.0

    k1, k2, k3 = st.columns(3)
    k1.metric("Anzahl Varianten", len(df_agg))
    k2.metric("Ø Preis", f"{df_agg['preis'].mean():.2f} €")
    best_row = df_agg.loc[df_agg["err_nom"].idxmin()]
    k3.metric(
        "Beste Genauigkeit (Nenn)", f"{best_row['err_nom']:.3f}%", best_row["legend"]
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_s = px.scatter(
            df_agg,
            x="preis",
            y="err_nom",
            color="legend",
            size=[15] * len(df_agg),
            title="Preis vs. Genauigkeit (Nennstrom)",
            color_discrete_sequence=df_agg["color"].tolist(),
        )
        fig_s.update_layout(
            legend=dict(orientation="h", y=-0.2), template="plotly_white"
        )
        st.plotly_chart(fig_s, use_container_width=True)

    with c2:
        # Ranking
        df_rank = df_agg.sort_values("err_nom")
        fig_b = px.bar(
            df_rank,
            x="err_nom",
            y="legend",
            orientation="h",
            title="Ranking: Genauigkeit (weniger ist besser)",
            color="legend",
            color_discrete_sequence=df_rank["color"].tolist(),
        )
        fig_b.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_b, use_container_width=True)

with tab3:
    st.markdown("### 📋 Datenbank-Vorschau (Read-Only)")
    st.info(
        "Hier sehen Sie alle Rohdaten, die in der Datenbank hinterlegt sind. Änderungen sind hier nicht möglich."
    )
    st.dataframe(df, use_container_width=True)
