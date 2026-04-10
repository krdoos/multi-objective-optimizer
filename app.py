# -*- coding: utf-8 -*-
"""
Multi-Objective Drilling Optimizer Tool (MODO)
Based on: Applied Sciences
"Analysis and Multi-Objective Optimization of the Rate of Penetration 
and Mechanical Specific Energy"

Methodology:
1. Calculate MSE using Teale (1965) equation
2. Apply Drill Rate Test (DRT) play-back methodology
3. Group data by RPM intervals (±5 rpm around central values)
4. Aggregate WOB bins and fit 2nd-degree polynomial curves
5. Apply Desirability Function (Derringer & Suich, 1980)
6. Identify optimal WOB/RPM combination (max ROP + min MSE)
7. Segment analysis by geological layer if available
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Rock / Formation Database
# Extended with UCS (MPa), recommended WOB (klb), RPM ranges
# Sources: Bourgoyne et al. (1986), Teale (1965),
#          standard drilling engineering references
# ============================================================
ROCK_DB = {
    # ---- Carbonates ----
    "Carbonate (Pre-Salt)": {
        "UCS_MPa": (80, 200),
        "description": "High-strength carbonate reservoir typical of pre-salt operations. "
                       "Highly heterogeneous with vugs and fractures.",
        "WOB_klb": (10, 25),
        "RPM": (100, 150),
        "bit_type": "PDC",
        "color": "#1a6b8a",
    },
    "Limestone": {
        "UCS_MPa": (30, 250),
        "description": "Sedimentary carbonate rock. Strength varies widely with porosity.",
        "WOB_klb": (8, 20),
        "RPM": (80, 130),
        "bit_type": "PDC/Roller",
        "color": "#2196a0",
    },
    "Dolomite": {
        "UCS_MPa": (40, 260),
        "description": "Diagenetically altered carbonate. Generally harder than limestone.",
        "WOB_klb": (10, 22),
        "RPM": (80, 130),
        "bit_type": "PDC/Roller",
        "color": "#0e7c7b",
    },
    # ---- Clastics ----
    "Sandstone (Soft)": {
        "UCS_MPa": (5, 60),
        "description": "Poorly cemented sandstone, low UCS.",
        "WOB_klb": (3, 8),
        "RPM": (60, 100),
        "bit_type": "PDC",
        "color": "#c8892a",
    },
    "Sandstone (Hard)": {
        "UCS_MPa": (60, 170),
        "description": "Well-cemented quartz arenite, moderate to high UCS.",
        "WOB_klb": (8, 18),
        "RPM": (80, 130),
        "bit_type": "PDC/Roller",
        "color": "#a0522d",
    },
    "Shale": {
        "UCS_MPa": (5, 100),
        "description": "Fine-grained clastic. Prone to balling with PDC bits.",
        "WOB_klb": (3, 10),
        "RPM": (60, 100),
        "bit_type": "PDC",
        "color": "#607d8b",
    },
    "Conglomerate": {
        "UCS_MPa": (30, 120),
        "description": "Coarse clastic; heterogeneous strength due to mixed clasts.",
        "WOB_klb": (8, 18),
        "RPM": (70, 120),
        "bit_type": "Roller",
        "color": "#8d6e63",
    },
    # ---- Evaporites ----
    "Salt (Halite)": {
        "UCS_MPa": (10, 30),
        "description": "Evaporite; very low UCS but prone to creep and borehole closure.",
        "WOB_klb": (3, 8),
        "RPM": (50, 80),
        "bit_type": "PDC",
        "color": "#e1bee7",
    },
    "Anhydrite": {
        "UCS_MPa": (50, 150),
        "description": "Hard evaporite. Significant bit wear.",
        "WOB_klb": (8, 20),
        "RPM": (70, 120),
        "bit_type": "Roller",
        "color": "#ce93d8",
    },
    # ---- Igneous / Metamorphic ----
    "Granite": {
        "UCS_MPa": (100, 250),
        "description": "Very hard igneous rock. Requires high WOB and specialized bits.",
        "WOB_klb": (20, 40),
        "RPM": (60, 120),
        "bit_type": "Roller/PDC",
        "color": "#b71c1c",
    },
    "Basalt": {
        "UCS_MPa": (150, 350),
        "description": "Extremely hard volcanic rock. Very high MSE expected.",
        "WOB_klb": (25, 45),
        "RPM": (50, 100),
        "bit_type": "Roller",
        "color": "#37474f",
    },
    # ---- Unknown / Custom ----
    "Custom / Unknown": {
        "UCS_MPa": (1, 500),
        "description": "Use when rock type is unknown. UCS range is set wide; "
                       "enter a specific UCS value below.",
        "WOB_klb": (2, 50),
        "RPM": (40, 200),
        "bit_type": "Any",
        "color": "#9e9e9e",
    },
}

MPA_TO_PSI = 145.038   # 1 MPa = 145.038 psi


# ============================================================
#  Utility functions
# ============================================================

def ucs_psi_range(rock_name):
    """Return (min_psi, max_psi, avg_psi) for a rock type."""
    entry = ROCK_DB.get(rock_name, ROCK_DB["Custom / Unknown"])
    lo, hi = entry["UCS_MPa"]
    return lo * MPA_TO_PSI, hi * MPA_TO_PSI, 0.5 * (lo + hi) * MPA_TO_PSI


def col_lower(name):
    return str(name).strip().lower()


def best_match(columns, patterns):
    """Return the first column whose lower-cased name contains any pattern."""
    for pat in patterns:
        pat_l = pat.lower()
        for c in columns:
            if pat_l in col_lower(c):
                return c
    return None


def to_numeric_safe(series):
    return pd.to_numeric(series, errors='coerce')


def wob_to_klb(series, col_name):
    """Convert WOB to klb based on column name heuristics."""
    cl = col_lower(col_name)
    if 'tonnes' in cl or 'ton' in cl:
        # 1 metric tonne = 2204.62 lbs → klb = tonnes * 2.20462
        return series * 2.20462
    elif 'kg' in cl:
        return series * 0.00220462
    elif 'k-lbs' in cl or 'klbs' in cl or 'k_lbs' in cl:
        return series  # already klb
    elif 'lbs' in cl:
        return series / 1000.0
    else:
        # Default: assume tonnes (common in international drilling)
        return series * 2.20462


def torque_to_ftlb(series, col_name):
    """Convert Torque to ft-lb."""
    cl = col_lower(col_name)
    if 'kft' in cl or 'k_ft' in cl or 'kftlb' in cl or 'trq' in cl:
        return series * 1000.0   # kft-lb → ft-lb
    elif 'nm' in cl or 'n.m' in cl:
        return series * 0.737562
    elif 'psi' in cl:
        # Surface Torque in PSI is actually a pressure reading from the kelly/top drive.
        # Convert using: T (ft-lb) = SPP(psi) * Flow(gal/min) / (0.1714 * RPM)
        # We cannot do that directly without flow/rpm, so we store raw & handle later.
        return series  # flagged below
    else:
        return series * 1000.0   # assume kft-lb


def rop_to_fthr(series, col_name):
    """Convert ROP to ft/hr."""
    cl = col_lower(col_name)
    if 'm/hr' in cl or 'm/h' in cl or 'm_hr' in cl:
        return series * 3.28084
    else:
        return series   # assume ft/hr


def fit_poly2(x, y):
    """Fit 2nd-degree polynomial; fall back to linear or constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    for deg in (2, 1):
        if len(x) > deg:
            try:
                coeffs = np.polyfit(x, y, deg)
                return np.poly1d(coeffs)
            except Exception:
                pass
    return np.poly1d([np.nanmean(y)])


def founder_point(poly_rop, x_min, x_max, n=500):
    """
    Find the 'foundering point' = WOB where ROP curve peaks (Region III onset).
    This is the local maximum of the polynomial in [x_min, x_max].
    """
    xs = np.linspace(x_min, x_max, n)
    ys = poly_rop(xs)
    idx = np.argmax(ys)
    return xs[idx], ys[idx]


def desirability_D(rop_val, mse_val,
                   rop_L, rop_U,
                   mse_L, mse_U,
                   w_rop=1.0, w_mse=1.0):
    """
    Compute total desirability D (Derringer & Suich 1980).
    ROP: Larger-the-best (LTB) — Eq. (3) in paper
    MSE: Smaller-the-best (STB) — Eq. (4) in paper
    D = (d_rop^w_rop * d_mse^w_mse)^(1/(w_rop+w_mse))
    """
    # Individual desirabilities
    if rop_U > rop_L:
        d_rop = np.clip((rop_val - rop_L) / (rop_U - rop_L), 0.0, 1.0) ** w_rop
    else:
        d_rop = 1.0

    if mse_U > mse_L:
        d_mse = np.clip((mse_U - mse_val) / (mse_U - mse_L), 0.0, 1.0) ** w_mse
    else:
        d_mse = 1.0

    total_w = w_rop + w_mse
    return (d_rop * d_mse) ** (1.0 / total_w)


# ============================================================
#  MSE Calculation  — Teale (1965) Eq. 1 in paper
#  MSE = WOB/Ab + (120·π·RPM·TOB) / (Ab·ROP)
#  WOB in lbs, TOB in ft-lb, ROP in ft/hr, Ab in in²
#  Result in psi
# ============================================================
def calc_mse(wob_lbs, rpm, tob_ftlb, rop_fthr, bit_area_in2):
    rop_arr  = np.asarray(rop_fthr, dtype=float)
    wob_arr  = np.asarray(wob_lbs,  dtype=float)
    rpm_arr  = np.asarray(rpm,       dtype=float)
    tob_arr  = np.asarray(tob_ftlb, dtype=float)
    rop_safe = np.where(rop_arr <= 0, 0.001, rop_arr)
    term1    = wob_arr / bit_area_in2
    term2    = (120.0 * np.pi * rpm_arr * tob_arr) / (bit_area_in2 * rop_safe)
    return np.maximum(term1 + term2, 0.0)


# ============================================================
#  Torque estimation from Surface Torque (PSI) reading
#  Many surface torque gauges output hydraulic pressure (PSI).
#  Conversion: TOB (ft-lb) = C * SPT (psi)  where C depends on
#  the hydraulic circuit. A common approximation used in field
#  practice is C = 1.0 (i.e., 1 psi gauge ≈ 1 ft-lb of torque
#  when the gauge is calibrated to torque-equivalent units).
#  If the Pason system records "Surface Torque (psi)" it is
#  in fact the torque read-out in psi-equivalent and must be
#  converted to ft-lb via the bit diameter.
#  Teale's equation requires TOB in ft-lb:
#  TOB (ft-lb) = [Surface Torque (psi)] * [bit_area (in²)] / (2π)
#  This derivation comes from: Torque = Force × radius,
#  where Force = Pressure × Area and radius = d_bit/24 (ft)
# ============================================================
def surface_torque_psi_to_ftlb(spt_psi, bit_diameter_in):
    """Convert surface torque reading in PSI to ft-lb."""
    bit_area = np.pi * (bit_diameter_in / 2.0) ** 2   # in²
    radius_ft = (bit_diameter_in / 2.0) / 12.0         # in → ft
    return spt_psi * bit_area * radius_ft / (2.0 * np.pi)


# ============================================================
#  Geological layer auto-detection using depth intervals
#  and abrupt changes in MSE / ROP (formation change proxy)
# ============================================================
def auto_segment_layers(depth, mse, rop, n_segments=5):
    """
    Divide the well into n_segments equal-depth intervals and
    label each segment with a generic layer name.
    Returns a Series of labels.
    """
    depth = np.asarray(depth)
    labels = pd.cut(
        depth,
        bins=n_segments,
        labels=[f"Layer_{i+1}" for i in range(n_segments)]
    )
    return labels.astype(str)


# ============================================================
#  Streamlit App
# ============================================================
st.set_page_config(
    page_title="MODO – Multi-Objective Drilling Optimizer",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-title {font-size: 1.8rem; font-weight: 700; color: #1a237e; margin-bottom: 0;}
    .sub-title {font-size: 0.95rem; color: #546e7a; margin-top: 0;}
    .metric-card {background:#f5f7fa; border-left:4px solid #1a237e;
                  padding:12px 16px; border-radius:6px; margin-bottom:8px;}
    .metric-card h4 {margin:0 0 4px 0; color:#1a237e; font-size:0.85rem;}
    .metric-card p  {margin:0; font-size:1.3rem; font-weight:700; color:#0d47a1;}
    .info-box {background:#e3f2fd; border-radius:6px; padding:10px 14px;
               font-size:0.85rem; color:#1565c0; margin-bottom:8px;}
    .warn-box {background:#fff3e0; border-radius:6px; padding:10px 14px;
               font-size:0.85rem; color:#e65100; margin-bottom:8px;}
    .region-label {font-size:0.78rem; font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⛏️ MODO – Multi-Objective Drilling Optimizer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Based on Mantegazini et al. (2024) · Drill Rate Test Play-Back · '
    'Desirability Function Optimization (Derringer & Suich 1980)</p>',
    unsafe_allow_html=True
)

# ============================================================
#  Session state
# ============================================================
for key in ('opt_done', 'results_df', 'working_df'):
    if key not in st.session_state:
        st.session_state[key] = None if key != 'opt_done' else False

# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Drilling Setup")

    bit_diam = st.number_input("Bit Diameter (inches)", value=12.25, min_value=3.0,
                                max_value=36.0, step=0.25, format="%.2f",
                                help="PDC or Roller bit diameter. Paper uses 12.25 in.")
    bit_area = np.pi * (bit_diam / 2.0) ** 2
    st.caption(f"Bit cross-section area Ab = {bit_area:.2f} in²")

    rig_rate = st.number_input("Rig Daily Rate (USD/day)", value=1_300_000, step=50_000,
                                format="%d",
                                help="Paper uses USD 1.3M/day for pre-salt ultra-deep wells.")
    rig_hr   = rig_rate / 24.0

    st.divider()
    st.subheader("🪨 Formation / Rock Type")
    rock_name = st.selectbox("Select Rock Type", list(ROCK_DB.keys()), index=0)
    rock = ROCK_DB[rock_name]
    ucs_lo, ucs_hi, ucs_avg = ucs_psi_range(rock_name)

    if rock_name == "Custom / Unknown":
        custom_ucs_mpa = st.number_input("Enter UCS (MPa)", value=100.0, min_value=1.0,
                                          max_value=1000.0, step=5.0)
        ucs_avg = custom_ucs_mpa * MPA_TO_PSI
        ucs_lo  = ucs_avg * 0.7
        ucs_hi  = ucs_avg * 1.3

    ucs_avg_kpsi = ucs_avg / 1000.0
    st.markdown(f"""
    <div class="info-box">
    <b>{rock['description']}</b><br>
    UCS: {rock['UCS_MPa'][0]}–{rock['UCS_MPa'][1]} MPa
    ({ucs_lo/1000:.0f}–{ucs_hi/1000:.0f} kpsi) · Avg: {ucs_avg_kpsi:.0f} kpsi<br>
    Rec. WOB: {rock['WOB_klb'][0]}–{rock['WOB_klb'][1]} klb · 
    RPM: {rock['RPM'][0]}–{rock['RPM'][1]} rpm · Bit: {rock['bit_type']}
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("📐 DRT Parameters")
    st.markdown(
        "The **Drill Rate Test (DRT)** groups data by RPM intervals and WOB bins "
        "to reproduce pre-operational test curves (Sec. 3.3 of paper).",
        help="Methodology from Mantegazini et al. (2024)"
    )
    wob_bin = st.number_input(
        "WOB bin size (klb)", value=5.0, min_value=1.0, max_value=20.0, step=1.0,
        help="Paper tests 3, 5, and 7 klb bins. 5 klb and 7 klb gave best results."
    )
    rpm_half_bw = st.number_input(
        "RPM bin half-width (rpm)", value=5.0, min_value=2.0, max_value=20.0, step=1.0,
        help="Data within ±this value of each RPM centre is grouped together."
    )
    rpm_centers_input = st.text_input(
        "RPM centre values (comma-separated)",
        value="110, 120, 130, 140, 150",
        help="Paper uses 110, 120, 130, 140, 150 rpm."
    )
    try:
        rpm_centers = [float(x.strip()) for x in rpm_centers_input.split(',') if x.strip()]
    except Exception:
        rpm_centers = [110, 120, 130, 140, 150]

    min_pts_per_bin = st.number_input(
        "Min data points per WOB bin", value=3, min_value=1, max_value=20, step=1
    )

    st.divider()
    st.subheader("⚖️ Desirability Weights")
    st.caption("Paper uses equal weighting. Adjust to prioritize ROP or MSE.")
    w_rop = st.slider("Weight for ROP (maximize)", 0.0, 1.0, 0.5, 0.1)
    w_mse = round(1.0 - w_rop, 1)
    st.caption(f"Weight for MSE (minimize) = {w_mse:.1f}")

    st.divider()
    st.subheader("🔍 Data Filter")
    min_rop_filter = st.number_input("Min ROP (ft/hr) for analysis", value=1.0, step=0.5)
    max_rop_filter = st.number_input("Max ROP (ft/hr) for analysis", value=800.0, step=10.0)
    min_wob_filter = st.number_input("Min WOB (klb) for analysis", value=2.0, step=0.5)
    min_rpm_filter = st.number_input("Min RPM for analysis", value=30.0, step=5.0)

    st.divider()
    st.subheader("🗂️ Layer Segmentation")
    segment_mode = st.radio(
        "How to define layers?",
        ["Manual column in file", "Auto depth-based segmentation",
         "Single layer (no segmentation)"],
        index=2,
        help="Doctors' feedback: analysis should be layer-aware."
    )
    if segment_mode == "Auto depth-based segmentation":
        n_auto_layers = st.slider("Number of depth-based layers", 2, 10, 4)

# ============================================================
#  File Upload & Column Mapping
# ============================================================
st.subheader("📂 Upload Drilling Data")
uploaded = st.file_uploader(
    "Upload CSV or Excel file",
    type=['csv', 'xlsx', 'xls'],
    help="Required columns: Depth, ROP, WOB, RPM, Torque (or Surface Torque)."
)

if uploaded is None:
    st.markdown("""
    <div class="info-box">
    Upload a CSV or Excel file with drilling data. The app auto-detects columns for:<br>
    <b>Depth · ROP · WOB · RPM · Torque</b><br><br>
    Supported datasets: Pason log CSV · Excel exports from drilling software.<br>
    The app handles unit conversion automatically (tonnes→klb, kft-lb→ft-lb, m/hr→ft/hr, etc.)
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---- Load file ----
@st.cache_data(show_spinner=False)
def load_file(file_bytes, file_name):
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        return pd.read_excel(file_bytes)
    try:
        return pd.read_csv(file_bytes, encoding='utf-8-sig', sep=None, engine='python')
    except Exception:
        return pd.read_csv(file_bytes, encoding='latin-1', sep=None, engine='python')

raw_df = load_file(uploaded, uploaded.name)
st.success(f"✅ Loaded **{uploaded.name}** — {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")

with st.expander("📊 Data Preview (first 10 rows)", expanded=False):
    st.dataframe(raw_df.head(10))

# ---- Column mapping ----
st.subheader("🔗 Column Mapping")
cols = list(raw_df.columns)

detected = {
    'depth':  best_match(cols, ['depth(ft)', 'depth(m)', 'depth', 'dept', 'md']),
    'rop':    best_match(cols, ['rop(1 ft)', 'rop(1 m)', 'rop (m/hr)', 'rop', 'rate of penetration']),
    'wob':    best_match(cols, ['weight on bit', 'wob (tonnes)', 'wob (k-lbs)', 'wob']),
    'rpm':    best_match(cols, ['rotary speed', 'rpm', 'rotary', 'rev/min']),
    'torque': best_match(cols, ['trq_avg', 'surface torque', 'torque', 'trq']),
    'layer':  best_match(cols, ['lithology', 'formation', 'layer', 'rock', 'zone']),
}

idx = lambda k: cols.index(detected[k]) if detected[k] and detected[k] in cols else 0

c1, c2, c3 = st.columns(3)
with c1:
    depth_col  = st.selectbox("Depth column", cols, index=idx('depth'))
    rop_col    = st.selectbox("ROP column", cols, index=idx('rop'))
with c2:
    wob_col    = st.selectbox("WOB column", cols, index=idx('wob'))
    rpm_col    = st.selectbox("RPM column", cols, index=idx('rpm'))
with c3:
    torque_col = st.selectbox("Torque column", cols, index=idx('torque'))
    layer_col_options = ['None'] + cols
    default_layer_idx = (layer_col_options.index(detected['layer'])
                         if detected['layer'] and detected['layer'] in layer_col_options else 0)
    layer_col = st.selectbox("Layer/Lithology column (optional)", layer_col_options,
                              index=default_layer_idx)

# ---- Unit hints ----
rop_unit = st.radio("ROP unit in file:", ['ft/hr', 'm/hr'], horizontal=True, index=0,
                     help="App converts to ft/hr internally for Teale MSE formula.")
wob_unit = st.radio("WOB unit in file:", ['k-lbs', 'tonnes', 'lbs'], horizontal=True,
                     help="App converts to klb internally.")
torque_source = st.radio(
    "Torque column type:",
    ['kft-lb (direct torque)', 'Surface Torque PSI (Pason-style)'],
    horizontal=True,
    help=(
        "Pason logs report 'Surface Torque (psi)' which is a hydraulic pressure. "
        "Select 'Surface Torque PSI' and the app converts it to ft-lb using bit area and radius."
    )
)

# ============================================================
#  RUN Button
# ============================================================
run_btn = st.button("🚀 Run DRT Optimization", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("Processing drilling data..."):

        # ------ Build working DataFrame ------
        df = pd.DataFrame()
        df['Depth_ft']  = to_numeric_safe(raw_df[depth_col])

        # ROP → ft/hr
        rop_raw = to_numeric_safe(raw_df[rop_col])
        df['ROP_fthr'] = rop_raw * 3.28084 if rop_unit == 'm/hr' else rop_raw

        # WOB → klb
        wob_raw = to_numeric_safe(raw_df[wob_col])
        if wob_unit == 'tonnes':
            df['WOB_klb'] = wob_raw * 2.20462
        elif wob_unit == 'lbs':
            df['WOB_klb'] = wob_raw / 1000.0
        else:
            df['WOB_klb'] = wob_raw   # already klb

        # RPM
        df['RPM'] = to_numeric_safe(raw_df[rpm_col])

        # Torque → ft-lb
        torq_raw = to_numeric_safe(raw_df[torque_col])
        if 'PSI' in torque_source:
            df['TOB_ftlb'] = surface_torque_psi_to_ftlb(torq_raw, bit_diam)
        else:
            df['TOB_ftlb'] = torq_raw * 1000.0   # kft-lb → ft-lb

        # Layer column
        if segment_mode == "Manual column in file" and layer_col != 'None':
            df['Layer'] = raw_df[layer_col].astype(str)
        else:
            df['Layer'] = rock_name   # placeholder; re-assigned below

        # Drop NaN rows
        n_before = len(df)
        df.dropna(subset=['Depth_ft', 'ROP_fthr', 'WOB_klb', 'RPM', 'TOB_ftlb'], inplace=True)

        # Apply filters
        mask = (
            (df['ROP_fthr'] >= min_rop_filter) &
            (df['ROP_fthr'] <= max_rop_filter) &
            (df['WOB_klb']  >= min_wob_filter) &
            (df['RPM']      >= min_rpm_filter) &
            (df['TOB_ftlb'] > 0)
        )
        df = df[mask].reset_index(drop=True)
        n_after = len(df)

        if n_after < 10:
            st.error(f"Only {n_after} rows remain after filtering. "
                     "Adjust filter thresholds in the sidebar.")
            st.stop()

        st.info(f"Rows loaded: {n_before:,} → after filtering: **{n_after:,}**")

        # ------ Calculate MSE  (Teale 1965, Eq. 1) ------
        df['MSE_psi'] = calc_mse(
            df['WOB_klb'] * 1000.0,   # klb → lbs
            df['RPM'],
            df['TOB_ftlb'],
            df['ROP_fthr'],
            bit_area
        )
        df['MSE_psi'] = np.maximum(df['MSE_psi'], 0)
        df['MSE_kpsi'] = df['MSE_psi'] / 1000.0

        # ------ Layer assignment ------
        if segment_mode == "Auto depth-based segmentation":
            df['Layer'] = auto_segment_layers(
                df['Depth_ft'], df['MSE_psi'], df['ROP_fthr'], n_auto_layers
            )
        elif segment_mode == "Single layer (no segmentation)":
            df['Layer'] = rock_name

        # ------ RPM grouping ------
        def assign_rpm_group(rpm_val):
            for c in rpm_centers:
                if abs(rpm_val - c) <= rpm_half_bw:
                    return f"{int(c)} rpm"
            return None

        df['RPM_Group'] = df['RPM'].apply(assign_rpm_group)
        df = df[df['RPM_Group'].notna()].reset_index(drop=True)

        if len(df) < 5:
            st.error("No rows match the specified RPM centres. "
                     "Adjust RPM centre values in the sidebar.")
            st.stop()

        # Save working df
        st.session_state.working_df = df

        # ============================================================
        #  DRT Loop: iterate over (Layer × RPM_Group)
        # ============================================================
        results = []
        total_groups = df.groupby(['Layer', 'RPM_Group']).ngroups
        prog = st.progress(0)
        g_idx = 0

        for (layer, rpm_grp), gdata in df.groupby(['Layer', 'RPM_Group'], sort=True):
            g_idx += 1
            prog.progress(g_idx / total_groups)

            lbl = f"{layer} / {rpm_grp}"
            if len(gdata) < 4:
                st.warning(f"Skipping '{lbl}': only {len(gdata)} rows.")
                continue

            gdata = gdata.copy()
            max_wob = gdata['WOB_klb'].max()
            min_wob = gdata['WOB_klb'].min()

            # ---- WOB binning (paper uses 3, 5, 7 klb bins) ----
            bins = np.arange(np.floor(min_wob), np.ceil(max_wob) + wob_bin, wob_bin)
            if len(bins) < 3:
                bins = np.linspace(min_wob, max_wob, 4)

            gdata['WOB_bin'] = pd.cut(gdata['WOB_klb'], bins=bins, include_lowest=True)
            agg = (gdata.groupby('WOB_bin', observed=True)
                   .agg(WOB_klb=('WOB_klb', 'mean'),
                        ROP_fthr=('ROP_fthr', 'mean'),
                        MSE_kpsi=('MSE_kpsi', 'mean'),
                        Count=('ROP_fthr', 'count'))
                   .dropna()
                   .reset_index(drop=True))

            agg = agg[agg['Count'] >= min_pts_per_bin].reset_index(drop=True)

            if len(agg) < 3:
                st.warning(f"'{lbl}': insufficient WOB bins ({len(agg)}). "
                           "Try reducing WOB bin size or min-points threshold.")
                continue

            agg = agg.sort_values('WOB_klb').reset_index(drop=True)
            x = agg['WOB_klb'].values
            y_rop = agg['ROP_fthr'].values
            y_mse = agg['MSE_kpsi'].values

            # ---- Fit 2nd-degree polynomials (paper Sec. 3.3) ----
            poly_rop = fit_poly2(x, y_rop)
            poly_mse = fit_poly2(x, y_mse)

            x_fine = np.linspace(x.min(), x.max(), 400)
            y_rop_fine = poly_rop(x_fine)
            y_mse_fine = poly_mse(x_fine)

            # ---- Founder point (Region III onset) ----
            fp_wob, fp_rop = founder_point(poly_rop, x.min(), x.max())
            fp_mse = float(poly_mse(fp_wob))

            # ---- Desirability optimization ----
            rop_L = y_rop_fine.min();   rop_U = y_rop_fine.max()
            mse_L = y_mse_fine.min();   mse_U = y_mse_fine.max()

            des_vals = np.array([
                desirability_D(r, m, rop_L, rop_U, mse_L, mse_U, w_rop, w_mse)
                for r, m in zip(y_rop_fine, y_mse_fine)
            ])
            idx_opt     = np.argmax(des_vals)
            wob_opt     = x_fine[idx_opt]
            rop_opt     = y_rop_fine[idx_opt]
            mse_opt     = y_mse_fine[idx_opt]
            des_opt     = des_vals[idx_opt]

            # ---- UCS comparison (efficiency metric) ----
            # Per Teale (1965): MSE ≈ UCS at maximum efficiency
            mse_uccs_ratio_founder = fp_mse / ucs_avg_kpsi if ucs_avg_kpsi > 0 else np.nan
            mse_uccs_ratio_opt     = mse_opt  / ucs_avg_kpsi if ucs_avg_kpsi > 0 else np.nan

            # ---- Cost per foot ----
            cost_per_ft_historical = rig_hr / gdata['ROP_fthr'].mean() if gdata['ROP_fthr'].mean() > 0 else np.nan
            cost_per_ft_opt        = rig_hr / rop_opt if rop_opt > 0 else np.nan

            # ---- Avg depth ----
            avg_depth = gdata['Depth_ft'].mean()

            # ---- Store results ----
            results.append({
                'Layer':            layer,
                'RPM_Group':        rpm_grp,
                'Avg_Depth_ft':     round(avg_depth, 1),
                'N_points':         len(gdata),
                'WOB_founder_klb':  round(fp_wob, 2),
                'ROP_founder_fthr': round(fp_rop, 2),
                'MSE_founder_kpsi': round(fp_mse, 1),
                'WOB_opt_klb':      round(wob_opt, 2),
                'ROP_opt_fthr':     round(rop_opt, 2),
                'MSE_opt_kpsi':     round(mse_opt, 1),
                'Desirability_D':   round(des_opt, 4),
                'MSE_UCS_founder':  round(mse_uccs_ratio_founder, 3) if not np.isnan(mse_uccs_ratio_founder) else np.nan,
                'MSE_UCS_opt':      round(mse_uccs_ratio_opt, 3)     if not np.isnan(mse_uccs_ratio_opt) else np.nan,
                'Cost_per_ft_hist_USD': round(cost_per_ft_historical, 2) if not np.isnan(cost_per_ft_historical) else np.nan,
                'Cost_per_ft_opt_USD':  round(cost_per_ft_opt, 2)        if not np.isnan(cost_per_ft_opt) else np.nan,
                # store poly coefficients as JSON string for later use
                '_rop_coeffs':      poly_rop.coefficients.tolist(),
                '_mse_coeffs':      poly_mse.coefficients.tolist(),
                '_wob_min':         float(x.min()),
                '_wob_max':         float(x.max()),
                '_agg_wob':         x.tolist(),
                '_agg_rop':         y_rop.tolist(),
                '_agg_mse':         y_mse.tolist(),
            })

        prog.progress(1.0)

        if not results:
            st.error("No optimization results generated. Check data, RPM centres, and filter settings.")
            st.stop()

        results_df = pd.DataFrame(results)
        st.session_state.results_df  = results_df
        st.session_state.opt_done    = True

# ============================================================
#  Display Results
# ============================================================
if st.session_state.opt_done and st.session_state.results_df is not None:
    results_df = st.session_state.results_df
    df         = st.session_state.working_df
    ucs_avg_kpsi_disp = ucs_avg / 1000.0

    # ---- Summary KPIs ----
    st.subheader("📈 Optimization Results")
    best_row = results_df.loc[results_df['Desirability_D'].idxmax()]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Best Desirability D", f"{best_row['Desirability_D']:.4f}")
    k2.metric("Optimal WOB", f"{best_row['WOB_opt_klb']:.1f} klb",
              help="WOB at maximum desirability")
    k3.metric("Optimal ROP", f"{best_row['ROP_opt_fthr']:.1f} ft/hr")
    k4.metric("Optimal MSE", f"{best_row['MSE_opt_kpsi']:.0f} kpsi",
              delta=f"UCS ratio {best_row['MSE_UCS_opt']:.2f}" if not pd.isna(best_row['MSE_UCS_opt']) else "")
    k5.metric("Formation UCS (avg)", f"{ucs_avg_kpsi_disp:.0f} kpsi",
              help="Rock UCS used as efficiency benchmark (Teale 1965)")

    # ---- Results Table ----
    display_cols = [c for c in results_df.columns if not c.startswith('_')]
    st.dataframe(results_df[display_cols], use_container_width=True)

    csv_bytes = results_df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results CSV", csv_bytes,
                        "MODO_results.csv", "text/csv")

    # ================================================================
    #  DRT Plots — one per (Layer × RPM_Group)
    # ================================================================
    st.subheader("📉 Drill Rate Test Curves (ROP & MSE vs WOB)")
    st.markdown(
        "Each chart shows the fitted **ROP vs WOB** (blue) and **MSE vs WOB** (red dashed) curves. "
        "Orange line = Founder point (Region III onset). Green line = Optimal WOB from desirability."
    )

    for _, row in results_df.iterrows():
        lbl = f"{row['Layer']} / {row['RPM_Group']}"
        with st.expander(f"📊 {lbl}  |  D={row['Desirability_D']:.4f}  "
                          f"|  WOB_opt={row['WOB_opt_klb']:.1f} klb  "
                          f"|  ROP_opt={row['ROP_opt_fthr']:.1f} ft/hr", expanded=True):

            # Reconstruct curves
            x_fine = np.linspace(row['_wob_min'], row['_wob_max'], 400)
            poly_rop = np.poly1d(row['_rop_coeffs'])
            poly_mse = np.poly1d(row['_mse_coeffs'])
            y_rop_fine = poly_rop(x_fine)
            y_mse_fine = poly_mse(x_fine)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"DRT — {lbl}", fontsize=12, fontweight='bold')

            # --- Left: ROP vs WOB ---
            ax1.scatter(row['_agg_wob'], row['_agg_rop'],
                        color='#1565c0', s=60, zorder=5, label='Data (binned avg)')
            ax1.plot(x_fine, y_rop_fine, color='#1565c0', lw=2, label='Poly fit')
            ax1.axvline(row['WOB_founder_klb'], color='darkorange', ls=':', lw=2,
                        label=f"Founder {row['WOB_founder_klb']:.1f} klb")
            ax1.axvline(row['WOB_opt_klb'], color='green', ls='-.', lw=2.5,
                        label=f"Optimal {row['WOB_opt_klb']:.1f} klb")
            ax1.set_xlabel('WOB (klb)', fontsize=10)
            ax1.set_ylabel('ROP (ft/hr)', fontsize=10, color='#1565c0')
            ax1.set_title('ROP vs WOB', fontsize=10)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            # Annotate regions
            x_mid = (x_fine.min() + row['WOB_founder_klb']) / 2
            ax1.text(x_mid, ax1.get_ylim()[0] * 1.02 if ax1.get_ylim()[0] > 0
                     else max(y_rop_fine) * 0.05,
                     "Region II\n(efficient)",
                     ha='center', fontsize=7, color='green',
                     bbox=dict(boxstyle='round,pad=0.2', fc='#e8f5e9', ec='green', alpha=0.7))

            # --- Right: MSE vs WOB ---
            ax2.scatter(row['_agg_wob'], row['_agg_mse'],
                        color='#b71c1c', marker='s', s=60, zorder=5, label='Data (binned avg)')
            ax2.plot(x_fine, y_mse_fine, color='#b71c1c', lw=2, ls='--', label='Poly fit')
            ax2.axhline(ucs_avg_kpsi_disp, color='purple', ls=':', lw=1.5,
                        label=f"UCS avg {ucs_avg_kpsi_disp:.0f} kpsi")
            ax2.axvline(row['WOB_founder_klb'], color='darkorange', ls=':', lw=2,
                        label=f"Founder {row['WOB_founder_klb']:.1f} klb")
            ax2.axvline(row['WOB_opt_klb'], color='green', ls='-.', lw=2.5,
                        label=f"Optimal {row['WOB_opt_klb']:.1f} klb")
            ax2.set_xlabel('WOB (klb)', fontsize=10)
            ax2.set_ylabel('MSE (kpsi)', fontsize=10, color='#b71c1c')
            ax2.set_title('MSE vs WOB', fontsize=10)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ---- Interpretation ----
            eff_label = "✅ Efficient" if row['MSE_UCS_opt'] <= 2.0 else "⚠️ Inefficient"
            ratio_str = f"{row['MSE_UCS_opt']:.2f}" if not pd.isna(row['MSE_UCS_opt']) else "N/A"
            st.caption(
                f"**{eff_label}** — Optimal MSE/UCS ratio = {ratio_str} "
                f"(ratio ≤ 1 means drilling energy ≈ rock strength → maximum efficiency per Teale 1965). "
                f"Founder WOB = {row['WOB_founder_klb']:.1f} klb / "
                f"Optimal WOB = {row['WOB_opt_klb']:.1f} klb / "
                f"Desirability D = {row['Desirability_D']:.4f}"
            )

    # ================================================================
    #  MSE vs ROP scatter (Figure 2 from paper)
    # ================================================================
    st.subheader("🔵 MSE vs ROP Efficiency Plot (Fig. 2 from paper)")
    st.markdown(
        "Points in **Region III** (high ROP, low MSE) indicate efficient drilling. "
        "Points in **Region I** (low ROP, high MSE) indicate inefficiency (over-WOB)."
    )

    fig2, ax = plt.subplots(figsize=(10, 6))
    layers_in_data = df['Layer'].unique()
    cmap = plt.colormaps['tab10'].resampled(max(len(layers_in_data), 1))
    for i, lay in enumerate(layers_in_data):
        sub = df[df['Layer'] == lay]
        ax.scatter(sub['ROP_fthr'], sub['MSE_kpsi'],
                   alpha=0.35, s=12, color=cmap(i), label=lay)

    # Add optimal points
    for _, row in results_df.iterrows():
        ax.scatter(row['ROP_opt_fthr'], row['MSE_opt_kpsi'],
                   s=140, marker='*', color='gold', edgecolors='black', zorder=10,
                   linewidths=0.8)

    ax.axhline(ucs_avg_kpsi_disp, color='purple', ls='--', lw=1.5,
               label=f"UCS avg ({ucs_avg_kpsi_disp:.0f} kpsi) — ideal MSE target")
    ax.set_xlabel('ROP (ft/hr)', fontsize=11)
    ax.set_ylabel('MSE (kpsi)', fontsize=11)
    ax.set_title('MSE vs ROP — Drilling Efficiency Map', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    # Annotate regions
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.text(xlim[0] + (xlim[1]-xlim[0])*0.05, ylim[1]*0.9,
            "Region I\n(Inefficient)", fontsize=9, color='red',
            bbox=dict(boxstyle='round', fc='#ffebee', ec='red', alpha=0.8))
    ax.text(xlim[1]*0.7, ylim[0] + (ylim[1]-ylim[0])*0.05,
            "Region III\n(Efficient)", fontsize=9, color='green',
            bbox=dict(boxstyle='round', fc='#e8f5e9', ec='green', alpha=0.8))
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ================================================================
    #  2D Depth Logs (Figure 3 from paper)
    # ================================================================
    st.subheader("📏 2D Depth Logs (WOB · RPM · ROP · MSE)")
    st.markdown("Reproduces Figure 3 of the paper: input and response variables as a function of depth.")

    fig3, axes = plt.subplots(1, 4, figsize=(16, 9), sharey=True)
    fig3.suptitle('Drilling Variables vs Depth', fontsize=13, fontweight='bold')

    plot_vars = [
        ('WOB_klb',   'WOB (klb)',   '#1565c0'),
        ('RPM',       'RPM',          '#2e7d32'),
        ('ROP_fthr',  'ROP (ft/hr)', '#f57f17'),
        ('MSE_kpsi',  'MSE (kpsi)',  '#b71c1c'),
    ]
    for ax, (col, lbl, clr) in zip(axes, plot_vars):
        ax.scatter(df[col], df['Depth_ft'], s=2, alpha=0.4, color=clr)
        ax.axvline(df[col].mean(), color='black', ls='--', lw=1,
                   label=f'Mean {df[col].mean():.1f}')
        ax.set_xlabel(lbl, fontsize=9)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('Depth (ft)', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # ================================================================
    #  Cost / Time Savings Analysis
    # ================================================================
    st.subheader("💰 Cost & Time Savings Analysis")
    st.markdown(
        "Select a reference zone (depth interval) and compare actual drilling performance "
        "against the optimal parameters from the DRT optimization."
    )

    results_df['Group_Label'] = results_df['Layer'] + ' / ' + results_df['RPM_Group']
    sel_grp = st.selectbox("Select Layer/RPM group:", results_df['Group_Label'].unique())

    if sel_grp:
        sel_row = results_df[results_df['Group_Label'] == sel_grp].iloc[0]
        layer_mask = (df['Layer'] == sel_row['Layer']) & (df['RPM_Group'] == sel_row['RPM_Group'])
        grp_data = df[layer_mask]

        if len(grp_data) > 0:
            d_min = float(grp_data['Depth_ft'].min())
            d_max = float(grp_data['Depth_ft'].max())

            col_a, col_b = st.columns(2)
            with col_a:
                z_start = st.number_input("Zone start depth (ft)", value=d_min, step=1.0)
            with col_b:
                z_end   = st.number_input("Zone end depth (ft)",   value=d_max, step=1.0)

            if st.button("📊 Calculate Savings"):
                zone = grp_data[(grp_data['Depth_ft'] >= z_start) &
                                (grp_data['Depth_ft'] <= z_end)]
                if len(zone) < 2:
                    st.warning("Not enough data in that depth interval.")
                else:
                    # Distance drilled
                    dist_ft = zone['Depth_ft'].max() - zone['Depth_ft'].min()
                    avg_rop_hist = zone['ROP_fthr'].mean()
                    rop_opt_val  = sel_row['ROP_opt_fthr']

                    t_hist = dist_ft / avg_rop_hist if avg_rop_hist > 0 else np.nan
                    t_opt  = dist_ft / rop_opt_val  if rop_opt_val  > 0 else np.nan
                    t_saved = t_hist - t_opt if not (np.isnan(t_hist) or np.isnan(t_opt)) else np.nan
                    cost_saved = t_saved * rig_hr if not np.isnan(t_saved) else np.nan

                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Drilled interval", f"{dist_ft:.1f} ft ({dist_ft*0.3048:.1f} m)")
                    r2.metric("Historical avg ROP", f"{avg_rop_hist:.1f} ft/hr")
                    r3.metric("Optimal ROP", f"{rop_opt_val:.1f} ft/hr")
                    r4.metric("Time saved", f"{max(0, t_saved):.2f} hrs" if not np.isnan(t_saved) else "N/A",
                              delta=f"${max(0, cost_saved):,.0f} saved" if not np.isnan(cost_saved) else "")

                    if not np.isnan(cost_saved) and cost_saved > 0:
                        st.success(
                            f"⏱️ **Time saved: {max(0,t_saved):.2f} hrs** | "
                            f"💰 **Cost saved: ${max(0,cost_saved):,.0f}** "
                            f"(at ${rig_hr:,.0f}/hr rig rate)"
                        )
                    else:
                        st.info("Optimal ROP is similar to historical ROP — no significant savings expected.")

    # ================================================================
    #  Efficiency Classification Table
    # ================================================================
    st.subheader("📋 Drilling Efficiency Classification")
    st.markdown(
        f"Based on MSE/UCS ratio (Teale 1965). UCS used: **{ucs_avg_kpsi_disp:.0f} kpsi** "
        f"({rock_name}). Ratio < 1.5 → efficient; 1.5–3 → moderate; > 3 → inefficient."
    )
    eff_rows = []
    for _, row in results_df.iterrows():
        ratio = row['MSE_UCS_opt']
        if pd.isna(ratio):
            cls = "Unknown"
        elif ratio < 1.5:
            cls = "✅ Efficient"
        elif ratio < 3.0:
            cls = "⚠️ Moderately efficient"
        else:
            cls = "❌ Inefficient"
        eff_rows.append({
            'Layer / RPM':    row['Group_Label'],
            'MSE opt (kpsi)': row['MSE_opt_kpsi'],
            'UCS avg (kpsi)': round(ucs_avg_kpsi_disp, 0),
            'MSE/UCS ratio':  round(ratio, 3) if not pd.isna(ratio) else "N/A",
            'Classification': cls,
            'WOB opt (klb)':  row['WOB_opt_klb'],
            'ROP opt (ft/hr)':row['ROP_opt_fthr'],
            'Desirability D': row['Desirability_D'],
        })
    st.dataframe(pd.DataFrame(eff_rows), use_container_width=True)

else:
    st.info("Upload a file and click **Run DRT Optimization** to begin.")
