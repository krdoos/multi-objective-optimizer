# -*- coding: utf-8 -*-
"""
Multi-Objective Drilling Optimizer Tool
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import warnings

# Attempt to import the shape‑preserving PCHIP interpolator from SciPy.  If it
# is unavailable (for example, if SciPy is not installed), we fall back to
# polynomial fitting.  This interpolation method produces monotonic curves
# through the data points and often yields more realistic fits than simple
# quadratic regression, especially when the underlying relationship is not
# well approximated by a parabola.
try:
    from scipy.interpolate import PchipInterpolator
    HAS_PCHIP = True
except Exception:
    HAS_PCHIP = False
warnings.filterwarnings('ignore')

# ====================================================================
# ======================= Rock Properties Data ========================
# ====================================================================

# Dictionary of common rock types and their properties.  Each entry
# contains a range for unconfined compressive strength (UCS) in MPa
# and recommended ranges for weight‑on‑bit (WOB) and RPM.  These values
# can be adjusted based on field data or literature.  The average
# UCS will be used to compare against calculated MSE to evaluate
# drilling efficiency.
rock_properties = {
    'Granite': {
        'UCS_MPa': (100, 250),
        'WOB_range_klb': (20, 35),
        'RPM_range': (100, 150)
    },
    'Sandstone': {
        'UCS_MPa': (20, 170),
        'WOB_range_klb': (8, 15),
        'RPM_range': (80, 120)
    },
    'Shale': {
        'UCS_MPa': (5, 100),
        'WOB_range_klb': (4, 8),
        'RPM_range': (60, 90)
    },
    'Limestone': {
        'UCS_MPa': (30, 250),
        'WOB_range_klb': (10, 20),
        'RPM_range': (80, 130)
    },
    'Dolomite': {
        'UCS_MPa': (30, 250),
        'WOB_range_klb': (10, 20),
        'RPM_range': (80, 130)
    }
}

# Conversion factor from MPa to psi for UCS comparison
MPA_TO_PSI = 145.038

def get_ucs_psi(rock_type):
    """Return average UCS in psi for the given rock type.

    If the rock type exists in the rock_properties dictionary, the
    average of the UCS_MPa range is converted to psi.  Otherwise
    returns None.
    """
    props = rock_properties.get(rock_type)
    if props and 'UCS_MPa' in props:
        ucs_range = props['UCS_MPa']
        avg_mpa = np.mean(ucs_range)
        return avg_mpa * MPA_TO_PSI
    return None

# ====================================================================
# ====================== Helper Functions ============================
# ====================================================================

def clean_column_name(col):
    if isinstance(col, str):
        return col.strip().lower()
    return str(col).strip().lower()

def find_best_match(columns, patterns):
    columns_lower = [clean_column_name(col) for col in columns]
    for pattern in patterns:
        pattern_lower = pattern.lower()
        for i, col_lower in enumerate(columns_lower):
            if pattern_lower in col_lower:
                return columns[i]
    return None

def convert_wob_to_lbs(series, col_name):
    col_lower = clean_column_name(col_name)
    if 'tonnes' in col_lower or 'ton' in col_lower:
        st.info("ℹ️ Converting WOB from tonnes to lbs (multiply by 2204.62)")
        return series * 2204.62
    elif 'kg' in col_lower:
        st.info("ℹ️ Converting WOB from kg to lbs (multiply by 2.20462)")
        return series * 2.20462
    elif 'klbs' in col_lower or 'k-lbs' in col_lower:
        st.info("ℹ️ WOB is in klbs, converting to lbs (multiply by 1000)")
        return series * 1000
    elif 'lbs' in col_lower:
        return series
    else:
        st.warning("⚠️ WOB unit unknown. Assuming tonnes.")
        return series * 2204.62

def convert_torque_to_ftlbs(series, col_name):
    col_lower = clean_column_name(col_name)
    if 'kft-lb' in col_lower or 'kftlb' in col_lower or 'trq_avg' in col_lower:
        st.info("ℹ️ Torque in kft-lb, converting to ft-lb (multiply by 1000)")
        return series * 1000
    elif 'ft-lb' in col_lower or 'ftlb' in col_lower:
        return series
    elif 'psi' in col_lower:
        st.error("⚠️ Torque column appears to be in PSI. Results will be WRONG!")
        return series * 10  # تقدير تقريبي فقط
    else:
        st.warning("⚠️ Torque unit unknown. Assuming kft-lb.")
        return series * 1000

def convert_rop(series, col_name, target_unit):
    col_lower = clean_column_name(col_name)
    if 'm/hr' in col_lower or 'm/h' in col_lower:
        if target_unit == 'ft/hr':
            st.info("ℹ️ Converting ROP from m/hr to ft/hr")
            return series * 3.28084
        else:
            return series
    elif 'ft/hr' in col_lower or 'ft/h' in col_lower:
        if target_unit == 'm/hr':
            st.info("ℹ️ Converting ROP from ft/hr to m/hr")
            return series / 3.28084
        else:
            return series
    else:
        st.warning(f"⚠️ ROP unit unknown. Assuming {target_unit}.")
        return series

def calculate_mse(wob_lbs, rpm, torque_ftlbs, rop_fthr, bit_area_sqin):
    rop_fthr = np.where(rop_fthr <= 0, 0.1, rop_fthr)
    term1 = wob_lbs / bit_area_sqin
    term2 = (120 * np.pi * rpm * torque_ftlbs) / (bit_area_sqin * rop_fthr)
    return term1 + term2

def safe_polyfit(x, y, max_degree=2):
    for degree in range(max_degree, 0, -1):
        try:
            coeffs = np.polyfit(x, y, degree)
            return np.poly1d(coeffs), degree
        except:
            continue
    return np.poly1d([0, y.mean()]), 0

def desirability(rop_pred, mse_pred, rop_min, rop_max, mse_min, mse_max):
    d_rop = (rop_pred - rop_min) / (rop_max - rop_min) if rop_max > rop_min else 1
    d_mse = (mse_max - mse_pred) / (mse_max - mse_min) if mse_max > mse_min else 1
    d_rop = np.clip(d_rop, 0, 1)
    d_mse = np.clip(d_mse, 0, 1)
    return np.sqrt(d_rop * d_mse)

def find_founder_point(wob_klb, rop_fthr, degree=2):
    coeffs = np.polyfit(wob_klb, rop_fthr, degree)
    poly = np.poly1d(coeffs)
    deriv = poly.deriv()
    roots = deriv.roots
    real_roots = [r.real for r in roots if np.isreal(r) and wob_klb.min() <= r.real <= wob_klb.max()]
    if real_roots:
        return max(real_roots)
    else:
        wob_fine = np.linspace(wob_klb.min(), wob_klb.max(), 200)
        deriv_vals = deriv(wob_fine)
        idx = np.argmin(np.abs(deriv_vals))
        return wob_fine[idx]

# ====================================================================
# ====================== Streamlit App ==============================
# ====================================================================

st.set_page_config(page_title="MODO Tool", layout="wide")
st.title("Multi-Objective Drilling Optimizer Tool")
st.markdown("This app optimizes drilling parameters (WOB, RPM) to maximize ROP while minimizing MSE using desirability function.")

# تهيئة session state
if 'optimization_done' not in st.session_state:
    st.session_state.optimization_done = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'df' not in st.session_state:
    st.session_state.df = None

# ---------------------- الشريط الجانبي (معاملات الحفارة) ----------------------
st.sidebar.header("Drilling Parameters")
bit_diameter_in = st.sidebar.number_input("Bit Diameter (inches)", value=17.5, step=0.1, format="%.2f")
rig_rate = st.sidebar.number_input("Rig Hourly Rate (USD/hr)", value=15000, step=100, format="%d")
wob_bin_size = st.sidebar.number_input("WOB bin size (klb) for grouping", value=3.0, step=0.5, format="%.2f")

# ----------- Rock properties selection ---------------
st.sidebar.header("Rock Properties")
# Provide a dropdown for selecting the current rock/formation type.  This
# list is derived from the rock_properties dictionary defined above.
rock_type = st.sidebar.selectbox(
    "Rock Type (Layer)",
    options=list(rock_properties.keys()),
    index=0
)
# Retrieve the UCS in psi for the selected rock type
ucs_psi = get_ucs_psi(rock_type)
# Display information about the selected rock type to the user
if ucs_psi:
    props = rock_properties.get(rock_type, {})
    wob_range = props.get('WOB_range_klb', (None, None))
    rpm_range = props.get('RPM_range', (None, None))
    st.sidebar.markdown(
        f"**Average UCS:** {ucs_psi:.0f} psi\n\n"
        f"**Recommended WOB:** {wob_range[0]}–{wob_range[1]} klb\n\n"
        f"**Recommended RPM:** {rpm_range[0]}–{rpm_range[1]} rev/min"
    )

# Choice of curve fitting method.  Quadratic (2nd‑degree polynomial) is the
# default method described in the paper【436714205699149†L446-L462】.  PCHIP is a
# shape‑preserving interpolator that can provide smoother, more realistic fits
# for noisy data.  It is only available if SciPy is installed.
fitting_options = ["Quadratic"]
if HAS_PCHIP:
    fitting_options.append("PCHIP (shape‑preserving)")
fit_method = st.sidebar.selectbox(
    "Curve fitting method", fitting_options,
    index=0,
    help="Choose how to fit ROP and MSE against WOB. Quadratic fits a 2nd‑degree polynomial. PCHIP provides a monotonic curve through the data (if SciPy is available)."
)

# ---------------------- تحميل الملف ----------------------
uploaded_file = st.file_uploader("Upload drilling data file (Excel or CSV)", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    # قراءة الملف (يتم فقط إذا كان ملف جديد أو لم يتم التحميل من قبل)
    if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df_raw = pd.read_excel(uploaded_file)
            else:
                try:
                    df_raw = pd.read_csv(uploaded_file, encoding='utf-8-sig', sep=None, engine='python')
                except:
                    df_raw = pd.read_csv(uploaded_file, encoding='latin-1', sep=None, engine='python')
            st.session_state.df_raw = df_raw
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.optimization_done = False  # إعادة تعيين حالة التحسين عند رفع ملف جديد
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        df_raw = st.session_state.df_raw

    st.success(f"File loaded: {uploaded_file.name} – {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")
    st.subheader("Data Preview")
    st.dataframe(df_raw.head(10))

    # ====================== تحديد الأعمدة الرئيسية ======================
    st.subheader("🔍 Column Mapping")
    st.markdown("The app will try to automatically detect the required columns. Please verify below.")

    depth_patterns = ['depth', 'dept', 'depth(ft)', 'depth(m)', 'measured depth', 'md']
    rop_patterns   = ['rop', 'rop(m/hr)', 'rop (m/hr)', 'rop_mhr', 'rop(1 m)', 'rop(1 ft)']
    wob_patterns   = ['wob', 'weight on bit', 'wob (tonnes)', 'wob (k-lbs)', 'wob (klbs)']
    rpm_patterns   = ['rpm', 'rotary speed', 'rev/min', 'rotary']
    torque_patterns = ['trq_avg', 'torque', 'trq', 'kft-lb', 'torque (kft-lb)']

    cols = list(df_raw.columns)
    detected_depth = find_best_match(cols, depth_patterns)
    detected_rop   = find_best_match(cols, rop_patterns)
    detected_wob   = find_best_match(cols, wob_patterns)
    detected_rpm   = find_best_match(cols, rpm_patterns)
    detected_torque = find_best_match(cols, torque_patterns)

    # Attempt to detect a column that represents the geological layer, lithology or formation.  This
    # information can be used to segment the analysis by rock type.  The user can override or
    # select "None" if no such column exists.
    layer_patterns = ['layer', 'lithology', 'formation', 'rock type', 'rock_type', 'zone', 'section']
    detected_layer = find_best_match(cols, layer_patterns)


    col1, col2 = st.columns(2)
    with col1:
        depth_col = st.selectbox("Depth column", cols, index=cols.index(detected_depth) if detected_depth else 0)
        rop_col   = st.selectbox("ROP column", cols, index=cols.index(detected_rop) if detected_rop else 0)
        wob_col   = st.selectbox("WOB column", cols, index=cols.index(detected_wob) if detected_wob else 0)
        # Select optional layer/lithology column.  Include a 'None' option when no column should be used.
        layer_options = ['None'] + cols
        # Determine default index: if a layer column was detected, use that; otherwise 'None'
        default_layer_idx = layer_options.index(detected_layer) if detected_layer and detected_layer in layer_options else 0
        layer_col = st.selectbox(
            "Layer/Lithology column (optional)", layer_options, index=default_layer_idx,
            help="Choose a column that contains lithology or rock type for each row. Use 'None' if such data is not available."
        )
    with col2:
        rpm_col   = st.selectbox("RPM column", cols, index=cols.index(detected_rpm) if detected_rpm else 0)
        torque_col = st.selectbox("Torque column", cols, index=cols.index(detected_torque) if detected_torque else 0)

    # ====================== إعدادات الفلترة ======================
    st.subheader("🎛️ Data Filter Settings")
    rop_unit = st.radio("ROP unit in file", ('m/hr', 'ft/hr'), horizontal=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        min_rop_val = st.number_input(f"Min ROP ({rop_unit})", value=1.0, step=0.1, format="%.2f")
    with col2:
        default_max = 400.0 if rop_unit == 'm/hr' else 1200.0
        max_rop_val = st.number_input(f"Max ROP ({rop_unit})", value=default_max, step=1.0, format="%.2f")
    with col3:
        min_wob_lbs = st.number_input("Min WOB (lbs)", value=2000, step=100, format="%d")

    # تحديد وحدة الإخراج (نفس وحدة الإدخال)
    output_unit = rop_unit

    # ====================== تحليل RPM ======================
    st.subheader("🔄 Multi-RPM Analysis")
    rpm_option = st.radio("Group data by RPM?", ('Yes', 'No'), horizontal=True, index=0)

    rpm_groups = None
    if rpm_option == 'Yes':
        method = st.radio("Grouping method", ('Fixed values', 'Number of bins'), horizontal=True)
        if method == 'Fixed values':
            rpm_vals = st.text_input("Enter RPM values separated by commas (e.g., 110,120,130,140,150)", "110,120,130,140,150")
            rpm_breaks = [float(x.strip()) for x in rpm_vals.split(',') if x.strip()]
            if not rpm_breaks:
                rpm_breaks = [110,120,130,140,150]
        else:
            num_bins = st.number_input("Number of bins", min_value=2, max_value=10, value=5, step=1)

    # ====================== زر التشغيل ======================
    if st.button("🚀 Run Optimization", type="primary"):
        with st.spinner("Processing data and optimizing..."):
            # بناء DataFrame موحد
            df = pd.DataFrame()
            df['Depth'] = pd.to_numeric(df_raw[depth_col], errors='coerce')
            df['ROP_original'] = pd.to_numeric(df_raw[rop_col], errors='coerce')
            df['WOB_lbs'] = convert_wob_to_lbs(pd.to_numeric(df_raw[wob_col], errors='coerce'), wob_col)
            df['RPM'] = pd.to_numeric(df_raw[rpm_col], errors='coerce')
            df['Torque_ftlbs'] = convert_torque_to_ftlbs(pd.to_numeric(df_raw[torque_col], errors='coerce'), torque_col)

            # إزالة الصفوف الفارغة
            initial_len = len(df)
            df.dropna(subset=['Depth', 'ROP_original', 'WOB_lbs', 'RPM', 'Torque_ftlbs'], inplace=True)
            st.info(f"Removed {initial_len - len(df)} rows with missing values. Remaining: {len(df)}")

            # الفلترة حسب ROP و WOB
            if rop_unit == 'm/hr':
                df['ROP_mhr'] = df['ROP_original']
                filter_col = 'ROP_mhr'
                min_rop = min_rop_val
                max_rop = max_rop_val
            else:
                df['ROP_fthr'] = df['ROP_original']
                filter_col = 'ROP_fthr'
                min_rop = min_rop_val
                max_rop = max_rop_val

            initial_count = len(df)
            df = df[(df[filter_col] >= min_rop) & (df[filter_col] <= max_rop) &
                    (df['WOB_lbs'] >= min_wob_lbs) & (df['RPM'] > 0) & (df['Torque_ftlbs'] > 0)]
            st.info(f"Removed {initial_count - len(df)} invalid rows. Remaining: {len(df)}")

            if len(df) < 5:
                st.error("Insufficient data after filtering. Adjust filter parameters.")
                st.stop()

            # توحيد ROP إلى ft/hr للحسابات الداخلية (لأن معادلة MSE تتطلب ft/hr)
            if rop_unit == 'm/hr':
                df['ROP_fthr'] = df['ROP_mhr'] * 3.28084
            else:
                df['ROP_fthr'] = df['ROP_fthr']

            # حساب MSE
            bit_area = np.pi * (bit_diameter_in / 2)**2
            df['MSE_psi'] = calculate_mse(df['WOB_lbs'], df['RPM'], df['Torque_ftlbs'], df['ROP_fthr'], bit_area)
            df['MSE_psi'] = df['MSE_psi'].clip(lower=0)

            # تعيين عمود الطبقة لكل صف.  إذا كان المستخدم قد اختار عموداً في ملف البيانات
            # ليكون عمود الطبقة، يُستخدم هذا العمود بعد تحويله إلى نص.  خلاف ذلك
            # يتم استخدام نوع الصخر المختار في الشريط الجانبي لكل الصفوف.
            if 'layer_col' in locals() and layer_col != 'None':
                try:
                    df['Layer'] = df_raw[layer_col].astype(str)
                except Exception:
                    df['Layer'] = str(rock_type)
            else:
                df['Layer'] = str(rock_type)

            # حساب مقاومة الضغط غير المحصور لكل صف بناءً على نوع الطبقة
            try:
                df['UCS_psi_layer'] = df['Layer'].apply(lambda x: get_ucs_psi(str(x)))
            except Exception:
                df['UCS_psi_layer'] = np.nan

            # حساب نسبة MSE إلى UCS لكل صف (قد تكون NaN إذا لم تتوفر قيمة UCS)
            df['MSE_ratio'] = df['MSE_psi'] / df['UCS_psi_layer']

            # تجهيز عمود المجموعة RPM
            if rpm_option == 'Yes':
                if method == 'Fixed values':
                    labels = []
                    for i in range(len(rpm_breaks)):
                        if i == 0:
                            labels.append(f"≤{rpm_breaks[0]}")
                        elif i < len(rpm_breaks):
                            labels.append(f"{rpm_breaks[i-1]}-{rpm_breaks[i]}")
                    labels.append(f">{rpm_breaks[-1]}")
                    df['RPM_Group'] = pd.cut(df['RPM'], bins=[-np.inf] + rpm_breaks + [np.inf], labels=labels)
                else:
                    df['RPM_Group'] = pd.cut(df['RPM'], bins=num_bins)
            else:
                df['RPM_Group'] = 'All Data'

            st.write("**RPM Groups created:**")
            st.dataframe(df['RPM_Group'].value_counts().reset_index().rename(columns={'index': 'Group', 'RPM_Group': 'Count'}))

            # ====================== التحسين لكل مجموعة ولطبقة ======================
            results = []
            progress_bar = st.progress(0)

            # Create grouping keys combining the geological layer and the RPM group.  This allows
            # separate optimization for each layer (if provided) and RPM group.  If no layer
            # column was specified, all rows share the same layer name (the selected rock_type).
            group_keys = list(df.groupby(['Layer', 'RPM_Group']).groups.keys())
            total_groups = len(group_keys)

            for i, (layer_name, rpm_group) in enumerate(group_keys):
                group_data = df[(df['Layer'] == layer_name) & (df['RPM_Group'] == rpm_group)].copy()
                group_label = f"{layer_name} / {rpm_group}"
                if len(group_data) < 4:
                    st.warning(f"Skipping group '{group_label}': insufficient data ({len(group_data)} points).")
                    progress_bar.progress((i + 1) / total_groups)
                    continue

                group_data['WOB_klb'] = group_data['WOB_lbs'] / 1000.0
                max_wob_klb = group_data['WOB_klb'].max()
                min_wob_klb = group_data['WOB_klb'].min()
                wob_range = max_wob_klb - min_wob_klb

                # تجميع مرن
                if wob_range < wob_bin_size * 1.5:
                    agg = group_data[['WOB_klb', 'ROP_fthr', 'MSE_psi']].copy()
                else:
                    current_bin_size = wob_bin_size
                    agg = None
                    while current_bin_size >= wob_bin_size / 4:
                        bins = np.arange(min_wob_klb, max_wob_klb + current_bin_size, current_bin_size)
                        group_data['WOB_bin'] = pd.cut(group_data['WOB_klb'], bins=bins, include_lowest=True)
                        temp_agg = group_data.groupby('WOB_bin', observed=True).agg({
                            'WOB_klb': 'mean',
                            'ROP_fthr': 'mean',
                            'MSE_psi': 'mean'
                        }).dropna().reset_index()
                        if len(temp_agg) >= 3:
                            agg = temp_agg
                            break
                        current_bin_size /= 2
                    if agg is None:
                        agg = group_data[['WOB_klb', 'ROP_fthr', 'MSE_psi']].copy()

                if len(agg) < 2:
                    continue

                # ---------------------- Curve fitting ------------------------------
                # Sort aggregated points by WOB for monotonic interpolation
                agg_sorted = agg.sort_values('WOB_klb')

                # Determine recommended WOB range for this layer.  If the layer exists in
                # rock_properties, use its recommended range; otherwise fall back to the
                # globally selected rock_type.
                wob_rec_range = None
                try:
                    if layer_name in rock_properties:
                        wob_rec_range = rock_properties[layer_name].get('WOB_range_klb')
                    elif rock_type in rock_properties:
                        wob_rec_range = rock_properties[rock_type].get('WOB_range_klb')
                except Exception:
                    wob_rec_range = None

                # Prepare fine WOB grid; restrict to recommended range if available
                if wob_rec_range and isinstance(wob_rec_range, (list, tuple)) and len(wob_rec_range) == 2:
                    rec_low, rec_high = wob_rec_range
                    wob_search_min = max(agg_sorted['WOB_klb'].min(), rec_low)
                    wob_search_max = min(agg_sorted['WOB_klb'].max(), rec_high)
                    if wob_search_min < wob_search_max:
                        wob_fine = np.linspace(wob_search_min, wob_search_max, 200)
                    else:
                        wob_fine = np.linspace(agg_sorted['WOB_klb'].min(), agg_sorted['WOB_klb'].max(), 200)
                else:
                    wob_fine = np.linspace(agg_sorted['WOB_klb'].min(), agg_sorted['WOB_klb'].max(), 200)

                # Fit curves using the selected method
                if 'fit_method' in locals() and fit_method.startswith('PCHIP') and HAS_PCHIP and len(agg_sorted) >= 3:
                    # Use shape‑preserving PCHIP interpolation
                    try:
                        pchip_rop = PchipInterpolator(agg_sorted['WOB_klb'], agg_sorted['ROP_fthr'])
                        pchip_mse = PchipInterpolator(agg_sorted['WOB_klb'], agg_sorted['MSE_psi'])
                        rop_fine = pchip_rop(wob_fine)
                        mse_fine = pchip_mse(wob_fine)
                        # Derivative of ROP curve to find founder point
                        rop_deriv = pchip_rop.derivative()(wob_fine)
                        # Founder WOB is where derivative magnitude is minimal (approaches zero)
                        idx_founder = np.argmin(np.abs(rop_deriv))
                        founder_wob_klb = wob_fine[idx_founder]
                        founder_rop_fthr = rop_fine[idx_founder]
                        founder_rop_mhr = founder_rop_fthr / 3.28084
                    except Exception:
                        # Fallback to polynomial fit if PCHIP fails
                        pchip_rop = None
                        pchip_mse = None
                        poly_rop, deg_rop = safe_polyfit(agg_sorted['WOB_klb'], agg_sorted['ROP_fthr'], max_degree=2)
                        poly_mse, deg_mse = safe_polyfit(agg_sorted['WOB_klb'], agg_sorted['MSE_psi'], max_degree=2)
                        rop_fine = poly_rop(wob_fine)
                        mse_fine = poly_mse(wob_fine)
                        founder_wob_klb = find_founder_point(agg_sorted['WOB_klb'], agg_sorted['ROP_fthr'], 2)
                        founder_rop_fthr = poly_rop(founder_wob_klb)
                        founder_rop_mhr = founder_rop_fthr / 3.28084
                else:
                    # Default quadratic fit
                    poly_rop, deg_rop = safe_polyfit(agg_sorted['WOB_klb'], agg_sorted['ROP_fthr'], max_degree=2)
                    poly_mse, deg_mse = safe_polyfit(agg_sorted['WOB_klb'], agg_sorted['MSE_psi'], max_degree=2)
                    rop_fine = poly_rop(wob_fine)
                    mse_fine = poly_mse(wob_fine)
                    founder_wob_klb = find_founder_point(agg_sorted['WOB_klb'], agg_sorted['ROP_fthr'], 2)
                    founder_rop_fthr = poly_rop(founder_wob_klb)
                    founder_rop_mhr = founder_rop_fthr / 3.28084

                # Normalize desirability parameters
                rop_min_fine, rop_max_fine = rop_fine.min(), rop_fine.max()
                mse_min_fine, mse_max_fine = mse_fine.min(), mse_fine.max()
                # Compute desirability across the fine grid
                des_fine = np.array([
                    desirability(r, m, rop_min_fine, rop_max_fine, mse_min_fine, mse_max_fine)
                    for r, m in zip(rop_fine, mse_fine)
                ])
                idx_opt = np.argmax(des_fine)
                wob_opt_klb = wob_fine[idx_opt]
                rop_opt_fthr = rop_fine[idx_opt]
                rop_opt_mhr = rop_opt_fthr / 3.28084
                mse_opt_psi = mse_fine[idx_opt]
                desirability_opt = des_fine[idx_opt]

                # Compute MSE at founder point for ratio calculation
                if 'fit_method' in locals() and fit_method.startswith('PCHIP') and HAS_PCHIP and len(agg_sorted) >= 3 and 'pchip_mse' in locals() and pchip_mse:
                    founder_mse_psi = pchip_mse(founder_wob_klb)
                else:
                    founder_mse_psi = float(mse_fine[idx_founder]) if 'idx_founder' in locals() else float(mse_fine[np.argmin(np.abs(wob_fine - founder_wob_klb))])

                # Determine UCS for this layer
                ucs_layer = get_ucs_psi(layer_name) or (get_ucs_psi(rock_type) if rock_type else None)
                if ucs_layer:
                    founder_mse_ratio = founder_mse_psi / ucs_layer
                    opt_mse_ratio = mse_opt_psi / ucs_layer
                else:
                    founder_mse_ratio = np.nan
                    opt_mse_ratio = np.nan

                # حساب متوسط العمق في المجموعة
                avg_depth = group_data['Depth'].mean()

                # تحويل ROP للوحدة المختارة للمخرجات
                if output_unit == 'm/hr':
                    founder_rop_output = founder_rop_mhr
                    opt_rop_output = rop_opt_mhr
                else:
                    founder_rop_output = founder_rop_fthr
                    opt_rop_output = rop_opt_fthr

                # حساب التكلفة لكل وحدة مسافة حسب الوحدة المختارة
                if output_unit == 'm/hr':
                    # cost per meter = rig_rate / (ROP in m/hr)
                    cost_founder = rig_rate / founder_rop_output if founder_rop_output > 0 else np.nan
                    cost_opt = rig_rate / opt_rop_output if opt_rop_output > 0 else np.nan
                else:
                    # cost per foot = rig_rate / (ROP in ft/hr)
                    cost_founder = rig_rate / founder_rop_output if founder_rop_output > 0 else np.nan
                    cost_opt = rig_rate / opt_rop_output if opt_rop_output > 0 else np.nan

                # حفظ النتائج، بما في ذلك نسب MSE/UCS عند نقطتي المؤسس والمثلى.  يتم تضمين اسم الطبقة
                # والمجموعة لتسهيل تحليل النتائج لاحقًا.
                results.append({
                    'Layer': layer_name,
                    'RPM_Group': rpm_group,
                    'Avg_Depth': avg_depth,
                    'WOB_founder_klb': founder_wob_klb,
                    f'ROP_founder_{output_unit}': founder_rop_output,
                    f'Cost_founder_USD_per_{output_unit.split("/")[0]}': cost_founder,
                    'WOB_opt_klb': wob_opt_klb,
                    f'ROP_opt_{output_unit}': opt_rop_output,
                    f'Cost_opt_USD_per_{output_unit.split("/")[0]}': cost_opt,
                    'MSE_founder_psi': founder_mse_psi,
                    'MSE_opt_psi': mse_opt_psi,
                    'Founder_MSE_ratio': founder_mse_ratio,
                    'Opt_MSE_ratio': opt_mse_ratio,
                    'Desirability': desirability_opt
                })

                # رسم بياني لكل مجموعة (يبقى بوحدة ft/hr لأن المحاور لا تتأثر باختيار المستخدم)
                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax1.set_xlabel('WOB (klb)')
                ax1.set_ylabel('ROP (ft/hr)', color='blue')
                ax1.scatter(agg['WOB_klb'], agg['ROP_fthr'], color='blue', alpha=0.6)
                ax1.plot(wob_fine, rop_fine, color='blue', linestyle='-', linewidth=2)
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.grid(True, alpha=0.3)

                ax2 = ax1.twinx()
                ax2.set_ylabel('MSE (psi)', color='red')
                ax2.scatter(agg['WOB_klb'], agg['MSE_psi'], color='red', marker='s', alpha=0.6)
                ax2.plot(wob_fine, mse_fine, color='red', linestyle='--', linewidth=2)
                ax2.tick_params(axis='y', labelcolor='red')

                ax1.axvline(x=founder_wob_klb, color='orange', linestyle=':', linewidth=2, label=f'Founder {founder_wob_klb:.1f} klb')
                ax1.axvline(x=wob_opt_klb, color='green', linestyle='-.', linewidth=2.5, label=f'Optimal {wob_opt_klb:.1f} klb')
                ax1.legend(loc='upper left')
                plt.title(f'Group: {group_label}')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # -------------------------------------------------------------------------
                # تفسير الرسم البياني: بعد عرض الشكل، يتم تقديم شرح يوضح معنى المنحنى
                # الأزرق (العلاقة بين ROP وWOB) والمنحنى الأحمر المقطَّع (العلاقة بين
                # MSE وWOB). يتم الإشارة إلى نقطة المؤسس والخط العمودي البرتقالي ونقطة
                # الأمثل والخط الأخضر. كما تُعرض نسب MSE إلى قوة الضغط غير المحصور
                # (UCS) للمؤسس والأمثل، حيث تشير نسبة قريبة من 1 إلى كفاءة حفر عالية.
                # -------------------------------------------------------------------------
                caption_parts = [
                    "**Interpretation:** The blue curve shows the fitted relationship between WOB and ROP, while the red dashed curve represents the fitted MSE curve.",
                    "The orange vertical line marks the founder point where the ROP curve flattens, and the green line marks the optimal WOB obtained from the desirability function.",
                ]
                # Add information about recommended WOB range if available
                if wob_rec_range and isinstance(wob_rec_range, (list, tuple)):
                    caption_parts.append(
                        f"The search for optimal WOB was restricted to the recommended range for {layer_name} (\​{wob_rec_range[0]:.1f}–{wob_rec_range[1]:.1f} klb)."
                    )
                # Add MSE ratio details if UCS is known
                if not np.isnan(founder_mse_ratio):
                    caption_parts.append(
                        f"For this group, the founder MSE/UCS ratio is {founder_mse_ratio:.2f} and the optimal MSE/UCS ratio is {opt_mse_ratio:.2f}. "
                        "Ratios near 1.0 indicate that the drilling energy matches the rock's compressive strength, which is considered efficient【436714205699149†L204-L217】【612207715040690†L90-L101】."
                    )
                caption_text = "\n".join(caption_parts)
                st.caption(caption_text)

                # Update progress bar using total number of groups (layer x RPM combinations)
                progress_bar.progress((i + 1) / total_groups)

            # حفظ النتائج في session_state
            if results:
                st.session_state.results_df = pd.DataFrame(results)
                st.session_state.df = df
                st.session_state.optimization_done = True
            else:
                st.error("No optimization results generated for any group.")

    # ====================== عرض النتائج المحفوظة ======================
    if st.session_state.optimization_done and st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        df = st.session_state.df

        st.subheader("✅ Final Optimization Results")
        st.dataframe(results_df)

        # زر تحميل النتائج
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, "drilling_optimization_results.csv", "text/csv")

        # ====================== حساب التوفير (Savings) اختياري ======================
        st.subheader("💰 Time & Cost Savings Estimation")
        st.markdown("Estimate savings for a specific depth interval using the optimal parameters from a selected RPM group.")

        # إنشاء تسمية فريدة لكل نتيجة تضم الطبقة ومجموعة RPM لسهولة الاختيار
        results_df = results_df.copy()
        results_df['Group_Label'] = results_df['Layer'].astype(str) + ' / ' + results_df['RPM_Group'].astype(str)
        # اختيار المجموعة لحساب التوفير بناءً على الطبقة ومجموعة RPM
        selected_label = st.selectbox(
            "Select group (Layer / RPM) for savings calculation", results_df['Group_Label'].unique()
        )
        # استنتاج اسم الطبقة ومجموعة RPM من التسمية المختارة
        if selected_label:
            selected_layer, selected_rpm_group = selected_label.split(' / ', 1)
            # الحصول على البيانات الخاصة بهذه المجموعة
            group_data_for_savings = df[(df['Layer'] == selected_layer) & (df['RPM_Group'] == selected_rpm_group)].copy()
        else:
            group_data_for_savings = pd.DataFrame()

        if not group_data_for_savings.empty:
            col1, col2 = st.columns(2)
            with col1:
                start_depth = st.number_input("Start depth", value=float(group_data_for_savings['Depth'].min()))
            with col2:
                end_depth = st.number_input("End depth", value=float(group_data_for_savings['Depth'].max()))

            if st.button("Calculate Savings"):
                # فلترة البيانات ضمن الفترة
                zone_data = group_data_for_savings[(group_data_for_savings['Depth'] >= start_depth) & 
                                                    (group_data_for_savings['Depth'] <= end_depth)]
                if len(zone_data) > 0:
                    # حساب المسافة المحفورة (مجموع الفروق الإيجابية في العمق)
                    zone_depths = zone_data['Depth'].sort_values()
                    depth_diffs = zone_depths.diff().dropna()
                    # نفرض أن الفرق بين القراءات لا يزيد عن 5 أقدام (يمثل خطوة حفر)
                    drilled_distance = depth_diffs[(depth_diffs > 0) & (depth_diffs <= 5.0)].sum()
                    if drilled_distance <= 0:
                        drilled_distance = zone_depths.max() - zone_depths.min()

                    # متوسط ROP التاريخي في الفترة (بالوحدة المختارة)
                    if output_unit == 'm/hr':
                        avg_historical_rop = zone_data['ROP_fthr'].mean() / 3.28084
                        historical_rop_label = "m/hr"
                    else:
                        avg_historical_rop = zone_data['ROP_fthr'].mean()
                        historical_rop_label = "ft/hr"

                    # ROP المثلى من النتائج لهذه المجموعة (بالوحدة المختارة)
                    opt_row = results_df[results_df['Group_Label'] == selected_label].iloc[0]
                    opt_rop_output = opt_row[f'ROP_opt_{output_unit}']

                    # حساب الوقت والتكلفة بناءً على الوحدة
                    if output_unit == 'm/hr':
                        # المسافة المحفورة بالمتر
                        drilled_distance_m = drilled_distance * 0.3048  # تحويل ft إلى m
                        historical_time = drilled_distance_m / avg_historical_rop if avg_historical_rop > 0 else 0
                        optimal_time = drilled_distance_m / opt_rop_output if opt_rop_output > 0 else 0
                        cost_unit = "meter"
                    else:
                        historical_time = drilled_distance / avg_historical_rop if avg_historical_rop > 0 else 0
                        optimal_time = drilled_distance / opt_rop_output if opt_rop_output > 0 else 0
                        cost_unit = "ft"

                    time_saved = historical_time - optimal_time
                    cost_saved = time_saved * rig_rate

                    st.success(f"""
                    **Zone:** {start_depth:.2f} - {end_depth:.2f}  
                    **Drilled distance:** {drilled_distance:.2f} ft ({drilled_distance*0.3048:.2f} m)  
                    **Historical avg ROP:** {avg_historical_rop:.2f} {historical_rop_label}  
                    **Optimal ROP:** {opt_rop_output:.2f} {output_unit}  
                    **Time saved:** {max(0, time_saved):.2f} hrs  
                    **Cost saved:** ${max(0, cost_saved):,.2f}  
                    *(based on {cost_unit} of drilling)*
                    """)
                else:
                    st.warning("No data in the specified interval.")
        else:
            st.warning("No data available for the selected group.")
else:
    st.info("Please upload a data file to begin.")
