# 11.6.25 Model G.11 added.

# v31f - Process Timeframes Separately" Option add.  with one high and one low range, current process take 200+ seconds.  
# this mod should bring it down to under 90 seconds.

# v31e - Processing time optimization to remove redundant data points.  Zone Sorting Fix #2

# v31d - Asset ID in Filename; for Custom Ranges path, Changed sort order from ['Group', 'Output', 'Arrival'] to ['Range', 'Group', 'Output', 'Arrival']

# v31c - "Most Current" mode: Now uses dt.datetime.now() as the report time (current timestamp)
# This ensures that report_time always has a valid datetime value when processing, preventing the NaN conversion error.

# v31b - Updated file uploads (7 files), added asset selector, modified custom ranges to single tab output
# Previous: v30b - Added HOD/LOD report mode with multi-day processing

import streamlit as st
import pandas as pd
import datetime as dt
import io
from typing import Optional
from pandas import ExcelWriter
from custom_range_calculator_0813 import apply_custom_ranges_advanced, apply_full_range_advanced
from hod_lod_processor import (
    process_hod_lod_mode, 
    is_complete_trading_day, 
    get_trading_day_bounds
)

# Configure pandas to handle large datasets
pd.set_option("styler.render.max_elements", 2000000)

# Import functions
from a_helpers import (
    clean_timestamp, process_feed, get_input_at_day_start, apply_excel_highlighting,
    GROUP_1A_TRAVELERS, GROUP_1B_TRAVELERS, GROUP_2A_TRAVELERS, GROUP_2B_TRAVELERS,
    get_input_value, highlight_traveler_report, get_input_at_time, highlight_custom_traveler_report,
    generate_master_traveler_list,    
)

# Model imports
from model_g_manager import run_model_g_detection

try:
    from models.models_a_today import run_a_model_detection_today
except ImportError:
    def run_a_model_detection_today(df):
        st.warning("Model A detection not available in this environment")

try:
    from models.mod_b_05pg1 import run_b_model_detection
except ImportError:
    def run_b_model_detection(df):
        st.warning("Model B detection not available in this environment")

try:
    from models.mod_c_04gpr3 import run_c_model_detection
except ImportError:
    def run_c_model_detection(df, run_c01=True, run_c02=True, run_c04=True):
        st.warning("Model C detection not available in this environment")

try:
    from models.mod_x_03g import run_x_model_detection
except ImportError:
    def run_x_model_detection(df):
        st.warning("Model X detection not available in this environment")

try:
    from models.simple_mega_report2 import run_simple_single_line_analysis
except ImportError:
    def run_simple_single_line_analysis(df):
        st.warning("Single Line Mega Report not available in this environment")


# === Unified Export Helper ===
def render_unified_export(traveler_reports, report_time, asset_id=""):
    if not traveler_reports:
        return

    st.markdown("---")
    st.markdown("### 📥 Unified Excel Download")

    # Add asset_id to filename if provided
    asset_prefix = f"{asset_id.lower()}_" if asset_id else ""
    report_datetime_str = report_time.strftime("%d-%b-%y_%H-%M")

    def _coerce_arrival_datetime(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Arrival_datetime" in df.columns:
            df["Arrival"] = pd.to_datetime(df["Arrival_datetime"], errors="coerce")
        elif "Arrival" in df.columns:
            df["Arrival"] = pd.to_datetime(df["Arrival"], errors="coerce", infer_datetime_format=True)
        return df

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter", datetime_format="mm/dd/yyyy hh:mm") as writer:
        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True, "text_wrap": True, "valign": "top",
            "fg_color": "#D7E4BC", "border": 1
        })
        date_fmt = workbook.add_format({"num_format": "mm/dd/yyyy hh:mm"})

        for group_name, group_data in traveler_reports.items():
            if not isinstance(group_data, pd.DataFrame) or group_data.empty:
                continue

            sheet_name = group_name.replace(" ", "_").replace("-", "_")[:31]
            export_data = group_data.drop(columns=["Group"], errors="ignore").copy()
            export_data = _coerce_arrival_datetime(export_data)

            export_data.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]

            # headers
            for c, name in enumerate(export_data.columns):
                ws.write(0, c, name, header_fmt)

            # make Arrival display as a date in Excel
            if "Arrival" in export_data.columns:
                a_idx = export_data.columns.get_loc("Arrival")
                ws.set_column(a_idx, a_idx, 18, date_fmt)

            # conditional coloring
            try:
                apply_excel_highlighting(workbook, ws, export_data, False)
            except Exception as e:
                st.warning(f"Highlighting skipped for '{sheet_name}': {e}")

    excel_buffer.seek(0)
    total_entries = sum(len(df) for df in traveler_reports.values() if isinstance(df, pd.DataFrame))
    num_groups = sum(1 for v in traveler_reports.values() if isinstance(v, pd.DataFrame) and not v.empty)

    st.download_button(
        "📥 Download Excel Report",
        data=excel_buffer.getvalue(),
        file_name=f"{asset_prefix}traveler_report_{report_datetime_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help=f"Excel file contains {num_groups} sheets with {total_entries} total entries"
    )


# Streamlit interface
st.set_page_config(layout="wide")
st.header("🧬 Data Processor + HOD/LOD + Model Detection v31f")

# Asset ID Selector (above file uploads)
st.markdown("### Asset Selection")
asset_id = st.selectbox(
    "Select Asset ID",
    options=["NQ", "ES", "YM", "RTY"],
    index=0,
    help="Select the asset/instrument for this analysis"
)
st.info(f"Selected Asset: **{asset_id}**")

# File uploads - 7 files in 3 rows
st.markdown("---")
st.markdown("### File Uploads")

# Row 1: Small feeds (3m, 5m, 15m)
col1, col2, col3 = st.columns(3)
with col1:
    small_3m_file = st.file_uploader("Upload 3m small", type="csv", key="small_3m")
with col2:
    small_5m_file = st.file_uploader("Upload 5m small", type="csv", key="small_5m")
with col3:
    small_15m_file = st.file_uploader("Upload 15m small", type="csv", key="small_15m")

# Row 2: Big feeds (3m, 5m, 15m)
col4, col5, col6 = st.columns(3)
with col4:
    big_3m_file = st.file_uploader("Upload 3m big", type="csv", key="big_3m")
with col5:
    big_5m_file = st.file_uploader("Upload 5m big", type="csv", key="big_5m")
with col6:
    big_15m_file = st.file_uploader("Upload 15m big", type="csv", key="big_15m")

# Row 3: Measurement file
measurement_file = st.file_uploader("Upload measurement file", type=["xlsx", "xls"])

# Optional: Upload Final Traveler Report (bypass feed upload)
st.markdown("---")
st.markdown("### 📂 Optional: Upload Final Traveler Report (bypass feed upload)")
bypass_traveler_file = st.file_uploader(
    "Upload Final Traveler Report",
    type=['xlsx', 'csv'],
    help="Skip feed processing and upload traveler report directly"
)

# Report Time UI
report_mode = st.radio("Select Report Time & Date", ["Most Current", "Choose a time"])
if report_mode == "Choose a time":
    selected_date = st.date_input("Select Report Date", value=dt.date.today())
    selected_time = st.time_input("Select Report Time", value=dt.time(18, 0))
    report_time = dt.datetime.combine(selected_date, selected_time)
else:
    # Use current time as default for "Most Current" mode
    report_time = dt.datetime.now()

# Toggles
run_g_models = st.sidebar.checkbox("🟢 Run Model G Detection", value=False)
if run_g_models:
    st.sidebar.markdown("**G Model Controls:**")
    
    # Proximity slider - applies to all G models
    g05_g06_proximity = st.sidebar.slider(
        "Proximity Threshold (hrs)", 
        min_value=0.05, 
        max_value=3.0, 
        value=0.10, 
        step=0.05,
        help="Proximity threshold in hours for grouping sequences (applies to G.05/06 and G.11)"
    )
    
    # Show Rejected Groups - applies to all G models
    show_rejected_groups = st.sidebar.checkbox("Show Rejected Groups", value=False,
        help="Display groups/pairs that were rejected during detection")
    
    st.sidebar.markdown("---")
    
    run_g05_g06 = st.sidebar.checkbox("   • G.05 & G.06 (Proximity Groups)", value=False)
    run_g08 = st.sidebar.checkbox("   • G.08 (x0Pd.w Patterns)", value=False)
    run_g09 = st.sidebar.checkbox("   • G.09 (Flip Endings)", value=False)
    run_g10 = st.sidebar.checkbox("   • G.10 (Pair Detection)", value=False)
    
    if run_g10:
        st.sidebar.markdown("**G.10 Group Controls:**")
        g10_group_0 = st.sidebar.checkbox("      ○ Group 0", value=True)
        g10_group_1 = st.sidebar.checkbox("      ○ Group 1", value=True)
        g10_group_2 = st.sidebar.checkbox("      ○ Group 2", value=True)
        g10_group_3 = st.sidebar.checkbox("      ○ Group 3", value=False)
        g10_group_4 = st.sidebar.checkbox("      ○ Group 4", value=False)
    else:
        g10_group_0 = g10_group_1 = g10_group_2 = g10_group_3 = g10_group_4 = False
    
    run_g11 = st.sidebar.checkbox("   • G.11 (Pair Detection sTF)", value=False)
    
    if run_g11:
        st.sidebar.markdown("**G.11 Controls:**")
        g11_proximity = st.sidebar.slider("      Proximity", min_value=0.1, max_value=10.0, value=3.0, step=0.1, 
                                         help="Filter sequences by output spread (max - min)")
        st.sidebar.markdown("**G.11 Group Controls:**")
        g11_group_0 = st.sidebar.checkbox("      ○ Grp 0 TA", value=True)
        g11_group_1 = st.sidebar.checkbox("      ○ Grp 1 sAA", value=True)
        g11_group_2 = st.sidebar.checkbox("      ○ Grp 2 AA", value=True)
        g11_group_3 = st.sidebar.checkbox("      ○ Grp 3 oA", value=True)
        g11_group_4 = st.sidebar.checkbox("      ○ Grp 4 Ao", value=True)
        st.sidebar.markdown("**G.11 Display Filters:**")
        g11_display_recipes = st.sidebar.checkbox("      Display Recips", value=True)
        g11_display_others = st.sidebar.checkbox("      Display others", value=True)
    else:
        g11_proximity = 3.0
        g11_group_0 = g11_group_1 = g11_group_2 = g11_group_3 = g11_group_4 = True
        g11_display_recipes = g11_display_others = True
    
    debug_g08 = st.sidebar.checkbox("Debug G.08 Detection", value=False)
    if debug_g08:
        st.session_state['debug_g08'] = True
    else:
        st.session_state['debug_g08'] = False
    
    debug_g_models = st.sidebar.checkbox("Debug G Models (show DataFrame info)", value=False)
    if debug_g_models:
        st.session_state['debug_g_models'] = True
    else:
        st.session_state['debug_g_models'] = False
else:
    run_g05_g06 = False
    run_g08 = False
    run_g09 = False
    run_g10 = False
    run_g11 = False
    g05_g06_proximity = 0.10
    show_rejected_groups = False
    g10_group_0 = g10_group_1 = g10_group_2 = g10_group_3 = g10_group_4 = False
    g11_proximity = 3.0
    g11_group_0 = g11_group_1 = g11_group_2 = g11_group_3 = g11_group_4 = True
    g11_display_recipes = g11_display_others = True
    debug_g08 = False
    debug_g_models = False

run_a_models = st.sidebar.checkbox("🔵 Run Model A Detection", value=False)
run_b_models = st.sidebar.checkbox("🟡 Run Model B Detection", value=False)
run_c_models = st.sidebar.checkbox("🟠 Run Model C Detection", value=False)
if run_c_models:
    st.sidebar.markdown("**C Model Sub-detections:**")
    run_c01 = st.sidebar.checkbox("   • C.01", value=True)
    run_c02 = st.sidebar.checkbox("   • C.02", value=True)
    run_c04 = st.sidebar.checkbox("   • C.04", value=True)
else:
    run_c01 = run_c02 = run_c04 = False

run_x_models = st.sidebar.checkbox("🟣 Run Model X Detection", value=False)
run_single_line = st.sidebar.checkbox("📊 Simple Single Line Mega Report", value=False)

# Performance toggles
fast_mode = st.sidebar.checkbox("⚡ Fast Mode (skip dataframe displays)", value=False)
minimal_display = st.sidebar.checkbox("📉 Minimal Display (summaries only)", value=False)
filter_future_data = st.sidebar.checkbox("🔮 Filter Future Data", value=False, 
    help="Remove entries with arrival times after report time")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Processing Mode")
processing_mode = st.sidebar.radio(
    "Select Processing Mode",
    ["Full Range", "Custom Ranges", "HOD/LOD Mode"],
    help="Choose how to process the data"
)

# Full Range options
if processing_mode == "Full Range":
    st.sidebar.markdown("**Full Range Settings:**")
    window_radius = st.sidebar.number_input("Window Radius", value=60.0, min_value=1.0, step=1.0,
        help="Full range window radius around center value")
    input_value_at_start = st.sidebar.number_input("Input Value @ Start (optional)", value=0.0,
        help="Leave at 0 to auto-calculate from feed data")
    run_g_on_full = st.sidebar.checkbox("Run Model G on Full Range", value=False)

# Custom Ranges options
if processing_mode == "Custom Ranges":
    st.sidebar.markdown("**Custom Range Settings:**")
    st.sidebar.info("Define up to 4 custom ranges (2 High, 2 Low)")
    
    use_high1 = st.sidebar.checkbox("Enable High Range 1", value=False)
    high1 = st.sidebar.number_input("High Range 1 Center", value=0.0, step=0.1,
        help="Range will be [value-24, value]") if use_high1 else 0.0
    
    use_high2 = st.sidebar.checkbox("Enable High Range 2", value=False)
    high2 = st.sidebar.number_input("High Range 2 Center", value=0.0, step=0.1,
        help="Range will be [value-24, value]") if use_high2 else 0.0
    
    use_low1 = st.sidebar.checkbox("Enable Low Range 1", value=False)
    low1 = st.sidebar.number_input("Low Range 1 Center", value=0.0, step=0.1,
        help="Range will be [value, value+24]") if use_low1 else 0.0
    
    use_low2 = st.sidebar.checkbox("Enable Low Range 2", value=False)
    low2 = st.sidebar.number_input("Low Range 2 Center", value=0.0, step=0.1,
        help="Range will be [value, value+24]") if use_low2 else 0.0
    
    run_g_on_custom = st.sidebar.checkbox("Run Model G on Custom Ranges", value=False)

# HOD/LOD options
if processing_mode == "HOD/LOD Mode":
    st.sidebar.markdown("**HOD/LOD Settings:**")
    hod_lod_num_days = st.sidebar.number_input("Number of Days to Analyze", value=1, min_value=1, max_value=30,
        help="Number of complete trading days to analyze")
    include_partial_day = st.sidebar.checkbox("Include Partial Current Day", value=True,
        help="Include analysis of current incomplete trading day")
    st.sidebar.info("HOD/LOD mode uses 15m feed with strict time boundaries")

# Common settings
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌍 Common Settings")
scope_value = st.sidebar.number_input("Scope (days)", value=20, min_value=1, max_value=365,
    help="Look back period for data analysis")
day_start_hour = st.sidebar.number_input("Day Start Hour", value=18, min_value=0, max_value=23,
    help="Hour when trading day begins (default: 18:00)")

# Performance optimization
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Performance Settings")
separate_timeframes = st.sidebar.checkbox(
    "Process Timeframes Separately", 
    value=False,
    help="Faster processing: Run 3m, 5m, 15m independently and combine results (Recommended for speed)"
)

# Set flags based on processing mode
use_full_range = (processing_mode == "Full Range")
use_custom_ranges = (processing_mode == "Custom Ranges")
use_hod_lod = (processing_mode == "HOD/LOD Mode")

# === BYPASS MODE ===
if bypass_traveler_file is not None:
    st.success("✅ Bypass mode: Traveler report uploaded directly")
    
    try:
        if bypass_traveler_file.name.endswith('.csv'):
            bypass_df = pd.read_csv(bypass_traveler_file)
        else:
            bypass_df = pd.read_excel(bypass_traveler_file, sheet_name=0)
        
        st.info(f"Loaded {len(bypass_df)} entries from traveler report")
        
        if not fast_mode:
            st.dataframe(bypass_df.head(50), use_container_width=True)
        
        traveler_reports = {"Bypass_Report": bypass_df}
        render_unified_export(traveler_reports, report_time, asset_id)
        
        # Model detection on bypass data
        final_df_filtered = bypass_df.copy()
        
        if filter_future_data and report_time and not final_df_filtered.empty:
            if 'Arrival_datetime' in final_df_filtered.columns:
                final_df_filtered = final_df_filtered[final_df_filtered['Arrival_datetime'] <= report_time]
            elif 'Arrival' in final_df_filtered.columns:
                tmp_dt = pd.to_datetime(final_df_filtered['Arrival'], errors='coerce', infer_datetime_format=True)
                final_df_filtered = final_df_filtered[tmp_dt <= report_time]
        
        if run_g_models and not final_df_filtered.empty:
            st.markdown("---")
            st.markdown("### Model G Detection Results")
            try:
                g_results = run_model_g_detection(
                    final_df_filtered,
                    proximity_threshold=g05_g06_proximity,
                    report_time=report_time,
                    key_suffix="_bypass",
                    run_g05_g06=run_g05_g06,
                    run_g08=run_g08,
                    run_g09=run_g09,
                    run_g10=run_g10,
                    run_g11=run_g11,
                    g10_group_0=g10_group_0,
                    g10_group_1=g10_group_1,
                    g10_group_2=g10_group_2,
                    g10_group_3=g10_group_3,
                    g10_group_4=g10_group_4,
                    g11_proximity=g11_proximity,
                    g11_group_0=g11_group_0,
                    g11_group_1=g11_group_1,
                    g11_group_2=g11_group_2,
                    g11_group_3=g11_group_3,
                    g11_group_4=g11_group_4,
                    g11_display_recipes=g11_display_recipes,
                    g11_display_others=g11_display_others,
                    show_rejected_groups=show_rejected_groups
                )
                if isinstance(g_results, dict) and 'success' in g_results and g_results['success']:
                    summary = g_results['summary']
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("o1 (Today)", summary['total_o1'])
                    with c2: st.metric("o2 (Other Day)", summary['total_o2'])
                    with c3: st.metric("Total Sequences", summary['total_sequences'])
                    
                    # Debug output
                    if 'results_df' in g_results:
                        st.write(f"📊 DataFrame exists: Shape = {g_results['results_df'].shape}")
                        if not g_results['results_df'].empty:
                            st.markdown("#### 📋 Cluster Table")
                            st.dataframe(g_results['results_df'], use_container_width=True)
                        else:
                            st.info("No sequences to display in cluster table (DataFrame is empty)")
                    else:
                        st.error("❌ results_df not found in g_results dictionary")
                else:
                    st.warning(f"Unexpected g_results format or failed: {g_results}")
            except Exception as e:
                st.error(f"Model G detection error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        if run_single_line and not final_df_filtered.empty:
            st.markdown("---")
            run_simple_single_line_analysis(final_df_filtered)
        
        if run_a_models and not final_df_filtered.empty:
            st.markdown("---")
            run_a_model_detection_today(final_df_filtered)
        
        if run_b_models and not final_df_filtered.empty:
            st.markdown("---")
            run_b_model_detection(final_df_filtered)
        
        if run_c_models and not final_df_filtered.empty:
            st.markdown("---")
            run_c_model_detection(final_df_filtered, run_c01=run_c01, run_c02=run_c02, run_c04=run_c04)
        
        if run_x_models and not final_df_filtered.empty:
            st.markdown("---")
            run_x_model_detection(final_df_filtered)
    
    except Exception as e:
        st.error(f"Error processing bypass file: {e}")
        import traceback
        st.text(traceback.format_exc())

# === NORMAL PROCESSING MODE ===
elif small_15m_file and big_15m_file and measurement_file:
    try:
        import time
        start_time = time.time()
        
        # Load measurements
        measurements_df = pd.read_excel(measurement_file, sheet_name=0)
        
        # Load and combine all small feeds
        small_feeds = []
        small_feed_info = []
        small_feeds_dict = {}  # For separate processing
        
        if small_3m_file:
            df_3m = pd.read_csv(small_3m_file)
            small_feeds.append(df_3m)
            small_feeds_dict['3m'] = df_3m
            small_feed_info.append(f"3m: {len(df_3m)} rows")
        
        if small_5m_file:
            df_5m = pd.read_csv(small_5m_file)
            small_feeds.append(df_5m)
            small_feeds_dict['5m'] = df_5m
            small_feed_info.append(f"5m: {len(df_5m)} rows")
        
        if small_15m_file:
            df_15m = pd.read_csv(small_15m_file)
            small_feeds.append(df_15m)
            small_feeds_dict['15m'] = df_15m
            small_feed_info.append(f"15m: {len(df_15m)} rows")
        
        # Combine all small feeds (for combined processing mode)
        if small_feeds and not separate_timeframes:
            small_df = pd.concat(small_feeds, ignore_index=True)
            # Remove duplicates based on time and origin columns to avoid redundant processing
            # First identify time column
            time_col = 'time' if 'time' in small_df.columns else small_df.columns[0]
            # Get all origin-related columns (columns ending with ' H', ' L', ' C')
            origin_cols = [col for col in small_df.columns if col.endswith((' H', ' L', ' C'))]
            key_cols = [time_col] + origin_cols
            small_df = small_df.drop_duplicates(subset=key_cols, keep='first')
        elif small_feeds and separate_timeframes:
            small_df = None  # Will process separately
        else:
            small_df = pd.DataFrame()
        
        # Load and combine all big feeds
        big_feeds = []
        big_feed_info = []
        big_feeds_dict = {}  # For separate processing
        
        if big_3m_file:
            df_3m = pd.read_csv(big_3m_file)
            big_feeds.append(df_3m)
            big_feeds_dict['3m'] = df_3m
            big_feed_info.append(f"3m: {len(df_3m)} rows")
        
        if big_5m_file:
            df_5m = pd.read_csv(big_5m_file)
            big_feeds.append(df_5m)
            big_feeds_dict['5m'] = df_5m
            big_feed_info.append(f"5m: {len(df_5m)} rows")
        
        if big_15m_file:
            df_15m = pd.read_csv(big_15m_file)
            big_feeds.append(df_15m)
            big_feeds_dict['15m'] = df_15m
            big_feed_info.append(f"15m: {len(df_15m)} rows")
        
        # Combine all big feeds (for combined processing mode)
        if big_feeds and not separate_timeframes:
            big_df = pd.concat(big_feeds, ignore_index=True)
            # Remove duplicates based on time and origin columns to avoid redundant processing
            time_col = 'time' if 'time' in big_df.columns else big_df.columns[0]
            origin_cols = [col for col in big_df.columns if col.endswith((' H', ' L', ' C'))]
            key_cols = [time_col] + origin_cols
            big_df = big_df.drop_duplicates(subset=key_cols, keep='first')
        elif big_feeds and separate_timeframes:
            big_df = None  # Will process separately
        else:
            big_df = pd.DataFrame()
        
        # Display file upload confirmation
        st.success(f"✅ Files loaded successfully")
        
        if separate_timeframes:
            st.info("⚡ **Separate Timeframe Mode**: Processing 3m, 5m, 15m independently for optimal speed")
        
        upload_info_cols = st.columns(3)
        with upload_info_cols[0]:
            if separate_timeframes:
                total_small = sum(len(df) for df in small_feeds_dict.values())
                st.metric("Total Small Feed Rows", f"{total_small} rows")
            else:
                st.metric("Combined Small Feed", f"{len(small_df) if small_df is not None else 0} rows")
            if small_feed_info:
                st.caption("Sources: " + ", ".join(small_feed_info))
        with upload_info_cols[1]:
            if separate_timeframes:
                total_big = sum(len(df) for df in big_feeds_dict.values())
                st.metric("Total Big Feed Rows", f"{total_big} rows")
            else:
                st.metric("Combined Big Feed", f"{len(big_df) if big_df is not None else 0} rows")
            if big_feed_info:
                st.caption("Sources: " + ", ".join(big_feed_info))
        with upload_info_cols[2]:
            st.metric("Measurements", f"{len(measurements_df)} rows")
        
        traveler_reports = {}
        
        # ---------- 1) FULL RANGE MODE ----------
        if use_full_range:
            st.markdown("---")
            st.markdown("### Full Range Processing Mode")
            
            input_val = None if input_value_at_start == 0.0 else input_value_at_start
            
            if separate_timeframes and (small_feeds_dict or big_feeds_dict):
                # Process each timeframe separately and combine results
                st.info("Processing timeframes separately for optimal performance...")
                all_timeframe_results = []
                
                timeframes = sorted(set(list(small_feeds_dict.keys()) + list(big_feeds_dict.keys())))
                
                for tf in timeframes:
                    st.text(f"Processing {tf} timeframe...")
                    tf_small = small_feeds_dict.get(tf, pd.DataFrame())
                    tf_big = big_feeds_dict.get(tf, pd.DataFrame())
                    
                    if tf_small.empty and tf_big.empty:
                        continue
                    
                    # Process this timeframe
                    tf_result = apply_full_range_advanced(
                        measurements_df,
                        tf_small if not tf_small.empty else None,
                        report_time,
                        window_radius,
                        day_start_hour=day_start_hour,
                        input_value_at_start=input_val,
                        big_df=tf_big if not tf_big.empty else None,
                        run_model_g=False  # Run G models only on combined data
                    )
                    
                    if tf_result is not None and not tf_result.empty:
                        all_timeframe_results.append(tf_result)
                
                # Combine all timeframe results
                if all_timeframe_results:
                    final_df_filtered = pd.concat(all_timeframe_results, ignore_index=True)
                else:
                    final_df_filtered = pd.DataFrame()
            else:
                # Combined processing (original logic)
                if small_df is None and big_df is None:
                    st.error("No feed data available for combined processing. Please disable 'Process Timeframes Separately' or ensure files are uploaded.")
                    final_df_filtered = pd.DataFrame()
                else:
                    final_df_filtered = apply_full_range_advanced(
                        measurements_df,
                        small_df if small_df is not None else pd.DataFrame(),
                        report_time,
                        window_radius,
                        day_start_hour=day_start_hour,
                        input_value_at_start=input_val,
                        big_df=big_df if big_df is not None else pd.DataFrame(),
                        run_model_g=run_g_on_full
                    )
            
            if final_df_filtered is None or final_df_filtered.empty:
                st.warning("No entries found in full range processing")
                traveler_reports = {}
            else:
                # Build 4 groups
                traveler_reports = {}
                traveler_reports["Grp 1a"] = final_df_filtered[final_df_filtered['M #'].isin(GROUP_1A_TRAVELERS)].copy()
                traveler_reports["Grp 1b"] = final_df_filtered[final_df_filtered['M #'].isin(GROUP_1B_TRAVELERS)].copy()
                traveler_reports["Grp 2a"] = final_df_filtered[final_df_filtered['M #'].isin(GROUP_2A_TRAVELERS)].copy()
                traveler_reports["Grp 2b"] = final_df_filtered[final_df_filtered['M #'].isin(GROUP_2B_TRAVELERS)].copy()
                
                # Sort each group
                for gname, gdf in traveler_reports.items():
                    if not gdf.empty and 'Output' in gdf.columns and 'Arrival' in gdf.columns:
                        traveler_reports[gname] = gdf.sort_values(['Output', 'Arrival'], ascending=[False, True])
                    elif not gdf.empty and 'Output' in gdf.columns:
                        traveler_reports[gname] = gdf.sort_values(['Output'], ascending=[False])
                
                # Display compact summaries
                if not minimal_display:
                    for gname, gdf in traveler_reports.items():
                        if not gdf.empty:
                            st.markdown(f"#### {gname}")
                            st.info(f"{len(gdf)} entries")
                            if not fast_mode:
                                st.dataframe(gdf, use_container_width=True)
                
                st.success("Full range processing complete.")

        # ---------- 2) CUSTOM RANGES MODE ----------
        elif use_custom_ranges:
            st.markdown("---")
            st.markdown("### Custom Ranges Processing Mode")
            
            if separate_timeframes and (small_feeds_dict or big_feeds_dict):
                # Process each timeframe separately and combine results
                st.info("Processing timeframes separately for optimal performance...")
                all_timeframe_results = []
                
                timeframes = sorted(set(list(small_feeds_dict.keys()) + list(big_feeds_dict.keys())))
                
                for tf in timeframes:
                    st.text(f"Processing {tf} timeframe...")
                    tf_small = small_feeds_dict.get(tf, pd.DataFrame())
                    tf_big = big_feeds_dict.get(tf, pd.DataFrame())
                    
                    if tf_small.empty and tf_big.empty:
                        continue
                    
                    # Process this timeframe
                    tf_result = apply_custom_ranges_advanced(
                        measurements_df,
                        tf_small if not tf_small.empty else None,
                        report_time,
                        high1, high2, low1, low2,
                        use_high1, use_high2, use_low1, use_low2,
                        big_df=tf_big if not tf_big.empty else None,
                        run_model_g=False,  # Run G models only on combined data
                        day_start_hour=day_start_hour
                    )
                    
                    if tf_result is not None and not tf_result.empty:
                        all_timeframe_results.append(tf_result)
                
                # Combine all timeframe results
                if all_timeframe_results:
                    final_df_filtered = pd.concat(all_timeframe_results, ignore_index=True)
                else:
                    final_df_filtered = pd.DataFrame()
            else:
                # Combined processing (original logic)
                if small_df is None and big_df is None:
                    st.error("No feed data available for combined processing. Please disable 'Process Timeframes Separately' or ensure files are uploaded.")
                    final_df_filtered = pd.DataFrame()
                else:
                    final_df_filtered = apply_custom_ranges_advanced(
                        measurements_df,
                        small_df if small_df is not None else pd.DataFrame(),
                        report_time,
                        high1, high2, low1, low2,
                        use_high1, use_high2, use_low1, use_low2,
                        big_df=big_df if big_df is not None else pd.DataFrame(),
                        run_model_g=run_g_on_custom,
                        day_start_hour=day_start_hour
                    )

            if final_df_filtered is None or final_df_filtered.empty:
                st.warning("No entries found using advanced H/L/C calculation")
                traveler_reports = {}
            else:
                # Modified: Output all groups to a single tab named 'Grp_All'
                # Combine all valid entries into one dataframe
                all_groups_df = final_df_filtered.copy()
                
                # Add Group column to identify which group each entry belongs to
                def assign_group(m_num):
                    if m_num in GROUP_1A_TRAVELERS:
                        return "Grp 1a"
                    elif m_num in GROUP_1B_TRAVELERS:
                        return "Grp 1b"
                    elif m_num in GROUP_2A_TRAVELERS:
                        return "Grp 2a"
                    elif m_num in GROUP_2B_TRAVELERS:
                        return "Grp 2b"
                    else:
                        return "Other"
                
                if 'Group' not in all_groups_df.columns:
                    all_groups_df['Group'] = all_groups_df['M #'].apply(assign_group)
                
                # Create a custom sort order for Zone column
                zone_order = {"0 to 6": 1, "6 to 12": 2, "12 to 18": 3, "18 to 24": 4, "Out of Range": 5}
                if 'Zone' in all_groups_df.columns:
                    all_groups_df['Zone_sort'] = all_groups_df['Zone'].map(zone_order).fillna(99)
                
                # Sort by Range FIRST (to keep all High 1, High 2, Low 1, Low 2 together),
                # then by Zone (0-6, 6-12, 12-18, 18-24), then by Group, then by Output descending, then by Arrival
                # This ensures each custom range is printed consecutively with zones in order
                if 'Range' in all_groups_df.columns and 'Zone' in all_groups_df.columns and 'Output' in all_groups_df.columns and 'Arrival' in all_groups_df.columns:
                    all_groups_df = all_groups_df.sort_values(['Range', 'Zone_sort', 'Group', 'Output', 'Arrival'], 
                                                               ascending=[True, True, True, False, True])
                    all_groups_df = all_groups_df.drop(columns=['Zone_sort'], errors='ignore')
                elif 'Range' in all_groups_df.columns and 'Zone' in all_groups_df.columns and 'Output' in all_groups_df.columns:
                    all_groups_df = all_groups_df.sort_values(['Range', 'Zone_sort', 'Group', 'Output'], 
                                                               ascending=[True, True, True, False])
                    all_groups_df = all_groups_df.drop(columns=['Zone_sort'], errors='ignore')
                elif 'Range' in all_groups_df.columns and 'Output' in all_groups_df.columns and 'Arrival' in all_groups_df.columns:
                    all_groups_df = all_groups_df.sort_values(['Range', 'Group', 'Output', 'Arrival'], 
                                                               ascending=[True, True, False, True])
                elif 'Range' in all_groups_df.columns and 'Output' in all_groups_df.columns:
                    all_groups_df = all_groups_df.sort_values(['Range', 'Group', 'Output'], 
                                                               ascending=[True, True, False])
                elif 'Output' in all_groups_df.columns and 'Arrival' in all_groups_df.columns:
                    all_groups_df = all_groups_df.sort_values(['Group', 'Output', 'Arrival'], 
                                                               ascending=[True, False, True])
                elif 'Output' in all_groups_df.columns:
                    all_groups_df = all_groups_df.sort_values(['Group', 'Output'], 
                                                               ascending=[True, False])
                
                # Store in single tab called 'Grp_All'
                traveler_reports = {"Grp_All": all_groups_df}
                
                # Display summary
                if not minimal_display:
                    st.markdown(f"#### Grp_All (Combined)")
                    st.info(f"{len(all_groups_df)} total entries across all groups")
                    
                    # Show breakdown by group
                    group_counts = all_groups_df['Group'].value_counts()
                    st.markdown("**Breakdown by Group:**")
                    for grp in ["Grp 1a", "Grp 1b", "Grp 2a", "Grp 2b", "Other"]:
                        if grp in group_counts:
                            st.text(f"  {grp}: {group_counts[grp]} entries")
                    
                    if not fast_mode:
                        st.dataframe(all_groups_df, use_container_width=True)

                st.success("Advanced custom range processing complete - all groups in single tab.")

        # ---------- 3) HOD/LOD MODE ----------
        elif use_hod_lod:
            st.markdown("---")
            st.markdown("### HOD/LOD Processing Mode")
            
            if separate_timeframes:
                st.warning("HOD/LOD mode is not optimized for separate timeframe processing. Using combined feeds.")
                # Force use of 15m feeds for HOD/LOD
                hod_small = small_feeds_dict.get('15m', pd.DataFrame()) if small_feeds_dict else pd.DataFrame()
                hod_big = big_feeds_dict.get('15m', pd.DataFrame()) if big_feeds_dict else pd.DataFrame()
            else:
                hod_small = small_df if small_df is not None else pd.DataFrame()
                hod_big = big_df if big_df is not None else pd.DataFrame()
            
            # Process HOD/LOD mode
            hod_lod_results = process_hod_lod_mode(
                measurement_df=measurements_df,
                small_df=hod_small,
                big_df=hod_big,
                report_time=report_time,
                num_days=hod_lod_num_days,
                include_partial_day=include_partial_day,
                scope_days=scope_value,  # Use existing scope setting
                day_start_hour=day_start_hour
            )
            
            if hod_lod_results:
                traveler_reports = hod_lod_results
                st.success(f"HOD/LOD processing complete: {len(hod_lod_results)} days analyzed")
                
                # Display summary for each day
                if not minimal_display:
                    for day_key, day_df in hod_lod_results.items():
                        st.markdown(f"#### {day_key}")
                        st.info(f"{len(day_df)} entries")
                        if not fast_mode:
                            st.dataframe(day_df.head(20), use_container_width=True)
            else:
                st.warning("No HOD/LOD data found")
                traveler_reports = {}

        # ---------- Unified Export ----------
        processing_time = time.time() - start_time
        if traveler_reports:
            render_unified_export(traveler_reports, report_time, asset_id)

            st.markdown("---")
            st.markdown("### Performance Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Processing Time", f"{processing_time:.1f}s")
            with col2:
                st.metric("Reports Generated", len(traveler_reports))
            with col3:
                total_entries = sum(
                    len(df) for df in traveler_reports.values()
                    if isinstance(df, pd.DataFrame)
                )
                st.metric("Total Entries", total_entries)

            if processing_time < 60:
                st.success(f"Excellent performance: {processing_time:.1f}s")
            elif processing_time < 180:
                st.info(f"Good performance: {processing_time:.1f}s")
            else:
                st.warning(f"Consider enabling Fast Mode (took {processing_time:.1f}s)")
        else:
            st.info("No traveler reports to export.")

        # ===============================
        # === Model Detections Section ===
        # ===============================
        # Build generic DataFrame for model detections
        if 'final_df_filtered' not in locals():
            if traveler_reports:
                try:
                    final_df_filtered = pd.concat(
                        [df for df in traveler_reports.values() if isinstance(df, pd.DataFrame)],
                        ignore_index=True
                    )
                except Exception:
                    final_df_filtered = pd.DataFrame()
            else:
                final_df_filtered = pd.DataFrame()

        # Filter future entries if requested
        if filter_future_data and report_time and not final_df_filtered.empty:
            if 'Arrival_datetime' in final_df_filtered.columns:
                final_df_filtered = final_df_filtered[final_df_filtered['Arrival_datetime'] <= report_time]
            elif 'Arrival' in final_df_filtered.columns:
                tmp_dt = pd.to_datetime(final_df_filtered['Arrival'], errors='coerce', infer_datetime_format=True)
                final_df_filtered = final_df_filtered[tmp_dt <= report_time]

        if run_g_models:
            st.markdown("---")
            st.markdown("### Model G Detection Results")
            st.info(f"Running G models with settings: G.05/06={run_g05_g06}, G.08={run_g08}, G.09={run_g09}, G.10={run_g10}, G.11={run_g11}")
            try:
                g_results = run_model_g_detection(
                    final_df_filtered,
                    proximity_threshold=g05_g06_proximity,
                    report_time=report_time,
                    key_suffix="_main",
                    run_g05_g06=run_g05_g06,
                    run_g08=run_g08,
                    run_g09=run_g09,
                    run_g10=run_g10,
                    run_g11=run_g11,
                    g10_group_0=g10_group_0,
                    g10_group_1=g10_group_1,
                    g10_group_2=g10_group_2,
                    g10_group_3=g10_group_3,
                    g10_group_4=g10_group_4,
                    g11_proximity=g11_proximity,
                    g11_group_0=g11_group_0,
                    g11_group_1=g11_group_1,
                    g11_group_2=g11_group_2,
                    g11_group_3=g11_group_3,
                    g11_group_4=g11_group_4,
                    g11_display_recipes=g11_display_recipes,
                    g11_display_others=g11_display_others,
                    show_rejected_groups=show_rejected_groups
                )
                if isinstance(g_results, dict) and 'success' in g_results:
                    if g_results['success']:
                        summary = g_results['summary']
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric("o1 (Today)", summary['total_o1'])
                        with c2: st.metric("o2 (Other Day)", summary['total_o2'])
                        with c3: st.metric("Total Sequences", summary['total_sequences'])
                        
                        # Debug output
                        if 'results_df' in g_results:
                            st.write(f"📊 DataFrame exists: Shape = {g_results['results_df'].shape}")
                            if not g_results['results_df'].empty:
                                st.markdown("#### 📋 Cluster Table")
                                st.dataframe(g_results['results_df'], use_container_width=True)
                            else:
                                st.info("No Model G sequences detected matching criteria (DataFrame is empty)")
                        else:
                            st.error("❌ results_df not found in g_results dictionary")
                    else:
                        st.error(f"Model G detection error: {g_results['error']}")
                else:
                    st.error("Model G detection returned unexpected format")
            except Exception as e:
                st.error(f"Model G detection error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.info("Make sure model_g_manager.py and all G model files exist")

        if run_single_line:
            st.markdown("---")
            run_simple_single_line_analysis(final_df_filtered)

        if run_a_models:
            st.markdown("---")
            run_a_model_detection_today(final_df_filtered)

        if run_b_models:
            st.markdown("---")
            run_b_model_detection(final_df_filtered)

        if run_c_models:
            st.markdown("---")
            run_c_model_detection(final_df_filtered, run_c01=run_c01, run_c02=run_c02, run_c04=run_c04)

        if run_x_models:
            st.markdown("---")
            run_x_model_detection(final_df_filtered)

    except Exception as e:
        st.error(f"Error processing files: {e}")
        import traceback
        st.text(traceback.format_exc())

else:
    st.info("Please upload all required files to begin processing:")
    st.markdown("""
    **Required files:**
    - **Small feeds**: At least the 15m timeframe (CSV) is required
    - **Big feeds**: At least the 15m timeframe (CSV) is required
    - **Measurement file**: Excel file (XLSX/XLS)
    
    **Optional files:**
    - 3m and 5m small/big feeds will be combined with 15m feeds if provided
    
    *Note: All uploaded timeframes will be processed simultaneously and combined into a single comprehensive report.*
    """)
