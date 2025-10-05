# v30b - Added HOD/LOD report mode with multi-day processing
# HOD/LOD uses combined feed analysis and 15-minute strict cutoff

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
def render_unified_export(traveler_reports, report_time):
    if not traveler_reports:
        return

    st.markdown("---")
    st.markdown("### 📥 Unified Excel Download")

    report_datetime_str = (
        report_time.strftime("%d-%b-%y_%H-%M") if report_time
        else dt.datetime.now().strftime("%d-%b-%y_%H-%M")
    )

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
        file_name=f"traveler_report_{report_datetime_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help=f"Excel file contains {num_groups} sheets with {total_entries} total entries"
    )


# Streamlit interface
st.set_page_config(layout="wide")
st.header("🧬 Data Processor + HOD/LOD + Model Detection v30b")

# File uploads
small_feed_file = st.file_uploader("Upload small feed", type="csv")
big_feed_file = st.file_uploader("Upload big feed", type="csv")
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
    report_time = None

# Toggles
run_g_models = st.sidebar.checkbox("🟢 Run Model G Detection", value=False)
if run_g_models:
    st.sidebar.markdown("**G Model Controls:**")
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
    
    debug_g08 = st.sidebar.checkbox("Debug G.08 Detection", value=False)
    if debug_g08:
        st.session_state['debug_g08'] = True
    else:
        st.session_state['debug_g08'] = False
else:
    run_g05_g06 = False
    run_g08 = False
    run_g09 = False
    run_g10 = False
    g10_group_0 = g10_group_1 = g10_group_2 = g10_group_3 = g10_group_4 = False
    debug_g08 = False
    st.session_state['debug_g08'] = False

run_g_on_custom = st.sidebar.checkbox("🎯 Run Model G on Custom Ranges", value=False)
run_single_line = st.sidebar.checkbox("🎯 Run Single Line Mega Report", value=False)
run_a_models = st.sidebar.checkbox("Run Model A Detection")
run_b_models = st.sidebar.checkbox("Run Model B Detection")
run_c_models = st.sidebar.checkbox("Run Model C Detection")
run_c01 = st.sidebar.checkbox("C Flips", value=True)
run_c02 = st.sidebar.checkbox("C Opposites", value=True)
run_c04 = st.sidebar.checkbox("C Ascending", value=True)
run_x_models = st.sidebar.checkbox("Run Model X Detection")

filter_future_data = st.checkbox(
    "Restrict analysis to Report Time or earlier only",
    value=True
)

# Analysis parameters
day_start_choice = st.radio("Select Day Start Time", ["18:00", "17:00"])
day_start_hour = int(day_start_choice.split(":")[0])
scope_type = st.radio("Scope by", ["Days", "Rows"])
scope_value = st.number_input(
    f"Enter number of {scope_type.lower()}",
    min_value=1,
    value=20
)

# Traveler Report Settings
st.markdown("---")
st.markdown("### 🎯 Traveler Report Settings")

# Report type selection (mutually exclusive)
report_type = st.radio(
    "Select Report Type",
    ["Full Range", "Custom Ranges", "HOD/LOD"],
    key="global_report_type"
)

# Initialize all variables
use_full_range = False
use_custom_ranges = False
use_hod_lod = False
use_advanced_ranges = False
full_range_value = 0
high1 = high2 = low1 = low2 = 0
use_high1 = use_high2 = use_low1 = use_low2 = False
hod_lod_num_days = 5
include_partial_day = False

if report_type == "Full Range":
    use_full_range = True
    col1, col2 = st.columns(2)
    with col1:
        full_range_value = st.number_input(
            "Full Range Value (±)",
            min_value=1,
            value=250,
            key="global_full_range"
        )
    with col2:
        st.markdown("**Range will be:** Input @ Day Start ± Full Range Value")

elif report_type == "Custom Ranges":
    use_custom_ranges = True
    use_advanced_ranges = True
    st.write("Configure up to 4 custom ranges:")
    st.info("🧮 **Advanced H/L/C calculation enabled**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        use_high1 = st.checkbox("High 1", key="global_use_high1")
        high1 = st.number_input("High 1 Value", value=0, key="global_high1") if use_high1 else 0

    with col2:
        use_high2 = st.checkbox("High 2", key="global_use_high2")
        high2 = st.number_input("High 2 Value", value=0, key="global_high2") if use_high2 else 0

    with col3:
        use_low1 = st.checkbox("Low 1", key="global_use_low1")
        low1 = st.number_input("Low 1 Value", value=0, key="global_low1") if use_low1 else 0

    with col4:
        use_low2 = st.checkbox("Low 2", key="global_use_low2")
        low2 = st.number_input("Low 2 Value", value=0, key="global_low2") if use_low2 else 0

elif report_type == "HOD/LOD":
    use_hod_lod = True
    st.write("**HOD/LOD Mode:** Analyze High of Day and Low of Day")
    st.info("🧮 Uses combined small + big feed analysis with 15-minute strict cutoff")
    
    col1, col2 = st.columns(2)
    with col1:
        hod_lod_num_days = st.number_input(
            "Number of days to analyze",
            min_value=1,
            max_value=30,
            value=5,
            key="hod_lod_days"
        )
    with col2:
        include_partial_day = st.checkbox(
            "Include most current partial day",
            value=False,
            key="include_partial",
            help="If checked, includes current day even if trading day is not complete (before 16:45)"
        )
