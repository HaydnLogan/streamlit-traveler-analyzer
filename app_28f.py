# v28f - Full Range inclusive helper integrated; Full Range runs first.
# Model G now split into multiple files.

import streamlit as st
import pandas as pd
import datetime as dt
import io
from typing import Optional
from pandas import ExcelWriter
from custom_range_calculator import apply_custom_ranges_advanced, apply_full_range_advanced

# Configure pandas to handle large datasets
pd.set_option("styler.render.max_elements", 2000000)

# Import functions - these paths are confirmed working
from a_helpers import (
    clean_timestamp, process_feed, get_input_at_day_start, apply_excel_highlighting,
    get_input_value, highlight_traveler_report, get_input_at_time, highlight_custom_traveler_report, generate_master_traveler_list,
    GROUP_1A_TRAVELERS, GROUP_1B_TRAVELERS, GROUP_2A_TRAVELERS, GROUP_2B_TRAVELERS,
)

# Model imports - work in your environment, fallbacks for this environment
try:
    from model_g_manager import run_model_g_detection
except ImportError:
    try:
        from models_g_updated import run_model_g_detection
    except ImportError:
        try:
            from models.models_g import run_model_g_detection
        except ImportError:
            from model_g_detector import run_model_g_detection

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
    """Render a single Excel download with all traveler groups."""
    if not traveler_reports:
        return

    st.markdown("---")
    st.markdown("### 📥 Unified Excel Download - Master Traveler Groups")

    report_datetime_str = (
        report_time.strftime("%d-%b-%y_%H-%M") if report_time
        else dt.datetime.now().strftime("%d-%b-%y_%H-%M")
    )

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        workbook = writer.book
        for group_name, group_data in traveler_reports.items():
            if isinstance(group_data, pd.DataFrame) and not group_data.empty:
                # Excel sheet name rules
                sheet_name = group_name.replace(" ", "_").replace("-", "_")[:31]

                # Drop a 'Group' column if present (it is only for display)
                export_data = group_data.drop(columns=['Group'], errors='ignore').copy()
                export_data.to_excel(writer, sheet_name=sheet_name, index=False)

                # Highlighting
                worksheet = writer.sheets[sheet_name]
                try:
                    apply_excel_highlighting(workbook, worksheet, export_data, False)
                except Exception as e:
                    # Do not crash export if highlighting fails
                    st.warning(f"Highlighting skipped for '{sheet_name}': {e}")

    excel_buffer.seek(0)
    total_entries = sum(
        len(df) for df in traveler_reports.values()
        if isinstance(df, pd.DataFrame)
    )
    num_groups = len(
        [k for k, v in traveler_reports.items()
         if isinstance(v, pd.DataFrame) and not v.empty]
    )

    st.download_button(
        label="📥 Download All Traveler Groups (Unified Excel)",
        data=excel_buffer.getvalue(),
        file_name=f"master_traveler_groups_{report_datetime_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help=f"Excel file contains {num_groups} groups with {total_entries} total entries"
    )
    st.success(
        f"✅ Unified Excel file ready with {len(traveler_reports)} traveler groups "
        f"({total_entries} total entries)"
    )


# 🔌 Streamlit interface (UI + orchestration)
st.set_page_config(layout="wide")
st.header("🧬 Data Processor + Model A/B/C/G Detector with fast mode. v28f")

# 📤 Uploads
small_feed_file = st.file_uploader("Upload small feed", type="csv")
big_feed_file = st.file_uploader("Upload big feed", type="csv")
measurement_file = st.file_uploader("Upload measurement file", type=["xlsx", "xls"])

# 📂 Optional: Upload Final Traveler Report (bypass feed upload)
st.markdown("---")
st.markdown("### 📂 Optional: Upload Final Traveler Report (bypass feed upload)")
bypass_traveler_file = st.file_uploader(
    "Upload Final Traveler Report",
    type=['xlsx', 'csv'],
    help="Skip feed processing and upload traveler report directly"
)

# 📅 Report Time UI
report_mode = st.radio("Select Report Time & Date", ["Most Current", "Choose a time"])
if report_mode == "Choose a time":
    selected_date = st.date_input("Select Report Date", value=dt.date.today())
    selected_time = st.time_input("Select Report Time", value=dt.time(18, 0))
    report_time = dt.datetime.combine(selected_date, selected_time)
else:
    report_time = None

# 🚥 Toggles
run_g_models = st.sidebar.checkbox("🟢 Run Model G Detection", value=False)
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

# ⚙️ Analysis parameters
day_start_choice = st.radio("Select Day Start Time", ["18:00", "17:00"])
day_start_hour = int(day_start_choice.split(":")[0])
scope_type = st.radio("Scope by", ["Days", "Rows"])
scope_value = st.number_input(
    f"Enter number of {scope_type.lower()}",
    min_value=1,
    value=20
)

# 🎯 Traveler Report Settings (Global Configuration)
st.markdown("---")
st.markdown("### 🎯 Traveler Report Settings")

# Mutually exclusive radio button selection
report_type = st.radio(
    "Select Report Type",
    ["Full Range", "Custom Ranges"],
    key="global_report_type"
)

# Initialize all variables with defaults first
use_full_range = False
use_custom_ranges = False
use_advanced_ranges = False
full_range_value = 0
high1 = high2 = low1 = low2 = 0
use_high1 = use_high2 = use_low1 = use_low2 = False

if report_type == "Full Range":
    use_full_range = True
    col1, col2 = st.columns(2)
    with col1:
        full_range_value = st.number_input(
            "Full Range Value (±)",
            min_value=1,
            value=1100,
            key="global_full_range"
        )
    with col2:
        st.markdown("**Range will be:** Input @ Day Start ± Full Range Value")

elif report_type == "Custom Ranges":
    use_custom_ranges = True
    use_advanced_ranges = True  # Always use advanced calculation
    st.write("Configure up to 4 custom ranges:")
    st.info("🧮 **Advanced H/L/C calculation enabled** - Uses sophisticated market data analysis with H/L/C values from small CSV to calculate raw M values and filter measurement data")

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


# ===========================
# === BYPASS TRAVELER FILE ===
# ===========================
if bypass_traveler_file:
    try:
        st.markdown("### 📊 Processing Bypass Traveler Report")

        # Read the bypass file
        if bypass_traveler_file.name.endswith('.csv'):
            final_df = pd.read_csv(bypass_traveler_file)
        else:
            xls = pd.ExcelFile(bypass_traveler_file)
            if len(xls.sheet_names) > 1:
                sheet_choice = st.selectbox("Select traveler report tab", xls.sheet_names, key="bypass_sheet")
                final_df = pd.read_excel(bypass_traveler_file, sheet_name=sheet_choice)
            else:
                final_df = pd.read_excel(bypass_traveler_file)

        st.success(f"✅ Bypass file loaded successfully: {len(final_df)} rows")

        # Extract report time from 'Arrival' or fallback to now
        if 'Arrival' in final_df.columns and not final_df.empty:
            try:
                arrival_times = pd.to_datetime(final_df['Arrival'], format='%m/%d/%Y %H:%M', errors='coerce')
                valid_times = arrival_times.dropna()
                report_time = valid_times.max() if len(valid_times) > 0 else dt.datetime.now()
            except Exception:
                report_time = dt.datetime.now()
        else:
            report_time = dt.datetime.now()

        st.info(f"Report time set to: {report_time.strftime('%m/%d/%Y %H:%M')}")

        # Copy to final_df_filtered for consistency
        final_df_filtered = final_df.copy()

        # Sort by Output desc, Arrival asc
        if 'Output' in final_df_filtered.columns and 'Arrival' in final_df_filtered.columns:
            final_df_filtered = final_df_filtered.sort_values(['Output', 'Arrival'], ascending=[False, True])

        st.markdown(f"**Total Entries:** {len(final_df_filtered)}")
        st.dataframe(final_df_filtered, use_container_width=True)

        # Optional Model G on bypass
        if run_g_models:
            st.markdown("---")
            st.markdown("### 🟢 Model G Detection on Bypass Report")
            try:
                from a_helpers import GROUP_1B_TRAVELERS
                if 'M #' in final_df_filtered.columns:
                    grp_1b_mask = final_df_filtered['M #'].isin(GROUP_1B_TRAVELERS)
                    grp_1b_df = final_df_filtered[grp_1b_mask].copy()
                    if grp_1b_df.empty:
                        st.info("No Group 1b entries found in bypass report for Model G detection")
                    else:
                        try:
                            from model_g_manager import run_model_g_detection as _g
                        except ImportError:
                            try:
                                from model_g import run_model_g_detection as _g
                            except ImportError:
                                from model_g_detector import run_model_g_detection as _g

                        g_results = _g(grp_1b_df, report_time, key_suffix="_bypass")
                        if isinstance(g_results, dict) and 'success' in g_results:
                            if g_results['success']:
                                summary = g_results['summary']
                                c1, c2, c3 = st.columns(3)
                                with c1: st.metric("o1 (Today)", summary['total_o1'])
                                with c2: st.metric("o2 (Other Day)", summary['total_o2'])
                                with c3: st.metric("Total Sequences", summary['total_sequences'])
                                if not g_results['results_df'].empty:
                                    st.markdown("#### Bypass Report Model G Results")
                                    st.dataframe(g_results['results_df'], use_container_width=True)
                                else:
                                    st.info("No Model G sequences detected in bypass report Grp 1b data")
                            else:
                                st.error(f"Model G detection error: {g_results['error']}")
                else:
                    st.warning("No 'M #' column found in bypass report - cannot run Model G detection")
            except ImportError as e:
                st.warning(f"Model G detection not available: {e}")
            except Exception as e:
                st.error(f"Model G detection error: {str(e)}")

        # Downloads
        timestamp = report_time.strftime("%y-%m-%d_%H-%M")

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            final_df_filtered.to_excel(writer, sheet_name='Final Traveler Report', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Final Traveler Report']
            header_format = workbook.add_format({
                'bold': True, 'text_wrap': True, 'valign': 'top',
                'fg_color': '#D7E4BC', 'border': 1
            })
            for col_num, value in enumerate(final_df_filtered.columns.values):
                worksheet.write(0, col_num, value, header_format)

        excel_filename = f"final_traveler_report_{timestamp}.xlsx"
        st.download_button(
            label="📥 Download Final Traveler Report (Excel)",
            data=excel_buffer.getvalue(),
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        csv_filename = f"final_traveler_report_{timestamp}.csv"
        st.download_button(
            label="📥 Download Final Traveler Report (CSV)",
            data=final_df_filtered.to_csv(index=False),
            file_name=csv_filename,
            mime="text/csv"
        )

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
        st.error(f"Error processing bypass file: {str(e)}")
        st.stop()

# ===========================
# === MAIN FEED PROCESSOR ===
# ===========================
elif small_feed_file and big_feed_file and measurement_file:
    try:
        st.markdown("### 📊 Processing Feed Files")

        # Read the uploaded files
        small_df = pd.read_csv(small_feed_file)
        big_df = pd.read_csv(big_feed_file)

        # Measurement file (Excel with sheets)
        xls = pd.ExcelFile(measurement_file)
        available_tabs = xls.sheet_names
        if len(xls.sheet_names) > 1:
            sheet_choice = st.selectbox("Select primary measurement tab", xls.sheet_names)
            measurements_df = pd.read_excel(measurement_file, sheet_name=sheet_choice)
        else:
            measurements_df = pd.read_excel(measurement_file)
            sheet_choice = xls.sheet_names[0]

        st.success(f"✅ Files loaded successfully:")
        st.markdown(f"- Small feed: {len(small_df)} rows")
        st.markdown(f"- Big feed: {len(big_df)} rows")
        st.markdown(f"- Measurements: {len(measurements_df)} rows from '{sheet_choice}' tab")

        # === Processing Options
        st.markdown("---")
        st.markdown("### ⚙️ Processing Options")

        perf_c1, perf_c2, perf_c3 = st.columns(3)
        with perf_c1:
            fast_mode = st.checkbox("🚀 Fast Mode", value=False, help="Skip debug info and detailed analysis")
        with perf_c2:
            parallel_processing = st.checkbox("⚡ Parallel Processing", value=True, help="Process measurement tabs simultaneously")
        with perf_c3:
            minimal_display = st.checkbox("📋 Minimal Display", value=False, help="Show only summary statistics")

        # Store Model G custom setting — guard to avoid rerun loops
        if st.session_state.get('run_model_g_on_custom') != run_g_on_custom:
            st.session_state['run_model_g_on_custom'] = run_g_on_custom

        # Ensure report_time
        if report_time is None:
            big_df['time'] = big_df['time'].apply(clean_timestamp)
            report_time = big_df['time'].max()
            st.info(f"Report time auto-set to most current: {report_time.strftime('%m/%d/%Y %H:%M')}")

        # Input @ day start
        input_value_at_start = get_input_at_day_start(small_df, report_time, day_start_hour)
        if input_value_at_start is not None:
            st.info(f"Input @ {day_start_hour:02d}:00: {input_value_at_start}")

        import time
        start_time = time.time()

        # Single source of truth for export
        traveler_reports = {}

        # ---------- 1) FULL RANGE (first/primary path) ----------
        if use_full_range:
            st.markdown("---")
            st.markdown("### 🌐 Full Range Processing Mode (Advanced / HLC → raw-M)")
            result_df = apply_full_range_advanced(
                df=measurements_df,
                small_df=small_df,
                report_time=report_time,
                window_radius=full_range_value,
                day_start_hour=day_start_hour,
                input_value_at_start=input_value_at_start,   # keep your center
                big_df=big_df,
                run_model_g=False,
            )

            traveler_reports = {}
            if result_df is not None and not result_df.empty:
                traveler_reports["Grp 1a"] = result_df[result_df['M #'].isin(GROUP_1A_TRAVELERS)].copy()
                traveler_reports["Grp 1b"] = result_df[result_df['M #'].isin(GROUP_1B_TRAVELERS)].copy()
                traveler_reports["Grp 2a"] = result_df[result_df['M #'].isin(GROUP_2A_TRAVELERS)].copy()
                traveler_reports["Grp 2b"] = result_df[result_df['M #'].isin(GROUP_2B_TRAVELERS)].copy()

                for gname, gdf in traveler_reports.items():
                    if not gdf.empty and {'Output','Arrival'}.issubset(gdf.columns):
                        traveler_reports[gname] = gdf.sort_values(['Output','Arrival'], ascending=[False, True])

                st.success("✅ Full Range (Advanced) complete.")
            else:
                st.warning("Full Range (Advanced) produced no rows.")

        # ---------- 2) CUSTOM RANGES (advanced) ----------
        elif use_custom_ranges and use_advanced_ranges:
            st.markdown("---")
            st.markdown("### 🧮 Advanced Custom Range Processing")

            # DEBUG tails (optional)
            st.warning("DEBUG: Bottom 4 rows of Small Feed CSV:")
            if len(small_df) > 0:
                small_bottom = small_df.tail(4)[['time', 'origin', 'high', 'low', 'close']] if all(
                    col in small_df.columns for col in ['time', 'origin', 'high', 'low', 'close']
                ) else small_df.tail(4)
                for i, (_, row) in enumerate(small_bottom.iterrows()):
                    st.text(f"  Row {len(small_df)-4+i+1}: {row.to_dict()}")
            else:
                st.text("  Small feed is empty")

            st.warning("DEBUG: Bottom 4 rows of Big Feed CSV:")
            if len(big_df) > 0:
                big_bottom = big_df.tail(4)[['time', 'origin', 'close']] if all(
                    col in big_df.columns for col in ['time', 'origin', 'close']
                ) else big_df.tail(4)
                for i, (_, row) in enumerate(big_bottom.iterrows()):
                    st.text(f"  Row {len(big_df)-4+i+1}: {row.to_dict()}")
            else:
                st.text("  Big feed is empty")

            # Use first measurement tab for advanced calc
            master_tab_name = available_tabs[0]
            master_measurements_df = pd.read_excel(measurement_file, sheet_name=master_tab_name)

            final_df_filtered = apply_custom_ranges_advanced(
                master_measurements_df, small_df.copy(), report_time,
                high1, high2, low1, low2,
                use_high1, use_high2, use_low1, use_low2,
                big_df=big_df,
                run_model_g=run_g_on_custom
            )

            if final_df_filtered is None or final_df_filtered.empty:
                st.warning("No entries found using advanced H/L/C calculation")
                traveler_reports = {}
            else:
                # Build 4 groups from filtered results
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

                # Display compact summaries (optional)
                if not minimal_display:
                    for gname, gdf in traveler_reports.items():
                        if not gdf.empty:
                            st.markdown(f"#### 📋 {gname}")
                            st.info(f"{len(gdf)} entries")
                            if not fast_mode:
                                st.dataframe(gdf, use_container_width=True)

                st.success("✅ Advanced custom range processing complete.")

        # ---------- 3) MASTER TRAVELER LIST fallback ----------
        else:
            st.markdown("---")
            st.markdown("### 🚀 Master Traveler List (Fallback)")

            # Build master list from the first measurement tab
            master_tab_name = available_tabs[0]
            master_measurements_df = pd.read_excel(measurement_file, sheet_name=master_tab_name)

            all_traveler_data = []

            # Big feed
            if len(big_df) > 0:
                big_df['time'] = big_df['time'].apply(clean_timestamp)
                big_data = process_feed(
                    big_df, "Big", report_time, scope_type, scope_value,
                    day_start_hour, master_measurements_df, input_value_at_start, small_df,
                    use_full_range=False, full_range_value=0
                )
                all_traveler_data.extend(big_data)

            # Small feed
            if len(small_df) > 0:
                small_df['time'] = small_df['time'].apply(clean_timestamp)
                small_data = process_feed(
                    small_df, "Small", report_time, scope_type, scope_value,
                    day_start_hour, master_measurements_df, input_value_at_start, small_df,
                    use_full_range=False, full_range_value=0
                )
                all_traveler_data.extend(small_data)

            if not all_traveler_data:
                st.error("Failed to generate master traveler list")
                traveler_reports = {}
            else:
                master_df = pd.DataFrame(all_traveler_data)

                traveler_reports = {
                    "Grp 1a": master_df[master_df['M #'].isin(GROUP_1A_TRAVELERS)].copy(),
                    "Grp 1b": master_df[master_df['M #'].isin(GROUP_1B_TRAVELERS)].copy(),
                    "Grp 2a": master_df[master_df['M #'].isin(GROUP_2A_TRAVELERS)].copy(),
                    "Grp 2b": master_df[master_df['M #'].isin(GROUP_2B_TRAVELERS)].copy(),
                }

                for gname, gdf in traveler_reports.items():
                    if not gdf.empty and {'Output', 'Arrival'}.issubset(gdf.columns):
                        traveler_reports[gname] = gdf.sort_values(['Output', 'Arrival'], ascending=[False, True])

                if not minimal_display:
                    for gname, gdf in traveler_reports.items():
                        if not gdf.empty:
                            st.markdown(f"#### 📋 {gname}")
                            st.info(f"{len(gdf)} entries")
                            if not fast_mode:
                                st.dataframe(gdf, use_container_width=True)

                st.success("✅ Master list created and filtered.")

        # ---------- Unified Export (for whichever path produced traveler_reports) ----------
        processing_time = time.time() - start_time
        if traveler_reports:
            render_unified_export(traveler_reports, report_time)

            st.markdown("---")
            st.markdown("### ⏱️ Performance Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Processing Time", f"{processing_time:.1f}s")
            with col2:
                st.metric("Groups Generated", len(traveler_reports))
            with col3:
                total_entries = sum(
                    len(df) for df in traveler_reports.values()
                    if isinstance(df, pd.DataFrame)
                )
                st.metric("Total Entries", total_entries)

            if processing_time < 60:
                st.success(f"🚀 Excellent performance: {processing_time:.1f}s")
            elif processing_time < 180:
                st.info(f"⚡ Good performance: {processing_time:.1f}s")
            else:
                st.warning(f"⏱️ Consider enabling Fast Mode (took {processing_time:.1f}s)")
        else:
            st.info("No traveler groups to export.")

        # ===============================
        # === Model Detections Section ===
        # ===============================
        # Build a generic DataFrame for model detections
        if 'final_df_filtered' not in locals():
            # Try to build from traveler_reports if available
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

        # Optional: Filter out future entries if requested and Arrival_datetime present
        if filter_future_data and report_time and not final_df_filtered.empty:
            if 'Arrival_datetime' in final_df_filtered.columns:
                final_df_filtered = final_df_filtered[final_df_filtered['Arrival_datetime'] <= report_time]
            elif 'Arrival' in final_df_filtered.columns:
                tmp_dt = pd.to_datetime(final_df_filtered['Arrival'], errors='coerce', infer_datetime_format=True)
                final_df_filtered = final_df_filtered[tmp_dt <= report_time]

        if run_g_models:
            st.markdown("---")
            try:
                g_results = run_model_g_detection(final_df_filtered, report_time, key_suffix="_main")
                if isinstance(g_results, dict) and 'success' in g_results:
                    if g_results['success']:
                        summary = g_results['summary']
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric("o1 (Today)", summary['total_o1'])
                        with c2: st.metric("o2 (Other Day)", summary['total_o2'])
                        with c3: st.metric("Total Sequences", summary['total_sequences'])
                        if not g_results['results_df'].empty:
                            st.markdown("#### Detection Results")
                            st.dataframe(g_results['results_df'], use_container_width=True)
                        else:
                            st.info("No Model G sequences detected matching criteria")
                    else:
                        st.error(f"Model G detection error: {g_results['error']}")
            except Exception as e:
                st.error(f"Model G detection error: {str(e)}")
                st.info("Make sure model_g_manager.py exists and contains run_model_g_detection function")

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
        st.error(f"❌ Error processing files: {e}")
        import traceback
        st.text(traceback.format_exc())

else:
    st.info("Please upload small feed, big feed, and measurement files to begin processing.")
