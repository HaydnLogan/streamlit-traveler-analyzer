# v27f - Master Traveler List
# This is cluttered and needs to be cleaned up


import streamlit as st
import pandas as pd
import datetime as dt
import io
from pandas import ExcelWriter

# Configure pandas to handle large datasets
pd.set_option("styler.render.max_elements", 2000000)

# Import functions - these paths are confirmed working
from a_helpers import clean_timestamp, process_feed, get_input_value, highlight_traveler_report, get_input_at_time, get_input_at_day_start, highlight_custom_traveler_report, apply_excel_highlighting, generate_master_traveler_list, GROUP_1A_TRAVELERS, GROUP_1B_TRAVELERS, GROUP_2A_TRAVELERS, GROUP_2B_TRAVELERS

# Import custom range calculator - no fallback, fail if not available
from custom_range_calculator import apply_custom_ranges_advanced

# Model imports - work in your environment, fallbacks for this environment
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

# 🔌 Streamlit interface (UI + orchestration)

st.set_page_config(layout="wide")
st.header("🧬 Data Processor + Model A/B/C/G Detector with fast mode. v27f")

# 📤 Uploads
small_feed_file = st.file_uploader("Upload small feed", type="csv")
big_feed_file = st.file_uploader("Upload big feed", type="csv")
measurement_file = st.file_uploader("Upload measurement file", type=["xlsx", "xls"])

# 📂 Optional: Upload Final Traveler Report (bypass feed upload)
st.markdown("---")
st.markdown("### 📂 Optional: Upload Final Traveler Report (bypass feed upload)")
bypass_traveler_file = st.file_uploader("Upload Final Traveler Report", type=['xlsx', 'csv'], help="Skip feed processing and upload traveler report directly")

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

filter_future_data = st.checkbox("Restrict analysis to Report Time or earlier only", value=True)

# ⚙️ Analysis parameters
day_start_choice = st.radio("Select Day Start Time", ["18:00", "17:00"])
day_start_hour = int(day_start_choice.split(":")[0])
scope_type = st.radio("Scope by", ["Days", "Rows"])
scope_value = st.number_input(f"Enter number of {scope_type.lower()}", min_value=1, value=20)

# 🎯 Traveler Report Settings (Global Configuration)
st.markdown("---")
st.markdown("### 🎯 Traveler Report Settings")

# Mutually exclusive radio button selection
report_type = st.radio("Select Report Type", ["Full Range", "Custom Ranges"], key="global_report_type")

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
        full_range_value = st.number_input("Full Range Value (±)", min_value=1, value=1100, key="global_full_range")
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

# Process bypass upload if provided
if bypass_traveler_file:
    try:
        st.markdown("### 📊 Processing Bypass Traveler Report")
        
        # Read the bypass file
        if bypass_traveler_file.name.endswith('.csv'):
            final_df = pd.read_csv(bypass_traveler_file)
        else:
            # Handle Excel files with multiple sheets
            xls = pd.ExcelFile(bypass_traveler_file)
            if len(xls.sheet_names) > 1:
                sheet_choice = st.selectbox("Select traveler report tab", xls.sheet_names, key="bypass_sheet")
                final_df = pd.read_excel(bypass_traveler_file, sheet_name=sheet_choice)
            else:
                final_df = pd.read_excel(bypass_traveler_file)
        
        st.success(f"✅ Bypass file loaded successfully: {len(final_df)} rows")
        
        # Extract report time from filename or use current time
        if 'Arrival' in final_df.columns and not final_df.empty:
            try:
                # Try to parse arrival times to determine report time
                arrival_times = pd.to_datetime(final_df['Arrival'], format='%m/%d/%Y %H:%M', errors='coerce')
                # Filter out NaT values and get the maximum valid time
                valid_times = arrival_times.dropna()
                if len(valid_times) > 0:
                    report_time = valid_times.max()
                else:
                    report_time = dt.datetime.now()
            except:
                report_time = dt.datetime.now()
        else:
            report_time = dt.datetime.now()
        
        st.info(f"Report time set to: {report_time.strftime('%m/%d/%Y %H:%M')}")
        
        # Copy to final_df_filtered for consistency
        final_df_filtered = final_df.copy()
        
        # Sort by Output descending, then by Arrival ascending (oldest to newest)
        if 'Output' in final_df_filtered.columns and 'Arrival' in final_df_filtered.columns:
            final_df_filtered = final_df_filtered.sort_values(['Output', 'Arrival'], ascending=[False, True])
        
        # Display results without highlighting (for performance)
        st.markdown(f"**Total Entries:** {len(final_df_filtered)}")
        
        # Display the DataFrame without highlighting
        st.dataframe(final_df_filtered, use_container_width=True)
        
        # Run G.06 Model Detection on bypass report
        if run_g_models:
            st.markdown("---")
            st.markdown("### 🟢 Model G Detection on Bypass Report")
            
            try:
                # Import Group 1a travelers list
                from a_helpers import GROUP_1A_TRAVELERS
                
                # Filter for Group 1a only (M# values in GROUP_1A_TRAVELERS)
                if 'M #' in final_df_filtered.columns:
                    grp_1a_mask = final_df_filtered['M #'].isin(GROUP_1A_TRAVELERS)
                    grp_1a_df = final_df_filtered[grp_1a_mask].copy()
                    
                    if grp_1a_df.empty:
                        st.info("No Group 1a entries found in bypass report for Model G detection")
                    else:
                        st.info(f"Running Model G detection on {len(grp_1a_df)} Group 1a entries from bypass report")
                        
                        # Import the correct Model G function
                        try:
                            from models_g_updated import run_model_g_detection
                        except ImportError:
                            try:
                                from model_g import run_model_g_detection
                            except ImportError:
                                from model_g_detector import run_model_g_detection
                        
                        # Run Model G detection on Group 1a data from bypass report
                        g_results = run_model_g_detection(grp_1a_df, report_time, key_suffix="_bypass")
                        
                        # Handle different return types
                        if isinstance(g_results, dict) and 'success' in g_results:
                            if g_results['success']:
                                # Display summary
                                summary = g_results['summary']
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("o1 (Today)", summary['total_o1'])
                                with col2:
                                    st.metric("o2 (Other Day)", summary['total_o2'])
                                with col3:
                                    st.metric("Total Sequences", summary['total_sequences'])
                                
                                # Display results table if any sequences found
                                if not g_results['results_df'].empty:
                                    st.markdown("#### Bypass Report Model G Results")
                                    st.dataframe(g_results['results_df'], use_container_width=True)
                                else:
                                    st.info("No Model G sequences detected in bypass report Grp 1a data")
                            else:
                                st.error(f"Model G detection error: {g_results['error']}")
                        # If function displays results directly, no additional handling needed
                else:
                    st.warning("No 'M #' column found in bypass report - cannot run Model G detection")
                        
            except ImportError as e:
                st.warning(f"Model G detection not available: {e}")
            except Exception as e:
                st.error(f"Model G detection error: {str(e)}")
        
        # Generate download with report time in filename
        timestamp = report_time.strftime("%y-%m-%d_%H-%M")
        
        # Excel download with highlighting
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            final_df_filtered.to_excel(writer, sheet_name='Final Traveler Report', index=False)
            
            # Get workbook and worksheet for formatting
            workbook = writer.book
            worksheet = writer.sheets['Final Traveler Report']
            
            # Add header formatting
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # Write headers with formatting
            for col_num, value in enumerate(final_df_filtered.columns.values):
                worksheet.write(0, col_num, value, header_format)
        
        excel_filename = f"final_traveler_report_{timestamp}.xlsx"
        
        st.download_button(
            label="📥 Download Final Traveler Report (Excel)", 
            data=excel_buffer.getvalue(),
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # CSV download
        csv_data = final_df_filtered.to_csv(index=False)
        csv_filename = f"final_traveler_report_{timestamp}.csv"
        st.download_button(
            label="📥 Download Final Traveler Report (CSV)",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv"
        )
        
        # Model G detection is already handled above in the bypass section, no need for duplicate
        
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

# Process feeds section (full functionality enabled)
elif small_feed_file and big_feed_file and measurement_file:
    try:
        st.markdown("### 📊 Processing Feed Files")
        
        # Read the uploaded files
        small_df = pd.read_csv(small_feed_file)
        big_df = pd.read_csv(big_feed_file)
        
        # Read measurement file (Excel or CSV) - Support multiple tabs for traveler reports
        if measurement_file.name.endswith('.csv'):
            measurements_df = pd.read_csv(measurement_file)
            available_tabs = ["Single CSV"]
        else:
            # Handle Excel files with multiple sheets
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
        
        # === PRE-RUN TRAVELER REPORTS FROM FIRST FOUR MEASUREMENT TABS ===
        st.markdown("---")
        st.markdown("### 📊 Pre-Run Traveler Reports Generation")
        
        # Performance optimization options
        st.markdown("#### ⚡ Performance Options")
        performance_col1, performance_col2, performance_col3 = st.columns(3)
        
        with performance_col1:
            fast_mode = st.checkbox("🚀 Fast Mode", value=False, help="Skip debug info and detailed analysis")
        with performance_col2:
            parallel_processing = st.checkbox("⚡ Parallel Processing", value=True, help="Process measurement tabs simultaneously")
        with performance_col3:
            minimal_display = st.checkbox("📋 Minimal Display", value=False, help="Show only summary statistics")
        
        # Store Model G custom range setting in session state
        st.session_state['run_model_g_on_custom'] = run_g_on_custom
        
        # Check if we should bypass normal processing first  
        if use_full_range:
            # Full range mode - skip this section entirely, will be handled later
            st.info("🚀 Full Range Mode: Skipping advanced custom range section - will process in dedicated full range section")
            skip_master_list_creation = True
        elif (use_custom_ranges and use_advanced_ranges):
            st.markdown("### 🧮 Advanced Custom Range Processing - Bypassing Full Range")
            st.info("Using H/L/C calculation directly - skipping full range processing for performance")
            st.warning(f"DEBUG: Custom ranges mode detected - High1: {high1 if use_high1 else 'N/A'}, Low1: {low1 if use_low1 else 'N/A'}")
            
            # DEBUG: Show bottom 4 rows of both CSV feeds
            st.warning("DEBUG: Bottom 4 rows of Small Feed CSV:")
            if len(small_df) > 0:
                small_bottom = small_df.tail(4)[['time', 'origin', 'high', 'low', 'close']] if all(col in small_df.columns for col in ['time', 'origin', 'high', 'low', 'close']) else small_df.tail(4)
                for i, (_, row) in enumerate(small_bottom.iterrows()):
                    st.text(f"  Row {len(small_df)-4+i+1}: {row.to_dict()}")
            else:
                st.text("  Small feed is empty")
            
            st.warning("DEBUG: Bottom 4 rows of Big Feed CSV:")
            if len(big_df) > 0:
                big_bottom = big_df.tail(4)[['time', 'origin', 'close']] if all(col in big_df.columns for col in ['time', 'origin', 'close']) else big_df.tail(4)
                for i, (_, row) in enumerate(big_bottom.iterrows()):
                    st.text(f"  Row {len(big_df)-4+i+1}: {row.to_dict()}")
            else:
                st.text("  Big feed is empty")
            
            # Use first measurement tab for advanced processing only
            master_tab_name = available_tabs[0]
            master_measurements_df = pd.read_excel(measurement_file, sheet_name=master_tab_name)
            skip_master_list_creation = True
            st.info("🧮 Bypassing master list creation for advanced/full range processing")
        else:
            # Use MASTER TRAVELER LIST APPROACH for faster processing
            st.markdown("**🚀 Master Traveler List Processing - Using First Measurement Tab**")
            
            # Use first measurement tab for master list generation
            master_tab_name = available_tabs[0]
            master_measurements_df = pd.read_excel(measurement_file, sheet_name=master_tab_name)
            
            st.info(f"📊 Creating master traveler list from '{master_tab_name}' tab, then filtering for 4 sub-reports")
            skip_master_list_creation = False
        
        # Store traveler reports for model access
        traveler_reports = {}
        
        # Set report time if not already set
        if report_time is None:
            # Use most current time from big feed
            big_df['time'] = big_df['time'].apply(clean_timestamp)
            report_time = big_df['time'].max()
            st.info(f"Report time auto-set to most current: {report_time.strftime('%m/%d/%Y %H:%M')}")
        
        # Get input value at day start time
        input_value_at_start = get_input_at_day_start(small_df, report_time, day_start_hour)
        if input_value_at_start is not None:
            st.info(f"Input @ {day_start_hour:02d}:00: {input_value_at_start}")
        
        # Performance timing
        import time
        start_time = time.time()
        
        # Generate master traveler list and filter into 4 sub-reports
        if fast_mode:
            st.success("⚡ Fast Mode Enabled: Skipping debug output and detailed analysis")
        if minimal_display:
            st.success("📋 Minimal Display Enabled: Showing summary statistics only")
        
        # Advanced custom range processing (for custom ranges only, not full range)
        if skip_master_list_creation and not use_full_range:
            # Use the same small_df that was loaded above (which has current data)
            small_csv_data = small_df.copy()
            st.success(f"Using small CSV data: {len(small_csv_data)} rows")
            
            # Use advanced calculation directly on measurement data
            final_df_filtered = apply_custom_ranges_advanced(
                master_measurements_df, small_csv_data, report_time,
                high1, high2, low1, low2, 
                use_high1, use_high2, use_low1, use_low2,
                big_df=big_df,  # Pass big_df for fallback H/L/C data
                run_model_g=run_g_on_custom  # Pass Model G checkbox state
            )
            
            if not final_df_filtered.empty:
                st.success(f"Advanced calculation complete: {len(final_df_filtered)} entries found")
                
                # Debug: Check columns and show output range
                st.info(f"DEBUG: Advanced range results columns: {list(final_df_filtered.columns)}")
                if 'Output' in final_df_filtered.columns:
                    output_min = final_df_filtered['Output'].min()
                    output_max = final_df_filtered['Output'].max()
                    st.warning(f"DEBUG: Output range {output_min} to {output_max}")
                else:
                    st.warning("DEBUG: No 'Output' column found in advanced range results")
                
                # Create traveler reports from filtered data for group classification
                traveler_reports = {}
                
                # Group 1a
                group_1a_mask = final_df_filtered['M #'].isin(GROUP_1A_TRAVELERS)
                traveler_reports["Grp 1a"] = final_df_filtered[group_1a_mask].copy()
                
                # Group 1b
                group_1b_mask = final_df_filtered['M #'].isin(GROUP_1B_TRAVELERS)
                traveler_reports["Grp 1b"] = final_df_filtered[group_1b_mask].copy()
                
                # Group 2a  
                group_2a_mask = final_df_filtered['M #'].isin(GROUP_2A_TRAVELERS)
                traveler_reports["Grp 2a"] = final_df_filtered[group_2a_mask].copy()
                
                # Group 2b
                group_2b_mask = final_df_filtered['M #'].isin(GROUP_2B_TRAVELERS) 
                traveler_reports["Grp 2b"] = final_df_filtered[group_2b_mask].copy()
                
                # Sort each group: Output descending, then Arrival ascending (old to new)
                for group_name, group_df in traveler_reports.items():
                    if not group_df.empty:
                        # Sort by Output if column exists, otherwise keep original order
                        if 'Output' in group_df.columns and 'Arrival' in group_df.columns:
                            traveler_reports[group_name] = group_df.sort_values(
                                ['Output', 'Arrival'], ascending=[False, True]
                            )
                        elif 'Output' in group_df.columns:
                            traveler_reports[group_name] = group_df.sort_values(
                                ['Output'], ascending=[False]
                            )
                        else:
                            traveler_reports[group_name] = group_df.copy()
                
                # Display summary for advanced mode
                processing_time = time.time() - start_time
                st.success(f"✅ Advanced range calculation completed in {processing_time:.1f} seconds")
                
                for group_name, group_df in traveler_reports.items():
                    if not minimal_display and not group_df.empty:
                        st.markdown(f"#### 📋 {group_name}")
                        st.info(f"{len(group_df)} entries")
                        if not fast_mode:
                            st.dataframe(group_df, use_container_width=True)
                    elif not group_df.empty:
                        st.info(f"✅ {group_name}: {len(group_df)} entries processed")
                
                # Set flag to skip normal processing
                skip_normal_processing = True
                
                # Skip the rest of the normal processing and jump to Excel export
                st.markdown("---")
                st.markdown("### ✅ Advanced Custom Range Processing Complete")
                
                # Display range summary  
                range_summary = []
                if use_high1: range_summary.append(f"High 1: {high1}")
                if use_high2: range_summary.append(f"High 2: {high2}")
                if use_low1: range_summary.append(f"Low 1: {low1}")
                if use_low2: range_summary.append(f"Low 2: {low2}")
                st.info(f"Processed ranges: {', '.join(range_summary)}")
                
            else:
                st.warning("No entries found using advanced H/L/C calculation")
                traveler_reports = {}
                skip_normal_processing = True
        
        # Create master list with performance timing (only if not using advanced custom ranges)
        if not skip_master_list_creation:
            with st.spinner("Creating master traveler list and filtering for 4 sub-reports..."):
                try:
                    # Process big and small feeds separately with proper feed labels
                    all_traveler_data = []
                    
                    # Process Big feed data  
                    if len(big_df) > 0:
                        big_df['time'] = big_df['time'].apply(clean_timestamp)
                        big_data = process_feed(big_df, "Big", report_time, scope_type, scope_value, 
                                              day_start_hour, master_measurements_df, input_value_at_start, small_df,
                                              use_full_range, full_range_value)
                        all_traveler_data.extend(big_data)
                    
                    # Process Small feed data
                    if len(small_df) > 0:
                        small_df['time'] = small_df['time'].apply(clean_timestamp)  
                        small_data = process_feed(small_df, "Small", report_time, scope_type, scope_value,
                                                day_start_hour, master_measurements_df, input_value_at_start, small_df,
                                                use_full_range, full_range_value)
                        all_traveler_data.extend(small_data)
                
                    # Convert to DataFrame for group filtering
                    if all_traveler_data:
                        master_df = pd.DataFrame(all_traveler_data)
                        
                        # Filter into 4 groups based on M# values
                        traveler_reports = {}
                        
                        # Group 1a
                        group_1a_mask = master_df['M #'].isin(GROUP_1A_TRAVELERS)
                        traveler_reports["Grp 1a"] = master_df[group_1a_mask].copy()
                        
                        # Group 1b
                        group_1b_mask = master_df['M #'].isin(GROUP_1B_TRAVELERS)
                        traveler_reports["Grp 1b"] = master_df[group_1b_mask].copy()
                        
                        # Group 2a  
                        group_2a_mask = master_df['M #'].isin(GROUP_2A_TRAVELERS)
                        traveler_reports["Grp 2a"] = master_df[group_2a_mask].copy()
                        
                        # Group 2b
                        group_2b_mask = master_df['M #'].isin(GROUP_2B_TRAVELERS) 
                        traveler_reports["Grp 2b"] = master_df[group_2b_mask].copy()
                        
                        # Sort each group: Output descending, then Arrival ascending (old to new)
                        for group_name, group_df in traveler_reports.items():
                            if not group_df.empty:
                                # Keep Arrival_datetime column - user wants it for Excel datetime recognition
                                traveler_reports[group_name] = group_df.sort_values(
                                    ['Output', 'Arrival'], ascending=[False, True]
                                )
                    else:
                        traveler_reports = {}
                    
                    if traveler_reports:
                        # Display summary
                        processing_time = time.time() - start_time
                        st.success(f"✅ Master list created and filtered in {processing_time:.1f} seconds")
                        
                        for group_name, group_df in traveler_reports.items():
                            if not minimal_display and not group_df.empty:
                                st.markdown(f"#### 📋 {group_name}")
                                st.info(f"{len(group_df)} entries")
                                if not fast_mode:
                                    st.dataframe(group_df, use_container_width=True)
                            elif not group_df.empty:
                                st.info(f"✅ {group_name}: {len(group_df)} entries processed")
                    else:
                        st.error("Failed to generate master traveler list")
                        
                except Exception as e:
                    st.error(f"Error generating master traveler list: {str(e)}")
                    traveler_reports = {}
        
        # Excel Export for Master Traveler Lists
        if traveler_reports:
            st.markdown("---")
            st.markdown("### 📥 Unified Excel Download - Master Traveler Groups")
            
            import io
            
            # Use report datetime for filename instead of current time
            report_datetime_str = report_time.strftime("%d-%b-%y_%H-%M") if report_time else dt.datetime.now().strftime("%d-%b-%y_%H-%M")
            
            # Create unified Excel file with all 4 groups
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                for group_name, group_data in traveler_reports.items():
                    if not group_data.empty:
                        # Clean sheet name (Excel limit: 31 chars)
                        sheet_name = group_name.replace(" ", "_").replace("-", "_")[:31]
                        
                        # Remove Group column if present and write data to sheet
                        export_data = group_data.copy()
                        if 'Group' in export_data.columns:
                            export_data = export_data.drop('Group', axis=1)
                        export_data.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Apply highlighting to worksheet (use export_data without Group column)
                        worksheet = writer.sheets[sheet_name]
                        apply_excel_highlighting(workbook, worksheet, export_data, False)  # No custom ranges for master list
            
            excel_buffer.seek(0)
            
            # Show download button
            total_entries = sum(len(df) for df in traveler_reports.values())
            st.download_button(
                label=f"📥 Download All 4 Traveler Groups (Unified Excel)",
                data=excel_buffer,
                file_name=f"master_traveler_groups_{report_datetime_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help=f"Excel file contains {len(traveler_reports)} groups with {total_entries} total entries"
            )
            
            st.success(f"✅ Unified Excel file ready with {len(traveler_reports)} traveler groups ({total_entries} total entries)")
            
            # Performance summary
            processing_time = time.time() - start_time
            st.markdown("---")
            st.markdown("### ⏱️ Performance Summary")
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.metric("Processing Time", f"{processing_time:.1f}s")
            with perf_col2:
                st.metric("Groups Generated", len(traveler_reports))
            with perf_col3:
                st.metric("Total Entries", total_entries)
            
            # Show performance improvement
            if processing_time < 60:  # Less than 1 minute
                st.success(f"🚀 **Excellent Performance**: Master list approach completed in {processing_time:.1f}s")
                st.info("💡 This is much faster than processing each measurement tab separately (previously ~500+ seconds)")
            elif processing_time < 180:  # Less than 3 minutes  
                st.info(f"⚡ **Good Performance**: Completed in {processing_time:.1f}s with master list approach")
            else:
                st.warning(f"⏱️ Processing took {processing_time:.1f}s - consider enabling Fast Mode for better performance")
                
            st.markdown("---")
            st.info("🎯 **Processing Complete**: Master traveler list generated and filtered into 4 groups. Download your unified Excel file above.")
            st.markdown("**Optional**: You can continue below for additional model detection and analysis.")
        
        # Legacy processing fallback (remove old complex loop):
        if False:  # Disabled - using master list approach instead
            for tab_label, tab_name in []:
                st.markdown(f"#### 📋 {tab_label} - {tab_name}")
                
                # Read the specific measurement tab
                if len(available_tabs) > 1:
                    current_measurements_df = pd.read_excel(measurement_file, sheet_name=tab_name)
                else:
                    current_measurements_df = measurements_df
                
                # Process feeds with current measurement tab
                small_data = process_feed(small_df, "Small", report_time, scope_type, scope_value, 
                                        day_start_hour, current_measurements_df, input_value_at_start, small_df)
                
                big_data = process_feed(big_df, "Big", report_time, scope_type, scope_value,
                                      day_start_hour, current_measurements_df, input_value_at_start, small_df)
                
                # Debug: Check feed processing results (skip in fast mode)
                if not fast_mode:
                    st.info(f"Small feed processed: {len(small_data)} entries")
                    st.info(f"Big feed processed: {len(big_data)} entries")
                
                # Combine processed data for this measurement tab
                tab_data = small_data + big_data
                if tab_data:
                    tab_df = pd.DataFrame(tab_data)
                    
                    # Apply filtering if future data restriction is enabled
                    if filter_future_data and report_time:
                        if 'Arrival_datetime' in tab_df.columns:
                            tab_df = tab_df[tab_df['Arrival_datetime'] <= report_time]
                        else:
                            tab_df['Arrival_datetime'] = pd.to_datetime(tab_df['Arrival'], format='%m/%d/%Y %H:%M', errors='coerce')
                            tab_df = tab_df[tab_df['Arrival_datetime'] <= report_time]
                    
                    # Apply traveler report filtering based on global settings
                    tab_df_filtered = tab_df.copy()
                    
                    if use_full_range:
                        # Skip pre-run processing for full range - handled in dedicated section
                        st.info(f"⏩ Skipping {tab_label} pre-run processing - Full Range mode uses dedicated workflow")
                        
                    elif use_custom_ranges:
                        # Custom Ranges mode - WITH screen highlighting
                        if use_advanced_ranges:
                            # Advanced H/L/C calculation mode
                            if not fast_mode:
                                st.markdown("#### 🧮 Advanced H/L/C Range Processing")
                            
                            # Try to use small CSV data for advanced calculation
                            small_csv_data = None
                            try:
                                # Check if small test file exists for fallback
                                small_csv_data = pd.read_csv('small_test.csv')
                                if not fast_mode:
                                    st.success(f"Using small CSV data: {len(small_csv_data)} rows")
                            except:
                                if not fast_mode:
                                    st.warning("No small CSV data found. Falling back to simple range filtering.")
                                use_advanced_ranges = False
                            
                            if use_advanced_ranges and small_csv_data is not None:
                                # Use advanced calculation
                                tab_df_filtered = apply_custom_ranges_advanced(
                                    tab_df_filtered, small_csv_data, report_time,
                                    high1, high2, low1, low2, 
                                    use_high1, use_high2, use_low1, use_low2
                                )
                                
                                if not fast_mode:
                                    if not tab_df_filtered.empty:
                                        st.success(f"Advanced filtering complete: {len(tab_df_filtered)} entries found")
                                    else:
                                        st.warning("No entries found using advanced H/L/C calculation")
                        
                        if not use_advanced_ranges:
                            # Simple range filtering (original method)
                            filter_ranges = []
                            
                            if use_high1 and high1 > 0:
                                filter_ranges.append({"name": "High 1", "type": "high", "upper": high1, "lower": high1 - 24})
                            if use_high2 and high2 > 0:
                                filter_ranges.append({"name": "High 2", "type": "high", "upper": high2, "lower": high2 - 24})
                            if use_low1 and low1 > 0:
                                filter_ranges.append({"name": "Low 1", "type": "low", "upper": low1 + 24, "lower": low1})
                            if use_low2 and low2 > 0:
                                filter_ranges.append({"name": "Low 2", "type": "low", "upper": low2 + 24, "lower": low2})
                            
                            if filter_ranges:
                                # Debug: Show data before filtering (skip in fast mode)
                                original_count = len(tab_df_filtered)
                                
                                if not fast_mode:
                                    # Show sample of outputs before filtering
                                    if 'Output' in tab_df_filtered.columns:
                                        output_values = sorted(tab_df_filtered['Output'].unique())
                                        st.info(f"Sample outputs before filtering: {output_values[:10]}...{output_values[-10:] if len(output_values) > 10 else ''}")
                                    
                                    # Check feed distribution before filtering
                                    if 'Feed' in tab_df_filtered.columns:
                                        feed_counts = tab_df_filtered['Feed'].value_counts()
                                        st.info(f"Feed distribution before filtering: {dict(feed_counts)}")
                                
                                # Filter data to custom ranges
                                mask = pd.Series([False] * len(tab_df_filtered), index=tab_df_filtered.index)
                                for range_info in filter_ranges:
                                    range_mask = (tab_df_filtered['Output'] >= range_info['lower']) & (tab_df_filtered['Output'] <= range_info['upper'])
                                    mask = mask | range_mask
                                    
                                    if not fast_mode:
                                        # Debug info for each range
                                        range_count = range_mask.sum()
                                    st.info(f"Range {range_info['name']}: [{range_info['lower']:.1f}, {range_info['upper']:.1f}] - {range_count} entries")
                                    
                                    # Show which origins are in this range
                                    if range_count > 0:
                                        range_data = tab_df_filtered[range_mask]
                                        origins_in_range = sorted(range_data['Origin'].unique())
                                        st.info(f"Origins in {range_info['name']}: {origins_in_range}")
                                        
                                        # Show breakdown by origin
                                        origin_counts = range_data['Origin'].value_counts()
                                        st.info(f"Entries per origin in {range_info['name']}: {dict(origin_counts)}")
                            
                            tab_df_filtered = tab_df_filtered[mask]
                            filtered_count = len(tab_df_filtered)
                            
                            if not fast_mode:
                                # Check feed distribution after filtering
                                if 'Feed' in tab_df_filtered.columns and filtered_count > 0:
                                    feed_counts_after = tab_df_filtered['Feed'].value_counts()
                                    st.info(f"Feed distribution after filtering: {dict(feed_counts_after)}")
                                
                                st.info(f"Total entries: {original_count} → {filtered_count} after custom range filtering")
                            
                            # Add Range and Zone columns
                            tab_df_filtered['Range'] = tab_df_filtered['Output'].apply(
                                lambda x: next((r['name'] for r in filter_ranges if r['lower'] <= x <= r['upper']), 'Other')
                            )
                            
                            def get_zone(output_val):
                                for range_info in filter_ranges:
                                    if range_info['lower'] <= output_val <= range_info['upper']:
                                        # Calculate distance using the same logic as shared_updated.py
                                        if range_info['type'] == 'high':
                                            # For highs, measure distance from upper limit
                                            distance = range_info['upper'] - output_val
                                        else:
                                            # For lows, measure distance from lower limit
                                            distance = output_val - range_info['lower']
                                        
                                        # Apply the correct zone labels
                                        if distance <= 6:
                                            return "0 to 6"
                                        elif distance <= 12:
                                            return "6 to 12"
                                        elif distance <= 18:
                                            return "12 to 18"
                                        else:
                                            return "18 to 24"
                                return "0 to 6"
                            
                            tab_df_filtered['Zone'] = tab_df_filtered['Output'].apply(get_zone)
                            
                            # Sort by Output descending, then Arrival ascending (oldest to newest)
                            tab_df_filtered = tab_df_filtered.sort_values(['Output', 'Arrival'], ascending=[False, True])
                            
                            if not fast_mode:
                                st.info(f"Custom Ranges: {len(tab_df_filtered)} entries across {len(filter_ranges)} ranges")
                            
                            # Display WITH custom highlighting (Custom Ranges mode) - minimal in fast mode
                            if not minimal_display:
                                display_columns = [col for col in tab_df_filtered.columns if col != 'Arrival_datetime']
                                highlighted_df = highlight_custom_traveler_report(tab_df_filtered[display_columns])
                                st.dataframe(highlighted_df, use_container_width=True)
                            else:
                                st.info(f"✅ {tab_label}: {len(tab_df_filtered)} entries processed")
                        else:
                            st.info("No custom ranges configured - showing full data")
                            # Sort by Output descending, then Arrival ascending (oldest to newest)
                            tab_df_filtered = tab_df_filtered.sort_values(['Output', 'Arrival'], ascending=[False, True])
                            display_columns = [col for col in tab_df_filtered.columns if col != 'Arrival_datetime']
                            st.dataframe(tab_df_filtered[display_columns], use_container_width=True)
                    else:
                        # No filtering applied
                        st.info(f"No filtering applied - showing all {len(tab_df_filtered)} entries")
                        # Sort by Output descending, then Arrival ascending (oldest to newest)
                        tab_df_filtered = tab_df_filtered.sort_values(['Output', 'Arrival'], ascending=[False, True])
                        display_columns = [col for col in tab_df_filtered.columns if col != 'Arrival_datetime']
                        st.dataframe(tab_df_filtered[display_columns], use_container_width=True)
                    
                    # Store the traveler report for model access
                    traveler_reports[tab_label] = tab_df_filtered.copy()
                    
                    # Store data for unified Excel export (no individual downloads)
                else:
                    st.warning(f"No data generated for {tab_label}")
            
            # Unified Excel download with all 4 tabs
            st.markdown("---")
            st.markdown("### 📥 Unified Excel Download")
            
            if traveler_reports:
                # Use report datetime for filename instead of current time
                report_datetime_str = report_time.strftime("%d-%b-%y_%H-%M") if report_time else dt.datetime.now().strftime("%d-%b-%y_%H-%M")
                
                # Create unified Excel file with all tabs
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    workbook = writer.book
                    
                    for tab_label, tab_data in traveler_reports.items():
                        if not tab_data.empty:
                            # Apply full range filtering directly if enabled
                            if use_full_range and input_value_at_start is not None:
                                # Filter using custom range method
                                high_limit = input_value_at_start + full_range_value
                                low_limit = input_value_at_start - full_range_value
                                
                                mask = (tab_data['Output'] >= low_limit) & (tab_data['Output'] <= high_limit)
                                tab_data = tab_data[mask].copy()
                                
                                # Add Range and Zone columns for full range
                                tab_data['Range'] = 'Full Range'
                                
                                def get_full_range_zone(output_val):
                                    distance_from_center = abs(output_val - input_value_at_start)
                                    if distance_from_center <= 6:
                                        return "0 to 6"
                                    elif distance_from_center <= 12:
                                        return "6 to 12"
                                    elif distance_from_center <= 18:
                                        return "12 to 18"
                                    else:
                                        return "18 to 24"
                                
                                tab_data['Zone'] = tab_data['Output'].apply(get_full_range_zone)
                            
                            # Prepare data for export
                            display_columns = [col for col in tab_data.columns if col != 'Arrival_datetime']
                            sheet_name = tab_label.replace(" ", "_").replace("-", "_")[:31]  # Excel sheet name limit
                            
                            # Ensure Arrival column is properly formatted as datetime for Excel (use CSV formatting approach)
                            tab_data_for_export = tab_data[display_columns].copy()
                            if 'Arrival' in tab_data_for_export.columns:
                                # Convert to datetime to ensure proper Excel formatting like CSV
                                tab_data_for_export['Arrival'] = pd.to_datetime(tab_data_for_export['Arrival'], errors='coerce')
                            
                            # Write data to sheet
                            tab_data_for_export.to_excel(writer, sheet_name=sheet_name, index=False)
                            
                            # ALWAYS apply Excel highlighting (regardless of fast mode) and format datetime
                            worksheet = writer.sheets[sheet_name]
                            
                            # Format datetime columns in Excel (match CSV format - no leading zeros)
                            if 'Arrival' in tab_data_for_export.columns:
                                arrival_col = tab_data_for_export.columns.get_loc('Arrival')
                                # Convert datetime values to string format to match CSV exactly: "7/31/2025 22:00"
                                for row_num in range(1, len(tab_data_for_export) + 1):
                                    cell_value = tab_data_for_export.iloc[row_num - 1]['Arrival']
                                    if pd.notna(cell_value):
                                        # Format as string to match CSV: no leading zeros
                                        if hasattr(cell_value, 'strftime'):
                                            formatted_date = cell_value.strftime('%m/%d/%Y %H:%M').lstrip('0').replace('/0', '/')
                                            worksheet.write(row_num, arrival_col, formatted_date)
                                        else:
                                            worksheet.write(row_num, arrival_col, str(cell_value))
                            
                            # Determine if custom ranges were used for this tab
                            has_custom_ranges = 'Range' in tab_data_for_export.columns and 'Zone' in tab_data_for_export.columns
                            # Apply highlighting using the properly formatted export data
                            apply_excel_highlighting(workbook, worksheet, tab_data_for_export, 
                                                    has_custom_ranges)
                
                excel_buffer.seek(0)
                
                # Show download button
                st.download_button(
                    label=f"📥 Download All {len(traveler_reports)} Measurement Tabs (Unified Excel)",
                    data=excel_buffer,
                    file_name=f"unified_traveler_reports_{report_datetime_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help=f"Excel file contains {len(traveler_reports)} tabs with all measurement data"
                )
                
                st.success(f"✅ Unified Excel file ready with {len(traveler_reports)} measurement tabs")
                
                # Stop processing here - user can download and continue if needed
                st.markdown("---")
                st.info("🎯 **Processing Complete**: All 4 measurement tabs have been processed. Download your unified Excel file above.")
                st.markdown("**Optional**: You can continue below for additional model detection and analysis.")
                
            else:
                st.warning("No traveler report data available for download")
                
            # Performance summary
            end_time = time.time()
            processing_time = end_time - start_time
            st.markdown("---")
            st.markdown("### ⏱️ Performance Summary")
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.metric("Processing Time", f"{processing_time:.1f}s")
            with perf_col2:
                st.metric("Tabs Processed", len(selected_tabs))
            with perf_col3:
                avg_time = processing_time / len(selected_tabs) if selected_tabs else 0
                st.metric("Avg Time/Tab", f"{avg_time:.1f}s")
            
            # Performance tips
            if processing_time > 30:
                st.info("💡 **Performance Tips:**")
                if not fast_mode:
                    st.markdown("- Enable **🚀 Fast Mode** to skip debug output")
                if not minimal_display:
                    st.markdown("- Enable **📋 Minimal Display** to show only summaries")
                st.markdown("- Consider processing fewer measurement tabs simultaneously")
            else:
                st.success(f"✅ All {len(selected_tabs)} measurement tab reports completed in {processing_time:.1f} seconds")
        
        # COMPLETELY DISABLE LEGACY PROCESSING FOR FULL RANGE MODE
        if use_full_range:
            # Full range mode - skip ALL legacy processing
            small_data = []
            big_data = []
            st.info("🧮 Skipping legacy feed processing - using advanced custom range method instead")
        else:
            # Use the primary measurement tab for main processing
            # Process small and big feeds with debug info
            st.info("🔍 Processing feeds - checking origins...")
            
            small_data = process_feed(small_df, "Small", report_time, scope_type, scope_value, 
                                    day_start_hour, measurements_df, input_value_at_start, small_df)
            
            big_data = process_feed(big_df, "Big", report_time, scope_type, scope_value,
                                  day_start_hour, measurements_df, input_value_at_start, small_df)
        
        # Show debug info about big feed processing (only for legacy mode)
        if not use_full_range:
            st.info(f"Debug: Small feed generated {len(small_data)} rows")
            st.info(f"Debug: Big feed generated {len(big_data)} rows")
        
        # Display big feed debug info if available (only for legacy mode)
        if not use_full_range:
            try:
                from a_helpers import big_feed_debug_info
                if 'big_feed_debug_info' in globals() or hasattr(a_helpers, 'big_feed_debug_info'):
                    debug_info = getattr(a_helpers, 'big_feed_debug_info', {})
                    if debug_info:
                        st.info(f"🔍 Big Feed Analysis:")
                        st.info(f"- Total columns: {debug_info.get('column_count', 'unknown')}")
                        st.info(f"- Origins detected: {debug_info.get('total_origins', 'unknown')}")
                        st.info(f"- Origin names: {debug_info.get('origin_names', [])}")
            except:
                pass
        
        # Combine processed data (skip for full range mode)
        if not use_full_range:
            all_data = small_data + big_data
            if not all_data:
                st.warning("No data generated from feed processing")
                st.stop()
            
            # Create final DataFrame
            final_df = pd.DataFrame(all_data)
        else:
            # For full range, skip legacy data processing - handled in dedicated section
            final_df = pd.DataFrame()  # Temporary empty dataframe
        
        # Drop the datetime column from display but keep for filtering
        display_columns = [col for col in final_df.columns if col != 'Arrival_datetime']
        
        # Filter future data if requested (skip for full range mode)
        if not use_full_range and filter_future_data and report_time:
            initial_count = len(final_df)
            # Use datetime column for comparison, fallback to converting Arrival if needed
            if 'Arrival_datetime' in final_df.columns:
                final_df = final_df[final_df['Arrival_datetime'] <= report_time]
            else:
                # Convert Arrival column to datetime for comparison
                final_df['Arrival_datetime'] = pd.to_datetime(final_df['Arrival'], format='%d-%b-%Y %H:%M', errors='coerce')
                final_df = final_df[final_df['Arrival_datetime'] <= report_time]
            filtered_count = len(final_df)
            if initial_count != filtered_count:
                st.info(f"Filtered {initial_count - filtered_count} future entries. Showing {filtered_count} entries at or before report time.")
        
        # FULL RANGE PROCESSING - Rewritten to match custom range workflow
        if use_full_range:
            st.markdown("---")
            st.markdown("### 🧮 Advanced Custom Range Processing - Bypassing Full Range")
            st.info("Using H/L/C calculation directly - skipping full range processing for performance")
            
            # Processing message matching custom range format
            if input_value_at_start is not None:
                range_low = input_value_at_start - full_range_value
                range_high = input_value_at_start + full_range_value
                st.markdown(f"**Processing full range from \"{range_low:.1f}\" to \"{range_high:.1f}\"**")
            else:
                st.error("No input value at day start found - cannot calculate full range")
                st.stop()
            
            # Small CSV Feed Debug Info
            st.markdown("**Small CSV Feed**")
            origins_small = small_df['origin'].unique() if 'origin' in small_df.columns else []
            st.markdown(f"Debug: Found {len(origins_small)} origins: {', '.join(origins_small) if len(origins_small) <= 10 else ', '.join(origins_small[:10]) + '...'}")
            
            if len(origins_small) > 0:
                # Show first origin's datetime entries
                first_origin = origins_small[0]
                origin_data = small_df[small_df['origin'] == first_origin] if 'origin' in small_df.columns else small_df
                datetime_entries = len(origin_data)
                st.markdown(f"Debug: Found {datetime_entries} datetime entries for {first_origin}:")
                if len(origin_data) > 0:
                    first_entry = origin_data.iloc[0]
                    st.markdown(f"First entry – DateTime: {first_entry.get('time', 'N/A')}")
            
            # Big CSV Feed Debug Info  
            st.markdown("**Big CSV Feed**")
            origins_big = big_df['origin'].unique() if 'origin' in big_df.columns else []
            st.markdown(f"Debug: Found {len(origins_big)} origins: {', '.join(origins_big) if len(origins_big) <= 10 else ', '.join(origins_big[:10]) + '...'}")
            
            if len(origins_big) > 0:
                # Show first origin's datetime entries
                first_origin_big = origins_big[0]
                origin_data_big = big_df[big_df['origin'] == first_origin_big] if 'origin' in big_df.columns else big_df
                datetime_entries_big = len(origin_data_big)
                st.markdown(f"Debug: Found {datetime_entries_big} datetime entries for {first_origin_big}:")
                if len(origin_data_big) > 0:
                    first_entry_big = origin_data_big.iloc[0]
                    st.markdown(f"First entry – DateTime: {first_entry_big.get('time', 'N/A')}")
            
            # Process feeds exactly like custom ranges - using first measurement tab only
            performance_start = time.time()
            
            # Use first measurement tab (master tab)
            master_measurements_df = pd.read_excel(measurement_file, sheet_name=available_tabs[0])
            
            # Use custom range approach for full range processing
            from custom_range_calculator import apply_custom_ranges_advanced
            
            # Convert full range to custom range format (one big range)
            range_center = input_value_at_start
            high_range = range_center + full_range_value
            low_range = range_center - full_range_value
            
            # DEBUG: Show range calculation
            st.info(f"DEBUG: Range calculation - Center: {range_center}, Full Range Value: {full_range_value}")
            st.info(f"DEBUG: Calculated High: {high_range}, Calculated Low: {low_range}")
            st.info(f"DEBUG: Total range span: {high_range - low_range}")
            
            # Use advanced calculation directly on measurement data (like custom ranges)
            full_range_results = apply_custom_ranges_advanced(
                master_measurements_df, small_df, report_time,
                high_range, high_range, low_range, low_range,  # Set both high1/high2 and low1/low2 to same values
                True, False, True, False,  # use_high1=True, use_high2=False, use_low1=True, use_low2=False
                big_df=big_df,  # Pass big_df for fallback H/L/C data
                run_model_g=False  # Skip Model G for now
            )
            
            # Convert results to same format as old process_feed output
            st.info(f"DEBUG: apply_custom_ranges_advanced returned: {type(full_range_results)}")
            st.info(f"DEBUG: Is empty? {full_range_results.empty if hasattr(full_range_results, 'empty') else 'Not a DataFrame'}")
            
            if not full_range_results.empty:
                # Debug: Check what columns we actually have
                st.info(f"DEBUG: Full range results columns: {list(full_range_results.columns)}")
                st.info(f"DEBUG: Full range results shape: {full_range_results.shape}")
                
                # The apply_custom_ranges_advanced already filtered the data
                final_df_filtered = full_range_results.copy()
                
                # Add Range column for consistency
                final_df_filtered['Range'] = 'Full Range'
                
                # Sort by Output descending, then Arrival ascending (if Output column exists)
                if 'Output' in final_df_filtered.columns:
                    final_df_filtered = final_df_filtered.sort_values(['Output', 'Arrival'], ascending=[False, True])
                else:
                    st.warning("DEBUG: No 'Output' column found - using unsorted results")
            else:
                final_df_filtered = pd.DataFrame()
                st.warning("DEBUG: apply_custom_ranges_advanced returned empty DataFrame")
            
            st.markdown(f"**Full range: found {len(final_df_filtered)} valid entries**")
            
            # M Values from Measurement File debug
            st.markdown("**M Values from Measurement File**")
            st.markdown(f"Available columns: {list(master_measurements_df.columns)}")
            
            # Count M# and M Values 
            m_hash_count = len(master_measurements_df) if 'M #' in master_measurements_df.columns or 'm #' in master_measurements_df.columns else 0
            m_value_count = len(master_measurements_df) if any('value' in col.lower() for col in master_measurements_df.columns) else 0
            st.markdown(f"Total M # entries: {m_hash_count}\t\tTotal M Values: {m_value_count}")
            st.markdown("ℹ️ Note: Raw M calculations compare against 'M Value' column, not 'M #' column")
            
            # Processing Summary
            st.markdown("**Processing Summary**")
            processing_time = time.time() - performance_start
            st.markdown(f"Advanced Calculation for full range complete: {len(final_df_filtered)} entries found.")
            
            # Debug output range (if Output column exists)
            if len(final_df_filtered) > 0:
                if 'Output' in final_df_filtered.columns:
                    output_min = final_df_filtered['Output'].min()
                    output_max = final_df_filtered['Output'].max() 
                    st.markdown(f"DEBUG: Output range {output_min:.1f} to {output_max:.1f}")
                else:
                    st.warning("DEBUG: Cannot show output range - 'Output' column missing")
                
            st.markdown(f"Advanced full range calculation completed in {processing_time:.1f} seconds.")
            
            # Group results exactly like custom ranges (if we have data)
            st.info(f"DEBUG: About to group {len(final_df_filtered)} entries")
            if not final_df_filtered.empty:
                traveler_reports = {}
                
                # Group 1a
                if 'M #' in final_df_filtered.columns:
                    st.info("DEBUG: 'M #' column found, proceeding with grouping")
                    group_1a_mask = final_df_filtered['M #'].isin(GROUP_1A_TRAVELERS)
                    traveler_reports["Grp 1a"] = final_df_filtered[group_1a_mask].copy()
                    st.info(f"DEBUG: Grp 1a created with {len(traveler_reports['Grp 1a'])} entries")
                    
                    # Group 1b  
                    group_1b_mask = final_df_filtered['M #'].isin(GROUP_1B_TRAVELERS)
                    traveler_reports["Grp 1b"] = final_df_filtered[group_1b_mask].copy()
                    st.info(f"DEBUG: Grp 1b created with {len(traveler_reports['Grp 1b'])} entries")
                    
                    # Group 2a
                    group_2a_mask = final_df_filtered['M #'].isin(GROUP_2A_TRAVELERS)
                    traveler_reports["Grp 2a"] = final_df_filtered[group_2a_mask].copy()
                    st.info(f"DEBUG: Grp 2a created with {len(traveler_reports['Grp 2a'])} entries")
                    
                    # Group 2b
                    group_2b_mask = final_df_filtered['M #'].isin(GROUP_2B_TRAVELERS)
                    traveler_reports["Grp 2b"] = final_df_filtered[group_2b_mask].copy()
                    st.info(f"DEBUG: Grp 2b created with {len(traveler_reports['Grp 2b'])} entries")
                    
                    # Sort each group by Output descending, Arrival ascending (if columns exist)
                    st.info("DEBUG: About to sort groups...")
                    for group_name, group_df in traveler_reports.items():
                        if not group_df.empty:
                            if 'Output' in group_df.columns and 'Arrival' in group_df.columns:
                                traveler_reports[group_name] = group_df.sort_values(
                                    ['Output', 'Arrival'], ascending=[False, True]
                                )
                            elif 'Output' in group_df.columns:
                                traveler_reports[group_name] = group_df.sort_values(
                                    ['Output'], ascending=[False]
                                )
                    st.info("DEBUG: Groups sorted, about to display unified report...")
                    
                    # Display unified report FIRST
                    st.markdown("---")
                    st.markdown("### 📊 Unified Report by Groups")
                    st.info("DEBUG: Starting unified report display...")
                    
                    total_entries = sum(len(df) for df in traveler_reports.values())
                    st.info(f"Found {len(traveler_reports)} groups with {total_entries} total entries")
                    st.info("DEBUG: About to enter group display loop...")
                    
                    # Display each group
                    for group_name, group_df in traveler_reports.items():
                        st.markdown(f"#### 📋 {group_name}")
                        st.info(f"{len(group_df)} entries")
                        
                        if not group_df.empty:
                            st.dataframe(group_df, use_container_width=True)
                        else:
                            st.warning(f"{group_name} is empty")
                            
                        # Add separator between groups
                        if group_name != "Grp 2b":
                            st.markdown("---")
                    
                    # Processing summary at the end
                    st.markdown("---")
                    st.markdown("### ✅ Full Range Processing Complete")
                    st.success(f"Advanced range calculation completed in {processing_time:.1f} seconds")
                    st.info(f"Processed full range: {low_range:.1f} to {high_range:.1f}")
                    st.success("🎉 Full Range Processing Successfully Completed!")
                    st.stop()
                    
                else:
                    st.warning("DEBUG: No 'M #' column found in full range results - cannot group by travelers")
                    traveler_reports = {}
            else:
                st.warning("No data generated from full range processing")
                traveler_reports = {}
        elif use_custom_ranges:
            # Custom Ranges mode - only advanced calculation method supported
            if use_advanced_ranges:
                st.error("⚠️ Custom range processing should have been handled earlier in the advanced processing section. This code path should not be reached.")
            else:
                st.error("⚠️ Simple range filtering has been removed. Please use Advanced H/L/C Calculation mode only.")
        else:
            # Default mode - use processed data as-is
            final_df_filtered = final_df.copy()
        
        # Sort main data for model processing (no display section - replaced by pre-run reports)
        if 'final_df_filtered' in locals() and not final_df_filtered.empty and 'Output' in final_df_filtered.columns:
            final_df_filtered = final_df_filtered.sort_values(['Output', 'Arrival'], ascending=[False, True])
        elif 'final_df_filtered' not in locals():
            final_df_filtered = pd.DataFrame()  # Initialize empty DataFrame if not set
        
        # Model detections on the processed data
        if run_g_models:
            st.markdown("---")
            
            # Use Meas tab 1 data if available (preferred for Model G)
            if "traveler_reports" in locals() and "Meas tab 1" in traveler_reports:
                st.info("🎯 Using Meas tab 1 data for Model G detection")
                detection_data = traveler_reports["Meas tab 1"]
            else:
                st.info("Using main processed data for Model G detection")
                detection_data = final_df_filtered
            
            # Run Model G detection (handles its own display now)
            try:
                g_results = run_model_g_detection(detection_data, report_time, key_suffix="_main")
                # If it returns a structured result, display it
                if isinstance(g_results, dict) and 'success' in g_results:
                    if g_results['success']:
                        # Display summary
                        summary = g_results['summary']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("o1 (Today)", summary['total_o1'])
                        with col2:
                            st.metric("o2 (Other Day)", summary['total_o2'])
                        with col3:
                            st.metric("Total Sequences", summary['total_sequences'])
                        
                        # Display results table if any sequences found
                        if not g_results['results_df'].empty:
                            st.markdown("#### Detection Results")
                            st.dataframe(g_results['results_df'], use_container_width=True)
                        else:
                            st.info("No Model G sequences detected matching criteria")
                    else:
                        st.error(f"Model G detection error: {g_results['error']}")
                # If it doesn't return structured results, it displays directly in Streamlit
            except Exception as e:
                st.error(f"Model G detection error: {str(e)}")
        
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
