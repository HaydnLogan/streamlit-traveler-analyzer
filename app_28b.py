# v28b - first cleanup attempt



import streamlit as st
import pandas as pd
import datetime as dt
import io
from pandas import ExcelWriter
from custom_range_calculator import apply_custom_ranges_advanced

# Configure pandas to handle large datasets
pd.set_option("styler.render.max_elements", 2000000)

# Import functions - these paths are confirmed working
from a_helpers import clean_timestamp, process_feed, get_input_value, highlight_traveler_report, get_input_at_time, get_input_at_day_start, highlight_custom_traveler_report, apply_excel_highlighting, generate_master_traveler_list, GROUP_1A_TRAVELERS, GROUP_1B_TRAVELERS, GROUP_2A_TRAVELERS, GROUP_2B_TRAVELERS


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
st.header("🧬 Data Processor + Model A/B/C/G Detector with fast mode. v28b")

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
        
        # Create master list with performance timing (only if not using advanced custom ranges AND not using full range)
        if not skip_master_list_creation and not use_full_range:
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
        
        # Legacy processing section removed - using master list approach instead
        
        # Process feeds for non-full-range modes
        if not use_full_range:
            st.info("🔍 Processing feeds - checking origins...")
            
            small_data = process_feed(small_df, "Small", report_time, scope_type, scope_value, 
                                    day_start_hour, measurements_df, input_value_at_start, small_df)
            
            big_data = process_feed(big_df, "Big", report_time, scope_type, scope_value,
                                  day_start_hour, measurements_df, input_value_at_start, small_df)
            
            st.info(f"Debug: Small feed generated {len(small_data)} rows")
            st.info(f"Debug: Big feed generated {len(big_data)} rows")
        else:
            # Full range mode skips legacy feed processing
            small_data = []
            big_data = []
        
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
        
        # ===== FULL RANGE PROCESSING =====
        if use_full_range:
            from full_range_processor import process_full_range, display_full_range_results
            st.markdown("---")
            st.markdown("### 🌐 Full Range Processing Mode")
            
            # Process full range
            result_data = process_full_range(
                measurements_df, small_df, big_df, report_time,
                input_value_at_start, full_range_value,
                GROUP_1A_TRAVELERS, GROUP_1B_TRAVELERS, GROUP_2A_TRAVELERS, GROUP_2B_TRAVELERS
            )
            
            # Display results and get traveler_reports for Excel export
            traveler_reports = display_full_range_results(result_data)
            
            # Continue to Excel export section instead of stopping here
            # (removed st.stop() so Excel export can process the results)
        
        # ===== CUSTOM RANGE PROCESSING =====  
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
                st.info("Make sure models_g_updated.py exists and contains run_model_g_detection function")
        
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
