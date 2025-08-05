# v26f - Final cleanup, auto-generate traveler reports without button press, remove debug messages

import streamlit as st
import pandas as pd
import datetime as dt
import io
from pandas import ExcelWriter

# Configure pandas to handle large datasets
pd.set_option("styler.render.max_elements", 2000000)

# Import functions - these paths are confirmed working
from a_helpers import clean_timestamp, process_feed, get_input_value, highlight_traveler_report, get_input_at_time, get_input_at_day_start, highlight_custom_traveler_report
from models.models_g import run_model_g_detection
from models.models_a_today import run_a_model_detection_today
from models.mod_b_05pg1 import run_b_model_detection
from models.mod_c_04gpr3 import run_c_model_detection
from models.mod_x_03g import run_x_model_detection
from models.simple_mega_report2 import run_simple_single_line_analysis

# 🔌 Streamlit interface (UI + orchestration)

st.set_page_config(layout="wide")
st.header("🧬 Data Feed Processor + Model A/B/C Detector. v26k")

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
    st.write("Configure up to 4 custom ranges:")
    
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
                arrival_times = pd.to_datetime(final_df['Arrival'], format='%d-%b-%Y %H:%M', errors='coerce')
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
        
        st.info(f"Report time set to: {report_time.strftime('%d-%b-%y %H:%M')}")
        
        # Copy to final_df_filtered for consistency
        final_df_filtered = final_df.copy()
        
        # Sort by Output descending, then by Arrival ascending (oldest to newest)
        if 'Output' in final_df_filtered.columns and 'Arrival' in final_df_filtered.columns:
            final_df_filtered = final_df_filtered.sort_values(['Output', 'Arrival'], ascending=[False, True])
        
        # Display results without highlighting (for performance)
        st.markdown(f"**Total Entries:** {len(final_df_filtered)}")
        
        # Display the DataFrame without highlighting
        st.dataframe(final_df_filtered, use_container_width=True)
        
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
        
        # Add model detection options for bypass mode
        if run_g_models:
            st.markdown("---")
            st.markdown("### 🟢 Model G Detection System")
            
            proximity_threshold = st.slider(
                "Proximity Threshold", 
                min_value=0.01, 
                max_value=1.0, 
                value=0.10, 
                step=0.01,
                help="Maximum distance between outputs to group them together",
                key="model_g_slider_bypass"
            )
            
            with st.spinner("Running Model G detection..."):
                try:
                    run_model_g_detection(final_df_filtered, proximity_threshold)
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
        st.error(f"Error processing bypass file: {str(e)}")
        st.stop()

# Process feeds section (full functionality enabled)
elif small_feed_file and big_feed_file and measurement_file:
    try:
        st.markdown("### 📊 Processing Feed Files")
        
        # Read the uploaded files
        small_df = pd.read_csv(small_feed_file)
        big_df = pd.read_csv(big_feed_file)
        
        # Read measurement file (Excel or CSV)
        if measurement_file.name.endswith('.csv'):
            measurements_df = pd.read_csv(measurement_file)
        else:
            # Handle Excel files with multiple sheets
            xls = pd.ExcelFile(measurement_file)
            if len(xls.sheet_names) > 1:
                sheet_choice = st.selectbox("Select measurement tab", xls.sheet_names)
                measurements_df = pd.read_excel(measurement_file, sheet_name=sheet_choice)
            else:
                measurements_df = pd.read_excel(measurement_file)
        
        st.success(f"✅ Files loaded successfully:")
        st.markdown(f"- Small feed: {len(small_df)} rows")
        st.markdown(f"- Big feed: {len(big_df)} rows") 
        st.markdown(f"- Measurements: {len(measurements_df)} rows")
        
        # Set report time if not already set
        if report_time is None:
            # Use most current time from big feed
            big_df['time'] = big_df['time'].apply(clean_timestamp)
            report_time = big_df['time'].max()
            st.info(f"Report time auto-set to most current: {report_time.strftime('%d-%b-%y %H:%M')}")
        
        # Get input value at day start time
        input_value_at_start = get_input_at_day_start(small_df, report_time, day_start_hour)
        if input_value_at_start is not None:
            st.info(f"Input @ {day_start_hour:02d}:00: {input_value_at_start}")
        
        # Process small and big feeds (without debug messages)
        small_data = process_feed(small_df, "Small", report_time, scope_type, scope_value, 
                                day_start_hour, measurements_df, input_value_at_start, small_df)
        
        big_data = process_feed(big_df, "Big", report_time, scope_type, scope_value,
                              day_start_hour, measurements_df, input_value_at_start, small_df)
        
        # Combine processed data
        all_data = small_data + big_data
        if not all_data:
            st.warning("No data generated from feed processing")
            st.stop()
        
        # Create final DataFrame
        final_df = pd.DataFrame(all_data)
        
        # Drop the datetime column from display but keep for filtering
        display_columns = [col for col in final_df.columns if col != 'Arrival_datetime']
        
        # Filter future data if requested
        if filter_future_data and report_time:
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
        
        # Generate traveler report automatically
        st.markdown("---")
        st.markdown("### 📋 Final Traveler Report")
        st.markdown(f"**Report Time:** {report_time.strftime('%d-%b-%y %H:%M')}")
        st.markdown(f"**Day Start Time:** {day_start_hour:02d}:00")
        
        # Apply traveler report filtering if configured
        final_df_filtered = final_df.copy()
        
        if use_full_range:
            # Full Range mode - filter based on Input @ day start time
            if input_value_at_start is not None:
                high_limit = input_value_at_start + full_range_value
                low_limit = input_value_at_start - full_range_value
                
                # Filter the data
                mask = (final_df_filtered['Output'] >= low_limit) & (final_df_filtered['Output'] <= high_limit)
                final_df_filtered = final_df_filtered[mask]
                
                # Add Range and Zone columns
                final_df_filtered['Range'] = 'Full Range'
                final_df_filtered['Zone'] = final_df_filtered['Output'].apply(
                    lambda x: f"Zone {min(int((abs(x - input_value_at_start) / full_range_value) * 4), 3) + 1}"
                )
                
                st.info(f"Full Range filtering: {input_value_at_start} ± {full_range_value} = [{low_limit:.1f}, {high_limit:.1f}]")
                st.info(f"Filtered to {len(final_df_filtered)} entries within range")
        
        elif use_custom_ranges:
            # Custom Ranges mode
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
                # Filter data to custom ranges
                mask = pd.Series([False] * len(final_df_filtered))
                for range_info in filter_ranges:
                    range_mask = (final_df_filtered['Output'] >= range_info['lower']) & (final_df_filtered['Output'] <= range_info['upper'])
                    mask = mask | range_mask
                
                final_df_filtered = final_df_filtered[mask]
                
                # Add Range and Zone columns
                final_df_filtered['Range'] = final_df_filtered['Output'].apply(
                    lambda x: next((r['name'] for r in filter_ranges if r['lower'] <= x <= r['upper']), 'Other')
                )
                
                def get_zone(output_val):
                    for range_info in filter_ranges:
                        if range_info['lower'] <= output_val <= range_info['upper']:
                            if range_info['type'] == 'high':
                                distance = range_info['upper'] - output_val
                            else:
                                distance = output_val - range_info['lower']
                            
                            if distance <= 6:
                                return "0 to 6"
                            elif distance <= 12:
                                return "6 to 12"
                            elif distance <= 18:
                                return "12 to 18"
                            else:
                                return "18 to 24"
                    return ""
                
                final_df_filtered['Zone'] = final_df_filtered['Output'].apply(get_zone)
                st.info(f"Custom Ranges filtering: {len(filter_ranges)} ranges active")
                st.info(f"Filtered to {len(final_df_filtered)} entries within ranges")
        
        # Sort by Output descending, then by Arrival ascending (oldest to newest)
        final_df_filtered = final_df_filtered.sort_values(['Output', 'Arrival'], ascending=[False, True])
        
        # Display results without highlighting (for performance)
        st.markdown(f"**Total Entries:** {len(final_df_filtered)}")
        
        display_df = final_df_filtered[display_columns]
        st.dataframe(display_df, use_container_width=True)
        
        # Download buttons with report time in filename
        timestamp = report_time.strftime("%y-%m-%d_%H-%M")
        
        # CSV download (exclude datetime column)
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Final Traveler Report (CSV)",
            data=csv_data,
            file_name=f"final_traveler_report_{timestamp}.csv",
            mime="text/csv"
        )
        
        # Excel download with highlighting
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            display_df.to_excel(writer, sheet_name='Final Traveler Report', index=False)
            
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
            for col_num, value in enumerate(display_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Apply highlighting based on the report type
            if use_custom_ranges:
                # Apply custom range highlighting for Excel
                highlighted_excel_df = highlight_custom_traveler_report(display_df.copy(), filter_ranges)
            else:
                # Apply standard highlighting for Excel
                highlighted_excel_df = highlight_traveler_report(display_df.copy())
        
        st.download_button(
            label="📥 Download Final Traveler Report (Excel)",
            data=excel_buffer.getvalue(), 
            file_name=f"final_traveler_report_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Model detections on the processed data
        if run_g_models:
            st.markdown("---")
            st.markdown("### 🟢 Model G Detection System")
            
            proximity_threshold = st.slider(
                "Proximity Threshold", 
                min_value=0.01, 
                max_value=1.0, 
                value=0.10, 
                step=0.01,
                help="Maximum distance between outputs to group them together",
                key="model_g_slider_processed"
            )
            
            with st.spinner("Running Model G detection..."):
                try:
                    run_model_g_detection(final_df_filtered, proximity_threshold)
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
