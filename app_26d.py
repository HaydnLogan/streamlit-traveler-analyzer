# v26d - Cleaned up version with proper imports and measurement tab selection

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
st.header("🧬 Data Feed Processor + Model A/B/C Detector. v26d")

# 📤 Uploads
small_feed_file = st.file_uploader("Upload small feed", type="csv")
big_feed_file = st.file_uploader("Upload big feed", type="csv")
measurement_file = st.file_uploader("Upload measurement file", type=["xlsx", "xls", "csv"])

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
full_range_value = 1100.0
use_high1 = use_high2 = use_low1 = use_low2 = False
high1 = high2 = low1 = low2 = 0.0

if report_type == "Full Range":
    st.markdown("**Full Range Configuration**")
    full_range_value = st.number_input("Range (+/-)", value=1100.0, format="%.1f", key="global_full_range_input")
    use_full_range = True
    use_custom_ranges = False
  
elif report_type == "Custom Ranges":
    st.markdown("**Custom Ranges Configuration**")
    use_full_range = False
    use_custom_ranges = True
    
    # Create 4-column layout with individual toggles
    cols = st.columns(4)
    with cols[0]:
        use_high1 = st.checkbox("High 1", value=True, key="global_use_high1")
        high1 = st.number_input("High 1 Value", value=500.0, format="%.1f", key="global_high1", disabled=not use_high1)
    with cols[1]:
        use_high2 = st.checkbox("High 2", value=True, key="global_use_high2") 
        high2 = st.number_input("High 2 Value", value=480.0, format="%.1f", key="global_high2", disabled=not use_high2)
    with cols[2]:
        use_low1 = st.checkbox("Low 1", value=True, key="global_use_low1")
        low1 = st.number_input("Low 1 Value", value=150.0, format="%.1f", key="global_low1", disabled=not use_low1)
    with cols[3]:
        use_low2 = st.checkbox("Low 2", value=True, key="global_use_low2")
        low2 = st.number_input("Low 2 Value", value=250.0, format="%.1f", key="global_low2", disabled=not use_low2)
      
st.markdown("---")

# 📥 Optional bypass: Upload a Final Traveler Report
st.markdown("### 📥 Optional: Upload Final Traveler Report (bypass feed upload)")
final_report_file = st.file_uploader("Upload Final Traveler Report", type="csv", key="final_report_bypass")

if final_report_file:
    final_df = pd.read_csv(final_report_file)
    final_df.columns = final_df.columns.str.strip()
    
    # Ensure Output column exists
    if 'Output' not in final_df.columns:
        st.error("Required 'Output' column not found in uploaded file. Available columns: " + str(list(final_df.columns)))
        st.stop()
    
    # Parse datetime with improved format handling and validation
    final_df["Arrival"] = pd.to_datetime(final_df["Arrival"], errors="coerce")
    
    # Validate reasonable date ranges (should be within reasonable bounds)
    current_year = pd.Timestamp.now().year
    min_year = current_year - 10  # 10 years ago
    max_year = current_year + 10   # 10 years from now
    
    # Filter out unreasonable dates
    mask = (final_df["Arrival"].dt.year >= min_year) & (final_df["Arrival"].dt.year <= max_year)
    if not mask.all():
        corrupted_count = (~mask).sum()
        st.warning(f"Found {corrupted_count} entries with corrupted dates (outside {min_year}-{max_year}). These will be excluded from analysis.")
        final_df = final_df[mask]

    # Set report_time using raw Arrival
    if report_mode == "Most Current":
        report_time = final_df["Arrival"].max()
        
    # Calculate Day column if it doesn't exist or has incorrect values
    if 'Day' not in final_df.columns or not final_df['Day'].astype(str).str.contains(r'\[').any():
        st.info("Calculating Day column based on arrival times and report time...")
        
        # Calculate day differences from report time
        final_df['Day_calc'] = (final_df['Arrival'].dt.date - report_time.date()).dt.days
        
        # Format as [X] strings
        final_df['Day'] = final_df['Day_calc'].apply(lambda x: f'[{int(x)}]')
        
        # Drop temporary calculation column
        final_df = final_df.drop('Day_calc', axis=1)
        
        st.success(f"Day column calculated. Today=[0] sequences: {(final_df['Day'] == '[0]').sum()}, Other day sequences: {(final_df['Day'] != '[0]').sum()}")

    st.success(f"✅ Loaded Final Traveler Report with {len(final_df)} rows. Report time set to: {report_time.strftime('%d-%b-%y %H:%M')}")
    
    # Model G Detection
    if run_g_models:
        st.markdown("---")
        st.markdown("### 🟢 Model G Detection System")
        
        # Proximity threshold setting
        proximity_threshold = st.slider(
            "Proximity Threshold", 
            min_value=0.01, 
            max_value=1.0, 
            value=0.10, 
            step=0.01,
            help="Maximum distance between outputs to group them together"
        )
        
        if st.button("🔍 Run Model G Detection", key="model_g_btn"):
            with st.spinner("Running Model G detection..."):
                run_model_g_detection(final_df, proximity_threshold)
    
    # Other model detections
    if run_single_line:
        st.markdown("---")
        run_simple_single_line_analysis(final_df)
    
    if run_a_models:
        st.markdown("---")
        run_a_model_detection_today(final_df)
        
    if run_b_models:
        st.markdown("---")
        run_b_model_detection(final_df)
        
    if run_c_models:
        st.markdown("---")
        run_c_model_detection(final_df, run_c01=run_c01, run_c02=run_c02, run_c04=run_c04)
        
    if run_x_models:
        st.markdown("---")
        run_x_model_detection(final_df)

# Main processing section (feeds upload)
elif small_feed_file and big_feed_file and measurement_file:
    
    # Processing feeds
    if True:  # Always process when files are uploaded
        st.markdown("### 📊 Processing Feed Files")
        
        # Read the uploaded files
        small_df = pd.read_csv(small_feed_file)
        big_df = pd.read_csv(big_feed_file)
        
        # Read measurement file (Excel or CSV)
        if measurement_file.name.endswith('.csv'):
            measurements_df = pd.read_csv(measurement_file)
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
        
        # Process small and big feeds
        st.markdown("#### Processing Small Feed...")
        small_data = process_feed(small_df, "Small", report_time, scope_type, scope_value, 
                                day_start_hour, measurements_df, input_value_at_start, small_df)
        st.success(f"✅ Small feed processed: {len(small_data)} entries generated")
        
        st.markdown("#### Processing Big Feed...")
        big_data = process_feed(big_df, "Big", report_time, scope_type, scope_value,
                              day_start_hour, measurements_df, input_value_at_start, small_df)
        st.success(f"✅ Big feed processed: {len(big_data)} entries generated")
        
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
        
        # Measurement selection tab
        st.markdown("---")
        st.markdown("### 📋 Measurement Selection")
        
        # Show available measurements for selection
        if 'Output' in final_df.columns:
            unique_outputs = sorted(final_df['Output'].unique())
            st.markdown(f"**Available Outputs:** {len(unique_outputs)} unique values")
            
            # Show first few outputs as examples
            sample_outputs = unique_outputs[:10] if len(unique_outputs) > 10 else unique_outputs
            st.markdown(f"**Sample Outputs:** {', '.join(map(str, sample_outputs))}")
            
            if len(unique_outputs) > 10:
                st.markdown(f"... and {len(unique_outputs) - 10} more")
            
            # Option to proceed to traveler report
            if st.button("📊 Generate Traveler Report"):
                # Apply traveler report filtering if configured
                final_df_filtered = final_df.copy()
                
                if use_full_range:
                    # Full Range mode - filter based on Input @ day start time
                    if input_value_at_start is not None:
                        high_limit = input_value_at_start + full_range_value
                        low_limit = input_value_at_start - full_range_value
                        
                        filter_ranges = [{
                            "name": "Full Range",
                            "type": "full",
                            "upper": high_limit,
                            "lower": low_limit
                        }]
                        
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
                        filter_ranges.append({
                            "name": "High 1",
                            "type": "high", 
                            "upper": high1,
                            "lower": high1 - 24
                        })
                    if use_high2 and high2 > 0:
                        filter_ranges.append({
                            "name": "High 2",
                            "type": "high",
                            "upper": high2, 
                            "lower": high2 - 24
                        })
                    if use_low1 and low1 > 0:
                        filter_ranges.append({
                            "name": "Low 1",
                            "type": "low",
                            "upper": low1 + 24,
                            "lower": low1
                        })
                    if use_low2 and low2 > 0:
                        filter_ranges.append({
                            "name": "Low 2", 
                            "type": "low",
                            "upper": low2 + 24,
                            "lower": low2
                        })
                    
                    if filter_ranges:
                        # Filter data to custom ranges
                        mask = pd.Series([False] * len(final_df_filtered))
                        for range_info in filter_ranges:
                            range_mask = (final_df_filtered['Output'] >= range_info['lower']) & (final_df_filtered['Output'] <= range_info['upper'])
                            mask = mask | range_mask
                        
                        final_df_filtered = final_df_filtered[mask]
                        
                        # Add Range and Zone columns
                        final_df_filtered['Range'] = final_df_filtered['Output'].apply(
                            lambda x: next((r['name'] for r in filter_ranges 
                                          if r['lower'] <= x <= r['upper']), 'Other')
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
                
                # Display results without highlighting (for performance)
                st.markdown("### 📋 Final Traveler Report")
                st.markdown(f"**Report Time:** {report_time.strftime('%d-%b-%y %H:%M')}")
                st.markdown(f"**Day Start Time:** {day_start_hour:02d}:00")
                st.markdown(f"**Total Entries:** {len(final_df_filtered)}")
                
                display_df = final_df_filtered[display_columns]
                st.dataframe(display_df, use_container_width=True)
                
                # Download buttons
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                
                # CSV download (exclude datetime column)
                csv_data = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Final Traveler Report (CSV)",
                    data=csv_data,
                    file_name=f"final_traveler_report_{timestamp}.csv",
                    mime="text/csv"
                )
                
                # Excel download with highlighting (exclude datetime column)
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    # Apply highlighting for Excel export
                    if use_custom_ranges:
                        styled_df = highlight_custom_traveler_report(display_df)
                    else:
                        styled_df = highlight_traveler_report(display_df)
                    
                    # Convert styled DataFrame to Excel
                    styled_df.to_excel(writer, sheet_name='Traveler Report', index=False)
                    
                    workbook = writer.book
                    worksheet = writer.sheets['Traveler Report']
                    
                    # Add header formatting
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'top',
                        'fg_color': '#D7E4BC',
                        'border': 1
                    })
                    
                    for col_num, value in enumerate(display_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                
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
                        help="Maximum distance between outputs to group them together"
                    )
                    
                    if st.button("🔍 Run Model G Detection", key="model_g_processed_btn"):
                        with st.spinner("Running Model G detection..."):
                            run_model_g_detection(final_df_filtered, proximity_threshold)
                
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
        else:
            st.error("Output column not found in processed data")

else:
    st.info("Please upload small feed, big feed, and measurement files to begin processing.")
