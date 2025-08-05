# v24 - UI redesign to show Full Range or Custom Range. 7.30.25
# v25 - v24 did not produce the custom ranges first.  Reverting back to raw data production first, then custom printing afterwards. 7.31.25
# v26a - First effort to find sequences in a range of outputs is too slow.  Moving to a hybrid solution with Octave in the next iteration.  

import streamlit as st
import pandas as pd
import datetime as dt
import io
from pandas import ExcelWriter

# Configure pandas to handle large datasets
pd.set_option("styler.render.max_elements", 2000000)

from a_helpers import clean_timestamp, process_feed, get_input_value, highlight_traveler_report, get_input_at_time, get_input_at_day_start, highlight_custom_traveler_report
from models.models_g import run_model_g_detection
from models.models_a_today import run_a_model_detection_today
from models.mod_b_05pg1 import run_b_model_detection
from models.mod_c_04gpr3 import run_c_model_detection
from models.mod_x_03g import run_x_model_detection
from models.simple_mega_report2 import run_simple_single_line_analysis

# For any missing model imports, we'll create simple placeholder functions
# def run_b_model_detection(df):
#     st.warning("Model B detection not available")
# def run_x_model_detection(df):
#     st.warning("Model X detection not available")
    
# 🔌 Streamlit interface (UI + orchestration)

st.set_page_config(layout="wide")
st.header("🧬 Data Feed Processor + Model A/B/C Detector. v26c1")

# 📤 Uploads
small_feed_file = st.file_uploader("Upload small feed", type="csv")
big_feed_file = st.file_uploader("Upload big feed", type="csv")
measurement_file = st.file_uploader("Upload measurement file", type=["xlsx", "xls"])

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
    try:
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
        
        # Model G Detection with Hybrid Octave System
        if run_g_models:
            st.markdown("---")
            st.markdown("### 🟢 Model G Detection System")
            
            # Computation method selection
            computation_method = st.radio(
                "Computation Method:",
                ["Python (Standard)", "Octave Hybrid (High Performance)"],
                help="Octave hybrid method uses optimized mathematical operations for faster processing of large datasets"
            )
            
            # Add display controls for both methods
            st.markdown("### 🎛️ Display Controls")
            st.markdown("*Select which results to display after detection runs:*")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                show_today = st.checkbox("Today", value=True, help="Show G.05.o1[0] sequences", key="app_today")
            with col2:
                show_recent = st.checkbox("Recent", value=True, help="Show Other Days [-1] to [-5]", key="app_recent")
            with col3:
                show_other_days = st.checkbox("Other Days", value=True, help="Show Other Days [-6] and beyond", key="app_other")
            with col4:
                show_rejected = st.checkbox("Rejected Groups", value=False, help="Show rejected groups for debugging", key="app_rejected")
            
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
                if computation_method == "Octave Hybrid (High Performance)":
                    try:
                        import hybrid_model_g
                        
                        if not hybrid_model_g.test_octave_installation():
                            st.error("Octave is not properly installed. Falling back to Python method.")
                            with st.spinner("Running Model G detection (Python)..."):
                                run_model_g_detection(final_df, proximity_threshold)
                        else:
                            # Show dataset size and method selection
                            if len(final_df) > 5000:
                                method_desc = "Advanced Octave (Large Dataset Optimized)"
                            else:
                                method_desc = "Standard Octave (High Performance)"
                                
                            with st.spinner(f"Running Model G detection ({method_desc})..."):
                                results = hybrid_model_g.run_octave_model_g(final_df, proximity_threshold)
                                if results:
                                    hybrid_model_g.display_hybrid_results(results, proximity_threshold, final_df)
                                else:
                                    st.info("Octave processing complete. Checking for result files...")
                                    # The large dataset implementation saves results to files
                                    import os
                                    if os.path.exists('octave_today_groups.txt') or os.path.exists('octave_other_groups.txt'):
                                        st.success("Large dataset processing completed successfully!")
                                        try:
                                            hybrid_model_g.display_large_dataset_results(proximity_threshold)
                                        except:
                                            st.info("Results saved to files. Processing completed.")
                                    else:
                                        st.warning("No results returned from Octave. Using Python method as backup.")
                                        run_model_g_detection(final_df, proximity_threshold)
                                    
                    except ImportError as e:
                        st.error(f"Could not import hybrid module: {e}")
                        run_model_g_detection(final_df, proximity_threshold)
                else:
                    # Standard Python method
                    with st.spinner("Running Model G detection (Python)..."):
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
            
    except Exception as e:
        st.error(f"❌ Error processing Final Traveler Report: {e}")
        import traceback
        st.text(traceback.format_exc())

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
        
        st.markdown("#### Processing Big Feed...")
        big_data = process_feed(big_df, "Big", report_time, scope_type, scope_value,
                              day_start_hour, measurements_df, input_value_at_start, small_df)
        
        # Combine processed data
        all_data = small_data + big_data
        if not all_data:
            st.warning("No data generated from feed processing")
            st.stop()
        
        # Create final DataFrame
        final_df = pd.DataFrame(all_data)
        
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
        
        # Display results
        st.markdown("### 📋 Final Traveler Report")
        st.markdown(f"**Report Time:** {report_time.strftime('%d-%b-%y %H:%M')}")
        st.markdown(f"**Day Start Time:** {day_start_hour:02d}:00")
        st.markdown(f"**Total Entries:** {len(final_df_filtered)}")
        
        # Display results without highlighting (for performance)
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
        
        # Now run model detections on the processed data
        if run_g_models:
            st.markdown("---")
            st.markdown("### 🟢 Model G Detection System")
            
            # Computation method selection
            computation_method = st.radio(
                "Computation Method:",
                ["Python (Standard)", "Octave Hybrid (High Performance)"],
                help="Octave hybrid method uses optimized mathematical operations for faster processing of large datasets"
            )
            
            # Add display controls for both methods
            st.markdown("### 🎛️ Display Controls")
            st.markdown("*Select which results to display after detection runs:*")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                show_today = st.checkbox("Today", value=True, help="Show G.05.o1[0] sequences", key="feed_today")
            with col2:
                show_recent = st.checkbox("Recent", value=True, help="Show Other Days [-1] to [-5]", key="feed_recent")
            with col3:
                show_other_days = st.checkbox("Other Days", value=True, help="Show Other Days [-6] and beyond", key="feed_other")
            with col4:
                show_rejected = st.checkbox("Rejected Groups", value=False, help="Show rejected groups for debugging", key="feed_rejected")
            
            # Proximity threshold setting
            proximity_threshold = st.slider(
                "Proximity Threshold", 
                min_value=0.01, 
                max_value=1.0, 
                value=0.10, 
                step=0.01,
                help="Maximum distance between outputs to group them together"
            )
            
            if st.button("🔍 Run Model G Detection", key="feed_model_g_btn"):
                if computation_method == "Octave Hybrid (High Performance)":
                    try:
                        import hybrid_model_g
                        
                        if not hybrid_model_g.test_octave_installation():
                            st.error("Octave is not properly installed. Falling back to Python method.")
                            with st.spinner("Running Model G detection (Python)..."):
                                run_model_g_detection(final_df_filtered, proximity_threshold)
                        else:
                            with st.spinner("Running Model G detection (Octave Hybrid)..."):
                                results = hybrid_model_g.run_octave_model_g(final_df_filtered, proximity_threshold)
                                if results:
                                    hybrid_model_g.display_hybrid_results(results, proximity_threshold, final_df_filtered)
                                else:
                                    st.warning("No results returned from Octave. Using Python method as backup.")
                                    run_model_g_detection(final_df_filtered, proximity_threshold)
                                    
                    except ImportError as e:
                        st.error(f"Could not import hybrid module: {e}")
                        run_model_g_detection(final_df_filtered, proximity_threshold)
                else:
                    # Standard Python method
                    with st.spinner("Running Model G detection (Python)..."):
                        run_model_g_detection(final_df_filtered, proximity_threshold)
        
        # Other model detections
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
        st.error(f"❌ Error processing feeds: {e}")
        import traceback
        st.text(traceback.format_exc())
else:
    st.info("👆 Upload files above to begin analysis, or use the 'Upload Final Traveler Report' option to test the hybrid Model G system directly.")
