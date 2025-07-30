# v24 - UI redesign to show Full Range or Custom Range. 7.30.25


import streamlit as st
import pandas as pd
import datetime as dt
import io
from pandas import ExcelWriter

from shared.shared import clean_timestamp, process_feed, get_input_value, highlight_traveler_report, get_input_at_time, get_input_at_day_start, highlight_custom_traveler_report
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
st.header("🧬 Data Feed Processor + Model A/B/C Detector. v24")

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
day_start_choice = st.radio("Select Day Start Time", ["17:00", "18:00"])
day_start_hour = int(day_start_choice.split(":")[0])
scope_type = st.radio("Scope by", ["Rows", "Days"])
scope_value = st.number_input(f"Enter number of {scope_type.lower()}", min_value=1, value=10)


# 🎯 Traveler Report Settings (Global Configuration)
st.markdown("---")
st.markdown("### 🎯 Traveler Report Settings")
# Create side-by-side layout for Full Range and Custom Ranges
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("**Full Range**")
    use_full_range = st.checkbox("Full Range", value=True, key="global_full_range_check")
    full_range_value = st.number_input("Range (+/-)", value=1100.0, format="%.1f", key="global_full_range_input")
    
with col2:
    st.markdown("**Custom Ranges**")
    use_custom_ranges = st.checkbox("Custom Ranges", value=False, key="global_custom_ranges_check")
    
    # Create 4-column layout for High1, High2, Low1, Low2
    cols = st.columns(4)
    with cols[0]:
        st.markdown("**High 1**")
        high1 = st.number_input("", value=500.0, format="%.1f", key="global_high1", label_visibility="collapsed")
    with cols[1]:
        st.markdown("**High 2**")
        high2 = st.number_input("", value=480.0, format="%.1f", key="global_high2", label_visibility="collapsed")
    with cols[2]:
        st.markdown("**Low 1**")
        low1 = st.number_input("", value=150.0, format="%.1f", key="global_low1", label_visibility="collapsed")
    with cols[3]:
        st.markdown("**Low 2**")
        low2 = st.number_input("", value=250.0, format="%.1f", key="global_low2", label_visibility="collapsed")
st.markdown("---")

# 📥 Optional bypass: Upload a Final Traveler Report
st.markdown("### 📥 Optional: Upload Final Traveler Report (bypass feed upload)")
final_report_file = st.file_uploader("Upload Final Traveler Report", type="csv", key="final_report_bypass")

if final_report_file:
    try:
        final_df = pd.read_csv(final_report_file)
        final_df.columns = final_df.columns.str.strip()
        final_df["Arrival"] = pd.to_datetime(final_df["Arrival"], errors="coerce")

        # ✅ Set report_time using raw Arrival
        if report_mode == "Most Current":
            report_time = final_df["Arrival"].max()

        # 🔧 Create properly formatted display version with new column structure
        display_df = final_df.copy()
        
        # Format Arrival column: ddd yy-mm-dd hh:mm
        display_df["Arrival"] = display_df["Arrival"].dt.strftime("%a %y-%m-%d %H:%M")
        
        # Ensure column order: Feed, Arrival, Day, Origin, M Name, M #, R #, Tag, Family, Input @ 18:00, Diff @ 18:00, Input @ Arrival, Diff @ Arrival, Input @ Report, Diff @ Report, Output
        ordered_columns = [
            "Feed", "Arrival", "Day", "Origin", "M Name", "M #", "R #", "Tag", "Family",
            "Input @ 18:00", "Diff @ 18:00", "Input @ Arrival", "Diff @ Arrival", 
            "Input @ Report", "Diff @ Report", "Output"
        ]
        
        # Only include columns that exist in the dataframe
        display_columns = [col for col in ordered_columns if col in display_df.columns]
        display_df = display_df[display_columns]

        st.success(f"✅ Using Final Traveler Report with {len(final_df)} rows. Report time set to: {report_time.strftime('%d-%b-%y %H:%M')}")

        # 📊 Display with properly formatted Arrival column
        st.subheader("📊 Final Traveler Report (Bypass Mode)")
        st.dataframe(display_df)

        # 📥 Excel download with highlighting
        timestamp_str = report_time.strftime("%y-%m-%d_%H-%M")
        filename = f"origin_report_{timestamp_str}.xlsx"

        excel_buffer = io.BytesIO()
        with ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            styled_excel = highlight_traveler_report(display_df)
            styled_excel.to_excel(writer, index=False, sheet_name="Traveler Report")

        st.download_button(
            "📥 Download Final Traveler Report (Excel)",
            data=excel_buffer.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Apply Traveler Report Settings configured above
        st.markdown("---")
        st.markdown("### 📊 Applying Traveler Report Settings")        
        
        # Run Traveler Report based on selection
        if use_full_range and not use_custom_ranges:
            # Get Input @ 18:00 as reference
            input_18_ref = None
            if f"Input @ {day_start_hour:02d}:00" in final_df.columns:
                input_18_ref = final_df[f"Input @ {day_start_hour:02d}:00"].iloc[0] if len(final_df) > 0 else None
            
            if input_18_ref is not None:
                # Calculate full range boundaries
                upper_bound = input_18_ref + full_range_value
                lower_bound = input_18_ref - full_range_value
                
                # Filter data within range
                range_df = final_df[(final_df["Output"] >= lower_bound) & (final_df["Output"] <= upper_bound)].copy()
                
                if len(range_df) > 0:
                    # Add Range and Zone columns for Full Range
                    range_df["Range"] = "Full Range"
                    range_df["Zone"] = ""  # No zone categorization for full range
                    
                    # Reorder columns to include Range and Zone after Output
                    ordered_columns = [
                        "Feed", "Arrival", "Day", "Origin", "M Name", "M #", "R #", "Tag", "Family",
                        f"Input @ {day_start_hour:02d}:00", f"Diff @ {day_start_hour:02d}:00", "Input @ Arrival", "Diff @ Arrival", 
                        "Input @ Report", "Diff @ Report", "Output", "Range", "Zone"
                    ]
                    display_columns = [col for col in ordered_columns if col in range_df.columns]
                    range_df = range_df[display_columns]
                    
                    st.subheader("📊 Final Traveler Report")
                    st.dataframe(range_df)
                    
                    # Excel download for Full Range
                    excel_buffer_full = io.BytesIO()
                    with ExcelWriter(excel_buffer_full, engine="xlsxwriter") as writer:
                        styled_excel = highlight_traveler_report(range_df)
                        styled_excel.to_excel(writer, index=False, sheet_name="Final Traveler Report")
                    
                    st.download_button(
                        "📥 Download Final Traveler Report (Excel)",
                        data=excel_buffer_full.getvalue(),
                        file_name=f"final_traveler_report_{timestamp_str}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("No data found within the specified full range.")
            else:
                st.error("Could not determine Input @ 18:00 reference value.")
                
        elif use_custom_ranges and not use_full_range:
            # Custom Ranges logic
            custom_ranges = [
                {"name": "High 1", "upper": high1, "lower": high1 - 24, "type": "high"},
                {"name": "High 2", "upper": high2, "lower": high2 - 24, "type": "high"},
                {"name": "Low 1", "upper": low1 + 24, "lower": low1, "type": "low"},
                {"name": "Low 2", "upper": low2 + 24, "lower": low2, "type": "low"}
            ]
            
            all_custom_data = []
            
            for range_info in custom_ranges:
                # Filter data for this range
                range_df = final_df[
                    (final_df["Output"] >= range_info["lower"]) & 
                    (final_df["Output"] <= range_info["upper"])
                ].copy()
                
                if len(range_df) > 0:
                    # Add Range column
                    range_df["Range"] = range_info["name"]
                    
                    # Calculate Zone based on distance from upper/lower limits
                    def calculate_zone(output, range_info):
                        if range_info["type"] == "high":
                            # For highs, measure distance from upper limit
                            distance = range_info["upper"] - output
                        else:
                            # For lows, measure distance from lower limit  
                            distance = output - range_info["lower"]
                        
                        if distance <= 6:
                            return "0 to 6"
                        elif distance <= 12:
                            return "6 to 12"
                        elif distance <= 18:
                            return "12 to 18"
                        else:
                            return "18 to 24"
                    
                    range_df["Zone"] = range_df["Output"].apply(lambda x: calculate_zone(x, range_info))
                    all_custom_data.append(range_df)
            
            if all_custom_data:
                # Combine all custom range data
                combined_df = pd.concat(all_custom_data, ignore_index=True)
                
                # Reorder columns
                ordered_columns = [
                    "Feed", "Arrival", "Day", "Origin", "M Name", "M #", "R #", "Tag", "Family",
                    f"Input @ {day_start_hour:02d}:00", f"Diff @ {day_start_hour:02d}:00", "Input @ Arrival", "Diff @ Arrival", 
                    "Input @ Report", "Diff @ Report", "Output", "Range", "Zone"
                ]
                display_columns = [col for col in ordered_columns if col in combined_df.columns]
                combined_df = combined_df[display_columns]
                
                st.subheader("📊 Custom Traveler Report")
                
                # Apply zone highlighting for screen display
                styled_df = highlight_custom_traveler_report(combined_df, show_highlighting=True)
                st.dataframe(styled_df)
                
                # Excel download for Custom Ranges with highlighting
                excel_buffer_custom = io.BytesIO()
                with ExcelWriter(excel_buffer_custom, engine="xlsxwriter") as writer:
                    styled_excel = highlight_custom_traveler_report(combined_df, show_highlighting=True)
                    styled_excel.to_excel(writer, index=False, sheet_name="Custom Traveler Report")
                
                st.download_button(
                    "📥 Download Custom Traveler Report (Excel)",
                    data=excel_buffer_custom.getvalue(),
                    file_name=f"custom_traveler_report_{timestamp_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No data found within any of the specified custom ranges.")
        else:
            st.warning("Please select either Full Range or Custom Ranges (not both).")      

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

# Process feeds
if small_feed_file and big_feed_file and measurement_file:
    try:
        small_df = pd.read_csv(small_feed_file)
        big_df = pd.read_csv(big_feed_file)

        for df in [small_df, big_df]:
            df.columns = df.columns.str.strip().str.lower()
            df["time"] = df["time"].apply(clean_timestamp)

        xls = pd.ExcelFile(measurement_file)
        sheet_choice = st.selectbox("Select measurement tab", xls.sheet_names)
        measurements = pd.read_excel(measurement_file, sheet_name=sheet_choice)
        measurements.columns = measurements.columns.str.strip().str.lower()

        if report_mode == "Most Current":
            report_time = max(small_df["time"].max(), big_df["time"].max())

        if filter_future_data and report_time:
            small_df = small_df[small_df["time"] <= report_time]
            big_df = big_df[big_df["time"] <= report_time]
            st.info(f"🔒 Data filtered up to: {report_time}")

        st.success(f"📅 Using report time: {report_time.strftime('%d-%b-%y %H:%M')}")

        input_value_18 = get_input_at_day_start(small_df, report_time, day_start_hour) or get_input_at_day_start(big_df, report_time, day_start_hour)
        if input_value_18 is None:
            st.error(f"⚠️ Could not determine Report Time or Input Value @ {day_start_hour:02d}:00.")
        else:
            st.success(f"📌 Input value @ {day_start_hour:02d}:00: {input_value_18:.3f}")

            results = []
            # Pass small_df for input calculations at different times
            sm_results = process_feed(small_df, "Sm", report_time, scope_type, scope_value, day_start_hour, measurements, input_value_18, small_df)
            bg_results = process_feed(big_df, "Bg", report_time, scope_type, scope_value, day_start_hour, measurements, input_value_18, small_df)
            
            st.info(f"📊 Small feed generated {len(sm_results)} rows, Big feed generated {len(bg_results)} rows")
            
            results += sm_results
            results += bg_results

            final_df = pd.DataFrame(results)
            final_df.sort_values(by=["Output", "Arrival"], ascending=[False, True], inplace=True)
            final_df["Arrival"] = pd.to_datetime(final_df["Arrival"], errors="coerce")
            
            # 🔧 Create properly formatted display version
            display_df = final_df.copy()
            
            # Format Arrival column: ddd yy-mm-dd hh:mm (e.g., Sun 25-07-27 18:00)
            display_df["Arrival"] = display_df["Arrival"].dt.strftime("%a %y-%m-%d %H:%M")
            
            # Ensure proper column order with dynamic start hour
            ordered_columns = [
                "Feed", "Arrival", "Day", "Origin", "M Name", "M #", "R #", "Tag", "Family",
                f"Input @ {day_start_hour:02d}:00", f"Diff @ {day_start_hour:02d}:00", "Input @ Arrival", "Diff @ Arrival", 
                "Input @ Report", "Diff @ Report", "Output"
            ]
            
            # Only include columns that exist in the dataframe
            display_columns = [col for col in ordered_columns if col in display_df.columns]
            display_df = display_df[display_columns]
            
            st.subheader("📊 Final Traveler Report")
            st.dataframe(display_df)
            
            # 🧾 Excel export setup
            timestamp_str = report_time.strftime("%y-%m-%d_%H-%M")
            filename = f"origin_report_{timestamp_str}.xlsx"
            
            excel_buffer = io.BytesIO()
            with ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                styled_excel = highlight_traveler_report(display_df)
                styled_excel.to_excel(writer, index=False, sheet_name="Traveler Report")
            
            # 📥 Download button
            st.download_button(
                "📥 Download Final Traveler Report (Excel)",
                data=excel_buffer.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

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
        st.error(f"❌ Error processing feeds: {e}")
