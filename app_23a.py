# v22 - Single Output line development. 7.27.25


import streamlit as st
import pandas as pd
import datetime as dt
import io
from pandas import ExcelWriter

from shared.shared import clean_timestamp, process_feed, get_input_value, highlight_traveler_report, get_input_at_time, get_input_at_day_start
from models.models_a_today import run_a_model_detection_today
from models.mod_b_05pg1 import run_b_model_detection
from models.mod_c_04gpr3 import run_c_model_detection
from models.mod_x_03g import run_x_model_detection
from models.simple_mega_report2 import run_simple_single_line_analysis

# For any missing model imports, we'll create simple placeholder functions
def run_b_model_detection(df):
    st.warning("Model B detection not available")
def run_x_model_detection(df):
    st.warning("Model X detection not available")
    
# 🔌 Streamlit interface (UI + orchestration)

st.set_page_config(layout="wide")
st.header("🧬 Data Feed Processor + Model A/B/C Detector. v23a single output line report")

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

            # Custom Traveler Report
            st.markdown("### 🎯 Run Custom Traveler Report")
            st.markdown("Use the table below to define up to 4 custom output ranges.")

            custom_range_results = []
            for i in range(4):
                cols = st.columns([1, 2, 2])
                enabled = cols[0].checkbox(f"Range {i+1}", key=f"enable_{i}")
                largest = cols[1].number_input("Largest Output", key=f"max_{i}", value=0.0, format="%.3f")
                smallest = cols[2].number_input("Smallest Output", key=f"min_{i}", value=0.0, format="%.3f")
                custom_range_results.append({"enabled": enabled, "largest": largest, "smallest": smallest, "label": f"Range {i+1}"})

            if st.button("▶️ Run Custom Traveler Report"):
                custom_outputs = {}
                for r in custom_range_results:
                    if r["enabled"] and r["largest"] > r["smallest"]:
                        filtered = display_df[(display_df["Output"] <= r["largest"]) & (display_df["Output"] >= r["smallest"])].copy()
                        filtered["Range Label"] = r["label"]
                        custom_outputs[r["label"]] = filtered

                if not custom_outputs:
                    st.warning("No custom ranges matched.")
                else:
                    all_custom_df = pd.concat(custom_outputs.values(), ignore_index=True)
                    for label, df in custom_outputs.items():
                        st.subheader(f"📌 {label}")
                        st.dataframe(df.drop(columns=["Range Label"]))

                    excel_buffer_custom = io.BytesIO()
                    with ExcelWriter(excel_buffer_custom, engine="xlsxwriter") as writer:
                        styled_custom = highlight_traveler_report(all_custom_df)
                        styled_custom.to_excel(writer, index=False, sheet_name="Custom Report")
                        
                    st.download_button("📥 Download Custom Traveler Report Excel", data=excel_buffer_custom.getvalue(), file_name="custom_traveler_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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
