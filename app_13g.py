# v13 - Data Feed Processor with Excel download + Anchor Highlighting in Export Only. GroundTech 7.20.25
# Table format for Custom Traveler Report.
# Model X sandbox added 7.20.25

import streamlit as st
import pandas as pd
import datetime as dt
import io
from pandas import ExcelWriter

from shared.shared import clean_timestamp, process_feed, get_input_value, highlight_anchor_origins
from models.models_a import run_a_model_detection
from models.mod_b_05pg1 import run_b_model_detection
from models.mod_c_03gp import run_c_model_detection
from models.mod_x_02g import run_x_model_detection

# 🔌 Streamlit interface (UI + orchestration)

st.set_page_config(layout="wide")
st.header("🧬 Data Feed Processor + Model A/B/C Detector. v13")

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
        final_df["Arrival Display"] = final_df["Arrival"].dt.strftime("%#d-%b-%y %H:%M")

        if report_mode == "Most Current":
            report_time = final_df["Arrival"].max()

        st.success(f"✅ Using Final Traveler Report with {len(final_df)} rows. Report time set to: {report_time.strftime('%d-%b-%y %H:%M')}")

        st.subheader("📊 Final Traveler Report (Bypass Mode)")
        st.dataframe(final_df)

        # Excel download with highlighting
        timestamp_str = report_time.strftime("%y-%m-%d_%H-%M")
        filename = f"origin_report_{timestamp_str}.xlsx"

        excel_buffer = io.BytesIO()
        with ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            styled_excel = highlight_anchor_origins(final_df)
            styled_excel.to_excel(writer, index=False, sheet_name="Traveler Report")

        st.download_button("📥 Download Report Excel", data=excel_buffer.getvalue(), file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        if run_a_models:
            st.markdown("---")
            run_a_model_detection(final_df)
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
            df["time"] = pd.to_datetime(df["time"].apply(clean_timestamp), errors="coerce").dt.tz_localize(None)

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

        input_value = get_input_value(small_df, report_time) or get_input_value(big_df, report_time)
        if input_value is None:
            st.error("⚠️ Could not determine Report Time or Input Value.")
        else:
            st.success(f"📌 Input value: {input_value:.3f}")

            results = []
            results += process_feed(small_df, "Sm", report_time, scope_type, scope_value, day_start_hour, measurements, input_value)
            results += process_feed(big_df, "Bg", report_time, scope_type, scope_value, day_start_hour, measurements, input_value)

            final_df = pd.DataFrame(results)
            final_df.sort_values(by=["Output", "Arrival"], ascending=[False, True], inplace=True)
            final_df["Arrival"] = pd.to_datetime(final_df["Arrival"], errors="coerce")
            final_df["Arrival Display"] = final_df["Arrival"].dt.strftime("%#d-%b-%y %H:%M")

            st.subheader("📊 Final Traveler Report")
            st.dataframe(final_df)

            timestamp_str = report_time.strftime("%y-%m-%d_%H-%M")
            filename = f"origin_report_{timestamp_str}.xlsx"

            excel_buffer = io.BytesIO()
            with ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                styled_excel = highlight_anchor_origins(final_df)
                styled_excel.to_excel(writer, index=False, sheet_name="Traveler Report")
       
            st.download_button("📥 Download Report Excel", data=excel_buffer.getvalue(), file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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
                        filtered = final_df[(final_df["Output"] <= r["largest"]) & (final_df["Output"] >= r["smallest"])].copy()
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
                        styled_custom = highlight_anchor_origins(all_custom_df)
                        styled_custom.to_excel(writer, index=False, sheet_name="Custom Report")
                        
                    st.download_button("📥 Download Custom Traveler Report Excel", data=excel_buffer_custom.getvalue(), file_name="custom_traveler_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            if run_a_models:
                st.markdown("---")
                run_a_model_detection(final_df)
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
