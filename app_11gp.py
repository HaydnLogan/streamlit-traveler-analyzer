import streamlit as st
import pandas as pd
import datetime as dt

from shared.shared import clean_timestamp, process_feed, get_input_value
from models.models_a import run_a_model_detection
from models.mod_b_04p3 import run_b_model_detection
from models.mod_c_02gp import run_c_model_detection
# to facilitate CavAir, Mod B.v04p3

# 🔌 Streamlit interface (UI + orchestration)

st.set_page_config(layout="wide")
st.header("🧬 Data Feed Processor + Model A/B/C Detector. v10gp")
# CavAir_Mod B_04p3 and mod C_2gp
# Table format for Custom Traveler Report

# 📤 Upload feeds
small_feed_file = st.file_uploader("Upload small feed", type="csv")
big_feed_file = st.file_uploader("Upload big feed", type="csv")
measurement_file = st.file_uploader("Upload measurement file", type=["xlsx", "xls"])

# 📅 Report time settings
report_mode = st.radio("Select Report Time & Date", ["Most Current", "Choose a time"], key="report_mode_radio")
if report_mode == "Choose a time":
    selected_date = st.date_input("Select Report Date", value=dt.date.today(), key="report_date_picker")
    selected_time = st.time_input("Select Report Time", value=dt.time(18, 0), key="report_time_picker")
    report_time = dt.datetime.combine(selected_date, selected_time)
else:
    report_time = None  # will be determined after loading feeds

# ✅ Model toggles
run_a_models = st.sidebar.checkbox("Run Model A Detection")
run_b_models = st.sidebar.checkbox("Run Model B Detection")
run_c_models = st.sidebar.checkbox("Run Model C Detection")
st.sidebar.subheader("🔧 C Model Filters")
run_c01 = st.sidebar.checkbox("C Flips", value=True)
run_c02 = st.sidebar.checkbox("C Opposites", value=True)
run_c04 = st.sidebar.checkbox("C Ascending", value=True)


# ✅ Option to restrict data
filter_future_data = st.checkbox("Restrict analysis to Report Time or earlier only", value=True)

# ⚙️ Analysis parameters
day_start_choice = st.radio("Select Day Start Time", ["17:00", "18:00"])
day_start_hour = int(day_start_choice.split(":")[0])
scope_type = st.radio("Scope by", ["Rows", "Days"])
scope_value = st.number_input(f"Enter number of {scope_type.lower()}", min_value=1, value=10)

# 📥 Optional bypass: Final Traveler Report
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
        st.dataframe(final_df[["Feed", "Arrival Display", "Origin", "M Name", "M #", "Output"]])

        # 📥 Download (optional)
        timestamp_str = report_time.strftime("%y-%m-%d_%H-%M")
        filename = f"origin_report_{timestamp_str}.csv"
        st.download_button("📥 Download Report CSV", data=final_df.to_csv(index=False).encode(), file_name=filename, mime="text/csv")

        # 🔍 Run model detectors
        if run_a_models:
            st.markdown("---")
            run_a_model_detection(final_df)

        if run_b_models:
            st.markdown("---")
            run_b_model_detection(final_df)

        if run_c_models:
            st.markdown("---")
            run_c_model_detection(final_df, run_c01=run_c01, run_c02=run_c02, run_c04=run_c04)

    except Exception as e:
        st.error(f"❌ Error processing Final Traveler Report: {e}")


# 🧠 Process feeds if ready
if small_feed_file and big_feed_file and measurement_file:
    try:
        # Load and clean feeds
        small_df = pd.read_csv(small_feed_file)
        big_df = pd.read_csv(big_feed_file)

        small_df.columns = small_df.columns.str.strip().str.lower()
        big_df.columns = big_df.columns.str.strip().str.lower()

        small_df["time"] = small_df["time"].apply(clean_timestamp)
        big_df["time"] = big_df["time"].apply(clean_timestamp)

        small_df["time"] = pd.to_datetime(small_df["time"], errors="coerce").dt.tz_localize(None)
        big_df["time"] = pd.to_datetime(big_df["time"], errors="coerce").dt.tz_localize(None)

        # 📈 Measurements (even if not currently used for input)
        xls = pd.ExcelFile(measurement_file)
        sheet_choice = st.selectbox("Select measurement tab", xls.sheet_names)
        measurements = pd.read_excel(measurement_file, sheet_name=sheet_choice)
        measurements.columns = measurements.columns.str.strip().str.lower()

        # Set report_time if "Most Current"
        if report_mode == "Most Current":
            report_time = max(small_df["time"].max(), big_df["time"].max())

        # ✅ Filter future rows
        if filter_future_data and report_time:
            small_df = small_df[small_df["time"] <= report_time]
            big_df = big_df[big_df["time"] <= report_time]
            st.info(f"🔒 Data filtered up to: {report_time}")
        elif report_time:
            st.warning("⚠️ Future data is included.")

        # Display report time
        st.success(f"📅 Using report time: {report_time.strftime('%d-%b-%y %H:%M')}")

        # 🧮 Get input value from 'open' column in small_df
        input_row = None
        input_value = get_input_value(small_df, report_time)
        if input_value is None:
            input_value = get_input_value(big_df, report_time)
            
      
        # Final check and print
        if report_time is None or input_value is None:
            st.error("⚠️ Could not determine Report Time or Input Value.")
        else:
            st.success(f"📌 Input value: {input_value:.3f}")
    

            # Finalize feed parsing
            results = []
            results += process_feed(small_df, "Sm", report_time, scope_type, scope_value, day_start_hour, measurements, input_value)
            results += process_feed(big_df, "Bg", report_time, scope_type, scope_value, day_start_hour, measurements, input_value)

            final_df = pd.DataFrame(results)
            final_df.sort_values(by=["Output", "Arrival"], ascending=[False, True], inplace=True)
            final_df["Arrival"] = pd.to_datetime(final_df["Arrival"], errors="coerce")  # keeps as datetime for models
            final_df["Arrival Display"] = final_df["Arrival"].dt.strftime("%#d-%b-%y %H:%M")  # human-readable version

            # 📐 Custom Traveler Report Table
            st.markdown("### 🎯 Run Custom Traveler Report")
            st.markdown("Use the table below to define up to 4 custom output ranges. Check the box to enable any of the ranges.")
            
            custom_range_data = {
                "Enable": [False, False, False, False],
                "Largest Output": [None, None, None, None],
                "Smallest Output": [None, None, None, None]
            }
            
            # Create editable table-like layout
            custom_range_df = pd.DataFrame(custom_range_data, index=["Range 1", "Range 2", "Range 3", "Range 4"])
            
            # Manual form layout
            custom_range_results = []
            for i, range_name in enumerate(custom_range_df.index):
                cols = st.columns([1, 2, 2])
                enabled = cols[0].checkbox(f"{range_name}", key=f"enable_{i}")
                largest = cols[1].number_input("Largest Output", key=f"max_{i}", value=0.0, format="%.3f")
                smallest = cols[2].number_input("Smallest Output", key=f"min_{i}", value=0.0, format="%.3f")
            
                custom_range_results.append({
                    "enabled": enabled,
                    "largest": largest,
                    "smallest": smallest,
                    "label": range_name
                })
            
            st.markdown("---")
            if st.button("▶️ Run Custom Traveler Report"):
                custom_outputs = []
                for r in custom_range_results:
                    if r["enabled"] and r["largest"] > r["smallest"]:
                        filtered = final_df[
                            (final_df["Output"] <= r["largest"]) & (final_df["Output"] >= r["smallest"])
                        ].copy()
                        filtered["Range Label"] = r["label"]
                        custom_outputs.append(filtered)
            
                if not custom_outputs:
                    st.warning("No custom ranges matched.")
                else:
                    all_custom_df = pd.concat(custom_outputs, ignore_index=True)
                    for r in custom_outputs:
                        label = r["Range Label"].iloc[0]
                        st.subheader(f"📌 {label}")
                        st.dataframe(r.drop(columns=["Range Label"]))
            
                    csv_data = all_custom_df.to_csv(index=False).encode()
                    st.download_button("📥 Download Custom Traveler Report CSV", data=csv_data, file_name="custom_traveler_report.csv", mime="text/csv")

          
      
            # run_custom = st.checkbox("Enable Custom Traveler Report")
            
            # custom_results = []
            # if run_custom:
            #     st.markdown("Select up to 4 output ranges:")
            
            #     custom_ranges = []
            #     for i in range(1, 5):
            #         enabled = st.checkbox(f"Enable Range {i}", key=f"range_{i}")
            #         if enabled:
            #             largest = st.number_input(f"Range {i}: Largest Output", value=25000.0, key=f"max_{i}")
            #             smallest = st.number_input(f"Range {i}: Smallest Output", value=20000.0, key=f"min_{i}")
            #             custom_ranges.append((i, smallest, largest))
            
            #     if st.button("▶️ Run Custom Traveler Report"):
            #         if not custom_ranges:
            #             st.warning("Please enable and define at least one range.")
            #         else:
            #             st.markdown("---")
            #             for idx, min_out, max_out in custom_ranges:
            #                 filtered = final_df[(final_df["Output"] >= min_out) & (final_df["Output"] <= max_out)]
            #                 st.subheader(f"📈 Range {idx}: {min_out:.3f} to {max_out:.3f}")
            #                 st.dataframe(filtered[["Feed", "Arrival Display", "Origin", "M Name", "M #", "Output"]])
            #                 filtered["Custom Range"] = f"Range {idx}"
            #                 custom_results.append(filtered)
            
            #             # 📥 Download all ranges
            #             if custom_results:
            #                 combined = pd.concat(custom_results)
            #                 st.download_button(
            #                     label="📥 Download Custom Traveler Report CSV",
            #                     data=combined.to_csv(index=False).encode(),
            #                     file_name="custom_traveler_report.csv",
            #                     mime="text/csv"
            #                 )

            
            st.subheader("📊 Final Traveler Report")
            st.dataframe(final_df[["Feed", "Arrival Display", "Origin", "M Name", "M #", "Output"]])


            # 📥 Download
            timestamp_str = report_time.strftime("%y-%m-%d_%H-%M")
            filename = f"origin_report_{timestamp_str}.csv"
            st.download_button("📥 Download Report CSV", data=final_df.to_csv(index=False).encode(), file_name=filename, mime="text/csv")

            # 🔍 Run model detectors
            if run_a_models:
                st.markdown("---")
                run_a_model_detection(final_df)

            if run_b_models:
                st.markdown("---")
                run_b_model_detection(final_df)

            if run_c_models:
                st.markdown("---")
                run_c_model_detection(final_df, run_c01=run_c01, run_c02=run_c02, run_c04=run_c04)


    except Exception as e:
        st.error(f"❌ Error processing feeds: {e}")
