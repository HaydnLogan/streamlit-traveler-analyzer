import streamlit as st
import pandas as pd
import datetime as dt

from shared.shared import clean_timestamp, process_feed
from models.models_a import run_a_model_detection
from models.TestB_Mod_B_01 import run_b_model_detection
from models.TestB_Mod_C_01 import run_c_model_detection

# *** This is Tester B  ***

# 🔌 Streamlit UI
st.set_page_config(layout="wide")
st.header("🧬 Data Feed Processor + Model A/B/C Detector. v08gp")
# CavAir_Mod B_04p3 and mod C_2gp

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
        st.success(f"✅ Using report time: {report_time.strftime('%d-%b-%y %H:%M')}")

        # 🧮 Get input value from 'open' column in small_df
        input_value = None
        if "open" in small_df.columns:
            if report_mode == "Most Current":
                input_value = small_df["open"].dropna().iloc[-1]
            else:
                valid = small_df[small_df["time"] <= report_time]
                if not valid.empty:
                    input_value = valid["open"].iloc[-1]

        if report_time is None or input_value is None:
            st.error("⚠️ Could not determine Report Time or Input Value.")
        else:
            st.success(f"✅ Input value: {input_value:.3f}")

            # Finalize feed parsing
            results = []
            results += process_feed(small_df, "Sm", report_time, scope_type, scope_value, day_start_hour, measurements, input_value)
            results += process_feed(big_df, "Bg", report_time, scope_type, scope_value, day_start_hour, measurements, input_value)

            final_df = pd.DataFrame(results)
            final_df.sort_values(by=["Output", "Arrival"], ascending=[False, True], inplace=True)
            final_df["Arrival"] = pd.to_datetime(final_df["Arrival"], errors="coerce")  # keeps as datetime for models
            final_df["Arrival Display"] = final_df["Arrival"].dt.strftime("%#d-%b-%y %H:%M")  # human-readable version

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
