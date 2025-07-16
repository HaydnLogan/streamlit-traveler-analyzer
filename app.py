import streamlit as st
import pandas as pd
import datetime as dt

from shared import clean_timestamp, get_input_value, process_feed
# from models import run_a_model_detection, run_b_model_detection, run_c_model_detection

# 🔌 Streamlit interface (UI + orchestration)

st.set_page_config(layout="wide")
st.header("🧬 Data Feed Processor + Model A/B/C Detector")

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
    report_time = None  # will be computed later

# ✅ Add option to filter out future data
filter_future_data = st.checkbox("Restrict analysis to Report Time or earlier only", value=True)

# ⚙️ Analysis parameters
day_start_choice = st.radio("Select Day Start Time", ["17:00", "18:00"])
day_start_hour = int(day_start_choice.split(":")[0])
scope_type = st.radio("Scope by", ["Rows", "Days"])
scope_value = st.number_input(f"Enter number of {scope_type.lower()}", min_value=1, value=10)

# 🧠 Process feeds if ready
if small_feed_file and big_feed_file and measurement_file:
    try:
        # 📥 Load and clean feeds
        small_df = pd.read_csv(small_feed_file)
        big_df = pd.read_csv(big_feed_file)
        small_df.columns = small_df.columns.str.strip().str.lower()
        big_df.columns = big_df.columns.str.strip().str.lower()
        small_df["time"] = small_df["time"].apply(clean_timestamp)
        big_df["time"] = big_df["time"].apply(clean_timestamp)

        # 📈 Measurements
        xls = pd.ExcelFile(measurement_file)
        sheet_choice = st.selectbox("Select measurement tab", xls.sheet_names)
        measurements = pd.read_excel(measurement_file, sheet_name=sheet_choice)
        measurements.columns = measurements.columns.str.strip().str.lower()

        # ⏱️ Set report time (if not already chosen)
        if report_mode == "Most Current":
            report_time = max(small_df["time"].max(), big_df["time"].max())

        # ✅ Filter feeds to exclude data after report_time
        if filter_future_data and report_time:
            small_df = small_df[small_df["time"] <= report_time]
            big_df = big_df[big_df["time"] <= report_time]
            st.info(f"🔒 Data filtered up to: {report_time}")
        elif report_time:
            st.warning("⚠️ Future data is included.")

        st.success(f"✅ Using report time: {report_time.strftime('%d-%b-%y %H:%M')}")

        # 🧮 Input value
        input_value = get_input_value(small_df, report_time)
        if input_value is None:
            input_value = get_input_value(big_df, report_time)

        if report_time is None or input_value is None:
            st.error("⚠️ Could not determine Report Time or Input Value.")
        else:
            st.success(f"✅ Input value: {input_value:.3f}")

            results = []
            results += process_feed(small_df, "Sm", report_time, scope_type, scope_value, day_start_hour, measurements, input_value)
            results += process_feed(big_df, "Bg", report_time, scope_type, scope_value, day_start_hour, measurements, input_value)

            final_df = pd.DataFrame(results)
            final_df.sort_values(by=["Output", "Arrival"], ascending=[False, True], inplace=True)
            final_df["Arrival"] = pd.to_datetime(final_df["Arrival"]).dt.strftime("%#d-%b-%y %H:%M")

            st.subheader("📊 Final Traveler Report")
            st.dataframe(final_df)

            # 📥 Download
            timestamp_str = report_time.strftime("%y-%m-%d_%H-%M")
            filename = f"origin_report_{timestamp_str}.csv"
            st.download_button("📥 Download Report CSV", data=final_df.to_csv(index=False).encode(), file_name=filename, mime="text/csv")

            # 📣 Model detection placeholder
            # st.markdown("---")
            # if run_models:
            #     run_a_model_detection(final_df)

    except Exception as e:
        st.error(f"❌ Error processing feeds: {e}")
