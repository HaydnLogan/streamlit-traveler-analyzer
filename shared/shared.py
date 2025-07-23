import pandas as pd
import datetime as dt
from dateutil import parser

# Constants: Origin Classifications   🌌🪐💫☄️🌍🏝️🍹⛱️🌞 🌊
EPIC_ORIGINS = {"trinidad", "tobago", "wasp-12b", "macedonia"}
ANCHOR_ORIGINS = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}


def clean_timestamp(value):
    try:
        return pd.to_datetime(value)
    except Exception:
        return pd.NaT

# ✅ Extract origin column groups
def extract_origins(columns):
    origins = {}
    for col in columns:
        col = col.strip().lower()
        if col in ["time", "open"]:
            continue
        if any(suffix in col for suffix in [" h", " l", " c"]):
            bracket = ""
            if "[" in col and "]" in col:
                bracket = col[col.find("["):col.find("]")+1]
            core = col.replace(" h", "").replace(" l", "").replace(" c", "")
            if bracket and not core.endswith(bracket):
                core += bracket
            origins.setdefault(core, []).append(col)
    return {origin: cols for origin, cols in origins.items() if len(cols) == 3}

# ✅ Get input value for a given report_time
def get_input_value(df, report_time):
    match = df[df["time"] == report_time]
    return match.iloc[-1]["open"] if not match.empty and "open" in match.columns else None

# ✅ Calculate pivot output
def calculate_pivot(H, L, C, M_value):
    return ((H + L + C) / 3) + M_value * (H - L)

# ✅ Get day index label
def get_day_index(arrival, report_time, start_hour):
    if not report_time:
        return "[0] Today"
    report_day_start = report_time.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    if report_time.hour < start_hour:
        report_day_start -= dt.timedelta(days=1)
    days_diff = (arrival - report_day_start) // dt.timedelta(days=1)
    return f"[{int(days_diff)}]"

# 🎨🖌️ Highlight Traveler Report
def highlight_traveler_report(df):
    """🧾 Apply full highlighting to Origin, Day '[0]', and M # / M Name rows in traveler report."""

    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    for i, row in df.iterrows():
        # 🌍 Origin-based highlight
        origin = str(row.get("Origin", "")).lower()
        if origin in ["spain", "saturn"]:
            styles.at[i, "Origin"] = "background-color: #d4edda;"  # light green
        elif origin == "jupiter":
            styles.at[i, "Origin"] = "background-color: #d1ecf1;"  # light blue
        elif origin in ["kepler-62", "kepler-44"]:
            styles.at[i, "Origin"] = "background-color: #fff3cd;"  # light orange

        # 🟨 Highlight Day == '[0]' across other columns
        day_val = str(row.get("Day", "")).strip().lower()
        if day_val == "[0]":
            for col in df.columns:
                if col not in ["Origin", "M Name", "M #"]:
                    styles.at[i, col] = "background-color: yellow;"
            # Make '[0]' cell bold
            if "Day" in df.columns:
                styles.at[i, "Day"] += "font-weight: bold;"

        # 🔢 Highlight M # and M Name
        m_val = int(row.get("M #", -1))
        m_style = ""
        if m_val == 0:
            m_style = "background-color: lavender; font-weight: bold;"  # 💜 M # 0
        elif m_val == 40:
            m_style = "background-color: lightgray; font-weight: bold;"  # 🩶 M # 40
        elif m_val == 54:
            m_style = "background-color: lightblue; font-weight: bold;"  # 🔵 M # 54

        for col in ["M #", "M Name"]:
            if col in df.columns:
                styles.at[i, col] = m_style

    return df.style.apply(lambda _: styles, axis=None)



# ✅ Calculate weekly anchor time
def get_weekly_anchor(report_time, weeks_back, start_hour):
    days_since_sunday = (report_time.weekday() + 1) % 7
    anchor_date = report_time - dt.timedelta(days=days_since_sunday + 7 * (weeks_back - 1))
    return anchor_date.replace(hour=start_hour, minute=0, second=0, microsecond=0)

# ✅ Calculate monthly anchor time
def get_monthly_anchor(report_time, months_back, start_hour):
    year = report_time.year
    month = report_time.month - (months_back - 1)
    while month <= 0:
        month += 12
        year -= 1
    return dt.datetime(year, month, 1, hour=start_hour, minute=0, second=0, microsecond=0)

# ✅ Main feed processor
def process_feed(df, feed_type, report_time, scope_type, scope_value, start_hour, measurements, input_value):
    df.columns = df.columns.str.strip().str.lower()
    df["time"] = df["time"].apply(clean_timestamp)
    df = df.iloc[::-1]  # reverse chronological

    if report_time:
        if scope_type == "Rows":
            try:
                start_index = df[df["time"] == report_time].index[0]
                df = df.iloc[start_index:start_index + scope_value]
            except:
                pass
        else:
            cutoff = report_time - pd.Timedelta(days=scope_value)
            df = df[df["time"] >= cutoff]

    origins = extract_origins(df.columns)
    new_data_rows = []

    for origin, cols in origins.items():
        relevant_rows = df[["time", "open"] + cols].dropna()
        origin_name = origin.lower()
        is_special = any(tag in origin_name for tag in ["wasp", "macedonia"])

        if is_special:
            report_row = relevant_rows[relevant_rows["time"] == report_time]
            if report_row.empty:
                continue
            current = report_row.iloc[0]
            bracket_number = 0
            if "[" in origin_name and "]" in origin_name:
                try:
                    bracket_number = int(origin_name.split("[")[-1].replace("]", ""))
                except:
                    pass
            if "wasp" in origin_name:
                arrival_time = get_weekly_anchor(report_time, max(1, bracket_number), start_hour)
            elif "macedonia" in origin_name:
                arrival_time = get_monthly_anchor(report_time, max(1, bracket_number), start_hour)
            else:
                arrival_time = report_time
            H, L, C = current[cols[0]], current[cols[1]], current[cols[2]]
            for _, row in measurements.iterrows():
                output = calculate_pivot(H, L, C, row["m value"])
                day = get_day_index(arrival_time, report_time, start_hour)
                new_data_rows.append({
                    "Feed": feed_type,
                    "Arrival": arrival_time,
                    "Origin": origin,
                    "M Name": row["m name"],
                    "M #": row["m #"],
                    "R #": row["r #"],
                    "Tag": row["tag"],
                    "Family": row["family"],
                    "Input": input_value,
                    "Output": output,
                    "Diff": output - input_value,
                    "Day": day
                })
            continue

        for i in range(len(relevant_rows) - 1):
            current = relevant_rows.iloc[i]
            above = relevant_rows.iloc[i + 1]
            changed = any(current[col] != above[col] for col in cols)
            if changed:
                arrival_time = current["time"]
                H, L, C = current[cols[0]], current[cols[1]], current[cols[2]]
                for _, row in measurements.iterrows():
                    output = calculate_pivot(H, L, C, row["m value"])
                    day = get_day_index(arrival_time, report_time, start_hour)
                    new_data_rows.append({
                        "Feed": feed_type,
                        "Arrival": arrival_time,
                        "Origin": origin,
                        "M Name": row["m name"],
                        "M #": row["m #"],
                        "R #": row["r #"],
                        "Tag": row["tag"],
                        "Family": row["family"],
                        "Input": input_value,
                        "Output": output,
                        "Diff": output - input_value,
                        "Day": day
                    })

    return new_data_rows
