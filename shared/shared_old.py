import pandas as pd
import datetime as dt
from dateutil import parser

# Constants: Origin Classifications   🌌 🪐 💫 ☄️ 🌍 🏝️ 🍹 ⛱️ 🌞 🌊
EPIC_ORIGINS = {"trinidad", "tobago", "wasp-12b", "macedonia"}
ANCHOR_ORIGINS = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}

# Constants: M# Traveler Family Classifications 🧭 📊 🎯 ⚡ 🔢 🌈 📈 📉 🎨 🔍
# Strength travelers (M# values of 0, 40, -40, 54, -54)
STRENGTH_TRAVELERS = {0, 40, -40, 54, -54}

# Tag B travelers (M# values of 78.01, -78.01, 90.5, -90.5, 95.5, -95.5)
TAG_B_TRAVELERS = {78.01, -78.01, 90.5, -90.5, 95.5, -95.5}

# Family Gry travelers (M# values of 62, -62, 78.01, -78.01, 83, -83, 90.5, -90.5, 95.5, -95.5)
FAMILY_GRY_TRAVELERS = {62, -62, 78.01, -78.01, 83, -83, 90.5, -90.5, 95.5, -95.5}

# Family Orn travelers (M# values of 12, -12, 24, -24, 47, -47, 57, -57, 71, -71, 85, -85, 93.5, -93.5)
FAMILY_ORN_TRAVELERS = {12, -12, 24, -24, 47, -47, 57, -57, 71, -71, 85, -85, 93.5, -93.5}

# Family Blu travelers (M# values of 15, -15, 27, -27, 33, -33, 38, -38, 45, -45, 54, -54, 67, -67, 74, -74, 80, -80, 85, -85, 89, -89, 92, -92, 95, -95)
FAMILY_BLU_TRAVELERS = {15, -15, 27, -27, 33, -33, 38, -38, 45, -45, 54, -54, 67, -67, 74, -74, 80, -80, 85, -85, 89, -89, 92, -92, 95, -95}

# Family Alpha travelers (M# values of 2, -2, 10, -10, 22, -22, 30, -30, 36, -36, 39, -39, 41, -41, 43, -43, 50, -50, 60, -60, 77, -77, 107, -107)
FAMILY_ALPHA_TRAVELERS = {2, -2, 10, -10, 22, -22, 30, -30, 36, -36, 39, -39, 41, -41, 43, -43, 50, -50, 60, -60, 77, -77, 107, -107}

# Family Bravo travelers (M# values of 5, -5, 14, -14, 55, -55, 68, -68, 96, -96)
FAMILY_BRAVO_TRAVELERS = {5, -5, 14, -14, 55, -55, 68, -68, 96, -96}

# Family Charlie travelers (M# values of 6, -6, 87, -87)
FAMILY_CHARLIE_TRAVELERS = {6, -6, 87, -87}

# Family Delta travelers (M# values of 3, -3, 103, -103)
FAMILY_DELTA_TRAVELERS = {3, -3, 103, -103}

# Family Echo travelers (M# values of 1, -1, 111, -111)
FAMILY_ECHO_TRAVELERS = {1, -1, 111, -111}

def clean_timestamp(value):
    try:
        # Handle ISO format with timezone (e.g., 2025-07-24T15:30:00-04:00)
        # Parse as naive datetime, ignoring timezone offset
        if isinstance(value, str) and 'T' in value:
            # Remove timezone offset (+/-XX:XX) to keep local time as-is
            if '+' in value:
                value = value.split('+')[0]
            elif value.count('-') >= 3:  # Has timezone offset like -04:00
                # Find last dash that's part of timezone (after 'T')
                parts = value.split('T')
                if len(parts) == 2:
                    date_part = parts[0]
                    time_part = parts[1]
                    if '-' in time_part:
                        time_part = time_part.split('-')[0]
                    value = f"{date_part}T{time_part}"
        return pd.to_datetime(value)
    except Exception:
        try:
            # Fallback to standard parsing
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

# ✅ Get input value at day start time (17:00 or 18:00) looking back from report time
def get_input_at_day_start(df, report_time, start_hour):
    """Get input value at the most recent day start time (17:00 or 18:00) before or at report time"""
    if df is None or report_time is None:
        return None
    
    # Start with the day start time on the same date as report_time
    target_time = report_time.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    
    # If the report time is before the day start time on the same day,
    # we need to go back to the previous day's day start time
    if report_time < target_time:
        target_time = target_time - pd.Timedelta(days=1)
    
    # First try exact match at the target time
    exact_match = df[df["time"] == target_time]
    if not exact_match.empty and "open" in exact_match.columns:
        return exact_match.iloc[-1]["open"]
    
    # If no exact match, find the closest time to the target time that's <= report_time
    if "time" in df.columns and "open" in df.columns:
        # Filter to times that are <= report_time
        valid_times_df = df[df["time"] <= report_time]
        if not valid_times_df.empty:
            valid_times_df = valid_times_df.copy()
            valid_times_df["time_diff"] = abs(valid_times_df["time"] - target_time)
            closest_row = valid_times_df.loc[valid_times_df["time_diff"].idxmin()]
            return closest_row["open"]
    
    return None

# ✅ Backward compatibility wrapper for get_input_at_18
def get_input_at_18(df, report_time):
    """Backward compatibility wrapper - defaults to 18:00"""
    return get_input_at_day_start(df, report_time, 18)

# ✅ Get input value at specific time from small feed (with fallback to closest time)
def get_input_at_time(small_df, target_time):
    """Get input value from small feed at specific time, or closest available time"""
    if small_df is None or target_time is None:
        return None
    
    # First try exact match
    exact_match = small_df[small_df["time"] == target_time]
    if not exact_match.empty and "open" in exact_match.columns:
        return exact_match.iloc[-1]["open"]
    
    # If no exact match, find closest time
    if "time" in small_df.columns and "open" in small_df.columns:
        small_df_copy = small_df.copy()
        small_df_copy["time_diff"] = abs(small_df_copy["time"] - target_time)
        closest_row = small_df_copy.loc[small_df_copy["time_diff"].idxmin()]
        return closest_row["open"]
    
    return None

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

# ✅ Highlight Anchor Origins ( old method)
def highlight_anchor_origins(df):
    def highlight(cell):
        origin = str(cell).lower()
        if origin in ["spain", "saturn"]:
            return "background-color: #d4edda;"  # light green
        elif origin == "jupiter":
            return "background-color: #d1ecf1;"  # light blue
        elif origin in ["kepler-62", "kepler-44"]:
            return "background-color: #fff3cd;"  # light orange
        return ""
    
    return df.style.applymap(highlight, subset=["Origin"])

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

# ✅ Range filtering helper functions
def _output_in_ranges(output, filter_ranges):
    """Check if output falls within any of the specified ranges"""
    if not filter_ranges:
        return True
    
    for range_info in filter_ranges:
        if range_info["lower"] <= output <= range_info["upper"]:
            return True
    return False

# Debug counter for filtering
_filter_debug_counter = {"total": 0, "filtered": 0, "passed": 0}

def _get_range_info(output, filter_ranges):
    """Get range name and zone for a given output value"""
    for range_info in filter_ranges:
        if range_info["lower"] <= output <= range_info["upper"]:
            # Calculate zone based on distance from upper/lower limits
            if range_info["type"] == "high":
                # For highs, measure distance from upper limit
                distance = range_info["upper"] - output
            else:
                # For lows, measure distance from lower limit  
                distance = output - range_info["lower"]
            
            if distance <= 6:
                zone = "0 to 6"
            elif distance <= 12:
                zone = "6 to 12"
            elif distance <= 18:
                zone = "12 to 18"
            else:
                zone = "18 to 24"
                
            return {"name": range_info["name"], "zone": zone}
    
    # Fallback for full range mode
    return {"name": "Full Range", "zone": ""}

# ✅ Main feed processor - UPDATED with new columns (back to raw generation)
def process_feed(df, feed_type, report_time, scope_type, scope_value, start_hour, measurements, input_value_at_start, small_df):
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
            
            # Calculate additional input values
            input_at_arrival = get_input_at_time(small_df, arrival_time)
            input_at_report = get_input_at_time(small_df, report_time)
            
            for _, row in measurements.iterrows():
                output = calculate_pivot(H, L, C, row["m value"])
                day = get_day_index(arrival_time, report_time, start_hour)
                new_data_rows.append({
                    "Feed": feed_type,
                    "Arrival": arrival_time,
                    "Day": day,
                    "Origin": origin,
                    "M Name": row["m name"],
                    "M #": row["m #"],
                    "R #": row["r #"],
                    "Tag": row["tag"],
                    "Family": row["family"],
                    f"Input @ {start_hour:02d}:00": input_value_at_start,
                    f"Diff @ {start_hour:02d}:00": output - input_value_at_start,
                    "Input @ Arrival": input_at_arrival,
                    "Diff @ Arrival": output - input_at_arrival if input_at_arrival is not None else None,
                    "Input @ Report": input_at_report,
                    "Diff @ Report": output - input_at_report if input_at_report is not None else None,
                    "Output": output
                })
            continue

        for i in range(len(relevant_rows) - 1):
            current = relevant_rows.iloc[i]
            above = relevant_rows.iloc[i + 1]
            changed = any(current[col] != above[col] for col in cols)
            if changed:
                arrival_time = current["time"]
                H, L, C = current[cols[0]], current[cols[1]], current[cols[2]]
                
                # Calculate additional input values
                input_at_arrival = get_input_at_time(small_df, arrival_time)
                input_at_report = get_input_at_time(small_df, report_time)
                
                for _, row in measurements.iterrows():
                    output = calculate_pivot(H, L, C, row["m value"])          
                    day = get_day_index(arrival_time, report_time, start_hour)
                    new_data_rows.append({
                        "Feed": feed_type,
                        "Arrival": arrival_time,
                        "Day": day,
                        "Origin": origin,
                        "M Name": row["m name"],
                        "M #": row["m #"],
                        "R #": row["r #"],
                        "Tag": row["tag"],
                        "Family": row["family"],
                        f"Input @ {start_hour:02d}:00": input_value_at_start,
                        f"Diff @ {start_hour:02d}:00": output - input_value_at_start,
                        "Input @ Arrival": input_at_arrival,
                        "Diff @ Arrival": output - input_at_arrival if input_at_arrival is not None else None,
                        "Input @ Report": input_at_report,
                        "Diff @ Report": output - input_at_report if input_at_report is not None else None,
                        "Output": output
                    })

    return new_data_rows

# 🎨 🖌️ ✅ Enhanced highlighting for traveler reports with updated colors  
def highlight_traveler_report(df):
    """Apply highlighting to traveler report with updated origin colors, output duplicates, Day '[0]', and M# values"""
    def apply_styles(row):
        style = [""] * len(row)
        col_map = {col: i for i, col in enumerate(df.columns)}

        # Enhanced Origin-based highlighting with new colors
        origin = str(row.get("Origin", "")).lower()
        if "Origin" in col_map:
            if origin in ["spain", "saturn"]:
                style[col_map["Origin"]] = "background-color: #39FF14;"  # 🌿 Neon Green
            elif origin == "jupiter":
                style[col_map["Origin"]] = "background-color: #d1ecf1;"  # 🌌 light blue (unchanged)
            elif origin in ["kepler-62", "kepler-44"]:
                style[col_map["Origin"]] = "background-color: #ff4d00;"  # 🟠 Red Orange
            elif origin in ["trinidad", "tobago"]:
                style[col_map["Origin"]] = "background-color: #f0cb59;"  # Gold for Trinidad/Tobago
            elif "wasp" in origin:
                style[col_map["Origin"]] = "background-color: lightgray;"  # Light Gray for Wasp
            elif "macedonia" in origin:
                style[col_map["Origin"]] = "background-color: #e022d7;"  # Magenta for Macedonia

        # Restored: Highlight Day == '[0]' across other columns (excluding Origin, M Name, M #)
        day_val = str(row.get("Day", "")).strip().lower()
        if day_val == "[0]":
            for col in df.columns:
                if col not in ["Origin", "M Name", "M #"] and col in col_map:
                    style[col_map[col]] = "background-color: yellow;"
            if "Day" in col_map:
                style[col_map["Day"]] += " font-weight: bold;"

        # Restored: Highlight M # and M Name based on M# values
        try:
            m_val = int(row.get("M #", -999))
            m_style = ""
            if m_val == 0:
                m_style = "background-color: lavender; font-weight: bold;"  # M # 0
            elif abs(m_val) == 40:
                m_style = "background-color: lightgray; font-weight: bold;"  # M # ±40
            elif abs(m_val) == 54:
                m_style = "background-color: lightblue; font-weight: bold;"  # M # ±54

            if m_style:
                for col in ["M #", "M Name"]:
                    if col in col_map:
                        style[col_map[col]] = m_style
        except Exception:
            pass

        return style
    
    def highlight_output_duplicates(series):
        """Highlight outputs that appear more than once with yellow"""
        value_counts = series.value_counts()
        duplicates = value_counts[value_counts > 1].index
        return ['background-color: yellow' if val in duplicates else '' for val in series]
    
    # Apply row-based styling first
    styled = df.style.apply(apply_styles, axis=1)
    
    # Then apply output duplicates highlighting
    if "Output" in df.columns:
        styled = styled.apply(highlight_output_duplicates, subset=["Output"])
    
    return styled
    
# ✅ Custom traveler report highlighting with zone colors and enhanced origin highlighting
def highlight_custom_traveler_report(df, show_highlighting=True):
    """Apply highlighting for custom traveler report with zone colors and updated origin colors"""
    if not show_highlighting:
        return df.style
    
    def apply_styles(row):
        style = [""] * len(row)
        col_map = {col: i for i, col in enumerate(df.columns)}

        # Enhanced Origin-based highlighting with new colors
        origin = str(row.get("Origin", "")).lower()
        if "Origin" in col_map:
            if origin in ["spain", "saturn"]:
                style[col_map["Origin"]] = "background-color: #39FF14;"  # Neon Green
            elif origin == "jupiter":
                style[col_map["Origin"]] = "background-color: #d1ecf1;"  # light blue (unchanged)
            elif origin in ["kepler-62", "kepler-44"]:
                style[col_map["Origin"]] = "background-color: #ff4d00;"  # Red Orange
            elif origin in ["trinidad", "tobago"]:
                style[col_map["Origin"]] = "background-color: #f0cb59;"  # Gold for Trinidad/Tobago
            elif "wasp" in origin:
                style[col_map["Origin"]] = "background-color: lightgray;"  # Light Gray for Wasp
            elif "macedonia" in origin:
                style[col_map["Origin"]] = "background-color: #e022d7;"  # Magenta for Macedonia

        # Restored: Highlight Day == '[0]' across other columns (excluding Origin, M Name, M #)
        day_val = str(row.get("Day", "")).strip().lower()
        if day_val == "[0]":
            for col in df.columns:
                if col not in ["Origin", "M Name", "M #"] and col in col_map:
                    style[col_map[col]] = "background-color: yellow;"
            if "Day" in col_map:
                style[col_map["Day"]] += " font-weight: bold;"

        # Restored: Highlight M # and M Name based on M# values
        try:
            m_val = int(row.get("M #", -999))
            m_style = ""
            if m_val == 0:
                m_style = "background-color: lavender; font-weight: bold;"  # M # 0
            elif abs(m_val) == 40:
                m_style = "background-color: lightgray; font-weight: bold;"  # M # ±40
            elif abs(m_val) == 54:
                m_style = "background-color: lightblue; font-weight: bold;"  # M # ±54

            if m_style:
                for col in ["M #", "M Name"]:
                    if col in col_map:
                        style[col_map[col]] = m_style
        except Exception:
            pass

        # Zone highlighting based on range type and zone
        if "Zone" in col_map and "Range" in col_map:
            zone_val = str(row.get("Zone", ""))
            range_val = str(row.get("Range", ""))
            
            if zone_val and range_val and zone_val != "":
                if "High" in range_val:
                    # High ranges: red gradient (bright red to orange)
                    if zone_val == "0 to 6":
                        style[col_map["Zone"]] = "background-color: #ff4444;"  # Bright red
                    elif zone_val == "6 to 12":
                        style[col_map["Zone"]] = "background-color: #ff7744;"  # Red-orange
                    elif zone_val == "12 to 18":
                        style[col_map["Zone"]] = "background-color: #ffaa44;"  # Orange
                    elif zone_val == "18 to 24":
                        style[col_map["Zone"]] = "background-color: #ffdd44;"  # Light orange
                elif "Low" in range_val:
                    # Low ranges: blue gradient (bright blue to gray)
                    if zone_val == "0 to 6":
                        style[col_map["Zone"]] = "background-color: #4488ff;"  # Bright blue
                    elif zone_val == "6 to 12":
                        style[col_map["Zone"]] = "background-color: #77aaff;"  # Medium blue
                    elif zone_val == "12 to 18":
                        style[col_map["Zone"]] = "background-color: #aaccff;"  # Light blue
                    elif zone_val == "18 to 24":
                        style[col_map["Zone"]] = "background-color: #cccccc;"  # Gray
        
        return style
    
    def highlight_zone(cell, range_name):
        """Apply zone highlighting based on range type and zone"""
        zone = str(cell)
        if "High" in range_name:
            # High ranges: red gradient (bright red to orange)
            if zone == "0 to 6":
                return "background-color: #ff4444;"  # Bright red
            elif zone == "6 to 12":
                return "background-color: #ff7744;"  # Red-orange
            elif zone == "12 to 18":
                return "background-color: #ffaa44;"  # Orange
            elif zone == "18 to 24":
                return "background-color: #ffdd44;"  # Light orange
        else:
            # Low ranges: blue gradient (bright blue to gray)
            if zone == "0 to 6":
                return "background-color: #4488ff;"  # Bright blue
            elif zone == "6 to 12":
                return "background-color: #77aaff;"  # Medium blue
            elif zone == "12 to 18":
                return "background-color: #aaccff;"  # Light blue
            elif zone == "18 to 24":
                return "background-color: #cccccc;"  # Gray
        return ""
    
    def highlight_output_duplicates(series):
        """Highlight outputs that appear more than once with yellow"""
        value_counts = series.value_counts()
        duplicates = value_counts[value_counts > 1].index
        return ['background-color: yellow' if val in duplicates else '' for val in series]
    
    def apply_zone_highlighting(row):
        """Apply zone highlighting based on the range for this row"""
        if "Zone" in row.index and "Range" in row.index:
            return highlight_zone(row["Zone"], row["Range"])
        return ""
    
    # Apply row-based styling first
    styled = df.style.apply(apply_styles, axis=1)
    
    # Then apply output duplicates highlighting
    if "Output" in df.columns:
        styled = styled.apply(highlight_output_duplicates, subset=["Output"])
    
    if "Zone" in df.columns:
        # Apply zone highlighting row by row
        for idx, row in df.iterrows():
            if "Range" in df.columns:
                zone_color = highlight_zone(row["Zone"], row["Range"])
                if zone_color:
                    styled = styled.set_properties(subset=(idx, "Zone"), **{"background-color": zone_color.split(":")[1].strip().rstrip(";")})
    
    return styled
