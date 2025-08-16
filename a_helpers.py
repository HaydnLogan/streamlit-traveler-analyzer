import pandas as pd
import numpy as np
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

# Grouping Classifications for M# values
# Group 1a (Green Family, X0p)
GROUP_1A_TRAVELERS = {111, 107, 103, 96, 87, 77, 68, 60, 50, -50, -60, -68, -77, -87, -96, -103, -107, -111}

# Group 1b (Green Family, X0p & d) - Additional 30 values including decimals and middle ranges
GROUP_1B_TRAVELERS = {
    111, 107, 103, 96, 87, 77, 68, 60, 55, 50, 43, 41, 40, 39, 36, 30, 22, 14,
    10, 6, 5, 3, 2, 1, 0, -1, -2, -3, -5, -6, -10, -14, -22, -30, -36, -39,
    -40, -41, -43, -50, -55, -60, -68, -77, -87, -96, -103, -107, -111
}

# Group 2a (Indigo Family, P only)
GROUP_2A_TRAVELERS = {
    101, 98.2, 99.1, 98.2, 98.1, 97.2, 97.1, 96.1, 95.5, 95, 93.5, 92, 90.5,
    89, 86.5, 85, 83, 80, 78.01, 74, 71, 67, 62, 54, 40, 0, -40, -54, -62, -67,
    -71, -74, -78.01, -80, -83, -85, -86.5, -89, -90.5, -92, -93.5, -95, -95.5,
    -96.1, -97.1, -97.2, -98.1, -98.2, -99.1, -98.2, -101
}

# Group 2b (Indigo Family, P & D)
GROUP_2B_TRAVELERS = {
    101, 98.2, 99.1, 98.2, 98.1, 97.2, 97.1, 96.1, 95.5, 95, 93.5, 92, 90.5,
    89, 86.5, 85, 83.0, 80, 78.01, 74, 71, 67, 62, 57, 54, 47, 45, 40, 38, 33,
    27, 24, 15, 12, 0, -12, -15, -24, -27, -33, -38, -40, -45, -47, -54, -57,
    -62, -67, -71, -74, -78.01, -80, -83.0, -85, -86.5, -89, -90.5, -92, -93.5,
    -95, -95.5, -96.1, -97.1, -97.2, -98.1, -98.2, -99.1, -98.2, -101
}

def generate_master_traveler_list(data, measurements, small_df, report_time, start_hour=17, fast_mode=True):
    """Generate master traveler list using first measurement tab, then filter for 4 sub-reports"""
    
    # Ensure time column exists and is datetime
    if 'time' not in data.columns:
        return {}
    
    data['time'] = pd.to_datetime(data['time'])
    
    # Define measurement columns
    price_cols = ['high', 'low', 'close']
    if not all(col in data.columns for col in price_cols):
        return {}
    
    # Get unique origins
    origins = data['origin'].unique() if 'origin' in data.columns else ['default']
    
    # Process each origin to create master list
    all_traveler_data = []
    
    for origin in origins:
        origin_data = data[data['origin'] == origin] if 'origin' in data.columns else data
        
        if len(origin_data) < 2:
            continue
            
        # Sort by time
        origin_data = origin_data.sort_values('time')
        
        # Find price changes
        for i in range(len(origin_data) - 1):
            current = origin_data.iloc[i]
            next_row = origin_data.iloc[i + 1]
            
            # Check if any price changed
            price_changed = any(current[col] != next_row[col] for col in price_cols)
            
            if price_changed:
                arrival_time = current['time']
                H, L, C = current['high'], current['low'], current['close']
                
                # Calculate input values
                input_at_arrival = get_input_at_time(small_df, arrival_time)
                input_at_report = get_input_at_time(small_df, report_time)
                input_at_start = get_input_at_time(small_df, report_time.replace(hour=start_hour, minute=0, second=0))
                
                # Generate entries for measurements from Excel file
                for _, measurement_row in measurements.iterrows():
                    m_value = get_measurement_value(measurement_row)
                    output = calculate_pivot(H, L, C, m_value)
                    day = get_day_index(arrival_time, report_time, start_hour)
                    
                    # Format arrival time
                    try:
                        day_abbrev = arrival_time.strftime('%a')
                        arrival_excel = arrival_time.strftime('%Y-%m-%d %H:%M')
                    except:
                        day_abbrev = ""
                        arrival_excel = str(arrival_time)
                    
                    # Classify group
                    group = classify_traveler_group(m_value)
                    
                    traveler_entry = {
                        "Feed": "auto",
                        "ddd": day_abbrev,
                        "Arrival": arrival_excel,
                        "Arrival_datetime": arrival_time,
                        "Day": day,
                        "Origin": origin,
                        "M Name": measurement_row.get("m name", measurement_row.get("M Name", measurement_row.get("M name", f"M{m_value}"))),
                        "M #": m_value,
                        "R #": measurement_row.get("r #", measurement_row.get("R #", "")),
                        "Tag": measurement_row.get("tag", measurement_row.get("Tag", "")),
                        "Family": measurement_row.get("family", measurement_row.get("Family", "")),
                        "Group": group,
                        f"Input @ {start_hour:02d}:00": input_at_start,
                        f"Diff @ {start_hour:02d}:00": output - input_at_start if input_at_start is not None else None,
                        "Input @ Arrival": input_at_arrival,
                        "Diff @ Arrival": output - input_at_arrival if input_at_arrival is not None else None,
                        "Input @ Report": input_at_report,
                        "Diff @ Report": output - input_at_report if input_at_report is not None else None,
                        "Output": output
                    }
                    
                    all_traveler_data.append(traveler_entry)
    
    # Convert to DataFrame
    master_df = pd.DataFrame(all_traveler_data)
    
    if master_df.empty:
        return {}
    
    # Remove Arrival_datetime for final output
    master_display_df = master_df.drop('Arrival_datetime', axis=1) if 'Arrival_datetime' in master_df.columns else master_df
    
    # Filter into 4 sub-reports
    reports = {}
    
    # Group 1a
    group_1a_df = master_display_df[master_display_df['Group'] == 'Group 1a'].copy()
    reports["Grp 1a"] = group_1a_df
    
    # Group 1b  
    group_1b_df = master_display_df[master_display_df['Group'] == 'Group 1b'].copy()
    reports["Grp 1b"] = group_1b_df
    
    # Group 2a
    group_2a_df = master_display_df[master_display_df['Group'] == 'Group 2a'].copy()
    reports["Grp 2a"] = group_2a_df
    
    # Group 2b
    group_2b_df = master_display_df[master_display_df['Group'] == 'Group 2b'].copy()
    reports["Grp 2b"] = group_2b_df
    
    return reports

def get_measurement_value(row):
    """Extract measurement value from row using flexible column naming"""
    for col in ['m #', 'M #', 'M value', 'm value', 'measurement']:
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return 0

def calculate_pivot(H, L, C, m_value):
    """Calculate output using pivot formula"""
    try:
        H, L, C, m_value = float(H), float(L), float(C), float(m_value)
        return ((H + L + C) / 3) + (m_value / 1000)
    except:
        return 0

def get_day_index(arrival_time, report_time, start_hour):
    """Calculate day index based on arrival and report times"""
    try:
        # Convert to datetime if needed
        if isinstance(arrival_time, str):
            arrival_time = pd.to_datetime(arrival_time)
        if isinstance(report_time, str):
            report_time = pd.to_datetime(report_time)
        
        # Calculate day difference
        day_diff = (report_time.date() - arrival_time.date()).days
        return f"[{day_diff}]"
    except:
        return "[0]"

def classify_traveler_group(m_value):
    """Classify M# value into new grouping system"""
    try:
        # Handle both int and float values
        m_val = float(m_value) if not pd.isna(m_value) else None
        if m_val is None:
            return "Unclassified"
            
        # Check Group 1a (empty)
        if m_val in GROUP_1A_TRAVELERS:
            return "Group 1a"
            
        # Check Group 1b (complete integer sequence)
        if m_val in GROUP_1B_TRAVELERS:
            return "Group 1b"
            
        # Check Group 2a (decimal sequence - higher ranges)
        if m_val in GROUP_2A_TRAVELERS:
            return "Group 2a"
            
        # Check Group 2b (extended decimal sequence)
        if m_val in GROUP_2B_TRAVELERS:
            return "Group 2b"
            
        return "Unclassified"
    except:
        return "Unclassified"

def get_traveler_family_summary(m_value):
    """Get comprehensive traveler family classification"""
    families = []
    try:
        m_val = float(m_value) if not pd.isna(m_value) else None
        if m_val is None:
            return "None"
            
        # Check all family classifications
        if m_val in STRENGTH_TRAVELERS:
            families.append("Strength")
        if m_val in TAG_B_TRAVELERS:
            families.append("Tag B")
        if m_val in FAMILY_GRY_TRAVELERS:
            families.append("Gry")
        if m_val in FAMILY_ORN_TRAVELERS:
            families.append("Orn")
        if m_val in FAMILY_BLU_TRAVELERS:
            families.append("Blu")
        if m_val in FAMILY_ALPHA_TRAVELERS:
            families.append("Alpha")
        if m_val in FAMILY_BRAVO_TRAVELERS:
            families.append("Bravo")
        if m_val in FAMILY_CHARLIE_TRAVELERS:
            families.append("Charlie")
        if m_val in FAMILY_DELTA_TRAVELERS:
            families.append("Delta")
        if m_val in FAMILY_ECHO_TRAVELERS:
            families.append("Echo")
            
        # Add new groupings
        group = classify_traveler_group(m_val)
        if group != "Unclassified":
            families.append(group)
            
        return " | ".join(families) if families else "Unclassified"
    except:
        return "Error"

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
    # Ensure time column is datetime for comparison and handle timezone compatibility
    df_copy = df.copy()
    df_copy["time"] = pd.to_datetime(df_copy["time"]).dt.tz_localize(None)  # Remove timezone info
    report_time_naive = pd.to_datetime(report_time).tz_localize(None) if hasattr(report_time, 'tz') and report_time.tz else report_time
    match = df_copy[df_copy["time"] == report_time_naive]
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
    # Ensure time column is datetime for comparison
    df_copy = df.copy()
    df_copy["time"] = pd.to_datetime(df_copy["time"]).dt.tz_localize(None)  # Remove timezone info
    target_time_naive = pd.to_datetime(target_time).tz_localize(None) if hasattr(target_time, 'tz') and target_time.tz else target_time
    report_time_naive = pd.to_datetime(report_time).tz_localize(None) if hasattr(report_time, 'tz') and report_time.tz else report_time
    
    exact_match = df_copy[df_copy["time"] == target_time_naive]
    if not exact_match.empty and "open" in exact_match.columns:
        return exact_match.iloc[-1]["open"]
    
    # If no exact match, find the closest time to the target time that's <= report_time
    if "time" in df.columns and "open" in df.columns:
        # Filter to times that are <= report_time
        valid_times_df = df_copy[df_copy["time"] <= report_time_naive]
        if not valid_times_df.empty:
            valid_times_df = valid_times_df.copy()
            valid_times_df["time_diff"] = abs(valid_times_df["time"] - target_time_naive)
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
    # Ensure time column is datetime for comparison and handle timezone compatibility
    small_df_copy = small_df.copy()
    small_df_copy["time"] = pd.to_datetime(small_df_copy["time"]).dt.tz_localize(None)  # Remove timezone info
    target_time_naive = pd.to_datetime(target_time).tz_localize(None) if hasattr(target_time, 'tz') and target_time.tz else target_time
    
    exact_match = small_df_copy[small_df_copy["time"] == target_time_naive]
    if not exact_match.empty and "open" in exact_match.columns:
        return exact_match.iloc[-1]["open"]
    
    # If no exact match, find closest time
    if "time" in small_df.columns and "open" in small_df.columns:
        small_df_copy["time_diff"] = abs(small_df_copy["time"] - target_time_naive)
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

# Helper function to get measurement value with flexible column matching
def get_measurement_value(row, possible_columns=None):
    """Get M value from measurement row with flexible column name matching"""
    if possible_columns is None:
        possible_columns = ["M value", "m value", "M_value", "m_value", "M #", "m #", "m#", "M#"]
    
    # First try exact case-sensitive matches
    for col in possible_columns:
        if col in row.index:
            return row[col]
    
    # If no exact match, try case-insensitive search
    row_lower = {k.lower(): v for k, v in row.items()}
    for col in ["m value", "m_value", "m #", "m#"]:
        if col in row_lower:
            return row_lower[col]
    
    # Return the first numeric column if nothing else works
    for col, val in row.items():
        if isinstance(val, (int, float)) and not pd.isna(val):
            return val
    
    return 0  # Default fallback

# ✅ Main feed processor - UPDATED with new columns and flexible measurement column handling
def process_feed(df,
                 feed_type,
                 report_time,
                 scope_type,
                 scope_value,
                 start_hour,
                 measurements,
                 input_value_at_start,
                 small_df,
                 use_full_range=False,
                 full_range_value=24):
    # PERFORMANCE DEBUG: Log if full range filtering is active
    if use_full_range:
        print(
            f"🚀 PERFORMANCE: Full range filtering active for {feed_type} feed - range: {full_range_value}, measurements: {len(measurements)}"
        )
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
                    bracket_number = int(
                        origin_name.split("[")[-1].replace("]", ""))
                except:
                    pass
            if "wasp" in origin_name:
                arrival_time = get_weekly_anchor(report_time,
                                                 max(1, bracket_number),
                                                 start_hour)
            elif "macedonia" in origin_name:
                arrival_time = get_monthly_anchor(report_time,
                                                  max(1, bracket_number),
                                                  start_hour)
            else:
                arrival_time = report_time
            H, L, C = current[cols[0]], current[cols[1]], current[cols[2]]

            # Calculate additional input values
            input_at_arrival = get_input_at_time(small_df, arrival_time)
            input_at_report = get_input_at_time(small_df, report_time)

            for _, row in measurements.iterrows():
                # Use flexible measurement value extraction
                m_value = get_measurement_value(row)
                output = calculate_pivot(H, L, C, m_value)

                # OPTIMIZATION: Skip processing if using full range and output is outside range
                if use_full_range and input_value_at_start is not None:
                    high_limit = input_value_at_start + full_range_value
                    low_limit = input_value_at_start - full_range_value
                    if output < low_limit or output > high_limit:
                        continue  # Skip this output

                day = get_day_index(arrival_time, report_time, start_hour)

                # Format arrival time into separate columns
                try:
                    day_abbrev = arrival_time.strftime(
                        '%a')  # Mon, Tue, Wed, etc.
                    arrival_excel = arrival_time.strftime(
                        '%Y-%m-%d %H:%M')  # Excel-friendly format
                except:
                    day_abbrev = ""
                    arrival_excel = str(arrival_time)

                new_data_rows.append({
                    "Feed": feed_type,
                    "ddd": day_abbrev,
                    "Arrival": arrival_excel,
                    "Arrival_datetime": arrival_time,
                    "Day": day,
                    "Origin": origin,
                    "M Name": row.get("m name", row.get("M Name", row.get("M name", ""))),
                    "M #": row.get("m #", row.get("M #", row.get("M value", et_measurement_value(row)))),
                    "R #": row.get("r #", row.get("R #", "")),
                    "Tag": row.get("tag", row.get("Tag", "")),
                    "Family": row.get("family", row.get("Family", "")),
                    f"Input @ {start_hour:02d}:00": input_value_at_start,
                    f"Diff @ {start_hour:02d}:00": output - input_value_at_start,
                    "Input @ Arrival": input_at_arrival,
                    "Diff @ Arrival": output - input_at_arrival
                    if input_at_arrival is not None else None,
                    "Input @ Report": input_at_report,
                    "Diff @ Report": output - input_at_report
                    if input_at_report is not None else None,
                    "Output": output
                })
        else:
            # Process regular (non-special) origins
            for i in range(len(relevant_rows) - 1):
                current = relevant_rows.iloc[i]
                above = relevant_rows.iloc[i + 1]
                changed = any(current[col] != above[col] for col in cols)
                if changed:
                    arrival_time = current["time"]
                    H, L, C = current[cols[0]], current[cols[1]], current[
                        cols[2]]

                    # Calculate additional input values
                    input_at_arrival = get_input_at_time(
                        small_df, arrival_time)
                    input_at_report = get_input_at_time(small_df, report_time)

                    for _, row in measurements.iterrows():
                        # Use flexible measurement value extraction
                        m_value = get_measurement_value(row)
                        output = calculate_pivot(H, L, C, m_value)

                # OPTIMIZATION: Skip processing if using full range and output is outside range
                if use_full_range and input_value_at_start is not None:
                    high_limit = input_value_at_start + full_range_value
                    low_limit = input_value_at_start - full_range_value
                    if output < low_limit or output > high_limit:
                        continue  # Skip this output

                        day = get_day_index(arrival_time, report_time,
                                            start_hour)

                        # Format arrival time into separate columns
                        try:
                            day_abbrev = arrival_time.strftime(
                                '%a')  # Mon, Tue, Wed, etc.
                            arrival_excel = arrival_time.strftime(
                                '%Y-%m-%d %H:%M')  # Excel-friendly format
                        except:
                            day_abbrev = ""
                            arrival_excel = str(arrival_time)

                        new_data_rows.append({
                            "Feed": feed_type,
                            "ddd": day_abbrev,
                            "Arrival": arrival_excel,
                            "Arrival_datetime": arrival_time,
                            "Day": day,
                            "Origin": origin,
                            "M Name": row.get("m name", row.get("M Name", row.get("M name", ""))),
                            "M #": row.get("m #", row.get("M #", row.get("M value", et_measurement_value(row)))),
                            "R #": row.get("r #", row.get("R #", "")),
                            "Tag": row.get("tag", row.get("Tag", "")),
                            "Family": row.get("family", row.get("Family", "")),
                            f"Input @ {start_hour:02d}:00": input_value_at_start,
                            f"Diff @ {start_hour:02d}:00": output - input_value_at_start,
                            "Input @ Arrival": input_at_arrival,
                            "Diff @ Arrival": output - input_at_arrival
                            if input_at_arrival is not None else None,
                            "Input @ Report": input_at_report,
                            "Diff @ Report": output - input_at_report
                            if input_at_report is not None else None,
                            "Output": output
                        })

    return new_data_rows


# ✅ Apply Excel highlighting using xlsxwriter formatting
def apply_excel_highlighting(workbook, worksheet, df, is_custom_ranges=False):
    """Apply highlighting to Excel export using xlsxwriter formatting"""

    # Formats
    header_format = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'top',
        'fg_color': '#D7E4BC', 'border': 1
    })

    # Date formats (normal + Day[0] yellow)
    date_fmt = workbook.add_format({'num_format': 'mm/dd/yyyy hh:mm'})
    day_zero_format = workbook.add_format({'fg_color': '#FFFF00'})
    day_zero_bold_format = workbook.add_format({'fg_color': '#FFFF00', 'bold': True})
    day_zero_date_fmt = workbook.add_format({'fg_color': '#FFFF00', 'num_format': 'mm/dd/yyyy hh:mm'})

    # Origin color formats
    spain_saturn_format = workbook.add_format({'fg_color': '#39FF14'})  # Neon Green
    jupiter_format = workbook.add_format({'fg_color': '#d1ecf1'})       # Light blue
    kepler_format = workbook.add_format({'fg_color': '#ff4d00'})        # Red Orange
    trinidad_tobago_format = workbook.add_format({'fg_color': '#f0cb59'})  # Gold
    wasp_format = workbook.add_format({'fg_color': '#C0C0C0'})          # Light Gray
    macedonia_format = workbook.add_format({'fg_color': '#e022d7'})     # Magenta

    # M# formats
    m0_format = workbook.add_format({'fg_color': '#E6E6FA', 'bold': True})   # Lavender
    m40_format = workbook.add_format({'fg_color': '#D3D3D3', 'bold': True})  # Light gray
    m54_format = workbook.add_format({'fg_color': '#ADD8E6', 'bold': True})  # Light blue

    # Zone formats (custom ranges only)
    zone_high_0to6_format = workbook.add_format({'fg_color': '#ff0000'})
    zone_high_6to12_format = workbook.add_format({'fg_color': '#ff6666'})
    zone_high_12to18_format = workbook.add_format({'fg_color': '#ffaaaa'})
    zone_high_18to24_format = workbook.add_format({'fg_color': '#ffdd44'})
    zone_low_0to6_format  = workbook.add_format({'fg_color': '#4444ff'})
    zone_low_6to12_format = workbook.add_format({'fg_color': '#7777ff'})
    zone_low_12to18_format = workbook.add_format({'fg_color': '#aaaaff'})
    zone_low_18to24_format = workbook.add_format({'fg_color': '#dddddd'})

    # Header row
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)

    # 👉 Make the whole Arrival column a real Excel date/time
    arrival_col_idx = df.columns.get_loc('Arrival') if 'Arrival' in df.columns else None
    if arrival_col_idx is not None:
        # width ~ 19 fits "mm/dd/yyyy hh:mm"
        worksheet.set_column(arrival_col_idx, arrival_col_idx, 19, date_fmt)

    # Helper: safe datetime conversion for write_datetime
    def _as_pydt(x):
        if pd.isna(x):
            return None
        if isinstance(x, dt.datetime):
            return x
        try:
            return pd.to_datetime(x).to_pydatetime()
        except Exception:
            return None

    # Row formatting
    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        # ----- Origin highlighting -----
        if 'Origin' in df.columns:
            origin_col = df.columns.get_loc('Origin')
            origin = str(row.get('Origin', '')).lower()
            origin_fmt = None
            if origin in ['spain', 'saturn']:
                origin_fmt = spain_saturn_format
            elif origin == 'jupiter':
                origin_fmt = jupiter_format
            elif origin in ['kepler-62', 'kepler-44']:
                origin_fmt = kepler_format
            elif origin in ['trinidad', 'tobago']:
                origin_fmt = trinidad_tobago_format
            elif 'wasp' in origin:
                origin_fmt = wasp_format
            elif 'macedonia' in origin:
                origin_fmt = macedonia_format
            if origin_fmt:
                val = row['Origin']
                worksheet.write(row_idx, origin_col, '' if pd.isna(val) else val, origin_fmt)

        # ----- Day [0] highlighting (special-case Arrival to preserve date typing) -----
        day_val = str(row.get('Day', '')).strip().lower()
        if day_val == '[0]':
            # Bold Day cell
            if 'Day' in df.columns:
                day_col = df.columns.get_loc('Day')
                worksheet.write(row_idx, day_col, '' if pd.isna(row['Day']) else row['Day'], day_zero_bold_format)

            for col_name in df.columns:
                if col_name in ['Origin', 'M Name', 'M #', 'Day', 'Arrival']:
                    continue  # handled separately or intentionally skipped

                col_idx = df.columns.get_loc(col_name)
                val = row[col_name]
                if pd.isna(val) or (isinstance(val, float) and (np.isinf(val) or np.isnan(val))):
                    val = ''
                worksheet.write(row_idx, col_idx, val, day_zero_format)

            # ✅ Handle Arrival as a true Excel date with yellow fill
            if arrival_col_idx is not None:
                # Prefer an explicit Arrival_datetime column if present
                arr_dt = None
                if 'Arrival_datetime' in df.columns and pd.notna(row.get('Arrival_datetime')):
                    arr_dt = _as_pydt(row['Arrival_datetime'])
                if arr_dt is None:
                    arr_dt = _as_pydt(row.get('Arrival'))
                if arr_dt is not None:
                    worksheet.write_datetime(row_idx, arrival_col_idx, arr_dt, day_zero_date_fmt)
                else:
                    # Fallback: write as text (rare)
                    worksheet.write(row_idx, arrival_col_idx, row.get('Arrival', ''), day_zero_date_fmt)

        # ----- M# highlighting -----
        if 'M #' in df.columns:
            try:
                m_val = int(row.get('M #', -999))
                m_fmt = None
                if m_val == 0:
                    m_fmt = m0_format
                elif abs(m_val) == 40:
                    m_fmt = m40_format
                elif abs(m_val) == 54:
                    m_fmt = m54_format
                if m_fmt:
                    m_col = df.columns.get_loc('M #')
                    m_value = row['M #']
                    worksheet.write(row_idx, m_col, '' if pd.isna(m_value) else m_value, m_fmt)
                    if 'M Name' in df.columns:
                        m_name_col = df.columns.get_loc('M Name')
                        m_name_value = row['M Name']
                        worksheet.write(row_idx, m_name_col, '' if pd.isna(m_name_value) else m_name_value, m_fmt)
            except Exception:
                pass

        # ----- Zone highlighting (custom ranges only) -----
        if is_custom_ranges and 'Zone' in df.columns and 'Range' in df.columns:
            zone_val = str(row.get('Zone', ''))
            range_val = str(row.get('Range', ''))
            zone_col = df.columns.get_loc('Zone')
            zone_fmt = None
            if 'High' in range_val:
                zone_fmt = {
                    '0 to 6': zone_high_0to6_format,
                    '6 to 12': zone_high_6to12_format,
                    '12 to 18': zone_high_12to18_format,
                    '18 to 24': zone_high_18to24_format
                }.get(zone_val)
            elif 'Low' in range_val:
                zone_fmt = {
                    '0 to 6': zone_low_0to6_format,
                    '6 to 12': zone_low_6to12_format,
                    '12 to 18': zone_low_12to18_format,
                    '18 to 24': zone_low_18to24_format
                }.get(zone_val)
            if zone_fmt:
                z = row['Zone']
                worksheet.write(row_idx, zone_col, '' if pd.isna(z) else z, zone_fmt)


# ✅ Enhanced highlighting for traveler reports with updated colors and restored M# highlighting
def highlight_traveler_report(df):
    """Apply highlighting to traveler report with updated origin colors, output duplicates, Day '[0]', and M# values"""
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
                        style[col_map["Zone"]] = "background-color: #4444ff;"  # Bright blue
                    elif zone_val == "6 to 12":
                        style[col_map["Zone"]] = "background-color: #7777ff;"  # Medium blue
                    elif zone_val == "12 to 18":
                        style[col_map["Zone"]] = "background-color: #aaaaff;"  # Light blue
                    elif zone_val == "18 to 24":
                        style[col_map["Zone"]] = "background-color: #dddddd;"  # Gray

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



