"""
Custom Range Calculator for Market Data Analysis - FINAL FIXED VERSION
Eliminates all Series boolean ambiguity errors.
"""

import datetime as dt
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import re

def safe_to_datetime(series_or_value, errors='coerce'):
    """Safely convert to datetime with proper error handling - NO SERIES BOOLEAN CHECKS"""
    try:
        if isinstance(series_or_value, pd.Series):
            # Handle Series conversion
            str_series = series_or_value.astype(str)
            clean_series = str_series.str.replace(r'[+-]\d{2}:?\d{2}$', '', regex=True)
            clean_series = clean_series.str.replace('T', ' ')
            return pd.to_datetime(clean_series, errors=errors)
        else:
            # Handle single value conversion
            if pd.isna(series_or_value):
                return pd.NaT
            str_value = str(series_or_value)
            clean_value = re.sub(r'[+-]\d{2}:?\d{2}$', '', str_value)
            clean_value = clean_value.replace('T', ' ')
            return pd.to_datetime(clean_value, errors=errors)
    except Exception:
        # Silent fallback - no warnings in loops
        if isinstance(series_or_value, pd.Series):
            return pd.Series([pd.NaT] * len(series_or_value))
        else:
            return pd.NaT

def ensure_timezone_naive(dt_value):
    """Ensure datetime is timezone naive - COMPLETELY FIXED VERSION"""
    try:
        # Handle None/NaT cases first
        if dt_value is None:
            return None
        if isinstance(dt_value, pd.Series):
            # For Series, just try to remove timezone if present
            if pd.api.types.is_datetime64_any_dtype(dt_value):
                try:
                    # Simple approach: if any timezone info exists, remove it
                    return dt_value.dt.tz_localize(None) if hasattr(dt_value.dtype, 'tz') and dt_value.dtype.tz is not None else dt_value
                except:
                    return dt_value
            else:
                return dt_value
        else:
            # Single value - check for timezone attribute
            if hasattr(dt_value, 'tz') and dt_value.tz is not None:
                return dt_value.replace(tzinfo=None)
            return dt_value
    except Exception:
        # Silent return on error
        return dt_value

def get_input_at_time(df, target_time):
    """Get 'open' input value from dataframe at specific time - SIMPLIFIED VERSION"""
    if df is None or target_time is None:
        return None
    
    try:
        # Check required columns
        if 'time' not in df.columns or 'open' not in df.columns:
            return None
        
        # Handle empty dataframe
        if df.empty:
            return None
        
        # Make a copy and convert time column
        df_copy = df.copy()
        df_copy["time"] = safe_to_datetime(df_copy["time"])
        
        # Simple timezone removal
        df_copy["time"] = ensure_timezone_naive(df_copy["time"])
        
        # Convert target time
        target_time_clean = ensure_timezone_naive(safe_to_datetime(target_time))
        
        # First try exact match
        exact_match = df_copy[df_copy["time"] == target_time_clean]
        if not exact_match.empty:
            return exact_match.iloc[-1]["open"]
        
        # Find closest time
        df_copy["time_diff"] = abs(df_copy["time"] - target_time_clean)
        closest_idx = df_copy["time_diff"].idxmin()
        return df_copy.loc[closest_idx, "open"]
        
    except Exception:
        return None

def get_input_values_batch(small_df, big_df, report_time, start_hour=18):
    """
    Calculate input values for both feeds at once - SIMPLIFIED VERSION
    Returns: (small_start, big_start, small_report, big_report)
    """
    try:
        # Handle report_time
        if report_time is None:
            return None, None, None, None
            
        report_clean = ensure_timezone_naive(safe_to_datetime(report_time))
        if pd.isna(report_clean):
            return None, None, None, None
        
        # Calculate start time
        start_time = report_clean.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        if report_clean.hour < start_hour:
            start_time = start_time - pd.Timedelta(days=1)
        
        # Get values
        small_start = get_input_at_time(small_df, start_time)
        big_start = get_input_at_time(big_df, start_time)
        small_report = get_input_at_time(small_df, report_time)
        big_report = get_input_at_time(big_df, report_time)
        
        return small_start, big_start, small_report, big_report
        
    except Exception:
        return None, None, None, None

def calculate_wasp_datetime(report_time, weeks_back, start_hour):
    """Calculate WASP datetime - SIMPLIFIED VERSION"""
    try:
        report_dt = safe_to_datetime(report_time)
        if pd.isna(report_dt):
            return None
            
        # Find days since last Sunday (Monday=0, Sunday=6)
        days_since_sunday = (report_dt.weekday() + 1) % 7
        
        # Go back to target Sunday
        target_sunday = report_dt - timedelta(days=days_since_sunday + 7 * (weeks_back - 1))
        wasp_dt = target_sunday.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        
        return ensure_timezone_naive(wasp_dt)
    except Exception:
        return None

def calculate_macedonia_datetime(report_time, months_back, start_hour):
    """Calculate Macedonia datetime - SIMPLIFIED VERSION"""
    try:
        report_dt = safe_to_datetime(report_time)
        if pd.isna(report_dt):
            return None
            
        # Start with first day of current month
        macedonia_dt = report_dt.replace(day=1, hour=start_hour, minute=0, second=0, microsecond=0)
        
        # Go back months
        for _ in range(months_back - 1):
            if macedonia_dt.month == 1:
                macedonia_dt = macedonia_dt.replace(year=macedonia_dt.year - 1, month=12)
            else:
                macedonia_dt = macedonia_dt.replace(month=macedonia_dt.month - 1)
        
        return ensure_timezone_naive(macedonia_dt)
    except Exception:
        return None

def find_new_data_changes(small_df, report_time, origin_name, scope_days=20):
    """Find new data changes - SIMPLIFIED VERSION"""
    try:
        if small_df is None or small_df.empty:
            return []
            
        report_clean = safe_to_datetime(report_time)
        if pd.isna(report_clean):
            return []

        # Look for H/L/C columns
        h_col = f"{origin_name} H"
        l_col = f"{origin_name} L" 
        c_col = f"{origin_name} C"
        
        # Handle bracket variations
        if '[' in origin_name and ']' in origin_name:
            base_name = origin_name[:origin_name.find('[')]
            bracket_part = origin_name[origin_name.find('['):]
            alt_h_col = f"{base_name} H{bracket_part}"
            alt_l_col = f"{base_name} L{bracket_part}"
            alt_c_col = f"{base_name} C{bracket_part}"
            
            if all(col in small_df.columns for col in [alt_h_col, alt_l_col, alt_c_col]):
                h_col, l_col, c_col = alt_h_col, alt_l_col, alt_c_col

        if not all(col in small_df.columns for col in [h_col, l_col, c_col]):
            return []

        # Process dataframe
        df_copy = small_df.copy()
        df_copy['time_clean'] = safe_to_datetime(df_copy['time'])
        df_copy['time_clean'] = ensure_timezone_naive(df_copy['time_clean'])
        
        # Filter by scope
        scope_start = report_clean - timedelta(days=scope_days)
        mask = (df_copy['time_clean'] >= scope_start) & (df_copy['time_clean'] <= report_clean) & df_copy['time_clean'].notna()
        scoped_df = df_copy[mask].sort_values('time_clean', ascending=True).reset_index(drop=True)

        if scoped_df.empty:
            return []

        # Find changes
        changes = []
        prev_h = prev_l = prev_c = None
        
        for _, row in scoped_df.iterrows():
            h, l, c = row[h_col], row[l_col], row[c_col]
            
            if pd.notna(h) and pd.notna(l) and pd.notna(c):
                h, l, c = float(h), float(l), float(c)
                
                if prev_h is None or h != prev_h or l != prev_l or c != prev_c:
                    changes.append({
                        'H': h, 'L': l, 'C': c,
                        'datetime': row['time_clean'],
                        'origin': origin_name
                    })
                    prev_h, prev_l, prev_c = h, l, c

        return changes

    except Exception:
        return []

def find_most_current_data(small_df, report_time, origin_name, scope_days=20):
    """Find most current data - SIMPLIFIED VERSION"""
    try:
        if small_df is None or small_df.empty:
            return None
            
        report_clean = safe_to_datetime(report_time)
        if pd.isna(report_clean):
            return None

        # Look for H/L/C columns
        h_col = f"{origin_name} H"
        l_col = f"{origin_name} L" 
        c_col = f"{origin_name} C"
        
        # Handle bracket variations
        if '[' in origin_name and ']' in origin_name:
            base_name = origin_name[:origin_name.find('[')]
            bracket_part = origin_name[origin_name.find('['):]
            alt_h_col = f"{base_name} H{bracket_part}"
            alt_l_col = f"{base_name} L{bracket_part}"
            alt_c_col = f"{base_name} C{bracket_part}"
            
            if all(col in small_df.columns for col in [alt_h_col, alt_l_col, alt_c_col]):
                h_col, l_col, c_col = alt_h_col, alt_l_col, alt_c_col

        if not all(col in small_df.columns for col in [h_col, l_col, c_col]):
            return None

        # Process dataframe
        df_copy = small_df.copy()
        df_copy['time_clean'] = safe_to_datetime(df_copy['time'])
        df_copy['time_clean'] = ensure_timezone_naive(df_copy['time_clean'])
        
        # Filter and sort
        mask = (df_copy['time_clean'] <= report_clean) & df_copy['time_clean'].notna()
        filtered_df = df_copy[mask].sort_values('time_clean', ascending=False)

        if filtered_df.empty:
            return None

        # Find first valid row
        for _, row in filtered_df.iterrows():
            h, l, c = row[h_col], row[l_col], row[c_col]
            if pd.notna(h) and pd.notna(l) and pd.notna(c):
                return {
                    'H': float(h),
                    'L': float(l),
                    'C': float(c),
                    'datetime': row['time_clean'],
                    'origin': origin_name
                }

        return None

    except Exception:
        return None

def calculate_raw_m_values(hlc_data, range_low, range_high):
    """Calculate raw M values - UNCHANGED"""
    try:
        H = hlc_data['H']
        L = hlc_data['L']
        C = hlc_data['C']

        avg = (H + L + C) / 3
        spread = H - L

        if spread == 0:
            return None

        raw_m_low = (range_low - avg) / spread
        raw_m_high = (range_high - avg) / spread

        return {
            'raw_m_low': raw_m_low,
            'raw_m_high': raw_m_high,
            'avg': avg,
            'spread': spread
        }

    except Exception:
        return None

def find_valid_m_values(measurement_df, raw_m_low, raw_m_high, hlc_data, range_low, range_high, is_high_range=False, data_source="Unknown", report_time=None, small_df=None, big_df=None, batch_inputs=None):
    """Find valid M values - SIMPLIFIED VERSION"""
    try:
        valid_entries = []
        valid_m_values = []

        # Find M value column
        m_value_col = None
        for col in ['M value', 'M Value', 'M_Value', 'M_value', 'm value', 'm_value']:
            if col in measurement_df.columns:
                m_value_col = col
                break

        if m_value_col is None:
            return {'valid_entries': [], 'valid_m_list': []}

        m_values = measurement_df[m_value_col].dropna().unique()

        # Process each M value
        for m_val in m_values:
            try:
                m_float = float(m_val)

                # Check if within raw M range
                if raw_m_low <= m_float <= raw_m_high:
                    output = hlc_data['avg'] + m_float * hlc_data['spread']
                    valid_m_values.append(m_float)
                    
                    matching_rows = measurement_df[measurement_df[m_value_col] == m_val]

                    for _, row in matching_rows.iterrows():
                        # Calculate zone
                        if range_low <= output <= range_high:
                            if is_high_range:
                                distance = range_high - output
                            else:
                                distance = output - range_low
                            
                            if distance <= 6:
                                zone_value = "0 to 6"
                            elif distance <= 12:
                                zone_value = "6 to 12"
                            elif distance <= 18:
                                zone_value = "12 to 18"
                            else:
                                zone_value = "18 to 24"
                        else:
                            zone_value = "Out of Range"

                        # Format arrival time
                        try:
                            arrival_dt = hlc_data['datetime']
                            if pd.notna(arrival_dt):
                                day_abbrev = arrival_dt.strftime('%a')
                                arrival_excel = arrival_dt.strftime('%Y-%m-%d %H:%M')
                                
                                # Calculate day index
                                try:
                                    from a_helpers import get_day_index
                                    day_index = get_day_index(arrival_dt, report_time, 18)
                                except:
                                    day_index = "[0]"
                            else:
                                day_abbrev = ""
                                arrival_excel = ""
                                day_index = "[0]"
                        except:
                            day_abbrev = ""
                            arrival_excel = str(hlc_data.get('datetime', ''))
                            day_index = "[0]"

                        # Determine feed type
                        feed_type = "Small" if data_source == "Small CSV" else "Big"

                        # Use batch input values
                        if batch_inputs is not None:
                            small_start, big_start, small_report, big_report = batch_inputs
                            
                            if feed_type == "Small":
                                input_18 = small_start if small_start is not None else 0
                                input_report = small_report if small_report is not None else 0
                            else:
                                input_18 = big_start if big_start is not None else 0
                                input_report = big_report if big_report is not None else 0
                        else:
                            input_18 = 0
                            input_report = 0

                        # Calculate input at arrival
                        if feed_type == "Small":
                            input_arrival = get_input_at_time(small_df, arrival_dt)
                        else:
                            input_arrival = get_input_at_time(big_df, arrival_dt)
                        
                        if input_arrival is None:
                            input_arrival = 0

                        valid_entries.append({
                            'Feed': feed_type,
                            'ddd': day_abbrev,
                            'Arrival': arrival_excel,
                            'Day': day_index,
                            'Origin': hlc_data['origin'],
                            'M Name': row.get('M Name', row.get('m name', f"M{m_val}")),
                            'M #': row.get('M #', row.get('m #', m_val)),
                            'M Value': m_val,
                            'R #': row.get('R #', row.get('r #', '')),
                            'Tag': row.get('Tag', row.get('tag', '')),
                            'Family': row.get('Family', row.get('family', '')),
                            'Input @ 18:00': input_18,
                            'Diff @ 18:00': output - input_18,
                            'Input @ Arrival': input_arrival,
                            'Diff @ Arrival': output - input_arrival,
                            'Input @ Report': input_report,
                            'Diff @ Report': output - input_report,
                            'Output': output,
                            'Range': f"{range_low:.1f}-{range_high:.1f}",
                            'Zone': zone_value
                        })

            except (ValueError, TypeError):
                continue

        return {'valid_entries': valid_entries, 'valid_m_list': valid_m_values}

    except Exception:
        return {'valid_entries': [], 'valid_m_list': []}

def process_custom_ranges_advanced(measurement_df, small_df, report_time, custom_ranges, scope_days=20, big_df=None, run_model_g=False):
    """Process custom ranges - SIMPLIFIED VERSION"""
    try:
        all_valid_entries = []
        
        # Calculate batch inputs once
        batch_inputs = get_input_values_batch(small_df, big_df, report_time, 18)
        
        # Prepare data sources
        data_sources = []
        if small_df is not None and not small_df.empty:
            data_sources.append((small_df.copy(), "Small CSV"))
        if big_df is not None and not big_df.empty:
            data_sources.append((big_df.copy(), "Big CSV"))

        # Get origins
        all_origins = set()
        for hlc_df, _ in data_sources:
            for col in hlc_df.columns:
                if col.endswith(' H'):
                    all_origins.add(col[:-2])

        # Add WASP and Macedonia variants
        for variant in ['WASP-12b', 'WASP-12b[1]', 'WASP-12b[2]', 'Macedonia', 'Macedonia[1]', 'Macedonia[2]']:
            all_origins.add(variant)

        origins = list(all_origins)

        # Process each range
        for range_name, config in custom_ranges.items():
            if not config.get('enabled', False) or config.get('value', 0) == 0:
                continue

            range_value = config['value']
            if range_name.startswith('High'):
                range_low, range_high = range_value - 24, range_value
                is_high_range = True
            else:
                range_low, range_high = range_value, range_value + 24
                is_high_range = False

            # Process each data source and origin
            for hlc_df, data_source in data_sources:
                for origin in origins:
                    # Handle special origins
                    if origin.lower().startswith('wasp-12b') or origin.lower() == 'wasp':
                        weeks_back = 1  # Default
                        if '[1]' in origin:
                            weeks_back = 2
                        elif '[2]' in origin:
                            weeks_back = 3
                        
                        wasp_dt = calculate_wasp_datetime(report_time, weeks_back, 18)
                        if wasp_dt is None:
                            continue
                            
                        current_data = find_most_current_data(hlc_df, report_time, origin, scope_days)
                        if current_data:
                            current_data['datetime'] = wasp_dt
                            hlc_list = [current_data]
                        else:
                            hlc_list = []

                    elif origin.lower().startswith('macedonia'):
                        months_back = 1  # Default
                        if '[1]' in origin:
                            months_back = 2
                        elif '[2]' in origin:
                            months_back = 3
                        
                        macedonia_dt = calculate_macedonia_datetime(report_time, months_back, 18)
                        if macedonia_dt is None:
                            continue
                            
                        current_data = find_most_current_data(hlc_df, report_time, origin, scope_days)
                        if current_data:
                            current_data['datetime'] = macedonia_dt
                            hlc_list = [current_data]
                        else:
                            hlc_list = []
                    else:
                        # Regular origins
                        hlc_list = find_new_data_changes(hlc_df, report_time, origin, scope_days)

                    # Process each HLC entry
                    for hlc_data in hlc_list:
                        calc = calculate_raw_m_values(hlc_data, range_low, range_high)
                        if not calc:
                            continue

                        enhanced = hlc_data.copy()
                        enhanced.update(calc)

                        validation = find_valid_m_values(
                            measurement_df, calc['raw_m_low'], calc['raw_m_high'],
                            enhanced, range_low, range_high, is_high_range,
                            data_source, report_time, small_df, big_df, batch_inputs
                        )

                        all_valid_entries.extend(validation.get('valid_entries', []))

        return all_valid_entries

    except Exception as e:
        st.error(f"Error in process_custom_ranges_advanced: {e}")
        return []

def process_full_range_advanced(measurement_df, small_df, report_time, center, window_radius, scope_days=20, big_df=None, run_model_g=False):
    """Process full range - SIMPLIFIED VERSION"""
    try:
        lo = center - window_radius
        hi = center + window_radius
        
        all_valid_entries = []
        
        # Calculate batch inputs once
        batch_inputs = get_input_values_batch(small_df, big_df, report_time, 18)
        
        # Prepare data sources
        data_sources = []
        if small_df is not None and not small_df.empty:
            data_sources.append((small_df.copy(), "Small CSV"))
        if big_df is not None and not big_df.empty:
            data_sources.append((big_df.copy(), "Big CSV"))

        # Get origins
        all_origins = set()
        for hlc_df, _ in data_sources:
            for col in hlc_df.columns:
                if col.endswith(" H"):
                    all_origins.add(col[:-2])

        # Add WASP and Macedonia variants
        for variant in ['WASP-12b', 'WASP-12b[1]', 'WASP-12b[2]', 'Macedonia', 'Macedonia[1]', 'Macedonia[2]']:
            all_origins.add(variant)

        origins = list(all_origins)

        # Process each data source and origin
        for hlc_df, data_source in data_sources:
            for origin in origins:
                # Handle special origins (same logic as custom ranges)
                if origin.lower().startswith('wasp-12b') or origin.lower() == 'wasp':
                    weeks_back = 1  # Default
                    if '[1]' in origin:
                        weeks_back = 2
                    elif '[2]' in origin:
                        weeks_back = 3
                    
                    wasp_dt = calculate_wasp_datetime(report_time, weeks_back, 18)
                    if wasp_dt is None:
                        continue
                        
                    current_data = find_most_current_data(hlc_df, report_time, origin, scope_days)
                    if current_data:
                        current_data['datetime'] = wasp_dt
                        hlc_list = [current_data]
                    else:
                        hlc_list = []

                elif origin.lower().startswith('macedonia'):
                    months_back = 1  # Default
                    if '[1]' in origin:
                        months_back = 2
                    elif '[2]' in origin:
                        months_back = 3
                    
                    macedonia_dt = calculate_macedonia_datetime(report_time, months_back, 18)
                    if macedonia_dt is None:
                        continue
                        
                    current_data = find_most_current_data(hlc_df, report_time, origin, scope_days)
                    if current_data:
                        current_data['datetime'] = macedonia_dt
                        hlc_list = [current_data]
                    else:
                        hlc_list = []
                        
                else:
                    # Regular origins
                    hlc_list = find_new_data_changes(hlc_df, report_time, origin, scope_days)

                # Process each HLC entry
                for hlc_data in hlc_list:
                    calc = calculate_raw_m_values(hlc_data, lo, hi)
                    if not calc:
                        continue

                    enhanced = hlc_data.copy()
                    enhanced.update(calc)

                    validation = find_valid_m_values(
                        measurement_df, calc['raw_m_low'], calc['raw_m_high'],
                        enhanced, lo, hi, False, data_source, report_time, 
                        small_df, big_df, batch_inputs
                    )

                    all_valid_entries.extend(validation.get('valid_entries', []))

        return all_valid_entries

    except Exception as e:
        st.error(f"Error in process_full_range_advanced: {e}")
        return []

def apply_custom_ranges_advanced(df, small_df, report_time, high1, high2, low1, low2, use_high1, use_high2, use_low1, use_low2, big_df=None, run_model_g=False):
    """Apply custom ranges - SIMPLIFIED VERSION"""
    try:
        # Prepare ranges
        custom_ranges = {}
        if use_high1 and high1 > 0:
            custom_ranges['High 1'] = {'enabled': True, 'value': high1}
        if use_high2 and high2 > 0:
            custom_ranges['High 2'] = {'enabled': True, 'value': high2}
        if use_low1 and low1 > 0:
            custom_ranges['Low 1'] = {'enabled': True, 'value': low1}
        if use_low2 and low2 > 0:
            custom_ranges['Low 2'] = {'enabled': True, 'value': low2}

        if not custom_ranges:
            return pd.DataFrame()

        # Process ranges
        valid_entries = process_custom_ranges_advanced(df, small_df, report_time, custom_ranges, big_df=big_df, run_model_g=run_model_g)

        if not valid_entries:
            return pd.DataFrame()

        return pd.DataFrame(valid_entries)
        
    except Exception as e:
        st.error(f"Error in apply_custom_ranges_advanced: {e}")
        return pd.DataFrame()

def apply_full_range_advanced(df, small_df, report_time, window_radius, day_start_hour=18, input_value_at_start=None, big_df=None, run_model_g=False):
    """Apply full range - SIMPLIFIED VERSION"""
    try:
        # Determine center
        center = input_value_at_start
        if center is None or pd.isna(center):
            # Try to derive from small_df
            try:
                from a_helpers import clean_timestamp
                sdf = small_df.copy()
                sdf['time'] = sdf['time'].apply(clean_timestamp)
                sdf = sdf[sdf['time'] <= report_time]

                if not sdf.empty:
                    base = dt.datetime(report_time.year, report_time.month, report_time.day, day_start_hour, 0, 0)
                    if report_time < base:
                        base = base - dt.timedelta(days=1)

                    center_row = sdf[sdf['time'] == base]
                    if not center_row.empty:
                        center_row = center_row.iloc[-1]
                    else:
                        center_row = sdf.iloc[-1]

                    for col in ['Open', 'open', 'close']:
                        if col in center_row.index and pd.notna(center_row[col]):
                            center = float(center_row[col])
                            break
            except:
                center = None

        if center is None:
            st.error("Cannot determine center for Full Range")
            return pd.DataFrame()

        # Process
        valid_entries = process_full_range_advanced(
            measurement_df=df, small_df=small_df, report_time=report_time,
            center=center, window_radius=window_radius, scope_days=20,
            big_df=big_df, run_model_g=run_model_g
        )

        if not valid_entries:
            return pd.DataFrame()

        # Convert to DataFrame and clean up
        out_df = pd.DataFrame(valid_entries)
        out_df = out_df.drop(columns=['Range', 'Zone'], errors='ignore')

        # Order columns consistently
        preferred_cols = [
            'Feed', 'ddd', 'Arrival', 'Day', 'Origin',
            'M Name', 'M #', 'M Value', 'R #', 'Tag', 'Family',
            'Input @ 18:00', 'Diff @ 18:00', 'Input @ Arrival', 'Diff @ Arrival',
            'Input @ Report', 'Diff @ Report', 'Output'
        ]
        ordered = [c for c in preferred_cols if c in out_df.columns]
        remaining = [c for c in out_df.columns if c not in ordered]
        out_df = out_df[ordered + remaining]

        st.success(f"Full Range (Advanced): {len(out_df)} entries processed")
        return out_df
        
    except Exception as e:
        st.error(f"Error in apply_full_range_advanced: {e}")
        return pd.DataFrame()
