"""
Optimized Custom Range Calculator for Market Data Analysis
Implements batch input value calculation for improved performance.
"""

import datetime as dt
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np

def safe_to_datetime(series_or_value, errors='coerce'):
    """
    Safely convert to datetime with proper error handling
    """
    try:
        if isinstance(series_or_value, pd.Series):
            # First convert to string to handle mixed types
            str_series = series_or_value.astype(str)
            # Clean timezone info using string operations
            clean_series = str_series.str.replace(r'[+-]\d{2}:?\d{2}$', '', regex=True)
            clean_series = clean_series.str.replace('T', ' ')
            return pd.to_datetime(clean_series, errors=errors)
        else:
            # Single value
            if pd.isna(series_or_value):
                return pd.NaT
            str_value = str(series_or_value)
            # Clean timezone info
            import re
            clean_value = re.sub(r'[+-]\d{2}:?\d{2}$', '', str_value)
            clean_value = clean_value.replace('T', ' ')
            return pd.to_datetime(clean_value, errors=errors)
    except Exception as e:
        st.warning(f"Datetime conversion error: {e}")
        if isinstance(series_or_value, pd.Series):
            return pd.Series([pd.NaT] * len(series_or_value))
        else:
            return pd.NaT

def ensure_timezone_naive(dt_value):
    """
    Ensure datetime is timezone naive
    """
    try:
        if pd.isna(dt_value):
            return dt_value
        
        if isinstance(dt_value, pd.Series):
            # For series, check if it's datetime type first
            if not pd.api.types.is_datetime64_any_dtype(dt_value):
                dt_value = safe_to_datetime(dt_value)
            
            # Only use .dt accessor if we have datetime type
            if pd.api.types.is_datetime64_any_dtype(dt_value):
                return dt_value.dt.tz_localize(None) if dt_value.dt.tz is not None else dt_value
            else:
                return dt_value
        else:
            # Single value
            if hasattr(dt_value, 'tz') and dt_value.tz is not None:
                return dt_value.replace(tzinfo=None)
            return dt_value
    except Exception as e:
        st.warning(f"Timezone conversion error: {e}")
        return dt_value

def get_input_values_batch(small_df, big_df, report_time, start_hour=18):
    """
    Efficiently calculate input values for both small and big feeds at once.
    Fixed version with robust datetime handling.
    """
    def get_single_input_value(df, target_time):
        """Helper to get input value from a single DataFrame at target time"""
        if df is None or target_time is None or df.empty:
            return None
        
        try:
            # Ensure we have required columns
            if 'time' not in df.columns or 'open' not in df.columns:
                return None
            
            # Make a copy to avoid modifying original
            df_copy = df.copy()
            
            # Safe datetime conversion
            df_copy["time"] = safe_to_datetime(df_copy["time"])
            
            # Check if conversion was successful
            if not pd.api.types.is_datetime64_any_dtype(df_copy["time"]):
                st.warning("Could not convert time column to datetime")
                return None
            
            # Remove timezone info safely
            df_copy["time"] = ensure_timezone_naive(df_copy["time"])
            target_time_naive = ensure_timezone_naive(pd.to_datetime(target_time))
            
            # First try exact match
            exact_match = df_copy[df_copy["time"] == target_time_naive]
            if not exact_match.empty:
                return exact_match.iloc[-1]["open"]
            
            # If no exact match, find closest time
            df_copy["time_diff"] = abs(df_copy["time"] - target_time_naive)
            closest_row = df_copy.loc[df_copy["time_diff"].idxmin()]
            return closest_row["open"]
            
        except Exception as e:
            st.warning(f"Error getting input value: {e}")
            return None
    
    try:
        # Calculate start time
        report_time_naive = ensure_timezone_naive(pd.to_datetime(report_time))
        start_time = report_time_naive.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        
        # If report time is before start_hour on the same day, go back to previous day
        if report_time_naive.hour < start_hour:
            start_time = start_time - pd.Timedelta(days=1)
        
        # Get all four values in one pass
        small_input_at_start = get_single_input_value(small_df, start_time)
        big_input_at_start = get_single_input_value(big_df, start_time)
        small_input_at_report = get_single_input_value(small_df, report_time)
        big_input_at_report = get_single_input_value(big_df, report_time)
        
        return small_input_at_start, big_input_at_start, small_input_at_report, big_input_at_report
        
    except Exception as e:
        st.error(f"Error in batch input calculation: {e}")
        return None, None, None, None

def find_new_data_changes(small_df, report_time, origin_name, scope_days=20):
    """
    Find the first time new data appears for an origin by detecting changes in H/L/C values.
    Fixed version with robust datetime handling.
    """
    try:
        # Convert report_time safely
        if isinstance(report_time, str):
            report_time = safe_to_datetime(report_time)

        # Look for columns ending with H, L, C for this origin
        h_col = f"{origin_name} H"
        l_col = f"{origin_name} L" 
        c_col = f"{origin_name} C"
        
        # Alternative format for bracket variations
        if '[' in origin_name and ']' in origin_name:
            base_name = origin_name[:origin_name.find('[')]
            bracket_part = origin_name[origin_name.find('['):]
            alt_h_col = f"{base_name} H{bracket_part}"
            alt_l_col = f"{base_name} L{bracket_part}"
            alt_c_col = f"{base_name} C{bracket_part}"
            
            if all(col in small_df.columns for col in [alt_h_col, alt_l_col, alt_c_col]):
                h_col, l_col, c_col = alt_h_col, alt_l_col, alt_c_col

        # Check if these columns exist
        if not all(col in small_df.columns for col in [h_col, l_col, c_col]):
            return []

        # Safe datetime conversion for the dataframe
        small_df_copy = small_df.copy()
        if 'time' in small_df_copy.columns:
            small_df_copy['time_dt'] = safe_to_datetime(small_df_copy['time'])
        else:
            small_df_copy['time_dt'] = safe_to_datetime(small_df_copy.iloc[:, 0])

        # Ensure report_time is timezone-naive datetime
        report_time = ensure_timezone_naive(safe_to_datetime(report_time))

        # Get data within scope and at or before report time
        scope_start = report_time - timedelta(days=scope_days)
        
        # Filter data safely
        valid_time_mask = pd.notna(small_df_copy['time_dt'])
        scoped_df = small_df_copy[
            valid_time_mask &
            (small_df_copy['time_dt'] >= scope_start) & 
            (small_df_copy['time_dt'] <= report_time)
        ].copy()

        if scoped_df.empty:
            return []

        # Sort by time ASCENDING (oldest first) to analyze changes properly
        scoped_df = scoped_df.sort_values('time_dt', ascending=True).reset_index(drop=True)

        new_data_entries = []
        previous_h, previous_l, previous_c = None, None, None

        # Go through rows chronologically to detect when data changes
        for idx, row in scoped_df.iterrows():
            h_val = row[h_col]
            l_val = row[l_col]
            c_val = row[c_col]

            # Skip if any values are null
            if pd.isna(h_val) or pd.isna(l_val) or pd.isna(c_val):
                continue

            current_h, current_l, current_c = float(h_val), float(l_val), float(c_val)

            # Check if this is new data (different from previous row)
            if (previous_h is None or 
                current_h != previous_h or 
                current_l != previous_l or 
                current_c != previous_c):

                # This is new data!
                new_data_entries.append({
                    'H': current_h,
                    'L': current_l,
                    'C': current_c,
                    'datetime': row['time_dt'],
                    'origin': origin_name
                })

                # Update previous values
                previous_h, previous_l, previous_c = current_h, current_l, current_c

        return new_data_entries

    except Exception as e:
        st.error(f"Error finding new data changes for {origin_name}: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return []

def find_most_current_data(small_df, report_time, origin_name, scope_days=20):
    """
    Find the most current data for an origin at report time.
    Fixed version with robust datetime handling.
    """
    try:
        # Convert report_time safely
        if isinstance(report_time, str):
            report_time = safe_to_datetime(report_time)

        # Look for columns ending with H, L, C for this origin
        h_col = f"{origin_name} H"
        l_col = f"{origin_name} L" 
        c_col = f"{origin_name} C"
        
        # Alternative format for bracket variations
        if '[' in origin_name and ']' in origin_name:
            base_name = origin_name[:origin_name.find('[')]
            bracket_part = origin_name[origin_name.find('['):]
            alt_h_col = f"{base_name} H{bracket_part}"
            alt_l_col = f"{base_name} L{bracket_part}"
            alt_c_col = f"{base_name} C{bracket_part}"
            
            if all(col in small_df.columns for col in [alt_h_col, alt_l_col, alt_c_col]):
                h_col, l_col, c_col = alt_h_col, alt_l_col, alt_c_col

        # Check if these columns exist
        if not all(col in small_df.columns for col in [h_col, l_col, c_col]):
            return None

        # Safe datetime conversion for the dataframe
        small_df_copy = small_df.copy()
        if 'time' in small_df_copy.columns:
            small_df_copy['time_dt'] = safe_to_datetime(small_df_copy['time'])
        else:
            small_df_copy['time_dt'] = safe_to_datetime(small_df_copy.iloc[:, 0])

        # Ensure report_time is timezone-naive datetime
        report_time = ensure_timezone_naive(safe_to_datetime(report_time))

        # Priority 1: Look for data from the same day as report_time
        report_date = report_time.date()
        
        # Filter for same day safely
        valid_time_mask = pd.notna(small_df_copy['time_dt'])
        if not valid_time_mask.any():
            return None
            
        # Only apply .dt.date on datetime columns
        same_day_mask = valid_time_mask & (small_df_copy['time_dt'].dt.date == report_date)
        same_day_df = small_df_copy[same_day_mask].copy()

        if not same_day_df.empty:
            # Sort by time descending to get most recent first
            same_day_df = same_day_df.sort_values('time_dt', ascending=False)
            # Find data at or before report_time on the same day
            valid_same_day = same_day_df[same_day_df['time_dt'] <= report_time]

            if not valid_same_day.empty:
                for _, row in valid_same_day.iterrows():
                    h_val = row[h_col]
                    l_val = row[l_col]
                    c_val = row[c_col]

                    if not (pd.isna(h_val) or pd.isna(l_val) or pd.isna(c_val)):
                        return {
                            'H': float(h_val),
                            'L': float(l_val),
                            'C': float(c_val),
                            'datetime': row['time_dt'],
                            'origin': origin_name
                        }

        # Priority 2: If no same-day data, look within scope_days (as fallback)
        scope_start = report_time - timedelta(days=scope_days)
        scoped_df = small_df_copy[
            valid_time_mask &
            (small_df_copy['time_dt'] >= scope_start) & 
            (small_df_copy['time_dt'] <= report_time)
        ].copy()

        if scoped_df.empty:
            return None

        # Sort by time descending to start with most recent
        scoped_df = scoped_df.sort_values('time_dt', ascending=False)

        # Find the most current data
        for i, row in scoped_df.iterrows():
            h_val = row[h_col]
            l_val = row[l_col]
            c_val = row[c_col]

            if not (pd.isna(h_val) or pd.isna(l_val) or pd.isna(c_val)):
                return {
                    'H': float(h_val),
                    'L': float(l_val),
                    'C': float(c_val),
                    'datetime': row['time_dt'],
                    'origin': origin_name
                }

        return None

    except Exception as e:
        st.error(f"Error finding current data for {origin_name}: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def calculate_raw_m_values(hlc_data, range_low, range_high):
    """
    Calculate raw M values for a price range.
    """
    try:
        H = hlc_data['H']
        L = hlc_data['L']
        C = hlc_data['C']

        # Calculate average
        avg = (H + L + C) / 3

        # Calculate spread
        spread = H - L

        if spread == 0:
            return None  # Cannot calculate with zero spread

        # Calculate raw M values
        raw_m_low = (range_low - avg) / spread
        raw_m_high = (range_high - avg) / spread

        return {
            'raw_m_low': raw_m_low,
            'raw_m_high': raw_m_high,
            'avg': avg,
            'spread': spread
        }

    except Exception as e:
        st.error(f"Error calculating raw M values: {e}")
        return None

def find_valid_m_values(measurement_df, raw_m_low, raw_m_high, hlc_data, range_low, range_high, is_high_range=False, data_source="Unknown", report_time=None, small_df=None, batch_inputs=None):
    """
    Find valid M values from measurement file within the raw M range.
    Uses batch input values for performance optimization.
    """
    try:
        valid_entries = []
        valid_m_values = []

        # Get M values from measurement file - use flexible column detection
        m_value_col = None
        for col in ['M value', 'M Value', 'M_Value', 'M_value', 'm value', 'm_value']:
            if col in measurement_df.columns:
                m_value_col = col
                break

        if m_value_col is None:
            return {
                'valid_entries': valid_entries,
                'valid_m_list': valid_m_values
            }

        m_values = measurement_df[m_value_col].dropna().unique()

        # Filter M values within raw M range
        for m_val in m_values:
            try:
                m_float = float(m_val)

                # Check if M value is within range
                if raw_m_low <= m_float <= raw_m_high:
                    # Calculate output for this M value
                    output = hlc_data['avg'] + m_float * hlc_data['spread']

                    # This M value is valid (within raw M range)
                    valid_m_values.append(m_float)
                    # Get all rows with this M value
                    matching_rows = measurement_df[measurement_df[m_value_col] == m_val]

                    for _, row in matching_rows.iterrows():
                        # Determine zone based on output value
                        zone_value = ""
                        if range_low <= output <= range_high:
                            if is_high_range:
                                # High range: zones measured from the top (range_high)
                                distance_from_top = range_high - output
                                if distance_from_top <= 6:
                                    zone_value = "0 to 6"
                                elif distance_from_top <= 12:
                                    zone_value = "6 to 12"
                                elif distance_from_top <= 18:
                                    zone_value = "12 to 18"
                                else:
                                    zone_value = "18 to 24"
                            else:
                                # Low range: zones measured from the bottom (range_low)
                                distance_from_bottom = output - range_low
                                if distance_from_bottom <= 6:
                                    zone_value = "0 to 6"
                                elif distance_from_bottom <= 12:
                                    zone_value = "6 to 12"
                                elif distance_from_bottom <= 18:
                                    zone_value = "12 to 18"
                                else:
                                    zone_value = "18 to 24"
                        else:
                            zone_value = "Out of Range"

                        # Format arrival time from H/L/C data
                        try:
                            arrival_dt = hlc_data['datetime']
                            day_abbrev = arrival_dt.strftime('%a')
                            arrival_excel = arrival_dt.strftime('%Y-%m-%d %H:%M')
                            
                            # Calculate day index
                            try:
                                from a_helpers import get_day_index
                                if isinstance(arrival_dt, str):
                                    arrival_dt = pd.to_datetime(arrival_dt)
                                day_index = get_day_index(arrival_dt, report_time, 18)
                            except Exception:
                                day_index = "[0]"
                        except:
                            day_abbrev = ""
                            arrival_excel = str(hlc_data['datetime'])
                            day_index = "[0]"

                        # Determine feed type from data source name
                        feed_type = "Small" if data_source == "Small CSV" else "Big"

                        # Use batch input values for performance optimization
                        if batch_inputs is not None:
                            small_input_at_start, big_input_at_start, small_input_at_report, big_input_at_report = batch_inputs
                            
                            if feed_type == "Small":
                                input_18 = small_input_at_start if small_input_at_start is not None else 0
                                input_report = small_input_at_report if small_input_at_report is not None else 0
                            else:
                                input_18 = big_input_at_start if big_input_at_start is not None else 0
                                input_report = big_input_at_report if big_input_at_report is not None else 0
                        else:
                            # Fallback to individual lookups (slower)
                            try:
                                from a_helpers import get_input_at_day_start, get_input_at_time
                                
                                if small_df is not None and not small_df.empty and 'time' in small_df.columns and 'open' in small_df.columns:
                                    input_18 = get_input_at_day_start(small_df, report_time, 18)
                                    input_report = get_input_at_time(small_df, report_time)
                                else:
                                    input_18 = 0
                                    input_report = 0
                            except Exception:
                                input_18 = 0
                                input_report = 0

                        # Calculate input at arrival - this still needs individual lookup
                        try:
                            from a_helpers import get_input_at_time
                            if feed_type == "Small" and small_df is not None:
                                input_arrival = get_input_at_time(small_df, arrival_dt)
                            else:
                                input_arrival = get_input_at_time(small_df, arrival_dt)  # fallback to small
                        except Exception:
                            input_arrival = 0

                        if input_18 is None:
                            input_18 = 0
                        if input_arrival is None:
                            input_arrival = 0
                        if input_report is None:
                            input_report = 0

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

        return {
            'valid_entries': valid_entries,
            'valid_m_list': valid_m_values
        }

    except Exception as e:
        st.error(f"Error finding valid M values: {e}")
        return {
            'valid_entries': [],
            'valid_m_list': []
        }

def process_custom_ranges_advanced(measurement_df, small_df, report_time, custom_ranges, scope_days=20, big_df=None, run_model_g=False):
    """
    Process custom ranges using advanced H/L/C calculation method with batch optimization.
    Fixed version with robust datetime handling.
    """
    all_valid_entries = []
    processing_summary = []

    try:
        # PERFORMANCE OPTIMIZATION: Calculate batch input values once upfront
        st.info("Calculating batch input values for performance optimization...")
        batch_inputs = get_input_values_batch(small_df, big_df, report_time, 18)
        small_input_at_start, big_input_at_start, small_input_at_report, big_input_at_report = batch_inputs
        
        st.info(f"Batch inputs calculated - Small @ 18:00: {small_input_at_start}, Big @ 18:00: {big_input_at_start}")

        # Process both Big and Small feeds if available
        data_sources = []
        if small_df is not None and not small_df.empty:
            data_sources.append((small_df.copy(), "Small CSV"))
        if big_df is not None and not big_df.empty:
            data_sources.append((big_df.copy(), "Big CSV"))

        # Get all unique origins from both data sources
        all_origins = set()
        for hlc_df, data_source in data_sources:
            for col in hlc_df.columns:
                if col.endswith(' H'):
                    origin_name = col[:-2]
                    all_origins.add(origin_name)

        # Add variants
        wasp_variants = ['WASP-12b[1]', 'WASP-12b[2]']
        macedonia_variants = ['Macedonia[1]', 'Macedonia[2]']
        for variant in wasp_variants + macedonia_variants:
            all_origins.add(variant)

        origins = list(all_origins)
        st.info(f"Processing origins: {', '.join(origins)}")

        # Process each custom range
        for range_name, range_config in custom_ranges.items():
            if not range_config.get('enabled', False):
                continue

            range_value = range_config.get('value', 0)
            if range_value == 0:
                continue

            # Determine range bounds
            if range_name.startswith('High'):
                range_low = range_value - 24
                range_high = range_value
                is_high_range = True
            else:
                range_low = range_value
                range_high = range_value + 24
                is_high_range = False

            st.markdown(f"### Processing {range_name}: {range_low:.3f} to {range_high:.3f}")

            range_entries = []

            # Process each data source for this range
            for hlc_df, data_source in data_sources:
                st.markdown(f"#### {data_source} Feed")

                # Process each origin for this data source
                for origin in origins:
                    # Handle special origins (same logic as before)
                    if (origin.lower() == 'wasp-12b' or origin.lower() == 'wasp' or 
                        'wasp-12b[1]' in origin.lower() or 'wasp-12b[2]' in origin.lower()):
                        
                        report_dt = safe_to_datetime(report_time)
                        if pd.isna(report_dt):
                            continue
                            
                        days_since_sunday = report_dt.weekday() + 1
                        if days_since_sunday == 7:
                            days_since_sunday = 0

                        wasp_datetime = report_dt - timedelta(days=days_since_sunday)
                        
                        if '[1]' in origin:
                            wasp_datetime = wasp_datetime - timedelta(weeks=1)
                        elif '[2]' in origin:
                            wasp_datetime = wasp_datetime - timedelta(weeks=2)
                        
                        wasp_datetime = wasp_datetime.replace(hour=18, minute=0, second=0, microsecond=0)
                        wasp_datetime = ensure_timezone_naive(wasp_datetime)

                        hlc_data_single = find_most_current_data(hlc_df, report_time, origin, scope_days)
                        if hlc_data_single:
                            hlc_data_single['datetime'] = wasp_datetime
                            hlc_data_single['origin'] = f"{origin}"
                            hlc_data_list = [hlc_data_single]
                        else:
                            hlc_data_list = []

                    elif (origin.lower() == 'macedonia' or 
                          'macedonia[1]' in origin.lower() or 'macedonia[2]' in origin.lower() or
                          origin.lower().startswith('macedonia')):
                        
                        report_dt = safe_to_datetime(report_time)
                        if pd.isna(report_dt):
                            continue
                            
                        macedonia_datetime = report_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                        
                        if '[1]' in origin:
                            if macedonia_datetime.month == 1:
                                macedonia_datetime = macedonia_datetime.replace(year=macedonia_datetime.year - 1, month=12)
                            else:
                                macedonia_datetime = macedonia_datetime.replace(month=macedonia_datetime.month - 1)
                        elif '[2]' in origin:
                            target_month = macedonia_datetime.month - 2
                            if target_month <= 0:
                                macedonia_datetime = macedonia_datetime.replace(year=macedonia_datetime.year - 1, month=target_month + 12)
                            else:
                                macedonia_datetime = macedonia_datetime.replace(month=target_month)
                        
                        macedonia_datetime = ensure_timezone_naive(macedonia_datetime)
                        
                        hlc_data_single = find_most_current_data(hlc_df, report_time, origin, scope_days)
                        if hlc_data_single:
                            hlc_data_single['datetime'] = macedonia_datetime
                            hlc_data_single['origin'] = f"{origin}"
                            hlc_data_list = [hlc_data_single]
                        else:
                            hlc_data_list = []

                    else:
                        # Regular origins - GET NEW DATA CHANGES ONLY
                        hlc_data_list = find_new_data_changes(hlc_df, report_time, origin, scope_days)

                    if not hlc_data_list:
                        continue

                    # Process each datetime entry for this origin
                    for hlc_data in hlc_data_list:
                        # Calculate raw M values
                        raw_m_calc = calculate_raw_m_values(hlc_data, range_low, range_high)
                        if not raw_m_calc:
                            continue

                        # Combine hlc_data with calculation results
                        enhanced_hlc_data = hlc_data.copy()
                        enhanced_hlc_data.update(raw_m_calc)

                        # Find valid M values with batch inputs for performance
                        validation_results = find_valid_m_values(
                            measurement_df, 
                            raw_m_calc['raw_m_low'], 
                            raw_m_calc['raw_m_high'],
                            enhanced_hlc_data,
                            range_low, 
                            range_high, 
                            is_high_range,
                            data_source,
                            report_time,
                            small_df,
                            batch_inputs  # Pass batch inputs for performance
                        )

                        valid_entries = validation_results['valid_entries']
                        valid_m_list = validation_results['valid_m_list']

                        range_entries.extend(valid_entries)

                        # Add to processing summary
                        if hasattr(hlc_data['datetime'], 'strftime'):
                            datetime_str = hlc_data['datetime'].strftime('%m/%d/%Y %H:%M')
                        else:
                            datetime_str = str(hlc_data['datetime'])

                        valid_m_count = len(valid_entries) if valid_entries else 0
                        valid_list_str = ', '.join([f'{m:.1f}' if isinstance(m, float) else str(m) for m in valid_m_list]) if valid_m_list else 'None'

                        processing_summary.append({
                            'Range': f"{range_low:.1f}-{range_high:.1f}",
                            'Feed': data_source.replace(' CSV', ''),
                            'DateTime': datetime_str,
                            'Origin': origin,
                            'H': hlc_data['H'],
                            'L': hlc_data['L'], 
                            'C': hlc_data['C'],
                            'Raw M Low': raw_m_calc['raw_m_low'] if raw_m_calc else 0,
                            'Raw M High': raw_m_calc['raw_m_high'] if raw_m_calc else 0,
                            'Valid M Values': valid_m_count,
                            'Valid list': valid_list_str
                        })

            all_valid_entries.extend(range_entries)
            st.info(f"{range_name}: Found {len(range_entries)} valid entries")

        # Display processing summary
        if processing_summary:
            st.markdown("### Processing Summary")
            summary_df = pd.DataFrame(processing_summary)
            st.dataframe(summary_df, use_container_width=True)

        return all_valid_entries

    except Exception as e:
        st.error(f"Error in process_custom_ranges_advanced: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return []

    # Run Model G detection on Grp 1a data if enabled
    if all_valid_entries and run_model_g:
        st.markdown("---")
        st.markdown("### Model G Detection on Grp 1a Data")

        try:
            from a_helpers import GROUP_1A_TRAVELERS
            custom_df = pd.DataFrame(all_valid_entries)
            grp_1a_mask = custom_df['M #'].isin(GROUP_1A_TRAVELERS)
            grp_1a_df = custom_df[grp_1a_mask].copy()

            if grp_1a_df.empty:
                st.info("No Group 1a entries found in custom range results for Model G detection")
            else:
                st.info(f"Running Model G detection on {len(grp_1a_df)} Group 1a entries")
                try:
                    from models_g_updated import run_model_g_detection
                except ImportError:
                    try:
                        from model_g import run_model_g_detection
                    except ImportError:
                        from model_g_detector import run_model_g_detection

                g_results = run_model_g_detection(grp_1a_df, report_time, key_suffix="_custom")

                if isinstance(g_results, dict) and 'success' in g_results:
                    if g_results['success']:
                        summary = g_results['summary']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("o1 (Today)", summary['total_o1'])
                        with col2:
                            st.metric("o2 (Other Day)", summary['total_o2'])
                        with col3:
                            st.metric("Total Sequences", summary['total_sequences'])

                        if not g_results['results_df'].empty:
                            st.markdown("#### Grp 1a Model G Results")
                            st.dataframe(g_results['results_df'], use_container_width=True)
                        else:
                            st.info("No Model G sequences detected in Grp 1a data")
                    else:
                        st.error(f"Model G detection error: {g_results['error']}")

        except ImportError as e:
            st.warning(f"Model G detection not available: {e}")
        except Exception as e:
            st.error(f"Model G detection error: {str(e)}")

    return all_valid_entries

def apply_custom_ranges_advanced(df, small_df, report_time, high1, high2, low1, low2, use_high1, use_high2, use_low1, use_low2, big_df=None, run_model_g=False):
    """
    Apply advanced custom ranges to dataframe with batch optimization.
    Fixed version with robust datetime handling.
    """
    try:
        st.info("Advanced Custom Range Processing with Batch Optimization Started")
        
        # Prepare custom ranges configuration
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

        # Process ranges using advanced method with batch optimization
        valid_entries = process_custom_ranges_advanced(df, small_df, report_time, custom_ranges, big_df=big_df, run_model_g=run_model_g)

        if not valid_entries:
            st.warning("No valid entries found using advanced custom range calculation")
            return pd.DataFrame()

        # Convert to dataframe
        filtered_df = pd.DataFrame(valid_entries)

        # Range and Zone columns are already calculated in find_valid_m_values
        return filtered_df
        
    except Exception as e:
        st.error(f"Error in apply_custom_ranges_advanced: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def process_full_range_advanced(measurement_df, small_df, report_time, center, window_radius, scope_days=20, big_df=None, run_model_g=False):
    """
    Advanced Full Range processing with batch optimization.
    Fixed version with robust datetime handling.
    """
    try:
        lo = center - window_radius
        hi = center + window_radius
        st.info(f"Full Range (Advanced) window: [{lo}, {hi}] around center={center}")

        all_valid_entries = []
        processing_summary = []

        # PERFORMANCE OPTIMIZATION: Calculate batch input values once upfront
        st.info("Calculating batch input values for Full Range processing...")
        batch_inputs = get_input_values_batch(small_df, big_df, report_time, 18)
        small_input_at_start, big_input_at_start, small_input_at_report, big_input_at_report = batch_inputs
        
        st.info(f"Batch inputs calculated - Small @ 18:00: {small_input_at_start}, Big @ 18:00: {big_input_at_start}")

        # Always include Small CSV if available; include Big CSV if present
        data_sources = []
        if small_df is not None and not small_df.empty:
            data_sources.append((small_df.copy(), "Small CSV"))
        if big_df is not None and not big_df.empty:
            data_sources.append((big_df.copy(), "Big CSV"))

        # Gather origins from both data sources
        all_origins = set()
        for hlc_df, _src in data_sources:
            for col in hlc_df.columns:
                if col.endswith(" H"):
                    all_origins.add(col[:-2])

        # Include special variants
        for variant in ['WASP-12b[1]', 'WASP-12b[2]', 'Macedonia[1]', 'Macedonia[2]']:
            all_origins.add(variant)

        origins = list(all_origins)
        if origins:
            st.info(f"Processing origins: {', '.join(origins)}")

        # Iterate the single full window across all sources/origins
        for hlc_df, data_source in data_sources:
            for origin in origins:

                # Special handling (same as custom path but with safe datetime)
                if (origin.lower() == 'wasp-12b' or origin.lower() == 'wasp' or
                    'wasp-12b[1]' in origin.lower() or 'wasp-12b[2]' in origin.lower()):
                    
                    report_dt = safe_to_datetime(report_time)
                    if pd.isna(report_dt):
                        continue
                        
                    days_since_sunday = report_dt.weekday() + 1
                    if days_since_sunday == 7:
                        days_since_sunday = 0
                    wasp_dt = report_dt - timedelta(days=days_since_sunday)
                    
                    if '[1]' in origin:
                        wasp_dt = wasp_dt - timedelta(weeks=1)
                    elif '[2]' in origin:
                        wasp_dt = wasp_dt - timedelta(weeks=2)
                    
                    wasp_dt = wasp_dt.replace(hour=18, minute=0, second=0, microsecond=0)
                    wasp_dt = ensure_timezone_naive(wasp_dt)

                    cur = find_most_current_data(hlc_df, report_time, origin, scope_days)
                    if cur:
                        cur['datetime'] = wasp_dt
                        cur['origin'] = f"{origin}"
                        hlc_data_list = [cur]
                    else:
                        hlc_data_list = []

                elif (origin.lower() == 'macedonia' or
                      'macedonia[1]' in origin.lower() or 'macedonia[2]' in origin.lower() or
                      origin.lower().startswith('macedonia')):
                    
                    report_dt = safe_to_datetime(report_time)
                    if pd.isna(report_dt):
                        continue
                        
                    macedonia_datetime = report_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                    
                    if '[1]' in origin:
                        if macedonia_datetime.month == 1:
                            macedonia_datetime = macedonia_datetime.replace(year=macedonia_datetime.year - 1, month=12)
                        else:
                            macedonia_datetime = macedonia_datetime.replace(month=macedonia_datetime.month - 1)
                    elif '[2]' in origin:
                        target_month = macedonia_datetime.month - 2
                        if target_month <= 0:
                            macedonia_datetime = macedonia_datetime.replace(year=macedonia_datetime.year - 1, month=target_month + 12)
                        else:
                            macedonia_datetime = macedonia_datetime.replace(month=target_month)
                    
                    macedonia_datetime = ensure_timezone_naive(macedonia_datetime)
                    
                    cur = find_most_current_data(hlc_df, report_time, origin, scope_days)
                    if cur:
                        cur['datetime'] = macedonia_datetime
                        cur['origin'] = f"{origin}"
                        hlc_data_list = [cur]
                    else:
                        hlc_data_list = []
                        
                else:
                    # Regular origins: NEW DATA CHANGES
                    hlc_data_list = find_new_data_changes(hlc_df, report_time, origin, scope_days)

                if not hlc_data_list:
                    continue

                # Process each datetime entry in this origin
                for hlc_data in hlc_data_list:
                    calc = calculate_raw_m_values(hlc_data, lo, hi)
                    if not calc:
                        continue

                    enhanced = hlc_data.copy()
                    enhanced.update(calc)

                    # Use batch inputs for performance
                    validation = find_valid_m_values(
                        measurement_df,
                        calc['raw_m_low'],
                        calc['raw_m_high'],
                        enhanced,
                        lo,
                        hi,
                        is_high_range=False,
                        data_source=data_source,
                        report_time=report_time,
                        small_df=small_df,
                        batch_inputs=batch_inputs  # Pass batch inputs for performance
                    )

                    valid_entries = validation.get('valid_entries', [])
                    all_valid_entries.extend(valid_entries)

                    # Optional processing summary
                    try:
                        if hasattr(hlc_data['datetime'], 'strftime'):
                            dt_str = hlc_data['datetime'].strftime('%m/%d/%Y %H:%M')
                        else:
                            dt_str = str(hlc_data['datetime'])
                    except:
                        dt_str = str(hlc_data.get('datetime', ''))

                    processing_summary.append({
                        'Range': f"{lo:.1f}-{hi:.1f}",
                        'Feed': data_source.replace(' CSV', ''),
                        'DateTime': dt_str,
                        'Origin': origin,
                        'H': hlc_data['H'],
                        'L': hlc_data['L'],
                        'C': hlc_data['C'],
                        'Raw M Low': calc['raw_m_low'],
                        'Raw M High': calc['raw_m_high'],
                        'Valid M Values': len(validation.get('valid_m_list', [])),
                        'Valid list': ', '.join([f"{m:.1f}" for m in validation.get('valid_m_list', [])]) or 'None'
                    })

        # Optional: show a compact summary
        if processing_summary:
            st.markdown("### Full Range Processing Summary")
            st.dataframe(pd.DataFrame(processing_summary), use_container_width=True)

        return all_valid_entries

    except Exception as e:
        st.error(f"Error in process_full_range_advanced: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return []


def apply_full_range_advanced(df, small_df, report_time, window_radius, day_start_hour=18, input_value_at_start=None, big_df=None, run_model_g=False):
    """
    Apply the advanced Full Range flow with batch optimization.
    Fixed version with robust datetime handling.
    """
    try:
        # Determine center
        center = None
        if input_value_at_start is not None and not pd.isna(input_value_at_start):
            center = float(input_value_at_start)
        else:
            # Derive from small_df
            try:
                sdf = small_df.copy()
                if 'time' in sdf.columns:
                    sdf['time'] = safe_to_datetime(sdf['time'])
                    if report_time is not None:
                        report_time_safe = safe_to_datetime(report_time)
                        if not pd.isna(report_time_safe):
                            sdf = sdf[sdf['time'] <= report_time_safe]

                if not sdf.empty:
                    report_time_safe = safe_to_datetime(report_time)
                    if not pd.isna(report_time_safe):
                        base = dt.datetime(report_time_safe.year, report_time_safe.month, report_time_safe.day, day_start_hour, 0, 0)
                        if report_time_safe < base:
                            base = base - dt.timedelta(days=1)

                        center_row = sdf[sdf['time'] == base]
                        if not center_row.empty:
                            center_row = center_row.iloc[-1]
                        else:
                            center_row = sdf.iloc[-1]

                        for cand in ('Open', 'open', 'close'):
                            if cand in center_row.index:
                                center = pd.to_numeric(pd.Series([center_row[cand]]), errors='coerce').iloc[0]
                                break
            except Exception as e:
                st.warning(f"Error deriving center from small_df: {e}")
                center = None

        if center is None or pd.isna(center):
            st.error("Full Range (Advanced): could not determine center. Provide input @ day start or ensure small feed has time/Open/close.")
            return pd.DataFrame()

        # Process using the fixed process_full_range_advanced function
        valid_entries = process_full_range_advanced(
            measurement_df=df,
            small_df=small_df,
            report_time=report_time,
            center=center,
            window_radius=window_radius,
            scope_days=20,
            big_df=big_df,
            run_model_g=run_model_g
        )

        if not valid_entries:
            st.warning("Full Range (Advanced): no valid entries found.")
            return pd.DataFrame()

        # Convert to DF and drop Range/Zone if present (Full Range doesn't need these)
        out_df = pd.DataFrame(valid_entries)
        out_df = out_df.drop(columns=['Range', 'Zone'], errors='ignore')

        # Order columns consistently
        preferred_cols = [
            'Feed','ddd','Arrival','Day','Origin',
            'M Name','M #','M Value','R #','Tag','Family',
            'Input @ 18:00','Diff @ 18:00','Input @ Arrival','Diff @ Arrival',
            'Input @ Report','Diff @ Report','Output'
        ]
        ordered = [c for c in preferred_cols if c in out_df.columns]
        remaining = [c for c in out_df.columns if c not in ordered]
        out_df = out_df[ordered + remaining]

        st.success(f"Full Range (Advanced): {len(out_df)} entries processed with batch optimization")
        return out_df
        
    except Exception as e:
        st.error(f"Error in apply_full_range_advanced: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()
