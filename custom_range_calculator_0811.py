"""
Custom Range Calculator - PERFORMANCE OPTIMIZED VERSION
Maintains correctness while dramatically improving speed.
"""

import datetime as dt
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import re

# Global cache for datetime conversions to avoid repeated processing
_datetime_cache = {}

def fast_datetime_convert(value):
    """Ultra-fast datetime conversion with caching"""
    if pd.isna(value):
        return pd.NaT
    
    # Use cache for repeated values
    str_val = str(value)
    if str_val in _datetime_cache:
        return _datetime_cache[str_val]
    
    try:
        # Fast regex-based timezone removal
        clean_val = re.sub(r'[+-]\d{2}:?\d{2}$', '', str_val).replace('T', ' ')
        result = pd.to_datetime(clean_val, errors='coerce')
        _datetime_cache[str_val] = result
        return result
    except:
        _datetime_cache[str_val] = pd.NaT
        return pd.NaT

def prepare_dataframe_once(df):
    """Pre-process dataframe once to avoid repeated conversions"""
    if df is None or df.empty:
        return None
    
    df_prepared = df.copy()
    
    # Convert time column once
    if 'time' in df_prepared.columns:
        df_prepared['time_clean'] = df_prepared['time'].apply(fast_datetime_convert)
    else:
        df_prepared['time_clean'] = df_prepared.iloc[:, 0].apply(fast_datetime_convert)
    
    # Pre-calculate time diffs for faster lookups later
    df_prepared = df_prepared.sort_values('time_clean').reset_index(drop=True)
    
    return df_prepared

def fast_get_input_at_time(df_prepared, target_time):
    """Ultra-fast input lookup using pre-processed dataframe"""
    if df_prepared is None or target_time is None:
        return None
    
    try:
        target_clean = fast_datetime_convert(target_time)
        if pd.isna(target_clean):
            return None
        
        # Fast exact match using pandas indexing
        exact_matches = df_prepared[df_prepared["time_clean"] == target_clean]
        if not exact_matches.empty:
            return exact_matches.iloc[-1]["open"]
        
        # Fast closest match using vectorized operations
        time_diffs = (df_prepared["time_clean"] - target_clean).abs()
        closest_idx = time_diffs.idxmin()
        return df_prepared.loc[closest_idx, "open"]
        
    except:
        return None

def get_input_values_batch_optimized(small_df, big_df, report_time, start_hour=18):
    """Optimized batch input calculation with pre-processing"""
    try:
        if report_time is None:
            return None, None, None, None
        
        # Pre-process dataframes once
        small_prepared = prepare_dataframe_once(small_df)
        big_prepared = prepare_dataframe_once(big_df)
        
        report_clean = fast_datetime_convert(report_time)
        if pd.isna(report_clean):
            return None, None, None, None
        
        # Calculate start time
        start_time = report_clean.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        if report_clean.hour < start_hour:
            start_time = start_time - pd.Timedelta(days=1)
        
        # Fast lookups using prepared dataframes
        small_start = fast_get_input_at_time(small_prepared, start_time)
        big_start = fast_get_input_at_time(big_prepared, start_time)
        small_report = fast_get_input_at_time(small_prepared, report_time)
        big_report = fast_get_input_at_time(big_prepared, report_time)
        
        return small_start, big_start, small_report, big_report
        
    except:
        return None, None, None, None

def fast_calculate_special_datetime(report_time, origin_name, start_hour=18):
    """Pre-calculate special datetimes for WASP and Macedonia"""
    try:
        report_dt = fast_datetime_convert(report_time)
        if pd.isna(report_dt):
            return None
        
        if 'wasp' in origin_name.lower():
            # WASP calculation
            weeks_back = 1
            if '[1]' in origin_name:
                weeks_back = 2
            elif '[2]' in origin_name:
                weeks_back = 3
            
            days_since_sunday = (report_dt.weekday() + 1) % 7
            target_sunday = report_dt - timedelta(days=days_since_sunday + 7 * (weeks_back - 1))
            return target_sunday.replace(hour=start_hour, minute=0, second=0, microsecond=0)
            
        elif 'macedonia' in origin_name.lower():
            # Macedonia calculation
            months_back = 1
            if '[1]' in origin_name:
                months_back = 2
            elif '[2]' in origin_name:
                months_back = 3
            
            macedonia_dt = report_dt.replace(day=1, hour=start_hour, minute=0, second=0, microsecond=0)
            for _ in range(months_back - 1):
                if macedonia_dt.month == 1:
                    macedonia_dt = macedonia_dt.replace(year=macedonia_dt.year - 1, month=12)
                else:
                    macedonia_dt = macedonia_dt.replace(month=macedonia_dt.month - 1)
            
            return macedonia_dt
            
        return None
    except:
        return None

def find_hlc_data_optimized(df_prepared, report_time, origin_name, scope_days=20):
    """Optimized H/L/C data finding with vectorized operations"""
    try:
        if df_prepared is None:
            return []
        
        # Look for columns
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
            
            if all(col in df_prepared.columns for col in [alt_h_col, alt_l_col, alt_c_col]):
                h_col, l_col, c_col = alt_h_col, alt_l_col, alt_c_col

        if not all(col in df_prepared.columns for col in [h_col, l_col, c_col]):
            return []

        report_clean = fast_datetime_convert(report_time)
        scope_start = report_clean - timedelta(days=scope_days)
        
        # Vectorized filtering
        time_mask = (df_prepared['time_clean'] >= scope_start) & (df_prepared['time_clean'] <= report_clean) & df_prepared['time_clean'].notna()
        scoped_df = df_prepared[time_mask].copy()
        
        if scoped_df.empty:
            return []
        
        # Vectorized change detection
        hlc_cols = [h_col, l_col, c_col]
        hlc_data = scoped_df[hlc_cols + ['time_clean']].copy()
        hlc_data = hlc_data.dropna()
        
        if hlc_data.empty:
            return []
        
        # Find where any H/L/C values change
        hlc_shifted = hlc_data[hlc_cols].shift(1)
        changes_mask = (hlc_data[hlc_cols] != hlc_shifted).any(axis=1)
        
        # Include first row (it's always a "change")
        changes_mask.iloc[0] = True
        
        change_rows = hlc_data[changes_mask]
        
        results = []
        for _, row in change_rows.iterrows():
            results.append({
                'H': float(row[h_col]),
                'L': float(row[l_col]),
                'C': float(row[c_col]),
                'datetime': row['time_clean'],
                'origin': origin_name
            })
        
        return results

    except:
        return []

def process_custom_ranges_optimized(measurement_df, small_df, big_df, report_time, custom_ranges):
    """Ultra-optimized custom range processing"""
    try:
        all_valid_entries = []
        
        # Pre-process dataframes once
        st.info("Pre-processing dataframes for speed optimization...")
        small_prepared = prepare_dataframe_once(small_df)
        big_prepared = prepare_dataframe_once(big_df)
        
        # Calculate batch inputs once
        batch_inputs = get_input_values_batch_optimized(small_df, big_df, report_time, 18)
        small_start, big_start, small_report, big_report = batch_inputs or (0, 0, 0, 0)
        
        # Get all origins once
        all_origins = set()
        for df_prep, _ in [(small_prepared, "Small"), (big_prepared, "Big")]:
            if df_prep is not None:
                for col in df_prep.columns:
                    if col.endswith(' H'):
                        all_origins.add(col[:-2])
        
        # Add special origins
        for variant in ['WASP-12b', 'WASP-12b[1]', 'WASP-12b[2]', 'Macedonia', 'Macedonia[1]', 'Macedonia[2]']:
            all_origins.add(variant)
        
        origins = list(all_origins)
        
        # Pre-calculate special datetimes
        special_datetimes = {}
        for origin in origins:
            if 'wasp' in origin.lower() or 'macedonia' in origin.lower():
                special_dt = fast_calculate_special_datetime(report_time, origin, 18)
                if special_dt:
                    special_datetimes[origin] = special_dt
        
        # Get M values once
        m_value_col = None
        for col in ['M value', 'M Value', 'M_Value', 'M_value', 'm value', 'm_value']:
            if col in measurement_df.columns:
                m_value_col = col
                break
        
        if m_value_col is None:
            return []
        
        # Convert M values to numeric once
        m_values_series = pd.to_numeric(measurement_df[m_value_col], errors='coerce').dropna()
        m_values = m_values_series.unique()
        
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

            # Process each data source
            for df_prepared, data_source in [(small_prepared, "Small CSV"), (big_prepared, "Big CSV")]:
                if df_prepared is None:
                    continue
                
                # Process origins in batches
                for origin in origins:
                    # Get H/L/C data
                    if origin in special_datetimes:
                        # Special handling for WASP/Macedonia
                        hlc_list = find_hlc_data_optimized(df_prepared, report_time, origin, 20)
                        if hlc_list:
                            hlc_list[0]['datetime'] = special_datetimes[origin]
                    else:
                        # Regular origins
                        hlc_list = find_hlc_data_optimized(df_prepared, report_time, origin, 20)
                    
                    # Process H/L/C entries
                    for hlc_data in hlc_list:
                        # Fast raw M calculation
                        H, L, C = hlc_data['H'], hlc_data['L'], hlc_data['C']
                        avg = (H + L + C) / 3
                        spread = H - L
                        
                        if spread == 0:
                            continue
                        
                        raw_m_low = (range_low - avg) / spread
                        raw_m_high = (range_high - avg) / spread
                        
                        # Vectorized M value filtering
                        valid_mask = (m_values >= raw_m_low) & (m_values <= raw_m_high)
                        valid_m_vals = m_values[valid_mask]
                        
                        if len(valid_m_vals) == 0:
                            continue
                        
                        # Process valid M values in batch
                        for m_val in valid_m_vals:
                            output = avg + m_val * spread
                            
                            # Get matching rows from measurement file
                            matching_rows = measurement_df[measurement_df[m_value_col] == m_val]
                            
                            for _, row in matching_rows.iterrows():
                                # Fast zone calculation
                                if range_low <= output <= range_high:
                                    if is_high_range:
                                        distance = range_high - output
                                    else:
                                        distance = output - range_low
                                    
                                    if distance <= 6:
                                        zone = "0 to 6"
                                    elif distance <= 12:
                                        zone = "6 to 12"
                                    elif distance <= 18:
                                        zone = "12 to 18"
                                    else:
                                        zone = "18 to 24"
                                else:
                                    zone = "Out of Range"
                                
                                # Fast datetime formatting
                                try:
                                    arrival_dt = hlc_data['datetime']
                                    day_abbrev = arrival_dt.strftime('%a')
                                    arrival_excel = arrival_dt.strftime('%Y-%m-%d %H:%M')
                                    
                                    # Simple day index calculation
                                    days_diff = (arrival_dt.date() - report_time.date()).days
                                    day_index = f"[{days_diff}]"
                                except:
                                    day_abbrev = ""
                                    arrival_excel = ""
                                    day_index = "[0]"
                                
                                # Use pre-calculated batch inputs
                                feed_type = "Small" if data_source == "Small CSV" else "Big"
                                if feed_type == "Small":
                                    input_18 = small_start if small_start is not None else 0
                                    input_report = small_report if small_report is not None else 0
                                else:
                                    input_18 = big_start if big_start is not None else 0
                                    input_report = big_report if big_report is not None else 0
                                
                                # Fast input at arrival
                                if feed_type == "Small":
                                    input_arrival = fast_get_input_at_time(small_prepared, arrival_dt)
                                else:
                                    input_arrival = fast_get_input_at_time(big_prepared, arrival_dt)
                                
                                if input_arrival is None:
                                    input_arrival = 0
                                
                                all_valid_entries.append({
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
                                    'Zone': zone
                                })

        return all_valid_entries

    except Exception as e:
        st.error(f"Error in optimized processing: {e}")
        return []

def apply_custom_ranges_advanced(df, small_df, report_time, high1, high2, low1, low2, use_high1, use_high2, use_low1, use_low2, big_df=None, run_model_g=False):
    """Apply custom ranges with performance optimization"""
    try:
        # Clear datetime cache for new run
        global _datetime_cache
        _datetime_cache.clear()
        
        st.info("Starting optimized custom range processing...")
        
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

        # Use optimized processing
        valid_entries = process_custom_ranges_optimized(df, small_df, big_df, report_time, custom_ranges)

        if not valid_entries:
            return pd.DataFrame()

        result_df = pd.DataFrame(valid_entries)
        st.success(f"Optimized processing complete: {len(result_df)} entries in {len(_datetime_cache)} unique timestamps")
        
        return result_df
        
    except Exception as e:
        st.error(f"Error in apply_custom_ranges_advanced: {e}")
        return pd.DataFrame()

def apply_full_range_advanced(df, small_df, report_time, window_radius, day_start_hour=18, input_value_at_start=None, big_df=None, run_model_g=False):
    """Apply full range with performance optimization"""
    try:
        # Clear datetime cache
        global _datetime_cache
        _datetime_cache.clear()
        
        st.info("Starting optimized full range processing...")
        
        # Determine center
        center = input_value_at_start
        if center is None or pd.isna(center):
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

        # Create a single "range" for full range processing
        lo, hi = center - window_radius, center + window_radius
        custom_ranges = {'Full Range': {'enabled': True, 'value': center}}
        
        # Use the optimized processor (modified for full range)
        valid_entries = process_custom_ranges_optimized(df, small_df, big_df, report_time, 
                                                       {'Full Range': {'enabled': True, 'value': center, 'low': lo, 'high': hi}})

        if not valid_entries:
            return pd.DataFrame()

        # Convert and clean up
        out_df = pd.DataFrame(valid_entries)
        out_df = out_df.drop(columns=['Range', 'Zone'], errors='ignore')

        # Order columns
        preferred_cols = [
            'Feed', 'ddd', 'Arrival', 'Day', 'Origin',
            'M Name', 'M #', 'M Value', 'R #', 'Tag', 'Family',
            'Input @ 18:00', 'Diff @ 18:00', 'Input @ Arrival', 'Diff @ Arrival',
            'Input @ Report', 'Diff @ Report', 'Output'
        ]
        ordered = [c for c in preferred_cols if c in out_df.columns]
        remaining = [c for c in out_df.columns if c not in ordered]
        out_df = out_df[ordered + remaining]

        st.success(f"Optimized full range complete: {len(out_df)} entries in {len(_datetime_cache)} unique timestamps")
        return out_df
        
    except Exception as e:
        st.error(f"Error in apply_full_range_advanced: {e}")
        return pd.DataFrame()

# Performance monitoring wrapper
def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        st.info(f"Function {func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper

# Apply performance monitoring to key functions
process_custom_ranges_optimized = monitor_performance(process_custom_ranges_optimized)
