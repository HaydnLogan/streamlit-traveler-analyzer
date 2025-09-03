""" Sulin ref (Asian international exchange student from Canada, Claude)
Custom Range Calculator with FIXED WASP and Macedonia handling
"""

import datetime as dt
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import re

def safe_datetime_convert(value):
    """Convert datetime and strip timezone to avoid comparison errors"""
    try:
        if pd.isna(value):
            return pd.NaT
        
        dt_val = pd.to_datetime(value)
        # Strip timezone if present
        if hasattr(dt_val, 'tz') and dt_val.tz is not None:
            dt_val = dt_val.tz_localize(None)
        
        return dt_val
    except:
        return pd.NaT

def get_wasp_macedonia_data(hlc_df, report_time, origin_name, start_hour=18):
    """
    Get H/L/C data for WASP and Macedonia origins with proper datetime calculation.
    FIXED VERSION - handles timezone issues
    """
    try:
        if hlc_df is None or hlc_df.empty:
            return None
        
        # Determine the exact column names based on origin
        if origin_name.lower().startswith('wasp-12b'):
            if '[1]' in origin_name:
                h_col, l_col, c_col = 'WASP-12b H[1]', 'WASP-12b L[1]', 'WASP-12b C[1]'
                weeks_back = 1  # Last week
                display_name = "WASP-12b[-1]"
            elif '[2]' in origin_name:
                h_col, l_col, c_col = 'WASP-12b H[2]', 'WASP-12b L[2]', 'WASP-12b C[2]'
                weeks_back = 2  # Two weeks ago
                display_name = "WASP-12b[-2]"
            else:
                h_col, l_col, c_col = 'WASP-12b H', 'WASP-12b L', 'WASP-12b C'
                weeks_back = 0  # This week
                display_name = "WASP-12b"
            
            # Calculate Sunday datetime
            report_dt = safe_datetime_convert(report_time)
            if pd.isna(report_dt):
                return None
                
            days_since_sunday = (report_dt.weekday() + 1) % 7  # Monday=0, Sunday=6
            
            # Go back to the target Sunday
            target_sunday = report_dt - timedelta(days=days_since_sunday + 7 * weeks_back)
            arrival_datetime = target_sunday.replace(hour=start_hour, minute=0, second=0, microsecond=0)
            
        elif origin_name.lower().startswith('macedonia'):
            if '[1]' in origin_name:
                h_col, l_col, c_col = 'Macedonia H[1]', 'Macedonia L[1]', 'Macedonia C[1]'
                months_back = 1  # Last month
                display_name = "Macedonia[-1]"
            elif '[2]' in origin_name:
                h_col, l_col, c_col = 'Macedonia H[2]', 'Macedonia L[2]', 'Macedonia C[2]'
                months_back = 2  # Two months ago
                display_name = "Macedonia[-2]"
            else:
                h_col, l_col, c_col = 'Macedonia H', 'Macedonia L', 'Macedonia C'
                months_back = 0  # This month
                display_name = "Macedonia"
            
            # Calculate first day of target month
            report_dt = safe_datetime_convert(report_time)
            if pd.isna(report_dt):
                return None
                
            arrival_datetime = report_dt.replace(day=1, hour=start_hour, minute=0, second=0, microsecond=0)
            
            # Go back the specified number of months
            for _ in range(months_back):
                if arrival_datetime.month == 1:
                    arrival_datetime = arrival_datetime.replace(year=arrival_datetime.year - 1, month=12)
                else:
                    arrival_datetime = arrival_datetime.replace(month=arrival_datetime.month - 1)
        else:
            return None
        
        # Check if required columns exist
        if not all(col in hlc_df.columns for col in [h_col, l_col, c_col]):
            return None
        
        # Get the most recent row with valid H/L/C data
        # Convert time column for filtering with timezone handling
        hlc_df_copy = hlc_df.copy()
        if 'time' in hlc_df_copy.columns:
            hlc_df_copy['time_clean'] = hlc_df_copy['time'].apply(safe_datetime_convert)
        else:
            hlc_df_copy['time_clean'] = hlc_df_copy.iloc[:, 0].apply(safe_datetime_convert)
        
        # Filter to data at or before report time (both timezone-naive now)
        report_clean = safe_datetime_convert(report_time)
        valid_data = hlc_df_copy[
            (hlc_df_copy['time_clean'] <= report_clean) &
            hlc_df_copy['time_clean'].notna()
        ].copy()
        
        if valid_data.empty:
            return None
        
        # Sort by time descending to get most recent first
        valid_data = valid_data.sort_values('time_clean', ascending=False)
        
        # Find first row with valid H/L/C data
        for _, row in valid_data.iterrows():
            h_val = row.get(h_col)
            l_val = row.get(l_col)
            c_val = row.get(c_col)
            
            if pd.notna(h_val) and pd.notna(l_val) and pd.notna(c_val):
                try:
                    return {
                        'H': float(h_val),
                        'L': float(l_val),
                        'C': float(c_val),
                        'datetime': arrival_datetime,  # Use calculated datetime, not data datetime
                        'origin': display_name
                    }
                except (ValueError, TypeError):
                    continue
        
        return None
        
    except Exception as e:
        st.error(f"Error getting WASP/Macedonia data for {origin_name}: {e}")
        return None

def get_input_at_time(df, target_time):
    """Get 'open' input value from dataframe at specific time - FIXED timezone handling"""
    if df is None or target_time is None or df.empty:
        return None
    
    try:
        if 'time' not in df.columns or 'open' not in df.columns:
            return None
        
        df_copy = df.copy()
        # Convert both to timezone-naive for comparison
        df_copy["time"] = df_copy["time"].apply(safe_datetime_convert)
        target_clean = safe_datetime_convert(target_time)
        
        if pd.isna(target_clean):
            return None
        
        # Exact match first
        exact_match = df_copy[df_copy["time"] == target_clean]
        if not exact_match.empty:
            return exact_match.iloc[-1]["open"]
        
        # Closest match
        df_copy["time_diff"] = abs(df_copy["time"] - target_clean)
        closest_idx = df_copy["time_diff"].idxmin()
        return df_copy.loc[closest_idx, "open"]
        
    except Exception as e:
        st.warning(f"Error getting input at time: {e}")
        return None

def get_start_time(report_time, start_hour):
    """Calculate start time (18:00 on day before if report is before 18:00) - FIXED timezone"""
    try:
        report_clean = safe_datetime_convert(report_time)
        if pd.isna(report_clean):
            return None
            
        start_time = report_clean.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        
        if report_clean.hour < start_hour:
            start_time = start_time - pd.Timedelta(days=1)
        
        return start_time
    except Exception as e:
        st.warning(f"Error calculating start time: {e}")
        return None

def find_new_data_changes(small_df, report_time, origin_name, scope_days=20):
    """
    Find new data changes for REGULAR origins (not WASP/Macedonia)
    FIXED VERSION - handles timezone issues
    """
    try:
        if small_df is None or small_df.empty:
            return []
        
        # Skip WASP and Macedonia - they should use get_wasp_macedonia_data instead
        if origin_name.lower().startswith('wasp-12b') or origin_name.lower().startswith('macedonia'):
            return []
        
        report_clean = safe_datetime_convert(report_time)
        if pd.isna(report_clean):
            return []

        # Look for H/L/C columns for regular origins
        h_col = f"{origin_name} H"
        l_col = f"{origin_name} L" 
        c_col = f"{origin_name} C"
        
        if not all(col in small_df.columns for col in [h_col, l_col, c_col]):
            return []

        # Process dataframe with timezone handling
        df_copy = small_df.copy()
        df_copy['time_clean'] = df_copy['time'].apply(safe_datetime_convert)
        
        # Filter by scope (both timezone-naive now)
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

    except Exception as e:
        st.error(f"Error finding changes for {origin_name}: {e}")
        return []

def apply_custom_ranges_advanced(measurement_df, small_df, big_df, report_time, custom_ranges, start_hour=18):
    """
    Process custom ranges with FIXED WASP and Macedonia handling
    FIXED VERSION - ensures both feeds are processed
    """
    all_valid_entries = []
    
    try:
        # Pre-calculate batch inputs for performance
        small_input_at_start = get_input_at_time(small_df, get_start_time(report_time, start_hour))
        big_input_at_start = get_input_at_time(big_df, get_start_time(report_time, start_hour))
        small_input_at_report = get_input_at_time(small_df, report_time)
        big_input_at_report = get_input_at_time(big_df, report_time)
        
        # FIXED: Process both data sources - ensure both are included
        data_sources = []
        if small_df is not None and not small_df.empty:
            data_sources.append((small_df.copy(), "Small CSV"))
            st.info(f"Small CSV: {len(small_df)} rows")
        else:
            st.warning("Small CSV is None or empty")
            
        if big_df is not None and not big_df.empty:
            data_sources.append((big_df.copy(), "Big CSV"))
            st.info(f"Big CSV: {len(big_df)} rows")
        else:
            st.warning("Big CSV is None or empty")
        
        if not data_sources:
            st.error("No valid data sources found")
            return []
        
        # Get all origins including WASP and Macedonia variants
        all_origins = set()
        for hlc_df, source_name in data_sources:
            origin_count = 0
            for col in hlc_df.columns:
                if col.endswith(' H'):
                    origin_name = col[:-2]
                    all_origins.add(origin_name)
                    origin_count += 1
            st.info(f"{source_name}: Found {origin_count} H/L/C origin sets")
        
        # FIXED: Add all WASP and Macedonia variants explicitly
        wasp_macedonia_origins = [
            'WASP-12b', 'WASP-12b[1]', 'WASP-12b[2]',
            'Macedonia', 'Macedonia[1]', 'Macedonia[2]'
        ]
        for variant in wasp_macedonia_origins:
            all_origins.add(variant)
        
        origins = list(all_origins)
        st.info(f"Total origins to process: {len(origins)}")
        st.info(f"Origins: {', '.join(sorted(origins))}")
        
        # Get M value column
        m_value_col = None
        for col in ['M value', 'M Value', 'M_Value', 'M_value', 'm value', 'm_value']:
            if col in measurement_df.columns:
                m_value_col = col
                break
        
        if m_value_col is None:
            st.error("No M value column found in measurement data")
            return []
        
        # Process each range
        for range_name, range_config in custom_ranges.items():
            if not range_config.get('enabled', False) or range_config.get('value', 0) == 0:
                continue
            
            range_value = range_config['value']
            if range_name.startswith('High'):
                range_low, range_high = range_value - 24, range_value
                is_high_range = True
            else:
                range_low, range_high = range_value, range_value + 24
                is_high_range = False
            
            st.markdown(f"### Processing {range_name}: {range_low:.3f} to {range_high:.3f}")
            
            # Process each data source
            for hlc_df, data_source in data_sources:
                st.info(f"Processing {data_source}")
                
                # Process each origin
                processed_origins = 0
                for origin in origins:
                    
                    # FIXED: Special handling for WASP and Macedonia
                    if origin.lower().startswith('wasp-12b') or origin.lower().startswith('macedonia'):
                        hlc_data = get_wasp_macedonia_data(hlc_df, report_time, origin, start_hour)
                        hlc_data_list = [hlc_data] if hlc_data else []
                        if hlc_data:
                            processed_origins += 1
                    else:
                        # Regular origins: use find_new_data_changes
def apply_full_range_advanced(measurement_df, small_df, big_df, report_time, center, window_radius, start_hour=18):
    """
    Process full range with FIXED WASP and Macedonia handling
    FIXED VERSION - ensures both feeds are processed and handles timezone issues
    """
    all_valid_entries = []
    
    try:
        lo = center - window_radius
        hi = center + window_radius
        st.info(f"Full Range window: [{lo:.1f}, {hi:.1f}] around center={center}")
        
        # Pre-calculate batch inputs for performance
        small_input_at_start = get_input_at_time(small_df, get_start_time(report_time, start_hour))
        big_input_at_start = get_input_at_time(big_df, get_start_time(report_time, start_hour))
        small_input_at_report = get_input_at_time(small_df, report_time)
        big_input_at_report = get_input_at_time(big_df, report_time)
        
        # FIXED: Process both data sources - ensure both are included
        data_sources = []
        if small_df is not None and not small_df.empty:
            data_sources.append((small_df.copy(), "Small CSV"))
            st.info(f"Small CSV: {len(small_df)} rows")
        else:
            st.warning("Small CSV is None or empty")
            
        if big_df is not None and not big_df.empty:
            data_sources.append((big_df.copy(), "Big CSV"))
            st.info(f"Big CSV: {len(big_df)} rows")
        else:
            st.warning("Big CSV is None or empty")
        
        if not data_sources:
            st.error("No valid data sources found")
            return []
        
        # Get all origins including WASP and Macedonia variants
        all_origins = set()
        for hlc_df, source_name in data_sources:
            origin_count = 0
            for col in hlc_df.columns:
                if col.endswith(' H'):
                    origin_name = col[:-2]
                    all_origins.add(origin_name)
                    origin_count += 1
            st.info(f"{source_name}: Found {origin_count} H/L/C origin sets")
        
        # FIXED: Add all WASP and Macedonia variants explicitly
        wasp_macedonia_origins = [
            'WASP-12b', 'WASP-12b[1]', 'WASP-12b[2]',
            'Macedonia', 'Macedonia[1]', 'Macedonia[2]'
        ]
        for variant in wasp_macedonia_origins:
            all_origins.add(variant)
        
        origins = list(all_origins)
        st.info(f"Total origins to process: {len(origins)}")
        
        # Get M value column
        m_value_col = None
        for col in ['M value', 'M Value', 'M_Value', 'M_value', 'm value', 'm_value']:
            if col in measurement_df.columns:
                m_value_col = col
                break
        
        if m_value_col is None:
            st.error("No M value column found in measurement data")
            return []
        
        # Process each data source
        for hlc_df, data_source in data_sources:
            st.info(f"Processing {data_source}")
            
            # Process each origin
            processed_origins = 0
            for origin in origins:
                
                # FIXED: Special handling for WASP and Macedonia
                if origin.lower().startswith('wasp-12b') or origin.lower().startswith('macedonia'):
                    hlc_data = get_wasp_macedonia_data(hlc_df, report_time, origin, start_hour)
                    hlc_data_list = [hlc_data] if hlc_data else []
                    if hlc_data:
                        processed_origins += 1
                else:
                    # Regular origins: use find_new_data_changes
                    hlc_data_list = find_new_data_changes(hlc_df, report_time, origin, 20)
                    if hlc_data_list:
                        processed_origins += 1
                
                if not hlc_data_list:
                    continue
                
                # Process each H/L/C data entry
                for hlc_data in hlc_data_list:
                    # Calculate raw M values for full range
                    H, L, C = hlc_data['H'], hlc_data['L'], hlc_data['C']
                    avg = (H + L + C) / 3
                    spread = H - L
                    
                    if spread == 0:
                        continue
                    
                    raw_m_low = (lo - avg) / spread
                    raw_m_high = (hi - avg) / spread
                    
                    # Find valid M values
                    m_values = pd.to_numeric(measurement_df[m_value_col], errors='coerce').dropna()
                    valid_m_mask = (m_values >= raw_m_low) & (m_values <= raw_m_high)
                    valid_m_vals = m_values[valid_m_mask].unique()
                    
                    # Process each valid M value
                    for m_val in valid_m_vals:
                        output = avg + m_val * spread
                        
                        # Get matching measurement rows
                        matching_rows = measurement_df[measurement_df[m_value_col] == m_val]
                        
                        for _, row in matching_rows.iterrows():
                            # Format arrival datetime
                            arrival_dt = hlc_data['datetime']
                            try:
                                day_abbrev = arrival_dt.strftime('%a')
                                arrival_excel = arrival_dt.strftime('%Y-%m-%d %H:%M')
                                
                                # Calculate day index
                                try:
                                    from a_helpers import get_day_index
                                    day_index = get_day_index(arrival_dt, report_time, start_hour)
                                except:
                                    day_index = "[0]"
                            except:
                                day_abbrev = ""
                                arrival_excel = ""
                                day_index = "[0]"
                            
                            # Get input values based on feed type
                            feed_type = "Small" if data_source == "Small CSV" else "Big"
                            if feed_type == "Small":
                                input_18 = small_input_at_start if small_input_at_start is not None else 0
                                input_report = small_input_at_report if small_input_at_report is not None else 0
                                input_arrival = get_input_at_time(small_df, arrival_dt)
                            else:
                                input_18 = big_input_at_start if big_input_at_start is not None else 0
                                input_report = big_input_at_report if big_input_at_report is not None else 0
                                input_arrival = get_input_at_time(big_df, arrival_dt)
                            
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
                                'Output': output
                            })
            
            st.info(f"{data_source}: Processed {processed_origins} origins successfully")
        
        return all_valid_entries
        
    except Exception as e:
        st.error(f"Error in apply_full_range_advanced: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return []changes(hlc_df, report_time, origin, 20)
                        if hlc_data_list:
                            processed_origins += 1
                    
                    if not hlc_data_list:
                        continue
                    
                    # Process each H/L/C data entry
                    for hlc_data in hlc_data_list:
                        # Calculate raw M values
                        H, L, C = hlc_data['H'], hlc_data['L'], hlc_data['C']
                        avg = (H + L + C) / 3
                        spread = H - L
                        
                        if spread == 0:
                            continue
                        
                        raw_m_low = (range_low - avg) / spread
                        raw_m_high = (range_high - avg) / spread
                        
                        # Find valid M values
                        m_values = pd.to_numeric(measurement_df[m_value_col], errors='coerce').dropna()
                        valid_m_mask = (m_values >= raw_m_low) & (m_values <= raw_m_high)
                        valid_m_vals = m_values[valid_m_mask].unique()
                        
                        # Process each valid M value
                        for m_val in valid_m_vals:
                            output = avg + m_val * spread
                            
                            # Get matching measurement rows
                            matching_rows = measurement_df[measurement_df[m_value_col] == m_val]
                            
                            for _, row in matching_rows.iterrows():
                                # Calculate zone
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
                                
                                # Format arrival datetime
                                arrival_dt = hlc_data['datetime']
                                try:
                                    day_abbrev = arrival_dt.strftime('%a')
                                    arrival_excel = arrival_dt.strftime('%Y-%m-%d %H:%M')
                                    
                                    # Calculate day index
                                    try:
                                        from a_helpers import get_day_index
                                        day_index = get_day_index(arrival_dt, report_time, start_hour)
                                    except:
                                        day_index = "[0]"
                                except:
                                    day_abbrev = ""
                                    arrival_excel = ""
                                    day_index = "[0]"
                                
                                # Get input values based on feed type
                                feed_type = "Small" if data_source == "Small CSV" else "Big"
                                if feed_type == "Small":
                                    input_18 = small_input_at_start if small_input_at_start is not None else 0
                                    input_report = small_input_at_report if small_input_at_report is not None else 0
                                    input_arrival = get_input_at_time(small_df, arrival_dt)
                                else:
                                    input_18 = big_input_at_start if big_input_at_start is not None else 0
                                    input_report = big_input_at_report if big_input_at_report is not None else 0
                                    input_arrival = get_input_at_time(big_df, arrival_dt)
                                
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
                
                st.info(f"{data_source}: Processed {processed_origins} origins successfully")
        
        return all_valid_entries
        
    except Exception as e:
        st.error(f"Error in apply_custom_ranges_advanced: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return []changes(hlc_df, report_time, origin, 20)
                    
                    if not hlc_data_list:
                        continue
                    
                    # Process each H/L/C data entry
                    for hlc_data in hlc_data_list:
                        # Calculate raw M values
                        H, L, C = hlc_data['H'], hlc_data['L'], hlc_data['C']
                        avg = (H + L + C) / 3
                        spread = H - L
                        
                        if spread == 0:
                            continue
                        
                        raw_m_low = (range_low - avg) / spread
                        raw_m_high = (range_high - avg) / spread
                        
                        # Find valid M values
                        m_values = pd.to_numeric(measurement_df[m_value_col], errors='coerce').dropna()
                        valid_m_mask = (m_values >= raw_m_low) & (m_values <= raw_m_high)
                        valid_m_vals = m_values[valid_m_mask].unique()
                        
                        # Process each valid M value
                        for m_val in valid_m_vals:
                            output = avg + m_val * spread
                            
                            # Get matching measurement rows
                            matching_rows = measurement_df[measurement_df[m_value_col] == m_val]
                            
                            for _, row in matching_rows.iterrows():
                                # Calculate zone
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
                                
                                # Format arrival datetime
                                arrival_dt = hlc_data['datetime']
                                try:
                                    day_abbrev = arrival_dt.strftime('%a')
                                    arrival_excel = arrival_dt.strftime('%Y-%m-%d %H:%M')
                                    
                                    # Calculate day index
                                    try:
                                        from a_helpers import get_day_index
                                        day_index = get_day_index(arrival_dt, report_time, start_hour)
                                    except:
                                        day_index = "[0]"
                                except:
                                    day_abbrev = ""
                                    arrival_excel = ""
                                    day_index = "[0]"
                                
                                # Get input values based on feed type
                                feed_type = "Small" if data_source == "Small CSV" else "Big"
                                if feed_type == "Small":
                                    input_18 = small_input_at_start if small_input_at_start is not None else 0
                                    input_report = small_input_at_report if small_input_at_report is not None else 0
                                    input_arrival = get_input_at_time(small_df, arrival_dt)
                                else:
                                    input_18 = big_input_at_start if big_input_at_start is not None else 0
                                    input_report = big_input_at_report if big_input_at_report is not None else 0
                                    input_arrival = get_input_at_time(big_df, arrival_dt)
                                
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
        st.error(f"Error in apply_custom_ranges_advanced: {e}")
        return []

def get_input_at_time(df, target_time):
    """Get 'open' input value from dataframe at specific time"""
    if df is None or target_time is None or df.empty:
        return None
    
    try:
        if 'time' not in df.columns or 'open' not in df.columns:
            return None
        
        df_copy = df.copy()
        df_copy["time"] = pd.to_datetime(df_copy["time"], errors='coerce')
        target_clean = pd.to_datetime(target_time, errors='coerce')
        
        # Exact match first
        exact_match = df_copy[df_copy["time"] == target_clean]
        if not exact_match.empty:
            return exact_match.iloc[-1]["open"]
        
        # Closest match
        df_copy["time_diff"] = abs(df_copy["time"] - target_clean)
        closest_idx = df_copy["time_diff"].idxmin()
        return df_copy.loc[closest_idx, "open"]
        
    except:
        return None

def get_start_time(report_time, start_hour):
    """Calculate start time (18:00 on day before if report is before 18:00)"""
    try:
        report_clean = pd.to_datetime(report_time)
        start_time = report_clean.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        
        if report_clean.hour < start_hour:
            start_time = start_time - pd.Timedelta(days=1)
        
        return start_time
    except:
        return None

def find_new_data_changes(small_df, report_time, origin_name, scope_days=20):
    """
    Find new data changes for REGULAR origins (not WASP/Macedonia)
    WASP and Macedonia should not use this function as they have dedicated columns
    """
    try:
        if small_df is None or small_df.empty:
            return []
        
        # Skip WASP and Macedonia - they should use get_wasp_macedonia_data instead
        if origin_name.lower().startswith('wasp-12b') or origin_name.lower().startswith('macedonia'):
            return []
        
        report_clean = pd.to_datetime(report_time, errors='coerce')
        if pd.isna(report_clean):
            return []

        # Look for H/L/C columns for regular origins
        h_col = f"{origin_name} H"
        l_col = f"{origin_name} L" 
        c_col = f"{origin_name} C"
        
        if not all(col in small_df.columns for col in [h_col, l_col, c_col]):
            return []

        # Process dataframe
        df_copy = small_df.copy()
        df_copy['time_clean'] = pd.to_datetime(df_copy['time'], errors='coerce')
        
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

    except Exception as e:
        st.error(f"Error finding changes for {origin_name}: {e}")
        return []

# Updated apply function
def apply_custom_ranges_advanced(df, small_df, report_time, high1, high2, low1, low2, use_high1, use_high2, use_low1, use_low2, big_df=None, run_model_g=False):
    """
    Apply custom ranges with FIXED WASP and Macedonia handling
    """
    try:
        st.info("Custom Range Processing with FIXED WASP/Macedonia handling started")
        
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

        # Use fixed processing
        valid_entries = apply_custom_ranges_advanced(df, small_df, big_df, report_time, custom_ranges, 18)

        if not valid_entries:
            st.warning("No valid entries found")
            return pd.DataFrame()

        result_df = pd.DataFrame(valid_entries)
        st.success(f"Processing complete: {len(result_df)} entries found")
        
        return result_df
        
    except Exception as e:
        st.error(f"Error in apply_custom_ranges_advanced: {e}")
        return pd.DataFrame()

def apply_full_range_advanced(measurement_df, small_df, big_df, report_time, center, window_radius, start_hour=18):
    """
    Process full range with FIXED WASP and Macedonia handling
    """
    all_valid_entries = []
    
    try:
        lo = center - window_radius
        hi = center + window_radius
        st.info(f"Full Range window: [{lo:.1f}, {hi:.1f}] around center={center}")
        
        # Pre-calculate batch inputs for performance
        small_input_at_start = get_input_at_time(small_df, get_start_time(report_time, start_hour))
        big_input_at_start = get_input_at_time(big_df, get_start_time(report_time, start_hour))
        small_input_at_report = get_input_at_time(small_df, report_time)
        big_input_at_report = get_input_at_time(big_df, report_time)
        
        # Process both data sources
        data_sources = []
        if small_df is not None and not small_df.empty:
            data_sources.append((small_df.copy(), "Small CSV"))
        if big_df is not None and not big_df.empty:
            data_sources.append((big_df.copy(), "Big CSV"))
        
        # Get all origins including WASP and Macedonia variants
        all_origins = set()
        for hlc_df, _ in data_sources:
            for col in hlc_df.columns:
                if col.endswith(' H'):
                    origin_name = col[:-2]
                    all_origins.add(origin_name)
        
        # FIXED: Add all WASP and Macedonia variants explicitly
        wasp_macedonia_origins = [
            'WASP-12b', 'WASP-12b[1]', 'WASP-12b[2]',
            'Macedonia', 'Macedonia[1]', 'Macedonia[2]'
        ]
        for variant in wasp_macedonia_origins:
            all_origins.add(variant)
        
        origins = list(all_origins)
        st.info(f"Processing origins: {', '.join(sorted(origins))}")
        
        # Get M value column
        m_value_col = None
        for col in ['M value', 'M Value', 'M_Value', 'M_value', 'm value', 'm_value']:
            if col in measurement_df.columns:
                m_value_col = col
                break
        
        if m_value_col is None:
            st.error("No M value column found in measurement data")
            return []
        
        # Process each data source
        for hlc_df, data_source in data_sources:
            
            # Process each origin
            for origin in origins:
                
                # FIXED: Special handling for WASP and Macedonia
                if origin.lower().startswith('wasp-12b') or origin.lower().startswith('macedonia'):
                    hlc_data = get_wasp_macedonia_data(hlc_df, report_time, origin, start_hour)
                    hlc_data_list = [hlc_data] if hlc_data else []
                else:
                    # Regular origins: use find_new_data_changes
                    hlc_data_list = find_new_data_changes(hlc_df, report_time, origin, 20)
                
                if not hlc_data_list:
                    continue
                
                # Process each H/L/C data entry
                for hlc_data in hlc_data_list:
                    # Calculate raw M values
                    H, L, C = hlc_data['H'], hlc_data['L'], hlc_data['C']
                    avg = (H + L + C) / 3
                    spread = H - L
                    
                    if spread == 0:
                        continue
                    
                    raw_m_low = (lo - avg) / spread
                    raw_m_high = (hi - avg) / spread
                    
                    # Find valid M values
                    m_values = pd.to_numeric(measurement_df[m_value_col], errors='coerce').dropna()
                    valid_m_mask = (m_values >= raw_m_low) & (m_values <= raw_m_high)
                    valid_m_vals = m_values[valid_m_mask].unique()
                    
                    # Process each valid M value
                    for m_val in valid_m_vals:
                        output = avg + m_val * spread
                        
                        # Get matching measurement rows
                        matching_rows = measurement_df[measurement_df[m_value_col] == m_val]
                        
                        for _, row in matching_rows.iterrows():
                            # Format arrival datetime
                            arrival_dt = hlc_data['datetime']
                            try:
                                day_abbrev = arrival_dt.strftime('%a')
                                arrival_excel = arrival_dt.strftime('%Y-%m-%d %H:%M')
                                
                                # Calculate day index
                                try:
                                    from a_helpers import get_day_index
                                    day_index = get_day_index(arrival_dt, report_time, start_hour)
                                except:
                                    day_index = "[0]"
                            except:
                                day_abbrev = ""
                                arrival_excel = ""
                                day_index = "[0]"
                            
                            # Get input values based on feed type
                            feed_type = "Small" if data_source == "Small CSV" else "Big"
                            if feed_type == "Small":
                                input_18 = small_input_at_start if small_input_at_start is not None else 0
                                input_report = small_input_at_report if small_input_at_report is not None else 0
                                input_arrival = get_input_at_time(small_df, arrival_dt)
                            else:
                                input_18 = big_input_at_start if big_input_at_start is not None else 0
                                input_report = big_input_at_report if big_input_at_report is not None else 0
                                input_arrival = get_input_at_time(big_df, arrival_dt)
                            
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
                                'Output': output
                            })
        
        return all_valid_entries
        
    except Exception as e:
        st.error(f"Error in apply_full_range_advanced: {e}")
        return []

def apply_full_range_advanced(df, small_df, report_time, window_radius, day_start_hour=18, input_value_at_start=None, big_df=None, run_model_g=False):
    """
    Apply full range with FIXED WASP and Macedonia handling
    """
    try:
        st.info("Full Range Processing with FIXED WASP/Macedonia handling started")
        
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

        # Use fixed processing
        valid_entries = apply_full_range_advanced(df, small_df, big_df, report_time, center, window_radius, day_start_hour)

        if not valid_entries:
            st.warning("No valid entries found")
            return pd.DataFrame()

        # Convert to DataFrame and clean up
        out_df = pd.DataFrame(valid_entries)
        
        # For full range, we don't need Range/Zone columns
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

        st.success(f"Full Range processing complete: {len(out_df)} entries found")
        return out_df
        
    except Exception as e:
        st.error(f"Error in apply_full_range_advanced: {e}")
        return pd.DataFrame()
