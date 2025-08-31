"""
Custom Range Calculator for Market Data Analysis
Implements sophisticated range calculation based on H/L/C data from small CSV files.
"""

import datetime as dt
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np

def find_new_data_changes(small_df, report_time, origin_name, scope_days=20):
    """
    Find the first time new data appears for an origin by detecting changes in H/L/C values.
    Starts from the bottom (most recent) and works backwards to find when data changed.
    
    Args:
        small_df: Small CSV dataframe
        report_time: Target datetime for analysis
        origin_name: Name of origin (e.g., 'Venus', 'Mercury', 'Fiji')
        scope_days: Maximum days to look back
        
    Returns:
        List of dictionaries with H, L, C values and datetime for each data change
    """
    try:
        # Convert report_time to pandas datetime if needed
        if isinstance(report_time, str):
            report_time = pd.to_datetime(report_time)
        
        # Look for columns ending with H, L, C for this origin
        h_col = f"{origin_name} H"
        l_col = f"{origin_name} L" 
        c_col = f"{origin_name} C"
        
        # Check if these columns exist
        if not all(col in small_df.columns for col in [h_col, l_col, c_col]):
            return []
        
        # Simple timezone stripping - no conversion
        small_df_copy = small_df.copy()
        # Strip timezone from ISO format: 2025-08-06T18:45:00-04:00 → 2025-08-06 18:45:00
        if 'time' in small_df_copy.columns:
            time_strings = small_df_copy['time'].astype(str)
            # Remove timezone offset (everything after + or - in time)
            clean_times = time_strings.str.replace(r'[+-]\d{2}:?\d{2}$', '', regex=True)
            # Replace T with space for standard datetime format
            clean_times = clean_times.str.replace('T', ' ')
            small_df_copy['time_dt'] = pd.to_datetime(clean_times, errors='coerce')
        else:
            small_df_copy['time_dt'] = pd.to_datetime(small_df_copy.iloc[:, 0], errors='coerce')
        
        # Ensure report_time is timezone-naive datetime
        if isinstance(report_time, str):
            # Strip timezone from string format
            clean_report_time = report_time.replace('T', ' ')
            clean_report_time = pd.Series([clean_report_time]).str.replace(r'[+-]\d{2}:?\d{2}$', '', regex=True).iloc[0]
            report_time = pd.to_datetime(clean_report_time)
        elif hasattr(report_time, 'tz') and report_time.tz is not None:
            report_time = report_time.replace(tzinfo=None)
        
        # Get data within scope and at or before report time
        scope_start = report_time - timedelta(days=scope_days)
        scoped_df = small_df_copy[
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
        return []

def find_most_current_data(small_df, report_time, origin_name, scope_days=20):
    """
    Find the most current data for an origin at report time.
    
    Args:
        small_df: Small CSV dataframe
        report_time: Target datetime for analysis
        origin_name: Name of origin (e.g., 'Venus', 'Mercury', 'Fiji')
        scope_days: Maximum days to look back
        
    Returns:
        Dictionary with H, L, C values and datetime, or None if not found
    """
    try:
        # Convert report_time to pandas datetime if needed
        if isinstance(report_time, str):
            report_time = pd.to_datetime(report_time)
        
        # Look for columns ending with H, L, C for this origin
        h_col = f"{origin_name} H"
        l_col = f"{origin_name} L" 
        c_col = f"{origin_name} C"
        
        # Check if these columns exist
        if not all(col in small_df.columns for col in [h_col, l_col, c_col]):
            return None
        
        # Simple timezone stripping - no conversion
        small_df_copy = small_df.copy()
        # Strip timezone from ISO format: 2025-08-06T18:45:00-04:00 → 2025-08-06 18:45:00
        if 'time' in small_df_copy.columns:
            time_strings = small_df_copy['time'].astype(str)
            # Remove timezone offset (everything after + or - in time)
            clean_times = time_strings.str.replace(r'[+-]\d{2}:?\d{2}$', '', regex=True)
            # Replace T with space for standard datetime format
            clean_times = clean_times.str.replace('T', ' ')
            small_df_copy['time_dt'] = pd.to_datetime(clean_times, errors='coerce')
        else:
            small_df_copy['time_dt'] = pd.to_datetime(small_df_copy.iloc[:, 0], errors='coerce')
        
        # Ensure report_time is timezone-naive datetime
        if isinstance(report_time, str):
            # Strip timezone from string format
            clean_report_time = report_time.replace('T', ' ')
            clean_report_time = pd.Series([clean_report_time]).str.replace(r'[+-]\d{2}:?\d{2}$', '', regex=True).iloc[0]
            report_time = pd.to_datetime(clean_report_time)
        elif hasattr(report_time, 'tz') and report_time.tz is not None:
            report_time = report_time.replace(tzinfo=None)
        
        # Priority 1: Look for data from the same day as report_time
        report_date = report_time.date()
        same_day_df = small_df_copy[small_df_copy['time_dt'].dt.date == report_date].copy()
        
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
            (small_df_copy['time_dt'] >= scope_start) & 
            (small_df_copy['time_dt'] <= report_time)
        ].copy()
        
        if scoped_df.empty:
            return None
        
        # Sort by time descending to start with most recent
        scoped_df = scoped_df.sort_values('time_dt', ascending=False)
        
        # Find the most current data by checking for different values
        for i, row in scoped_df.iterrows():
            h_val = row[h_col]
            l_val = row[l_col]
            c_val = row[c_col]
            
            # Skip if any values are null
            if pd.isna(h_val) or pd.isna(l_val) or pd.isna(c_val):
                continue
            
            # Check if this row has different data from previous rows
            is_current = True
            if i > 0:  # Not the first row
                # Look at next newer rows to see if data changed
                newer_rows = scoped_df[scoped_df['time_dt'] > row['time_dt']]
                if not newer_rows.empty:
                    for _, newer_row in newer_rows.iterrows():
                        if (newer_row[h_col] == h_val and 
                            newer_row[l_col] == l_val and 
                            newer_row[c_col] == c_val):
                            is_current = False
                            break
            
            if is_current:
                return {
                    'H': float(h_val),
                    'L': float(l_val),
                    'C': float(c_val),
                    'datetime': row['time_dt'],
                    'origin': origin_name
                }
        
        # If no "current" data found, use the most recent row with valid data
        for _, row in scoped_df.iterrows():
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
        return None

def calculate_raw_m_values(hlc_data, range_low, range_high):
    """
    Calculate raw M values for a price range.
    
    Args:
        hlc_data: Dictionary with H, L, C values
        range_low: Lower bound of price range
        range_high: Upper bound of price range
        
    Returns:
        Dictionary with raw_m_low and raw_m_high
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

def find_valid_m_values(measurement_df, raw_m_low, raw_m_high, hlc_data, range_low, range_high, is_high_range=False, data_source="Unknown", report_time=None):
    """
    Find valid M values from measurement file within the raw M range.
    
    Args:
        measurement_df: Measurement dataframe
        raw_m_low: Lower raw M boundary
        raw_m_high: Upper raw M boundary  
        hlc_data: H/L/C data for output calculation
        range_low: Lower price boundary
        range_high: Upper price boundary
        is_high_range: True if this is a High range, False for Low range
        
    Returns:
        Dictionary with 'valid_entries' and 'valid_m_list'
    """
    try:
        valid_entries = []
        valid_m_values = []  # Track valid M values
        rejected_m_values = []  # Track rejected M values with reasons
        
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
        debug_count = 0
        zone_summary = {"0 to 6": 0, "6 to 12": 0, "12 to 18": 0, "18 to 24": 0, "Out of Range": 0}
        for m_val in m_values:
            try:
                m_float = float(m_val)
                debug_count += 1
                
                # Check if M value is within range
                if raw_m_low <= m_float <= raw_m_high:
                    # Calculate output for this M value
                    output = hlc_data['avg'] + m_float * hlc_data['spread']
                    
                    
                    # This M value is valid (within raw M range)
                    valid_m_values.append(m_float)
                    # Get all rows with this M value (match by detected column, not 'M #')
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
                        
                        # Track zone distribution
                        zone_summary[zone_value] += 1
                        

                        
                        # Format arrival time from H/L/C data
                        try:
                            arrival_dt = hlc_data['datetime']
                            day_abbrev = arrival_dt.strftime('%a')  # Mon, Tue, Wed, etc.
                            arrival_excel = arrival_dt.strftime('%Y-%m-%d %H:%M')  # Excel-friendly format
                            
                            # Calculate day index using proper get_day_index function
                            try:
                                from a_helpers import get_day_index
                                if isinstance(arrival_dt, str):
                                    arrival_dt = pd.to_datetime(arrival_dt)
                                # Use the proper get_day_index function with start_hour (defaulting to 18)
                                day_index = get_day_index(arrival_dt, report_time, 18)

                            except Exception as e:

                                day_index = "[0]"
                        except:
                            day_abbrev = ""
                            arrival_excel = str(hlc_data['datetime'])
                            day_index = "[0]"
                        
                        # Determine feed type from data source name
                        feed_type = "Small" if data_source == "Small CSV" else "Big"
                        
                        # Calculate input values (basic implementation for now)
                        # These would normally come from small_df at specific times
                        input_18 = hlc_data.get('H', 0)  # Placeholder
                        input_arrival = hlc_data.get('C', 0)  # Placeholder  
                        input_report = hlc_data.get('L', 0)  # Placeholder
                        
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
                else:
                    # M value outside raw range
                    rejected_m_values.append({
                        'm_value': m_float,
                        'reason': 'outside_raw_range',
                        'distance_from_range': min(abs(m_float - raw_m_low), abs(m_float - raw_m_high))
                    })
                
            except (ValueError, TypeError):
                continue  # Skip invalid M values
        

        
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
    Process custom ranges using advanced H/L/C calculation method.
    
    Args:
        measurement_df: Measurement dataframe
        small_df: Small CSV dataframe with H/L/C data
        report_time: Report datetime
        custom_ranges: Dictionary with range specifications
        scope_days: Days to look back for current data
        big_df: Big CSV dataframe (fallback for H/L/C data if small_df lacks current data)
        
    Returns:
        List of valid entries for all ranges
    """
    all_valid_entries = []
    processing_summary = []
    
    # Process both Big and Small feeds if available
    data_sources = []
    
    # Always include Small CSV if available
    if small_df is not None and not small_df.empty:
        data_sources.append((small_df.copy(), "Small CSV"))
    
    # Always include Big CSV if available
    if big_df is not None and not big_df.empty:
        data_sources.append((big_df.copy(), "Big CSV"))
    
    # Get all unique origins from both data sources
    all_origins = set()
    for hlc_df, data_source in data_sources:
        for col in hlc_df.columns:
            if col.endswith(' H'):
                origin_name = col[:-2]  # Remove ' H' suffix
                all_origins.add(origin_name)
    
    # Add [1] and [2] versions of WASP-12b and Macedonia if not already present
    wasp_variants = ['WASP-12b[1]', 'WASP-12b[2]']
    macedonia_variants = ['Macedonia[1]', 'Macedonia[2]']
    
    for variant in wasp_variants + macedonia_variants:
        all_origins.add(variant)
    
    origins = list(all_origins)
    
    st.info(f"Processing origins from both feeds: {', '.join(origins)}")
    
    # DEBUG: Show H/L/C data for first origin around report time
    if len(origins) > 0:
        first_origin = origins[0]
        st.warning(f"DEBUG: Analyzing H/L/C data for first origin: {first_origin}")
        
        # Find data around report time for debugging
        h_col = f"{first_origin} H"
        l_col = f"{first_origin} L"  
        c_col = f"{first_origin} C"
        
        if all(col in small_df.columns for col in [h_col, l_col, c_col]):
            debug_df = small_df.copy()
            # Simple timezone stripping - no conversion
            if 'time' in debug_df.columns:
                time_strings = debug_df['time'].astype(str)
                # Remove timezone offset (everything after + or - in time)
                clean_times = time_strings.str.replace(r'[+-]\d{2}:?\d{2}$', '', regex=True)
                # Replace T with space for standard datetime format
                clean_times = clean_times.str.replace('T', ' ')
                debug_df['time_dt'] = pd.to_datetime(clean_times, errors='coerce')
            else:
                debug_df['time_dt'] = pd.to_datetime(debug_df.iloc[:, 0], errors='coerce')
            
            # Ensure report_time is timezone-naive datetime
            debug_report_time = report_time
            if isinstance(report_time, str):
                # Strip timezone from string format
                clean_report_time = report_time.replace('T', ' ')
                clean_report_time = pd.Series([clean_report_time]).str.replace(r'[+-]\d{2}:?\d{2}$', '', regex=True).iloc[0]
                debug_report_time = pd.to_datetime(clean_report_time)
            elif hasattr(report_time, 'tz') and report_time.tz is not None:
                debug_report_time = report_time.replace(tzinfo=None)
            
            st.warning(f"DEBUG: Report time: {debug_report_time}")
            st.text(f"DEBUG: Report date extracted: {debug_report_time.date()}")
            
            # Show sample timestamps from CSV for comparison
            st.text(f"DEBUG: Sample CSV timestamps (first 3):")
            for i, ts in enumerate(debug_df['time_dt'].head(3)):
                st.text(f"  {i+1}: {ts} | Date: {ts.date()}")
            
            st.text(f"DEBUG: Sample CSV timestamps (last 3):")
            for i, ts in enumerate(debug_df['time_dt'].tail(3)):
                st.text(f"  {i+1}: {ts} | Date: {ts.date()}")
            
            # Check data availability around report time
            report_date = debug_report_time.date()
            same_day_data = debug_df[debug_df['time_dt'].dt.date == report_date]
            
            st.text(f"DEBUG: Looking for date {report_date} in {len(debug_df)} rows")
            
            if same_day_data.empty:
                st.error(f"DEBUG: No data found for {report_date} - this explains why it falls back to previous day!")
                
                # Show what dates are available
                available_dates = sorted(debug_df['time_dt'].dt.date.unique())
                st.text(f"Available dates in CSV: {[str(d) for d in available_dates[-10:]]}")  # Last 10 dates
                
                # Show data around report time from any date
                closest_before = debug_df[debug_df['time_dt'] <= debug_report_time].sort_values('time_dt', ascending=False)
                if not closest_before.empty:
                    st.warning(f"DEBUG: Closest data before report time (fallback logic):")
                    for i, (_, row) in enumerate(closest_before.head(4).iterrows()):
                        marker = "➤ " if i == 0 else "  "
                        st.text(f"{marker}Time: {row['time_dt']} | H: {row.get(h_col, 'N/A')} | L: {row.get(l_col, 'N/A')} | C: {row.get(c_col, 'N/A')}")
            else:
                st.success(f"DEBUG: Found {len(same_day_data)} rows for {report_date}")
                
                # Sort by time and find rows around report time
                debug_df = debug_df.sort_values('time_dt')
                report_row = None
                
                # Find the row at or just before report time
                for idx, row in debug_df.iterrows():
                    if row['time_dt'] <= debug_report_time:
                        report_row = row
                    else:
                        break
                
                if report_row is not None:
                    # Get position of report row
                    report_idx = debug_df[debug_df['time_dt'] == report_row['time_dt']].index[0]
                    report_row_pos = debug_df.index.get_loc(report_idx)
                    
                    # Get the report time row and 3 rows above it
                    start_pos = max(0, report_row_pos - 3)
                    debug_rows = debug_df.iloc[start_pos:report_row_pos + 1]
                    
                    st.warning(f"DEBUG: H/L/C data for {first_origin} (last 4 rows up to report time):")
                    for i, (_, row) in enumerate(debug_rows.iterrows()):
                        marker = "➤ " if i == len(debug_rows) - 1 else "  "
                        st.text(f"{marker}Time: {row['time_dt']} | H: {row.get(h_col, 'N/A')} | L: {row.get(l_col, 'N/A')} | C: {row.get(c_col, 'N/A')}")
                else:
                    st.warning(f"DEBUG: No data found for {first_origin} at or before report time")
        else:
            st.warning(f"DEBUG: Missing H/L/C columns for {first_origin}")
    
    # Process each custom range for all data sources
    for range_name, range_config in custom_ranges.items():
        if not range_config.get('enabled', False):
            continue
            
        range_value = range_config.get('value', 0)
        if range_value == 0:
            continue
        
        # Determine range bounds
        if range_name.startswith('High'):
            # High ranges: value-24 to value
            range_low = range_value - 24
            range_high = range_value
            is_high_range = True
        else:
            # Low ranges: value to value+24
            range_low = range_value
            range_high = range_value + 24
            is_high_range = False
        
        st.markdown(f"### Processing {range_name}: {range_low:.3f} to {range_high:.3f}")
        
        range_entries = []
        
        # Process each data source for this range
        for hlc_df, data_source in data_sources:
        
            st.markdown(f"#### {data_source} Feed")
            
            # Process each origin for this data source
            first_origin_processed = False
            for origin in origins:
                # Handle special origins (WASP and Macedonia get single datetime)
                if (origin.lower() == 'wasp-12b' or origin.lower() == 'wasp' or 
                    'wasp-12b[1]' in origin.lower() or 'wasp-12b[2]' in origin.lower()):
                    # Wasp gets assigned to previous Sunday at 18:00
                    # Find the most recent Sunday before report_time
                    report_dt = pd.to_datetime(report_time) if isinstance(report_time, str) else report_time
                    days_since_sunday = report_dt.weekday() + 1  # Monday=0, so Sunday is 6
                    if days_since_sunday == 7:  # If report_time is Sunday
                        days_since_sunday = 0
                    
                    wasp_datetime = report_dt - timedelta(days=days_since_sunday)
                    wasp_datetime = wasp_datetime.replace(hour=18, minute=0, second=0, microsecond=0)
                    
                    # Ensure wasp_datetime is timezone-naive
                    if hasattr(wasp_datetime, 'tz') and wasp_datetime.tz is not None:
                        wasp_datetime = wasp_datetime.replace(tzinfo=None)
                    
                    # Use current data but with modified datetime (single entry)
                    hlc_data_single = find_most_current_data(hlc_df, report_time, origin, scope_days)
                    if hlc_data_single:
                        hlc_data_single['datetime'] = wasp_datetime
                        hlc_data_single['origin'] = f"{origin} (Sunday 18:00)"
                        hlc_data_list = [hlc_data_single]  # Wrap in list for consistent processing
                    else:
                        hlc_data_list = []
                
                elif (origin.lower() == 'macedonia' or 
                      'macedonia[1]' in origin.lower() or 'macedonia[2]' in origin.lower()):
                    # Macedonia follows special schedule (single entry)
                    hlc_data_single = find_most_current_data(hlc_df, report_time, origin, scope_days)
                    if hlc_data_single:
                        hlc_data_single['origin'] = f"{origin} (Special Schedule)"
                        hlc_data_list = [hlc_data_single]  # Wrap in list for consistent processing
                    else:
                        hlc_data_list = []
                
                else:
                    # Regular origins - GET NEW DATA CHANGES ONLY
                    hlc_data_list = find_new_data_changes(hlc_df, report_time, origin, scope_days)
            
                if not hlc_data_list:
                    if not first_origin_processed:  # Only show warning once per data source
                        st.warning(f"No H/L/C data found for {origin} in {data_source}")
                    continue
                
                # DEBUG: Show how many entries found and first entry for first origin
                if not first_origin_processed:
                    st.warning(f"DEBUG: Found {len(hlc_data_list)} datetime entries for {origin}:")
                    if hlc_data_list:
                        first_entry = hlc_data_list[0]
                        st.text(f"  First entry - DateTime: {first_entry['datetime']}")
                        st.text(f"  H: {first_entry['H']} | L: {first_entry['L']} | C: {first_entry['C']}")
                        st.text(f"  Origin: {first_entry['origin']}")
                        if len(hlc_data_list) > 1:
                            st.text(f"  ...and {len(hlc_data_list)-1} more datetime entries")
                    first_origin_processed = True
                
                # Process each datetime entry for this origin
                for hlc_data in hlc_data_list:
                    # Calculate raw M values
                    raw_m_calc = calculate_raw_m_values(hlc_data, range_low, range_high)
                    if not raw_m_calc:
                        st.warning(f"Cannot calculate raw M values for {origin} at {hlc_data['datetime']} (zero spread)")
                        continue
                    
                    # Combine hlc_data with calculation results
                    enhanced_hlc_data = hlc_data.copy()
                    enhanced_hlc_data.update(raw_m_calc)
                
                    # Find valid M values
                    validation_results = find_valid_m_values(
                        measurement_df, 
                        raw_m_calc['raw_m_low'], 
                        raw_m_calc['raw_m_high'],
                        enhanced_hlc_data,
                        range_low, 
                        range_high, 
                        is_high_range,
                        data_source,
                        report_time
                    )
                
                    valid_entries = validation_results['valid_entries']
                    valid_m_list = validation_results['valid_m_list']
                    
                    range_entries.extend(valid_entries)
                
                    # Add to processing summary with updated column structure
                    # Format datetime without timezone for display
                    if hasattr(hlc_data['datetime'], 'strftime'):
                        datetime_str = hlc_data['datetime'].strftime('%m/%d/%Y %H:%M')
                    else:
                        datetime_str = str(hlc_data['datetime'])
                
                    # Count valid M values for this range/origin combo
                    valid_m_count = len(valid_entries) if valid_entries else 0
                    
                    # Format valid list for display
                    valid_list_str = ', '.join([f'{m:.1f}' if isinstance(m, float) else str(m) for m in valid_m_list]) if valid_m_list else 'None'
                
                    processing_summary.append({
                        'Range': f"{range_low:.1f}-{range_high:.1f}",
                        'Feed': data_source.replace(' CSV', ''),  # Use actual data source (Big or Small)
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
    
    # Display M Values from measurement file
    st.markdown("### M Values from Measurement File")
    
    # Debug: Show what columns exist
    st.text(f"Available columns: {', '.join(measurement_df.columns.tolist())}")
    
    # Check for M # column
    m_hash_col = None
    for col in ['M #', 'M#', 'm #', 'm#']:
        if col in measurement_df.columns:
            m_hash_col = col
            break
    
    # Check for M Value column  
    m_value_col = None
    for col in ['M Value', 'M value', 'M_Value', 'M_value', 'm value', 'm_value']:
        if col in measurement_df.columns:
            m_value_col = col
            break
    
    if m_hash_col or m_value_col:
        col1, col2 = st.columns(2)
        
        with col1:
            if m_hash_col:
                m_numbers = measurement_df[m_hash_col].dropna().unique()
                st.text(f"Total M # entries: {len(m_numbers)}")
                m_numbers_str = ', '.join([str(m) for m in sorted(m_numbers)])
                st.text_area("M # List (identifiers):", m_numbers_str, height=100, disabled=True)
            else:
                st.text("M # column not found")
        
        with col2:
            if m_value_col:
                m_values_data = measurement_df[m_value_col].dropna().unique()
                sorted_m_values = sorted([float(m) for m in m_values_data if pd.notna(m)])
                st.text(f"Total M Values: {len(sorted_m_values)}")
                m_values_str = ', '.join([f'{m:.1f}' if isinstance(m, float) else str(m) for m in sorted_m_values])
                st.text_area("M Value List (used for calculations):", m_values_str, height=100, disabled=True)
            else:
                st.text("M Value column not found")
        
        if m_value_col:
            st.info("ℹ️ Note: Raw M calculations compare against 'M Value' column, not 'M #' column")
        else:
            st.warning("⚠️ M Value column not found - raw M calculations may not work properly")
    else:
        st.error("❌ Neither M # nor M Value columns found in measurement file")
    
    # Display processing summary
    if processing_summary:
        st.markdown("### Processing Summary")
        summary_df = pd.DataFrame(processing_summary)
        st.dataframe(summary_df, use_container_width=True)
    
    # Run Model G detection on Grp 1a data if enabled
    if all_valid_entries and run_model_g:
        st.markdown("---")
        st.markdown("### 🟢 Model G Detection on Grp 1a Data")
        
        try:
            # Import Group 1a travelers list
            from a_helpers import GROUP_1A_TRAVELERS
            
            # Convert valid entries to DataFrame
            custom_df = pd.DataFrame(all_valid_entries)
            
            # Filter for Group 1a only (M# values in GROUP_1A_TRAVELERS)
            grp_1a_mask = custom_df['M #'].isin(GROUP_1A_TRAVELERS)
            grp_1a_df = custom_df[grp_1a_mask].copy()
            
            if grp_1a_df.empty:
                st.info("No Group 1a entries found in custom range results for Model G detection")
            else:
                st.info(f"Running Model G detection on {len(grp_1a_df)} Group 1a entries")
                
                # Import the correct Model G function
                try:
                    from models_g_updated import run_model_g_detection
                except ImportError:
                    try:
                        from model_g import run_model_g_detection
                    except ImportError:
                        from model_g_detector import run_model_g_detection
                
                # Run Model G detection on Group 1a data
                g_results = run_model_g_detection(grp_1a_df, report_time, key_suffix="_custom")
                
                # Handle different return types
                if isinstance(g_results, dict) and 'success' in g_results:
                    if g_results['success']:
                        # Display summary
                        summary = g_results['summary']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("o1 (Today)", summary['total_o1'])
                        with col2:
                            st.metric("o2 (Other Day)", summary['total_o2'])
                        with col3:
                            st.metric("Total Sequences", summary['total_sequences'])
                        
                        # Display results table if any sequences found
                        if not g_results['results_df'].empty:
                            st.markdown("#### Grp 1a Model G Results")
                            st.dataframe(g_results['results_df'], use_container_width=True)
                        else:
                            st.info("No Model G sequences detected in Grp 1a data")
                    else:
                        st.error(f"Model G detection error: {g_results['error']}")
                # If function displays results directly, no additional handling needed
                
        except ImportError as e:
            st.warning(f"Model G detection not available: {e}")
        except Exception as e:
            st.error(f"Model G detection error: {str(e)}")
    
    return all_valid_entries

def apply_custom_ranges_advanced(df, small_df, report_time, high1, high2, low1, low2, use_high1, use_high2, use_low1, use_low2, big_df=None, run_model_g=False):
    """
    Apply advanced custom ranges to dataframe.
    
    Returns:
        Filtered dataframe with Range and Zone columns
    """
    # DEBUG: Confirm new module is being used
    st.error("🔥 MODULE RELOADED - NEW VERSION ACTIVE 🔥")
    st.info(f"🧮 Advanced Custom Range Processing Started - {len(df)} measurements to process")
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
        return df
    
    # Process ranges using advanced method
    valid_entries = process_custom_ranges_advanced(df, small_df, report_time, custom_ranges, big_df=big_df, run_model_g=run_model_g)
    
    if not valid_entries:
        st.warning("No valid entries found using advanced custom range calculation")
        return pd.DataFrame()  # Return empty dataframe
    
    # Convert to dataframe
    filtered_df = pd.DataFrame(valid_entries)
    
    # Add Range and Zone columns
    def get_range_name(output_val):
        for range_name, range_config in custom_ranges.items():
            range_value = range_config['value']
            if range_name.startswith('High'):
                range_low = range_value - 24
                range_high = range_value
            else:
                range_low = range_value
                range_high = range_value + 24
            
            if range_low <= output_val <= range_high:
                return range_name
        return 'Other'
    
    def get_zone(output_val):
        for range_name, range_config in custom_ranges.items():
            range_value = range_config['value']
            if range_name.startswith('High'):
                range_low = range_value - 24
                range_high = range_value
                if range_low <= output_val <= range_high:
                    distance = range_high - output_val
                    if distance <= 6:
                        return "0-6"
                    elif distance <= 12:
                        return "6-12"
                    elif distance <= 18:
                        return "12-18"
                    else:
                        return "18-24"
            else:
                range_low = range_value
                range_high = range_value + 24
                if range_low <= output_val <= range_high:
                    distance = output_val - range_low
                    if distance <= 6:
                        return "0-6"
                    elif distance <= 12:
                        return "6-12"
                    elif distance <= 18:
                        return "12-18"
                    else:
                        return "18-24"
        return ""
    
    # Don't overwrite the Zone column - it's already correctly calculated in find_valid_m_values
    filtered_df['Range'] = filtered_df['Output'].apply(get_range_name)
    # Keep the existing Zone column from find_valid_m_values function
    
    return filtered_df

def process_full_range_advanced(
    measurement_df: pd.DataFrame,
    small_df: pd.DataFrame,
    report_time: dt.datetime,
    center: float,
    window_radius: float,
    scope_days: int = 20,
    big_df: pd.DataFrame | None = None,
    run_model_g: bool = False,  # kept for API symmetry
):
    """
    Advanced Full Range processing:
      - Build a single price window [center - radius, center + radius]
      - For each origin/time (using your H/L/C logic), compute raw-M bounds for that window
      - Find valid M values in the measurement file via find_valid_m_values
      - Return the merged list of valid entries (same shape as custom ranges path)
    """
    try:
        lo = center - window_radius
        hi = center + window_radius
        st.info(f"🧮 Full Range (Advanced) window: [{lo}, {hi}] around center={center}")

        all_valid_entries = []
        processing_summary = []

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

        # Include your special variants
        for variant in ['WASP-12b[1]', 'WASP-12b[2]', 'Macedonia[1]', 'Macedonia[2]']:
            all_origins.add(variant)

        origins = list(all_origins)
        if origins:
            st.info(f"Processing origins: {', '.join(origins)}")

        # Iterate the single full window across all sources/origins
        for hlc_df, data_source in data_sources:
            first_origin_logged = False
            for origin in origins:

                # Special handling (same as your custom path)
                if (origin.lower() == 'wasp-12b' or origin.lower() == 'wasp' or
                    'wasp-12b[1]' in origin.lower() or 'wasp-12b[2]' in origin.lower()):
                    report_dt = pd.to_datetime(report_time) if isinstance(report_time, str) else report_time
                    days_since_sunday = report_dt.weekday() + 1  # Monday=0 -> Sunday=6
                    if days_since_sunday == 7:
                        days_since_sunday = 0
                    wasp_dt = report_dt - timedelta(days=days_since_sunday)
                    wasp_dt = wasp_dt.replace(hour=18, minute=0, second=0, microsecond=0)
                    if hasattr(wasp_dt, 'tz') and wasp_dt.tz is not None:
                        wasp_dt = wasp_dt.replace(tzinfo=None)

                    cur = find_most_current_data(hlc_df, report_time, origin, scope_days)
                    if cur:
                        cur['datetime'] = wasp_dt
                        cur['origin'] = f"{origin} (Sunday 18:00)"
                        hlc_data_list = [cur]
                    else:
                        hlc_data_list = []

                elif (origin.lower() == 'macedonia' or
                      'macedonia[1]' in origin.lower() or 'macedonia[2]' in origin.lower()):
                    cur = find_most_current_data(hlc_df, report_time, origin, scope_days)
                    if cur:
                        cur['origin'] = f"{origin} (Special Schedule)"
                        hlc_data_list = [cur]
                    else:
                        hlc_data_list = []
                else:
                    # Regular origins: NEW DATA CHANGES (same as custom)
                    hlc_data_list = find_new_data_changes(hlc_df, report_time, origin, scope_days)

                if not hlc_data_list:
                    if not first_origin_logged:
                        st.warning(f"No H/L/C data found for {origin} in {data_source}")
                        first_origin_logged = True
                    continue

                # Process each datetime entry in this origin
                for hlc_data in hlc_data_list:
                    calc = calculate_raw_m_values(hlc_data, lo, hi)
                    if not calc:
                        # e.g., zero spread
                        continue

                    enhanced = hlc_data.copy()
                    enhanced.update(calc)

                    # We don’t need Zone semantics for full range;
                    # pass False to is_high_range (we’ll drop Zone later).
                    validation = find_valid_m_values(
                        measurement_df,
                        calc['raw_m_low'],
                        calc['raw_m_high'],
                        enhanced,
                        lo,
                        hi,
                        is_high_range=False,
                        data_source=data_source,
                        report_time=report_time
                    )

                    valid_entries = validation.get('valid_entries', [])
                    all_valid_entries.extend(valid_entries)

                    # Optional processing summary (matches your style)
                    if hasattr(hlc_data['datetime'], 'strftime'):
                        dt_str = hlc_data['datetime'].strftime('%m/%d/%Y %H:%M')
                    else:
                        dt_str = str(hlc_data['datetime'])

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
            st.markdown("### Full Range – Processing Summary")
            st.dataframe(pd.DataFrame(processing_summary), use_container_width=True)

        return all_valid_entries

    except Exception as e:
        st.error(f"Error in process_full_range_advanced: {e}")
        return []


def apply_full_range_advanced(
    df: pd.DataFrame,
    small_df: pd.DataFrame,
    report_time: dt.datetime,
    window_radius: float,
    day_start_hour: int = 18,
    input_value_at_start: float | None = None,
    big_df: pd.DataFrame | None = None,
    run_model_g: bool = False,
):
    """
    Apply the advanced Full Range flow (mirrors apply_custom_ranges_advanced):
      - Determine center (prefer input_value_at_start; else derive from small_df).
      - Call process_full_range_advanced to collect valid entries via raw-M windows.
      - Return a dataframe; drop 'Range'/'Zone' columns if present.
    """
    # 1) Center
    center = None
    if input_value_at_start is not None and not pd.isna(input_value_at_start):
        center = float(input_value_at_start)
    else:
        # Derive from small_df: prefer Open @ day start, else last Open/close <= report_time
        try:
            sdf = small_df.copy()
            if 'time' in sdf.columns:
                # Use your existing clean_timestamp
                sdf['time'] = sdf['time'].apply(clean_timestamp)
                if report_time is not None:
                    sdf = sdf[sdf['time'] <= report_time]

            if not sdf.empty:
                # compute day start
                base = dt.datetime(report_time.year, report_time.month, report_time.day, day_start_hour, 0, 0)
                if report_time < base:
                    base = base - dt.timedelta(days=1)

                center_row = sdf[sdf['time'] == base]
                if not center_row.empty:
                    center_row = center_row.iloc[-1]
                else:
                    center_row = sdf.iloc[-1]

                for cand in ('Open', 'open', 'close'):
                    if cand in center_row.index:
                        center = pd.to_numeric(pd.Series(center_row[cand]), errors='coerce').iloc[0]
                        break
        except Exception:
            center = None

    if center is None or pd.isna(center):
        st.error("Full Range (Advanced): could not determine center. Provide input @ day start or ensure small feed has time/Open/close.")
        return pd.DataFrame()

    # 2) Process
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

    # 3) Convert to DF and drop Range/Zone if present
    out_df = pd.DataFrame(valid_entries)
    out_df = out_df.drop(columns=['Range', 'Zone'], errors='ignore')

    # Nice to have: order consistently
    preferred_cols = [
        'Feed','ddd','Arrival','Day','Origin',
        'M Name','M #','M Value','R #','Tag','Family',
        'Input @ 18:00','Diff @ 18:00','Input @ Arrival','Diff @ Arrival',
        'Input @ Report','Diff @ Report','Output'
    ]
    ordered = [c for c in preferred_cols if c in out_df.columns]
    remaining = [c for c in out_df.columns if c not in ordered]
    out_df = out_df[ordered + remaining]

    st.success(f"✅ Full Range (Advanced): {len(out_df)} entries")
    return out_df
