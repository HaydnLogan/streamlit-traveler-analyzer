"""
HOD/LOD Detection and Processing Module
Handles detection of High of Day and Low of Day across multiple trading days
"""

import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import streamlit as st

def get_trading_day_bounds(report_time, day_start_hour=18):
    """
    Get the start and end times for a trading day.
    Trading day starts at day_start_hour and ends at 16:45 the next calendar day.
    
    Returns: (start_time, end_time)
    """
    # If report_time is before day_start_hour, the trading day started yesterday
    if report_time.hour < day_start_hour:
        day_start = report_time.replace(hour=day_start_hour, minute=0, second=0, microsecond=0) - timedelta(days=1)
    else:
        day_start = report_time.replace(hour=day_start_hour, minute=0, second=0, microsecond=0)
    
    # End is 16:45 the next calendar day
    day_end = (day_start + timedelta(days=1)).replace(hour=16, minute=45, second=0, microsecond=0)
    
    return day_start, day_end

def is_complete_trading_day(report_time, day_start_hour=18):
    """
    Determine if the most recent trading day is complete.
    A day is complete if report_time >= 16:45 after the day start.
    """
    day_start, day_end = get_trading_day_bounds(report_time, day_start_hour)
    return report_time >= day_end

def find_hod_lod_for_day(small_df, big_df, day_start, day_end):
    """
    Find HOD and LOD for a specific trading day across both feeds.
    
    Returns: dict with HOD and LOD info including which feed each came from
    """
    def filter_day_data(df, day_start, day_end):
        """Filter dataframe to trading day window"""
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        return df[(df['time'] >= day_start) & (df['time'] <= day_end)]
    
    # Filter both feeds to the trading day
    small_day = filter_day_data(small_df, day_start, day_end)
    big_day = filter_day_data(big_df, day_start, day_end)
    
    results = {
        'day_start': day_start,
        'day_end': day_end,
        'hod': None,
        'hod_time': None,
        'hod_feed': None,
        'lod': None,
        'lod_time': None,
        'lod_feed': None,
        'small_hod': None,
        'small_hod_time': None,
        'small_lod': None,
        'small_lod_time': None,
        'big_hod': None,
        'big_hod_time': None,
        'big_lod': None,
        'big_lod_time': None
    }
    
    # Find small feed HOD/LOD
    if not small_day.empty and 'high' in small_day.columns and 'low' in small_day.columns:
        small_high_idx = small_day['high'].idxmax()
        small_low_idx = small_day['low'].idxmin()
        results['small_hod'] = small_day.loc[small_high_idx, 'high']
        results['small_hod_time'] = small_day.loc[small_high_idx, 'time']
        results['small_lod'] = small_day.loc[small_low_idx, 'low']
        results['small_lod_time'] = small_day.loc[small_low_idx, 'time']
    
    # Find big feed HOD/LOD
    if not big_day.empty and 'high' in big_day.columns and 'low' in big_day.columns:
        big_high_idx = big_day['high'].idxmax()
        big_low_idx = big_day['low'].idxmin()
        results['big_hod'] = big_day.loc[big_high_idx, 'high']
        results['big_hod_time'] = big_day.loc[big_high_idx, 'time']
        results['big_lod'] = big_day.loc[big_low_idx, 'low']
        results['big_lod_time'] = big_day.loc[big_low_idx, 'time']
    
    # Determine overall HOD (highest of both feeds)
    if results['small_hod'] is not None and results['big_hod'] is not None:
        if results['small_hod'] >= results['big_hod']:
            results['hod'] = results['small_hod']
            results['hod_time'] = results['small_hod_time']
            results['hod_feed'] = 'small'
        else:
            results['hod'] = results['big_hod']
            results['hod_time'] = results['big_hod_time']
            results['hod_feed'] = 'big'
    elif results['small_hod'] is not None:
        results['hod'] = results['small_hod']
        results['hod_time'] = results['small_hod_time']
        results['hod_feed'] = 'small'
    elif results['big_hod'] is not None:
        results['hod'] = results['big_hod']
        results['hod_time'] = results['big_hod_time']
        results['hod_feed'] = 'big'
    
    # Determine overall LOD (lowest of both feeds)
    if results['small_lod'] is not None and results['big_lod'] is not None:
        if results['small_lod'] <= results['big_lod']:
            results['lod'] = results['small_lod']
            results['lod_time'] = results['small_lod_time']
            results['lod_feed'] = 'small'
        else:
            results['lod'] = results['big_lod']
            results['lod_time'] = results['big_lod_time']
            results['lod_feed'] = 'big'
    elif results['small_lod'] is not None:
        results['lod'] = results['small_lod']
        results['lod_time'] = results['small_lod_time']
        results['lod_feed'] = 'small'
    elif results['big_lod'] is not None:
        results['lod'] = results['big_lod']
        results['lod_time'] = results['big_lod_time']
        results['lod_feed'] = 'big'
    
    return results

def get_multiple_days_hod_lod(small_df, big_df, report_time, num_days, 
                               include_partial_day, day_start_hour=18):
    """
    Get HOD/LOD for multiple trading days working backwards from report_time.
    
    Returns: list of dicts, one per day, with HOD/LOD information
    """
    days_data = []
    
    # Determine if current day is complete
    current_day_complete = is_complete_trading_day(report_time, day_start_hour)
    
    # Start from most recent complete day
    if current_day_complete or include_partial_day:
        # Include current/partial day
        current_start, current_end = get_trading_day_bounds(report_time, day_start_hour)
        # Use report_time as end if day not complete
        if not current_day_complete:
            current_end = report_time
        days_to_process = num_days
        latest_end = current_end
    else:
        # Skip current partial day, start from previous complete day
        day_start, _ = get_trading_day_bounds(report_time, day_start_hour)
        previous_day_start = day_start - timedelta(days=1)
        latest_end = previous_day_start.replace(hour=16, minute=45, second=0, microsecond=0) + timedelta(days=1)
        days_to_process = num_days
    
    # Work backwards to collect the requested number of days
    for i in range(days_to_process):
        # Calculate day boundaries
        if i == 0 and (current_day_complete or include_partial_day):
            day_end = current_end if not current_day_complete else latest_end
            day_start = get_trading_day_bounds(report_time, day_start_hour)[0]
        else:
            # Go back one day at a time
            if i == 0:
                day_end = latest_end
            else:
                day_end = days_data[-1]['day_start'] - timedelta(seconds=1)
                day_end = day_end.replace(hour=16, minute=45, second=0, microsecond=0)
            
            day_start = (day_end - timedelta(days=1)).replace(hour=day_start_hour, minute=0, second=0, microsecond=0)
        
        # Find HOD/LOD for this day
        day_info = find_hod_lod_for_day(small_df, big_df, day_start, day_end)
        day_info['day_index'] = i
        day_info['is_partial'] = (i == 0 and not current_day_complete and include_partial_day)
        days_data.append(day_info)
    
    return days_data

def apply_hod_lod_cutoff(df, hod_lod_time, scope_days=20):
    """
    Filter dataframe to only include arrivals within scope_days before 
    the strict 15-minute cutoff before HOD/LOD time.
    
    Args:
        df: DataFrame with 'Arrival' or 'Arrival_datetime' column
        hod_lod_time: datetime of HOD or LOD
        scope_days: number of days to look back from cutoff
    
    Returns: filtered DataFrame
    """
    if df is None or df.empty:
        return df
    
    # Calculate cutoff time (15 minutes before HOD/LOD)
    cutoff_time = hod_lod_time - timedelta(minutes=15)
    
    # Calculate scope start (scope_days before cutoff)
    scope_start = cutoff_time - timedelta(days=scope_days)
    
    # Determine which time column to use
    if 'Arrival_datetime' in df.columns:
        time_col = 'Arrival_datetime'
    elif 'Arrival' in df.columns:
        df['Arrival_datetime'] = pd.to_datetime(df['Arrival'], errors='coerce')
        time_col = 'Arrival_datetime'
    else:
        return df  # No time column found
    
    # Apply filter: arrivals between scope_start and cutoff_time (inclusive)
    filtered = df[(df[time_col] >= scope_start) & (df[time_col] <= cutoff_time)].copy()
    
    return filtered

def process_hod_lod_mode(measurement_df, small_df, big_df, report_time, num_days,
                         include_partial_day, scope_days, day_start_hour):
    """
    Main processing function for HOD/LOD mode.
    
    Returns: dict of DataFrames, one per day (key format: "Day_0", "Day_1", etc.)
    """
    from custom_range_calculator_0813 import apply_custom_ranges_advanced
    
    # Get HOD/LOD for multiple days
    days_data = get_multiple_days_hod_lod(
        small_df, big_df, report_time, num_days,
        include_partial_day, day_start_hour
    )
    
    if not days_data:
        st.warning("No HOD/LOD data found for the specified days")
        return {}
    
    results = {}
    
    # Process each day
    for day_info in days_data:
        day_idx = day_info['day_index']
        day_date = day_info['day_start'].strftime('%Y-%m-%d')
        is_partial = day_info.get('is_partial', False)
        
        # Skip if no HOD or LOD found
        if day_info['hod'] is None or day_info['lod'] is None:
            st.warning(f"Skipping {day_date}: HOD or LOD not found")
            continue
        
        # Use custom ranges path with HOD as High 1 and LOD as Low 1
        hod_value = day_info['hod']
        lod_value = day_info['lod']
        
        st.info(f"Processing {day_date} {'(Partial Day)' if is_partial else ''}: HOD={hod_value:.2f}, LOD={lod_value:.2f}")
        
        # Process using custom ranges advanced
        day_df = apply_custom_ranges_advanced(
            df=measurement_df,
            small_df=small_df,
            report_time=day_info['day_end'],  # Use day end as report time
            high1=hod_value,
            high2=0,
            low1=lod_value,
            low2=0,
            use_high1=True,
            use_high2=False,
            use_low1=True,
            use_low2=False,
            big_df=big_df,
            run_model_g=False,
            day_start_hour=day_start_hour
        )
        
        if day_df is None or day_df.empty:
            st.warning(f"No travelers found for {day_date}")
            continue
        
        # Apply strict 15-minute cutoff for HOD
        hod_df = day_df[day_df['Range'].str.contains('High', na=False)].copy()
        if not hod_df.empty:
            hod_df = apply_hod_lod_cutoff(hod_df, day_info['hod_time'], scope_days)
            
            # Add "Actual HOD/LOD" column
            hod_df['Actual HOD/LOD'] = hod_df.apply(
                lambda row: 'HOD' if (
                    (row['Feed'] == 'Small' and abs(row['Output'] - day_info['small_hod']) < 0.01) or
                    (row['Feed'] == 'Big' and abs(row['Output'] - day_info['big_hod']) < 0.01)
                ) else '',
                axis=1
            )
        
        # Apply strict 15-minute cutoff for LOD
        lod_df = day_df[day_df['Range'].str.contains('Low', na=False)].copy()
        if not lod_df.empty:
            lod_df = apply_hod_lod_cutoff(lod_df, day_info['lod_time'], scope_days)
            
            # Add "Actual HOD/LOD" column
            lod_df['Actual HOD/LOD'] = lod_df.apply(
                lambda row: 'LOD' if (
                    (row['Feed'] == 'Small' and abs(row['Output'] - day_info['small_lod']) < 0.01) or
                    (row['Feed'] == 'Big' and abs(row['Output'] - day_info['big_lod']) < 0.01)
                ) else '',
                axis=1
            )
        
        # Combine HOD and LOD data
        combined_df = pd.concat([hod_df, lod_df], ignore_index=True)
        
        if not combined_df.empty:
            # Sort by Output (descending) and Arrival (ascending)
            if 'Output' in combined_df.columns and 'Arrival' in combined_df.columns:
                combined_df = combined_df.sort_values(['Output', 'Arrival'], ascending=[False, True])
            
            # Store with descriptive key
            key = f"Day_{day_idx}_{day_date}{'_Partial' if is_partial else ''}"
            results[key] = combined_df
            
            st.success(f"Day {day_idx} ({day_date}): {len(combined_df)} travelers found")
    
    return results
