"""
FAST Custom Reciprocal Traveler Report Generator

KEY OPTIMIZATIONS:
1. Only generates travelers at origin UPDATE TIMES (not every bar)
2. Lookback parameter (default 20 days like App 31)
3. Filters to specific times per origin (18:00, 03:15, etc.)

Performance:
- OLD: ~159,000 travelers from every 15-min bar (10+ minutes)
- NEW: ~50-100 travelers per day from update times only (<10 seconds)
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Origin update times (when travelers are generated)
ORIGIN_UPDATE_TIMES = {
    'spain': [(3, 15), (9, 30), (15, 45)],  # 03:15, 09:30, 15:45
    'jupiter': [(6, 0), (12, 0), (18, 0)],   # 06:00, 12:00, 18:00
    'saturn': [(0, 0), (8, 0), (16, 0)],     # 00:00, 08:00, 16:00
    'trinidad': [(18, 0)],                   # 18:00 only
    'tobago': [(18, 0)],                     # 18:00 only
    'kepler-44': [(18, 0)],                  # 18:00 only
    'kepler-62': [(18, 0)],                  # 18:00 only
    'wasp-12b': [(18, 0)],                   # 18:00 for daily (weekly/monthly handled separately)
    'macedonia': [(18, 0)]                   # 18:00 for daily
}

# Origin types
EPIC_ORIGINS = {'trinidad', 'tobago', 'wasp-12b', 'wasp-12b[1]', 'wasp-12b[2]', 
                'macedonia', 'macedonia[1]', 'macedonia[2]'}
ANCHOR_ORIGINS = {'spain', 'saturn', 'jupiter', 'kepler-62', 'kepler-44'}

# Recipe M# pairs from Model G.11
RECIPE_PAIRS = {
    'GR': [(30, 50)],
    'X0': [(22, 60), (14, 68), (10, 77), (6, 87), (5, 96), (3, 103), (2, 107), (1, 111)],
    'X1': [(36, 43), (26, 55)],
    'X2': [(39, 41)]
}

# Flatten to get all Recipe M# values
ALL_RECIPE_M_NUMBERS = set()
for pair_list in RECIPE_PAIRS.values():
    for m1, m2 in pair_list:
        ALL_RECIPE_M_NUMBERS.add(m1)
        ALL_RECIPE_M_NUMBERS.add(-m1)
        ALL_RECIPE_M_NUMBERS.add(m2)
        ALL_RECIPE_M_NUMBERS.add(-m2)


def calculate_pivot(h: float, l: float, c: float, m_value: int) -> Optional[float]:
    """Calculate traveler output using pivot formula."""
    if pd.isna(h) or pd.isna(l) or pd.isna(c):
        return None
    
    pivot = (h + l + c) / 3
    spread = h - l
    output = pivot + (m_value * spread)
    
    return output


def extract_origins_from_hlc(df: pd.DataFrame) -> Dict[str, Tuple[str, str, str]]:
    """Extract origin names and their H/L/C columns from HLC data."""
    origins = {}
    
    for col in df.columns:
        if col.endswith(' H'):
            origin_name = col[:-2]
            h_col = col
            l_col = f"{origin_name} L"
            c_col = f"{origin_name} C"
            
            if l_col in df.columns and c_col in df.columns:
                origins[origin_name] = (h_col, l_col, c_col)
    
    return origins


def is_anchor_or_epic(origin_name: str) -> bool:
    """Check if origin is Anchor or Epic."""
    origin_lower = origin_name.lower()
    return origin_lower in ANCHOR_ORIGINS or origin_lower in EPIC_ORIGINS


def get_origin_type(origin_name: str) -> str:
    """Get origin type (Anchor or Epic)."""
    origin_lower = origin_name.lower()
    if origin_lower in ANCHOR_ORIGINS:
        return 'Anchor'
    elif origin_lower in EPIC_ORIGINS:
        return 'Epic'
    return 'Other'


def get_origin_update_times(origin_name: str) -> List[Tuple[int, int]]:
    """Get update times for an origin."""
    origin_lower = origin_name.lower()
    
    # Handle bracketed origins (WASP-12b[1], Macedonia[2], etc.)
    base_origin = origin_lower.split('[')[0]
    
    return ORIGIN_UPDATE_TIMES.get(base_origin, [(18, 0)])  # Default to 18:00


def get_last_trading_day(report_time: datetime) -> datetime:
    """Get the last trading day before report_time."""
    prev_day = report_time - timedelta(days=1)
    
    # If report_time is early Monday (before 8am), go back to Friday
    if report_time.weekday() == 0 and report_time.hour < 8:
        prev_day = report_time - timedelta(days=3)
    # If report_time is Sunday, go back to Friday
    elif report_time.weekday() == 6:
        prev_day = report_time - timedelta(days=2)
    
    # Keep going back while prev_day is weekend
    while prev_day.weekday() >= 5:
        prev_day -= timedelta(days=1)
    
    return prev_day


def generate_custom_recip_report_FAST(
    hlc_df: pd.DataFrame,
    measurement_df: pd.DataFrame,
    report_time: datetime,
    feed_label: str = 'Small',
    lookback_days: int = 20
) -> pd.DataFrame:
    """
    FAST version: Only generates travelers at origin update times.
    
    Parameters:
    -----------
    hlc_df : DataFrame
        Raw HLC data
    measurement_df : DataFrame
        Measurement file with 'M #' and 'R #' columns
    report_time : datetime
        Time to generate report for
    feed_label : str
        'Small' or 'Big'
    lookback_days : int
        How many days to look back (default 20, like App 31)
    
    Returns:
    --------
    DataFrame with traveler data
    """
    
    print(f"\n{'='*80}")
    print(f"GENERATING FAST CUSTOM RECIP REPORT - {feed_label.upper()} FEED")
    print(f"{'='*80}")
    print(f"Report Time: {report_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Lookback: {lookback_days} days")
    print(f"{'='*80}\n")
    
    # Ensure time column is datetime
    if 'time' not in hlc_df.columns:
        time_col = hlc_df.columns[0]
        hlc_df = hlc_df.rename(columns={time_col: 'time'})
    
    # Handle mixed timezones
    hlc_df['time'] = pd.to_datetime(hlc_df['time'], errors='coerce', utc=True)
    hlc_df['time'] = hlc_df['time'].dt.tz_localize(None)
    
    # Extract origins
    origins = extract_origins_from_hlc(hlc_df)
    print(f"📊 Found {len(origins)} origins in HLC data")
    
    # Filter to Anchor/Epic
    filtered_origins = {name: cols for name, cols in origins.items() 
                       if is_anchor_or_epic(name)}
    
    print(f"✅ Filtered to {len(filtered_origins)} Anchor/Epic origins: {list(filtered_origins.keys())}")
    
    # Filter measurements to Recipe M# values only
    recipe_measurements = measurement_df[
        measurement_df['M #'].isin(ALL_RECIPE_M_NUMBERS)
    ].copy()
    
    print(f"✅ Filtered to {len(recipe_measurements)} Recipe M# values")
    
    # Calculate date range
    end_date = report_time.date()
    start_date = end_date - timedelta(days=lookback_days)
    
    print(f"📅 Date range: {start_date} to {end_date} ({lookback_days} days)")
    
    # Filter HLC data to lookback window
    hlc_df['date'] = hlc_df['time'].dt.date
    hlc_df_filtered = hlc_df[
        (hlc_df['date'] >= start_date) & 
        (hlc_df['date'] <= end_date)
    ].copy()
    
    print(f"📊 Filtered to {len(hlc_df_filtered)} rows within lookback window")
    
    # Generate travelers ONLY at origin update times
    travelers = []
    total_updates = 0
    
    for origin_name, (h_col, l_col, c_col) in filtered_origins.items():
        origin_type = get_origin_type(origin_name)
        update_times = get_origin_update_times(origin_name)
        
        print(f"\n  Processing {origin_name} ({origin_type}):")
        print(f"    Update times: {update_times}")
        
        origin_traveler_count = 0
        
        # For each update time
        for hour, minute in update_times:
            # Find all rows that match this update time
            matching_rows = hlc_df_filtered[
                (hlc_df_filtered['time'].dt.hour == hour) &
                (hlc_df_filtered['time'].dt.minute == minute)
            ]
            
            if len(matching_rows) == 0:
                continue
            
            # For each matching row
            for _, row in matching_rows.iterrows():
                h = row[h_col]
                l = row[l_col]
                c = row[c_col]
                arrival_time = row['time']
                
                if pd.isna(h) or pd.isna(l) or pd.isna(c):
                    continue
                
                # Calculate day offset
                days_diff = (end_date - arrival_time.date()).days
                day_label = f"[{-days_diff}]" if days_diff > 0 else "[0]"
                
                # Generate travelers for each Recipe M#
                for _, m_row in recipe_measurements.iterrows():
                    m_num = m_row['M #']
                    r_num = m_row['R #']
                    
                    output = calculate_pivot(h, l, c, m_num)
                    
                    if output is not None:
                        travelers.append({
                            'M #': m_num,
                            'R #': r_num,
                            'Origin': origin_name,
                            'Output': output,
                            'Arrival': arrival_time,
                            'Day': day_label,
                            'Feed': feed_label,
                            'Origin_Type': origin_type
                        })
                        origin_traveler_count += 1
                        total_updates += 1
        
        print(f"    Generated {origin_traveler_count} travelers")
    
    print(f"\n✅ Total travelers generated: {total_updates}")
    
    return pd.DataFrame(travelers)


def find_reciprocal_matches(traveler_df: pd.DataFrame, max_spread: float = 3.0) -> List[Dict]:
    """Find reciprocal matches where M1's R# = M2's M# and outputs are close."""
    
    print(f"\n{'='*80}")
    print(f"FINDING RECIPROCAL MATCHES")
    print(f"{'='*80}")
    print(f"Max Spread: {max_spread} points")
    print(f"Total Travelers to search: {len(traveler_df)}")
    print(f"{'='*80}\n")
    
    matches = []
    
    # Group by Feed (same-feed matching only)
    for feed in traveler_df['Feed'].unique():
        feed_travelers = traveler_df[traveler_df['Feed'] == feed]
        
        print(f"Searching {feed} feed: {len(feed_travelers)} travelers")
        
        # For each traveler, find its reciprocal
        for idx1, t1 in feed_travelers.iterrows():
            # Find travelers where:
            # - Their M# equals this traveler's R#
            # - Their R# equals this traveler's M#
            # - Output spread <= max_spread
            # - Same feed
            
            potential_matches = feed_travelers[
                (feed_travelers['M #'] == t1['R #']) &
                (feed_travelers['R #'] == t1['M #']) &
                (feed_travelers.index > idx1)  # Avoid duplicates
            ]
            
            for idx2, t2 in potential_matches.iterrows():
                output_spread = abs(t1['Output'] - t2['Output'])
                
                if output_spread <= max_spread:
                    zone_price = (t1['Output'] + t2['Output']) / 2
                    
                    matches.append({
                        'Origin1': t1['Origin'],
                        'M1': t1['M #'],
                        'R1': t1['R #'],
                        'Output1': t1['Output'],
                        'Arrival1': t1['Arrival'],  # FIXED: Added arrival time
                        'Day1': t1['Day'],
                        'Origin2': t2['Origin'],
                        'M2': t2['M #'],
                        'R2': t2['R #'],
                        'Output2': t2['Output'],
                        'Arrival2': t2['Arrival'],  # FIXED: Added arrival time
                        'Day2': t2['Day'],
                        'Output_Spread': output_spread,
                        'Zone_Price': zone_price,
                        'Feed': feed
                    })
        
        print(f"  Found {sum(1 for m in matches if m['Feed'] == feed)} reciprocal matches\n")
    
    print(f"✅ Total reciprocal matches found: {len(matches)}\n")
    
    return matches


def generate_recip_traveler_reports(
    small_hlc_df: pd.DataFrame,
    big_hlc_df: pd.DataFrame,
    measurement_df: pd.DataFrame,
    report_time: datetime,
    max_spread: float = 3.0,
    lookback_days: int = 20
) -> Dict:
    """
    Generate reciprocal traveler reports for both feeds (FAST version).
    
    Returns dict with:
    - small_report: DataFrame
    - big_report: DataFrame
    - small_matches: List[Dict]
    - big_matches: List[Dict]
    - combined_travelers: DataFrame
    """
    
    print(f"\n{'='*80}")
    print(f"GENERATING CUSTOM RECIPROCAL TRAVELER REPORTS (FAST)")
    print(f"{'='*80}")
    print(f"Report Time: {report_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Max Spread: {max_spread} points")
    print(f"Lookback: {lookback_days} days")
    print(f"{'='*80}\n")
    
    # Generate reports
    small_report = generate_custom_recip_report_FAST(
        small_hlc_df, measurement_df, report_time, 'Small', lookback_days
    )
    
    big_report = generate_custom_recip_report_FAST(
        big_hlc_df, measurement_df, report_time, 'Big', lookback_days
    )
    
    # Find reciprocal matches
    small_matches = find_reciprocal_matches(small_report, max_spread)
    big_matches = find_reciprocal_matches(big_report, max_spread)
    
    # Combine travelers
    combined_travelers = pd.concat([small_report, big_report], ignore_index=True)
    
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Small Feed: {len(small_report)} travelers, {len(small_matches)} matches")
    print(f"Big Feed: {len(big_report)} travelers, {len(big_matches)} matches")
    print(f"Combined: {len(combined_travelers)} travelers")
    print(f"{'='*80}\n")
    
    return {
        'small_report': small_report,
        'big_report': big_report,
        'small_matches': small_matches,
        'big_matches': big_matches,
        'combined_travelers': combined_travelers
    }
