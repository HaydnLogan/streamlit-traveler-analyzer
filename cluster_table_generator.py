"""
Cluster Table Generator - FOGZ, Large Discounts, Recips PD
============================================================
Generates three cluster tables modeled after Model G.11:
1. FOGZ - M#{0, 1, 2, 3, 5, 6} matched against PwX2_1_0
2. Large Discounts - M#{36, 39} matched against PX2_1_0
3. Recips PD - DRecip list matched with R# mates

All tables use 'new' data rows only (no duplicates/old data)
Data window: From report_time back to previous trading day start (18:00)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONSTANTS
# ============================================================================

# FOGZ M# values (include 0)
FOGZ_M_NUMBERS = {0, -1, 1, -2, 2, -3, 3, -5, 5, -6, 6}

# PwX2_1_0 - Used for FOGZ matching
PwX2_1_0 = {-40, 40, -41, 41, -43, 43, -50, 50, -55, 55, -60, 60, -68, 68, 
            -77, 77, -87, 87, -96, 96, -103, 103, -107, 107, -111, 111}

# Large Discount M# values
LARGE_DISCOUNT_M = {-36, 36, -39, 39}

# PX2_1_0 - Used for Large Discount matching
PX2_1_0 = {-41, 41, -43, 43, -50, 50, -55, 55, -60, 60, -68, 68, 
           -77, 77, -87, 87, -96, 96, -103, 103, -107, 107, -111, 111}

# DRecip List - Recips with abs value < 40
DRECIP_LIST = {-1, 1, -2, 2, -3, 3, -5, 5, -6, 6, -10, 10, -14, 14, -15, 15,
               -22, 22, -27, 27, -30, 30, -36, 36, -38, 38, -39, 39}

# Reciprocal Pairs
RECIP_X0_PAIRS = [(30, 50), (22, 60), (14, 68), (10, 77), (6, 87), (5, 96), 
                  (3, 103), (2, 107), (1, 111)]
RECIP_XD0_PAIRS = [(27, 54), (15, 67)]
RECIP_X1_PAIRS = [(36, 43), (26, 55)]
RECIP_XD1_PAIRS = [(33, 45)]
RECIP_X2_PAIRS = [(39, 41)]
RECIP_XD2_PAIRS = [(38, 42)]

ALL_RECIP_PAIRS = (RECIP_X0_PAIRS + RECIP_XD0_PAIRS + RECIP_X1_PAIRS + 
                   RECIP_XD1_PAIRS + RECIP_X2_PAIRS + RECIP_XD2_PAIRS)

# Create reciprocal map for quick lookup
RECIP_MAP = {}
for pair in ALL_RECIP_PAIRS:
    m1, m2 = pair
    RECIP_MAP[m1] = m2
    RECIP_MAP[-m1] = -m2
    RECIP_MAP[m2] = m1
    RECIP_MAP[-m2] = -m1

# Origin update times
ORIGIN_UPDATE_TIMES = {
    'spain': [(3, 15), (9, 30), (15, 45)],
    'jupiter': [(6, 0), (12, 0), (18, 0)],
    'saturn': [(0, 0), (8, 0), (16, 0)],
    'trinidad': [(18, 0)],
    'tobago': [(18, 0)],
    'kepler-44': [(18, 0)],
    'kepler-62': [(18, 0)],
    'wasp-12b': [(18, 0)],
    'macedonia': [(18, 0)]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_pivot(h: float, l: float, c: float, m_value: int) -> Optional[float]:
    """Calculate traveler output using pivot formula."""
    if pd.isna(h) or pd.isna(l) or pd.isna(c):
        return None
    
    pivot = (h + l + c) / 3
    spread = h - l
    output = pivot + (m_value * spread)
    
    return output


def get_origin_update_times(origin_name: str) -> List[Tuple[int, int]]:
    """Get update times for an origin."""
    origin_lower = origin_name.lower().split('[')[0]
    return ORIGIN_UPDATE_TIMES.get(origin_lower, [(18, 0)])


def get_data_window(hlc_df: pd.DataFrame, report_time: datetime) -> pd.DataFrame:
    """
    Get data window: From report_time back to previous trading day start (18:00).
    
    Example: Report time 11/20/25 18:00
    - Grab 18:00 row (current day [0])
    - Grab all rows back to previous 18:00 (typically 93 rows on 15m data)
    
    Returns DataFrame with rows from current and previous trading day.
    """
    # Ensure time column
    if 'time' not in hlc_df.columns:
        hlc_df = hlc_df.rename(columns={hlc_df.columns[0]: 'time'})
    
    # Convert to datetime
    hlc_df['time'] = pd.to_datetime(hlc_df['time'], utc=True, errors='coerce')
    hlc_df['time'] = hlc_df['time'].dt.tz_localize(None)
    
    # Sort by time
    hlc_df = hlc_df.sort_values('time')
    
    # Find report_time row
    report_time_naive = report_time.replace(tzinfo=None)
    
    # Get all rows up to and including report_time
    rows_up_to_report = hlc_df[hlc_df['time'] <= report_time_naive]
    
    if len(rows_up_to_report) == 0:
        return pd.DataFrame()
    
    # Find previous trading day start (18:00)
    # Trading day starts at 18:00
    previous_day_start = report_time_naive - timedelta(days=1)
    previous_day_start = previous_day_start.replace(hour=18, minute=0, second=0, microsecond=0)
    
    # If report_time is exactly 18:00, we want rows from this 18:00 and previous 18:00
    # If report_time is after 18:00 (e.g., 19:30), we want from current day's 18:00
    if report_time_naive.hour >= 18:
        current_day_start = report_time_naive.replace(hour=18, minute=0, second=0, microsecond=0)
    else:
        # Before 18:00, current day start is yesterday's 18:00
        current_day_start = previous_day_start
        previous_day_start = previous_day_start - timedelta(days=1)
    
    # Get rows from previous_day_start to report_time
    window_df = hlc_df[
        (hlc_df['time'] >= previous_day_start) & 
        (hlc_df['time'] <= report_time_naive)
    ].copy()
    
    return window_df


def filter_new_data_rows(window_df: pd.DataFrame, origin_cols: Dict[str, Tuple[str, str, str]]) -> Dict[str, pd.DataFrame]:
    """
    Filter to 'new' data rows only per origin.
    Removes duplicate/old data rows, keeps only rows where origin has updated.
    
    Returns dict: {origin_name: DataFrame of new rows}
    """
    new_data_per_origin = {}
    
    for origin_name, (h_col, l_col, c_col) in origin_cols.items():
        update_times = get_origin_update_times(origin_name)
        
        # Filter to rows that match update times
        new_rows = []
        for hour, minute in update_times:
            matching = window_df[
                (window_df['time'].dt.hour == hour) &
                (window_df['time'].dt.minute == minute)
            ]
            if len(matching) > 0:
                new_rows.append(matching)
        
        if new_rows:
            origin_new_df = pd.concat(new_rows).sort_values('time')
            
            # Remove duplicates - keep only if H/L/C values changed
            origin_new_df = origin_new_df.copy()
            origin_new_df['hlc_combo'] = (
                origin_new_df[h_col].astype(str) + '_' +
                origin_new_df[l_col].astype(str) + '_' +
                origin_new_df[c_col].astype(str)
            )
            
            # Keep only first occurrence of each unique HLC combo
            origin_new_df = origin_new_df.drop_duplicates(subset='hlc_combo', keep='first')
            origin_new_df = origin_new_df.drop('hlc_combo', axis=1)
            
            new_data_per_origin[origin_name] = origin_new_df
    
    return new_data_per_origin


def calculate_outputs_for_m_numbers(
    new_data_per_origin: Dict[str, pd.DataFrame],
    origin_cols: Dict[str, Tuple[str, str, str]],
    m_numbers: set,
    report_time: datetime,
    feed_label: str
) -> pd.DataFrame:
    """
    Calculate outputs for given M# set across all new data rows.
    
    Returns DataFrame with: Origin, M#, R#, Output, Arrival, Day, Feed
    """
    travelers = []
    
    report_date = report_time.date()
    
    for origin_name, origin_df in new_data_per_origin.items():
        if origin_name not in origin_cols:
            continue
        
        h_col, l_col, c_col = origin_cols[origin_name]
        
        for _, row in origin_df.iterrows():
            h = row[h_col]
            l = row[l_col]
            c = row[c_col]
            arrival_time = row['time']
            
            if pd.isna(h) or pd.isna(l) or pd.isna(c):
                continue
            
            # Calculate day bracket
            arrival_date = arrival_time.date()
            days_diff = (report_date - arrival_date).days
            day_label = f"[{-days_diff}]" if days_diff > 0 else "[0]"
            
            # Determine type
            if days_diff == 0:
                arrival_type = 'Today'
            elif days_diff <= 2:
                arrival_type = 'Recent'
            else:
                arrival_type = 'Older'
            
            # Calculate outputs for each M#
            for m_num in m_numbers:
                output = calculate_pivot(h, l, c, m_num)
                
                if output is not None:
                    travelers.append({
                        'Origin': origin_name,
                        'M #': m_num,
                        'R #': 0,  # Will be filled later if needed
                        'Output': output,
                        'Arrival': arrival_time,
                        'Day': day_label,
                        'Type': arrival_type,
                        'Feed': feed_label
                    })
    
    return pd.DataFrame(travelers)


def find_matches_within_spread(
    arrival_df: pd.DataFrame,
    match_df: pd.DataFrame,
    max_spread: float = 3.0
) -> List[Dict]:
    """
    Find matches where match_df outputs are within max_spread of arrival_df outputs.
    
    Returns list of match dictionaries.
    """
    matches = []
    
    for _, arrival_row in arrival_df.iterrows():
        arrival_output = arrival_row['Output']
        arrival_m = arrival_row['M #']
        arrival_origin = arrival_row['Origin']
        arrival_time = arrival_row['Arrival']
        arrival_day = arrival_row['Day']
        arrival_feed = arrival_row['Feed']
        
        # Find matches within spread
        potential_matches = match_df[
            abs(match_df['Output'] - arrival_output) <= max_spread
        ]
        
        for _, match_row in potential_matches.iterrows():
            match_output = match_row['Output']
            match_m = match_row['M #']
            match_origin = match_row['Origin']
            match_time = match_row['Arrival']
            match_day = match_row['Day']
            match_feed = match_row['Feed']
            
            # Calculate spread
            output_spread = abs(arrival_output - match_output)
            avg_output = (arrival_output + match_output) / 2
            
            # Determine combined type
            if arrival_row['Type'] == 'Today' or match_row['Type'] == 'Today':
                combined_type = 'Today'
            elif arrival_row['Type'] == 'Recent' or match_row['Type'] == 'Recent':
                combined_type = 'Recent'
            else:
                combined_type = 'Older'
            
            matches.append({
                'Arrival_Output': avg_output,
                'Arrival_DateTime': arrival_time,
                'Arrival_Bracket': arrival_day,
                'Type': combined_type,
                'Category': f"M#{arrival_m} + M#{match_m}",
                'Origins': f"{arrival_origin}, {match_origin}",
                'Feed': f"{arrival_feed}+{match_feed}" if arrival_feed != match_feed else arrival_feed,
                'M_#s': f"{arrival_m}, {match_m}",
                'Outputs': f"{arrival_output:.2f}, {match_output:.2f}",
                'Prox': output_spread,
                'Arrival_M': arrival_m,
                'Match_M': match_m,
                'Arrival_Origin': arrival_origin,
                'Match_Origin': match_origin
            })
    
    return matches


# ============================================================================
# FOGZ CLUSTER TABLE
# ============================================================================

def generate_fogz_table(
    hlc_df: pd.DataFrame,
    origin_cols: Dict[str, Tuple[str, str, str]],
    measurement_df: pd.DataFrame,
    report_time: datetime,
    feed_label: str,
    max_spread: float = 3.0,
    lookback_days: int = 20
) -> pd.DataFrame:
    """
    Generate FOGZ cluster table.
    
    Process:
    1. Get data window (current + previous trading day)
    2. Filter to 'new' data rows per origin
    3. Calculate FOGZ outputs
    4. Find PwX2_1_0 matches within spread (with lookback)
    5. Return cluster table
    """
    print(f"\n{'='*80}")
    print(f"GENERATING FOGZ CLUSTER TABLE - {feed_label.upper()} FEED")
    print(f"{'='*80}\n")
    
    # Get data window
    window_df = get_data_window(hlc_df, report_time)
    print(f"Data window: {len(window_df)} rows")
    
    # Filter to new data rows per origin
    new_data_per_origin = filter_new_data_rows(window_df, origin_cols)
    total_new_rows = sum(len(df) for df in new_data_per_origin.values())
    print(f"New data rows: {total_new_rows} across {len(new_data_per_origin)} origins")
    
    # Calculate FOGZ outputs
    fogz_df = calculate_outputs_for_m_numbers(
        new_data_per_origin,
        origin_cols,
        FOGZ_M_NUMBERS,
        report_time,
        feed_label
    )
    print(f"FOGZ outputs: {len(fogz_df)}")
    
    # Get PwX2_1_0 travelers from full dataset with lookback
    start_date = report_time.date() - timedelta(days=lookback_days)
    hlc_lookback = hlc_df[hlc_df['time'].dt.date >= start_date].copy()
    
    # Calculate PwX2_1_0 outputs for lookback window
    pw_travelers = []
    for origin_name, (h_col, l_col, c_col) in origin_cols.items():
        for _, row in hlc_lookback.iterrows():
            h = row[h_col]
            l = row[l_col]
            c = row[c_col]
            arrival_time = row['time']
            
            if pd.isna(h) or pd.isna(l) or pd.isna(c):
                continue
            
            arrival_date = arrival_time.date()
            days_diff = (report_time.date() - arrival_date).days
            day_label = f"[{-days_diff}]" if days_diff > 0 else "[0]"
            
            if days_diff == 0:
                arrival_type = 'Today'
            elif days_diff <= 2:
                arrival_type = 'Recent'
            else:
                arrival_type = 'Older'
            
            for m_num in PwX2_1_0:
                output = calculate_pivot(h, l, c, m_num)
                if output is not None:
                    pw_travelers.append({
                        'Origin': origin_name,
                        'M #': m_num,
                        'Output': output,
                        'Arrival': arrival_time,
                        'Day': day_label,
                        'Type': arrival_type,
                        'Feed': feed_label
                    })
    
    pw_df = pd.DataFrame(pw_travelers)
    print(f"PwX2_1_0 travelers ({lookback_days}d lookback): {len(pw_df)}")
    
    # Find matches
    matches = find_matches_within_spread(fogz_df, pw_df, max_spread)
    print(f"FOGZ matches found: {len(matches)}")
    
    # Build cluster table
    if not matches:
        return pd.DataFrame()
    
    cluster_df = pd.DataFrame(matches)
    
    # Add required columns
    cluster_df['Model'] = 'FOGZ'
    cluster_df['Pattern_Type'] = 'FOGZ Match'
    cluster_df['Group'] = 'N/A'
    cluster_df['Is_Recip'] = 'No'
    
    # Sort by Arrival_Output
    cluster_df = cluster_df.sort_values('Arrival_Output', ascending=False)
    
    # Select columns in order
    columns = ['Arrival_Output', 'Arrival_DateTime', 'Arrival_Bracket', 'Model', 'Type', 
               'Category', 'Origins', 'Feed', 'M_#s', 'Outputs', 'Prox', 
               'Pattern_Type', 'Group', 'Is_Recip']
    
    cluster_df = cluster_df[columns]
    
    print(f"✅ FOGZ cluster table complete: {len(cluster_df)} rows\n")
    
    return cluster_df


# ============================================================================
# LARGE DISCOUNTS CLUSTER TABLE
# ============================================================================

def generate_large_discounts_table(
    hlc_df: pd.DataFrame,
    origin_cols: Dict[str, Tuple[str, str, str]],
    measurement_df: pd.DataFrame,
    report_time: datetime,
    feed_label: str,
    max_spread: float = 3.0,
    lookback_days: int = 20
) -> pd.DataFrame:
    """
    Generate Large Discounts cluster table.
    
    M#{36, 39} matched against PX2_1_0.
    """
    print(f"\n{'='*80}")
    print(f"GENERATING LARGE DISCOUNTS CLUSTER TABLE - {feed_label.upper()} FEED")
    print(f"{'='*80}\n")
    
    # Get data window
    window_df = get_data_window(hlc_df, report_time)
    print(f"Data window: {len(window_df)} rows")
    
    # Filter to new data rows
    new_data_per_origin = filter_new_data_rows(window_df, origin_cols)
    total_new_rows = sum(len(df) for df in new_data_per_origin.values())
    print(f"New data rows: {total_new_rows}")
    
    # Calculate Large Discount outputs
    ld_df = calculate_outputs_for_m_numbers(
        new_data_per_origin,
        origin_cols,
        LARGE_DISCOUNT_M,
        report_time,
        feed_label
    )
    print(f"Large Discount outputs: {len(ld_df)}")
    
    # Get PX2_1_0 travelers with lookback
    start_date = report_time.date() - timedelta(days=lookback_days)
    hlc_lookback = hlc_df[hlc_df['time'].dt.date >= start_date].copy()
    
    px_travelers = []
    for origin_name, (h_col, l_col, c_col) in origin_cols.items():
        for _, row in hlc_lookback.iterrows():
            h, l, c = row[h_col], row[l_col], row[c_col]
            arrival_time = row['time']
            
            if pd.isna(h) or pd.isna(l) or pd.isna(c):
                continue
            
            arrival_date = arrival_time.date()
            days_diff = (report_time.date() - arrival_date).days
            day_label = f"[{-days_diff}]" if days_diff > 0 else "[0]"
            
            if days_diff == 0:
                arrival_type = 'Today'
            elif days_diff <= 2:
                arrival_type = 'Recent'
            else:
                arrival_type = 'Older'
            
            for m_num in PX2_1_0:
                output = calculate_pivot(h, l, c, m_num)
                if output is not None:
                    px_travelers.append({
                        'Origin': origin_name,
                        'M #': m_num,
                        'Output': output,
                        'Arrival': arrival_time,
                        'Day': day_label,
                        'Type': arrival_type,
                        'Feed': feed_label
                    })
    
    px_df = pd.DataFrame(px_travelers)
    print(f"PX2_1_0 travelers: {len(px_df)}")
    
    # Find matches
    matches = find_matches_within_spread(ld_df, px_df, max_spread)
    print(f"Large Discount matches: {len(matches)}")
    
    if not matches:
        return pd.DataFrame()
    
    cluster_df = pd.DataFrame(matches)
    cluster_df['Model'] = 'Large Discounts'
    cluster_df['Pattern_Type'] = 'Discount Match'
    cluster_df['Group'] = 'N/A'
    cluster_df['Is_Recip'] = 'No'
    
    cluster_df = cluster_df.sort_values('Arrival_Output', ascending=False)
    
    columns = ['Arrival_Output', 'Arrival_DateTime', 'Arrival_Bracket', 'Model', 'Type', 
               'Category', 'Origins', 'Feed', 'M_#s', 'Outputs', 'Prox', 
               'Pattern_Type', 'Group', 'Is_Recip']
    
    cluster_df = cluster_df[columns]
    
    print(f"✅ Large Discounts table complete: {len(cluster_df)} rows\n")
    
    return cluster_df


# ============================================================================
# RECIPS PD CLUSTER TABLE
# ============================================================================

def generate_recips_pd_table(
    hlc_df: pd.DataFrame,
    origin_cols: Dict[str, Tuple[str, str, str]],
    measurement_df: pd.DataFrame,
    report_time: datetime,
    feed_label: str,
    max_spread: float = 3.0,
    lookback_days: int = 20
) -> pd.DataFrame:
    """
    Generate Recips PD cluster table.
    
    DRecip list (abs < 40) matched with R# mates only.
    """
    print(f"\n{'='*80}")
    print(f"GENERATING RECIPS PD CLUSTER TABLE - {feed_label.upper()} FEED")
    print(f"{'='*80}\n")
    
    # Get data window
    window_df = get_data_window(hlc_df, report_time)
    print(f"Data window: {len(window_df)} rows")
    
    # Filter to new data rows
    new_data_per_origin = filter_new_data_rows(window_df, origin_cols)
    total_new_rows = sum(len(df) for df in new_data_per_origin.values())
    print(f"New data rows: {total_new_rows}")
    
    # Calculate DRecip outputs
    drecip_df = calculate_outputs_for_m_numbers(
        new_data_per_origin,
        origin_cols,
        DRECIP_LIST,
        report_time,
        feed_label
    )
    print(f"DRecip outputs: {len(drecip_df)}")
    
    # Get R# mate outputs with lookback
    # For each DRecip M#, only look for its specific R# mate
    start_date = report_time.date() - timedelta(days=lookback_days)
    hlc_lookback = hlc_df[hlc_df['time'].dt.date >= start_date].copy()
    
    recip_mate_travelers = []
    for origin_name, (h_col, l_col, c_col) in origin_cols.items():
        for _, row in hlc_lookback.iterrows():
            h, l, c = row[h_col], row[l_col], row[c_col]
            arrival_time = row['time']
            
            if pd.isna(h) or pd.isna(l) or pd.isna(c):
                continue
            
            arrival_date = arrival_time.date()
            days_diff = (report_time.date() - arrival_date).days
            day_label = f"[{-days_diff}]" if days_diff > 0 else "[0]"
            
            if days_diff == 0:
                arrival_type = 'Today'
            elif days_diff <= 2:
                arrival_type = 'Recent'
            else:
                arrival_type = 'Older'
            
            # Calculate outputs for all possible R# mates
            r_mates = set()
            for m_num in DRECIP_LIST:
                if m_num in RECIP_MAP:
                    r_mates.add(RECIP_MAP[m_num])
            
            for m_num in r_mates:
                output = calculate_pivot(h, l, c, m_num)
                if output is not None:
                    recip_mate_travelers.append({
                        'Origin': origin_name,
                        'M #': m_num,
                        'Output': output,
                        'Arrival': arrival_time,
                        'Day': day_label,
                        'Type': arrival_type,
                        'Feed': feed_label
                    })
    
    recip_mate_df = pd.DataFrame(recip_mate_travelers)
    print(f"R# mate travelers: {len(recip_mate_df)}")
    
    # Find matches (only with R# mates)
    recip_matches = []
    
    for _, drecip_row in drecip_df.iterrows():
        drecip_m = drecip_row['M #']
        drecip_output = drecip_row['Output']
        
        # Get R# mate for this M#
        if drecip_m not in RECIP_MAP:
            continue
        
        r_mate = RECIP_MAP[drecip_m]
        
        # Find matches with this specific R# mate
        potential_matches = recip_mate_df[
            (recip_mate_df['M #'] == r_mate) &
            (abs(recip_mate_df['Output'] - drecip_output) <= max_spread)
        ]
        
        for _, match_row in potential_matches.iterrows():
            output_spread = abs(drecip_output - match_row['Output'])
            avg_output = (drecip_output + match_row['Output']) / 2
            
            # Determine combined type
            if drecip_row['Type'] == 'Today' or match_row['Type'] == 'Today':
                combined_type = 'Today'
            elif drecip_row['Type'] == 'Recent' or match_row['Type'] == 'Recent':
                combined_type = 'Recent'
            else:
                combined_type = 'Older'
            
            recip_matches.append({
                'Arrival_Output': avg_output,
                'Arrival_DateTime': drecip_row['Arrival'],
                'Arrival_Bracket': drecip_row['Day'],
                'Type': combined_type,
                'Category': f"M#{drecip_m} ↔ M#{r_mate}",
                'Origins': f"{drecip_row['Origin']}, {match_row['Origin']}",
                'Feed': f"{drecip_row['Feed']}+{match_row['Feed']}" if drecip_row['Feed'] != match_row['Feed'] else drecip_row['Feed'],
                'M_#s': f"{drecip_m}, {r_mate}",
                'Outputs': f"{drecip_output:.2f}, {match_row['Output']:.2f}",
                'Prox': output_spread
            })
    
    print(f"Recip PD matches: {len(recip_matches)}")
    
    if not recip_matches:
        return pd.DataFrame()
    
    cluster_df = pd.DataFrame(recip_matches)
    cluster_df['Model'] = 'Recips PD'
    cluster_df['Pattern_Type'] = 'Reciprocal'
    cluster_df['Group'] = 'N/A'
    cluster_df['Is_Recip'] = 'Yes'
    
    cluster_df = cluster_df.sort_values('Arrival_Output', ascending=False)
    
    columns = ['Arrival_Output', 'Arrival_DateTime', 'Arrival_Bracket', 'Model', 'Type', 
               'Category', 'Origins', 'Feed', 'M_#s', 'Outputs', 'Prox', 
               'Pattern_Type', 'Group', 'Is_Recip']
    
    cluster_df = cluster_df[columns]
    
    print(f"✅ Recips PD table complete: {len(cluster_df)} rows\n")
    
    return cluster_df


# ============================================================================
# MAIN GENERATOR
# ============================================================================

def generate_all_cluster_tables(
    small_hlc_df: pd.DataFrame,
    big_hlc_df: pd.DataFrame,
    measurement_df: pd.DataFrame,
    report_time: datetime,
    max_spread: float = 3.0,
    lookback_days: int = 20
) -> Dict[str, pd.DataFrame]:
    """
    Generate all three cluster tables for both feeds.
    
    Returns dict with:
    - fogz_small, fogz_big, fogz_combined
    - ld_small, ld_big, ld_combined (Large Discounts)
    - recips_small, recips_big, recips_combined
    """
    print(f"\n{'='*80}")
    print(f"GENERATING ALL CLUSTER TABLES")
    print(f"{'='*80}")
    print(f"Report Time: {report_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Max Spread: {max_spread}")
    print(f"Lookback: {lookback_days} days")
    print(f"{'='*80}\n")
    
    # Extract origins
    from recip_traveler_generator_FAST import extract_origins_from_hlc
    
    small_origins = extract_origins_from_hlc(small_hlc_df)
    big_origins = extract_origins_from_hlc(big_hlc_df)
    
    # Generate FOGZ tables
    fogz_small = generate_fogz_table(
        small_hlc_df, small_origins, measurement_df, report_time, 'Small', max_spread, lookback_days
    )
    fogz_big = generate_fogz_table(
        big_hlc_df, big_origins, measurement_df, report_time, 'Big', max_spread, lookback_days
    )
    fogz_combined = pd.concat([fogz_small, fogz_big], ignore_index=True) if len(fogz_small) > 0 or len(fogz_big) > 0 else pd.DataFrame()
    if len(fogz_combined) > 0:
        fogz_combined = fogz_combined.sort_values('Arrival_Output', ascending=False)
    
    # Generate Large Discounts tables
    ld_small = generate_large_discounts_table(
        small_hlc_df, small_origins, measurement_df, report_time, 'Small', max_spread, lookback_days
    )
    ld_big = generate_large_discounts_table(
        big_hlc_df, big_origins, measurement_df, report_time, 'Big', max_spread, lookback_days
    )
    ld_combined = pd.concat([ld_small, ld_big], ignore_index=True) if len(ld_small) > 0 or len(ld_big) > 0 else pd.DataFrame()
    if len(ld_combined) > 0:
        ld_combined = ld_combined.sort_values('Arrival_Output', ascending=False)
    
    # Generate Recips PD tables
    recips_small = generate_recips_pd_table(
        small_hlc_df, small_origins, measurement_df, report_time, 'Small', max_spread, lookback_days
    )
    recips_big = generate_recips_pd_table(
        big_hlc_df, big_origins, measurement_df, report_time, 'Big', max_spread, lookback_days
    )
    recips_combined = pd.concat([recips_small, recips_big], ignore_index=True) if len(recips_small) > 0 or len(recips_big) > 0 else pd.DataFrame()
    if len(recips_combined) > 0:
        recips_combined = recips_combined.sort_values('Arrival_Output', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"CLUSTER TABLES SUMMARY")
    print(f"{'='*80}")
    print(f"FOGZ: Small={len(fogz_small)}, Big={len(fogz_big)}, Combined={len(fogz_combined)}")
    print(f"Large Discounts: Small={len(ld_small)}, Big={len(ld_big)}, Combined={len(ld_combined)}")
    print(f"Recips PD: Small={len(recips_small)}, Big={len(recips_big)}, Combined={len(recips_combined)}")
    print(f"{'='*80}\n")
    
    return {
        'fogz_small': fogz_small,
        'fogz_big': fogz_big,
        'fogz_combined': fogz_combined,
        'ld_small': ld_small,
        'ld_big': ld_big,
        'ld_combined': ld_combined,
        'recips_small': recips_small,
        'recips_big': recips_big,
        'recips_combined': recips_combined
    }
