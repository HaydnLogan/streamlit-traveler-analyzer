"""
Swing Analysis Tool 05a
- 4-file OHLC upload (3m, 5m, 6m, 15m)
- Moving Averages Wick Detection Report
- Updated 15-minute Swing Detection (exits at 16:00)
- Identifies large trending moves, avoids ranging zones
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import io
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Swing Analysis Tool 05a",
    page_icon="📊",
    layout="wide"
)

# Trading day boundaries
TRADING_DAY_END_HOUR = 16
TRADING_DAY_END_MINUTE = 0

# Minimum swing size thresholds
SWING_THRESHOLDS = {
    'small': 100,
    'medium': 150,
    'large': 200,
    'epic': 500
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_ma_column_name(col_name):
    """
    Parse MA column names like '5m 200e', '30m 200s', '1Hr 200h'
    Returns dict with timeframe, period, ma_type or None if not an MA column
    """
    # Skip non-MA columns
    if col_name in ['time', 'open', 'high', 'low', 'close', 'Volume', 'Basis', 'Upper', 'Lower']:
        return None
    if col_name.startswith('h') or col_name.startswith('RSI'):
        return None
    
    # Pattern: timeframe + space + period + type
    # Examples: '5m 200e', '1Hr 500h', 's8.1 100e', 'D1 800s'
    pattern = r'^([a-zA-Z0-9.]+)\s+(\d+)([hse])$'
    match = re.match(pattern, col_name.strip())
    
    if match:
        timeframe = match.group(1)
        period = int(match.group(2))
        ma_type_code = match.group(3)
        
        ma_types = {'h': 'Hull', 's': 'Simple', 'e': 'Exponential'}
        
        return {
            'column': col_name,
            'timeframe': timeframe,
            'period': period,
            'ma_type': ma_types.get(ma_type_code, 'Unknown'),
            'ma_type_code': ma_type_code
        }
    return None


def get_all_ma_columns(df):
    """Extract all MA column information from dataframe"""
    ma_cols = []
    for col in df.columns:
        parsed = parse_ma_column_name(col)
        if parsed:
            ma_cols.append(parsed)
    return ma_cols


def load_ohlc_file(uploaded_file, file_label):
    """Load and validate an OHLC file"""
    if uploaded_file is None:
        return None
    
    try:
        df = pd.read_excel(uploaded_file)
        
        # Check for required columns
        required = ['time', 'open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"{file_label}: Missing required columns: {missing}")
            return None
        
        # Parse time column
        df['time'] = pd.to_datetime(df['time'])
        
        # Remove timezone if present
        if df['time'].dt.tz is not None:
            df['time'] = df['time'].dt.tz_localize(None)
        
        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        st.error(f"{file_label}: Error loading file - {str(e)}")
        return None


def get_trading_day(timestamp, day_start_hour=18):
    """
    Determine the trading day for a given timestamp.
    Trading day starts at day_start_hour (17 or 18) and ends at 16:00/16:45 next calendar day.
    """
    if timestamp.hour >= day_start_hour:
        # After day start - trading day is next calendar day
        return (timestamp + timedelta(days=1)).date()
    elif timestamp.hour < TRADING_DAY_END_HOUR or (timestamp.hour == TRADING_DAY_END_HOUR and timestamp.minute == 0):
        # Before day end - trading day is current calendar day
        return timestamp.date()
    else:
        # Between 16:00 and day_start - belongs to current calendar day
        return timestamp.date()


def is_in_wick(candle_row, ma_value):
    """
    Check if MA value is in the wick area of a candle.
    Wick area = between high/low and the candle body (open/close range)
    """
    if pd.isna(ma_value):
        return False, None
    
    high = candle_row['high']
    low = candle_row['low']
    open_price = candle_row['open']
    close_price = candle_row['close']
    
    body_high = max(open_price, close_price)
    body_low = min(open_price, close_price)
    
    # Upper wick: between high and body_high
    if body_high < ma_value <= high:
        return True, 'upper'
    
    # Lower wick: between low and body_low
    if low <= ma_value < body_low:
        return True, 'lower'
    
    return False, None


# ============================================================================
# MA WICK DETECTION
# ============================================================================

def detect_ma_wick_interactions(df, ma_columns):
    """
    For each candle, detect which MAs are in the wick area.
    Returns a list of interactions with time, MA info, and wick position.
    """
    interactions = []
    
    for idx, row in df.iterrows():
        for ma_info in ma_columns:
            col_name = ma_info['column']
            if col_name not in df.columns:
                continue
            
            ma_value = row[col_name]
            in_wick, wick_type = is_in_wick(row, ma_value)
            
            if in_wick:
                interactions.append({
                    'time': row['time'],
                    'candle_idx': idx,
                    'ma_column': col_name,
                    'ma_timeframe': ma_info['timeframe'],
                    'ma_period': ma_info['period'],
                    'ma_type': ma_info['ma_type'],
                    'ma_value': ma_value,
                    'wick_type': wick_type,
                    'high': row['high'],
                    'low': row['low'],
                    'open': row['open'],
                    'close': row['close']
                })
    
    return pd.DataFrame(interactions)


def calculate_distance_after_interaction(df, interactions_df, lookforward_candles=50):
    """
    For each MA wick interaction, calculate how far price traveled after.
    """
    if interactions_df.empty:
        return interactions_df
    
    results = []
    
    for _, interaction in interactions_df.iterrows():
        idx = interaction['candle_idx']
        wick_type = interaction['wick_type']
        
        # Get future candles
        future_df = df.iloc[idx+1:idx+1+lookforward_candles]
        
        if future_df.empty:
            max_move = 0
            direction = 'none'
        else:
            if wick_type == 'lower':
                # MA in lower wick - expect move up
                max_high = future_df['high'].max()
                max_move = max_high - interaction['low']
                direction = 'up'
            else:
                # MA in upper wick - expect move down
                min_low = future_df['low'].min()
                max_move = interaction['high'] - min_low
                direction = 'down'
        
        result = interaction.to_dict()
        result['distance_after'] = round(max_move, 2)
        result['expected_direction'] = direction
        results.append(result)
    
    return pd.DataFrame(results)


def generate_ma_wick_report(df, ma_columns, min_distance=100):
    """
    Generate comprehensive MA wick interaction report.
    Filters to interactions followed by moves >= min_distance.
    """
    # Detect all interactions
    interactions = detect_ma_wick_interactions(df, ma_columns)
    
    if interactions.empty:
        return pd.DataFrame()
    
    # Calculate distances
    interactions = calculate_distance_after_interaction(df, interactions)
    
    # Filter by minimum distance
    significant = interactions[interactions['distance_after'] >= min_distance].copy()
    
    # Sort by distance descending
    significant = significant.sort_values('distance_after', ascending=False)
    
    return significant


# ============================================================================
# SWING DETECTION (15-MINUTE STYLE)
# ============================================================================

def detect_swings_15m(df, day_start_hour=18, min_swing_size=100, exit_at_16=True):
    """
    Detect swings on 15-minute (or any) timeframe data.
    
    Rules:
    - Identify larger trending moves
    - Avoid logging swings inside ranging zones  
    - Exit analysis at 16:00 if exit_at_16 is True
    - Log swing from entry point to exit point with move size
    - Identify multiple possible entry points for same major move
    """
    if df.empty:
        return pd.DataFrame()
    
    # Add trading day column
    df = df.copy()
    df['trading_day'] = df['time'].apply(lambda x: get_trading_day(x, day_start_hour))
    
    all_swings = []
    
    # Process each trading day
    for trading_day in df['trading_day'].unique():
        day_df = df[df['trading_day'] == trading_day].copy()
        
        if exit_at_16:
            # Filter to only include data up to 16:00
            day_df = day_df[
                (day_df['time'].dt.hour < TRADING_DAY_END_HOUR) | 
                (day_df['time'].dt.date != trading_day)
            ]
        
        if len(day_df) < 3:
            continue
        
        # Detect major swings with multiple entry points
        day_swings = detect_major_swings_with_entries(day_df, trading_day, min_swing_size)
        all_swings.extend(day_swings)
    
    return pd.DataFrame(all_swings)


def detect_major_swings_with_entries(day_df, trading_day, min_swing_size):
    """
    Detect major swings for a trading day and identify multiple entry points.
    
    Logic:
    1. Find HOD and LOD for the day
    2. Identify the major moves (HOD->LOD or LOD->HOD)
    3. Find all valid entry points within acceptable zone (24 units)
    4. Return all entry variations for each major move
    """
    swings = []
    
    if len(day_df) < 5:
        return swings
    
    day_df = day_df.reset_index(drop=True)
    
    # Find HOD and LOD
    hod_idx = day_df['high'].idxmax()
    lod_idx = day_df['low'].idxmin()
    
    hod_row = day_df.loc[hod_idx]
    lod_row = day_df.loc[lod_idx]
    
    hod_time = hod_row['time']
    lod_time = lod_row['time']
    hod_price = hod_row['high']
    lod_price = lod_row['low']
    
    # Identify significant swing points using adaptive window
    swing_points = identify_swing_points(day_df, sensitivity=3)
    
    # Find major moves and their entry points
    for i, sp in enumerate(swing_points):
        if i == len(swing_points) - 1:
            continue
            
        next_sp = swing_points[i + 1]
        
        if sp['type'] == 'low' and next_sp['type'] == 'high':
            # Upward move
            move_size = next_sp['price'] - sp['price']
            if move_size >= min_swing_size:
                # Find all valid entry points (swing lows within 24 units of the base)
                entries = find_entry_points(day_df, sp['time'], next_sp['time'], 
                                          sp['price'], 'up', zone_size=24)
                
                for entry in entries:
                    swings.append({
                        'trading_day': trading_day,
                        'swing_type': 'high',
                        'direction': 'up',
                        'from_datetime': entry['time'],
                        'from_price': entry['price'],
                        'to_datetime': next_sp['time'],
                        'to_price': next_sp['price'],
                        'move_size': round(next_sp['price'] - entry['price'], 2),
                        'category': categorize_move(next_sp['price'] - entry['price'])
                    })
        
        elif sp['type'] == 'high' and next_sp['type'] == 'low':
            # Downward move
            move_size = sp['price'] - next_sp['price']
            if move_size >= min_swing_size:
                # Find all valid entry points (swing highs within 24 units of the top)
                entries = find_entry_points(day_df, sp['time'], next_sp['time'],
                                          sp['price'], 'down', zone_size=24)
                
                for entry in entries:
                    swings.append({
                        'trading_day': trading_day,
                        'swing_type': 'low',
                        'direction': 'down',
                        'from_datetime': entry['time'],
                        'from_price': entry['price'],
                        'to_datetime': next_sp['time'],
                        'to_price': next_sp['price'],
                        'move_size': round(entry['price'] - next_sp['price'], 2),
                        'category': categorize_move(entry['price'] - next_sp['price'])
                    })
    
    # Also add HOD/LOD major moves if not already captured
    total_range = hod_price - lod_price
    if total_range >= min_swing_size:
        if hod_time < lod_time:
            # Down day - add HOD to LOD move
            existing = [s for s in swings if s['to_price'] == lod_price and s['direction'] == 'down']
            if not existing:
                swings.append({
                    'trading_day': trading_day,
                    'swing_type': 'low',
                    'direction': 'down',
                    'from_datetime': hod_time,
                    'from_price': hod_price,
                    'to_datetime': lod_time,
                    'to_price': lod_price,
                    'move_size': round(total_range, 2),
                    'category': categorize_move(total_range)
                })
        else:
            # Up day - add LOD to HOD move
            existing = [s for s in swings if s['to_price'] == hod_price and s['direction'] == 'up']
            if not existing:
                swings.append({
                    'trading_day': trading_day,
                    'swing_type': 'high',
                    'direction': 'up',
                    'from_datetime': lod_time,
                    'from_price': lod_price,
                    'to_datetime': hod_time,
                    'to_price': hod_price,
                    'move_size': round(total_range, 2),
                    'category': categorize_move(total_range)
                })
    
    return swings


def identify_swing_points(df, sensitivity=3):
    """
    Identify significant swing highs and lows.
    A swing high/low is confirmed when price moves away by 'sensitivity' candles.
    """
    swing_points = []
    
    if len(df) < sensitivity * 2 + 1:
        return swing_points
    
    df = df.reset_index(drop=True)
    
    for i in range(sensitivity, len(df) - sensitivity):
        # Check for swing high
        is_swing_high = True
        for j in range(1, sensitivity + 1):
            if df.loc[i, 'high'] <= df.loc[i - j, 'high'] or df.loc[i, 'high'] <= df.loc[i + j, 'high']:
                is_swing_high = False
                break
        
        if is_swing_high:
            swing_points.append({
                'type': 'high',
                'time': df.loc[i, 'time'],
                'price': df.loc[i, 'high'],
                'idx': i
            })
        
        # Check for swing low
        is_swing_low = True
        for j in range(1, sensitivity + 1):
            if df.loc[i, 'low'] >= df.loc[i - j, 'low'] or df.loc[i, 'low'] >= df.loc[i + j, 'low']:
                is_swing_low = False
                break
        
        if is_swing_low:
            swing_points.append({
                'type': 'low',
                'time': df.loc[i, 'time'],
                'price': df.loc[i, 'low'],
                'idx': i
            })
    
    # Sort by time
    swing_points.sort(key=lambda x: x['time'])
    
    return swing_points


def find_entry_points(df, start_time, end_time, base_price, direction, zone_size=24):
    """
    Find all valid entry points for a swing within the acceptable zone.
    For 'up' direction: find swing lows within zone_size of base_price
    For 'down' direction: find swing highs within zone_size of base_price
    """
    entries = []
    
    # Filter to time range
    mask = (df['time'] >= start_time) & (df['time'] <= end_time)
    range_df = df[mask].copy()
    
    if range_df.empty:
        return [{'time': start_time, 'price': base_price}]
    
    if direction == 'up':
        # Find local lows within zone
        for idx, row in range_df.iterrows():
            if abs(row['low'] - base_price) <= zone_size:
                # Check if it's a local low (lower than neighbors)
                idx_pos = range_df.index.get_loc(idx)
                if idx_pos > 0 and idx_pos < len(range_df) - 1:
                    prev_low = range_df.iloc[idx_pos - 1]['low']
                    next_low = range_df.iloc[idx_pos + 1]['low']
                    if row['low'] <= prev_low and row['low'] <= next_low:
                        entries.append({'time': row['time'], 'price': row['low']})
                elif idx_pos == 0:  # First candle
                    entries.append({'time': row['time'], 'price': row['low']})
    else:
        # Find local highs within zone
        for idx, row in range_df.iterrows():
            if abs(row['high'] - base_price) <= zone_size:
                # Check if it's a local high
                idx_pos = range_df.index.get_loc(idx)
                if idx_pos > 0 and idx_pos < len(range_df) - 1:
                    prev_high = range_df.iloc[idx_pos - 1]['high']
                    next_high = range_df.iloc[idx_pos + 1]['high']
                    if row['high'] >= prev_high and row['high'] >= next_high:
                        entries.append({'time': row['time'], 'price': row['high']})
                elif idx_pos == 0:  # First candle
                    entries.append({'time': row['time'], 'price': row['high']})
    
    # Always include the base entry
    if not any(e['time'] == start_time for e in entries):
        entries.insert(0, {'time': start_time, 'price': base_price})
    
    # Remove duplicates and sort by time
    seen = set()
    unique_entries = []
    for e in entries:
        key = (e['time'], e['price'])
        if key not in seen:
            seen.add(key)
            unique_entries.append(e)
    
    unique_entries.sort(key=lambda x: x['time'])
    
    return unique_entries if unique_entries else [{'time': start_time, 'price': base_price}]


def detect_swing_entries(day_df, trading_day, min_swing_size):
    """
    Legacy function - now calls detect_major_swings_with_entries
    """
    return detect_major_swings_with_entries(day_df, trading_day, min_swing_size)


def categorize_move(move_size):
    """Categorize move size"""
    if move_size >= 500:
        return '500+'
    elif move_size >= 200:
        return '200+'
    elif move_size >= 150:
        return '150-200'
    elif move_size >= 100:
        return '100-150'
    else:
        return '<100'


def detect_entry_zones(df, swings_df, zone_size=24):
    """
    For each swing, identify the acceptable entry zone (within zone_size units).
    Returns swings with entry zone information.
    """
    if swings_df.empty:
        return swings_df
    
    enhanced_swings = []
    
    for _, swing in swings_df.iterrows():
        swing_dict = swing.to_dict()
        
        from_price = swing['from_price']
        direction = swing['direction']
        
        if direction == 'up':
            # Entry zone is from_price to from_price + zone_size
            swing_dict['entry_zone_low'] = from_price
            swing_dict['entry_zone_high'] = from_price + zone_size
        else:
            # Entry zone is from_price - zone_size to from_price
            swing_dict['entry_zone_low'] = from_price - zone_size
            swing_dict['entry_zone_high'] = from_price
        
        enhanced_swings.append(swing_dict)
    
    return pd.DataFrame(enhanced_swings)


# ============================================================================
# MA CONFLUENCE ANALYSIS
# ============================================================================

def find_ma_confluence_at_swings(df, swings_df, ma_columns, zone_size=24):
    """
    For each swing point, find MAs that were in the wick area near the entry.
    This helps identify confluence between MAs and significant turns.
    """
    if swings_df.empty or df.empty:
        return pd.DataFrame()
    
    confluence_records = []
    
    for _, swing in swings_df.iterrows():
        from_time = swing['from_datetime']
        from_price = swing['from_price']
        direction = swing['direction']
        
        # Get candles around the swing entry (±3 candles)
        time_mask = (df['time'] >= from_time - timedelta(hours=1)) & \
                   (df['time'] <= from_time + timedelta(minutes=15))
        nearby_candles = df[time_mask]
        
        for _, candle in nearby_candles.iterrows():
            for ma_info in ma_columns:
                col_name = ma_info['column']
                if col_name not in df.columns:
                    continue
                
                ma_value = candle[col_name]
                if pd.isna(ma_value):
                    continue
                
                # Check if MA is within zone_size of the swing price
                distance_from_swing = abs(ma_value - from_price)
                
                if distance_from_swing <= zone_size:
                    in_wick, wick_type = is_in_wick(candle, ma_value)
                    
                    confluence_records.append({
                        'swing_time': from_time,
                        'swing_price': from_price,
                        'swing_direction': direction,
                        'swing_move_size': swing['move_size'],
                        'candle_time': candle['time'],
                        'ma_column': col_name,
                        'ma_timeframe': ma_info['timeframe'],
                        'ma_period': ma_info['period'],
                        'ma_type': ma_info['ma_type'],
                        'ma_value': ma_value,
                        'distance_from_swing': round(distance_from_swing, 2),
                        'in_wick': in_wick,
                        'wick_type': wick_type if in_wick else 'N/A'
                    })
    
    return pd.DataFrame(confluence_records)


def track_ma_history(df, ma_column, min_move_after=100):
    """
    Track the history of a specific MA - every time it appears in a wick,
    and what happened after.
    """
    if ma_column not in df.columns:
        return pd.DataFrame()
    
    history = []
    
    for idx, row in df.iterrows():
        ma_value = row[ma_column]
        in_wick, wick_type = is_in_wick(row, ma_value)
        
        if in_wick:
            # Calculate move after
            future_df = df.iloc[idx+1:idx+51]  # Next 50 candles
            
            if not future_df.empty:
                if wick_type == 'lower':
                    max_move = future_df['high'].max() - row['low']
                    direction = 'up'
                else:
                    max_move = row['high'] - future_df['low'].min()
                    direction = 'down'
            else:
                max_move = 0
                direction = 'unknown'
            
            history.append({
                'time': row['time'],
                'ma_value': ma_value,
                'wick_type': wick_type,
                'high': row['high'],
                'low': row['low'],
                'move_after': round(max_move, 2),
                'direction': direction,
                'significant': max_move >= min_move_after
            })
    
    return pd.DataFrame(history)


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.title("📊 Market Swing Analysis Tool 05a")
    st.markdown("---")
    
    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    
    day_start_hour = st.sidebar.radio(
        "Trading Day Start",
        [18, 17],
        format_func=lambda x: f"{x}:00"
    )
    
    min_swing_size = st.sidebar.slider(
        "Minimum Swing Size (units)",
        min_value=50,
        max_value=500,
        value=100,
        step=25
    )
    
    min_ma_distance = st.sidebar.slider(
        "Min MA Move Distance (for report)",
        min_value=50,
        max_value=500,
        value=100,
        step=25
    )
    
    entry_zone_size = st.sidebar.slider(
        "Entry Zone Size (units)",
        min_value=12,
        max_value=48,
        value=24,
        step=6
    )
    
    exit_at_16 = st.sidebar.checkbox("Exit analysis at 16:00", value=True)
    
    # Main content - File uploads
    st.header("📁 Upload OHLC Files")
    st.markdown("Upload your OHLC data files for each timeframe.")
    
    # 4 file uploaders in one row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        file_3m = st.file_uploader("3m", type=['xlsx', 'xls'], key="file_3m")
    
    with col2:
        file_5m = st.file_uploader("5m", type=['xlsx', 'xls'], key="file_5m")
    
    with col3:
        file_6m = st.file_uploader("6m", type=['xlsx', 'xls'], key="file_6m")
    
    with col4:
        file_15m = st.file_uploader("15m", type=['xlsx', 'xls'], key="file_15m")
    
    # Load files
    files = {
        '3m': load_ohlc_file(file_3m, '3m'),
        '5m': load_ohlc_file(file_5m, '5m'),
        '6m': load_ohlc_file(file_6m, '6m'),
        '15m': load_ohlc_file(file_15m, '15m')
    }
    
    # Check if any files loaded
    loaded_files = {k: v for k, v in files.items() if v is not None}
    
    if not loaded_files:
        st.info("👆 Please upload at least one OHLC file to begin analysis.")
        return
    
    # Display loaded files info
    st.success(f"✅ Loaded {len(loaded_files)} file(s): {', '.join(loaded_files.keys())}")
    
    # Show file details
    with st.expander("📋 File Details"):
        for tf, df in loaded_files.items():
            ma_cols = get_all_ma_columns(df)
            st.markdown(f"**{tf}**: {len(df)} rows, {len(df.columns)} columns, {len(ma_cols)} MA columns")
            st.markdown(f"  Time range: {df['time'].min()} to {df['time'].max()}")
    
    st.markdown("---")
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔄 Swing Detection",
        "📈 MA Wick Report",
        "🎯 MA Confluence",
        "📜 MA History Tracker"
    ])
    
    # ========================
    # TAB 1: SWING DETECTION
    # ========================
    with tab1:
        st.header("15-Minute Swing Detection")
        
        # Use 15m file if available, otherwise use the first available
        swing_tf = '15m' if '15m' in loaded_files else list(loaded_files.keys())[0]
        swing_df = loaded_files[swing_tf]
        
        st.info(f"Using {swing_tf} data for swing detection")
        
        if st.button("🔍 Detect Swings", key="detect_swings"):
            with st.spinner("Analyzing swings..."):
                swings = detect_swings_15m(
                    swing_df,
                    day_start_hour=day_start_hour,
                    min_swing_size=min_swing_size,
                    exit_at_16=exit_at_16
                )
                
                if not swings.empty:
                    # Add entry zones
                    swings = detect_entry_zones(swing_df, swings, entry_zone_size)
                    
                    st.success(f"Found {len(swings)} significant swings")
                    
                    # Display swings grouped by trading day
                    for trading_day in swings['trading_day'].unique():
                        day_swings = swings[swings['trading_day'] == trading_day]
                        st.markdown(f"### Trading Day: {trading_day}")
                        
                        display_cols = ['swing_type', 'direction', 'from_datetime', 'from_price',
                                       'to_datetime', 'to_price', 'move_size', 'category',
                                       'entry_zone_low', 'entry_zone_high']
                        st.dataframe(day_swings[display_cols], use_container_width=True)
                    
                    # Store in session state for export
                    st.session_state['swings_df'] = swings
                    
                    # Export button
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        for trading_day in swings['trading_day'].unique():
                            day_swings = swings[swings['trading_day'] == trading_day]
                            sheet_name = str(trading_day).replace('-', '.')
                            day_swings.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                    
                    st.download_button(
                        "📥 Download Swing Report",
                        data=excel_buffer.getvalue(),
                        file_name=f"swing_analysis_05a_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("No significant swings found with current settings.")
    
    # ========================
    # TAB 2: MA WICK REPORT
    # ========================
    with tab2:
        st.header("Moving Averages Wick Interaction Report")
        
        # Select timeframe
        ma_timeframe = st.selectbox(
            "Select timeframe for MA analysis",
            list(loaded_files.keys()),
            key="ma_tf_select"
        )
        
        if ma_timeframe and st.button("📊 Generate MA Wick Report", key="gen_ma_report"):
            analysis_df = loaded_files[ma_timeframe]
            ma_columns = get_all_ma_columns(analysis_df)
            
            if not ma_columns:
                st.warning("No MA columns found in this file.")
            else:
                with st.spinner(f"Analyzing {len(ma_columns)} MA columns..."):
                    report = generate_ma_wick_report(
                        analysis_df,
                        ma_columns,
                        min_distance=min_ma_distance
                    )
                    
                    if not report.empty:
                        st.success(f"Found {len(report)} significant MA wick interactions")
                        
                        # Summary by MA
                        st.markdown("### Summary by Moving Average")
                        ma_summary = report.groupby('ma_column').agg({
                            'distance_after': ['count', 'mean', 'max'],
                            'wick_type': lambda x: (x == 'lower').sum()
                        }).round(2)
                        ma_summary.columns = ['Count', 'Avg Move', 'Max Move', 'Lower Wick Count']
                        ma_summary = ma_summary.sort_values('Max Move', ascending=False)
                        st.dataframe(ma_summary, use_container_width=True)
                        
                        # Full report
                        st.markdown("### Detailed Interactions")
                        display_cols = ['time', 'ma_column', 'ma_value', 'wick_type',
                                       'distance_after', 'expected_direction', 'high', 'low']
                        st.dataframe(report[display_cols].head(100), use_container_width=True)
                        
                        # Export
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            report.to_excel(writer, sheet_name='MA_Wick_Report', index=False)
                            ma_summary.to_excel(writer, sheet_name='MA_Summary')
                        
                        st.download_button(
                            "📥 Download MA Wick Report",
                            data=excel_buffer.getvalue(),
                            file_name=f"ma_wick_report_{ma_timeframe}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.warning("No significant MA wick interactions found.")
    
    # ========================
    # TAB 3: MA CONFLUENCE
    # ========================
    with tab3:
        st.header("MA Confluence at Swing Points")
        st.markdown("Find which MAs were present near significant swing entry points.")
        
        # Need swings first
        if 'swings_df' not in st.session_state or st.session_state['swings_df'].empty:
            st.info("👆 Please run Swing Detection first (Tab 1)")
        else:
            conf_timeframe = st.selectbox(
                "Select timeframe for confluence analysis",
                list(loaded_files.keys()),
                key="conf_tf_select"
            )
            
            if st.button("🎯 Find MA Confluence", key="find_confluence"):
                analysis_df = loaded_files[conf_timeframe]
                ma_columns = get_all_ma_columns(analysis_df)
                swings = st.session_state['swings_df']
                
                with st.spinner("Finding confluence..."):
                    confluence = find_ma_confluence_at_swings(
                        analysis_df,
                        swings,
                        ma_columns,
                        zone_size=entry_zone_size
                    )
                    
                    if not confluence.empty:
                        st.success(f"Found {len(confluence)} MA confluence points")
                        
                        # Summary
                        st.markdown("### MAs Most Frequently at Swing Points")
                        ma_freq = confluence.groupby('ma_column').agg({
                            'swing_time': 'count',
                            'swing_move_size': 'mean',
                            'in_wick': 'sum'
                        }).round(2)
                        ma_freq.columns = ['Occurrences', 'Avg Swing Size', 'In Wick Count']
                        ma_freq = ma_freq.sort_values('Occurrences', ascending=False)
                        st.dataframe(ma_freq.head(20), use_container_width=True)
                        
                        # Detail
                        st.markdown("### Detailed Confluence")
                        st.dataframe(confluence.head(100), use_container_width=True)
                        
                        # Export
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            confluence.to_excel(writer, sheet_name='Confluence', index=False)
                            ma_freq.to_excel(writer, sheet_name='MA_Frequency')
                        
                        st.download_button(
                            "📥 Download Confluence Report",
                            data=excel_buffer.getvalue(),
                            file_name=f"ma_confluence_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.warning("No MA confluence found at swing points.")
    
    # ========================
    # TAB 4: MA HISTORY TRACKER
    # ========================
    with tab4:
        st.header("MA History Tracker")
        st.markdown("Track the history of a specific MA - every wick appearance and subsequent move.")
        
        # Select timeframe and MA
        track_tf = st.selectbox(
            "Select timeframe",
            list(loaded_files.keys()),
            key="track_tf_select"
        )
        
        if track_tf:
            track_df = loaded_files[track_tf]
            ma_columns = get_all_ma_columns(track_df)
            ma_names = [ma['column'] for ma in ma_columns]
            
            # Filter options
            st.markdown("#### Filter MAs")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                period_filter = st.multiselect(
                    "Periods",
                    [25, 50, 100, 200, 500, 800],
                    default=[500, 800]
                )
            
            with col2:
                type_filter = st.multiselect(
                    "MA Types",
                    ['Hull', 'Simple', 'Exponential'],
                    default=['Exponential']
                )
            
            # Filter MA names
            filtered_mas = [
                ma['column'] for ma in ma_columns
                if ma['period'] in period_filter and ma['ma_type'] in type_filter
            ]
            
            selected_ma = st.selectbox(
                "Select MA to track",
                filtered_mas if filtered_mas else ma_names,
                key="selected_ma"
            )
            
            if selected_ma and st.button("📜 Track MA History", key="track_ma"):
                with st.spinner(f"Tracking {selected_ma}..."):
                    history = track_ma_history(track_df, selected_ma, min_move_after=min_ma_distance)
                    
                    if not history.empty:
                        st.success(f"Found {len(history)} wick appearances for {selected_ma}")
                        
                        # Stats
                        significant_count = history['significant'].sum()
                        st.markdown(f"**Significant moves (≥{min_ma_distance} units):** {significant_count} / {len(history)} ({100*significant_count/len(history):.1f}%)")
                        
                        # Display
                        st.dataframe(history, use_container_width=True)
                        
                        # Export
                        excel_buffer = io.BytesIO()
                        history.to_excel(excel_buffer, index=False)
                        
                        st.download_button(
                            "📥 Download MA History",
                            data=excel_buffer.getvalue(),
                            file_name=f"ma_history_{selected_ma.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.warning(f"No wick appearances found for {selected_ma}")


if __name__ == "__main__":
    main()
