"""
v05e_7. CRITICAL FIX: All 5 issues identified by user now resolved.
- Window Radius slider moved to common area (applies to all modes) and max increased to 1000
- Each feed's window now displayed: "🧮 Small Feed Window: [24299.75, 24599.75] around Open = 24449.75"
- Input @ start values now correctly tracked and will display in prep tables
- Raw M calculations now use window range instead of HLC L/H
- Range column in prep tables now shows window range
- Zone column removed from prep tables
- Window_radius parameter now passed to all cluster table calls

v05e_6. CRITICAL FIX for find_valid_m_values calling pattern + comprehensive HLC tracking.
- Processing summaries now ALWAYS display (even when 0 results found)
- Shows detailed HLC examination: which HLCs were processed, dates examined, Raw M ranges
- Expanded by default to help diagnose issues
- Displays both Pass 1 and Pass 2 HLC examination details in separate tabs
- This will help identify why no results are found (e.g., wrong dates, Raw M ranges too narrow)

v05e_5. CRITICAL FIX for M # filtering in two-pass processing.
- Measurement file now pre-filtered by M # before searching (was causing 0 results)
- Added diagnostic output showing M # availability and overlap with valid lists
- This fixes the root cause: valid lists specify M #s, but find_valid_m_values searches by M value
- Pre-filtering ensures only relevant M values (corresponding to desired M #s) are searched

v05e_4. Adds three enhancements to two-pass cluster processing:
1. Processing Summary - Shows Pass 1 vs Pass 2 counts, M numbers, origins, and feeds
2. Export Functionality - Download buttons for All, Pass 1 only, and Pass 2 only data
3. Visual Comparison Mode - Side-by-side view of Pass 1 (recent 2 days) vs Pass 2 (all scope)
   
Also fixes M # type mismatch issue causing 0 results (now converts to int for comparison).
"""
- Identifies large trending moves, avoids ranging zones
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta, time
from typing import List, Dict, Tuple, Optional
import io
import time as time_module  # For timing cluster table generation
import re
import sys

# Add current directory to path for imports
sys.path.insert(0, '/home/claude')
sys.path.insert(0, '/mnt/user-data/outputs')

# Import strategic zone detector
from strategic_zone_detector import (
    detect_huge_hmas, get_high_rank_mas, get_ma_rank,
    find_recip_pairs, find_wildcards_near_zone,
    track_ma_role_at_price, detect_ma_role_flip,
    generate_zone_recommendations, format_recommendation_report,
    get_next_origin_updates
)

# Import reciprocal traveler generator (FAST VERSION)
from recip_traveler_generator_FAST import generate_recip_traveler_reports

# Import custom range calculator with two-pass cluster processing
from custom_range_calculator_1125_7 import (
    process_cluster_tables_two_pass,
    match_cluster_table_entries,
    clean_timestamp
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Swing Analysis Tool 05e_7",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# CONSTANTS
# ============================================================================

# Trading day boundaries
TRADING_DAY_END_HOUR = 16
TRADING_DAY_END_MINUTE = 0

# NY Session boundaries
NY_SESSION_START = dt.time(8, 0)
NY_SESSION_END = dt.time(16, 0)

# Swing thresholds
SWING_THRESHOLDS = {
    'small': 100,
    'medium': 150,
    'large': 200,
    'epic': 500
}

# Origin Classifications (from a_helpers.py)
EPIC_ORIGINS = {"trinidad", "tobago", "wasp-12b", "macedonia"}
ANCHOR_ORIGINS = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}

# Traveler Family Classifications
STRENGTH_TRAVELERS = {0, 40, -40, 54, -54}
FAMILY_ALPHA_TRAVELERS = {2, -2, 10, -10, 22, -22, 30, -30, 36, -36, 39, -39, 41, -41, 43, -43, 50, -50, 60, -60, 77, -77, 107, -107}
FAMILY_BRAVO_TRAVELERS = {5, -5, 14, -14, 55, -55, 68, -68, 96, -96}
GROUP_1A_TRAVELERS = {111, 107, 103, 96, 87, 77, 68, 60, 50, -50, -60, -68, -77, -87, -96, -103, -107, -111}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_timestamp_naive(timestamp_str):
    """Parse timestamp and return naive datetime (remove timezone info)"""
    try:
        if isinstance(timestamp_str, str):
            # Handle ISO format with timezone
            if 'T' in timestamp_str and ('-' in timestamp_str[-6:] or '+' in timestamp_str[-6:]):
                for tz_sep in ['+', '-']:
                    if tz_sep in timestamp_str[-6:]:
                        timestamp_str = timestamp_str[:timestamp_str.rfind(tz_sep)]
                        break
            return pd.to_datetime(timestamp_str, format='mixed')
        else:
            return pd.to_datetime(timestamp_str)
    except:
        return pd.to_datetime(timestamp_str, errors='coerce')


def clean_timestamp(x):
    """Clean timestamp - tolerant of strings/naive/aware datetimes."""
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.replace(tzinfo=None) if getattr(x, "tzinfo", None) is not None else x
    if isinstance(x, str):
        x = x.replace("T", " ")
        x = pd.Series([x]).str.replace(r"[+-]\d{2}:?\d{2}$", "", regex=True).iloc[0]
        try:
            return pd.to_datetime(x, errors="coerce")
        except Exception:
            return pd.NaT
    return pd.NaT


def parse_ma_column_name(col_name):
    """
    Parse MA column names like '5m 200e', '30m 200s', '1Hr 200h'
    Returns dict with timeframe, period, ma_type or None if not an MA column
    """
    if col_name in ['time', 'open', 'high', 'low', 'close', 'Volume', 'Basis', 'Upper', 'Lower']:
        return None
    if col_name.startswith('h') or col_name.startswith('RSI'):
        return None
    
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
    """Load and validate an OHLC file (CSV or Excel)"""
    if uploaded_file is None:
        return None
    
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        else:  # Excel file
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            # Use first sheet or let user select if multiple
            if len(sheet_names) > 1:
                # For simplicity, use first sheet - can be enhanced with sheet selection
                sheet_name = sheet_names[0]
            else:
                sheet_name = sheet_names[0]
            
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        
        # Check for required columns
        required = ['time', 'open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"{file_label}: Missing required columns: {missing}")
            return None
        
        # Parse time column
        df['time'] = df['time'].apply(parse_timestamp_naive)
        
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
    """Get trading day based on start hour (default 18:00)"""
    if timestamp.hour < day_start_hour:
        return timestamp.date()
    else:
        return (timestamp + timedelta(days=1)).date()


def categorize_range(range_val):
    """Categorize daily range"""
    if range_val < 100:
        return "< 100"
    elif range_val < 150:
        return "100-150"
    elif range_val < 200:
        return "150-200"
    elif range_val < 250:
        return "200-250"
    elif range_val < 350:
        return "250-350"
    elif range_val < 500:
        return "350-500"
    elif range_val < 1000:
        return "500-1000"
    else:
        return "1000+"


def categorize_swing(move_size):
    """Categorize swing moves"""
    if move_size < 60:
        return "30-60"
    elif move_size < 100:
        return "60-100"
    elif move_size < 150:
        return "100-150"
    elif move_size < 200:
        return "150-200"
    else:
        return "200+"


def is_in_wick(candle_row, ma_value):
    """Check if MA value is in the wick area of a candle."""
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


def get_origin_type(origin_name):
    """Classify origin type"""
    if not origin_name:
        return 'Other'
    
    normalized = str(origin_name).lower().strip()
    if '[' in normalized and ']' in normalized:
        normalized = normalized[:normalized.find('[')]
    
    if normalized in EPIC_ORIGINS:
        return 'EPC'
    if normalized in ANCHOR_ORIGINS:
        return 'Anchor'
    return 'Other'


# ============================================================================
# SWING DETECTION (from 03h with enhancements)
# ============================================================================

def detect_swings(df, swing_threshold=50, drawdown_limit=25):
    """
    Detect swings by tracking actual price progression bar by bar.
    FIXED: Eliminates duplicate from/to times and ensures chronological order.
    """
    swings = []
    
    if len(df) == 0:
        return swings
    
    current_extreme_high = df.iloc[0]['high']
    current_extreme_low = df.iloc[0]['low']
    extreme_high_time = df.iloc[0]['time']
    extreme_low_time = df.iloc[0]['time']
    
    measuring_from = 'low'
    swing_start_price = current_extreme_low
    swing_start_time = extreme_low_time
    
    for idx, row in df.iterrows():
        bar_high = row['high']
        bar_low = row['low']
        bar_time = row['time']
        
        if measuring_from == 'low':
            if bar_low < current_extreme_low:
                current_extreme_low = bar_low
                extreme_low_time = bar_time
                swing_start_price = current_extreme_low
                swing_start_time = extreme_low_time
            
            if bar_high > current_extreme_high:
                current_extreme_high = bar_high
                extreme_high_time = bar_time
            
            upward_move = current_extreme_high - current_extreme_low
            if upward_move >= swing_threshold:
                drawdown = current_extreme_high - bar_low
                if drawdown >= drawdown_limit:
                    if (swing_start_time < extreme_high_time and 
                        swing_start_price != current_extreme_high):
                        swings.append({
                            'type': 'high',
                            'swing_price': current_extreme_high,
                            'swing_time': extreme_high_time,
                            'move_size': upward_move,
                            'category': categorize_swing(upward_move),
                            'from_price': swing_start_price,
                            'from_time': swing_start_time,
                            'to_price': current_extreme_high,
                            'to_time': extreme_high_time,
                            'direction': 'up'
                        })
                    
                    measuring_from = 'high'
                    current_extreme_low = bar_low
                    extreme_low_time = bar_time
                    swing_start_price = current_extreme_high
                    swing_start_time = extreme_high_time
        
        else:  # measuring_from == 'high'
            if bar_high > current_extreme_high:
                current_extreme_high = bar_high
                extreme_high_time = bar_time
                swing_start_price = current_extreme_high
                swing_start_time = extreme_high_time
            
            if bar_low < current_extreme_low:
                current_extreme_low = bar_low
                extreme_low_time = bar_time
            
            downward_move = current_extreme_high - current_extreme_low
            if downward_move >= swing_threshold:
                bounce = bar_high - current_extreme_low
                if bounce >= drawdown_limit:
                    if (swing_start_time < extreme_low_time and 
                        swing_start_price != current_extreme_low):
                        swings.append({
                            'type': 'low',
                            'swing_price': current_extreme_low,
                            'swing_time': extreme_low_time,
                            'move_size': downward_move,
                            'category': categorize_swing(downward_move),
                            'from_price': swing_start_price,
                            'from_time': swing_start_time,
                            'to_price': current_extreme_low,
                            'to_time': extreme_low_time,
                            'direction': 'down'
                        })
                    
                    measuring_from = 'low'
                    current_extreme_high = bar_high
                    extreme_high_time = bar_time
                    swing_start_price = current_extreme_low
                    swing_start_time = extreme_low_time
    
    return swings


def analyze_daily_data(df, day_start_hour=18, swing_threshold=50, drawdown_limit=25, exit_at_16=True):
    """Analyze daily market structure based on trading day definition"""
    daily_stats = []
    
    df = df.copy()
    df['trading_day'] = df['time'].apply(lambda x: get_trading_day(x, day_start_hour))
    
    for trading_day, day_data in df.groupby('trading_day'):
        day_data = day_data.sort_values('time').copy()
        
        # Apply 16:00 exit if requested
        if exit_at_16:
            day_data = day_data[
                (day_data['time'].dt.hour < TRADING_DAY_END_HOUR) | 
                (day_data['time'].dt.date != trading_day)
            ]
        
        if len(day_data) < 3:
            continue
        
        # Basic daily stats
        daily_high = day_data['high'].max()
        daily_low = day_data['low'].min()
        daily_range = daily_high - daily_low
        
        high_time = day_data[day_data['high'] == daily_high]['time'].iloc[0]
        low_time = day_data[day_data['low'] == daily_low]['time'].iloc[0]
        
        session_start = day_data['time'].min()
        session_end = day_data['time'].max()
        
        # Detect swings
        swings = detect_swings(day_data.reset_index(drop=True), swing_threshold, drawdown_limit)
        
        # NY Session swings (8 AM - 12 PM start time)
        ny_swings = []
        for swing in swings:
            swing_start = swing['from_time']
            if dt.time(8, 0) <= swing_start.time() <= dt.time(12, 0):
                ny_swings.append(swing)
        
        ny_swings = sorted(ny_swings, key=lambda x: x['from_time'])[:3]
        
        # Top swings
        swing_moves = [s['move_size'] for s in swings]
        swing_moves.sort(reverse=True)
        top_3_swings = swing_moves[:3]
        
        # Category counts
        swing_categories = [s['category'] for s in swings]
        category_counts = {
            '30-60': swing_categories.count('30-60'),
            '60-100': swing_categories.count('60-100'),
            '100-150': swing_categories.count('100-150'),
            '150-200': swing_categories.count('150-200'),
            '200+': swing_categories.count('200+')
        }
        
        daily_stats.append({
            'trading_day': trading_day,
            'session_start': session_start,
            'session_end': session_end,
            'daily_high': daily_high,
            'daily_high_time': high_time,
            'daily_low': daily_low,
            'daily_low_time': low_time,
            'daily_range': daily_range,
            'range_category': categorize_range(daily_range),
            'ny_swings_count': len(ny_swings),
            'ny_1': ny_swings[0]['move_size'] if len(ny_swings) > 0 else 'none',
            'ny_2': ny_swings[1]['move_size'] if len(ny_swings) > 1 else 'none',
            'ny_3': ny_swings[2]['move_size'] if len(ny_swings) > 2 else 'none',
            'total_swings': len(swings),
            'top_1_swing': top_3_swings[0] if len(top_3_swings) > 0 else 0,
            'top_2_swing': top_3_swings[1] if len(top_3_swings) > 1 else 0,
            'top_3_swing': top_3_swings[2] if len(top_3_swings) > 2 else 0,
            'swings_30_60': category_counts['30-60'],
            'swings_60_100': category_counts['60-100'],
            'swings_100_150': category_counts['100-150'],
            'swings_150_200': category_counts['150-200'],
            'swings_200_plus': category_counts['200+'],
            'all_swings': swings,
            'ny_swings': ny_swings
        })
    
    return daily_stats


# ============================================================================
# NY SESSION ANALYSIS
# ============================================================================

def analyze_ny_session(df, day_start_hour=18, swing_threshold=50, drawdown_limit=25):
    """
    Analyze NY session specifically (8 AM - 4 PM).
    Returns swings that occur within this window.
    """
    ny_results = []
    
    df = df.copy()
    df['trading_day'] = df['time'].apply(lambda x: get_trading_day(x, day_start_hour))
    
    for trading_day, day_data in df.groupby('trading_day'):
        day_data = day_data.sort_values('time').copy()
        
        # Filter to NY session (8 AM - 4 PM)
        ny_data = day_data[
            (day_data['time'].dt.time >= NY_SESSION_START) &
            (day_data['time'].dt.time <= NY_SESSION_END)
        ]
        
        if len(ny_data) < 3:
            continue
        
        # NY session stats
        ny_high = ny_data['high'].max()
        ny_low = ny_data['low'].min()
        ny_range = ny_high - ny_low
        
        ny_high_time = ny_data[ny_data['high'] == ny_high]['time'].iloc[0]
        ny_low_time = ny_data[ny_data['low'] == ny_low]['time'].iloc[0]
        
        # Detect swings within NY session
        ny_swings = detect_swings(ny_data.reset_index(drop=True), swing_threshold, drawdown_limit)
        
        ny_results.append({
            'trading_day': trading_day,
            'ny_high': ny_high,
            'ny_high_time': ny_high_time,
            'ny_low': ny_low,
            'ny_low_time': ny_low_time,
            'ny_range': ny_range,
            'ny_range_category': categorize_range(ny_range),
            'ny_swing_count': len(ny_swings),
            'ny_swings': ny_swings
        })
    
    return ny_results


# ============================================================================
# MA WICK DETECTION
# ============================================================================

def detect_ma_wick_interactions(df, ma_columns):
    """Detect MA wick interactions for all candles."""
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
    """Calculate distance traveled after each MA wick interaction."""
    if interactions_df.empty:
        return interactions_df
    
    results = []
    
    for _, interaction in interactions_df.iterrows():
        idx = interaction['candle_idx']
        wick_type = interaction['wick_type']
        
        future_df = df.iloc[idx+1:idx+1+lookforward_candles]
        
        if future_df.empty:
            max_move = 0
            direction = 'none'
        else:
            if wick_type == 'lower':
                max_high = future_df['high'].max()
                max_move = max_high - interaction['low']
                direction = 'up'
            else:
                min_low = future_df['low'].min()
                max_move = interaction['high'] - min_low
                direction = 'down'
        
        result = interaction.to_dict()
        result['distance_after'] = round(max_move, 2)
        result['expected_direction'] = direction
        results.append(result)
    
    return pd.DataFrame(results)


def generate_ma_wick_report(df, ma_columns, min_distance=100):
    """Generate comprehensive MA wick interaction report."""
    interactions = detect_ma_wick_interactions(df, ma_columns)
    
    if interactions.empty:
        return pd.DataFrame()
    
    interactions = calculate_distance_after_interaction(df, interactions)
    significant = interactions[interactions['distance_after'] >= min_distance].copy()
    significant = significant.sort_values('distance_after', ascending=False)
    
    return significant


def track_ma_history(df, ma_column, min_move_after=100):
    """Track the history of a specific MA."""
    if ma_column not in df.columns:
        return pd.DataFrame()
    
    history = []
    
    for idx, row in df.iterrows():
        ma_value = row[ma_column]
        in_wick, wick_type = is_in_wick(row, ma_value)
        
        if in_wick:
            future_df = df.iloc[idx+1:idx+51]
            
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
# TRAVELER / PIVOT CALCULATIONS
# ============================================================================

def calculate_pivot_output(H, L, C, m_value):
    """Calculate traveler output using pivot formula: avg + m_value * spread"""
    try:
        H, L, C, m_value = float(H), float(L), float(C), float(m_value)
        avg = (H + L + C) / 3.0
        spread = H - L
        return avg + m_value * spread
    except:
        return None


def calculate_raw_m_values(hlc_data, range_low, range_high):
    """Calculate raw M value range for a given price range."""
    try:
        H, L, C = hlc_data['H'], hlc_data['L'], hlc_data['C']
        avg = (H + L + C) / 3.0
        spread = H - L
        if spread == 0:
            return None
        return {
            'raw_m_low': (range_low - avg) / spread,
            'raw_m_high': (range_high - avg) / spread,
            'avg': avg,
            'spread': spread
        }
    except:
        return None


def find_travelers_at_price(measurement_df, target_price, tolerance=24, hlc_sets=None):
    """
    Find travelers (M values) that produce outputs near a target price.
    Returns list of travelers with their details.
    """
    if hlc_sets is None or not hlc_sets:
        return []
    
    valid_travelers = []
    
    m_value_col = next((c for c in ['M value', 'M Value', 'M_Value', 'm value', 'm_value', 'M #', 'm #'] 
                       if c in measurement_df.columns), None)
    if m_value_col is None:
        return []
    
    for hlc in hlc_sets:
        H, L, C = hlc.get('H'), hlc.get('L'), hlc.get('C')
        if H is None or L is None or C is None:
            continue
        
        avg = (H + L + C) / 3.0
        spread = H - L
        if spread == 0:
            continue
        
        # Calculate M range that would produce outputs near target
        m_low = (target_price - tolerance - avg) / spread
        m_high = (target_price + tolerance - avg) / spread
        
        for _, row in measurement_df.iterrows():
            try:
                m_val = float(row[m_value_col])
            except:
                continue
            
            if m_low <= m_val <= m_high:
                output = avg + m_val * spread
                if abs(output - target_price) <= tolerance:
                    valid_travelers.append({
                        'origin': hlc.get('origin', 'Unknown'),
                        'datetime': hlc.get('datetime'),
                        'M #': m_val,
                        'M Name': row.get('M Name', row.get('m name', f'M{m_val}')),
                        'R #': row.get('R #', row.get('r #', '')),
                        'Tag': row.get('Tag', row.get('tag', '')),
                        'Family': row.get('Family', row.get('family', '')),
                        'Output': round(output, 2),
                        'Distance': round(abs(output - target_price), 2),
                        'H': H, 'L': L, 'C': C
                    })
    
    return valid_travelers


def detect_recip_pairs(travelers_df, tolerance=0.5):
    """
    Detect Reciprocal pairs in travelers list.
    Recip = when M# A has R# of B, and M# B has R# of A.
    """
    if travelers_df.empty:
        return pd.DataFrame()
    
    recips = []
    
    # Group by origin for same-origin matching
    for origin in travelers_df['origin'].unique():
        origin_df = travelers_df[travelers_df['origin'] == origin]
        
        for i, row1 in origin_df.iterrows():
            m1 = row1.get('M #')
            r1 = row1.get('R #')
            
            if pd.isna(m1) or pd.isna(r1):
                continue
            
            try:
                m1 = float(m1)
                r1 = float(r1)
            except:
                continue
            
            # Look for matching pair
            for j, row2 in origin_df.iterrows():
                if i >= j:
                    continue
                
                m2 = row2.get('M #')
                r2 = row2.get('R #')
                
                if pd.isna(m2) or pd.isna(r2):
                    continue
                
                try:
                    m2 = float(m2)
                    r2 = float(r2)
                except:
                    continue
                
                # Check for reciprocal relationship
                # M1's R# = M2, and M2's R# = M1
                if abs(r1 - m2) < 0.01 and abs(r2 - m1) < 0.01:
                    output1 = row1.get('Output', 0)
                    output2 = row2.get('Output', 0)
                    output_spread = abs(output1 - output2)
                    
                    if output_spread <= tolerance:
                        recips.append({
                            'origin': origin,
                            'M1 #': m1,
                            'M1 R#': r1,
                            'M1 Output': output1,
                            'M2 #': m2,
                            'M2 R#': r2,
                            'M2 Output': output2,
                            'Output Spread': round(output_spread, 4),
                            'Avg Output': round((output1 + output2) / 2, 2),
                            'datetime': row1.get('datetime')
                        })
    
    return pd.DataFrame(recips)


# ============================================================================
# HOD/LOD ANALYSIS
# ============================================================================

def detect_hod_lod(df, day_start_hour=18):
    """
    Detect High of Day (HOD) and Low of Day (LOD) for each trading day.
    Returns summary with timestamps and prices.
    """
    df = df.copy()
    df['trading_day'] = df['time'].apply(lambda x: get_trading_day(x, day_start_hour))
    
    hod_lod_results = []
    
    for trading_day, day_data in df.groupby('trading_day'):
        day_data = day_data.sort_values('time')
        
        if len(day_data) < 2:
            continue
        
        # Find HOD
        hod_price = day_data['high'].max()
        hod_row = day_data[day_data['high'] == hod_price].iloc[0]
        hod_time = hod_row['time']
        
        # Find LOD
        lod_price = day_data['low'].min()
        lod_row = day_data[day_data['low'] == lod_price].iloc[0]
        lod_time = lod_row['time']
        
        # Determine if HOD or LOD came first
        hod_first = hod_time < lod_time
        
        hod_lod_results.append({
            'trading_day': trading_day,
            'hod_price': hod_price,
            'hod_time': hod_time,
            'lod_price': lod_price,
            'lod_time': lod_time,
            'daily_range': hod_price - lod_price,
            'hod_first': hod_first,
            'session_start': day_data['time'].min(),
            'session_end': day_data['time'].max()
        })
    
    return pd.DataFrame(hod_lod_results)


def find_travelers_at_hod_lod(hod_lod_df, measurement_df, hlc_df, tolerance=24, day_start_hour=18):
    """
    Find travelers that produce outputs near HOD/LOD prices.
    Uses the traveler calculation logic from custom_range_calculator.
    """
    if hod_lod_df.empty:
        return pd.DataFrame()
    
    all_travelers = []
    
    # Get origins from HLC dataframe
    origins = []
    for col in hlc_df.columns:
        if col.endswith(' H'):
            origin_name = col[:-2]
            if f'{origin_name} L' in hlc_df.columns and f'{origin_name} C' in hlc_df.columns:
                origins.append(origin_name)
    
    for _, hod_lod_row in hod_lod_df.iterrows():
        trading_day = hod_lod_row['trading_day']
        hod_price = hod_lod_row['hod_price']
        lod_price = hod_lod_row['lod_price']
        hod_time = hod_lod_row['hod_time']
        lod_time = hod_lod_row['lod_time']
        
        # For each origin, find HLC values
        for origin in origins:
            h_col = f'{origin} H'
            l_col = f'{origin} L'
            c_col = f'{origin} C'
            
            # Get HLC at or before the HOD/LOD time
            hlc_df_sorted = hlc_df.copy()
            if 'time' in hlc_df_sorted.columns:
                hlc_df_sorted['time'] = pd.to_datetime(hlc_df_sorted['time'], errors='coerce')
                hlc_df_sorted = hlc_df_sorted.sort_values('time')
            
            # Use the most recent HLC values
            for idx, hlc_row in hlc_df_sorted.iterrows():
                if pd.isna(hlc_row[h_col]) or pd.isna(hlc_row[l_col]) or pd.isna(hlc_row[c_col]):
                    continue
                
                H = float(hlc_row[h_col])
                L = float(hlc_row[l_col])
                C = float(hlc_row[c_col])
                
                avg = (H + L + C) / 3.0
                spread = H - L
                
                if spread == 0:
                    continue
                
                # Check HOD
                m_value_col = next((c for c in ['M value', 'M Value', 'M #', 'm #'] 
                                   if c in measurement_df.columns), None)
                if m_value_col is None:
                    continue
                
                for target_price, target_type, target_time in [
                    (hod_price, 'HOD', hod_time),
                    (lod_price, 'LOD', lod_time)
                ]:
                    # Calculate M range for this target
                    m_low = (target_price - tolerance - avg) / spread
                    m_high = (target_price + tolerance - avg) / spread
                    
                    for _, m_row in measurement_df.iterrows():
                        try:
                            m_val = float(m_row[m_value_col])
                        except:
                            continue
                        
                        if m_low <= m_val <= m_high:
                            output = avg + m_val * spread
                            distance = abs(output - target_price)
                            
                            if distance <= tolerance:
                                all_travelers.append({
                                    'trading_day': trading_day,
                                    'target_type': target_type,
                                    'target_price': target_price,
                                    'target_time': target_time,
                                    'origin': origin,
                                    'M #': m_val,
                                    'M Name': m_row.get('M Name', m_row.get('m name', f'M{m_val}')),
                                    'R #': m_row.get('R #', m_row.get('r #', '')),
                                    'Tag': m_row.get('Tag', m_row.get('tag', '')),
                                    'Family': m_row.get('Family', m_row.get('family', '')),
                                    'Output': round(output, 2),
                                    'Distance': round(distance, 2),
                                    'H': H, 'L': L, 'C': C
                                })
    
    return pd.DataFrame(all_travelers)


# ============================================================================
# CONFLUENCE ANALYSIS
# ============================================================================

def find_ma_confluence_at_swings(df, swings_df, ma_columns, zone_size=24):
    """Find MAs present near swing entry points."""
    if swings_df.empty or df.empty:
        return pd.DataFrame()
    
    confluence_records = []
    
    for _, swing in swings_df.iterrows():
        from_time = swing.get('from_datetime') or swing.get('from_time')
        from_price = swing.get('from_price')
        direction = swing.get('direction')
        
        if pd.isna(from_time) or pd.isna(from_price):
            continue
        
        # Get candles around the swing entry
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
                
                distance_from_swing = abs(ma_value - from_price)
                
                if distance_from_swing <= zone_size:
                    in_wick, wick_type = is_in_wick(candle, ma_value)
                    
                    confluence_records.append({
                        'swing_time': from_time,
                        'swing_price': from_price,
                        'swing_direction': direction,
                        'swing_move_size': swing.get('move_size', 0),
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


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.title("📊 Market Swing Analysis Tool 05e_7 Two Pass")
    st.markdown("**Integrated Version** - Swing Detection, MA Analysis, NY Session, Traveler/Pivot Calculations")
    st.markdown("---")
    
    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    
    day_start_hour = st.sidebar.radio(
        "Trading Day Start",
        [18, 17],
        format_func=lambda x: f"{x}:00"
    )
    
    swing_threshold = st.sidebar.slider(
        "Swing Threshold (min move)",
        min_value=20,
        max_value=100,
        value=50,
        step=5
    )
    
    drawdown_limit = st.sidebar.slider(
        "Drawdown/Bounce Confirmation",
        min_value=10,
        max_value=50,
        value=25,
        step=5
    )
    
    min_ma_distance = st.sidebar.slider(
        "Min MA Move Distance",
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
    st.markdown("Upload your OHLC data files (CSV or Excel)")
    
    # 4 file uploaders in one row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        file_3m = st.file_uploader("3m OHLC", type=['csv', 'xlsx', 'xls'], key="file_3m")
    
    with col2:
        file_5m = st.file_uploader("5m OHLC", type=['csv', 'xlsx', 'xls'], key="file_5m")
    
    with col3:
        file_6m = st.file_uploader("6m OHLC", type=['csv', 'xlsx', 'xls'], key="file_6m")
    
    with col4:
        file_15m = st.file_uploader("15m OHLC", type=['csv', 'xlsx', 'xls'], key="file_15m")
    
    # Traveler files (for Strategic Zones)
    st.markdown("---")
    st.markdown("### 📊 Feed Data (For Strategic Zones Tab 8)")
    st.info("💡 Upload RAW HLC feed data - the app will generate custom reciprocal traveler reports")
    
    with st.expander("ℹ️ About Feed Data Format"):
        st.markdown("""
        **These should be RAW HLC feed files from your trading system.**
        
        Required format:
        - CSV files with columns: `time`, `Spain H`, `Spain L`, `Spain C`, `Jupiter H`, `Jupiter L`, etc.
        - One row per timestamp
        - Three columns per origin (H, L, C)
        
        The app will:
        1. Extract Anchor and Epic origins only
        2. Focus on Day [0] arrivals (today at report time, e.g., 18:00)
        3. Include previous trading day (for weekend handling)
        4. Generate travelers using Recipe M# pairs from G.11
        5. Find reciprocal matches within Max Spread
        
        **This is DIFFERENT from Tab 7:**
        - Tab 7 uses a separate HLC file upload inside the tab
        - Both use the same HLC format, just uploaded in different places
        """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        small_feed_file = st.file_uploader(
            "Small Feed HLC CSV",
            type=['csv'],
            key="small_feed",
            help="Raw HLC data with columns like 'Spain H', 'Spain L', 'Spain C'"
        )
        if small_feed_file:
            st.success(f"✅ {small_feed_file.name}")
    
    with col2:
        big_feed_file = st.file_uploader(
            "Big Feed HLC CSV",
            type=['csv'],
            key="big_feed",
            help="Raw HLC data with columns like 'Jupiter H', 'Jupiter L', 'Jupiter C'"
        )
        if big_feed_file:
            st.success(f"✅ {big_feed_file.name}")
    
    with col3:
        measurement_file = st.file_uploader(
            "Measurement File",
            type=['xlsx', 'xls'],
            key="measurement",
            help="Excel file with M# and R# relationships"
        )
        if measurement_file:
            st.success(f"✅ {measurement_file.name}")
    
    # Load files
    files = {
        '3m': load_ohlc_file(file_3m, '3m'),
        '5m': load_ohlc_file(file_5m, '5m'),
        '6m': load_ohlc_file(file_6m, '6m'),
        '15m': load_ohlc_file(file_15m, '15m')
    }
    
    loaded_files = {k: v for k, v in files.items() if v is not None}
    
    if not loaded_files:
        st.info("👆 Please upload at least one OHLC file to begin analysis.")
        
        # Show expected format
        st.subheader("📋 Expected File Format")
        sample_data = {
            'time': ['2025-06-15T18:00:00-04:00', '2025-06-15T18:15:00-04:00'],
            'open': [21784, 21821.25],
            'high': [21850.75, 21842],
            'low': [21722, 21815],
            'close': [21821.25, 21835]
        }
        st.table(pd.DataFrame(sample_data))
        return
    
    # Display loaded files info
    st.success(f"✅ Loaded {len(loaded_files)} file(s): {', '.join(loaded_files.keys())}")
    
    with st.expander("📋 File Details"):
        for tf, df in loaded_files.items():
            ma_cols = get_all_ma_columns(df)
            st.markdown(f"**{tf}**: {len(df)} rows, {len(df.columns)} columns, {len(ma_cols)} MA columns")
            st.markdown(f"  Time range: {df['time'].min()} to {df['time'].max()}")
    
    st.markdown("---")
    
    # Check minimum files for Strategic Zones
    min_files_met = all([
        small_feed_file is not None,
        big_feed_file is not None,
        measurement_file is not None,
        len(loaded_files) >= 1  # At least one OHLC file
    ])
    
    if min_files_met:
        st.success(f"✅ **Tab 8 - Strategic Zones READY!** All required files loaded ({len(loaded_files)} OHLC + 3 feed files)")
        st.info("👉 Click the '🎯 Strategic Zones' tab below to generate custom reciprocal traveler reports")
    elif len(loaded_files) > 0:
        missing = []
        if not small_feed_file:
            missing.append("Small Feed HLC CSV")
        if not big_feed_file:
            missing.append("Big Feed HLC CSV")
        if not measurement_file:
            missing.append("Measurement File")
        if missing:
            st.warning(f"⚠️ **Tab 8 needs:** {', '.join(missing)}")
    else:
        st.info("ℹ️ Upload OHLC and Feed files above to enable Tab 8 - Strategic Zones")
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔄 Swing Detection",
        "🗽 NY Session",
        "📈 MA Wick Report",
        "🎯 MA Confluence",
        "📜 MA History",
        "📊 HOD/LOD Analysis",
        "🧭 Traveler Calculator",
        "🎯 Strategic Zones"  # NEW TAB
    ])
    
    # ========================
    # TAB 1: SWING DETECTION
    # ========================
    with tab1:
        st.header("Full Swing Detection")
        
        swing_tf = st.selectbox(
            "Select timeframe for swing analysis",
            list(loaded_files.keys()),
            key="swing_tf"
        )
        
        if st.button("🔍 Analyze Swings", key="analyze_swings"):
            analysis_df = loaded_files[swing_tf]
            
            with st.spinner("Analyzing daily swings..."):
                daily_stats = analyze_daily_data(
                    analysis_df,
                    day_start_hour=day_start_hour,
                    swing_threshold=swing_threshold,
                    drawdown_limit=drawdown_limit,
                    exit_at_16=exit_at_16
                )
                
                if daily_stats:
                    # Summary table
                    summary_data = []
                    detailed_swings = []
                    
                    for day in daily_stats:
                        summary_data.append({
                            'Trading Day': day['trading_day'],
                            'Range': day['daily_range'],
                            'Category': day['range_category'],
                            'Total Swings': day['total_swings'],
                            'NY Swings': day['ny_swings_count'],
                            'Top 1': day['top_1_swing'],
                            'Top 2': day['top_2_swing'],
                            'Top 3': day['top_3_swing'],
                            '200+': day['swings_200_plus'],
                            '150-200': day['swings_150_200'],
                            '100-150': day['swings_100_150']
                        })
                        
                        # Detailed swings
                        for i, swing in enumerate(day['all_swings']):
                            detailed_swings.append({
                                'trading_day': day['trading_day'],
                                'swing_id': i + 1,
                                'swing_type': swing['type'],
                                'direction': swing['direction'],
                                'from_datetime': swing['from_time'],
                                'from_price': swing['from_price'],
                                'to_datetime': swing['to_time'],
                                'to_price': swing['to_price'],
                                'move_size': swing['move_size'],
                                'category': swing['category']
                            })
                    
                    summary_df = pd.DataFrame(summary_data)
                    detailed_df = pd.DataFrame(detailed_swings)
                    
                    st.markdown("### Daily Summary")
                    st.dataframe(summary_df, use_container_width=True)
                    
                    st.markdown("### Detailed Swings")
                    st.dataframe(detailed_df, use_container_width=True)
                    
                    # Store for other tabs
                    st.session_state['swings_df'] = detailed_df
                    st.session_state['daily_stats'] = daily_stats
                    
                    # Export
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        summary_df.to_excel(writer, sheet_name='Daily Summary', index=False)
                        detailed_df.to_excel(writer, sheet_name='All Swings', index=False)
                        
                        # Per-day sheets
                        for day in daily_stats:
                            day_swings = [s for s in detailed_swings if s['trading_day'] == day['trading_day']]
                            if day_swings:
                                sheet_name = str(day['trading_day']).replace('-', '.')[:31]
                                pd.DataFrame(day_swings).to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    st.download_button(
                        "📥 Download Swing Report",
                        data=excel_buffer.getvalue(),
                        file_name=f"swing_analysis_05e_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("No trading days found in the data.")
    
    # ========================
    # TAB 2: NY SESSION
    # ========================
    with tab2:
        st.header("🗽 NY Session Analysis (8 AM - 4 PM)")
        
        ny_tf = st.selectbox(
            "Select timeframe for NY session",
            list(loaded_files.keys()),
            key="ny_tf"
        )
        
        if st.button("🔍 Analyze NY Session", key="analyze_ny"):
            analysis_df = loaded_files[ny_tf]
            
            with st.spinner("Analyzing NY session..."):
                ny_results = analyze_ny_session(
                    analysis_df,
                    day_start_hour=day_start_hour,
                    swing_threshold=swing_threshold,
                    drawdown_limit=drawdown_limit
                )
                
                if ny_results:
                    ny_summary = []
                    ny_detailed = []
                    
                    for day in ny_results:
                        ny_summary.append({
                            'Trading Day': day['trading_day'],
                            'NY High': day['ny_high'],
                            'NY High Time': day['ny_high_time'],
                            'NY Low': day['ny_low'],
                            'NY Low Time': day['ny_low_time'],
                            'NY Range': day['ny_range'],
                            'Category': day['ny_range_category'],
                            'NY Swings': day['ny_swing_count']
                        })
                        
                        for i, swing in enumerate(day['ny_swings']):
                            ny_detailed.append({
                                'trading_day': day['trading_day'],
                                'swing_id': f'NY {i+1}',
                                'direction': swing['direction'],
                                'from_datetime': swing['from_time'],
                                'from_price': swing['from_price'],
                                'to_datetime': swing['to_time'],
                                'to_price': swing['to_price'],
                                'move_size': swing['move_size'],
                                'category': swing['category']
                            })
                    
                    st.markdown("### NY Session Summary")
                    st.dataframe(pd.DataFrame(ny_summary), use_container_width=True)
                    
                    if ny_detailed:
                        st.markdown("### NY Session Swings")
                        st.dataframe(pd.DataFrame(ny_detailed), use_container_width=True)
                    
                    # Export
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        pd.DataFrame(ny_summary).to_excel(writer, sheet_name='NY Summary', index=False)
                        if ny_detailed:
                            pd.DataFrame(ny_detailed).to_excel(writer, sheet_name='NY Swings', index=False)
                    
                    st.download_button(
                        "📥 Download NY Session Report",
                        data=excel_buffer.getvalue(),
                        file_name=f"ny_session_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("No NY session data found.")
    
    # ========================
    # TAB 3: MA WICK REPORT
    # ========================
    with tab3:
        st.header("Moving Averages Wick Interaction Report")
        
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
    # TAB 4: MA CONFLUENCE
    # ========================
    with tab4:
        st.header("MA Confluence at Swing Points")
        
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
                        
                        st.markdown("### MAs Most Frequently at Swing Points")
                        ma_freq = confluence.groupby('ma_column').agg({
                            'swing_time': 'count',
                            'swing_move_size': 'mean',
                            'in_wick': 'sum'
                        }).round(2)
                        ma_freq.columns = ['Occurrences', 'Avg Swing Size', 'In Wick Count']
                        ma_freq = ma_freq.sort_values('Occurrences', ascending=False)
                        st.dataframe(ma_freq.head(20), use_container_width=True)
                        
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
    # TAB 5: MA HISTORY TRACKER
    # ========================
    with tab5:
        st.header("MA History Tracker")
        
        track_tf = st.selectbox(
            "Select timeframe",
            list(loaded_files.keys()),
            key="track_tf_select"
        )
        
        if track_tf:
            track_df = loaded_files[track_tf]
            ma_columns = get_all_ma_columns(track_df)
            ma_names = [ma['column'] for ma in ma_columns]
            
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
                        
                        significant_count = history['significant'].sum()
                        st.markdown(f"**Significant moves (≥{min_ma_distance} units):** {significant_count} / {len(history)} ({100*significant_count/len(history):.1f}%)")
                        
                        st.dataframe(history, use_container_width=True)
                        
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
    
    # ========================
    # TAB 6: HOD/LOD ANALYSIS
    # ========================
    with tab6:
        st.header("📊 HOD/LOD Analysis")
        st.markdown("Analyze High of Day (HOD) and Low of Day (LOD) for each trading day")
        
        hod_tf = st.selectbox(
            "Select timeframe for HOD/LOD analysis",
            list(loaded_files.keys()),
            key="hod_tf_select"
        )
        
        if st.button("🔍 Detect HOD/LOD", key="detect_hod_lod"):
            analysis_df = loaded_files[hod_tf]
            
            with st.spinner("Analyzing HOD/LOD..."):
                hod_lod_df = detect_hod_lod(analysis_df, day_start_hour=day_start_hour)
                
                if not hod_lod_df.empty:
                    st.success(f"Found HOD/LOD for {len(hod_lod_df)} trading days")
                    
                    # Display summary
                    display_df = hod_lod_df.copy()
                    display_df['daily_range'] = display_df['daily_range'].round(2)
                    display_df['hod_first'] = display_df['hod_first'].map({True: '↑ HOD First', False: '↓ LOD First'})
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Statistics
                    st.markdown("### Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_range = hod_lod_df['daily_range'].mean()
                        st.metric("Avg Daily Range", f"{avg_range:.1f}")
                    
                    with col2:
                        max_range = hod_lod_df['daily_range'].max()
                        st.metric("Max Daily Range", f"{max_range:.1f}")
                    
                    with col3:
                        hod_first_pct = (hod_lod_df['hod_first'] == True).sum() / len(hod_lod_df) * 100
                        st.metric("HOD First %", f"{hod_first_pct:.1f}%")
                    
                    with col4:
                        lod_first_pct = 100 - hod_first_pct
                        st.metric("LOD First %", f"{lod_first_pct:.1f}%")
                    
                    # Store for traveler analysis
                    st.session_state['hod_lod_df'] = hod_lod_df
                    
                    # Export
                    excel_buffer = io.BytesIO()
                    hod_lod_df.to_excel(excel_buffer, index=False)
                    
                    st.download_button(
                        "📥 Download HOD/LOD Report",
                        data=excel_buffer.getvalue(),
                        file_name=f"hod_lod_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("No trading days found in the data.")
    
    # ========================
    # TAB 7: TRAVELER CALCULATOR
    # ========================
    with tab7:
        st.header("🧭 Traveler (Pivot) Calculator")
        st.markdown("""
        Calculate travelers that produce outputs at specific price targets.
        Uses the pivot formula: **Output = (H + L + C) / 3 + M × (H - L)**
        """)
        
        st.info("ℹ️ **Note:** This tab requires DIFFERENT files than the main upload section above.")
        
        # File uploads for HLC and Measurement data
        st.markdown("### Data Sources (Upload Here)")
        
        st.markdown("""
        **Required format:**
        - **HLC Data:** CSV/Excel with columns like "Spain H", "Spain L", "Spain C", "Jupiter H", etc.
        - **Measurement:** Excel file with M# and R# lookup table
        
        ⚠️ These are DIFFERENT from the Feed CSV files used in Tab 8.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            tab7_hlc_file = st.file_uploader(
                "Upload HLC Data (CSV with origin H/L/C columns)",
                type=['csv', 'xlsx', 'xls'],
                key="hlc_upload"
            )
        
        with col2:
            tab7_measurement_file = st.file_uploader(
                "Upload Measurement File (Excel with M values)",
                type=['xlsx', 'xls'],
                key="measurement_upload"
            )
        
        if tab7_hlc_file is not None and tab7_measurement_file is not None:
            try:
                # Load HLC data
                if tab7_hlc_file.name.endswith('.csv'):
                    hlc_df = pd.read_csv(tab7_hlc_file)
                else:
                    hlc_df = pd.read_excel(tab7_hlc_file)
                
                # Load measurement data
                measurement_df = pd.read_excel(tab7_measurement_file)
                
                st.success(f"Loaded HLC data: {len(hlc_df)} rows, Measurements: {len(measurement_df)} rows")
                
                # Detect origins
                origins = []
                for col in hlc_df.columns:
                    if col.endswith(' H'):
                        origin_name = col[:-2]
                        if f'{origin_name} L' in hlc_df.columns and f'{origin_name} C' in hlc_df.columns:
                            origins.append(origin_name)
                
                st.info(f"Detected {len(origins)} origins: {', '.join(origins[:10])}{'...' if len(origins) > 10 else ''}")
                
                st.markdown("---")
                st.markdown("### Calculate Travelers at Price Target")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    target_price = st.number_input(
                        "Target Price",
                        value=25000.0,
                        step=1.0,
                        help="Price level to find travelers for"
                    )
                
                with col2:
                    price_tolerance = st.number_input(
                        "Tolerance (±)",
                        value=24.0,
                        min_value=1.0,
                        max_value=100.0,
                        step=1.0,
                        help="Acceptable distance from target"
                    )
                
                if st.button("🔍 Find Travelers", key="find_travelers"):
                    with st.spinner("Calculating travelers..."):
                        # Build HLC sets for each origin
                        hlc_sets = []
                        
                        for origin in origins:
                            h_col = f'{origin} H'
                            l_col = f'{origin} L'
                            c_col = f'{origin} C'
                            
                            # Get the most recent non-null row
                            for idx in range(len(hlc_df) - 1, -1, -1):
                                row = hlc_df.iloc[idx]
                                if not pd.isna(row[h_col]) and not pd.isna(row[l_col]) and not pd.isna(row[c_col]):
                                    hlc_sets.append({
                                        'origin': origin,
                                        'H': float(row[h_col]),
                                        'L': float(row[l_col]),
                                        'C': float(row[c_col]),
                                        'datetime': row.get('time', None)
                                    })
                                    break
                        
                        travelers = find_travelers_at_price(
                            measurement_df,
                            target_price,
                            tolerance=price_tolerance,
                            hlc_sets=hlc_sets
                        )
                        
                        if travelers:
                            travelers_df = pd.DataFrame(travelers)
                            travelers_df = travelers_df.sort_values('Distance')
                            
                            st.success(f"Found {len(travelers_df)} travelers near {target_price}")
                            st.dataframe(travelers_df, use_container_width=True)
                            
                            # Check for Recip pairs
                            st.markdown("### Reciprocal Pair Detection")
                            recips = detect_recip_pairs(travelers_df, tolerance=1.0)
                            
                            if not recips.empty:
                                st.success(f"Found {len(recips)} Recip pairs!")
                                st.dataframe(recips, use_container_width=True)
                            else:
                                st.info("No Recip pairs found at this price level")
                            
                            # Export
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                travelers_df.to_excel(writer, sheet_name='Travelers', index=False)
                                if not recips.empty:
                                    recips.to_excel(writer, sheet_name='Recips', index=False)
                            
                            st.download_button(
                                "📥 Download Traveler Report",
                                data=excel_buffer.getvalue(),
                                file_name=f"travelers_{target_price:.0f}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            st.warning(f"No travelers found within {price_tolerance} units of {target_price}")
                
                # HOD/LOD Traveler Analysis
                st.markdown("---")
                st.markdown("### Find Travelers at HOD/LOD")
                
                if 'hod_lod_df' in st.session_state and not st.session_state['hod_lod_df'].empty:
                    hod_lod_df = st.session_state['hod_lod_df']
                    
                    selected_day = st.selectbox(
                        "Select Trading Day",
                        hod_lod_df['trading_day'].tolist(),
                        key="hod_lod_day_select"
                    )
                    
                    day_data = hod_lod_df[hod_lod_df['trading_day'] == selected_day].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("HOD", f"{day_data['hod_price']:.2f}")
                    with col2:
                        st.metric("LOD", f"{day_data['lod_price']:.2f}")
                    
                    if st.button("🎯 Find Travelers at HOD/LOD", key="find_hod_lod_travelers"):
                        with st.spinner("Finding travelers at HOD/LOD..."):
                            # Build HLC sets
                            hlc_sets = []
                            for origin in origins:
                                h_col = f'{origin} H'
                                l_col = f'{origin} L'
                                c_col = f'{origin} C'
                                
                                for idx in range(len(hlc_df) - 1, -1, -1):
                                    row = hlc_df.iloc[idx]
                                    if not pd.isna(row[h_col]) and not pd.isna(row[l_col]) and not pd.isna(row[c_col]):
                                        hlc_sets.append({
                                            'origin': origin,
                                            'H': float(row[h_col]),
                                            'L': float(row[l_col]),
                                            'C': float(row[c_col]),
                                            'datetime': row.get('time', None)
                                        })
                                        break
                            
                            # Find at HOD
                            hod_travelers = find_travelers_at_price(
                                measurement_df,
                                day_data['hod_price'],
                                tolerance=price_tolerance,
                                hlc_sets=hlc_sets
                            )
                            
                            # Find at LOD
                            lod_travelers = find_travelers_at_price(
                                measurement_df,
                                day_data['lod_price'],
                                tolerance=price_tolerance,
                                hlc_sets=hlc_sets
                            )
                            
                            st.markdown("#### Travelers at HOD")
                            if hod_travelers:
                                hod_df = pd.DataFrame(hod_travelers).sort_values('Distance')
                                st.dataframe(hod_df.head(20), use_container_width=True)
                            else:
                                st.info("No travelers found at HOD")
                            
                            st.markdown("#### Travelers at LOD")
                            if lod_travelers:
                                lod_df = pd.DataFrame(lod_travelers).sort_values('Distance')
                                st.dataframe(lod_df.head(20), use_container_width=True)
                            else:
                                st.info("No travelers found at LOD")
                else:
                    st.info("👆 Run HOD/LOD Detection first (Tab 6) to analyze travelers at HOD/LOD")
            
            except Exception as e:
                st.error(f"Error loading files: {str(e)}")
        else:
            st.info("👆 Upload HLC data and Measurement file to use the Traveler Calculator")
    
    # ========================
    # TAB 8: STRATEGIC ZONES
    # ========================
    with tab8:
        st.header("🎯 Strategic Zone Recommendations")
        st.markdown("""
        **High-Probability Turning Zones** identified through:
        - 📊 High-rank Recip pairs (Epic + Anchor combinations)
        - 📈 HUGE HMA confluences (h1-h20 + high-timeframe MAs)
        - ⚡ Wildcard M# emergence (0, ±40, ±54)
        - 🔄 MA role transitions (resistance ↔ support)
        """)
        
        # DEBUG: Show file status
        with st.expander("🔍 Debug: File Upload Status"):
            st.markdown("**Main Section Files:**")
            st.markdown(f"- Small Feed: {'✅ ' + small_feed_file.name if small_feed_file else '❌ Not uploaded'}")
            st.markdown(f"- Big Feed: {'✅ ' + big_feed_file.name if big_feed_file else '❌ Not uploaded'}")
            st.markdown(f"- Measurement: {'✅ ' + measurement_file.name if measurement_file else '❌ Not uploaded'}")
            st.markdown(f"- OHLC Files: {len(loaded_files)} files loaded")
            st.markdown(f"\n**min_files_met:** {min_files_met}")
        
        # Check minimum files
        if not min_files_met:
            st.warning("⚠️ Strategic Zones requires minimum 7 files:")
            st.markdown("""
            **Minimum Requirements:**
            - ✅ At least 1 OHLC file (3m, 5m, 6m, or 15m)
            - ✅ Small Feed 15m CSV (for travelers)
            - ✅ Big Feed 15m CSV (for travelers)
            - ✅ Measurement File (for M# values)
            """)
            
            files_status = {
                'OHLC files': len(loaded_files),
                'Small Feed': '✅' if small_feed_file else '❌',
                'Big Feed': '✅' if big_feed_file else '❌',
                'Measurement': '✅' if measurement_file else '❌'
            }
            
            for file_type, status in files_status.items():
                st.markdown(f"- {file_type}: {status}")
            
            st.info("👆 Please upload all required files above to enable Strategic Zones analysis")
            
        elif not (small_feed_file and big_feed_file and measurement_file):
            # Double-check that all traveler files are uploaded
            st.error("⚠️ Missing traveler files!")
            st.markdown("Please upload:")
            if not small_feed_file:
                st.markdown("- ❌ Small Feed 15m CSV")
            if not big_feed_file:
                st.markdown("- ❌ Big Feed 15m CSV")
            if not measurement_file:
                st.markdown("- ❌ Measurement File (Excel)")
            
        else:
            # All files present, proceed with analysis
            try:
                st.markdown("### 📊 Loading Feed Data...")
                
                # Load HLC feeds
                small_hlc_df = pd.read_csv(small_feed_file)
                big_hlc_df = pd.read_csv(big_feed_file)
                measurement_df = pd.read_excel(measurement_file)
                
                st.success(f"Loaded: Small feed ({len(small_hlc_df)} rows), Big feed ({len(big_hlc_df)} rows), Measurements ({len(measurement_df)} rows)")
                
                # Show detected HLC columns
                with st.expander("🔍 Detected HLC Columns"):
                    st.markdown("**Small Feed:**")
                    hlc_cols_small = [col for col in small_hlc_df.columns if col.endswith((' H', ' L', ' C'))]
                    origins_small = list(set([col[:-2] for col in hlc_cols_small if col.endswith(' H')]))
                    st.code(f"Origins: {', '.join(origins_small)}")
                    
                    st.markdown("**Big Feed:**")
                    hlc_cols_big = [col for col in big_hlc_df.columns if col.endswith((' H', ' L', ' C'))]
                    origins_big = list(set([col[:-2] for col in hlc_cols_big if col.endswith(' H')]))
                    st.code(f"Origins: {', '.join(origins_big)}")
                
                st.markdown("---")
                st.markdown("### ⚙️ Report Settings")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    report_time_mode = st.radio(
                        "Report Time",
                        ["Now (Real-time)", "Custom Time"],
                        key="report_time_mode"
                    )
                    
                    if report_time_mode == "Custom Time":
                        report_date = st.date_input("Date", value=dt.date.today(), key="strat_date")
                        report_time_input = st.time_input("Time", value=dt.time(18, 0), key="strat_time")
                        report_time = datetime.combine(report_date, report_time_input)
                    else:
                        report_time = datetime.now()
                    
                    st.info(f"📅 Report: {report_time.strftime('%Y-%m-%d %H:%M')}")
                
                with col2:
                    lookback_days = st.slider("Lookback Days", 1, 60, 20, 1, key="lookback_days")
                    st.markdown("*How many days to look back (default: 20)*")
                
                with col3:
                    max_spread = st.slider("Max Output Spread", 0.5, 10.0, 3.0, 0.5, key="max_output_spread")
                    st.markdown("*Maximum output spread for cluster matches*")
                
                with col4:
                    max_zones = st.slider("Max Zones to Show", 1, 6, 4, key="max_zones")
                    zone_tolerance = st.slider("Zone Tolerance (±)", 12, 48, 24, 6, key="zone_tol")
                
                st.markdown("---")
                
                # Processing Mode Selection
                st.markdown("### 🔧 Processing Mode")
                processing_mode = st.radio(
                    "Select processing mode for cluster table generation:",
                    ["Full Range", "Custom Ranges", "HOD/LOD Mode"],
                    index=0,
                    key="cluster_processing_mode",
                    horizontal=True
                )
                
                # Mode-specific settings
                # Common settings for all modes
                st.markdown("**Window Settings:**")
                window_radius = st.slider("Window Radius", 50, 1000, 150, 50, key="window_radius")
                st.caption("Points above/below each feed's Open (Input @ start) for range calculation")
                
                st.markdown("---")
                
                if processing_mode == "Full Range":
                    st.markdown("**Full Range Settings:**")
                    st.info("Full Range: Processes all M#s within window around each feed's Open")
                
                elif processing_mode == "Custom Ranges":
                    st.markdown("**Custom Ranges Settings:**")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        use_high1 = st.checkbox("Use High 1", value=True, key="use_high1")
                        high1 = st.number_input("High 1", value=25500.0, step=10.0, key="high1") if use_high1 else None
                    with col_b:
                        use_high2 = st.checkbox("Use High 2", value=False, key="use_high2")
                        high2 = st.number_input("High 2", value=26000.0, step=10.0, key="high2") if use_high2 else None
                    with col_c:
                        use_low1 = st.checkbox("Use Low 1", value=True, key="use_low1")
                        low1 = st.number_input("Low 1", value=24500.0, step=10.0, key="low1") if use_low1 else None
                    with col_d:
                        use_low2 = st.checkbox("Use Low 2", value=False, key="use_low2")
                        low2 = st.number_input("Low 2", value=24000.0, step=10.0, key="low2") if use_low2 else None
                    st.caption("Custom Ranges: Define specific high/low ranges to analyze")
                
                elif processing_mode == "HOD/LOD Mode":
                    st.markdown("**HOD/LOD Settings:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        hod_cutoff_time = st.time_input("HOD Cutoff Time", value=dt.time(9, 45), key="hod_cutoff")
                        st.caption("Ignore HOD/LOD if occurs before this time")
                    with col_b:
                        st.info("HOD/LOD: Focuses on High of Day / Low of Day analysis")
                
                st.markdown("---")
                
                # ========================================
                # CLUSTER TABLES: FOGZ, Large Discounts, Recips PD
                # ========================================
                st.markdown("### 📊 Cluster Tables Analysis")
                st.markdown(f"**Modeled after Model G.11** - FOGZ, Large Discounts, and Recips PD analysis ({processing_mode} mode)")
                
                if st.button("🎯 Generate Cluster Tables", key="gen_cluster_tables", type="primary"):
                    with st.spinner("Generating cluster tables with two-pass processing..."):
                        total_start = time_module.time()
                        
                        # Define valid lists for each table
                        # Pass 1: Most recent 2 days
                        FOGZ_PASS1 = {0, 1, -1, 2, -2, 3, -3, 5, -5, 6, -6}
                        LD_PASS1 = {36, -36, 38, -38, 39, -39}
                        RECIPS_PASS1 = {1, -1, 2, -2, 3, -3, 5, -5, 6, -6, 10, -10, 14, -14, 15, -15, 22, -22, 27, -27, 30, -30, 36, -36, 38, -38, 39, -39}
                        
                        # Pass 2: All data within lookback
                        FOGZ_PASS2 = {40, -40, 41, -41, 43, -43, 50, -50, 55, -55, 60, -60, 68, -68, 77, -77, 87, -87, 96, -96, 103, -103, 107, -107, 111, -111}
                        LD_PASS2 = {41, -41, 43, -43, 50, -50, 55, -55, 60, -60, 68, -68, 77, -77, 87, -87, 96, -96, 103, -103, 107, -107, 111, -111}
                        RECIPS_PASS2 = {41, -41, 42, -42, 43, -43, 45, -45, 50, -50, 54, -54, 55, -55, 60, -60, 67, -67, 68, -68, 77, -77, 87, -87, 96, -96, 103, -103, 107, -107, 111, -111}
                        
                        cluster_results = {}
                        prep_tables = {}
                        
                        # ========================================
                        # FOGZ TABLE
                        # ========================================
                        st.info("⏳ Generating FOGZ prep table (two-pass)...")
                        fogz_start = time_module.time()
                        
                        fogz_prep, fogz_summary = process_cluster_tables_two_pass(
                            measurement_df=measurement_df,
                            small_df=small_hlc_df,
                            big_df=big_hlc_df,
                            report_time=report_time,
                            scope_days=lookback_days,
                            valid_list_pass1=FOGZ_PASS1,
                            valid_list_pass2=FOGZ_PASS2,
                            max_output_spread=max_spread,
                            window_radius=window_radius
                        )
                        
                        # Store summary for export
                        prep_tables['fogz_summary'] = fogz_summary
                        
                        # ALWAYS display processing summary (even if no results)
                        with st.expander("📊 FOGZ Processing Summary", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Pass 1 Results", fogz_summary['pass1_count'])
                                st.caption(f"HLCs Examined: {fogz_summary.get('pass1_hlcs_examined', 0)}")
                                if fogz_summary.get('pass1_dates'):
                                    st.caption(f"Dates: {', '.join(map(str, fogz_summary['pass1_dates'][:3]))}")
                                if fogz_summary['pass1_m_numbers']:
                                    st.caption(f"M #s found: {', '.join(map(str, fogz_summary['pass1_m_numbers'][:10]))}")
                            with col2:
                                st.metric("Pass 2 Results", fogz_summary['pass2_count'])
                                st.caption(f"HLCs Examined: {fogz_summary.get('pass2_hlcs_examined', 0)}")
                                if fogz_summary.get('pass2_dates'):
                                    date_list = sorted(list(fogz_summary['pass2_dates']))
                                    st.caption(f"Date range: {date_list[0] if date_list else 'N/A'} to {date_list[-1] if date_list else 'N/A'}")
                                if fogz_summary['pass2_m_numbers']:
                                    st.caption(f"M #s found: {', '.join(map(str, fogz_summary['pass2_m_numbers'][:10]))}")
                            with col3:
                                st.metric("Total Results", fogz_summary['total_count'])
                                st.caption(f"Feeds: {', '.join(fogz_summary.get('pass1_feeds', []))}")
                            
                            # Show detailed HLC processing
                            if fogz_summary.get('pass1_hlcs_examined', 0) > 0 or fogz_summary.get('pass2_hlcs_examined', 0) > 0:
                                st.markdown("---")
                                st.markdown("**🔍 Detailed HLC Examination:**")
                                
                                tab1, tab2 = st.tabs(["Pass 1 HLCs", "Pass 2 HLCs"])
                                
                                with tab1:
                                    if not fogz_summary.get('pass1_processing_details', pd.DataFrame()).empty:
                                        st.dataframe(fogz_summary['pass1_processing_details'], use_container_width=True, height=200)
                                        st.caption(f"Examined {len(fogz_summary['pass1_processing_details'])} HLC entries in Pass 1")
                                    else:
                                        st.info("No HLCs examined in Pass 1")
                                
                                with tab2:
                                    if not fogz_summary.get('pass2_processing_details', pd.DataFrame()).empty:
                                        st.dataframe(fogz_summary['pass2_processing_details'], use_container_width=True, height=200)
                                        st.caption(f"Examined {len(fogz_summary['pass2_processing_details'])} HLC entries in Pass 2")
                                    else:
                                        st.info("No HLCs examined in Pass 2")
                        
                        if not fogz_prep.empty:
                            st.success(f"✅ FOGZ prep: {len(fogz_prep)} entries")
                            prep_tables['fogz_prep'] = fogz_prep
                            
                            # Now match pass1 with pass2
                            st.info("⏳ Matching FOGZ entries...")
                            fogz_matches = match_cluster_table_entries(
                                prep_df=fogz_prep,
                                valid_list_pass1=FOGZ_PASS1,
                                valid_list_pass2=FOGZ_PASS2,
                                max_output_spread=max_spread
                            )
                            cluster_results['fogz_combined'] = fogz_matches
                            st.success(f"✅ FOGZ matches: {len(fogz_matches)} found")
                        else:
                            st.warning("⚠️ No FOGZ prep entries found")
                            prep_tables['fogz_prep'] = pd.DataFrame()
                            cluster_results['fogz_combined'] = pd.DataFrame()
                        
                        fogz_time = time_module.time() - fogz_start
                        
                        # ========================================
                        # LARGE DISCOUNTS TABLE
                        # ========================================
                        st.info("⏳ Generating Large Discounts prep table (two-pass)...")
                        ld_start = time_module.time()
                        
                        ld_prep, ld_summary = process_cluster_tables_two_pass(
                            measurement_df=measurement_df,
                            small_df=small_hlc_df,
                            big_df=big_hlc_df,
                            report_time=report_time,
                            scope_days=lookback_days,
                            valid_list_pass1=LD_PASS1,
                            valid_list_pass2=LD_PASS2,
                            max_output_spread=max_spread,
                            window_radius=window_radius
                        )
                        
                        # Store summary for export
                        prep_tables['ld_summary'] = ld_summary
                        
                        # ALWAYS display processing summary (even if no results)
                        with st.expander("📊 Large Discounts Processing Summary", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Pass 1 Results", ld_summary['pass1_count'])
                                st.caption(f"HLCs Examined: {ld_summary.get('pass1_hlcs_examined', 0)}")
                                if ld_summary.get('pass1_dates'):
                                    st.caption(f"Dates: {', '.join(map(str, ld_summary['pass1_dates'][:3]))}")
                                if ld_summary['pass1_m_numbers']:
                                    st.caption(f"M #s found: {', '.join(map(str, ld_summary['pass1_m_numbers'][:10]))}")
                            with col2:
                                st.metric("Pass 2 Results", ld_summary['pass2_count'])
                                st.caption(f"HLCs Examined: {ld_summary.get('pass2_hlcs_examined', 0)}")
                                if ld_summary.get('pass2_dates'):
                                    date_list = sorted(list(ld_summary['pass2_dates']))
                                    st.caption(f"Date range: {date_list[0] if date_list else 'N/A'} to {date_list[-1] if date_list else 'N/A'}")
                                if ld_summary['pass2_m_numbers']:
                                    st.caption(f"M #s found: {', '.join(map(str, ld_summary['pass2_m_numbers'][:10]))}")
                            with col3:
                                st.metric("Total Results", ld_summary['total_count'])
                                st.caption(f"Feeds: {', '.join(ld_summary.get('pass1_feeds', []))}")
                            
                            # Show detailed HLC processing
                            if ld_summary.get('pass1_hlcs_examined', 0) > 0 or ld_summary.get('pass2_hlcs_examined', 0) > 0:
                                st.markdown("---")
                                st.markdown("**🔍 Detailed HLC Examination:**")
                                
                                tab1, tab2 = st.tabs(["Pass 1 HLCs", "Pass 2 HLCs"])
                                
                                with tab1:
                                    if not ld_summary.get('pass1_processing_details', pd.DataFrame()).empty:
                                        st.dataframe(ld_summary['pass1_processing_details'], use_container_width=True, height=200)
                                        st.caption(f"Examined {len(ld_summary['pass1_processing_details'])} HLC entries in Pass 1")
                                    else:
                                        st.info("No HLCs examined in Pass 1")
                                
                                with tab2:
                                    if not ld_summary.get('pass2_processing_details', pd.DataFrame()).empty:
                                        st.dataframe(ld_summary['pass2_processing_details'], use_container_width=True, height=200)
                                        st.caption(f"Examined {len(ld_summary['pass2_processing_details'])} HLC entries in Pass 2")
                                    else:
                                        st.info("No HLCs examined in Pass 2")
                        
                        if not ld_prep.empty:
                            st.success(f"✅ Large Discounts prep: {len(ld_prep)} entries")
                            prep_tables['ld_prep'] = ld_prep
                            
                            # Now match pass1 with pass2
                            st.info("⏳ Matching Large Discounts entries...")
                            ld_matches = match_cluster_table_entries(
                                prep_df=ld_prep,
                                valid_list_pass1=LD_PASS1,
                                valid_list_pass2=LD_PASS2,
                                max_output_spread=max_spread
                            )
                            cluster_results['ld_combined'] = ld_matches
                            st.success(f"✅ Large Discounts matches: {len(ld_matches)} found")
                        else:
                            st.warning("⚠️ No Large Discounts prep entries found")
                            prep_tables['ld_prep'] = ld_prep
                            cluster_results['ld_combined'] = pd.DataFrame()
                        
                        ld_time = time_module.time() - ld_start
                        
                        # ========================================
                        # RECIPS PD TABLE
                        # ========================================
                        st.info("⏳ Generating Recips PD prep table (two-pass)...")
                        recips_start = time_module.time()
                        
                        recips_prep, recips_summary = process_cluster_tables_two_pass(
                            measurement_df=measurement_df,
                            small_df=small_hlc_df,
                            big_df=big_hlc_df,
                            report_time=report_time,
                            scope_days=lookback_days,
                            valid_list_pass1=RECIPS_PASS1,
                            valid_list_pass2=RECIPS_PASS2,
                            max_output_spread=max_spread,
                            window_radius=window_radius
                        )
                        
                        # Store summary for export
                        prep_tables['recips_summary'] = recips_summary
                        
                        # ALWAYS display processing summary (even if no results)
                        with st.expander("📊 Recips PD Processing Summary", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Pass 1 Results", recips_summary['pass1_count'])
                                st.caption(f"HLCs Examined: {recips_summary.get('pass1_hlcs_examined', 0)}")
                                if recips_summary.get('pass1_dates'):
                                    st.caption(f"Dates: {', '.join(map(str, recips_summary['pass1_dates'][:3]))}")
                                if recips_summary['pass1_m_numbers']:
                                    st.caption(f"M #s found: {', '.join(map(str, recips_summary['pass1_m_numbers'][:10]))}")
                            with col2:
                                st.metric("Pass 2 Results", recips_summary['pass2_count'])
                                st.caption(f"HLCs Examined: {recips_summary.get('pass2_hlcs_examined', 0)}")
                                if recips_summary.get('pass2_dates'):
                                    date_list = sorted(list(recips_summary['pass2_dates']))
                                    st.caption(f"Date range: {date_list[0] if date_list else 'N/A'} to {date_list[-1] if date_list else 'N/A'}")
                                if recips_summary['pass2_m_numbers']:
                                    st.caption(f"M #s found: {', '.join(map(str, recips_summary['pass2_m_numbers'][:10]))}")
                            with col3:
                                st.metric("Total Results", recips_summary['total_count'])
                                st.caption(f"Feeds: {', '.join(recips_summary.get('pass1_feeds', []))}")
                            
                            # Show detailed HLC processing
                            if recips_summary.get('pass1_hlcs_examined', 0) > 0 or recips_summary.get('pass2_hlcs_examined', 0) > 0:
                                st.markdown("---")
                                st.markdown("**🔍 Detailed HLC Examination:**")
                                
                                tab1, tab2 = st.tabs(["Pass 1 HLCs", "Pass 2 HLCs"])
                                
                                with tab1:
                                    if not recips_summary.get('pass1_processing_details', pd.DataFrame()).empty:
                                        st.dataframe(recips_summary['pass1_processing_details'], use_container_width=True, height=200)
                                        st.caption(f"Examined {len(recips_summary['pass1_processing_details'])} HLC entries in Pass 1")
                                    else:
                                        st.info("No HLCs examined in Pass 1")
                                
                                with tab2:
                                    if not recips_summary.get('pass2_processing_details', pd.DataFrame()).empty:
                                        st.dataframe(recips_summary['pass2_processing_details'], use_container_width=True, height=200)
                                        st.caption(f"Examined {len(recips_summary['pass2_processing_details'])} HLC entries in Pass 2")
                                    else:
                                        st.info("No HLCs examined in Pass 2")
                        
                        if not recips_prep.empty:
                            st.success(f"✅ Recips PD prep: {len(recips_prep)} entries")
                            prep_tables['recips_prep'] = recips_prep
                            
                            # Now match pass1 with pass2
                            st.info("⏳ Matching Recips PD entries...")
                            recips_matches = match_cluster_table_entries(
                                prep_df=recips_prep,
                                valid_list_pass1=RECIPS_PASS1,
                                valid_list_pass2=RECIPS_PASS2,
                                max_output_spread=max_spread
                            )
                            cluster_results['recips_combined'] = recips_matches
                            st.success(f"✅ Recips PD matches: {len(recips_matches)} found")
                        else:
                            st.warning("⚠️ No Recips PD prep entries found")
                            prep_tables['recips_prep'] = recips_prep
                            cluster_results['recips_combined'] = pd.DataFrame()
                        
                        recips_time = time_module.time() - recips_start
                        
                        # ========================================
                        # FINALIZE
                        # ========================================
                        total_time = time_module.time() - total_start
                        
                        cluster_results['timings'] = {
                            'fogz': fogz_time,
                            'large_discounts': ld_time,
                            'recips_pd': recips_time,
                            'total': total_time
                        }
                        
                        # Store in session state
                        st.session_state['cluster_tables'] = cluster_results
                        st.session_state['prep_tables'] = prep_tables
                        
                        st.success(f"✅ All cluster tables generated in {total_time:.2f}s!")
                        st.info(f"⏱️ **Timing Breakdown:** FOGZ: {fogz_time:.2f}s | Large Discounts: {ld_time:.2f}s | Recips PD: {recips_time:.2f}s")
                
                # Display prep tables first
                if 'prep_tables' in st.session_state:
                    prep_tables = st.session_state['prep_tables']
                    
                    st.markdown("---")
                    st.markdown("### 📋 Preparation Tables (Two-Pass Results)")
                    
                    # FOGZ Prep
                    with st.expander("🔍 FOGZ Prep Table", expanded=False):
                        if 'fogz_prep' in prep_tables and not prep_tables['fogz_prep'].empty:
                            # Summary info
                            st.markdown(f"**Pass 1 M#s (Recent 2 days):** {list(sorted(set(prep_tables['fogz_prep'][prep_tables['fogz_prep']['Pass'] == 'Pass1']['M #'].unique())))}")
                            st.markdown(f"**Pass 2 M#s (Within lookback):** {list(sorted(set(prep_tables['fogz_prep'][prep_tables['fogz_prep']['Pass'] == 'Pass2']['M #'].unique())))}")
                            
                            # Export buttons
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                csv_all = prep_tables['fogz_prep'].to_csv(index=False)
                                st.download_button("📥 Export All", csv_all, "fogz_prep_all.csv", "text/csv", key="fogz_all")
                            with col2:
                                pass1_df = prep_tables['fogz_prep'][prep_tables['fogz_prep']['Pass'] == 'Pass1']
                                if not pass1_df.empty:
                                    csv_pass1 = pass1_df.to_csv(index=False)
                                    st.download_button("📥 Export Pass 1", csv_pass1, "fogz_prep_pass1.csv", "text/csv", key="fogz_p1")
                            with col3:
                                pass2_df = prep_tables['fogz_prep'][prep_tables['fogz_prep']['Pass'] == 'Pass2']
                                if not pass2_df.empty:
                                    csv_pass2 = pass2_df.to_csv(index=False)
                                    st.download_button("📥 Export Pass 2", csv_pass2, "fogz_prep_pass2.csv", "text/csv", key="fogz_p2")
                            
                            # Visual comparison toggle
                            show_comparison = st.checkbox("📊 Show Pass 1 vs Pass 2 Comparison", key="fogz_comp")
                            if show_comparison:
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.markdown("**Pass 1 (Recent 2 days)**")
                                    st.dataframe(pass1_df, use_container_width=True, height=300)
                                    st.caption(f"📊 {len(pass1_df)} entries")
                                with col_b:
                                    st.markdown("**Pass 2 (All scope)**")
                                    st.dataframe(pass2_df, use_container_width=True, height=300)
                                    st.caption(f"📊 {len(pass2_df)} entries")
                            else:
                                st.dataframe(prep_tables['fogz_prep'], use_container_width=True, height=300)
                                st.caption(f"📊 Total entries: {len(prep_tables['fogz_prep'])}")
                        else:
                            st.info("No FOGZ prep entries")
                    
                    # Large Discounts Prep
                    with st.expander("🔍 Large Discounts Prep Table", expanded=False):
                        if 'ld_prep' in prep_tables and not prep_tables['ld_prep'].empty:
                            # Summary info
                            st.markdown(f"**Pass 1 M#s (Recent 2 days):** {list(sorted(set(prep_tables['ld_prep'][prep_tables['ld_prep']['Pass'] == 'Pass1']['M #'].unique())))}")
                            st.markdown(f"**Pass 2 M#s (Within lookback):** {list(sorted(set(prep_tables['ld_prep'][prep_tables['ld_prep']['Pass'] == 'Pass2']['M #'].unique())))}")
                            
                            # Export buttons
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                csv_all = prep_tables['ld_prep'].to_csv(index=False)
                                st.download_button("📥 Export All", csv_all, "ld_prep_all.csv", "text/csv", key="ld_all")
                            with col2:
                                pass1_df = prep_tables['ld_prep'][prep_tables['ld_prep']['Pass'] == 'Pass1']
                                if not pass1_df.empty:
                                    csv_pass1 = pass1_df.to_csv(index=False)
                                    st.download_button("📥 Export Pass 1", csv_pass1, "ld_prep_pass1.csv", "text/csv", key="ld_p1")
                            with col3:
                                pass2_df = prep_tables['ld_prep'][prep_tables['ld_prep']['Pass'] == 'Pass2']
                                if not pass2_df.empty:
                                    csv_pass2 = pass2_df.to_csv(index=False)
                                    st.download_button("📥 Export Pass 2", csv_pass2, "ld_prep_pass2.csv", "text/csv", key="ld_p2")
                            
                            # Visual comparison toggle
                            show_comparison = st.checkbox("📊 Show Pass 1 vs Pass 2 Comparison", key="ld_comp")
                            if show_comparison:
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.markdown("**Pass 1 (Recent 2 days)**")
                                    st.dataframe(pass1_df, use_container_width=True, height=300)
                                    st.caption(f"📊 {len(pass1_df)} entries")
                                with col_b:
                                    st.markdown("**Pass 2 (All scope)**")
                                    st.dataframe(pass2_df, use_container_width=True, height=300)
                                    st.caption(f"📊 {len(pass2_df)} entries")
                            else:
                                st.dataframe(prep_tables['ld_prep'], use_container_width=True, height=300)
                                st.caption(f"📊 Total entries: {len(prep_tables['ld_prep'])}")
                        else:
                            st.info("No Large Discounts prep entries")
                    
                    # Recips PD Prep
                    with st.expander("🔍 Recips PD Prep Table", expanded=False):
                        if 'recips_prep' in prep_tables and not prep_tables['recips_prep'].empty:
                            # Summary info
                            st.markdown(f"**Pass 1 M#s (Recent 2 days):** {list(sorted(set(prep_tables['recips_prep'][prep_tables['recips_prep']['Pass'] == 'Pass1']['M #'].unique())))}")
                            st.markdown(f"**Pass 2 M#s (Within lookback):** {list(sorted(set(prep_tables['recips_prep'][prep_tables['recips_prep']['Pass'] == 'Pass2']['M #'].unique())))}")
                            
                            # Export buttons
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                csv_all = prep_tables['recips_prep'].to_csv(index=False)
                                st.download_button("📥 Export All", csv_all, "recips_prep_all.csv", "text/csv", key="recips_all")
                            with col2:
                                pass1_df = prep_tables['recips_prep'][prep_tables['recips_prep']['Pass'] == 'Pass1']
                                if not pass1_df.empty:
                                    csv_pass1 = pass1_df.to_csv(index=False)
                                    st.download_button("📥 Export Pass 1", csv_pass1, "recips_prep_pass1.csv", "text/csv", key="recips_p1")
                            with col3:
                                pass2_df = prep_tables['recips_prep'][prep_tables['recips_prep']['Pass'] == 'Pass2']
                                if not pass2_df.empty:
                                    csv_pass2 = pass2_df.to_csv(index=False)
                                    st.download_button("📥 Export Pass 2", csv_pass2, "recips_prep_pass2.csv", "text/csv", key="recips_p2")
                            
                            # Visual comparison toggle
                            show_comparison = st.checkbox("📊 Show Pass 1 vs Pass 2 Comparison", key="recips_comp")
                            if show_comparison:
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.markdown("**Pass 1 (Recent 2 days)**")
                                    st.dataframe(pass1_df, use_container_width=True, height=300)
                                    st.caption(f"📊 {len(pass1_df)} entries")
                                with col_b:
                                    st.markdown("**Pass 2 (All scope)**")
                                    st.dataframe(pass2_df, use_container_width=True, height=300)
                                    st.caption(f"📊 {len(pass2_df)} entries")
                            else:
                                st.dataframe(prep_tables['recips_prep'], use_container_width=True, height=300)
                                st.caption(f"📊 Total entries: {len(prep_tables['recips_prep'])}")
                        else:
                            st.info("No Recips PD prep entries")
                
                # Display cluster tables if generated
                if 'cluster_tables' in st.session_state:
                    cluster_results = st.session_state['cluster_tables']
                    
                    st.markdown("---")
                    st.markdown("### 📊 Matched Results (Final Cluster Tables)")
                    
                    # Helper function for highlighting
                    def highlight_cluster_rows(row):
                        if row['Type'] == 'Today':
                            return ['background-color: #fff9c4'] * len(row)  # Yellow
                        elif row['Type'] == 'Recent':
                            return ['background-color: #bbdefb'] * len(row)  # Blue
                        else:
                            return [''] * len(row)
                    
                    # FOGZ Table
                    with st.expander("📋 FOGZ Matched Results", expanded=True):
                        if 'fogz_combined' in cluster_results and len(cluster_results['fogz_combined']) > 0:
                            st.markdown("""
                            **Legend:** 
                            🟡 **Yellow** = Day [0] (Today)  |  
                            🔵 **Blue** = Day [-1] or [-2] (Recent)  |  
                            ⚪ **White** = Older
                            
                            **FOGZ M#s:** {0, ±1, ±2, ±3, ±5, ±6} matched with **PwX2_1_0:** {±40, ±41, ±43, ±50, ±55, ±60, ±68, ±77, ±87, ±96, ±103, ±107, ±111}
                            """)
                            
                            styled_fogz = cluster_results['fogz_combined'].style.apply(highlight_cluster_rows, axis=1)
                            st.dataframe(styled_fogz, use_container_width=True, height=400)
                            timing_text = f" | ⏱️ Generated in {cluster_results.get('timings', {}).get('fogz', 0):.2f}s" if 'timings' in cluster_results else ""
                            st.caption(f"📊 Total FOGZ Matches: {len(cluster_results['fogz_combined'])}{timing_text}")
                        else:
                            st.info("No FOGZ matches found within current spread settings.")
                    
                    # Large Discounts Table
                    with st.expander("📋 Large Discounts Matched Results", expanded=False):
                        if 'ld_combined' in cluster_results and len(cluster_results['ld_combined']) > 0:
                            st.markdown("""
                            **Legend:** 
                            🟡 **Yellow** = Day [0] (Today)  |  
                            🔵 **Blue** = Day [-1] or [-2] (Recent)  |  
                            ⚪ **White** = Older
                            
                            **Large Discount M#s:** {±36, ±38, ±39} matched with **PX2_1_0:** {±41, ±43, ±50, ±55, ±60, ±68, ±77, ±87, ±96, ±103, ±107, ±111}
                            """)
                            
                            styled_ld = cluster_results['ld_combined'].style.apply(highlight_cluster_rows, axis=1)
                            st.dataframe(styled_ld, use_container_width=True, height=400)
                            timing_text = f" | ⏱️ Generated in {cluster_results.get('timings', {}).get('large_discounts', 0):.2f}s" if 'timings' in cluster_results else ""
                            st.caption(f"📊 Total Large Discount Matches: {len(cluster_results['ld_combined'])}{timing_text}")
                        else:
                            st.info("No Large Discount matches found within current spread settings.")
                    
                    # Recips PD Table
                    with st.expander("📋 Recips PD Matched Results", expanded=False):
                        if 'recips_combined' in cluster_results and len(cluster_results['recips_combined']) > 0:
                            st.markdown("""
                            **Legend:** 
                            🟡 **Yellow** = Day [0] (Today)  |  
                            🔵 **Blue** = Day [-1] or [-2] (Recent)  |  
                            ⚪ **White** = Older
                            
                            **DRecip List (abs < 40):** {±1, ±2, ±3, ±5, ±6, ±10, ±14, ±15, ±22, ±27, ±30, ±36, ±38, ±39}
                            
                            **Matched with R# mates:**
                            - X0: (30,50), (22,60), (14,68), (10,77), (6,87), (5,96), (3,103), (2,107), (1,111)
                            - XD0: (27,54), (15,67)
                            - X1: (36,43), (26,55)
                            - XD1: (33,45)
                            - X2: (39,41)
                            - XD2: (38,42)
                            """)
                            
                            styled_recips = cluster_results['recips_combined'].style.apply(highlight_cluster_rows, axis=1)
                            st.dataframe(styled_recips, use_container_width=True, height=400)
                            timing_text = f" | ⏱️ Generated in {cluster_results.get('timings', {}).get('recips_pd', 0):.2f}s" if 'timings' in cluster_results else ""
                            st.caption(f"📊 Total Recip PD Matches: {len(cluster_results['recips_combined'])}{timing_text}")
                        else:
                            st.info("No Recip PD matches found within current spread settings.")
                
                st.markdown("---")
                



            except Exception as e:
                st.error(f"Error in Tab 8 analysis: {str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
