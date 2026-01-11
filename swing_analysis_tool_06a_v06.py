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

# Configure Pandas to handle large dataframes
pd.set_option("styler.render.max_elements", 1000000)  # Allow up to 1M cells for styling

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
from custom_range_calculator_1125_21c import (
    process_cluster_tables_two_pass,
    match_cluster_table_entries,
    clean_timestamp
)

# Import model definitions for all 23 trading models
from model_definitions_v21 import MODELS, get_reciprocal_lookup, apply_special_matching

# Import model processor for batch processing all 23 models
from model_processor_v21 import (
    process_all_models,
    get_model_display_info,
    organize_results_by_category,
    create_summary_stats
)

# Import Excel exporter v23 v6 with enhanced formatting, Today/Recent identification, and export timer
from excel_exporter_v23_v6 import create_download_button

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Swing Analysis Tool 06a v06",
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
    st.title(" Market Swing Analysis Tool 06a_06 Two Pass")
    st.markdown("**Integrated Version** - Swing Detection, MA Analysis, NY Session, Traveler/Pivot Calculations")
    st.markdown("---")
    
    # Sidebar settings
    st.sidebar.header("️ Settings")
    
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
    st.header(" Upload OHLC Files")
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
    st.markdown("###  Tab 8 Data Source Selection")
    
    # Mode selection for Tab 8
    tab8_mode = st.radio(
        "Choose Tab 8 data source:",
        ["Traditional Mode (3 Feed Files)", "Bypass Mode (Traveler Report)"],
        key="tab8_mode",
        horizontal=True
    )
    
    # Initialize variables
    small_feed_file = None
    big_feed_file = None
    measurement_file = None
    traveler_report_file = None
    selected_tab_name = None
    
    if tab8_mode == "Traditional Mode (3 Feed Files)":
        st.markdown("###  Feed Data (For Strategic Zones Tab 8)")
        st.info(" Upload RAW HLC feed data - the app will generate custom reciprocal traveler reports")
        
        with st.expander("INFO: About Feed Data Format"):
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
                st.success(f"[OK] {small_feed_file.name}")
        
        with col2:
            big_feed_file = st.file_uploader(
                "Big Feed HLC CSV",
                type=['csv'],
                key="big_feed",
                help="Raw HLC data with columns like 'Jupiter H', 'Jupiter L', 'Jupiter C'"
            )
            if big_feed_file:
                st.success(f"[OK] {big_feed_file.name}")
        
        with col3:
            measurement_file = st.file_uploader(
                "Measurement File",
                type=['xlsx', 'xls'],
                key="measurement",
                help="Excel file with M# and R# relationships"
            )
            if measurement_file:
                st.success(f"[OK] {measurement_file.name}")
    
    else:  # Bypass Mode
        st.markdown("###  Traveler Report Upload (Bypass Mode)")
        st.info(" Upload pre-generated traveler report - skips the need for 3 feed files")
        
        with st.expander("INFO: About Bypass Mode"):
            st.markdown("""
            **Bypass Mode uses pre-generated Traveler Reports from your main app.**
            
            Benefits:
            - No need to upload Small Feed, Big Feed, or Measurement files
            - Faster setup - just upload the traveler report
            - Works with multi-tab Excel files
            
            Requirements:
            - Traveler report Excel file (can have multiple tabs)
            - At least one OHLC file (3m, 5m, 6m, or 15m)
            
            The app will:
            1. Let you select which tab to analyze (if multi-tab file)
            2. Use the traveler data directly for Trading Model Analysis
            3. Skip the traveler generation step
            """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            traveler_report_file = st.file_uploader(
                "Traveler Report (Excel)",
                type=['xlsx', 'xls'],
                key="traveler_report",
                help="Pre-generated traveler report from main app"
            )
            
            if traveler_report_file:
                st.success(f"[OK] {traveler_report_file.name}")
                
                # Load Excel file and get sheet names
                try:
                    excel_file = pd.ExcelFile(traveler_report_file)
                    sheet_names = excel_file.sheet_names
                    
                    if len(sheet_names) > 1:
                        st.info(f"📊 Found {len(sheet_names)} tabs in file")
                    else:
                        st.info(f"📊 Single-tab file detected")
                except Exception as e:
                    st.error(f"Error reading Excel file: {str(e)}")
                    sheet_names = []
        
        with col2:
            if traveler_report_file:
                try:
                    excel_file = pd.ExcelFile(traveler_report_file)
                    sheet_names = excel_file.sheet_names
                    
                    if len(sheet_names) > 1:
                        st.markdown("**Select Tab:**")
                        selected_tab_name = st.selectbox(
                            "Choose which tab to analyze",
                            sheet_names,
                            key="selected_tab",
                            label_visibility="collapsed"
                        )
                        st.caption(f"Selected: {selected_tab_name}")
                    else:
                        selected_tab_name = sheet_names[0]
                        st.info(f"Using: {selected_tab_name}")
                except:
                    pass
    
    # Load files
    files = {
        '3m': load_ohlc_file(file_3m, '3m'),
        '5m': load_ohlc_file(file_5m, '5m'),
        '6m': load_ohlc_file(file_6m, '6m'),
        '15m': load_ohlc_file(file_15m, '15m')
    }
    
    loaded_files = {k: v for k, v in files.items() if v is not None}
    
    if not loaded_files:
        st.info(" Please upload at least one OHLC file to begin analysis.")
        
        # Show expected format
        st.subheader(" Expected File Format")
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
    st.success(f"[OK] Loaded {len(loaded_files)} file(s): {', '.join(loaded_files.keys())}")
    
    with st.expander(" File Details"):
        for tf, df in loaded_files.items():
            ma_cols = get_all_ma_columns(df)
            st.markdown(f"**{tf}**: {len(df)} rows, {len(df.columns)} columns, {len(ma_cols)} MA columns")
            st.markdown(f"  Time range: {df['time'].min()} to {df['time'].max()}")
    
    st.markdown("---")
    
    # Check minimum files for Strategic Zones - depends on mode
    if tab8_mode == "Traditional Mode (3 Feed Files)":
        min_files_met = all([
            small_feed_file is not None,
            big_feed_file is not None,
            measurement_file is not None,
            len(loaded_files) >= 1  # At least one OHLC file
        ])
        
        if min_files_met:
            st.success(f"[OK] **Tab 8 - Strategic Zones READY!** All required files loaded ({len(loaded_files)} OHLC + 3 feed files)")
            st.info(" Click the ' Strategic Zones' tab below to generate custom reciprocal traveler reports")
        elif len(loaded_files) > 0:
            missing = []
            if not small_feed_file:
                missing.append("Small Feed HLC CSV")
            if not big_feed_file:
                missing.append("Big Feed HLC CSV")
            if not measurement_file:
                missing.append("Measurement File")
            if missing:
                st.warning(f"WARNING: **Tab 8 needs:** {', '.join(missing)}")
        else:
            st.info("INFO: Upload OHLC and Feed files above to enable Tab 8 - Strategic Zones")
    else:  # Bypass Mode
        min_files_met = all([
            traveler_report_file is not None,
            selected_tab_name is not None,
            len(loaded_files) >= 1  # At least one OHLC file
        ])
        
        if min_files_met:
            st.success(f"[OK] **Tab 8 - Strategic Zones READY! (Bypass Mode)** Files loaded: {len(loaded_files)} OHLC + Traveler Report")
            st.info(f" Using traveler report tab: '{selected_tab_name}'")
        elif len(loaded_files) > 0:
            if not traveler_report_file:
                st.warning("WARNING: **Tab 8 needs:** Traveler Report (Excel)")
            else:
                st.warning("WARNING: Select a tab from the traveler report")
        else:
            st.info("INFO: Upload OHLC file and Traveler Report above to enable Tab 8 - Strategic Zones")
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        " Swing Detection",
        " NY Session",
        " MA Wick Report",
        " MA Confluence",
        " MA History",
        " HOD/LOD Analysis",
        " Traveler Calculator",
        " Strategic Zones"  # NEW TAB
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
        
        if st.button("DEBUG: Analyze Swings", key="analyze_swings"):
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
                        " Download Swing Report",
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
        st.header(" NY Session Analysis (8 AM - 4 PM)")
        
        ny_tf = st.selectbox(
            "Select timeframe for NY session",
            list(loaded_files.keys()),
            key="ny_tf"
        )
        
        if st.button("DEBUG: Analyze NY Session", key="analyze_ny"):
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
                        " Download NY Session Report",
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
        
        if ma_timeframe and st.button(" Generate MA Wick Report", key="gen_ma_report"):
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
                            " Download MA Wick Report",
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
            st.info(" Please run Swing Detection first (Tab 1)")
        else:
            conf_timeframe = st.selectbox(
                "Select timeframe for confluence analysis",
                list(loaded_files.keys()),
                key="conf_tf_select"
            )
            
            if st.button(" Find MA Confluence", key="find_confluence"):
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
                            " Download Confluence Report",
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
            
            if selected_ma and st.button(" Track MA History", key="track_ma"):
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
                            " Download MA History",
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
        st.header(" HOD/LOD Analysis")
        st.markdown("Analyze High of Day (HOD) and Low of Day (LOD) for each trading day")
        
        hod_tf = st.selectbox(
            "Select timeframe for HOD/LOD analysis",
            list(loaded_files.keys()),
            key="hod_tf_select"
        )
        
        if st.button("DEBUG: Detect HOD/LOD", key="detect_hod_lod"):
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
                        " Download HOD/LOD Report",
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
        st.header(" Traveler (Pivot) Calculator")
        st.markdown("""
        Calculate travelers that produce outputs at specific price targets.
        Uses the pivot formula: **Output = (H + L + C) / 3 + M * (H - L)**
        """)
        
        st.info("INFO: **Note:** This tab requires DIFFERENT files than the main upload section above.")
        
        # File uploads for HLC and Measurement data
        st.markdown("### Data Sources (Upload Here)")
        
        st.markdown("""
        **Required format:**
        - **HLC Data:** CSV/Excel with columns like "Spain H", "Spain L", "Spain C", "Jupiter H", etc.
        - **Measurement:** Excel file with M# and R# lookup table
        
        **WARNING:** These are DIFFERENT from the Feed CSV files used in Tab 8.
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
                        "Tolerance (+/-)",
                        value=24.0,
                        min_value=1.0,
                        max_value=100.0,
                        step=1.0,
                        help="Acceptable distance from target"
                    )
                
                if st.button("DEBUG: Find Travelers", key="find_travelers"):
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
                                " Download Traveler Report",
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
                    
                    if st.button(" Find Travelers at HOD/LOD", key="find_hod_lod_travelers"):
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
                    st.info(" Run HOD/LOD Detection first (Tab 6) to analyze travelers at HOD/LOD")
            
            except Exception as e:
                st.error(f"Error loading files: {str(e)}")
        else:
            st.info(" Upload HLC data and Measurement file to use the Traveler Calculator")
    
    # ========================
    # TAB 8: STRATEGIC ZONES
    # ========================
    with tab8:
        # Initialize session state for Tab 8 processing
        if 'tab8_processed' not in st.session_state:
            st.session_state.tab8_processed = False
        if 'tab8_bypass_data' not in st.session_state:
            st.session_state.tab8_bypass_data = None
        
        st.header(" Strategic Zone Recommendations")
        st.markdown("""
        **High-Probability Turning Zones** identified through:
        - High-rank Recip pairs (Epic + Anchor combinations)
        - HUGE HMA confluences (h1-h20 + high-timeframe MAs)
        - Wildcard M# emergence (0, +/-40, +/-54)
        - MA role transitions (resistance to support and vice versa)
        """)
        
        # Show current mode
        mode_badge = "🔄 Traditional Mode" if tab8_mode == "Traditional Mode (3 Feed Files)" else "⚡ Bypass Mode"
        st.info(f"{mode_badge} - Tab 8 Data Source")
        
        # DEBUG: Show file status
        with st.expander("DEBUG: Debug: File Upload Status"):
            st.markdown(f"**Mode:** {tab8_mode}")
            if tab8_mode == "Traditional Mode (3 Feed Files)":
                st.markdown("**Traditional Mode Files:**")
                st.markdown(f"- Small Feed: {'[OK] ' + small_feed_file.name if small_feed_file else '[X] Not uploaded'}")
                st.markdown(f"- Big Feed: {'[OK] ' + big_feed_file.name if big_feed_file else '[X] Not uploaded'}")
                st.markdown(f"- Measurement: {'[OK] ' + measurement_file.name if measurement_file else '[X] Not uploaded'}")
            else:
                st.markdown("**Bypass Mode Files:**")
                st.markdown(f"- Traveler Report: {'[OK] ' + traveler_report_file.name if traveler_report_file else '[X] Not uploaded'}")
                st.markdown(f"- Selected Tab: {selected_tab_name if selected_tab_name else '[X] Not selected'}")
            st.markdown(f"- OHLC Files: {len(loaded_files)} files loaded")
            st.markdown(f"\n**min_files_met:** {min_files_met}")
        
        # Check minimum files
        if not min_files_met:
            if tab8_mode == "Traditional Mode (3 Feed Files)":
                st.warning("WARNING: Strategic Zones (Traditional Mode) requires minimum 4 files:")
                st.markdown("""
                **Minimum Requirements:**
                - At least 1 OHLC file (3m, 5m, 6m, or 15m)
                - Small Feed 15m CSV (for travelers)
                - Big Feed 15m CSV (for travelers)
                - Measurement File (for M# values)
                """)
                
                files_status = {
                    'OHLC files': len(loaded_files),
                    'Small Feed': '[OK]' if small_feed_file else '[X]',
                    'Big Feed': '[OK]' if big_feed_file else '[X]',
                    'Measurement': '[OK]' if measurement_file else '[X]'
                }
            else:
                st.warning("WARNING: Strategic Zones (Bypass Mode) requires minimum 2 files:")
                st.markdown("""
                **Minimum Requirements:**
                - At least 1 OHLC file (3m, 5m, 6m, or 15m)
                - Traveler Report (Excel) with selected tab
                """)
                
                files_status = {
                    'OHLC files': len(loaded_files),
                    'Traveler Report': '[OK]' if traveler_report_file else '[X]',
                    'Tab Selected': '[OK]' if selected_tab_name else '[X]'
                }
            
            for file_type, status in files_status.items():
                st.markdown(f"- {file_type}: {status}")
            
            st.info(" Please upload all required files above to enable Strategic Zones analysis")
            
        elif tab8_mode == "Traditional Mode (3 Feed Files)" and not (small_feed_file and big_feed_file and measurement_file):
            # Double-check that all traveler files are uploaded
            st.error("WARNING: Missing traveler files!")
            st.markdown("Please upload:")
            if not small_feed_file:
                st.markdown("- [X] Small Feed 15m CSV")
            if not big_feed_file:
                st.markdown("- [X] Big Feed 15m CSV")
            if not measurement_file:
                st.markdown("- [X] Measurement File (Excel)")
            
        else:
            # All files present, proceed with analysis
            try:
                st.markdown("###  Loading Data...")
                
                # Load data based on mode
                if tab8_mode == "Traditional Mode (3 Feed Files)":
                    # Traditional mode: Load HLC feeds and generate travelers
                    small_hlc_df = pd.read_csv(small_feed_file)
                    big_hlc_df = pd.read_csv(big_feed_file)
                    measurement_df = pd.read_excel(measurement_file)
                    
                    st.success(f"Loaded: Small feed ({len(small_hlc_df)} rows), Big feed ({len(big_hlc_df)} rows), Measurements ({len(measurement_df)} rows)")
                    
                    # Show detected HLC columns
                    with st.expander("DEBUG: Detected HLC Columns"):
                        st.markdown("**Small Feed:**")
                        hlc_cols_small = [col for col in small_hlc_df.columns if col.endswith((' H', ' L', ' C'))]
                        origins_small = list(set([col[:-2] for col in hlc_cols_small if col.endswith(' H')]))
                        st.code(f"Origins: {', '.join(origins_small)}")
                        
                        st.markdown("**Big Feed:**")
                        hlc_cols_big = [col for col in big_hlc_df.columns if col.endswith((' H', ' L', ' C'))]
                        origins_big = list(set([col[:-2] for col in hlc_cols_big if col.endswith(' H')]))
                        st.code(f"Origins: {', '.join(origins_big)}")
                    
                    # Will generate travelers later in the process
                    bypass_mode_active = False
                    
                else:
                    # Bypass mode: Load pre-generated traveler report
                    excel_file = pd.ExcelFile(traveler_report_file)
                    small_hlc_df = pd.read_excel(excel_file, sheet_name=selected_tab_name)
                    
                    # In bypass mode, we don't have separate big feed or measurement file
                    # The traveler report should contain all necessary data
                    big_hlc_df = None
                    measurement_df = None
                    
                    st.success(f"Loaded: Traveler Report tab '{selected_tab_name}' ({len(small_hlc_df)} rows)")
                    
                    # Show detected columns
                    with st.expander("DEBUG: Detected Traveler Columns"):
                        st.markdown("**Columns in traveler report:**")
                        st.code(f"{', '.join(small_hlc_df.columns.tolist())}")
                    
                    bypass_mode_active = True
                
                st.markdown("---")
                st.markdown("### ️ Report Settings")
                
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
                    if not bypass_mode_active:
                        lookback_days = st.slider("Lookback Days", 1, 60, 20, 1, key="lookback_days")
                        st.markdown("*How many days to look back (default: 20)*")
                    else:
                        st.info("⚡ Bypass Mode")
                        st.caption("Using pre-generated traveler data")
                        lookback_days = None  # Not used in bypass mode
                
                with col3:
                    if not bypass_mode_active:
                        max_spread = st.slider("Max Output Spread", 0.5, 10.0, 3.0, 0.5, key="max_output_spread")
                        st.markdown("*Maximum output spread for cluster matches*")
                    else:
                        st.info("Report includes all matches")
                        st.caption("No spread filtering needed")
                        max_spread = None  # Not used in bypass mode
                
                with col4:
                    max_zones = st.slider("Max Zones to Show", 1, 6, 4, key="max_zones")
                    zone_tolerance = st.slider("Zone Tolerance (+/-)", 12, 48, 24, 6, key="zone_tol")
                
                st.markdown("---")
                
                # Processing Mode Selection - only in traditional mode
                if not bypass_mode_active:
                    st.markdown("### 🔧 Processing Mode")
                    processing_mode = st.radio(
                        "Select processing mode for cluster table generation:",
                        ["Full Range", "Custom Ranges", "HOD/LOD Mode"],
                        index=0,
                        key="cluster_processing_mode",
                        horizontal=True
                    )
                else:
                    # In bypass mode, we're just using the traveler report as-is
                    processing_mode = "Bypass"
                    st.markdown("### 🔧 Bypass Mode Active")
                    st.info("Using traveler report data directly - no cluster table generation needed")
                
                # Mode-specific settings (only for traditional mode)
                if not bypass_mode_active:
                    # Common settings for all modes - 3 column layout
                    col_win, col_feed, col_match = st.columns([1, 1, 1])
                    
                    with col_win:
                        st.markdown("**Window Settings:**")
                        window_radius = st.slider("Window Radius", 50, 1000, 150, 50, key="window_radius")
                        st.caption("Points above/below each feed's Open")
                    
                    with col_feed:
                        st.markdown("**Feed Selection:**")
                        st.caption("*Which feed(s) to process:*")
                        feed_selection = st.radio(
                            "Feed Selection",
                            ["Both feeds", "Small feed only", "Big feed only"],
                            index=0,
                            key="feed_selection",
                            label_visibility="collapsed"
                        )
                    
                    with col_match:
                        st.markdown("**Match Type:**")
                        st.caption("*Cross-feed matching:*")
                        match_type_selection = st.radio(
                            "Match Type",
                            ["Same feed only", "Allow mixed feed"],
                            index=0,
                            key="match_type_selection",
                            label_visibility="collapsed"
                        )
                    
                    # Status line below all three columns
                    if feed_selection == "Both feeds":
                        if match_type_selection == "Same feed only":
                            st.info("✓ Processing both feeds, matching within each feed separately")
                        else:
                            st.info("✓ Processing both feeds, allowing matches across feeds")
                    elif feed_selection == "Small feed only":
                        st.info("✓ Processing Small feed only, matching within Small feed")
                    elif feed_selection == "Big feed only":
                        st.info("✓ Processing Big feed only, matching within Big feed")
                    
                    st.markdown("**Origin Filtering:**")
                    
                    # Collect all origins from both feeds
                    all_origins = set()
                    if small_hlc_df is not None:
                        small_origins = [col[:-2] for col in small_hlc_df.columns if col.endswith(' H')]
                        all_origins.update(small_origins)
                    if big_hlc_df is not None:
                        big_origins = [col[:-2] for col in big_hlc_df.columns if col.endswith(' H')]
                        all_origins.update(big_origins)
                
                # Categorize origins (case-insensitive matching) - only for traditional mode
                if not bypass_mode_active:
                    EPIC_NAMES = {"trinidad", "tobago", "wasp-12b", "macedonia"}
                    ANCHOR_NAMES = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}
                    
                    epic_origins = [o for o in all_origins if o.lower() in EPIC_NAMES]
                    anchor_origins = [o for o in all_origins if o.lower() in ANCHOR_NAMES]
                    other_origins = [o for o in all_origins if o.lower() not in EPIC_NAMES and o.lower() not in ANCHOR_NAMES]
                    
                    # Sort alphabetically
                    epic_origins.sort()
                    anchor_origins.sort()
                    other_origins.sort()
                    
                    st.info(f"Detected {len(all_origins)} origins: {len(epic_origins)} Epic, {len(anchor_origins)} Anchor, {len(other_origins)} Other")
                    
                    col_filter1, col_filter2 = st.columns([2, 1])
                    with col_filter1:
                        filter_origins = st.checkbox(
                            "Filter Origins (Faster processing, smaller results)", 
                            value=False, 
                            key="filter_origins"
                        )
                        st.caption("Process only selected origins. All priority origins (Epic + Anchor) selected by default.")
                    
                    allowed_origins = None
                    if filter_origins:
                        # Epic Origins
                        if epic_origins:
                            st.markdown("**Epic Origins (Priority):**")
                            epic_selections = {}
                            cols = st.columns(min(len(epic_origins), 4))
                            for idx, origin in enumerate(epic_origins):
                                with cols[idx % len(cols)]:
                                    epic_selections[origin] = st.checkbox(origin, value=True, key=f"epic_{origin}")
                        
                        # Anchor Origins
                        if anchor_origins:
                            st.markdown("**Anchor Origins (Priority):**")
                            anchor_selections = {}
                            cols = st.columns(min(len(anchor_origins), 5))
                            for idx, origin in enumerate(anchor_origins):
                                with cols[idx % len(cols)]:
                                    anchor_selections[origin] = st.checkbox(origin, value=True, key=f"anchor_{origin}")
                        
                        # Other Origins
                        if other_origins:
                            st.markdown("**Other Origins (Optional):**")
                            other_selections = {}
                            cols = st.columns(min(len(other_origins), 4))
                            for idx, origin in enumerate(other_origins):
                                with cols[idx % len(cols)]:
                                    other_selections[origin] = st.checkbox(origin, value=False, key=f"other_{origin}")
                        
                        # Build allowed origins set (use exact names from CSV, case-sensitive)
                        allowed_origins = set()
                        if epic_origins:
                            allowed_origins.update([o for o, selected in epic_selections.items() if selected])
                        if anchor_origins:
                            allowed_origins.update([o for o, selected in anchor_selections.items() if selected])
                        if other_origins:
                            allowed_origins.update([o for o, selected in other_selections.items() if selected])
                        
                        epic_count = sum(1 for o, s in (epic_selections.items() if epic_origins else []) if s)
                        anchor_count = sum(1 for o, s in (anchor_selections.items() if anchor_origins else []) if s)
                        other_count = sum(1 for o, s in (other_selections.items() if other_origins else []) if s)
                        
                        st.info(f"Processing {len(allowed_origins)} origins ({epic_count} Epic + {anchor_count} Anchor + {other_count} Other)")
                    else:
                        st.info("Processing ALL origins (may take longer with large results)")
                    
                    st.markdown("---")
                    
                    # Window Segmentation
                    st.markdown("**Window Segmentation (Performance):**")
                    segment_window = st.checkbox(
                        "Segment window into smaller ranges",
                        value=False,
                        key="segment_window"
                    )
                    st.caption("Break 300-unit window into smaller segments. Reduces memory & processing time.")
                    
                    segment_size = 75
                    num_segments = 1
                    combine_segments = True  # Default value
                    
                    if segment_window:
                        col_seg1, col_seg2 = st.columns(2)
                        with col_seg1:
                            segment_size = st.slider(
                                "Segment Size (units)",
                                25, 150, 75, 25,
                                key="segment_size"
                            )
                        with col_seg2:
                            total_range = window_radius * 2
                            num_segments = int(np.ceil(total_range / segment_size))
                            st.metric("Number of Segments", num_segments)
                            st.caption(f"Each segment: ~{segment_size} units")
                        
                        combine_segments = st.checkbox(
                            "Combine segments into single report",
                            value=True,
                            key="combine_segments"
                        )
                        if not combine_segments:
                            st.info(f"Will generate {num_segments} separate reports per table")
                        else:
                            st.info(f"Will process {num_segments} segments and combine results")
                    else:
                        # When segment_window is False, set segment_size to None
                        segment_size = None
                else:
                    # Bypass mode - set default values
                    allowed_origins = None
                    segment_size = None
                    segment_window = False
                    combine_segments = True
                
                st.markdown("---")
                
                # Mode-specific settings (only for traditional mode)
                if not bypass_mode_active:
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
                # ALL MODELS ANALYSIS (V21)
                # ========================================
                st.markdown("### 🎯 Trading Models Analysis")
                st.markdown(f"**All 23 Trading Models** - Comprehensive pattern matching ({processing_mode} mode)")
                
                if st.button("🚀 Process All Models", key="gen_all_models", type="primary"):
                    # Set session state to indicate processing has occurred
                    st.session_state.tab8_processed = True
                    
                    if not bypass_mode_active:
                        # TRADITIONAL MODE - Generate travelers from feeds
                        with st.spinner("Processing all 23 trading models..."):
                            # Process all models using streamlined processor
                            all_results = process_all_models(
                                measurement_df=measurement_df,
                                small_df=small_hlc_df,
                                big_df=big_hlc_df,
                                report_time=report_time,
                                lookback_days=lookback_days,
                                max_spread=max_spread,
                                window_radius=window_radius,
                                allowed_origins=allowed_origins,
                                segment_size=segment_size,
                                combine_segments=combine_segments,
                                feed_selection=feed_selection,
                                match_type_selection=match_type_selection
                            )
                            
                            # Store in session state
                            st.session_state.cluster_results = all_results
                            
                            # Display summary
                            summary_stats = create_summary_stats(all_results)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Models", summary_stats['total_models'])
                            with col2:
                                st.metric("Models with Matches", summary_stats['models_with_matches'])
                            with col3:
                                st.metric("Total Matches", summary_stats['total_matches'])
                            with col4:
                                st.metric("Processing Time", f"{all_results['total_time']:.1f}s")
                            
                            st.success(f"✅ Processed {summary_stats['total_models']} models in {all_results['total_time']:.1f}s!")
                            
                            # Excel Export with Report Time and highlighting (v23 unified format)
                            st.markdown("---")
                            st.markdown("### 📥 Export Results")
                            create_download_button(all_results, report_time, measurement_df)
                            
                            # Display results by category
                            st.markdown("---")
                            st.markdown("### 📊 Results by Model")
                            
                            categories = organize_results_by_category(all_results)
                            
                            # Helper function for highlighting Arrival_Brackets
                            def highlight_arrival_brackets(df):
                                """Highlight only the Arrival_Brackets column."""
                                def apply_bracket_color(row):
                                    colors = [''] * len(row)
                                    if 'Arrival_Brackets' in df.columns:
                                        bracket_idx = df.columns.get_loc('Arrival_Brackets')
                                        bracket_val = str(row.get('Arrival_Brackets', ''))
                                        
                                        if '[0]' in bracket_val:
                                            colors[bracket_idx] = 'background-color: #fff9c4'  # Yellow
                                        elif any(x in bracket_val for x in ['[-1]', '[-2]', '[-3]']):
                                            colors[bracket_idx] = 'background-color: #bbdefb'  # Blue
                                    return colors
                                return apply_bracket_color
                            
                            # Display each category
                            for category_name, model_list in categories.items():
                                if not model_list:
                                    continue
                                
                                st.markdown(f"#### {category_name}")
                                
                                for model_name in model_list:
                                    model_info = get_model_display_info(model_name)
                                    matched_df = all_results['results'].get(model_name, pd.DataFrame())
                                    
                                    # Determine if expanded by default (first 3 models only)
                                    is_expanded = model_info['number'] <= 3
                                    
                                    with st.expander(
                                        f"🔹 Model #{model_info['number']}: {model_info['display_name']} "
                                        f"({len(matched_df)} matches)",
                                        expanded=is_expanded
                                    ):
                                        st.markdown(f"**Description:** {model_info['description']}")
                                        
                                        if len(matched_df) > 0:
                                            st.markdown("""
                                            **Legend:** 
                                            🟡 **Yellow** = Day [0] (Today)  |  
                                            🔵 **Blue** = Days [-1], [-2], [-3] (Recent)
                                            """)
                                            
                                            # Apply styling
                                            styled_df = matched_df.style.apply(
                                                highlight_arrival_brackets(matched_df),
                                                axis=1
                                            ).format({
                                                'Arrival_Output': '{:.3f}',
                                                'Prox': '{:.3f}'
                                            })
                                            
                                            st.dataframe(styled_df, use_container_width=True, height=400)
                                            
                                            # Show timing
                                            timing = all_results['timings'].get(model_name, 0)
                                            st.caption(
                                                f"📊 {len(matched_df)} matches | "
                                                f"⏱️ Generated in {timing:.2f}s"
                                            )
                                        else:
                                            st.info("No matches found within current spread settings.")
                    
                    else:
                        # BYPASS MODE - Use uploaded traveler report
                        with st.spinner("Processing all 23 trading models using uploaded traveler report..."):
                            st.info("⚡ Bypass Mode: Using pre-generated traveler data")
                            
                            try:
                                # Keep 'Output' column as-is for bypass_mode_matcher compatibility
                                # (bypass_mode_matcher expects 'Output', not 'Arrival_Output')
                                
                                # Extract M#s from traveler report to create measurement_df
                                if 'M #' in small_hlc_df.columns:
                                    unique_m_numbers = sorted(small_hlc_df['M #'].unique())
                                    st.write(f"📊 Traveler report has {len(unique_m_numbers)} unique M #s")
                                    
                                    # Create minimal measurement_df with M# column
                                    bypass_measurement_df = pd.DataFrame({'M #': unique_m_numbers})
                                    
                                    # Show Pass 1 and Pass 2 M#s available
                                    pass1_ms = [m for m in unique_m_numbers if -6 <= m <= 6]
                                    pass2_ms = [m for m in unique_m_numbers if m not in pass1_ms]
                                    st.write(f"✓ Pass 1 M #s available: {pass1_ms[:20]}{'...' if len(pass1_ms) > 20 else ''}")
                                    st.write(f"✓ Pass 2 M #s available: {pass2_ms[:20]}{'...' if len(pass2_ms) > 20 else ''}")
                                else:
                                    st.error("❌ No 'M #' column found in traveler report!")
                                    st.info("The traveler report must have a 'M #' column for model processing.")
                                    return
                                
                                # Extract origins information
                                if 'Origin' in small_hlc_df.columns:
                                    unique_origins = sorted(small_hlc_df['Origin'].unique())
                                    st.write(f"📋 Traveler report has {len(unique_origins)} unique origins: {', '.join(unique_origins)}")
                                else:
                                    st.warning("⚠️ No 'Origin' column found in traveler report")
                                
                                # Separate by feed if present
                                if 'Feed' in small_hlc_df.columns:
                                    small_feed_df = small_hlc_df[small_hlc_df['Feed'] == 'Small'].copy()
                                    big_feed_df = small_hlc_df[small_hlc_df['Feed'] == 'Big'].copy()
                                    
                                    st.write(f"📋 Processing 2 feed(s): Small, Big")
                                    st.write(f"  📊 Small feed: {len(small_feed_df)} rows")
                                    st.write(f"  📊 Big feed: {len(big_feed_df)} rows")
                                    
                                    # Show output range for each feed (use 'Output' column)
                                    if 'Output' in small_feed_df.columns:
                                        small_min = small_feed_df['Output'].min()
                                        small_max = small_feed_df['Output'].max()
                                        st.write(f"🧮 Small Feed Range: [{small_min:.2f}, {small_max:.2f}]")
                                    
                                    if 'Output' in big_feed_df.columns:
                                        big_min = big_feed_df['Output'].min()
                                        big_max = big_feed_df['Output'].max()
                                        st.write(f"🧮 Big Feed Range: [{big_min:.2f}, {big_max:.2f}]")
                                    
                                    # Use separated feeds
                                    cluster_small = small_feed_df
                                    cluster_big = big_feed_df
                                else:
                                    # No Feed column - treat all as one feed
                                    st.write(f"📋 Processing combined data")
                                    st.write(f"  📊 Total: {len(small_hlc_df)} rows")
                                    cluster_small = small_hlc_df.copy()
                                    cluster_big = None
                                
                                # Show overall coverage (use 'Output' column)
                                if 'Output' in small_hlc_df.columns:
                                    output_min = small_hlc_df['Output'].min()
                                    output_max = small_hlc_df['Output'].max()
                                    output_spread = output_max - output_min
                                    
                                    st.success(f"📊 Overall Output Coverage: {output_min:.2f} to {output_max:.2f} (Spread: {output_spread:.2f} units)")
                                
                                # Store bypass data in session state for use by button handlers
                                st.session_state.tab8_bypass_data = {
                                    'small_hlc_df': small_hlc_df,
                                    'cluster_small': cluster_small,
                                    'cluster_big': cluster_big,
                                    'report_time': report_time
                                }
                                
                            except Exception as e:
                                st.error(f"❌ Error during bypass mode processing: {str(e)}")
                                import traceback
                                with st.expander("🔍 Error Details"):
                                    st.code(traceback.format_exc())
                                st.session_state.tab8_processed = False  # Reset on error
                
                # SHOW ANALYSIS BUTTONS if processing has been completed
                if st.session_state.tab8_processed and bypass_mode_active:
                    # Retrieve data from session state
                    bypass_data = st.session_state.tab8_bypass_data
                    if bypass_data:
                        small_hlc_df = bypass_data['small_hlc_df']
                        report_time = bypass_data['report_time']
                        
                        st.markdown("---")
                        st.markdown("### 🎯 Trading Models Analysis")
                        
                        # MODEL SELECTION UI
                        st.markdown("**Select Models to Process:**")
                        
                        # Organize models by category
                        from model_definitions_v21 import MODELS
                        categories = organize_results_by_category({'results': {}})
                        
                        # Create columns for model selection
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            select_all = st.checkbox("✅ Select All Models", value=True, key="select_all_models")
                        
                        with col2:
                            st.caption("Choose which trading models to process. Results will display as each model completes.")
                        
                        # Model selection checkboxes organized by category
                        selected_models = []
                        
                        for category_name, model_list in categories.items():
                            if not model_list:
                                continue
                            
                            with st.expander(f"📁 {category_name}", expanded=True):
                                for model_name in model_list:
                                    model_info = get_model_display_info(model_name)
                                    if model_info:
                                        is_selected = st.checkbox(
                                            f"Model #{model_info['number']}: {model_info['display_name']}",
                                            value=select_all,
                                            key=f"select_{model_name}",
                                            help=model_info['description']
                                        )
                                        if is_selected:
                                            selected_models.append(model_name)
                        
                        # Show selected count
                        st.info(f"📊 {len(selected_models)} model(s) selected")
                        
                        # Process button
                        if st.button("🚀 Process Selected Models", key="run_selected_models", type="primary", disabled=len(selected_models)==0):
                            try:
                                st.markdown("---")
                                st.markdown("### ⚙️ Processing Models...")
                                
                                # Import the bypass mode matcher
                                from bypass_mode_matcher import match_travelers_bypass_mode, process_model_bypass_mode
                                
                                import time as time_module
                                total_start = time_module.time()
                                
                                all_results = {
                                    'results': {},
                                    'timings': {},
                                    'metadata': {
                                        'report_time': report_time,
                                        'mode': 'Bypass Mode',
                                        'max_spread': 3.0
                                    }
                                }
                                
                                # Progress tracking
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Create placeholder for results display
                                results_container = st.container()
                                
                                # Process each selected model
                                for idx, model_name in enumerate(selected_models):
                                    model_def = MODELS[model_name]
                                    model_start = time_module.time()
                                    
                                    # Update progress
                                    progress = (idx) / len(selected_models)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Processing {idx+1}/{len(selected_models)}: {model_name}...")
                                    
                                    # Get Pass 1 and Pass 2 M#s from model definition
                                    pass1_set = model_def.get('pass1', set())
                                    pass2_set = model_def.get('pass2', set())
                                    
                                    # Convert sets to sorted lists
                                    pass1_ms = sorted(list(pass1_set))
                                    pass2_ms = sorted(list(pass2_set))
                                    
                                    # Store all matches (Today + Recent)
                                    all_model_matches = []
                                    
                                    # === RUN 1: TODAY MATCHES ([0]) ===
                                    model_config_today = {
                                        'pass1_ms': pass1_ms,
                                        'pass2_ms': pass2_ms,
                                        'day_filter': '[0]',  # Today only
                                        'feed_selection': 'both'
                                    }
                                    
                                    matched_df_today = process_model_bypass_mode(
                                        small_hlc_df,
                                        model_config_today,
                                        max_spread=3.0
                                    )
                                    
                                    if not matched_df_today.empty:
                                        # Tag as "Today"
                                        matched_df_today['Recent'] = 'Today'
                                        all_model_matches.append(matched_df_today)
                                    
                                    # === RUN 2: RECENT MATCHES ([-1] if present, else [-3]) ===
                                    # Try [-1] first
                                    model_config_recent1 = {
                                        'pass1_ms': pass1_ms,
                                        'pass2_ms': pass2_ms,
                                        'day_filter': '[-1]',  # Recent: -1 day
                                        'feed_selection': 'both'
                                    }
                                    
                                    matched_df_recent = process_model_bypass_mode(
                                        small_hlc_df,
                                        model_config_recent1,
                                        max_spread=3.0
                                    )
                                    
                                    # If [-1] has no results, try [-3]
                                    if matched_df_recent.empty:
                                        model_config_recent3 = {
                                            'pass1_ms': pass1_ms,
                                            'pass2_ms': pass2_ms,
                                            'day_filter': '[-3]',  # Recent: -3 days (fallback)
                                            'feed_selection': 'both'
                                        }
                                        
                                        matched_df_recent = process_model_bypass_mode(
                                            small_hlc_df,
                                            model_config_recent3,
                                            max_spread=3.0
                                        )
                                    
                                    # Add Recent matches if any exist
                                    if not matched_df_recent.empty:
                                        # Tag as "Recent"
                                        matched_df_recent['Recent'] = 'Recent'
                                        all_model_matches.append(matched_df_recent)
                                    
                                    # Combine all matches (Today + Recent)
                                    if all_model_matches:
                                        matched_df = pd.concat(all_model_matches, ignore_index=True)
                                    else:
                                        matched_df = pd.DataFrame()
                                    
                                    # ENHANCEMENT: Extract Tag and Family from traveler report
                                    if not matched_df.empty and 'Tag' in small_hlc_df.columns and 'Family' in small_hlc_df.columns:
                                        # Create lookup dictionaries for Tag and Family based on M#, Origin, Feed, Arrival
                                        # This ensures we get the correct Tag/Family for each specific traveler
                                        
                                        def create_traveler_key(row):
                                            """Create unique key for traveler lookup"""
                                            try:
                                                return (
                                                    int(row['M #']), 
                                                    str(row['Origin']), 
                                                    str(row['Feed']),
                                                    pd.to_datetime(row['Arrival']).strftime('%Y-%m-%d %H:%M:%S')
                                                )
                                            except:
                                                return None
                                        
                                        # Build lookup dictionary: (M#, Origin, Feed, Arrival) -> (Tag, Family)
                                        traveler_lookup = {}
                                        for _, row in small_hlc_df.iterrows():
                                            key = create_traveler_key(row)
                                            if key:
                                                traveler_lookup[key] = {
                                                    'Tag': row.get('Tag', ''),
                                                    'Family': row.get('Family', '')
                                                }
                                        
                                        # Function to lookup Tag/Family for a matched row
                                        def lookup_tag_family(row, traveler_num):
                                            """Lookup Tag and Family for traveler 1 or 2"""
                                            try:
                                                m_num = int(row[f'M{traveler_num}'])
                                                origin = str(row[f'Origin{traveler_num}'])
                                                feed = str(row[f'Feed{traveler_num}'])
                                                arrival = pd.to_datetime(row[f'Arrival{traveler_num}']).strftime('%Y-%m-%d %H:%M:%S')
                                                
                                                key = (m_num, origin, feed, arrival)
                                                traveler_data = traveler_lookup.get(key, {})
                                                
                                                return pd.Series([
                                                    traveler_data.get('Tag', ''),
                                                    traveler_data.get('Family', '')
                                                ])
                                            except:
                                                return pd.Series(['', ''])
                                        
                                        # Extract Tag1, Family1 from first traveler
                                        matched_df[['Tag1', 'Family1']] = matched_df.apply(
                                            lambda row: lookup_tag_family(row, 1), axis=1
                                        )
                                        
                                        # Extract Tag2, Family2 from second traveler
                                        matched_df[['Tag2', 'Family2']] = matched_df.apply(
                                            lambda row: lookup_tag_family(row, 2), axis=1
                                        )
                                    
                                    # Apply any special model-specific filtering
                                    if model_def.get('check_recip', False):
                                        # Filter for reciprocal pairs
                                        # Reciprocal pairs can be EITHER:
                                        # 1. Same-sign: R1 == M2 AND R2 == M1 (e.g., M1=6, M2=87, R1=87, R2=6)
                                        # 2. Opposite-sign: M1 == -R2 AND M2 == -R1 (e.g., M1=6, M2=-87, R1=87, R2=-6)
                                        if not matched_df.empty and 'R1' in matched_df.columns and 'R2' in matched_df.columns:
                                            # Check for same-sign reciprocals
                                            same_sign_recip = (
                                                (matched_df['R1'] == matched_df['M2']) &
                                                (matched_df['R2'] == matched_df['M1'])
                                            )
                                            
                                            # Check for opposite-sign reciprocals
                                            opposite_sign_recip = (
                                                (matched_df['M1'] == -matched_df['R2']) &
                                                (matched_df['M2'] == -matched_df['R1'])
                                            )
                                            
                                            # Accept BOTH types of reciprocals
                                            matched_df = matched_df[same_sign_recip | opposite_sign_recip]
                                    
                                    # Store results
                                    all_results['results'][model_name] = matched_df
                                    model_time = time_module.time() - model_start
                                    all_results['timings'][model_name] = model_time
                                    
                                    # DISPLAY RESULTS IMMEDIATELY (Progressive Display)
                                    with results_container:
                                        model_info = get_model_display_info(model_name)
                                        
                                        with st.expander(
                                            f"✅ Model #{model_info['number']}: {model_info['display_name']} "
                                            f"({len(matched_df)} matches, {model_time:.2f}s)",
                                            expanded=(len(matched_df) > 0 and idx < 3)  # Expand first 3 with matches
                                        ):
                                            st.markdown(f"**Description:** {model_info['description']}")
                                            
                                            if len(matched_df) > 0:
                                                st.markdown("""
                                                **Legend:** 
                                                🟡 **Yellow** = Day [0] (Today)  |  
                                                🔵 **Blue** = Days [-1], [-2], [-3] (Recent)
                                                """)
                                                
                                                # Helper function for highlighting
                                                def highlight_arrival_brackets(df):
                                                    def apply_bracket_color(row):
                                                        colors = [''] * len(row)
                                                        if 'Arrival_Brackets' in df.columns:
                                                            bracket_idx = df.columns.get_loc('Arrival_Brackets')
                                                            bracket_val = str(row.get('Arrival_Brackets', ''))
                                                            if '[0]' in bracket_val:
                                                                colors[bracket_idx] = 'background-color: #fff9c4'
                                                            elif any(x in bracket_val for x in ['[-1]', '[-2]', '[-3]']):
                                                                colors[bracket_idx] = 'background-color: #bbdefb'
                                                        return colors
                                                    return apply_bracket_color
                                                
                                                styled_df = matched_df.style.apply(
                                                    highlight_arrival_brackets(matched_df),
                                                    axis=1
                                                ).format({
                                                    'Output1': lambda x: f'{x:.3f}' if pd.notna(x) else '',
                                                    'Output2': lambda x: f'{x:.3f}' if pd.notna(x) else '',
                                                    'Prox': lambda x: f'{x:.3f}' if pd.notna(x) else ''
                                                })
                                                
                                                st.dataframe(styled_df, use_container_width=True, height=400)
                                            else:
                                                st.info("No matches found for this model.")
                                
                                # Complete progress
                                progress_bar.progress(1.0)
                                status_text.text("")
                                
                                # Calculate total time
                                all_results['total_time'] = time_module.time() - total_start
                                
                                # Display summary
                                st.markdown("---")
                                st.markdown("### 📊 Processing Complete")
                                
                                summary_stats = create_summary_stats(all_results)
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Models Processed", len(selected_models))
                                with col2:
                                    st.metric("Models with Matches", summary_stats['models_with_matches'])
                                with col3:
                                    st.metric("Total Matches", summary_stats['total_matches'])
                                with col4:
                                    st.metric("Processing Time", f"{all_results['total_time']:.1f}s")
                                
                                st.success(f"✅ Processed {len(selected_models)} model(s) in {all_results['total_time']:.1f}s!")
                                
                                # Excel Export (v23 unified format)
                                if summary_stats['total_matches'] > 0:
                                    st.markdown("---")
                                    st.markdown("### 📥 Export Results")
                                    
                                    # Create minimal measurement_df for R# extraction if needed
                                    bypass_measurement_df = None
                                    if 'M #' in small_hlc_df.columns and 'R #' in small_hlc_df.columns:
                                        bypass_measurement_df = small_hlc_df[['M #', 'R #']].drop_duplicates()
                                    
                                    create_download_button(all_results, report_time, bypass_measurement_df)
                                
                            except Exception as e:
                                st.error(f"❌ Error processing models: {str(e)}")
                                import traceback
                                with st.expander("🔍 Error Details"):
                                    st.code(traceback.format_exc())
                
            except Exception as e:
                st.error(f"Error in Tab 8 analysis: {str(e)}")
                import traceback
                with st.expander("DEBUG: Error Details"):
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

# ==============================================================================
# ================================================================================
# FILE NOTES - SWING ANALYSIS TOOL
# ================================================================================
#
# RECENT UPDATES and Version History Summary:
# ====================
# 06a_v03 post run review: bug fix to show all Recip results creates new bug.  
# The previous bug was: same up and same down result missing, only flip results showed.
# the new bug: flip results now missing, only same up and down results show.

# 06a_v03 (2026-01-11) - Critical bug fix + export enhancements

# Fixed reciprocal filter (removed incorrect negative sign)
# Updated Prox colors (pink/yellow)
# Added M/R font colors (red/blue)
# Set custom column widths
# Format Prox to 2 decimals

# 06a_v02 - Tag/Family extraction

# 06a_v01 - Model selection, progressive display
# 06a_v01 detail belows:  
# posted on 1/10/2025 around 7 PM
# added Unified Excel Export Format (v23)
# Changed: Both normal and bypass modes now use the new excel_exporter_v23 for consistent 26-column output.

# Model X Detection Removed

# Individual Model Selection Panel
# New Feature: Interactive model selection UI organized by category.

# Progressive Results Display
# New Feature: Results display as each model completes processing.

# Enhanced Export Call
# Changed: Export function now receives measurement_df for R# extraction.

# End of File
# ==============================================================================
