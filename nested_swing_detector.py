"""
Nested Swing Detection Algorithm
Version: 1.0
Date: 2025-12-28

Detects multi-level swing patterns in price data, identifying both nested swings
and major turning points.
"""

import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime


def parse_timestamp_naive(timestamp):
    """
    Parse timestamp and return naive datetime (remove timezone info).
    
    This is the preferred method for swing analysis to avoid timezone
    comparison issues.
    
    Args:
        timestamp: Can be string, datetime, or pd.Timestamp
    
    Returns:
        Naive datetime object (timezone-unaware)
    """
    try:
        if isinstance(timestamp, str):
            # Handle ISO format with timezone
            if 'T' in timestamp:
                # Parse ISO format and remove timezone
                dt = pd.to_datetime(timestamp)
                return dt.replace(tzinfo=None)
            else:
                # Simple format without timezone
                return pd.to_datetime(timestamp)
        elif isinstance(timestamp, pd.Timestamp):
            # Remove timezone from pandas Timestamp
            return timestamp.replace(tzinfo=None)
        elif isinstance(timestamp, datetime):
            # Remove timezone from datetime
            return timestamp.replace(tzinfo=None)
        else:
            return timestamp
    except:
        return timestamp


def detect_nested_swings(
    trading_day_df: pd.DataFrame,
    min_swing_size: float = 60,
    pullback_tolerance: float = 30
) -> List[Dict]:
    """
    Detect all nested swings in price data.
    
    A swing remains active until price retraces beyond the pullback tolerance
    from its origin. This allows larger swings to contain smaller ones.
    
    Args:
        trading_day_df: DataFrame with columns ['time', 'high', 'low']
        min_swing_size: Minimum swing size in points (default: 60)
        pullback_tolerance: Points past origin that invalidate the swing (default: 30)
    
    Returns:
        List of swing dictionaries with:
            - from_time: Origin timestamp (naive datetime)
            - from_price: Origin price
            - to_time: Extreme timestamp (naive datetime)
            - to_price: Extreme price
            - swing_size: Size in points
            - direction: 'Up' or 'Down'
    """
    active_swings = []
    completed_swings = []
    
    trading_day_df = trading_day_df.copy()
    trading_day_df = trading_day_df.reset_index(drop=True)
    
    # Strip timezone from all timestamps
    trading_day_df['time'] = trading_day_df['time'].apply(parse_timestamp_naive)
    
    for idx, row in trading_day_df.iterrows():
        current_time = row['time']
        current_high = row['high']
        current_low = row['low']
        
        # Start potential down swing from new highs
        if idx == 0 or current_high > trading_day_df.loc[idx-1, 'high']:
            active_swings.append({
                'origin_time': current_time,
                'origin_price': current_high,
                'direction': 'Down',
                'extreme_price': current_high,
                'extreme_time': current_time,
                'max_swing': 0
            })
        
        # Start potential up swing from new lows
        if idx == 0 or current_low < trading_day_df.loc[idx-1, 'low']:
            active_swings.append({
                'origin_time': current_time,
                'origin_price': current_low,
                'direction': 'Up',
                'extreme_price': current_low,
                'extreme_time': current_time,
                'max_swing': 0
            })
        
        # Update all active swings
        swings_to_remove = []
        
        for swing in active_swings:
            if swing['direction'] == 'Down':
                # Update if we made a new low
                if current_low < swing['extreme_price']:
                    swing['extreme_price'] = current_low
                    swing['extreme_time'] = current_time
                    swing['max_swing'] = swing['origin_price'] - swing['extreme_price']
                
                # Check if price retraced beyond tolerance
                if current_high > swing['origin_price'] + pullback_tolerance:
                    if swing['max_swing'] >= min_swing_size:
                        completed_swings.append({
                            'from_time': swing['origin_time'],
                            'from_price': swing['origin_price'],
                            'to_time': swing['extreme_time'],
                            'to_price': swing['extreme_price'],
                            'swing_size': swing['max_swing'],
                            'direction': 'Down'
                        })
                    swings_to_remove.append(swing)
            
            elif swing['direction'] == 'Up':
                # Update if we made a new high
                if current_high > swing['extreme_price']:
                    swing['extreme_price'] = current_high
                    swing['extreme_time'] = current_time
                    swing['max_swing'] = swing['extreme_price'] - swing['origin_price']
                
                # Check if price retraced beyond tolerance
                if current_low < swing['origin_price'] - pullback_tolerance:
                    if swing['max_swing'] >= min_swing_size:
                        completed_swings.append({
                            'from_time': swing['origin_time'],
                            'from_price': swing['origin_price'],
                            'to_time': swing['extreme_time'],
                            'to_price': swing['extreme_price'],
                            'swing_size': swing['max_swing'],
                            'direction': 'Up'
                        })
                    swings_to_remove.append(swing)
        
        # Remove completed swings
        for swing in swings_to_remove:
            active_swings.remove(swing)
    
    # Close any remaining active swings at end of period
    for swing in active_swings:
        if swing['max_swing'] >= min_swing_size:
            completed_swings.append({
                'from_time': swing['origin_time'],
                'from_price': swing['origin_price'],
                'to_time': swing['extreme_time'],
                'to_price': swing['extreme_price'],
                'swing_size': swing['max_swing'],
                'direction': swing['direction']
            })
    
    return completed_swings


def identify_major_turning_points(
    completed_swings: List[Dict],
    min_time_separation_minutes: int = 90
) -> List[Dict]:
    """
    Identify major turning points from nested swings.
    
    Filters nested swings to find only the critical reversal points where
    price direction actually changed significantly.
    
    Args:
        completed_swings: List of swing dictionaries from detect_nested_swings()
        min_time_separation_minutes: Minimum time between major points (default: 90)
    
    Returns:
        List of major turning point dictionaries with:
            - time: Point timestamp
            - price: Point price
            - type: 'High' or 'Low'
            - is_reversal: Boolean
            - is_major_extreme: Boolean
            - as_origin_count: Number of swings originating from this point
            - as_extreme_count: Number of swings ending at this point
            - max_as_origin: Largest swing size from this origin
            - max_as_extreme: Largest swing size to this extreme
            - significance_score: Calculated importance score
    """
    # Build point significance data
    point_data = {}
    
    for swing in completed_swings:
        # Track origin points
        origin_key = (swing['from_time'], swing['from_price'])
        if origin_key not in point_data:
            point_data[origin_key] = {
                'time': swing['from_time'],
                'price': swing['from_price'],
                'type': 'High' if swing['direction'] == 'Down' else 'Low',
                'as_origin_count': 0,
                'as_extreme_count': 0,
                'max_as_origin': 0,
                'max_as_extreme': 0
            }
        point_data[origin_key]['as_origin_count'] += 1
        point_data[origin_key]['max_as_origin'] = max(
            point_data[origin_key]['max_as_origin'],
            swing['swing_size']
        )
        
        # Track extreme points
        extreme_key = (swing['to_time'], swing['to_price'])
        if extreme_key not in point_data:
            point_data[extreme_key] = {
                'time': swing['to_time'],
                'price': swing['to_price'],
                'type': 'Low' if swing['direction'] == 'Down' else 'High',
                'as_origin_count': 0,
                'as_extreme_count': 0,
                'max_as_origin': 0,
                'max_as_extreme': 0
            }
        point_data[extreme_key]['as_extreme_count'] += 1
        point_data[extreme_key]['max_as_extreme'] = max(
            point_data[extreme_key]['max_as_extreme'],
            swing['swing_size']
        )
    
    # Identify critical points (reversals or major extremes)
    critical_points = []
    
    for key, point in point_data.items():
        is_reversal = (point['as_origin_count'] > 0 and point['as_extreme_count'] > 0)
        is_major_extreme = point['as_extreme_count'] >= 5
        
        # Calculate significance score
        reversal_score = (point['max_as_origin'] + point['max_as_extreme']) if is_reversal else 0
        extreme_score = point['max_as_extreme'] * 2 if is_major_extreme else point['max_as_extreme']
        origin_score = point['max_as_origin']
        total_score = reversal_score + extreme_score + origin_score
        
        # Only include reversal points or major extremes
        if is_reversal or is_major_extreme:
            critical_points.append({
                **point,
                'is_reversal': is_reversal,
                'is_major_extreme': is_major_extreme,
                'significance_score': total_score
            })
    
    # Apply spatial filtering to get most significant points
    critical_points.sort(key=lambda x: x['time'])
    major_points = []
    
    for candidate in critical_points:
        too_close = False
        for selected in major_points:
            time_diff_minutes = abs((candidate['time'] - selected['time']).total_seconds() / 60)
            if time_diff_minutes < min_time_separation_minutes:
                if selected['significance_score'] >= candidate['significance_score']:
                    too_close = True
                    break
                else:
                    # This candidate is better, remove the previous one
                    major_points.remove(selected)
                    break
        
        if not too_close:
            major_points.append(candidate)
    
    major_points.sort(key=lambda x: x['time'])
    return major_points


def analyze_swings(
    df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    min_swing_size: float = 60,
    pullback_tolerance: float = 30,
    min_time_separation_minutes: int = 90
) -> Tuple[List[Dict], List[Dict]]:
    """
    Complete swing analysis: detect nested swings and identify major turning points.
    
    Args:
        df: DataFrame with columns ['time', 'high', 'low']
        start_time: Start of analysis period
        end_time: End of analysis period
        min_swing_size: Minimum swing size in points (default: 60)
        pullback_tolerance: Points past origin that invalidate swing (default: 30)
        min_time_separation_minutes: Minimum time between major points (default: 90)
    
    Returns:
        Tuple of (nested_swings, major_turning_points)
    """
    # Strip timezone from input times and dataframe
    start_time = parse_timestamp_naive(start_time)
    end_time = parse_timestamp_naive(end_time)
    
    df = df.copy()
    df['time'] = df['time'].apply(parse_timestamp_naive)
    
    # Filter to trading period
    trading_day = df[(df['time'] >= start_time) & (df['time'] <= end_time)].copy()
    
    # Detect all nested swings
    nested_swings = detect_nested_swings(
        trading_day,
        min_swing_size=min_swing_size,
        pullback_tolerance=pullback_tolerance
    )
    
    # Identify major turning points
    major_points = identify_major_turning_points(
        nested_swings,
        min_time_separation_minutes=min_time_separation_minutes
    )
    
    return nested_swings, major_points
