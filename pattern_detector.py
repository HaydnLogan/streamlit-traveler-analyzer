"""
Pattern Detection System for Traveler Analysis
Detects "M# arrives with opposites" patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# Family and Tag definitions
EPIC_ORIGINS = {"Trinidad", "Tobago", "Wasp-12b", "Macedonia"}
ANCHOR_ORIGINS = {"Spain", "Saturn", "Jupiter", "Kepler-62", "Kepler-44"}

# Green Family
FAMILY_ALPHA_TRAVELERS = {2, -2, 10, -10, 21, -21, 22, -22, 30, -30, 36, -36, 39, -39, 41, -41, 43, -43, 50, -50, 60, -60, 77, -77, 107, -107}
FAMILY_BRAVO_TRAVELERS = {5, -5, 14, -14, 26, -26, 55, -55, 56, -56, 68, -68, 91, -91, 96, -96}
FAMILY_CHARLIE_TRAVELERS = {6, -6, 25, -25, 49, -49, 79, -79, 87, -87}
FAMILY_DELTA_TRAVELERS = {3, -3, 37, -37, 63, -63, 99, -99.2, 103, -103}
FAMILY_ECHO_TRAVELERS = {1, -1, 73, -73, 111, -111}
FAMILY_FOXY_TRAVELERS = {52, -52, 59, -59, 65, -65, 70, -70, 76, -76, 82, -82, 86, -86, 88, -88, 97, -97}

# Indigo Family
FAMILY_WILD_TRAVELERS = {0, 40, -40}
FAMILY_BLU_TRAVELERS = {15, -15, 27, -27, 33, -33, 38, -38, 42, -42, 45, -45, 54, -54, 67, -67, 74, -74, 80, -80, 85, -85, 89, -89, 92, -92, 95, -95, 96.1, -96.1, 97.2, -97.2, 98.2, -98.2, 99.3, -99.3}
FAMILY_ORN_TRAVELERS = {4, -4, 12, -12, 24, -24, 31, -31, 47, -47, 57, -57, 71, -71, 85, -85, 93.5, -93.5, 97.2, -97.2, 101, -101}
FAMILY_GRY_TRAVELERS = {62, -62, 78.01, -78.01, 83, -83, 86.5, -86.5, 90.5, -90.5, 95.5, -95.5, 96.1, -96.1, 97.1, -97.1, 98.1, -98.1, 99.1, -99.1}

# X0 Tags
X0P_TAGS = {50, -50, 60, -60, 68, -68, 77, -77, 87, -87, 96, -96, 103, -103, 107, -107, 111, -111}
X0D_TAGS = {30, -30, 22, -22, 14, -14, 10, -10, 6, -6, 5, -5, 3, -3, 2, -2, 1, -1}

# XD0 Tags
XD0P_TAGS = {54, -54, 67, -67, 74, -74, 80, -80, 85, -85, 89, -89, 92, -92, 95, -95, 96.1, -96.1, 97.2, -97.2}
XD0D_TAGS = {27, -27, 15, -15}

def get_family(m_value: float) -> str:
    """Determine which family an M# belongs to"""
    if m_value in FAMILY_WILD_TRAVELERS:
        return "Indigo Wild"
    elif m_value in FAMILY_BLU_TRAVELERS:
        return "Indigo Blu"
    elif m_value in FAMILY_ORN_TRAVELERS:
        return "Indigo Orn"
    elif m_value in FAMILY_GRY_TRAVELERS:
        return "Indigo Gry"
    elif m_value in FAMILY_ALPHA_TRAVELERS:
        return "Green Alpha"
    elif m_value in FAMILY_BRAVO_TRAVELERS:
        return "Green Bravo"
    elif m_value in FAMILY_CHARLIE_TRAVELERS:
        return "Green Charlie"
    elif m_value in FAMILY_DELTA_TRAVELERS:
        return "Green Delta"
    elif m_value in FAMILY_ECHO_TRAVELERS:
        return "Green Echo"
    elif m_value in FAMILY_FOXY_TRAVELERS:
        return "Green Foxy"
    return "Unknown"

def get_tag_category(m_value: float) -> str:
    """Determine tag category"""
    if m_value in X0P_TAGS:
        return "X0p"
    elif m_value in X0D_TAGS:
        return "X0d"
    elif m_value in XD0P_TAGS:
        return "XD0p"
    elif m_value in XD0D_TAGS:
        return "XD0d"
    return "Other"

def is_epic_origin(origin: str) -> bool:
    """Check if origin is EPIC"""
    return any(epic in origin for epic in EPIC_ORIGINS)

def is_anchor_origin(origin: str) -> bool:
    """Check if origin is Anchor"""
    return any(anchor in origin for anchor in ANCHOR_ORIGINS)

class PatternDetector:
    def __init__(self, traveler_df: pd.DataFrame, ohlc_df: pd.DataFrame = None):
        """
        Initialize pattern detector
        
        Args:
            traveler_df: Traveler report dataframe
            ohlc_df: OHLC price data (optional, for validation)
        """
        self.traveler_df = traveler_df.copy()
        self.ohlc_df = ohlc_df
        
        # Parse datetime using naive method (drop timezone)
        # For traveler_df, already in consistent timezone
        self.traveler_df['Arrival'] = pd.to_datetime(self.traveler_df['Arrival'])
        if hasattr(self.traveler_df['Arrival'].dtype, 'tz') and self.traveler_df['Arrival'].dtype.tz is not None:
            self.traveler_df['Arrival'] = self.traveler_df['Arrival'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        if self.ohlc_df is not None:
            # Use utc=True to handle mixed timezones, then convert to naive
            self.ohlc_df['time'] = pd.to_datetime(self.ohlc_df['time'], utc=True)
            # Convert to naive by removing timezone
            self.ohlc_df['time'] = self.ohlc_df['time'].dt.tz_localize(None)
    
    def find_opposite_patterns(self, 
                                output_spread_tolerance: float = 3.0,
                                min_days_before: int = 1) -> pd.DataFrame:
        """
        Find patterns where M# arrives with opposites arriving before it
        
        Args:
            output_spread_tolerance: Maximum output spread for pattern (default 3.0)
            min_days_before: Minimum days before Day [0] to look for opposites
            
        Returns:
            DataFrame with detected patterns
        """
        patterns = []
        
        # Get Day [0] arrivals
        day_0_df = self.traveler_df[self.traveler_df['Day'] == '[0]'].copy()
        
        for idx, row in day_0_df.iterrows():
            m_value = row['M #']
            output = row['Output']
            arrival_time = row['Arrival']
            feed = row['Feed']
            origin = row['Origin']
            
            # Calculate opposite M# value
            opposite_m = -m_value
            
            # Find arrivals before this one in same feed
            earlier_df = self.traveler_df[
                (self.traveler_df['Feed'] == feed) &
                (self.traveler_df['Arrival'] < arrival_time)
            ].copy()
            
            # Look for positive opposite
            pos_opposite_df = earlier_df[
                (earlier_df['M #'] == abs(m_value)) &
                (abs(earlier_df['Output'] - output) <= output_spread_tolerance)
            ]
            
            # Look for negative opposite
            neg_opposite_df = earlier_df[
                (earlier_df['M #'] == -abs(m_value)) &
                (abs(earlier_df['Output'] - output) <= output_spread_tolerance)
            ]
            
            # If we found both opposites
            if len(pos_opposite_df) > 0 and len(neg_opposite_df) > 0:
                # Get the most recent of each opposite
                pos_latest = pos_opposite_df.nlargest(1, 'Arrival').iloc[0]
                neg_latest = neg_opposite_df.nlargest(1, 'Arrival').iloc[0]
                
                # Calculate max output spread
                outputs = [output, pos_latest['Output'], neg_latest['Output']]
                max_spread = max(outputs) - min(outputs)
                
                # Determine family and origin characteristics
                all_same_family = (get_family(m_value) == get_family(pos_latest['M #']) == get_family(neg_latest['M #']))
                all_indigo = all([get_family(v).startswith('Indigo') for v in [m_value, pos_latest['M #'], neg_latest['M #']]])
                all_green = all([get_family(v).startswith('Green') for v in [m_value, pos_latest['M #'], neg_latest['M #']]])
                
                all_x0_tags = all([get_tag_category(v).startswith('X0') for v in [m_value, pos_latest['M #'], neg_latest['M #']]])
                all_xd0_tags = all([get_tag_category(v).startswith('XD0') for v in [m_value, pos_latest['M #'], neg_latest['M #']]])
                
                all_epic = all([is_epic_origin(o) for o in [origin, pos_latest['Origin'], neg_latest['Origin']]])
                all_anchor = all([is_anchor_origin(o) for o in [origin, pos_latest['Origin'], neg_latest['Origin']]])
                
                patterns.append({
                    'Report_Time': arrival_time,
                    'Feed': feed,
                    'M#_Day0': m_value,
                    'Origin_Day0': origin,
                    'Output_Day0': output,
                    'M#_Pos_Prior': pos_latest['M #'],
                    'Origin_Pos_Prior': pos_latest['Origin'],
                    'Output_Pos_Prior': pos_latest['Output'],
                    'Arrival_Pos_Prior': pos_latest['Arrival'],
                    'M#_Neg_Prior': neg_latest['M #'],
                    'Origin_Neg_Prior': neg_latest['Origin'],
                    'Output_Neg_Prior': neg_latest['Output'],
                    'Arrival_Neg_Prior': neg_latest['Arrival'],
                    'Max_Output_Spread': max_spread,
                    'Family_Day0': get_family(m_value),
                    'Tag_Category_Day0': get_tag_category(m_value),
                    'All_Same_Family': all_same_family,
                    'All_Indigo': all_indigo,
                    'All_Green': all_green,
                    'All_X0_Tags': all_x0_tags,
                    'All_XD0_Tags': all_xd0_tags,
                    'All_EPIC_Origins': all_epic,
                    'All_Anchor_Origins': all_anchor,
                    'Avg_Output': np.mean(outputs),
                    'Output_Range': f"{min(outputs):.2f} to {max(outputs):.2f}"
                })
        
        return pd.DataFrame(patterns)
    
    def validate_pattern(self, pattern_row: pd.Series, lookahead_hours: int = 24) -> Dict:
        """
        Validate pattern by checking what happened after it appeared
        
        Args:
            pattern_row: Row from patterns dataframe
            lookahead_hours: Hours to look ahead for validation
            
        Returns:
            Dictionary with validation metrics
        """
        if self.ohlc_df is None:
            return {'error': 'No OHLC data provided'}
        
        report_time = pattern_row['Report_Time']
        avg_output = pattern_row['Avg_Output']
        
        # Find OHLC data around this time
        time_window_start = report_time - timedelta(hours=1)
        time_window_end = report_time + timedelta(hours=lookahead_hours)
        
        window_df = self.ohlc_df[
            (self.ohlc_df['time'] >= time_window_start) &
            (self.ohlc_df['time'] <= time_window_end)
        ].copy()
        
        if len(window_df) == 0:
            return {'error': 'No OHLC data in window'}
        
        # Get open price at report time (or nearest)
        open_price_row = window_df.iloc[(window_df['time'] - report_time).abs().argsort()[:1]]
        open_price = open_price_row['open'].values[0]
        
        # Calculate distance from pattern output
        distance_from_pattern = abs(open_price - avg_output)
        
        # Find if price reached the pattern zone
        reached_pattern = any(
            (window_df['low'] <= avg_output + 3.0) & 
            (window_df['high'] >= avg_output - 3.0)
        )
        
        # Calculate max move in direction of pattern
        if open_price > avg_output:
            # Gap up, look for downward move toward pattern
            max_move_toward = open_price - window_df['low'].min()
            direction = 'Down'
        else:
            # Gap down or at pattern, look for upward move
            max_move_toward = window_df['high'].max() - open_price
            direction = 'Up'
        
        return {
            'Open_Price': open_price,
            'Avg_Pattern_Output': avg_output,
            'Distance_From_Pattern': distance_from_pattern,
            'Reached_Pattern_Zone': reached_pattern,
            'Max_Move_Toward_Pattern': max_move_toward,
            'Direction': direction,
            'Window_High': window_df['high'].max(),
            'Window_Low': window_df['low'].min(),
            'Total_Range': window_df['high'].max() - window_df['low'].min()
        }
