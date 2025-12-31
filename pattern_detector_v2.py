"""
Pattern Detection System V2
Detects "M# arrives after opposite pair" patterns
where ANY M# can arrive after ANY opposite pair (e.g., M# 0 arrives after +40/-40 pair)
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import definitions from original
from pattern_detector import (
    EPIC_ORIGINS, ANCHOR_ORIGINS,
    FAMILY_WILD_TRAVELERS, FAMILY_BLU_TRAVELERS, FAMILY_ORN_TRAVELERS, FAMILY_GRY_TRAVELERS,
    FAMILY_ALPHA_TRAVELERS, FAMILY_BRAVO_TRAVELERS, FAMILY_CHARLIE_TRAVELERS,
    FAMILY_DELTA_TRAVELERS, FAMILY_ECHO_TRAVELERS, FAMILY_FOXY_TRAVELERS,
    X0P_TAGS, X0D_TAGS, XD0P_TAGS, XD0D_TAGS,
    get_family, get_tag_category, is_epic_origin, is_anchor_origin
)

class PatternDetectorV2:
    def __init__(self, traveler_df: pd.DataFrame, ohlc_df: pd.DataFrame = None):
        """
        Initialize pattern detector V2
        
        Args:
            traveler_df: Traveler report dataframe
            ohlc_df: OHLC price data (optional, for validation)
        """
        self.traveler_df = traveler_df.copy()
        self.ohlc_df = ohlc_df
        
        # Parse datetime using naive method
        self.traveler_df['Arrival'] = pd.to_datetime(self.traveler_df['Arrival'])
        if hasattr(self.traveler_df['Arrival'].dtype, 'tz') and self.traveler_df['Arrival'].dtype.tz is not None:
            self.traveler_df['Arrival'] = self.traveler_df['Arrival'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        if self.ohlc_df is not None:
            self.ohlc_df['time'] = pd.to_datetime(self.ohlc_df['time'], utc=True)
            self.ohlc_df['time'] = self.ohlc_df['time'].dt.tz_localize(None)
    
    def find_m_with_opposite_pairs(self, 
                                    output_spread_tolerance: float = 3.0,
                                    min_days_before: int = 1) -> pd.DataFrame:
        """
        Find patterns where ANY M# arrives with ANY opposite pair arriving before it
        
        Example: M# 0 arrives, with M# +40 and -40 having arrived earlier near same output
        
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
            
            # Find arrivals before this one in same feed
            earlier_df = self.traveler_df[
                (self.traveler_df['Feed'] == feed) &
                (self.traveler_df['Arrival'] < arrival_time)
            ].copy()
            
            # Filter to those within output spread
            earlier_near_df = earlier_df[
                abs(earlier_df['Output'] - output) <= output_spread_tolerance
            ].copy()
            
            if len(earlier_near_df) == 0:
                continue
            
            # Find all unique M# values that appear in earlier data
            unique_m_values = earlier_near_df['M #'].unique()
            
            # For each unique M# value, check if its opposite also exists
            for check_m in unique_m_values:
                if check_m == 0:  # M# 0 has no opposite
                    continue
                    
                opposite_m = -check_m
                
                # Find positive M#
                pos_m_df = earlier_near_df[earlier_near_df['M #'] == abs(check_m)]
                # Find negative M#
                neg_m_df = earlier_near_df[earlier_near_df['M #'] == -abs(check_m)]
                
                # If we found both opposites
                if len(pos_m_df) > 0 and len(neg_m_df) > 0:
                    # Get the most recent of each opposite
                    pos_latest = pos_m_df.nlargest(1, 'Arrival').iloc[0]
                    neg_latest = neg_m_df.nlargest(1, 'Arrival').iloc[0]
                    
                    # Calculate max output spread
                    outputs = [output, pos_latest['Output'], neg_latest['Output']]
                    max_spread = max(outputs) - min(outputs)
                    
                    # Determine family and origin characteristics
                    pair_m = abs(check_m)
                    all_same_family = (get_family(m_value) == get_family(pair_m) == get_family(-pair_m))
                    all_indigo = all([get_family(v).startswith('Indigo') for v in [m_value, pair_m, -pair_m]])
                    all_green = all([get_family(v).startswith('Green') for v in [m_value, pair_m, -pair_m]])
                    
                    all_x0_tags = all([get_tag_category(v).startswith('X0') for v in [m_value, pair_m, -pair_m]])
                    all_xd0_tags = all([get_tag_category(v).startswith('XD0') for v in [m_value, pair_m, -pair_m]])
                    
                    all_epic = all([is_epic_origin(o) for o in [origin, pos_latest['Origin'], neg_latest['Origin']]])
                    all_anchor = all([is_anchor_origin(o) for o in [origin, pos_latest['Origin'], neg_latest['Origin']]])
                    
                    # Check if this M# is from Wild family and pair is also Wild
                    day0_is_wild = m_value in FAMILY_WILD_TRAVELERS
                    pair_is_wild = pair_m in FAMILY_WILD_TRAVELERS
                    all_wild = day0_is_wild and pair_is_wild
                    
                    patterns.append({
                        'Report_Time': arrival_time,
                        'Feed': feed,
                        'M#_Arriving_Day0': m_value,
                        'Origin_Day0': origin,
                        'Output_Day0': output,
                        'Pair_M#_Positive': pair_m,
                        'Origin_Pair_Pos': pos_latest['Origin'],
                        'Output_Pair_Pos': pos_latest['Output'],
                        'Arrival_Pair_Pos': pos_latest['Arrival'],
                        'Days_Before_Pos': (arrival_time - pos_latest['Arrival']).days,
                        'Pair_M#_Negative': -pair_m,
                        'Origin_Pair_Neg': neg_latest['Origin'],
                        'Output_Pair_Neg': neg_latest['Output'],
                        'Arrival_Pair_Neg': neg_latest['Arrival'],
                        'Days_Before_Neg': (arrival_time - neg_latest['Arrival']).days,
                        'Max_Output_Spread': max_spread,
                        'Family_Day0': get_family(m_value),
                        'Family_Pair': get_family(pair_m),
                        'Tag_Day0': get_tag_category(m_value),
                        'Tag_Pair': get_tag_category(pair_m),
                        'All_Same_Family': all_same_family,
                        'All_Indigo': all_indigo,
                        'All_Green': all_green,
                        'All_Wild': all_wild,
                        'All_X0_Tags': all_x0_tags,
                        'All_XD0_Tags': all_xd0_tags,
                        'All_EPIC_Origins': all_epic,
                        'All_Anchor_Origins': all_anchor,
                        'Has_EPIC_Origin': is_epic_origin(origin),
                        'Avg_Output': np.mean(outputs),
                        'Output_Range': f"{min(outputs):.2f} to {max(outputs):.2f}"
                    })
        
        return pd.DataFrame(patterns)
