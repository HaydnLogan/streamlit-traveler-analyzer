"""
Haydn's Pattern Scanner v3
Comprehensive pattern detection including:
- All 23 existing models
- X0 Sequential Descents (Model 24)
- FOGZ presence detection
- Constellation patterns
- Indigo Wild pair detection
- Same-origin tag descent
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pattern_detector_v2 import get_family, get_tag_category
import sys
sys.path.append('/mnt/user-data/uploads')
from nested_swing_detector import analyze_swings

# Import model definitions
from model_definitions_v21 import (
    MODELS, FOGZ, LRG_D, MED_D, RECIP_PAIRS,
    get_reciprocal_lookup, apply_special_matching
)

# Pattern significance weights
WEIGHTS = {
    'x0_sequential_descent': 150,   # Model 24
    'fogz_presence': 120,            # FOGZ member at Day [0]
    'constellation': 100,            # Multi-M# cluster around anchor
    'wild_pair': 110,                # Both 0 and 40 present
    'same_origin_tag_descent': 90,  # Spain x03p → Spain x02d
    'epic_epic_same': 100,          # Tobago-Tobago, Trinidad-Trinidad
    'epic_epic_different': 50,      # Trinidad-Tobago
    'anchor_anchor_same': 40,       # Jupiter-Jupiter, etc.
    'downgrade': 30,
    'same_family': 25,
    'x0_alignment': 35,
    'large_m': 20,
    'very_large_m': 40,
    'flip_match': 20,
    'tight_spread': 50,
    'medium_spread': 25,
    'family_cluster': 15,
    'model_match': 40,              # Match from one of the 23 models
}

EPIC_ORIGINS = {'Trinidad', 'Tobago'}
ANCHOR_ORIGINS = {'Saturn', 'Jupiter', 'Spain', 'Kepler-44', 'Kepler-62'}
INDIGO_WILD = {0, -0, 40, -40}  # Both Wild members


class HaydnPatternScanner:
    def __init__(self, traveler_df: pd.DataFrame, ohlc_df: pd.DataFrame = None):
        """
        Initialize Haydn's comprehensive pattern scanner
        
        Args:
            traveler_df: Traveler report dataframe
            ohlc_df: OHLC price data for swing detection
        """
        self.traveler_df = traveler_df.copy()
        self.ohlc_df = ohlc_df
        
        # Add calculated columns
        self.traveler_df['Family_Calc'] = self.traveler_df['M #'].apply(get_family)
        self.traveler_df['Tag_Calc'] = self.traveler_df['M #'].apply(get_tag_category)
        
        # Parse datetimes
        self.traveler_df['Arrival'] = pd.to_datetime(self.traveler_df['Arrival'])
        
        # Build reciprocal lookup for reciprocal model checking
        self.recip_lookup = get_reciprocal_lookup()
    
    def detect_swing_zones(self, 
                          start_time: str,
                          end_time: str,
                          min_swing_size: float = 60) -> List[Dict]:
        """Detect key price zones using nested swing detector"""
        if self.ohlc_df is None:
            return []
        
        nested_swings, major_points = analyze_swings(
            self.ohlc_df,
            start_time=pd.to_datetime(start_time),
            end_time=pd.to_datetime(end_time),
            min_swing_size=min_swing_size,
            pullback_tolerance=30
        )
        
        zones = []
        for point in major_points:
            if point['is_reversal'] or point['is_major_extreme']:
                zones.append({
                    'price': point['price'],
                    'type': 'Reversal' if point['is_reversal'] else 'Major Extreme',
                    'subtype': point['type'],
                    'time': point['time'],
                    'significance': point['significance_score']
                })
        
        return zones
    
    def analyze_zone(self, 
                     center_price: float,
                     zone_width: float = 10.0,
                     match_tolerance: float = 1.0,
                     single_model: str = None,
                     # Progressive filtering parameters
                     filter_epic_origins: bool = False,
                     filter_same_origin: bool = False,
                     filter_by_prox: bool = False,
                     prox_threshold: float = None,
                     filter_day_zero: bool = False,
                     show_flip_matches: bool = False) -> Dict:
        """
        Comprehensive zone analysis including all pattern types
        
        Args:
            single_model: If provided, only analyze this specific model
            filter_epic_origins: If True, only include Trinidad/Tobago
            filter_same_origin: If True, only same origin pairs
            filter_by_prox: If True, filter by output spread
            prox_threshold: Max output spread for matches
            filter_day_zero: If True, only Day [0] arrivals
            show_flip_matches: If True, identify flip matches
        
        Returns:
            Dictionary with patterns, trigger analysis, and scoring
        """
        zone_df = self.traveler_df[
            (self.traveler_df['Output'] >= center_price - zone_width/2) &
            (self.traveler_df['Output'] <= center_price + zone_width/2)
        ].copy()
        
        if len(zone_df) == 0:
            return {'error': 'No arrivals in zone'}
        
        # If single model requested, only process that model with filters
        if single_model:
            model_matches, filtering_metrics = self._find_single_model_matches(
                zone_df, 
                single_model, 
                match_tolerance,
                filter_epic_origins=filter_epic_origins,
                filter_same_origin=filter_same_origin,
                filter_by_prox=filter_by_prox,
                prox_threshold=prox_threshold,
                filter_day_zero=filter_day_zero,
                show_flip_matches=show_flip_matches
            )
            
            return {
                'center_price': center_price,
                'zone_width': zone_width,
                'num_arrivals': len(zone_df),
                'single_model_name': single_model,
                'single_model_matches': model_matches,
                'filtering_metrics': filtering_metrics,  # NEW: Progressive filtering metrics
                'score': len(model_matches),  # Simple score based on match count
                'rank': None
            }
        
        # Detect all pattern types (full analysis)
        patterns = {
            'epic_same_origin': self._find_epic_same_origin(zone_df, match_tolerance),
            'epic_epic_pairs': self._find_epic_epic_pairs(zone_df, match_tolerance),
            'downgrades': self._find_downgrades(zone_df, match_tolerance),
            'x0_alignments': self._find_x0_alignments(zone_df, match_tolerance),
            'x0_sequential_descents': self._find_x0_sequential_descents(zone_df, match_tolerance),
            'large_m_presence': self._find_large_m(zone_df),
            'family_clusters': self._find_family_clusters(zone_df),
            'flip_matches': self._find_flip_matches(zone_df, match_tolerance),
            
            # NEW PATTERNS
            'fogz_presence': self._find_fogz_presence(zone_df),
            'constellations': self._find_constellations(zone_df, match_tolerance),
            'wild_pairs': self._find_wild_pairs(zone_df, match_tolerance),
            'same_origin_tag_descents': self._find_same_origin_tag_descents(zone_df, match_tolerance),
            'model_matches': self._find_model_matches(zone_df, match_tolerance),
        }
        
        # CRITICAL: Identify trigger patterns (patterns unique to this zone)
        trigger_patterns = self._identify_trigger_patterns(
            patterns, 
            center_price, 
            zone_width, 
            match_tolerance
        )
        
        score = self._calculate_zone_score(patterns)
        
        return {
            'center_price': center_price,
            'zone_width': zone_width,
            'num_arrivals': len(zone_df),
            'patterns': patterns,
            'trigger_patterns': trigger_patterns,  # NEW: Identifies actual triggers
            'score': score,
            'rank': None
        }
    
    def _find_single_model_matches(self, 
                                   zone_df: pd.DataFrame, 
                                   model_name: str, 
                                   tolerance: float,
                                   filter_epic_origins: bool = False,
                                   filter_same_origin: bool = False,
                                   filter_by_prox: bool = False,
                                   prox_threshold: float = None,
                                   filter_day_zero: bool = False,
                                   show_flip_matches: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Find matches for a single specific model with progressive filtering
        
        Returns:
            Tuple of (matches, filtering_metrics)
        """
        from model_definitions_v21 import MODELS, get_reciprocal_lookup, apply_special_matching
        
        if model_name not in MODELS:
            return [], {}
        
        model = MODELS[model_name]
        
        # Get pass1 and pass2 M# values
        pass1_values = set(model['pass1'])
        pass2_values = set(model['pass2'])
        
        # Filter zone_df for pass1 and pass2 values
        pass1_df = zone_df[zone_df['M #'].isin(pass1_values)].copy()
        pass2_df = zone_df[zone_df['M #'].isin(pass2_values)].copy()
        
        if len(pass1_df) == 0 or len(pass2_df) == 0:
            return [], {}
        
        # Find all matches within tolerance
        matches = []
        for _, p1_row in pass1_df.iterrows():
            for _, p2_row in pass2_df.iterrows():
                # Skip if same arrival
                if p1_row.name == p2_row.name:
                    continue
                
                # Check output spread
                spread = abs(p1_row['Output'] - p2_row['Output'])
                if spread > tolerance:
                    continue
                
                # Apply special matching rules if needed
                if model.get('special_matching'):
                    recip_lookup = get_reciprocal_lookup()
                    if not apply_special_matching(
                        p1_row['M #'], 
                        p2_row['M #'],
                        model['special_matching'],
                        recip_lookup
                    ):
                        continue
                
                # Check reciprocal requirement if needed
                if model.get('check_recip'):
                    m1, m2 = p1_row['M #'], p2_row['M #']
                    if not ((m1 > 0 and m2 < 0) or (m1 < 0 and m2 > 0)):
                        continue
                
                # Determine which arrival is more recent (for flip detection)
                day1 = p1_row.get('Day', 0) if 'Day' in p1_row else 0
                day2 = p2_row.get('Day', 0) if 'Day' in p2_row else 0
                
                # Most recent has lower Day value (0 is today, 1 is yesterday, etc.)
                if day1 <= day2:
                    more_recent_m = p1_row['M #']
                    less_recent_m = p2_row['M #']
                else:
                    more_recent_m = p2_row['M #']
                    less_recent_m = p1_row['M #']
                
                # Determine if flip match
                m1, m2 = p1_row['M #'], p2_row['M #']
                is_flip = (m1 > 0 and m2 < 0) or (m1 < 0 and m2 > 0)
                
                # Create match in swing tool format
                match = {
                    'M1': p1_row['M #'],
                    'Origin1': p1_row['Origin'],
                    'Output1': p1_row['Output'],
                    'Arrival1': pd.to_datetime(p1_row['Arrival']).strftime('%Y-%m-%d %H:%M:%S'),
                    'Day1': day1,
                    'M2': p2_row['M #'],
                    'Origin2': p2_row['Origin'],
                    'Output2': p2_row['Output'],
                    'Arrival2': pd.to_datetime(p2_row['Arrival']).strftime('%Y-%m-%d %H:%M:%S'),
                    'Day2': day2,
                    'Output_Spread': spread,
                    'Prox': spread,  # Prox is same as Output_Spread
                    'Match_Type': 'Reciprocal' if model.get('check_recip') else 'Standard',
                    'Is_Flip': is_flip,
                    'More_Recent_M': more_recent_m,
                    'Arrival_Output': (p1_row['Output'] + p2_row['Output']) / 2  # Average
                }
                
                # Add feed info if available
                if 'Feed' in p1_row:
                    match['Feed1'] = p1_row['Feed']
                if 'Feed' in p2_row:
                    match['Feed2'] = p2_row['Feed']
                
                # Determine primary feed (where most recent arrival is)
                if day1 <= day2 and 'Feed' in p1_row:
                    match['Feed'] = p1_row['Feed']
                elif day2 < day1 and 'Feed' in p2_row:
                    match['Feed'] = p2_row['Feed']
                elif 'Feed' in p1_row:
                    match['Feed'] = p1_row['Feed']
                else:
                    match['Feed'] = 'Unknown'
                
                matches.append(match)
        
        # PROGRESSIVE FILTERING - Track metrics at each step
        filtering_metrics = {
            'step0_initial': {
                'count': len(matches),
                'output_spread': self._calc_output_spread(matches),
                'unique_outputs': self._count_unique_outputs(matches),
                'description': 'Initial matches (within zone + tolerance)'
            }
        }
        
        # Filter 1: Day [0] only
        if filter_day_zero:
            matches = [m for m in matches if m['Day1'] == 0 and m['Day2'] == 0]
            filtering_metrics['step1_day_zero'] = {
                'count': len(matches),
                'output_spread': self._calc_output_spread(matches),
                'unique_outputs': self._count_unique_outputs(matches),
                'description': 'Day [0] arrivals only (Today)'
            }
        
        # Filter 2: Epic origins (Trinidad/Tobago)
        if filter_epic_origins:
            epic_origins = {'Trinidad', 'Tobago', 'trinidad', 'tobago'}
            matches = [m for m in matches 
                      if m['Origin1'] in epic_origins or m['Origin2'] in epic_origins]
            step_key = f'step{2 if filter_day_zero else 1}_epic_origins'
            filtering_metrics[step_key] = {
                'count': len(matches),
                'output_spread': self._calc_output_spread(matches),
                'unique_outputs': self._count_unique_outputs(matches),
                'description': 'Epic origins (Trinidad/Tobago) only'
            }
        
        # Filter 3: Same origin pairs
        if filter_same_origin:
            matches = [m for m in matches 
                      if m['Origin1'].lower() == m['Origin2'].lower()]
            step_num = sum([filter_day_zero, filter_epic_origins, True])
            step_key = f'step{step_num}_same_origin'
            filtering_metrics[step_key] = {
                'count': len(matches),
                'output_spread': self._calc_output_spread(matches),
                'unique_outputs': self._count_unique_outputs(matches),
                'description': 'Same origin pairs (Trinidad+Trinidad or Tobago+Tobago)'
            }
        
        # Filter 4: Prox threshold
        if filter_by_prox and prox_threshold is not None:
            matches = [m for m in matches if m['Prox'] < prox_threshold]
            step_num = sum([filter_day_zero, filter_epic_origins, filter_same_origin, True])
            step_key = f'step{step_num}_prox'
            filtering_metrics[step_key] = {
                'count': len(matches),
                'output_spread': self._calc_output_spread(matches),
                'unique_outputs': self._count_unique_outputs(matches),
                'description': f'Prox < {prox_threshold} points',
                'prox_threshold': prox_threshold
            }
        
        # Calculate rarity if flip matches requested
        if show_flip_matches:
            flip_matches = [m for m in matches if m['Is_Flip']]
            filtering_metrics['flip_analysis'] = {
                'total_matches': len(matches),
                'flip_matches': len(flip_matches),
                'flip_percentage': (len(flip_matches) / len(matches) * 100) if len(matches) > 0 else 0,
                'rarity_ratio': f"{len(flip_matches)} out of {filtering_metrics['step0_initial']['count']}"
            }
        
        return matches, filtering_metrics
    
    def _calc_output_spread(self, matches: List[Dict]) -> float:
        """Calculate output spread (max - min arrival output)"""
        if not matches:
            return 0.0
        outputs = [m['Arrival_Output'] for m in matches]
        return max(outputs) - min(outputs)
    
    def _count_unique_outputs(self, matches: List[Dict]) -> int:
        """Count unique arrival output locations"""
        if not matches:
            return 0
        outputs = [m['Arrival_Output'] for m in matches]
        # Round to 2 decimals to count "unique" locations
        unique = len(set(round(o, 2) for o in outputs))
        return unique
    
    # ========================================================================
    # EXISTING PATTERN DETECTORS (from v2)
    # ========================================================================
    
    def _find_epic_same_origin(self, zone_df: pd.DataFrame, tolerance: float) -> List[Dict]:
        """Find Tobago-Tobago or Trinidad-Trinidad matches"""
        matches = []
        for origin in EPIC_ORIGINS:
            origin_df = zone_df[zone_df['Origin'] == origin]
            for idx1, row1 in origin_df.iterrows():
                for idx2, row2 in origin_df.iterrows():
                    if idx1 >= idx2 or row1['M #'] == row2['M #']:
                        continue
                    spread = abs(row1['Output'] - row2['Output'])
                    if spread <= tolerance and row1['Feed'] == row2['Feed']:
                        matches.append({
                            'type': f'{origin}-{origin}',
                            'origin1': origin,
                            'origin2': origin,
                            'm1': row1['M #'],
                            'm2': row2['M #'],
                            'output1': row1['Output'],
                            'output2': row2['Output'],
                            'spread': spread,
                            'feed': row1['Feed'],
                            'family1': row1['Family_Calc'],
                            'family2': row2['Family_Calc'],
                            'flip_type': self._get_flip_type(row1['M #'], row2['M #']),
                            'is_downgrade': abs(row1['M #']) > abs(row2['M #']),
                            'day1': row1['Day'],
                            'day2': row2['Day'],
                            'significance': 'RARE - Same Epic Origin'
                        })
        return matches
    
    def _find_epic_epic_pairs(self, zone_df: pd.DataFrame, tolerance: float) -> List[Dict]:
        """Find Trinidad-Tobago matches"""
        matches = []
        trinidad_df = zone_df[zone_df['Origin'] == 'Trinidad']
        tobago_df = zone_df[zone_df['Origin'] == 'Tobago']
        
        for _, r1 in trinidad_df.iterrows():
            for _, r2 in tobago_df.iterrows():
                spread = abs(r1['Output'] - r2['Output'])
                if spread <= tolerance and r1['Feed'] == r2['Feed']:
                    matches.append({
                        'type': 'Trinidad-Tobago',
                        'origin1': 'Trinidad',
                        'origin2': 'Tobago',
                        'm1': r1['M #'],
                        'm2': r2['M #'],
                        'output1': r1['Output'],
                        'output2': r2['Output'],
                        'spread': spread,
                        'feed': r1['Feed'],
                        'family1': r1['Family_Calc'],
                        'family2': r2['Family_Calc'],
                        'flip_type': self._get_flip_type(r1['M #'], r2['M #']),
                        'is_downgrade': abs(r1['M #']) > abs(r2['M #']),
                        'day1': r1['Day'],
                        'day2': r2['Day'],
                        'significance': 'TT Match - Epic Origins'
                    })
        return matches
    
    def _find_downgrades(self, zone_df: pd.DataFrame, tolerance: float) -> List[Dict]:
        """Find downgrade patterns"""
        downgrades = []
        for idx1, row1 in zone_df.iterrows():
            for idx2, row2 in zone_df.iterrows():
                if idx1 >= idx2 or abs(row1['M #']) <= abs(row2['M #']):
                    continue
                spread = abs(row1['Output'] - row2['Output'])
                if spread <= tolerance and row1['Feed'] == row2['Feed']:
                    downgrades.append({
                        'origin1': row1['Origin'],
                        'origin2': row2['Origin'],
                        'm_large': row1['M #'],
                        'm_small': row2['M #'],
                        'differential': abs(row1['M #']) - abs(row2['M #']),
                        'output1': row1['Output'],
                        'output2': row2['Output'],
                        'spread': spread,
                        'feed': row1['Feed'],
                        'family1': row1['Family_Calc'],
                        'family2': row2['Family_Calc'],
                        'same_family': row1['Family_Calc'] == row2['Family_Calc'],
                        'flip_type': self._get_flip_type(row1['M #'], row2['M #']),
                        'day1': row1['Day'],
                        'day2': row2['Day']
                    })
        return downgrades
    
    def _find_x0_alignments(self, zone_df: pd.DataFrame, tolerance: float) -> List[Dict]:
        """Find X0 tag alignment patterns"""
        alignments = []
        x0_df = zone_df[zone_df['Tag_Calc'].str.startswith('X0', na=False)]
        
        for idx1, row1 in x0_df.iterrows():
            for idx2, row2 in x0_df.iterrows():
                if idx1 >= idx2:
                    continue
                spread = abs(row1['Output'] - row2['Output'])
                if spread <= tolerance and row1['Feed'] == row2['Feed']:
                    alignments.append({
                        'origin1': row1['Origin'],
                        'origin2': row2['Origin'],
                        'm1': row1['M #'],
                        'm2': row2['M #'],
                        'tag1': row1['Tag_Calc'],
                        'tag2': row2['Tag_Calc'],
                        'same_x0_type': row1['Tag_Calc'] == row2['Tag_Calc'],
                        'output1': row1['Output'],
                        'output2': row2['Output'],
                        'spread': spread,
                        'feed': row1['Feed'],
                        'is_downgrade': abs(row1['M #']) > abs(row2['M #']),
                        'day1': row1['Day'],
                        'day2': row2['Day']
                    })
        return alignments
    
    def _find_x0_sequential_descents(self, zone_df: pd.DataFrame, tolerance: float) -> List[Dict]:
        """Find X0 sequential descent patterns (Model 24)"""
        sequences = []
        
        for feed in zone_df['Feed'].unique():
            feed_df = zone_df[zone_df['Feed'] == feed].copy()
            day0_df = feed_df[feed_df['Day'] == '[0]']
            
            for _, anchor_row in day0_df.iterrows():
                anchor_output = anchor_row['Output']
                anchor_time = anchor_row['Arrival']
                anchor_m = anchor_row['M #']
                
                nearby_x0 = feed_df[
                    (abs(feed_df['Output'] - anchor_output) <= tolerance) &
                    (feed_df['Arrival'] <= anchor_time) &
                    (feed_df['Tag_Calc'].str.startswith('X0', na=False))
                ].copy()
                
                if len(nearby_x0) < 2:
                    continue
                
                nearby_x0 = nearby_x0.sort_values('Arrival')
                m_values = nearby_x0['M #'].tolist()
                abs_values = [abs(m) for m in m_values]
                
                is_descending = all(abs_values[i] >= abs_values[i+1] for i in range(len(abs_values)-1))
                crosses_zero = any(m_values[i] > 0 and m_values[j] < 0 
                                  for i in range(len(m_values)) 
                                  for j in range(i+1, len(m_values)))
                
                x0p_only = nearby_x0[nearby_x0['Tag_Calc'] == 'X0p']
                x0p_count = len(x0p_only)
                x0p_descending = False
                if x0p_count >= 2:
                    x0p_abs = [abs(m) for m in x0p_only['M #'].tolist()]
                    x0p_descending = all(x0p_abs[i] >= x0p_abs[i+1] for i in range(len(x0p_abs)-1))
                
                if (x0p_count >= 3 and x0p_descending) or (crosses_zero and is_descending):
                    sequence_length = len(nearby_x0)
                    max_m = max(abs_values)
                    min_m = min(abs_values)
                    spread_range = max(nearby_x0['Output']) - min(nearby_x0['Output'])
                    
                    sequences.append({
                        'anchor_m': anchor_m,
                        'anchor_origin': anchor_row['Origin'],
                        'anchor_output': anchor_output,
                        'anchor_time': anchor_time,
                        'feed': feed,
                        'sequence_length': sequence_length,
                        'x0p_count': x0p_count,
                        'is_descending': is_descending,
                        'x0p_descending': x0p_descending,
                        'crosses_zero': crosses_zero,
                        'max_m': max_m,
                        'min_m': min_m,
                        'output_spread': spread_range,
                        'sequence': ' → '.join([f'{m:+.0f}' for m in m_values]),
                        'x0p_sequence': ' → '.join([f'{m:+.0f}' for m in x0p_only['M #'].tolist()]) if x0p_count > 0 else '',
                        'm_values': m_values,
                        'pattern_type': self._classify_x0_sequence(m_values, x0p_count, x0p_descending, crosses_zero)
                    })
        
        return sequences
    
    def _classify_x0_sequence(self, m_values: List[float], x0p_count: int, 
                             x0p_descending: bool, crosses_zero: bool) -> str:
        """Classify X0 sequential pattern type"""
        if x0p_count >= 3 and x0p_descending:
            if crosses_zero:
                return "X0p Countdown with Flip"
            else:
                return "X0p Countdown"
        elif crosses_zero:
            return "Number Line Sweep"
        else:
            return "X0 Sequence"
    
    def _find_large_m(self, zone_df: pd.DataFrame) -> List[Dict]:
        """Find large M# presence (80+)"""
        large_m = []
        for _, row in zone_df.iterrows():
            if abs(row['M #']) >= 80:
                large_m.append({
                    'origin': row['Origin'],
                    'm': row['M #'],
                    'output': row['Output'],
                    'feed': row['Feed'],
                    'family': row['Family_Calc'],
                    'day': row['Day'],
                    'size': abs(row['M #'])
                })
        return large_m
    
    def _find_family_clusters(self, zone_df: pd.DataFrame) -> Dict:
        """Find family clusters"""
        clusters = {}
        for family in zone_df['Family_Calc'].unique():
            if pd.isna(family):
                continue
            family_df = zone_df[zone_df['Family_Calc'] == family]
            unique_m = family_df['M #'].nunique()
            
            if unique_m >= 2:
                clusters[family] = {
                    'count': len(family_df),
                    'unique_m_values': unique_m,
                    'members': []
                }
                for m_val in family_df['M #'].unique():
                    m_rows = family_df[family_df['M #'] == m_val]
                    clusters[family]['members'].append({
                        'm': m_val,
                        'count': len(m_rows),
                        'origins': m_rows['Origin'].unique().tolist()
                    })
        return clusters
    
    def _find_flip_matches(self, zone_df: pd.DataFrame, tolerance: float) -> List[Dict]:
        """Find flip patterns (PD, DD, PP, DP)"""
        flips = []
        for idx1, row1 in zone_df.iterrows():
            for idx2, row2 in zone_df.iterrows():
                if idx1 >= idx2:
                    continue
                spread = abs(row1['Output'] - row2['Output'])
                if spread <= tolerance and row1['Feed'] == row2['Feed']:
                    flip_type = self._get_flip_type(row1['M #'], row2['M #'])
                    if flip_type in ['PD', 'DD', 'PP', 'DP']:
                        flips.append({
                            'type': flip_type,
                            'origin1': row1['Origin'],
                            'origin2': row2['Origin'],
                            'm1': row1['M #'],
                            'm2': row2['M #'],
                            'output1': row1['Output'],
                            'output2': row2['Output'],
                            'spread': spread,
                            'feed': row1['Feed'],
                            'family1': row1['Family_Calc'],
                            'family2': row2['Family_Calc'],
                            'day1': row1['Day'],
                            'day2': row2['Day']
                        })
        return flips
    
    # ========================================================================
    # NEW PATTERN DETECTORS
    # ========================================================================
    
    def _find_fogz_presence(self, zone_df: pd.DataFrame) -> List[Dict]:
        """
        Find FOGZ members (M# +6 to -6) at Day [0]
        FOGZ = {0, 1, -1, 2, -2, 3, -3, 5, -5, 6, -6}
        """
        fogz_arrivals = []
        day0_df = zone_df[zone_df['Day'] == '[0]']
        
        for _, row in day0_df.iterrows():
            if row['M #'] in FOGZ:
                fogz_arrivals.append({
                    'm': row['M #'],
                    'origin': row['Origin'],
                    'output': row['Output'],
                    'feed': row['Feed'],
                    'family': row['Family_Calc'],
                    'is_zero': row['M #'] == 0,
                    'is_epic': row['Origin'] in EPIC_ORIGINS
                })
        return fogz_arrivals
    
    def _find_constellations(self, zone_df: pd.DataFrame, tolerance: float) -> List[Dict]:
        """
        Find constellation patterns: anchor M# with multiple M#s clustered around it.
        Particularly significant when:
        - Multiple Epic origins present
        - Both Indigo Wild members (0 and 40) appear
        - Multiple same-family M#s cluster
        """
        constellations = []
        
        # Look for potential anchors (Indigo Blue M#s from Anchor origins)
        indigo_blue_df = zone_df[zone_df['Family_Calc'] == 'Indigo Blu']
        anchor_df = indigo_blue_df[indigo_blue_df['Origin'].isin(ANCHOR_ORIGINS)]
        
        for _, anchor in anchor_df.iterrows():
            anchor_output = anchor['Output']
            
            # Find all M#s within tolerance
            nearby = zone_df[
                (abs(zone_df['Output'] - anchor_output) <= tolerance * 1.5) &
                (zone_df['Feed'] == anchor['Feed'])
            ]
            
            if len(nearby) < 3:  # Need at least 3 M#s including anchor
                continue
            
            # Check for significant patterns
            epic_count = len(nearby[nearby['Origin'].isin(EPIC_ORIGINS)])
            wild_members = nearby[nearby['M #'].isin(INDIGO_WILD)]
            has_both_wild = len(wild_members['M #'].unique()) >= 2  # Has both 0 and 40
            
            # Count family members
            same_family = nearby[nearby['Family_Calc'] == anchor['Family_Calc']]
            
            if epic_count >= 2 or has_both_wild or len(same_family) >= 3:
                constellations.append({
                    'anchor_m': anchor['M #'],
                    'anchor_origin': anchor['Origin'],
                    'anchor_output': anchor_output,
                    'anchor_family': anchor['Family_Calc'],
                    'feed': anchor['Feed'],
                    'member_count': len(nearby),
                    'epic_count': epic_count,
                    'has_both_wild': has_both_wild,
                    'same_family_count': len(same_family),
                    'members': [
                        {
                            'm': row['M #'],
                            'origin': row['Origin'],
                            'output': row['Output'],
                            'family': row['Family_Calc']
                        }
                        for _, row in nearby.iterrows()
                    ]
                })
        
        return constellations
    
    def _find_wild_pairs(self, zone_df: pd.DataFrame, tolerance: float) -> List[Dict]:
        """
        Find when both Indigo Wild members (0 and 40) appear together.
        This is significant for turning points.
        """
        wild_pairs = []
        
        for feed in zone_df['Feed'].unique():
            feed_df = zone_df[zone_df['Feed'] == feed]
            wild_df = feed_df[feed_df['M #'].isin(INDIGO_WILD)]
            
            # Look for M# 0
            zero_df = wild_df[wild_df['M #'] == 0]
            
            for _, zero_row in zero_df.iterrows():
                # Find M# 40 or -40 nearby
                forty_df = wild_df[
                    (wild_df['M #'].isin([40, -40])) &
                    (abs(wild_df['Output'] - zero_row['Output']) <= tolerance)
                ]
                
                for _, forty_row in forty_df.iterrows():
                    wild_pairs.append({
                        'zero_origin': zero_row['Origin'],
                        'forty_origin': forty_row['Origin'],
                        'forty_m': forty_row['M #'],
                        'zero_output': zero_row['Output'],
                        'forty_output': forty_row['Output'],
                        'spread': abs(zero_row['Output'] - forty_row['Output']),
                        'feed': feed,
                        'both_epic': (zero_row['Origin'] in EPIC_ORIGINS and 
                                     forty_row['Origin'] in EPIC_ORIGINS)
                    })
        
        return wild_pairs
    
    def _find_same_origin_tag_descents(self, zone_df: pd.DataFrame, tolerance: float) -> List[Dict]:
        """
        Find same-origin tag descent patterns.
        Example: Spain M# 68 (X0p/x03p) → Spain M# -22 (X0d/x02d)
        """
        descents = []
        
        for origin in zone_df['Origin'].unique():
            origin_df = zone_df[zone_df['Origin'] == origin]
            
            # Get X0p and X0d M#s
            x0p_df = origin_df[origin_df['Tag_Calc'] == 'X0p']
            x0d_df = origin_df[origin_df['Tag_Calc'] == 'X0d']
            
            for _, r1 in x0p_df.iterrows():
                for _, r2 in x0d_df.iterrows():
                    spread = abs(r1['Output'] - r2['Output'])
                    if spread <= tolerance and r1['Feed'] == r2['Feed']:
                        descents.append({
                            'origin': origin,
                            'm_x0p': r1['M #'],
                            'm_x0d': r2['M #'],
                            'output_x0p': r1['Output'],
                            'output_x0d': r2['Output'],
                            'spread': spread,
                            'feed': r1['Feed'],
                            'tag_x0p': r1['Tag'],
                            'tag_x0d': r2['Tag'],
                            'descent_type': f"{r1['Tag']} → {r2['Tag']}"
                        })
        
        return descents
    
    def _find_model_matches(self, zone_df: pd.DataFrame, tolerance: float) -> Dict:
        """
        Find matches for all 23 existing models.
        Returns dict with model names as keys and match lists as values.
        """
        model_matches = {}
        
        for model_name, model_config in MODELS.items():
            matches = []
            
            # Get Pass 1 and Pass 2 M#s
            pass1_m = model_config['pass1']
            pass2_m = model_config['pass2']
            
            # Filter zone to only those M#s
            pass1_df = zone_df[zone_df['M #'].isin(pass1_m)]
            pass2_df = zone_df[zone_df['M #'].isin(pass2_m)]
            
            # Look for matches
            for _, r1 in pass1_df.iterrows():
                for _, r2 in pass2_df.iterrows():
                    if r1.name == r2.name:  # Skip same row
                        continue
                    
                    spread = abs(r1['Output'] - r2['Output'])
                    if spread <= tolerance and r1['Feed'] == r2['Feed']:
                        
                        # Check reciprocal if needed
                        if model_config['check_recip']:
                            m1_int = int(r1['M #'])
                            m2_int = int(r2['M #'])
                            if (m1_int, m2_int) not in RECIP_PAIRS:
                                continue
                        
                        # Check special matching if needed
                        if model_config.get('special_matching'):
                            if not apply_special_matching(model_name, r1['M #'], r2['M #']):
                                continue
                        
                        matches.append({
                            'model': model_name,
                            'model_number': model_config['number'],
                            'm1': r1['M #'],
                            'm2': r2['M #'],
                            'origin1': r1['Origin'],
                            'origin2': r2['Origin'],
                            'output1': r1['Output'],
                            'output2': r2['Output'],
                            'spread': spread,
                            'feed': r1['Feed'],
                            'day1': r1['Day'],
                            'day2': r2['Day']
                        })
            
            model_matches[model_name] = matches
        
        return model_matches
    
    # ========================================================================
    # TRIGGER PATTERN DETECTION
    # ========================================================================
    
    def _identify_trigger_patterns(self, 
                                   patterns: Dict,
                                   center_price: float,
                                   zone_width: float,
                                   match_tolerance: float) -> Dict:
        """
        Identify which patterns are likely triggers for the turn.
        
        A pattern is a likely trigger if it appears IN the zone but NOT commonly
        outside the zone. Patterns that appear everywhere are less significant.
        
        Args:
            patterns: Detected patterns in the zone
            center_price: Zone center price
            zone_width: Zone width
            match_tolerance: Match tolerance used
            
        Returns:
            Dict with trigger likelihood for each pattern type
        """
        
        # Define the zone boundaries
        zone_min = center_price - zone_width/2
        zone_max = center_price + zone_width/2
        
        # Get data outside the zone for comparison
        outside_zone_df = self.traveler_df[
            (self.traveler_df['Output'] < zone_min - zone_width) |
            (self.traveler_df['Output'] > zone_max + zone_width)
        ].copy()
        
        if len(outside_zone_df) == 0:
            # If no data outside, all patterns are potential triggers
            return {
                'all_patterns_unique': True,
                'trigger_likelihood': 'HIGH',
                'reason': 'No similar patterns found outside zone'
            }
        
        trigger_analysis = {
            'pattern_specificity': {},
            'trigger_candidates': [],
            'common_patterns': [],
            'overall_trigger_likelihood': 0.0
        }
        
        # Check each pattern type
        pattern_checks = {
            'epic_same_origin': self._check_epic_outside_zone,
            'fogz_presence': self._check_fogz_outside_zone,
            'wild_pairs': self._check_wild_pairs_outside_zone,
            'constellations': self._check_constellations_outside_zone,
        }
        
        total_specificity = 0
        pattern_count = 0
        
        for pattern_type, check_func in pattern_checks.items():
            pattern_list = patterns.get(pattern_type, [])
            
            if pattern_list and len(pattern_list) > 0:
                # Check if this pattern exists outside the zone
                appears_outside = check_func(outside_zone_df, pattern_list, match_tolerance)
                
                # Calculate specificity (0 = common everywhere, 1 = unique to zone)
                if appears_outside:
                    specificity = 0.2  # Low specificity - appears elsewhere
                    trigger_analysis['common_patterns'].append(pattern_type)
                else:
                    specificity = 1.0  # High specificity - unique to zone
                    trigger_analysis['trigger_candidates'].append({
                        'pattern_type': pattern_type,
                        'count': len(pattern_list),
                        'specificity': specificity,
                        'reason': 'Unique to this zone'
                    })
                
                trigger_analysis['pattern_specificity'][pattern_type] = specificity
                total_specificity += specificity
                pattern_count += 1
        
        # Calculate overall trigger likelihood
        if pattern_count > 0:
            avg_specificity = total_specificity / pattern_count
            trigger_analysis['overall_trigger_likelihood'] = avg_specificity
            
            if avg_specificity >= 0.8:
                trigger_analysis['trigger_strength'] = 'STRONG'
            elif avg_specificity >= 0.5:
                trigger_analysis['trigger_strength'] = 'MODERATE'
            else:
                trigger_analysis['trigger_strength'] = 'WEAK'
        else:
            trigger_analysis['trigger_strength'] = 'UNKNOWN'
            trigger_analysis['overall_trigger_likelihood'] = 0.0
        
        return trigger_analysis
    
    def _check_epic_outside_zone(self, outside_df: pd.DataFrame, 
                                  patterns: List[Dict], 
                                  tolerance: float) -> bool:
        """Check if epic same origin patterns exist outside zone"""
        if len(patterns) == 0:
            return False
        
        # Look for any epic same origin matches outside the zone
        epic_outside = self._find_epic_same_origin(outside_df, tolerance)
        return len(epic_outside) > 0
    
    def _check_fogz_outside_zone(self, outside_df: pd.DataFrame,
                                  patterns: List[Dict],
                                  tolerance: float) -> bool:
        """Check if FOGZ presence exists outside zone"""
        if len(patterns) == 0:
            return False
        
        # Look for FOGZ members outside the zone
        fogz_outside = self._find_fogz_presence(outside_df)
        return len(fogz_outside) > 0
    
    def _check_wild_pairs_outside_zone(self, outside_df: pd.DataFrame,
                                        patterns: List[Dict],
                                        tolerance: float) -> bool:
        """Check if wild pairs exist outside zone"""
        if len(patterns) == 0:
            return False
        
        wild_outside = self._find_wild_pairs(outside_df, tolerance)
        return len(wild_outside) > 0
    
    def _check_constellations_outside_zone(self, outside_df: pd.DataFrame,
                                            patterns: List[Dict],
                                            tolerance: float) -> bool:
        """Check if constellations exist outside zone"""
        if len(patterns) == 0:
            return False
        
        const_outside = self._find_constellations(outside_df, tolerance)
        return len(const_outside) > 0
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _get_flip_type(self, m1: float, m2: float) -> str:
        """Determine flip type"""
        if m1 > 0 and m2 < 0:
            return 'PD'
        elif m1 < 0 and m2 > 0:
            return 'DP'
        elif m1 > 0 and m2 > 0:
            return 'PP'
        elif m1 < 0 and m2 < 0:
            return 'DD'
        return 'Unknown'
    
    def _calculate_zone_score(self, patterns: Dict) -> float:
        """Calculate comprehensive zone score"""
        score = 0
        
        # Epic same origin
        score += len(patterns.get('epic_same_origin', [])) * WEIGHTS['epic_epic_same']
        
        # Epic-epic pairs
        score += len(patterns.get('epic_epic_pairs', [])) * WEIGHTS['epic_epic_different']
        
        # Downgrades
        for dg in patterns.get('downgrades', []):
            score += WEIGHTS['downgrade']
            if dg.get('same_family'):
                score += WEIGHTS['same_family']
            if dg.get('spread', 999) < 0.5:
                score += WEIGHTS['tight_spread']
            elif dg.get('spread', 999) < 1.0:
                score += WEIGHTS['medium_spread']
        
        # X0 alignments
        for xa in patterns.get('x0_alignments', []):
            score += WEIGHTS['x0_alignment']
            if xa.get('same_x0_type'):
                score += WEIGHTS['x0_alignment'] * 0.5
        
        # X0 Sequential Descents (Model 24)
        for seq in patterns.get('x0_sequential_descents', []):
            base_score = WEIGHTS['x0_sequential_descent']
            if seq['x0p_count'] >= 3 and seq['x0p_descending']:
                base_score *= 1.5
            if seq['crosses_zero']:
                base_score *= 1.3
            if seq['sequence_length'] >= 5:
                base_score *= 1.2
            score += base_score
        
        # Large M# presence
        for lm in patterns.get('large_m_presence', []):
            if lm['size'] >= 100:
                score += WEIGHTS['very_large_m']
            else:
                score += WEIGHTS['large_m']
        
        # Family clusters
        for family, data in patterns.get('family_clusters', {}).items():
            score += data['unique_m_values'] * WEIGHTS['family_cluster']
        
        # Flip matches
        score += len(patterns.get('flip_matches', [])) * WEIGHTS['flip_match']
        
        # FOGZ presence
        for fogz in patterns.get('fogz_presence', []):
            base = WEIGHTS['fogz_presence']
            if fogz['is_zero']:
                base *= 1.5  # M# 0 is extra significant
            if fogz['is_epic']:
                base *= 1.3  # Epic origins are extra significant
            score += base
        
        # Constellations
        for const in patterns.get('constellations', []):
            base = WEIGHTS['constellation']
            if const['has_both_wild']:
                base *= 1.8  # Both Wild members = very strong
            if const['epic_count'] >= 3:
                base *= 1.5
            score += base
        
        # Wild pairs
        for wp in patterns.get('wild_pairs', []):
            base = WEIGHTS['wild_pair']
            if wp['both_epic']:
                base *= 1.5
            score += base
        
        # Same origin tag descents
        score += len(patterns.get('same_origin_tag_descents', [])) * WEIGHTS['same_origin_tag_descent']
        
        # Model matches (from the 23 models)
        model_matches = patterns.get('model_matches', {})
        for model_name, matches in model_matches.items():
            if len(matches) > 0:
                # FOGZ models are more significant
                model_num = MODELS[model_name]['number']
                if model_num <= 3:  # FOGZ models
                    score += len(matches) * WEIGHTS['model_match'] * 1.5
                elif model_num <= 8:  # Large Disc or Reciprocal
                    score += len(matches) * WEIGHTS['model_match'] * 1.3
                else:
                    score += len(matches) * WEIGHTS['model_match']
        
        return score
    
    def scan_all_zones(self, 
                       start_time: str,
                       end_time: str,
                       min_swing_size: float = 60,
                       zone_width: float = 10.0,
                       match_tolerance: float = 1.0,
                       single_model: str = None,
                       # Progressive filtering parameters
                       filter_epic_origins: bool = False,
                       filter_same_origin: bool = False,
                       filter_by_prox: bool = False,
                       prox_threshold: float = None,
                       filter_day_zero: bool = False,
                       show_flip_matches: bool = False) -> pd.DataFrame:
        """Complete scan with swing detection and analysis
        
        Args:
            single_model: If provided, only analyze this specific model (e.g., 'FOGZ_Premium_Output')
            filter_epic_origins: If True, only include Trinidad/Tobago origins
            filter_same_origin: If True, only Trinidad+Trinidad or Tobago+Tobago
            filter_by_prox: If True, filter by output spread (prox)
            prox_threshold: Maximum output spread for matches
            filter_day_zero: If True, only include Day [0] arrivals
            show_flip_matches: If True, identify flip matches
        """
        zones = self.detect_swing_zones(start_time, end_time, min_swing_size)
        
        if not zones:
            return pd.DataFrame()
        
        results = []
        for zone in zones:
            analysis = self.analyze_zone(
                center_price=zone['price'],
                zone_width=zone_width,
                match_tolerance=match_tolerance,
                single_model=single_model,
                # Progressive filtering
                filter_epic_origins=filter_epic_origins,
                filter_same_origin=filter_same_origin,
                filter_by_prox=filter_by_prox,
                prox_threshold=prox_threshold,
                filter_day_zero=filter_day_zero,
                show_flip_matches=show_flip_matches
            )
            
            if 'error' not in analysis:
                analysis['zone_type'] = zone['type']
                analysis['zone_subtype'] = zone['subtype']
                analysis['zone_time'] = zone['time']
                results.append(analysis)
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.sort_values('score', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df
