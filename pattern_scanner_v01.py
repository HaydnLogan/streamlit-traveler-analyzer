"""
Haydn's Pattern Scanner
Implements Haydn's manual pattern recognition methodology
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pattern_detector_v2 import get_family, get_tag_category
import sys
sys.path.append('/mnt/user-data/uploads')
from nested_swing_detector import analyze_swings

# Pattern significance weights
WEIGHTS = {
    'epic_epic_same': 100,      # Tobago-Tobago, Trinidad-Trinidad
    'epic_epic_different': 50,  # Trinidad-Tobago
    'anchor_anchor_same': 40,   # Jupiter-Jupiter, Saturn-Saturn, etc.
    'downgrade': 30,            # Larger → smaller M#
    'same_family': 25,          # Both from same family
    'x0_alignment': 35,         # Both X0p or both X0d
    'large_m': 20,              # M# >= 80
    'very_large_m': 40,         # M# >= 100
    'flip_match': 20,           # PD, DD, PP flip
    'tight_spread': 50,         # <0.5 output spread
    'medium_spread': 25,        # 0.5-1.0 spread
    'family_cluster': 15,       # Per additional family member
}

EPIC_ORIGINS = {'Trinidad', 'Tobago'}
ANCHOR_ORIGINS = {'Saturn', 'Jupiter', 'Spain', 'Kepler-44', 'Kepler-62'}

class HaydnPatternScanner:
    def __init__(self, traveler_df: pd.DataFrame, ohlc_df: pd.DataFrame = None):
        """
        Initialize Haydn's pattern scanner
        
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
        
    def detect_swing_zones(self, 
                          start_time: str,
                          end_time: str,
                          min_swing_size: float = 60) -> List[Dict]:
        """
        Detect key price zones using nested swing detector
        
        Returns:
            List of zone dictionaries with price and type
        """
        if self.ohlc_df is None:
            return []
        
        # Run swing analysis
        nested_swings, major_points = analyze_swings(
            self.ohlc_df,
            start_time=pd.to_datetime(start_time),
            end_time=pd.to_datetime(end_time),
            min_swing_size=min_swing_size,
            pullback_tolerance=30
        )
        
        # Convert to zones
        zones = []
        
        # Add swing origins (reversal points)
        for point in major_points:
            if point['is_reversal']:
                zones.append({
                    'price': point['price'],
                    'type': 'Reversal',
                    'subtype': point['type'],  # High or Low
                    'time': point['time'],
                    'significance': point['significance_score']
                })
        
        # Add major extremes
        for point in major_points:
            if point['is_major_extreme']:
                zones.append({
                    'price': point['price'],
                    'type': 'Major Extreme',
                    'subtype': point['type'],
                    'time': point['time'],
                    'significance': point['significance_score']
                })
        
        return zones
    
    def analyze_zone(self, 
                     center_price: float,
                     zone_width: float = 10.0,
                     match_tolerance: float = 1.0) -> Dict:
        """
        Analyze traveler patterns in a price zone
        
        Args:
            center_price: Center of zone
            zone_width: Total width of zone (±width/2)
            match_tolerance: Max spread for pattern matches
            
        Returns:
            Dictionary with all detected patterns and scoring
        """
        # Get all arrivals in zone (all days)
        zone_df = self.traveler_df[
            (self.traveler_df['Output'] >= center_price - zone_width/2) &
            (self.traveler_df['Output'] <= center_price + zone_width/2)
        ].copy()
        
        if len(zone_df) == 0:
            return {'error': 'No arrivals in zone'}
        
        # Detect all pattern types
        patterns = {
            'epic_same_origin': self._find_epic_same_origin(zone_df, match_tolerance),
            'epic_epic_pairs': self._find_epic_epic_pairs(zone_df, match_tolerance),
            'downgrades': self._find_downgrades(zone_df, match_tolerance),
            'x0_alignments': self._find_x0_alignments(zone_df, match_tolerance),
            'large_m_presence': self._find_large_m(zone_df),
            'family_clusters': self._find_family_clusters(zone_df),
            'flip_matches': self._find_flip_matches(zone_df, match_tolerance),
        }
        
        # Calculate composite score
        score = self._calculate_zone_score(patterns)
        
        return {
            'center_price': center_price,
            'zone_width': zone_width,
            'num_arrivals': len(zone_df),
            'patterns': patterns,
            'score': score,
            'rank': None  # Will be assigned later
        }
    
    def _find_epic_same_origin(self, zone_df: pd.DataFrame, tolerance: float) -> List[Dict]:
        """Find Tobago-Tobago or Trinidad-Trinidad matches"""
        matches = []
        
        for origin in EPIC_ORIGINS:
            origin_df = zone_df[zone_df['Origin'] == origin]
            
            # Look for pairs with different M# values
            for idx1, row1 in origin_df.iterrows():
                for idx2, row2 in origin_df.iterrows():
                    if idx1 >= idx2:
                        continue
                    
                    if row1['M #'] == row2['M #']:
                        continue  # Skip exact same M#
                    
                    # Check spread and feed
                    spread = abs(row1['Output'] - row2['Output'])
                    if spread <= tolerance and row1['Feed'] == row2['Feed']:
                        
                        # Determine flip type
                        flip_type = self._get_flip_type(row1['M #'], row2['M #'])
                        is_downgrade = abs(row1['M #']) > abs(row2['M #'])
                        
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
                            'flip_type': flip_type,
                            'is_downgrade': is_downgrade,
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
                    
                    flip_type = self._get_flip_type(r1['M #'], r2['M #'])
                    is_downgrade = abs(r1['M #']) > abs(r2['M #'])
                    
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
                        'flip_type': flip_type,
                        'is_downgrade': is_downgrade,
                        'day1': r1['Day'],
                        'day2': r2['Day'],
                        'significance': 'TT Match - Epic Origins'
                    })
        
        return matches
    
    def _find_downgrades(self, zone_df: pd.DataFrame, tolerance: float) -> List[Dict]:
        """Find downgrade patterns (larger M# → smaller M#)"""
        downgrades = []
        
        # Look for any pairs where |M#1| > |M#2| with tight spread
        for idx1, row1 in zone_df.iterrows():
            for idx2, row2 in zone_df.iterrows():
                if idx1 >= idx2:
                    continue
                
                if abs(row1['M #']) <= abs(row2['M #']):
                    continue  # Not a downgrade
                
                spread = abs(row1['Output'] - row2['Output'])
                if spread <= tolerance and row1['Feed'] == row2['Feed']:
                    
                    differential = abs(row1['M #']) - abs(row2['M #'])
                    flip_type = self._get_flip_type(row1['M #'], row2['M #'])
                    same_family = row1['Family_Calc'] == row2['Family_Calc']
                    
                    downgrades.append({
                        'origin1': row1['Origin'],
                        'origin2': row2['Origin'],
                        'm_large': row1['M #'],
                        'm_small': row2['M #'],
                        'differential': differential,
                        'output1': row1['Output'],
                        'output2': row2['Output'],
                        'spread': spread,
                        'feed': row1['Feed'],
                        'family1': row1['Family_Calc'],
                        'family2': row2['Family_Calc'],
                        'same_family': same_family,
                        'flip_type': flip_type,
                        'day1': row1['Day'],
                        'day2': row2['Day']
                    })
        
        return downgrades
    
    def _find_x0_alignments(self, zone_df: pd.DataFrame, tolerance: float) -> List[Dict]:
        """Find X0 tag alignment patterns (X0p→X0p, etc.)"""
        alignments = []
        
        # Filter to X0 tags
        x0_df = zone_df[zone_df['Tag_Calc'].str.startswith('X0', na=False)]
        
        for idx1, row1 in x0_df.iterrows():
            for idx2, row2 in x0_df.iterrows():
                if idx1 >= idx2:
                    continue
                
                spread = abs(row1['Output'] - row2['Output'])
                if spread <= tolerance and row1['Feed'] == row2['Feed']:
                    
                    # Check if same X0 subtype
                    same_x0_type = row1['Tag_Calc'] == row2['Tag_Calc']
                    is_downgrade = abs(row1['M #']) > abs(row2['M #'])
                    
                    alignments.append({
                        'origin1': row1['Origin'],
                        'origin2': row2['Origin'],
                        'm1': row1['M #'],
                        'm2': row2['M #'],
                        'tag1': row1['Tag_Calc'],
                        'tag2': row2['Tag_Calc'],
                        'same_x0_type': same_x0_type,
                        'output1': row1['Output'],
                        'output2': row2['Output'],
                        'spread': spread,
                        'feed': row1['Feed'],
                        'is_downgrade': is_downgrade,
                        'day1': row1['Day'],
                        'day2': row2['Day']
                    })
        
        return alignments
    
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
        """Find family clusters (multiple members of same family)"""
        clusters = {}
        
        for family in zone_df['Family_Calc'].unique():
            if pd.isna(family):
                continue
            
            family_df = zone_df[zone_df['Family_Calc'] == family]
            
            # Get unique M# values in this family
            unique_m = family_df['M #'].nunique()
            
            if unique_m >= 2:  # At least 2 different M#s
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
        """Find flip patterns (PD, DD, PP)"""
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
    
    def _get_flip_type(self, m1: float, m2: float) -> str:
        """Determine flip type (PD, DD, PP, DP)"""
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
        """Calculate composite significance score for zone"""
        score = 0
        
        # Epic same origin (Tobago-Tobago, Trinidad-Trinidad)
        epic_same = patterns.get('epic_same_origin', [])
        score += len(epic_same) * WEIGHTS['epic_epic_same']
        
        # Epic-epic different (Trinidad-Tobago)
        epic_pairs = patterns.get('epic_epic_pairs', [])
        score += len(epic_pairs) * WEIGHTS['epic_epic_different']
        
        # Downgrades
        downgrades = patterns.get('downgrades', [])
        for dg in downgrades:
            score += WEIGHTS['downgrade']
            if dg.get('same_family'):
                score += WEIGHTS['same_family']
            if dg.get('spread', 999) < 0.5:
                score += WEIGHTS['tight_spread']
            elif dg.get('spread', 999) < 1.0:
                score += WEIGHTS['medium_spread']
        
        # X0 alignments
        x0_aligns = patterns.get('x0_alignments', [])
        for xa in x0_aligns:
            score += WEIGHTS['x0_alignment']
            if xa.get('same_x0_type'):
                score += WEIGHTS['x0_alignment'] * 0.5  # Bonus for exact match
        
        # Large M# presence
        large_m = patterns.get('large_m_presence', [])
        for lm in large_m:
            if lm['size'] >= 100:
                score += WEIGHTS['very_large_m']
            else:
                score += WEIGHTS['large_m']
        
        # Family clusters
        clusters = patterns.get('family_clusters', {})
        for family, data in clusters.items():
            # Score based on number of unique M# values
            score += data['unique_m_values'] * WEIGHTS['family_cluster']
        
        # Flip matches
        flips = patterns.get('flip_matches', [])
        score += len(flips) * WEIGHTS['flip_match']
        
        return score
    
    def scan_all_zones(self, 
                       start_time: str,
                       end_time: str,
                       min_swing_size: float = 60,
                       zone_width: float = 10.0) -> pd.DataFrame:
        """
        Complete scan: detect swings and analyze all zones
        
        Returns:
            DataFrame with all zones ranked by significance
        """
        # Detect swing zones
        zones = self.detect_swing_zones(start_time, end_time, min_swing_size)
        
        if not zones:
            return pd.DataFrame()
        
        # Analyze each zone
        results = []
        for zone in zones:
            analysis = self.analyze_zone(
                center_price=zone['price'],
                zone_width=zone_width,
                match_tolerance=1.0
            )
            
            if 'error' not in analysis:
                analysis['zone_type'] = zone['type']
                analysis['zone_subtype'] = zone['subtype']
                analysis['zone_time'] = zone['time']
                results.append(analysis)
        
        # Convert to DataFrame and rank
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.sort_values('score', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df
