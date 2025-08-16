"""
G.09 Model Detection - Descends then Flips
Detects x0Pd.w descending sequences ending with polarity flips
"""

import pandas as pd
import streamlit as st
from model_g_core import (
    GROUP_1B_TRAVELERS, X0PDW_POS, X0PDW_NEG, _round_m, _chronological_arrivals,
    get_origin_type, group_by_proximity, has_required_origin, classify_by_day,
    find_temporal_descending_sequences, is_subsequence_contained
)

# G.09 Flip Classification Constants
SMALL_FLIP_VALUES = set(range(-6, 7))  # -6 to +6 inclusive
MEDIUM_FLIP_VALUES = {-39, -36, -30, -22, -14, -10, 10, 14, 22, 30, 36, 39}
LARGE_FLIP_MIN = 40  # >=40 or <=-40

def _is_star_origin(origin):
    """Check if origin is a 'star origin' (Anchor or EPC)"""
    return get_origin_type(origin) in ['Anchor', 'EPC']

def _count_star_origins(sequence):
    """Count number of star origins in sequence"""
    return sum(1 for item in sequence if _is_star_origin(item['Origin']))

def _get_flip_type(m_value):
    """Classify M# value flip type: small, medium, or large"""
    if m_value is None:
        return None
    
    abs_m = abs(m_value)
    
    if abs_m in SMALL_FLIP_VALUES:
        return 'small'
    elif abs_m in {abs(v) for v in MEDIUM_FLIP_VALUES}:
        return 'medium'
    elif abs_m >= LARGE_FLIP_MIN:
        return 'large'
    else:
        return None

def _has_opposite_flip(sequence):
    """Check if sequence ends with opposite flip (same absolute value, different polarities)"""
    if len(sequence) < 2:
        return False
    
    # Sort by chronological arrival
    def get_sort_key(item):
        arrival = item['Arrival']
        try:
            if isinstance(arrival, str):
                return pd.to_datetime(arrival)
            elif hasattr(arrival, 'isoformat'):
                return pd.to_datetime(arrival) if not isinstance(arrival, pd.Timestamp) else arrival
            else:
                return pd.to_datetime(str(arrival))
        except:
            return pd.NaT
    
    sorted_seq = sorted(sequence, key=get_sort_key)
    last_two_m = [_round_m(item['M #']) for item in sorted_seq[-2:]]
    
    if None in last_two_m:
        return False
    
    # Check if they have same absolute value but different signs
    return abs(last_two_m[0]) == abs(last_two_m[1]) and last_two_m[0] != last_two_m[1]

def _is_valid_g09_sequence(sequence):
    """Check if sequence meets G.09 basic requirements"""
    if len(sequence) < 3:
        return False
    
    # Must be x0Pd.w descending pattern (reuse G.08 logic)
    m_vals = [_round_m(it.get("M #")) for it in sequence]
    m_vals = [m for m in m_vals if m is not None]
    
    if len(m_vals) < 3:
        return False
    
    # Check positive pattern
    pos_matches = []
    for m in m_vals:
        if m in X0PDW_POS:
            pos_matches.append(X0PDW_POS.index(m))
    if len(pos_matches) == len(m_vals) and len(set(pos_matches)) == len(pos_matches):
        if all(pos_matches[i] <= pos_matches[i+1] for i in range(len(pos_matches)-1)):
            return True
    
    # Check negative pattern
    neg_matches = []
    for m in m_vals:
        if m in X0PDW_NEG:
            neg_matches.append(X0PDW_NEG.index(m))
    if len(neg_matches) == len(m_vals) and len(set(neg_matches)) == len(neg_matches):
        if all(neg_matches[i] <= neg_matches[i+1] for i in range(len(neg_matches)-1)):
            return True
    
    return False

def _excludes_small_m_values(sequence):
    """Check if sequence excludes M# 6 and below (6 to -6) except for flip endings"""
    # Sort by chronological arrival  
    def get_sort_key(item):
        arrival = item['Arrival']
        try:
            if isinstance(arrival, str):
                return pd.to_datetime(arrival)
            elif hasattr(arrival, 'isoformat'):
                return pd.to_datetime(arrival) if not isinstance(arrival, pd.Timestamp) else arrival
            else:
                return pd.to_datetime(str(arrival))
        except:
            return pd.NaT
    
    sorted_seq = sorted(sequence, key=get_sort_key)
    
    # Check all except the last M# value (last can be small for flip)
    for item in sorted_seq[:-1]:
        m_val = _round_m(item['M #'])
        if m_val is not None and abs(m_val) <= 6:
            return False
    
    return True

def _ends_with_star_origin(sequence):
    """Check if sequence ends with a star origin"""
    if not sequence:
        return False
    
    # Sort by chronological arrival to get final item
    def get_sort_key(item):
        arrival = item['Arrival']
        try:
            if isinstance(arrival, str):
                return pd.to_datetime(arrival)
            elif hasattr(arrival, 'isoformat'):
                return pd.to_datetime(arrival) if not isinstance(arrival, pd.Timestamp) else arrival
            else:
                return pd.to_datetime(str(arrival))
        except:
            return pd.NaT
    
    sorted_seq = sorted(sequence, key=get_sort_key)
    final_item = sorted_seq[-1]
    
    return _is_star_origin(final_item['Origin'])

def _g09_day_from_final(sequence):
    """Determine today/other based on the chronologically final item"""
    def get_sort_key(item):
        arrival = item['Arrival']
        try:
            if isinstance(arrival, str):
                return pd.to_datetime(arrival)
            elif hasattr(arrival, 'isoformat'):
                return pd.to_datetime(arrival) if not isinstance(arrival, pd.Timestamp) else arrival
            else:
                return pd.to_datetime(str(arrival))
        except:
            return pd.NaT
    
    sorted_seq = sorted(sequence, key=get_sort_key)
    final_item = sorted_seq[-1]
    
    try:
        final_arrival = pd.to_datetime(final_item['Arrival'])
        today = pd.Timestamp.now().normalize()
        return "today" if final_arrival.normalize() == today else "other"
    except:
        return "other"

def run_g09_detection(df, proximity_threshold=0.10):
    """
    G.09 Detection: x0Pd.w descending patterns with flip endings
    Categories: G.09.01opp, G.09.02opp, G.09.03sm, G.09.04md, G.09.05lg
    """
    
    # Filter to Group 1B travelers only
    group_1b_df = df[df['M #'].apply(lambda x: _round_m(x) in GROUP_1B_TRAVELERS)].copy()
    
    if group_1b_df.empty:
        return {
            'today_sequences': [],
            'other_day_sequences': [],
            'rejected_groups': []
        }
    
    results = {
        'today_sequences': [],
        'other_day_sequences': [],
        'rejected_groups': []
    }
    
    # Convert to list of dictionaries
    data_list = group_1b_df.to_dict('records')
    
    # Group by proximity
    proximity_groups = group_by_proximity(data_list, proximity_threshold)
    
    if st.session_state.get('debug_g06', False):
        st.write(f"🔍 **G.09 Detection Debug**")
        st.write(f"  - Group 1B filtered records: {len(group_1b_df)}")
        st.write(f"  - Total proximity groups: {len(proximity_groups)}")
    
    # Process each proximity group
    for group in proximity_groups:
        # Check G.09 specific requirements
        if not _is_valid_g09_sequence(group):
            continue
        
        if not _ends_with_star_origin(group):
            continue
        
        day_classification = _g09_day_from_final(group)
        
        # Determine category based on flip type and star origin count
        has_opposite_flip = _has_opposite_flip(group)
        star_origin_count = _count_star_origins(group)
        excludes_small = _excludes_small_m_values(group)
        
        category = None
        
        if has_opposite_flip and excludes_small:
            if star_origin_count >= 2:
                category = "G.09.01opp"
            else:
                category = "G.09.02opp"
        else:
            # Check flip type from final M# value
            def get_sort_key(item):
                arrival = item['Arrival']
                try:
                    if isinstance(arrival, str):
                        return pd.to_datetime(arrival)
                    elif hasattr(arrival, 'isoformat'):
                        return pd.to_datetime(arrival) if not isinstance(arrival, pd.Timestamp) else arrival
                    else:
                        return pd.to_datetime(str(arrival))
                except:
                    return pd.NaT
            
            sorted_seq = sorted(group, key=get_sort_key)
            final_m = _round_m(sorted_seq[-1]['M #'])
            
            if final_m is not None:
                flip_type = _get_flip_type(final_m)
                
                if flip_type == 'small':
                    category = "G.09.03sm"
                elif flip_type == 'medium':
                    category = "G.09.04md"
                elif flip_type == 'large':
                    category = "G.09.05lg"
        
        if category is None:
            continue
        
        # Store sequence info
        sequence_info = {
            'sequence': group,
            'category': category,
            'day_classification': day_classification,
            'star_origin_count': star_origin_count,
            'has_opposite_flip': has_opposite_flip,
            'excludes_small': excludes_small,
            'outputs': [item['Output'] for item in group],
            'origins': ', '.join([item['Origin'] for item in group]),
            'm_values': [_round_m(item['M #']) for item in group],
            'feeds': [item['Feed'] for item in group]
        }
        
        if day_classification == 'today':
            results['today_sequences'].append(sequence_info)
        else:
            results['other_day_sequences'].append(sequence_info)
    
    return results
