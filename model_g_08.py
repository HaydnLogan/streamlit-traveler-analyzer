"""
G.08 Model Detection - Advanced x0Pd.w Pattern Recognition
Enhanced detection with cross-group merging for "Both" category
"""

import pandas as pd
import streamlit as st
from model_g_core import (
    GROUP_1B_TRAVELERS, X0PDW_POS, X0PDW_NEG, _round_m, _chronological_arrivals,
    get_origin_type, group_by_proximity, has_required_origin, classify_by_day,
    find_temporal_descending_sequences, is_subsequence_contained, 
    ends_with_m50_and_anchor, ends_with_opposite_m50_pair
)

def _arrivals_ok_for_g08(items):
    """No repeated Arrival datetimes except possibly at the END"""
    arr = _chronological_arrivals(items)
    if not arr or all(pd.isna(x) for x in arr):
        return False
    # mask for NaT
    arr2 = [x for x in arr if pd.notna(x)]
    if not arr2:
        return False
    final_t = max(arr2)
    # Count duplicates not equal to final_t
    counts = {}
    for t in arr2:
        counts[t] = counts.get(t, 0) + 1
    for t, c in counts.items():
        if t != final_t and c > 1:
            return False
    return True

def _g08_day_from_final(items):
    """Determine today/other based on the chronologically final item"""
    arr = _chronological_arrivals(items)
    arr2 = [x for x in arr if pd.notna(x)]
    if not arr2:
        return "other"
    final_t = max(arr2)
    today = pd.Timestamp.now().normalize()
    return "today" if final_t.normalize() == today else "other"

def _is_x0pdw_descending(items):
    """Check M# follow x0Pd.w descending pattern"""
    m_vals = [_round_m(it.get("M #")) for it in items]
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

def _g08_end_type(items):
    """Determine if sequence ends with Anchor, EPC, or Both"""
    arr = _chronological_arrivals(items)
    arr2 = [(x, it) for x, it in zip(arr, items) if pd.notna(x)]
    if not arr2:
        return "Neither"
    
    final_t = max(arr2, key=lambda x: x[0])[0]
    final_items = [it for x, it in arr2 if x == final_t]
    
    origins = [it.get("Origin", "") for it in final_items]
    has_anchor = any(get_origin_type(o) == 'Anchor' for o in origins)
    has_epc = any(get_origin_type(o) == 'EPC' for o in origins)
    
    if has_anchor and has_epc:
        return "Both"
    elif has_anchor:
        return "Anchor"
    elif has_epc:
        return "EPC"
    else:
        return "Neither"

def run_g08_detection(df, proximity_threshold=0.10):
    """
    G.08 Detection: x0Pd.w descending patterns with Group 1B filtering
    Categories: G.08.01 (Anchor), G.08.02 (EPC), G.08.03 (Both)
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
        st.write(f"🔍 **G.08 Detection Debug**")
        st.write(f"  - Group 1B filtered records: {len(group_1b_df)}")
        st.write(f"  - Total proximity groups: {len(proximity_groups)}")
    
    # Process each proximity group
    for group in proximity_groups:
        # Check G.08 specific requirements
        if not _arrivals_ok_for_g08(group):
            continue
        
        if not _is_x0pdw_descending(group):
            continue
        
        end_type = _g08_end_type(group)
        if end_type == "Neither":
            continue
        
        day_classification = _g08_day_from_final(group)
        
        # Determine category based on end type
        if end_type == "Anchor":
            category = "G.08.01"
        elif end_type == "EPC":
            category = "G.08.02"
        else:  # Both
            category = "G.08.03"
        
        # Store sequence info
        sequence_info = {
            'sequence': group,
            'category': category,
            'day_classification': day_classification,
            'end_type': end_type,
            'outputs': [item['Output'] for item in group],
            'origins': ', '.join([item['Origin'] for item in group]),
            'm_values': [_round_m(item['M #']) for item in group],
            'feeds': [item['Feed'] for item in group]
        }
        
        if day_classification == 'today':
            results['today_sequences'].append(sequence_info)
        else:
            results['other_day_sequences'].append(sequence_info)
    
    # Post-process for cross-group G.08.03 detection
    _detect_cross_group_g08_both(results)
    
    return results

def _detect_cross_group_g08_both(results):
    """
    Post-processing step to detect identical sequences that end with different origin types
    Merges G.08.01 (Anchor) and G.08.02 (EPC) into G.08.03 (Both) when they match
    """
    
    # Process both today and other day sequences
    for day_type in ['today_sequences', 'other_day_sequences']:
        if day_type not in results:
            continue
        
        sequences = results[day_type]
        
        # Find G.08.01 and G.08.02 sequences for potential merging
        anchor_sequences = [seq for seq in sequences if seq['category'] == 'G.08.01']
        epc_sequences = [seq for seq in sequences if seq['category'] == 'G.08.02']
        
        matches_to_remove = []
        both_key = day_type
        
        # Compare each anchor sequence with each EPC sequence
        for i, anchor_seq in enumerate(anchor_sequences):
            anchor_m_values = tuple(sorted(anchor_seq['m_values']))
            
            # Safe timestamp parsing for final arrival
            anchor_arrivals = []
            for item in anchor_seq['sequence']:
                arrival = item['Arrival']
                try:
                    if isinstance(arrival, str):
                        if ':' in arrival and len(arrival.split(':')) >= 2:
                            # Bypass mode: time only format
                            from datetime import datetime
                            today = datetime.now().strftime('%Y-%m-%d')
                            full_timestamp = f"{today}T{arrival}"
                            anchor_arrivals.append(pd.to_datetime(full_timestamp))
                        else:
                            # ISO format or other date formats
                            anchor_arrivals.append(pd.to_datetime(arrival))
                    elif hasattr(arrival, 'isoformat'):
                        anchor_arrivals.append(pd.to_datetime(arrival) if not isinstance(arrival, pd.Timestamp) else arrival)
                    else:
                        anchor_arrivals.append(pd.to_datetime(str(arrival)))
                except:
                    continue
            
            if not anchor_arrivals:
                continue
                
            anchor_final_arrival = max(anchor_arrivals)
            
            for j, epc_seq in enumerate(epc_sequences):
                epc_m_values = tuple(sorted(epc_seq['m_values']))
                
                # Safe timestamp parsing for final arrival
                epc_arrivals = []
                for item in epc_seq['sequence']:
                    arrival = item['Arrival']
                    try:
                        if isinstance(arrival, str):
                            if ':' in arrival and len(arrival.split(':')) >= 2:
                                # Bypass mode: time only format
                                from datetime import datetime
                                today = datetime.now().strftime('%Y-%m-%d')
                                full_timestamp = f"{today}T{arrival}"
                                epc_arrivals.append(pd.to_datetime(full_timestamp))
                            else:
                                # ISO format or other date formats
                                epc_arrivals.append(pd.to_datetime(arrival))
                        elif hasattr(arrival, 'isoformat'):
                            epc_arrivals.append(pd.to_datetime(arrival) if not isinstance(arrival, pd.Timestamp) else arrival)
                        else:
                            epc_arrivals.append(pd.to_datetime(str(arrival)))
                    except:
                        continue
                
                if not epc_arrivals:
                    continue
                    
                epc_final_arrival = max(epc_arrivals)
                
                # Check if sequences match (same M# pattern and final arrival time)
                if (anchor_m_values == epc_m_values and 
                    abs((anchor_final_arrival - epc_final_arrival).total_seconds()) < 60):  # Within 1 minute
                    
                    # Create combined "both" sequence
                    both_seq = anchor_seq.copy()
                    both_seq['category'] = both_seq['category'].replace('G.08.01', 'G.08.03')
                    both_seq['origins'] = f"{anchor_seq['origins']} + {epc_seq['origins']}"
                    
                    # Add to both category
                    results[both_key].append(both_seq)
                    
                    # Mark for removal
                    matches_to_remove.append(('G.08.01', i))
                    matches_to_remove.append(('G.08.02', j))
                    
                    if st.session_state.get('debug_g06', False):
                        st.write(f"🔍 **Cross-group G.08.03 detected!**")
                        st.write(f"  - Merged identical sequences from G.08.01 and G.08.02")
                        st.write(f"  - M# pattern: {anchor_m_values}")
                        st.write(f"  - Final arrival: {anchor_final_arrival}")
        
        # Remove duplicates (in reverse order to preserve indices)
        sequences_to_remove = set()
        for category, idx in matches_to_remove:
            for k, seq in enumerate(sequences):
                if seq['category'] == category:
                    if category == 'G.08.01' and idx == len([s for s in sequences[:k+1] if s['category'] == 'G.08.01']) - 1:
                        sequences_to_remove.add(k)
                    elif category == 'G.08.02' and idx == len([s for s in sequences[:k+1] if s['category'] == 'G.08.02']) - 1:
                        sequences_to_remove.add(k)
        
        # Remove in reverse order to preserve indices
        for idx in sorted(sequences_to_remove, reverse=True):
            if idx < len(results[day_type]):
                results[day_type].pop(idx)
