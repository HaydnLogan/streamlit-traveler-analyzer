"""
G.05 and G.06 Model Detection

v31i (11.8.25) - Updated to use output_spread_filter instead of proximity_threshold
- Changed parameter name for consistency across all G models
- Added output spread filtering to reject groups with spread > threshold
- Temporal grouping still used (needed for temporal descending sequences)
"""

import pandas as pd
import streamlit as st
from model_g_core import *

def run_g05_g06_detection(df, output_spread_filter=3.0):
    """
    G.05 Detection: Standard proximity grouping with descending sequences
    G.06 Detection: Similar to G.05 but with different criteria
    
    Args:
        output_spread_filter: Maximum output spread (max - min) for filtering groups
    """
    
    results = {
        'today_sequences': [],
        'other_day_sequences': [],
        'rejected_groups': []
    }
    
    # Convert DataFrame to list of dictionaries
    data_list = df.to_dict('records')
    
    # Group by proximity (still needed for temporal descending sequences)
    # Note: This uses a small time window for finding temporal patterns
    proximity_groups = group_by_proximity(data_list, 0.10)  # Fixed small window for temporal patterns
    
    if st.session_state.get('debug_g06', False):
        st.write(f"🔍 **G.05/G.06 Detection Debug**")
        st.write(f"  - Total proximity groups: {len(proximity_groups)}")
        st.write(f"  - Output spread filter: {output_spread_filter}")
    
    # Process each proximity group
    for group in proximity_groups:
        # Calculate output spread for this group
        output_values = [item['Output'] for item in group]
        min_output = min(output_values)
        max_output = max(output_values)
        output_spread = max_output - min_output
        
        # Filter by output spread FIRST (main filtering criterion)
        if output_spread > output_spread_filter:
            results['rejected_groups'].append({
                'outputs': output_values,
                'reasons': [f'Output spread {output_spread:.3f} exceeds filter {output_spread_filter:.3f}'],
                'output_range': f"{min_output:.3f} to {max_output:.3f} (spread: {output_spread:.3f})"
            })
            continue
        
        # Check basic requirements
        has_anchor_epc = has_required_origin(group)
        
        if not has_anchor_epc:
            # Store rejected group info
            results['rejected_groups'].append({
                'outputs': output_values,
                'reasons': ['No Anchor/EPC origin'],
                'output_range': f"{min_output:.3f} to {max_output:.3f} (spread: {output_spread:.3f})"
            })
            continue
        
        # Find all valid temporal descending sequences within this group
        valid_sequences = find_temporal_descending_sequences(group)
        
        if not valid_sequences:
            # Store rejected group info
            results['rejected_groups'].append({
                'outputs': output_values,
                'reasons': ['No valid descending sequences found'],
                'output_range': f"{min_output:.3f} to {max_output:.3f} (spread: {output_spread:.3f})"
            })
            continue
        
        # Process each valid sequence
        for sequence in valid_sequences:
            day_classification = classify_by_day(sequence)
            
            # Determine G.05 vs G.06 categorization
            ends_with_m50_anchor = ends_with_m50_and_anchor(sequence)
            
            if ends_with_m50_anchor:
                # G.05 categories
                if day_classification == 'today':
                    category = 'G.05.o1'
                else:
                    category = 'G.05.o2'
            else:
                # G.06 categories (all other valid sequences)
                if day_classification == 'today':
                    category = 'G.05.o3'
                else:
                    category = 'G.05.o4'
            
            # Store sequence info
            sequence_info = {
                'sequence': sequence,
                'category': category,
                'day_classification': day_classification,
                'ends_with_m50_anchor': ends_with_m50_anchor,
                'outputs': [item['Output'] for item in sequence],
                'origins': [item['Origin'] for item in sequence],
                'm_values': [float(item['M #']) for item in sequence],
                'feeds': [item['Feed'] for item in sequence]
            }
            
            if day_classification == 'today':
                results['today_sequences'].append(sequence_info)
            else:
                results['other_day_sequences'].append(sequence_info)
    
    if st.session_state.get('debug_g06', False):
        st.write(f"📊 **G.05/G.06 Detection Summary**")
        st.write(f"  - Today sequences: {len(results['today_sequences'])}")
        st.write(f"  - Other day sequences: {len(results['other_day_sequences'])}")
        st.write(f"  - Rejected groups: {len(results['rejected_groups'])}")
    
    return results
