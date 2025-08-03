"""
Model G Detection System
Proximity-based traveler grouping with descending M# sequences.  Use with Meas 1a.

Classification:
- G.05.o1[0]: Descending Grn, * Origin included, Today
- G.05.o2[≠0]: Descending Grn, * Origin included, Other days

Requirements:
- Minimum of 3 M #s in the sequence
- M #s must arrive in descending order by absolute value
- An Anchor or EPC origin must be in the sequence
- Proximity grouping for outputs within specified distance
"""

import pandas as pd
import streamlit as st
from collections import defaultdict
import numpy as np

def get_origin_type(origin):
    """Classify origin as Anchor, EPC, or Other"""
    origin_lower = origin.lower()
    
    # Anchor origins
    anchor_keywords = ['spain', 'saturn', 'jupiter', 'kepler-62', 'kepler-44']
    if any(keyword in origin_lower for keyword in anchor_keywords):
        return 'Anchor'
    
    # EPC origins  
    epc_keywords = ['trinidad', 'tobago', 'wasp-12b', 'macedonia']
    if any(keyword in origin_lower for keyword in epc_keywords):
        return 'EPC'
    
    return 'Other'

def group_by_proximity(outputs, proximity_threshold):
    """Group outputs that are within proximity_threshold of each other"""
    if len(outputs) == 0:
        return []
    
    # Sort outputs for easier processing
    sorted_outputs = sorted(outputs, key=lambda x: x['Output'])
    groups = []
    current_group = [sorted_outputs[0]]
    
    for i in range(1, len(sorted_outputs)):
        current_output = sorted_outputs[i]['Output']
        last_in_group = current_group[-1]['Output']
        
        # Check if current output is within proximity of any output in current group
        within_proximity = False
        for item in current_group:
            if abs(current_output - item['Output']) <= proximity_threshold:
                within_proximity = True
                break
        
        if within_proximity:
            current_group.append(sorted_outputs[i])
        else:
            # Start new group
            if len(current_group) >= 3:  # Only keep groups with 3+ items
                groups.append(current_group)
            current_group = [sorted_outputs[i]]
    
    # Don't forget the last group
    if len(current_group) >= 3:
        groups.append(current_group)
    
    return groups

def check_descending_m_numbers(group):
    """Check if M# values are in descending order by absolute value without duplicates"""
    if len(group) < 3:
        return False
    
    # Sort by arrival time to get chronological order
    sorted_by_time = sorted(group, key=lambda x: x['Arrival'])
    
    # Extract absolute M# values in arrival order
    m_values = [abs(float(item['M #'])) for item in sorted_by_time]
    
    # Check for duplicates first
    if len(set(m_values)) != len(m_values):
        return False  # Has duplicates, reject
    
    # Check if they're in strictly descending order
    for i in range(len(m_values) - 1):
        if m_values[i] <= m_values[i + 1]:  # Changed from < to <= to catch equals
            return False
    
    return True

def has_required_origin(group):
    """Check if group contains at least one Anchor or EPC origin"""
    for item in group:
        origin_type = get_origin_type(item['Origin'])
        if origin_type in ['Anchor', 'EPC']:
            return True
    return False

def classify_by_day(group):
    """Classify as [0] (today) or [≠0] (other days)"""
    # Check if any item in the group has Day = '[0]'
    for item in group:
        if item['Day'] == '[0]':
            return '[0]'
    return '[≠0]'

def detect_model_g_sequences(df, proximity_threshold=0.10):
    """
    Detect Model G sequences in the dataframe
    
    Args:
        df: DataFrame with traveler data
        proximity_threshold: Maximum distance between outputs to be considered in same group
    
    Returns:
        Dictionary with detected sequences
    """
    results = {
        'G.05.o1[0]': [],    # Today sequences
        'G.05.o2[≠0]': [],   # Other day sequences
        'proximity_groups': [],
        'rejected_groups': []
    }
    
    if df.empty:
        return results
    
    # Convert dataframe to list of dictionaries for easier processing
    data_list = []
    for _, row in df.iterrows():
        data_list.append({
            'Output': float(row['Output']),
            'Arrival': row['Arrival'],
            'Origin': row['Origin'],
            'M #': row['M #'],
            'Day': row['Day'],
            'Feed': row['Feed'],
            'M Name': row['M Name'],
            'R #': row['R #'],
            'Tag': row['Tag'],
            'Family': row['Family']
        })
    
    # Group by proximity
    proximity_groups = group_by_proximity(data_list, proximity_threshold)
    results['proximity_groups'] = proximity_groups
    
    # Process each proximity group
    for group in proximity_groups:
        # Check requirements
        has_descending_m = check_descending_m_numbers(group)
        has_anchor_epc = has_required_origin(group)
        day_classification = classify_by_day(group)
        
        # Store group info for debugging
        sorted_by_time = sorted(group, key=lambda x: x['Arrival'])
        m_values = [abs(float(item['M #'])) for item in sorted_by_time]
        has_duplicates = len(set(m_values)) != len(m_values)
        
        group_info = {
            'outputs': [item['Output'] for item in group],
            'origins': [item['Origin'] for item in group],
            'days': [item['Day'] for item in group],
            'm_numbers': [item['M #'] for item in group],
            'm_abs_values': m_values,
            'has_duplicates': has_duplicates,
            'has_descending_m': has_descending_m,
            'has_anchor_epc': has_anchor_epc,
            'day_classification': day_classification,
            'size': len(group)
        }
        
        # Classify if meets all requirements
        if has_descending_m and has_anchor_epc:
            if day_classification == '[0]':
                results['G.05.o1[0]'].append({
                    'group': group,
                    'info': group_info,
                    'model': 'G.05.o1[0]'
                })
            else:
                results['G.05.o2[≠0]'].append({
                    'group': group,
                    'info': group_info,
                    'model': 'G.05.o2[≠0]'
                })
        else:
            # Track rejected groups for debugging
            group_info['rejection_reason'] = []
            if has_duplicates:
                group_info['rejection_reason'].append('Duplicate M# values found')
            if not has_descending_m:
                if has_duplicates:
                    group_info['rejection_reason'].append('M# not in descending order (duplicates)')
                else:
                    group_info['rejection_reason'].append('M# not in descending order')
            if not has_anchor_epc:
                group_info['rejection_reason'].append('No Anchor/EPC origin')
            
            results['rejected_groups'].append(group_info)
    
    return results

def run_model_g_detection(df, proximity_threshold=0.10):
    """Run Model G detection and display results in Streamlit"""
    
    st.subheader("🔍 Model G Detection - Proximity-Based Traveler Grouping")
    
    # Proximity input
    col1, col2 = st.columns([1, 3])
    with col1:
        proximity = st.number_input(
            "Proximity Threshold",
            min_value=0.001,
            max_value=10.0,
            value=proximity_threshold,
            step=0.001,
            format="%.3f",
            help="Maximum distance between outputs to be considered in same group"
        )
    
    with col2:
        st.info(f"Grouping outputs within ±{proximity:.3f} of each other")
    
    # Run detection
    results = detect_model_g_sequences(df, proximity)
    
    # Display results summary
    st.markdown("### 📊 Detection Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Proximity Groups", len(results['proximity_groups']))
    with col2:
        st.metric("G.05.o1[0] (Today)", len(results['G.05.o1[0]']))
    with col3:
        st.metric("G.05.o2[≠0] (Other Days)", len(results['G.05.o2[≠0]']))
    with col4:
        st.metric("Rejected Groups", len(results['rejected_groups']))
    
    # Display classified sequences
    if results['G.05.o1[0]'] or results['G.05.o2[≠0]']:
        st.markdown("### ✅ Classified Sequences")
        
        # Today sequences
        if results['G.05.o1[0]']:
            with st.expander(f"🟢 G.05.o1[0] - Today Sequences ({len(results['G.05.o1[0]'])})"):
                for i, seq in enumerate(results['G.05.o1[0]']):
                    st.markdown(f"**Sequence {i+1}:**")
                    seq_df = pd.DataFrame(seq['group'])
                    seq_df = seq_df.sort_values('Arrival')
                    st.dataframe(seq_df[['Output', 'Origin', 'M #', 'Day', 'Arrival']], use_container_width=True)
                    
                    info = seq['info']
                    st.markdown(f"- Outputs: {[f'{x:.3f}' for x in info['outputs']]}")
                    st.markdown(f"- M# sequence: {info['m_numbers']}")
                    st.markdown(f"- Origins: {info['origins']}")
                    st.markdown("---")
        
        # Other day sequences
        if results['G.05.o2[≠0]']:
            with st.expander(f"🔵 G.05.o2[≠0] - Other Day Sequences ({len(results['G.05.o2[≠0]'])})"):
                for i, seq in enumerate(results['G.05.o2[≠0]']):
                    st.markdown(f"**Sequence {i+1}:**")
                    seq_df = pd.DataFrame(seq['group'])
                    seq_df = seq_df.sort_values('Arrival')
                    st.dataframe(seq_df[['Output', 'Origin', 'M #', 'Day', 'Arrival']], use_container_width=True)
                    
                    info = seq['info']
                    st.markdown(f"- Outputs: {[f'{x:.3f}' for x in info['outputs']]}")
                    st.markdown(f"- M# sequence: {info['m_numbers']}")
                    st.markdown(f"- Origins: {info['origins']}")
                    st.markdown("---")
    
    # Display debugging information
    if results['rejected_groups']:
        with st.expander(f"❌ Rejected Groups ({len(results['rejected_groups'])}) - Debug Info"):
            for i, group_info in enumerate(results['rejected_groups']):
                st.markdown(f"**Rejected Group {i+1}:**")
                st.markdown(f"- Size: {group_info['size']} travelers")
                st.markdown(f"- Outputs: {[f'{x:.3f}' for x in group_info['outputs']]}")
                st.markdown(f"- M# values: {group_info['m_numbers']}")
                st.markdown(f"- M# absolute values: {group_info['m_abs_values']}")
                st.markdown(f"- Has duplicates: {group_info['has_duplicates']}")
                st.markdown(f"- Origins: {group_info['origins']}")
                st.markdown(f"- Days: {group_info['days']}")
                st.markdown(f"- Rejection reasons: {', '.join(group_info['rejection_reason'])}")
                st.markdown("---")
    
    # Display all proximity groups for debugging
    if results['proximity_groups']:
        with st.expander(f"🔧 All Proximity Groups ({len(results['proximity_groups'])}) - Technical Details"):
            for i, group in enumerate(results['proximity_groups']):
                st.markdown(f"**Group {i+1} ({len(group)} travelers):**")
                group_df = pd.DataFrame(group)
                group_df = group_df.sort_values('Output')
                st.dataframe(group_df[['Output', 'Origin', 'M #', 'Day']], use_container_width=True)
                
                # Show output range
                outputs = [item['Output'] for item in group]
                st.markdown(f"- Output range: {min(outputs):.3f} to {max(outputs):.3f} (spread: {max(outputs) - min(outputs):.3f})")
                st.markdown("---")
    
    return results

if __name__ == "__main__":
    # This allows testing the module independently
    st.title("Model G Detection Test")
    st.write("Upload a CSV file to test Model G detection")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.dataframe(df.head())
        
        if st.button("Run Model G Detection"):
            results = run_model_g_detection(df)
            st.write("Detection complete!")
