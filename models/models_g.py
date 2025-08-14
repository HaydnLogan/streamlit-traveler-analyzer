"""
Model G Detection System
Proximity-based traveler grouping with descending M# sequences.  Use with Meas 1a XO.

Classification:
- G.05.o1[0]: Descending Grn to m50, to * Origin, Today
- G.05.o2[≠0]: Descending Grn to m50, to * Origin, Other days
- G.05.o3[0]: Descending Grn to ≠ m50, * Origin included, Today
- G.05.o4[≠0]: Descending Grn to ≠ m50, * Origin included, Other days

Requirements:
- Minimum of 3 M #s in the sequence
- M #s must arrive in descending order by absolute value
- An Anchor or EPC origin must be in the sequence
- Proximity grouping for outputs within specified distance
- Categories o1/o2: Sequence must end in M# 50 AND Anchor Origin
- Categories o3/o4: All other valid sequences with required origins
"""

import pandas as pd
import streamlit as st
from collections import defaultdict
import numpy as np
import io
from datetime import datetime

def run_model_g_detection(df, report_time=None, key_suffix=""):
    """Main function to run Model G detection with full cluster display"""
    
    st.subheader("🔍 Model G Detection - Proximity-Based Traveler Grouping")
    
    # Show dataset info for large datasets
    if len(df) > 10000:
        st.info(f"📊 Processing dataset: {len(df):,} rows | Using optimized Python algorithm for large datasets")
    
    # Proximity input and display controls
    col1, col2 = st.columns([1, 3])
    with col1:
        proximity_threshold = st.number_input(
            "Proximity Threshold",
            min_value=0.001,
            max_value=10.0,
            value=0.10,
            step=0.001,
            format="%.3f",
            help="Maximum distance between outputs to be considered in same group",
            key=f"g_proximity_threshold{key_suffix}"
        )
    
    with col2:
        st.info(f"Grouping outputs within ±{proximity_threshold:.3f} of each other")
    
    # Display control toggles
    st.markdown("### 🎛️ Display Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_o1 = st.checkbox("G.05.o1[0]: Today M50+Anchor", value=True, key=f"g_show_o1{key_suffix}")
        show_g06_o1 = st.checkbox("G.06.o1[0]: Today Opposites", value=True, key=f"g_show_g06_o1{key_suffix}")
    with col2:
        show_o2 = st.checkbox("G.05.o2[≠0]: Other M50+Anchor", value=True, key=f"g_show_o2{key_suffix}")
        show_g06_o2 = st.checkbox("G.06.o2[≠0]: Other Opposites", value=True, key=f"g_show_g06_o2{key_suffix}")
    with col3:
        show_rejected = st.checkbox("❌ Rejected Groups", value=False, key=f"g_show_rejected{key_suffix}")
        st.session_state['debug_g06'] = st.checkbox("🔍 G.06 Debug", value=False, key=f"g_debug_g06{key_suffix}")
    
    try:
        # Run the detection
        results = detect_model_g_sequences(df, proximity_threshold)
        
        # Display cluster summary
        st.markdown("### 📊 Cluster Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Proximity Groups", len(results['proximity_groups']))
        with col2:
            st.metric("G.05 Today (o1)", len(results['G.05.o1[0]']))
        with col3:
            st.metric("G.05 Other (o2)", len(results['G.05.o2[≠0]']))
        with col4:
            st.metric("G.06 Today", len(results['G.06.o1[0]']))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("G.06 Other", len(results['G.06.o2[≠0]']))
        with col4:
            st.metric("Rejected Groups", len(results['rejected_groups']))
        
        # Create cluster table for enabled categories
        enabled_sequences = []
        if show_o1:
            enabled_sequences.extend(results['G.05.o1[0]'])
        if show_o2:
            enabled_sequences.extend(results['G.05.o2[≠0]'])
        if show_g06_o1:
            enabled_sequences.extend(results['G.06.o1[0]'])
        if show_g06_o2:
            enabled_sequences.extend(results['G.06.o2[≠0]'])
        
        if enabled_sequences:
            st.markdown("### 📊 Cluster Table")
            
            # Collect all sequences for cluster analysis
            all_sequences = enabled_sequences
            
            # Create cluster data by grouping sequences by their final output
            cluster_data = {}
            
            for seq_info in all_sequences:
                # Get the final output (last in chronological order)
                final_output = seq_info['outputs'][-1]  # Last output in the sequence
                
                if final_output not in cluster_data:
                    cluster_data[final_output] = {
                        'sequences': [],
                        'largest_length': 0,
                        'unique_count': 0,
                        'final_arrival': None,
                        'final_day': None,
                        'final_origin': None
                    }
                
                # Add this sequence to the cluster
                cluster_data[final_output]['sequences'].append(seq_info)
                
                # Update largest sequence length
                seq_length = len(seq_info['outputs'])
                if seq_length > cluster_data[final_output]['largest_length']:
                    cluster_data[final_output]['largest_length'] = seq_length
                    # Store arrival info from the largest sequence
                    cluster_data[final_output]['final_arrival'] = seq_info['arrivals'][-1]
                    cluster_data[final_output]['final_day'] = seq_info['days'][-1]
                    cluster_data[final_output]['final_origin'] = seq_info['origins'][-1]
                
                # Count unique sequences
                cluster_data[final_output]['unique_count'] = len(cluster_data[final_output]['sequences'])
            
            # Build cluster table data
            cluster_rows = []
            for output, cluster_info in cluster_data.items():
                # Get representative sequence (the largest one)
                largest_seq = max(cluster_info['sequences'], key=lambda x: len(x['outputs']))
                
                # Get the end-of-sequence row data from the original dataframe
                final_output = largest_seq['outputs'][-1]  # Last output in the sequence
                final_arrival = largest_seq['arrivals'][-1]  # Last arrival time
                
                # Find the corresponding row in the dataframe for end-of-sequence values
                end_sequence_row = None
                for idx, row in df.iterrows():
                    if (abs(row['Output'] - final_output) < 0.001 and  # Match output within tolerance
                        pd.to_datetime(row['Arrival']) == pd.to_datetime(final_arrival)):  # Match arrival time
                        end_sequence_row = row
                        break
                
                # Extract input values from the end-of-sequence row
                if end_sequence_row is not None:
                    input_at_18 = end_sequence_row.get('Input @ 18:00', "")
                    diff_18 = end_sequence_row.get('Diff @ 18:00', "")
                    input_at_arrival = end_sequence_row.get('Input @ Arrival', "")
                    diff_arrival = end_sequence_row.get('Diff @ Arrival', "")
                    input_at_report = end_sequence_row.get('Input @ Report', "")
                    diff_report = end_sequence_row.get('Diff @ Report', "")
                else:
                    # Fallback if no matching row found
                    input_at_18 = ""
                    diff_18 = ""
                    input_at_arrival = ""
                    diff_arrival = ""
                    input_at_report = ""
                    diff_report = ""
                
                # Format arrival time into separate day and datetime columns
                if cluster_info['final_arrival']:
                    try:
                        arrival_dt = pd.to_datetime(cluster_info['final_arrival'])
                        if pd.notna(arrival_dt):  # Check if it's not NaT
                            day_abbrev = arrival_dt.strftime('%a')  # Mon, Tue, Wed, etc.
                            arrival_excel = arrival_dt.strftime('%d-%b-%Y %H:%M')  # Excel-friendly format
                            
                            # Calculate hours ago from current time
                            hours_ago = (pd.Timestamp.now() - arrival_dt).total_seconds() / 3600
                            hours_ago_str = f"{int(hours_ago)} hours ago"
                        else:
                            day_abbrev = ""
                            arrival_excel = str(cluster_info['final_arrival'])
                            hours_ago_str = ""
                    except:
                        day_abbrev = ""
                        arrival_excel = str(cluster_info['final_arrival'])
                        hours_ago_str = ""
                else:
                    day_abbrev = ""
                    arrival_excel = ""
                    hours_ago_str = ""
                
                # Create pattern sequences description
                pattern_sequences = []
                for seq in cluster_info['sequences']:
                    m_values = [str(m) for m in seq['m_values']]
                    origins = [str(o) for o in seq['origins']]
                    
                    # Get M names from the original dataframe by matching M# values
                    m_names = []
                    for m_val in seq['m_values']:
                        # Find corresponding M name in the dataframe
                        matching_rows = df[df['M #'] == m_val]
                        if not matching_rows.empty:
                            m_name = matching_rows.iloc[0].get('M Name', f'M{m_val}')
                            m_names.append(str(m_name))
                        else:
                            m_names.append(f'M{m_val}')
                    
                    if len(seq['outputs']) >= 4:  # Only show sequences with 4+ elements
                        # Determine if this is G.05 or G.06 based on sequence category
                        seq_category = seq.get('category', 'G.05')  # Default to G.05 if not specified
                        model_type = 'G.06' if 'G.06' in seq_category else 'G.05'
                        pattern_desc = f"{model_type} | M Names: {','.join(m_names)} | M#: {','.join(m_values)} | Origins: {','.join(origins)}"
                        pattern_sequences.append(pattern_desc)
                
                pattern_str = " | ".join(pattern_sequences) if pattern_sequences else "No valid patterns"
                
                cluster_rows.append({
                    'Output': f"{output:.3f}",
                    'Largest sequence length': cluster_info['largest_length'],
                    'Unique Sequences': cluster_info['unique_count'],
                    'Stars': "",  # Blank for now
                    'Booster Score': "",  # Blank for now
                    'Input @ 18:00': input_at_18,
                    '18:00 Diff': diff_18,
                    'Input @ Arrival': input_at_arrival,
                    'Arrival Diff': diff_arrival,
                    'Input @ Report': input_at_report,
                    'Report Diff': diff_report,
                    'ddd': day_abbrev,  # Day abbreviation (Mon, Tue, Wed, etc.)
                    'Arrival': arrival_excel,  # Excel-friendly datetime format
                    'Day': cluster_info['final_day'] or "",
                    'Hours ago': hours_ago_str,
                    'Pattern Sequences': pattern_str
                })
            
            # Sort by output value (descending)
            cluster_rows.sort(key=lambda x: float(x['Output']), reverse=True)
            
            # Display cluster table
            if cluster_rows:
                cluster_df = pd.DataFrame(cluster_rows)
                st.dataframe(cluster_df, use_container_width=True)
                
                # Download button for cluster table with report time in filename
                try:
                    # Try to get report time from the dataframe
                    if 'Arrival' in df.columns and not df.empty:
                        arrival_times = pd.to_datetime(df['Arrival'], format='%d-%b-%Y %H:%M', errors='coerce')
                        # Filter out NaT values and get the maximum valid time
                        valid_times = arrival_times.dropna()
                        if len(valid_times) > 0:
                            report_time = valid_times.max()
                            timestamp = report_time.strftime("%y-%m-%d_%H-%M")
                        else:
                            timestamp = pd.Timestamp.now().strftime('%y-%m-%d_%H-%M')
                    else:
                        timestamp = pd.Timestamp.now().strftime('%y-%m-%d_%H-%M')
                except:
                    timestamp = pd.Timestamp.now().strftime('%y-%m-%d_%H-%M')
                    
                cluster_csv = cluster_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Cluster Table (CSV)",
                    data=cluster_csv,
                    file_name=f"model_g_cluster_table_{timestamp}.csv",
                    mime="text/csv",
                    key=f"g_download_cluster{key_suffix}"
                )
            else:
                st.info("No cluster data available")
            
            st.markdown("---")
        
        # Display o1 sequences
        if show_o1 and results['G.05.o1[0]']:
            st.markdown("### 🟢 G.05.o1[0]: Today Sequences (M50 + Anchor Origin)")
            for i, seq_info in enumerate(results['G.05.o1[0]']):
                with st.expander(f"Sequence {i+1}: {len(seq_info['sequence'])} travelers | Range: {seq_info['output_range']}"):
                    display_sequence_details(seq_info)
        
        # Display o2 sequences  
        if show_o2 and results['G.05.o2[≠0]']:
            st.markdown("### 🔵 G.05.o2[≠0]: Other Day Sequences (M50 + Anchor Origin)")
            for i, seq_info in enumerate(results['G.05.o2[≠0]']):
                with st.expander(f"Sequence {i+1}: {len(seq_info['sequence'])} travelers | Range: {seq_info['output_range']}"):
                    display_sequence_details(seq_info)
        
        # Display G.06.o1 sequences (Today Opposites)
        if show_g06_o1 and results['G.06.o1[0]']:
            st.markdown("### 🟡 G.06.o1[0]: Today Sequences (Opposite M# 50 Ending)")
            for i, seq_info in enumerate(results['G.06.o1[0]']):
                with st.expander(f"G.06 Sequence {i+1}: {len(seq_info['sequence'])} travelers | Range: {seq_info['output_range']}"):
                    display_sequence_details(seq_info)
        
        # Display G.06.o2 sequences (Other Day Opposites)  
        if show_g06_o2 and results['G.06.o2[≠0]']:
            st.markdown("### 🟠 G.06.o2[≠0]: Other Day Sequences (Opposite M# 50 Ending)")
            for i, seq_info in enumerate(results['G.06.o2[≠0]']):
                with st.expander(f"G.06 Sequence {i+1}: {len(seq_info['sequence'])} travelers | Range: {seq_info['output_range']}"):
                    display_sequence_details(seq_info)
        
        # Display rejected groups
        if show_rejected and results['rejected_groups']:
            st.markdown("### ❌ Rejected Groups")
            for i, rejected in enumerate(results['rejected_groups']):
                with st.expander(f"Rejected Group {i+1}: {len(rejected['outputs'])} outputs | {rejected['output_range']}"):
                    st.write(f"**Reasons:** {', '.join(rejected['reasons'])}")
                    st.write(f"**Outputs:** {rejected['outputs']}")
                    if 'sequence_details' in rejected:
                        st.write(f"**Details:** {rejected['sequence_details']}")
        
        # Format results for the main app interface
        all_sequences = []
        all_sequences.extend(results['G.05.o1[0]'])
        all_sequences.extend(results['G.05.o2[≠0]'])
        all_sequences.extend(results['G.06.o1[0]'])
        all_sequences.extend(results['G.06.o2[≠0]'])
        
        if all_sequences:
            # Create simplified DataFrame for main app display
            simple_results = []
            for seq_info in all_sequences:
                simple_results.append({
                    'category': seq_info.get('category', 'G.05.o1[0]' if seq_info in results['G.05.o1[0]'] else 'G.05.o2[≠0]'),
                    'count': len(seq_info['sequence']),
                    'day_type': seq_info['day_type'],
                    'output_range': seq_info['output_range']
                })
            results_df = pd.DataFrame(simple_results)
        else:
            results_df = pd.DataFrame()
        
        # Calculate summary
        summary = {
            'total_o1': len(results['G.05.o1[0]']) + len(results['G.06.o1[0]']),
            'total_o2': len(results['G.05.o2[≠0]']) + len(results['G.06.o2[≠0]']),
            'total_sequences': len(all_sequences),
            'g05_o1': len(results['G.05.o1[0]']),
            'g05_o2': len(results['G.05.o2[≠0]']),
            'g06_o1': len(results['G.06.o1[0]']),
            'g06_o2': len(results['G.06.o2[≠0]'])
        }
        
        return {
            'success': True,
            'results_df': results_df,
            'summary': summary,
            'full_results': results
        }
        
    except Exception as e:
        import traceback
        st.error(f"Model G detection error: {str(e)}")
        st.text(traceback.format_exc())
        return {
            'success': False,
            'error': str(e),
            'results_df': pd.DataFrame(),
            'summary': {'total_o1': 0, 'total_o2': 0, 'total_sequences': 0}
        }

def display_sequence_details(seq_info):
    """Display detailed sequence information"""
    sequence = seq_info['sequence']
    
    # Create DataFrame for display
    seq_df = pd.DataFrame([
        {
            'Output': item['Output'],
            'Arrival': item['Arrival'], 
            'Origin': item['Origin'],
            'M #': item['M #'],
            'Day': item['Day'],
            'Feed': item['Feed']
        } for item in sequence
    ])
    
    # Sort by arrival time for display
    seq_df = seq_df.sort_values('Arrival')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(seq_df, use_container_width=True)
    
    with col2:
        st.write("**Sequence Info:**")
        st.write(f"• Travelers: {len(sequence)}")
        st.write(f"• Output Range: {seq_info['output_range']}")
        st.write(f"• M# Values: {seq_info['m_values']}")
        st.write(f"• Day Type: {seq_info['day_type']}")
        st.write(f"• Descending: {'✓' if seq_info['is_descending'] else '✗'}")
        if seq_info['has_duplicates']:
            st.warning("⚠️ Contains duplicate M# values")


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
    """
    Group outputs using tight proximity clustering - top-to-bottom approach.
    Creates groups where ALL members are within threshold of at least one other member,
    preventing large transitive clusters.
    """
    if len(outputs) == 0:
        return []
    
    # Sort outputs by value for top-to-bottom processing
    sorted_outputs = sorted(outputs, key=lambda x: x['Output'])
    n = len(sorted_outputs)
    
    if n > 10000:
        st.info(f"Processing large dataset ({n:,} rows) with tight proximity clustering...")
    
    groups = []
    processed = [False] * n
    
    # Process from top to bottom
    for i in range(n):
        if processed[i]:
            continue
            
        # Start a new tight group with current output
        current_group = [sorted_outputs[i]]
        current_output = sorted_outputs[i]['Output']
        processed[i] = True
        
        # Find all outputs within direct proximity of the current output
        for j in range(i + 1, n):
            if processed[j]:
                continue
                
            candidate_output = sorted_outputs[j]['Output']
            
            # If candidate is too far from the current seed, stop checking (sorted order)
            if candidate_output - current_output > proximity_threshold:
                break
                
            # Check if candidate is within threshold of current seed
            if abs(candidate_output - current_output) <= proximity_threshold:
                current_group.append(sorted_outputs[j])
                processed[j] = True
        
        # Only keep groups with 3+ members
        if len(current_group) >= 3:
            groups.append(current_group)
    
    return groups

def find_temporal_descending_sequences(group):
    """
    Find all valid descending subsequences that respect temporal arrival order
    Returns list of subsequences, each with at least 3 elements
    """
    if len(group) < 3:
        return []
    
    # Sort by arrival time to get chronological order
    # Handle sorting with both datetime and string formats
    def get_sort_key(item):
        arrival = item['Arrival']
        if hasattr(arrival, 'isoformat'):
            return arrival
        else:
            # Try to parse string datetime if needed
            try:
                return pd.to_datetime(arrival)
            except:
                return arrival
    
    sorted_by_time = sorted(group, key=get_sort_key)
    
    # Find all possible descending subsequences of length 3+
    valid_sequences = []
    n = len(sorted_by_time)
    
    # Generate all possible subsequences of length 3+
    from itertools import combinations
    
    for length in range(3, n + 1):
        for indices in combinations(range(n), length):
            subsequence = [sorted_by_time[i] for i in indices]
            
            # Check if this subsequence has descending absolute M# values
            m_values = [abs(float(item['M #'])) for item in subsequence]
            

            # Must be strictly or mostly descending
            descending_pairs = sum(1 for i in range(len(m_values)-1) if m_values[i] > m_values[i+1])
            total_pairs = len(m_values) - 1
            
            # Require STRICTLY descending for Model G (100% descending pairs)
            if (total_pairs > 0 and descending_pairs == total_pairs and 
                len(set(m_values)) >= 3):
                valid_sequences.append(subsequence)
    
    # Remove smaller subsequences that are contained within larger ones
    deduplicated_sequences = deduplicate_sequences(valid_sequences)
    
    return deduplicated_sequences

def deduplicate_sequences(sequences):
    """
    Remove smaller subsequences that are completely contained within larger ones.
    Only keep the largest sequences.
    """
    if len(sequences) <= 1:
        return sequences
    
    # Sort sequences by length (longest first)
    sorted_sequences = sorted(sequences, key=len, reverse=True)
    
    final_sequences = []
    
    for current_seq in sorted_sequences:
        # Check if this sequence is contained within any already accepted larger sequence
        is_contained = False
        
        for larger_seq in final_sequences:
            if is_subsequence_contained(current_seq, larger_seq):
                is_contained = True
                break
        
        # Only add if not contained in a larger sequence
        if not is_contained:
            final_sequences.append(current_seq)
    
    return final_sequences

def is_subsequence_contained(small_seq, large_seq):
    """
    Check if small_seq is completely contained within large_seq.
    A sequence is contained if all its items are present in the larger sequence.
    """
    if len(small_seq) >= len(large_seq):
        return False
    
    # Convert Arrival to string format for comparison (handle both datetime and string formats)
    def format_arrival(arrival):
        if hasattr(arrival, 'isoformat'):
            return arrival.isoformat()
        else:
            return str(arrival)
    
    # Create sets of unique identifiers for each sequence (using Output and Arrival as unique key)
    small_items = {(item['Output'], format_arrival(item['Arrival'])) for item in small_seq}
    large_items = {(item['Output'], format_arrival(item['Arrival'])) for item in large_seq}
    
    # Check if all items in small sequence are present in large sequence
    return small_items.issubset(large_items)

def check_descending_m_numbers(group):
    """Check if group contains any valid descending sequences"""
    sequences = find_temporal_descending_sequences(group)
    return len(sequences) > 0

def has_required_origin(group):
    """Check if group contains at least one Anchor or EPC origin"""
    for item in group:
        origin_type = get_origin_type(item['Origin'])
        if origin_type in ['Anchor', 'EPC']:
            return True
    return False

def ends_with_m50_and_anchor(sequence):
    """Check if sequence ends in M# 50 and has Anchor origin at the end"""
    if not sequence:
        return False
    
    # Get the last item in temporal order (chronologically last arrival)
    def get_sort_key(item):
        arrival = item['Arrival']
        if hasattr(arrival, 'isoformat'):
            return arrival
        else:
            try:
                return pd.to_datetime(arrival)
            except:
                return arrival
    
    last_item = max(sequence, key=get_sort_key)
    
    # Check if last item has M# 50 and Anchor origin
    m_value = abs(float(last_item['M #']))
    origin_type = get_origin_type(last_item['Origin'])
    
    return m_value == 50 and origin_type == 'Anchor'

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
        'G.05.o1[0]': [],    # Today sequences ending in M50 + Anchor
        'G.05.o2[≠0]': [],   # Other day sequences ending in M50 + Anchor
        'G.06.o1[0]': [],    # Today sequences with opposite M# ending (+50/-50 or -50/+50)
        'G.06.o2[≠0]': [],   # Other day sequences with opposite M# ending
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
    
    # Debug proximity grouping (only when G.06 debug enabled)
    if st.session_state.get('debug_g06', False):
        st.write(f"Debug - Found {len(proximity_groups)} proximity groups")
        for i, group in enumerate(proximity_groups):
            st.write(f"Debug - Group {i+1}: {len(group)} items")
            st.write(f"Debug - Group {i+1} outputs: {[item['Output'] for item in group]}")
            st.write(f"Debug - Group {i+1} M# values: {[item['M #'] for item in group]}")
            st.write(f"Debug - Group {i+1} origins: {[item['Origin'] for item in group]}")
        st.write("---")
    
    # Process each proximity group and extract sequences
    for group in proximity_groups:
        # Check basic requirements
        has_anchor_epc = has_required_origin(group)
        
        if not has_anchor_epc:
            # Store rejected group info
            output_values = [item['Output'] for item in group]
            min_output = min(output_values)
            max_output = max(output_values)
            output_range_spread = max_output - min_output
            
            results['rejected_groups'].append({
                'outputs': output_values,
                'reasons': ['No Anchor/EPC origin'],
                'output_range': f"{min_output:.3f} to {max_output:.3f} (spread: {output_range_spread:.3f})"
            })
            continue
        
        # Find all valid temporal descending sequences within this group
        valid_sequences = find_temporal_descending_sequences(group)
        
        output_values = [item['Output'] for item in group]
        
        if not valid_sequences:
            # Store rejected group info
            min_output = min(output_values)
            max_output = max(output_values)
            output_range_spread = max_output - min_output
            
            results['rejected_groups'].append({
                'outputs': output_values,
                'reasons': ['No valid descending sequences found'],
                'output_range': f"{min_output:.3f} to {max_output:.3f} (spread: {output_range_spread:.3f})"
            })
            continue
        
        # Process each valid sequence separately
        for sequence in valid_sequences:
            day_classification = classify_by_day(sequence)
            
            # Store sequence info for debugging
            def get_sort_key(item):
                arrival = item['Arrival']
                if hasattr(arrival, 'isoformat'):
                    return arrival
                else:
                    try:
                        return pd.to_datetime(arrival)
                    except:
                        return arrival
            
            sorted_by_time = sorted(sequence, key=get_sort_key)
            m_values = [abs(float(item['M #'])) for item in sorted_by_time]
            has_duplicates = len(set(m_values)) != len(m_values)
            
            # Calculate output range for this specific sequence
            output_values = [item['Output'] for item in sequence]
            min_output = min(output_values)
            max_output = max(output_values)
            output_range_spread = max_output - min_output
            
            sequence_info = {
                'sequence': sequence,
                'outputs': output_values,
                'origins': [item['Origin'] for item in sequence],
                'm_values': [float(item['M #']) for item in sorted_by_time],
                'days': [item['Day'] for item in sorted_by_time],
                'feeds': [item['Feed'] for item in sequence],
                'arrivals': [item['Arrival'] for item in sorted_by_time],
                'has_duplicates': has_duplicates,
                'output_range': f"{min_output:.3f} to {max_output:.3f} (spread: {output_range_spread:.3f})",
                'group_size': len(sequence),
                'is_descending': True,  # Already validated by find_temporal_descending_sequences
                'day_type': day_classification
            }
            
            # Check if sequence ends with M# 50 and Anchor origin
            ends_with_m50_anchor = ends_with_m50_and_anchor(sequence)
            
            # Check if sequence ends with opposite M# values (+50/-50 or -50/+50) - G.06
            ends_with_opposite_m50 = ends_with_opposite_m50_pair(sequence)
            
            # Debug logging for G.06 (only when enabled)
            if st.session_state.get('debug_g06', False):
                st.write(f"Debug - Processing sequence with {len(sequence)} items")
                st.write(f"Debug - M# values: {[float(item['M #']) for item in sorted_by_time]}")
                st.write(f"Debug - Origins: {[item['Origin'] for item in sorted_by_time]}")
                st.write(f"Debug - Output values: {[float(item['Output']) for item in sorted_by_time]}")
                st.write(f"Debug - Days: {[item['Day'] for item in sorted_by_time]}")
                st.write(f"Debug - Feeds: {[item['Feed'] for item in sorted_by_time]}")
                st.write(f"Debug - G.05 check (M50+Anchor): {ends_with_m50_anchor}")
                st.write(f"Debug - G.06 check (Opposite M50): {ends_with_opposite_m50}")
                if len(sorted_by_time) >= 2:
                    st.write(f"Debug - Last two M# values: {[float(sorted_by_time[-2]['M #']), float(sorted_by_time[-1]['M #'])]}")
                    st.write(f"Debug - Final origin type: {get_origin_type(sorted_by_time[-1]['Origin'])}")
                st.write("---")
            
            # G.06 classification: sequences with opposite M# endings (independent check)
            if ends_with_opposite_m50:
                if day_classification == '[0]':
                    sequence_info['category'] = 'G.06.o1[0]'
                    results['G.06.o1[0]'].append(sequence_info)  # Today, opposite M# ending
                else:
                    sequence_info['category'] = 'G.06.o2[≠0]'
                    results['G.06.o2[≠0]'].append(sequence_info)  # Other days, opposite M# ending
            
            # G.05 classification: sequences that end with M50 + Anchor (independent check)
            if ends_with_m50_anchor:
                # Create a copy for G.05 since a sequence could match both G.05 and G.06
                g05_sequence_info = sequence_info.copy()
                if day_classification == '[0]':
                    g05_sequence_info['category'] = 'G.05.o1[0]'
                    results['G.05.o1[0]'].append(g05_sequence_info)  # Today, ends M50+Anchor
                else:
                    g05_sequence_info['category'] = 'G.05.o2[≠0]'
                    results['G.05.o2[≠0]'].append(g05_sequence_info)  # Other days, ends M50+Anchor
            
            # If sequence doesn't meet either G.05 or G.06 criteria, add to rejected
            if not ends_with_m50_anchor and not ends_with_opposite_m50:
                # Store rejected sequences for debugging
                last_m_value = float(sorted_by_time[-1]["M #"])
                second_last_m_value = float(sorted_by_time[-2]["M #"]) if len(sorted_by_time) >= 2 else None
                rejection_reason = f'Does not meet G.05 (M50+Anchor) or G.06 (opposite M# pair) criteria'
                if second_last_m_value is not None:
                    rejection_reason += f' (ends with M#{last_m_value}, M#{second_last_m_value} + {get_origin_type(sorted_by_time[-1]["Origin"])})'
                
                results['rejected_groups'].append({
                    'outputs': sequence_info['outputs'],
                    'reasons': [rejection_reason],
                    'output_range': sequence_info['output_range'],
                    'sequence_details': f"M# sequence: {sequence_info['m_values']}, Origins: {sequence_info['origins']}"
                })
    
    return results

def ends_with_opposite_m50_pair(sequence):
    """
    Check for G.06 pattern with updated requirements
    
    G.06 Requirements:
    - Minimum sequence length of 5 elements 
    - Sequence must end with TWO M# 50 values with opposite polarities
    - Final entry must have Anchor origin (Spain, Saturn, Jupiter, Kepler-62, Kepler-44)
    - All M# values before the last TWO must be 100% strictly descending by absolute value in chronological order
    - All M# values before the last TWO must have the same polarity as the second-to-last M# value
    
    Valid examples:
    - [96, 68, 60, 50, -50] (positive sequence ending with +50, -50)
    - [-96, -68, -60, -50, +50] (negative sequence ending with -50, +50)
    """
    if len(sequence) < 5:
        if st.session_state.get('debug_g06', False):
            st.write(f"G.06 Debug: Sequence too short ({len(sequence)} < 5)")
        return False
        
    # Sort by arrival time to ensure proper chronological order
    def get_sort_key(item):
        arrival = item['Arrival']
        if hasattr(arrival, 'isoformat'):
            return arrival
        else:
            try:
                return pd.to_datetime(arrival)
            except:
                return arrival
    
    sorted_by_time = sorted(sequence, key=get_sort_key)
    
    # Get M# values in chronological order
    m_values = [float(item['M #']) for item in sorted_by_time]
    
    # Must have at least 5 values
    if len(m_values) < 5:
        return False
    
    # Check that last two values are both M# 50 with opposite polarities
    second_last_m = m_values[-2]
    final_m = m_values[-1]
    
    if st.session_state.get('debug_g06', False):
        st.write(f"G.06 Debug: Second-to-last M#: {second_last_m}, Final M#: {final_m}")
        st.write(f"G.06 Debug: Second-to-last abs: {abs(second_last_m)}, Final abs: {abs(final_m)}")
    
    if abs(second_last_m) != 50 or abs(final_m) != 50:
        if st.session_state.get('debug_g06', False):
            st.write(f"G.06 Debug: FAILED - Not both M# 50 values")
        return False
    
    # Check that last two have opposite polarities
    if (second_last_m > 0) == (final_m > 0):
        if st.session_state.get('debug_g06', False):
            st.write(f"G.06 Debug: FAILED - Same polarity: {second_last_m > 0} == {final_m > 0}")
        return False
    
    # Check that final entry has Anchor origin
    final_entry = sorted_by_time[-1]
    if get_origin_type(final_entry['Origin']) != 'Anchor':
        return False
    
    # Get all preceding values (exclude last TWO M# values)
    preceding_values = m_values[:-2]
    
    if st.session_state.get('debug_g06', False):
        st.write(f"G.06 Debug: Preceding M# values: {preceding_values}")
        st.write(f"G.06 Debug: Second-to-last M# polarity: {second_last_m > 0}")
    
    # Check that all preceding values have same polarity as second-to-last
    second_last_polarity = second_last_m > 0
    for i, m in enumerate(preceding_values):
        if (m > 0) != second_last_polarity:  # Must have same polarity as second-to-last
            if st.session_state.get('debug_g06', False):
                st.write(f"G.06 Debug: FAILED - Preceding M#{m} has different polarity from second-to-last")
            return False
    
    # Check that preceding values are strictly descending by absolute value
    preceding_abs_values = [abs(m) for m in preceding_values]
    if preceding_abs_values != sorted(preceding_abs_values, reverse=True):
        return False
    
    return True

# def run_model_g_detection(df, proximity_threshold=0.10):
#     """Run Model G detection and display results in Streamlit"""
    
#     st.subheader("🔍 Model G Detection - Proximity-Based Traveler Grouping")
    
#     # Show dataset info for large datasets
#     if len(df) > 10000:
#         st.info(f"📊 Processing dataset: {len(df):,} rows | Using optimized Python algorithm for large datasets")
    
#     # Proximity input and display controls
#     col1, col2 = st.columns([1, 3])
#     with col1:
#         proximity = st.number_input(
#             "Proximity Threshold",
#             min_value=0.001,
#             max_value=10.0,
#             value=proximity_threshold,
#             step=0.001,
#             format="%.3f",
#             help="Maximum distance between outputs to be considered in same group"
#         )
    
#     with col2:
#         st.info(f"Grouping outputs within ±{proximity:.3f} of each other")
    
#     # Display control toggles
#     st.markdown("### 🎛️ Display Controls")
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         show_o1 = st.checkbox("G.05.o1[0]: Today M50+Anchor", value=True, key="g_show_o1")
#     with col2:
#         show_o2 = st.checkbox("G.05.o2[≠0]: Other M50+Anchor", value=True, key="g_show_o2")
#     with col3:
#         show_rejected = st.checkbox("❌ Rejected Groups", value=False, key="g_show_rejected")
    
#     # Run detection
#     results = detect_model_g_sequences(df, proximity)
    
#     # Display results summary
#     st.markdown("### 📊 Detection Summary")
    
#     # Categorize sequences for Recent vs Other Days display
#     def categorize_other_day_sequences(sequences):
#         recent = []  # [-1] to [-5]
#         other_days = []  # [-6] and more negative
        
#         for seq_info in sequences:
#             days = seq_info['days']
#             day_nums = []
#             for day_str in days:
#                 try:
#                     day_num = int(day_str.strip('[]'))
#                     day_nums.append(day_num)
#                 except:
#                     day_nums.append(-999)
            
#             most_recent_day = max(day_nums) if day_nums else -999
            
#             if -5 <= most_recent_day <= -1:
#                 recent.append(seq_info)
#             else:
#                 other_days.append(seq_info)
        
#         return recent, other_days

#     # Get recent vs other days breakdown for M50+Anchor category
#     o2_recent, o2_other_days = categorize_other_day_sequences(results['G.05.o2[≠0]'])
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total Proximity Groups", len(results['proximity_groups']))
#     with col2:
#         st.metric("G.05.o1[0]: Today M50+Anchor", len(results['G.05.o1[0]']))
#     with col3:
#         st.metric("G.05.o2[≠0]: Other M50+Anchor", len(results['G.05.o2[≠0]']))
    


#     # Create cluster table for enabled categories
#     enabled_sequences = []
#     if show_o1:
#         enabled_sequences.extend(results['G.05.o1[0]'])
#     if show_o2:
#         enabled_sequences.extend(results['G.05.o2[≠0]'])
    
#     if enabled_sequences:
#         st.markdown("### 📊 Cluster Table")
        
#         # Collect all sequences for cluster analysis
#         all_sequences = enabled_sequences
        
#         # Create cluster data by grouping sequences by their final output
#         cluster_data = {}
        
#         for seq_info in all_sequences:
#             # Get the final output (last in chronological order)
#             final_output = seq_info['outputs'][-1]  # Last output in the sequence
            
#             if final_output not in cluster_data:
#                 cluster_data[final_output] = {
#                     'sequences': [],
#                     'largest_length': 0,
#                     'unique_count': 0,
#                     'final_arrival': None,
#                     'final_day': None,
#                     'final_origin': None
#                 }
            
#             # Add this sequence to the cluster
#             cluster_data[final_output]['sequences'].append(seq_info)
            
#             # Update largest sequence length
#             seq_length = len(seq_info['outputs'])
#             if seq_length > cluster_data[final_output]['largest_length']:
#                 cluster_data[final_output]['largest_length'] = seq_length
#                 # Store arrival info from the largest sequence
#                 cluster_data[final_output]['final_arrival'] = seq_info['arrivals'][-1]
#                 cluster_data[final_output]['final_day'] = seq_info['days'][-1]
#                 cluster_data[final_output]['final_origin'] = seq_info['origins'][-1]
            
#             # Count unique sequences
#             cluster_data[final_output]['unique_count'] = len(cluster_data[final_output]['sequences'])
        
#         # Build cluster table data
#         cluster_rows = []
#         for output, cluster_info in cluster_data.items():
#             # Get representative sequence (the largest one)
#             largest_seq = max(cluster_info['sequences'], key=lambda x: len(x['outputs']))
            
#             # Get the end-of-sequence row data from the original dataframe
#             final_output = largest_seq['outputs'][-1]  # Last output in the sequence
#             final_arrival = largest_seq['arrivals'][-1]  # Last arrival time
            
#             # Find the corresponding row in the dataframe for end-of-sequence values
#             end_sequence_row = None
#             for idx, row in df.iterrows():
#                 if (abs(row['Output'] - final_output) < 0.001 and  # Match output within tolerance
#                     pd.to_datetime(row['Arrival']) == pd.to_datetime(final_arrival)):  # Match arrival time
#                     end_sequence_row = row
#                     break
            
#             # Extract input values from the end-of-sequence row
#             if end_sequence_row is not None:
#                 input_at_18 = end_sequence_row.get('Input @ 18:00', "")
#                 diff_18 = end_sequence_row.get('Diff @ 18:00', "")
#                 input_at_arrival = end_sequence_row.get('Input @ Arrival', "")
#                 diff_arrival = end_sequence_row.get('Diff @ Arrival', "")
#                 input_at_report = end_sequence_row.get('Input @ Report', "")
#                 diff_report = end_sequence_row.get('Diff @ Report', "")
#             else:
#                 # Fallback if no matching row found
#                 input_at_18 = ""
#                 diff_18 = ""
#                 input_at_arrival = ""
#                 diff_arrival = ""
#                 input_at_report = ""
#                 diff_report = ""
            
#             # Format arrival time into separate day and datetime columns
#             if cluster_info['final_arrival']:
#                 try:
#                     arrival_dt = pd.to_datetime(cluster_info['final_arrival'])
#                     if pd.notna(arrival_dt):  # Check if it's not NaT
#                         day_abbrev = arrival_dt.strftime('%a')  # Mon, Tue, Wed, etc.
#                         arrival_excel = arrival_dt.strftime('%d-%b-%Y %H:%M')  # Excel-friendly format
                        
#                         # Calculate hours ago from current time
#                         hours_ago = (pd.Timestamp.now() - arrival_dt).total_seconds() / 3600
#                         hours_ago_str = f"{int(hours_ago)} hours ago"
#                     else:
#                         day_abbrev = ""
#                         arrival_excel = str(cluster_info['final_arrival'])
#                         hours_ago_str = ""
#                 except:
#                     day_abbrev = ""
#                     arrival_excel = str(cluster_info['final_arrival'])
#                     hours_ago_str = ""
#             else:
#                 day_abbrev = ""
#                 arrival_excel = ""
#                 hours_ago_str = ""
            
#             # Create pattern sequences description
#             pattern_sequences = []
#             for seq in cluster_info['sequences']:
#                 m_values = [str(m) for m in seq['m_values']]
#                 origins = [str(o) for o in seq['origins']]
                
#                 # Get M names from the original dataframe by matching M# values
#                 m_names = []
#                 for m_val in seq['m_values']:
#                     # Find corresponding M name in the dataframe
#                     matching_rows = df[df['M #'] == m_val]
#                     if not matching_rows.empty:
#                         m_name = matching_rows.iloc[0].get('M Name', f'M{m_val}')
#                         m_names.append(str(m_name))
#                     else:
#                         m_names.append(f'M{m_val}')
                
#                 if len(seq['outputs']) >= 3:  # Only show sequences with 3+ elements
#                     pattern_desc = f"G.05 | M Names: {','.join(m_names)} | M#: {','.join(m_values)} | Origins: {','.join(origins)}"
#                     pattern_sequences.append(pattern_desc)
            
#             pattern_str = " | ".join(pattern_sequences) if pattern_sequences else "No valid patterns"
            
#             cluster_rows.append({
#                 'Output': f"{output:.3f}",
#                 'Largest sequence length': cluster_info['largest_length'],
#                 'Unique Sequences': cluster_info['unique_count'],
#                 'Stars': "",  # Blank for now
#                 'Booster Score': "",  # Blank for now
#                 'Input @ 18:00': input_at_18,
#                 '18:00 Diff': diff_18,
#                 'Input @ Arrival': input_at_arrival,
#                 'Arrival Diff': diff_arrival,
#                 'Input @ Report': input_at_report,
#                 'Report Diff': diff_report,
#                 'ddd': day_abbrev,  # Day abbreviation (Mon, Tue, Wed, etc.)
#                 'Arrival': arrival_excel,  # Excel-friendly datetime format
#                 'Day': cluster_info['final_day'] or "",
#                 'Hours ago': hours_ago_str,
#                 'Pattern Sequences': pattern_str
#             })
        
#         # Sort by output value (descending)
#         cluster_rows.sort(key=lambda x: float(x['Output']), reverse=True)
        
#         # Display cluster table
#         if cluster_rows:
#             cluster_df = pd.DataFrame(cluster_rows)
#             st.dataframe(cluster_df, use_container_width=True)
            
#             # Download button for cluster table with report time in filename
#             try:
#                 # Try to get report time from the dataframe
#                 if 'Arrival' in df.columns and not df.empty:
#                     arrival_times = pd.to_datetime(df['Arrival'], format='%d-%b-%Y %H:%M', errors='coerce')
#                     # Filter out NaT values and get the maximum valid time
#                     valid_times = arrival_times.dropna()
#                     if len(valid_times) > 0:
#                         report_time = valid_times.max()
#                         timestamp = report_time.strftime("%y-%m-%d_%H-%M")
#                     else:
#                         timestamp = pd.Timestamp.now().strftime('%y-%m-%d_%H-%M')
#                 else:
#                     timestamp = pd.Timestamp.now().strftime('%y-%m-%d_%H-%M')
#             except:
#                 timestamp = pd.Timestamp.now().strftime('%y-%m-%d_%H-%M')
                
#             cluster_csv = cluster_df.to_csv(index=False)
#             st.download_button(
#                 label="📥 Download Cluster Table (CSV)",
#                 data=cluster_csv,
#                 file_name=f"model_g_cluster_table_{timestamp}.csv",
#                 mime="text/csv"
#             )
#         else:
#             st.info("No cluster data available")
        
#         st.markdown("---")

#     # Display classified sequences
#     if enabled_sequences:
#         st.markdown("### ✅ Classified Sequences")
        
#         # Prepare data for download
#         all_classified_data = []
        
#         # Collect all enabled sequences
#         for category_name, sequences in [
#             ('G.05.o1[0]', results['G.05.o1[0]'] if show_o1 else []),
#             ('G.05.o2[≠0]', results['G.05.o2[≠0]'] if show_o2 else [])
#         ]:
#             for seq in sequences:
#                 # Create rows from the sequence data structure
#                 for i, output in enumerate(seq['outputs']):
#                     row = {
#                         'Output': output,
#                         'Origin': seq['origins'][i] if i < len(seq['origins']) else '',
#                         'M #': seq['m_values'][i] if i < len(seq['m_values']) else '',
#                         'Day': seq['days'][i] if i < len(seq['days']) else '',
#                         'Arrival': seq['arrivals'][i] if i < len(seq['arrivals']) else '',
#                         'Feed': seq['feeds'][i] if i < len(seq['feeds']) else '',
#                         'Model': category_name,
#                         'Classification': category_name.split('[')[0]  # Extract base name
#                     }
#                     all_classified_data.append(row)
        
#         # Create download DataFrame if we have data
#         if all_classified_data:
            
#             download_df = pd.DataFrame(all_classified_data)
#             # Reorder columns for better readability
#             column_order = ['Model', 'Classification', 'Output', 'Origin', 'M #', 'Day', 'Arrival', 'Feed', 'M Name', 'R #', 'Tag', 'Family']
#             download_columns = [col for col in column_order if col in download_df.columns]
#             download_df = download_df[download_columns]
            
#             # Sort by Model then by Output (descending)
#             download_df = download_df.sort_values(['Model', 'Output'], ascending=[True, False])
            
#             # Create Excel file with proper date formatting
#             excel_buffer = io.BytesIO()
#             with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
#                 # Ensure Arrival column is properly formatted as datetime for Excel
#                 if 'Arrival' in download_df.columns:
#                     download_df['Arrival'] = pd.to_datetime(download_df['Arrival'])
                
#                 download_df.to_excel(writer, sheet_name='Model G Results', index=False)
                
#                 # Get workbook and worksheet for formatting
#                 workbook = writer.book
#                 worksheet = writer.sheets['Model G Results']
                
#                 # Add some basic formatting
#                 header_format = workbook.add_format({
#                     'bold': True,
#                     'text_wrap': True,
#                     'valign': 'top',
#                     'fg_color': '#D7E4BC',
#                     'border': 1
#                 })
                
#                 # Write headers with formatting
#                 for col_num, value in enumerate(download_df.columns.values):
#                     worksheet.write(0, col_num, value, header_format)
                
#                 # Format Arrival column as datetime in Excel
#                 if 'Arrival' in download_df.columns:
#                     arrival_col = list(download_df.columns).index('Arrival')
#                     date_format = workbook.add_format({'num_format': 'yyyy-mm-dd hh:mm:ss'})
#                     worksheet.set_column(arrival_col, arrival_col, 20, date_format)
                
#                 # Auto-adjust column widths
#                 for i, col in enumerate(download_df.columns):
#                     max_length = max(
#                         download_df[col].astype(str).map(len).max(),
#                         len(str(col))
#                     ) + 2
#                     worksheet.set_column(i, i, min(max_length, 50))
            
#             # Generate timestamp for filename
#             timestamp = datetime.now().strftime("%y-%m-%d_%H-%M")
#             filename = f"model_g_results_{timestamp}.xlsx"
            
#             # Download button
#             st.download_button(
#                 label="📥 Download Model G Results (Excel)",
#                 data=excel_buffer.getvalue(),
#                 file_name=filename,
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )
        
#         # Display each category based on toggles
#         if show_o1 and results['G.05.o1[0]']:
#             with st.expander(f"🟢 G.05.o1[0] - Today M50+Anchor ({len(results['G.05.o1[0]'])})"):
#                 for i, seq_info in enumerate(results['G.05.o1[0]']):
#                     st.markdown(f"**Sequence {i+1}:**")
                    
#                     outputs = [f"{x:.3f}" for x in seq_info['outputs']]
#                     m_values = seq_info['m_values']
#                     abs_m_values = [abs(float(m)) for m in m_values]
#                     origins = seq_info['origins']
#                     days = seq_info['days']
#                     arrivals = seq_info['arrivals']
                    
#                     st.markdown(f"- **Output Range:** {seq_info['output_range']}")
#                     st.markdown(f"- **Outputs:** {outputs}")
#                     st.markdown(f"- **M# sequence:** {m_values}")
#                     st.markdown(f"- **Absolute M# values:** {abs_m_values}")
#                     st.markdown(f"- **Origins:** {origins}")
#                     st.markdown(f"- **Days:** {days}")
#                     st.markdown(f"- **Group Size:** {seq_info['group_size']} travelers")
#                     st.markdown(f"- **Is Descending:** {'✅ Yes' if seq_info['is_descending'] else '❌ No'}")
                    
#                     # Show time span
#                     if arrivals:
#                         first_arrival = arrivals[0]
#                         last_arrival = arrivals[-1]
#                         st.markdown(f"- **Time span:** {first_arrival} → {last_arrival}")
                    
#                     # Create DataFrame for display
#                     display_data = {
#                         'Arrival': arrivals,
#                         'Output': seq_info['outputs'],
#                         'Origin': origins,
#                         'M #': m_values,
#                         'Day': days,
#                         'Feed': seq_info['feeds']
#                     }
#                     display_df = pd.DataFrame(display_data)
#                     st.dataframe(display_df, use_container_width=True)
#                     st.markdown("---")
        
#         if show_o2 and results['G.05.o2[≠0]']:
#             # Split o2 into Recent and Other Days for display
#             o2_recent, o2_other = categorize_other_day_sequences(results['G.05.o2[≠0]'])
            
#             # Recent sequences ([-1] to [-5])
#             if o2_recent:
#                 with st.expander(f"🔶 G.05.o2[Recent] M50+Anchor ([-1] to [-5]) ({len(o2_recent)})"):
#                     for i, seq_info in enumerate(o2_recent):
#                         st.markdown(f"**Sequence {i+1}:**")
                        
#                         # Extract data from the new sequence_info structure
#                         outputs = [f"{x:.3f}" for x in seq_info['outputs']]
#                     m_values = seq_info['m_values']
#                     abs_m_values = [abs(float(m)) for m in m_values]
#                     origins = seq_info['origins']
#                     days = seq_info['days']
#                     arrivals = seq_info['arrivals']
                    
#                     st.markdown(f"- **Output Range:** {seq_info['output_range']}")
#                     st.markdown(f"- **Outputs:** {outputs}")
#                     st.markdown(f"- **M# sequence:** {m_values}")
#                     st.markdown(f"- **Absolute M# values:** {abs_m_values}")
#                     st.markdown(f"- **Origins:** {origins}")
#                     st.markdown(f"- **Days:** {days}")
#                     st.markdown(f"- **Group Size:** {seq_info['group_size']} travelers")
#                     st.markdown(f"- **Is Descending:** {'✅ Yes' if seq_info['is_descending'] else '❌ No'}")
                    
#                     # Show time span
#                     if arrivals:
#                         first_arrival = arrivals[0]
#                         last_arrival = arrivals[-1]
#                         st.markdown(f"- **Time span:** {first_arrival} → {last_arrival}")
                    
#                     # Create DataFrame for display
#                     display_data = {
#                         'Arrival': arrivals,
#                         'Output': seq_info['outputs'],
#                         'Origin': origins,
#                         'M #': m_values,
#                         'Day': days,
#                         'Feed': seq_info['feeds']
#                     }
#                     display_df = pd.DataFrame(display_data)
#                     st.dataframe(display_df, use_container_width=True)
#                     st.markdown("---")
                    
#             # Other day sequences ([-6] and beyond)  
#             if o2_other:
#                 with st.expander(f"🔵 G.05.o2[Other] M50+Anchor ([-6] and beyond) ({len(o2_other)})"):
#                     for i, seq_info in enumerate(o2_other):
#                         st.markdown(f"**Sequence {i+1}:**")
                        
#                         # Extract data from the new sequence_info structure
#                         outputs = [f"{x:.3f}" for x in seq_info['outputs']]
#                     m_values = seq_info['m_values']
#                     abs_m_values = [abs(float(m)) for m in m_values]
#                     origins = seq_info['origins']
#                     days = seq_info['days']
#                     arrivals = seq_info['arrivals']
                    
#                     st.markdown(f"- **Output Range:** {seq_info['output_range']}")
#                     st.markdown(f"- **Outputs:** {outputs}")
#                     st.markdown(f"- **M# sequence:** {m_values}")
#                     st.markdown(f"- **Absolute M# values:** {abs_m_values}")
#                     st.markdown(f"- **Origins:** {origins}")
#                     st.markdown(f"- **Days:** {days}")
#                     st.markdown(f"- **Group Size:** {seq_info['group_size']} travelers")
#                     st.markdown(f"- **Is Descending:** {'✅ Yes' if seq_info['is_descending'] else '❌ No'}")
                    
#                     # Show time span
#                     if arrivals:
#                         first_arrival = arrivals[0]
#                         last_arrival = arrivals[-1]
#                         st.markdown(f"- **Time span:** {first_arrival} → {last_arrival}")
                    
#                     # Create DataFrame for display
#                     display_data = {
#                         'Arrival': arrivals,
#                         'Output': seq_info['outputs'],
#                         'Origin': origins,
#                         'M #': m_values,
#                         'Day': days,
#                         'Feed': seq_info['feeds']
#                     }
#                     display_df = pd.DataFrame(display_data)
#                     st.dataframe(display_df, use_container_width=True)
#                     st.markdown("---")

    
#     # Display debugging information (conditionally displayed)
#     if show_rejected and results['rejected_groups']:
#         with st.expander(f"❌ Rejected Groups ({len(results['rejected_groups'])}) - Debug Info"):
#             for i, group_info in enumerate(results['rejected_groups']):
#                 st.markdown(f"**Rejected Group {i+1}:**")
#                 # Sort outputs in descending order for display
#                 sorted_outputs = sorted(group_info['outputs'], reverse=True)
#                 st.markdown(f"- Outputs: {[f'{x:.3f}' for x in sorted_outputs]}")
#                 st.markdown(f"- Output Range: {group_info['output_range']}")
#                 st.markdown(f"- Rejection reasons: {', '.join(group_info['reasons'])}")
#                 st.markdown("---")
    
#     return results

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
