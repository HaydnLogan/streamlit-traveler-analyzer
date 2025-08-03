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
import io
from datetime import datetime

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
    Group outputs that are within proximity_threshold of each other using graph-based clustering.
    This allows for more flexible grouping where outputs don't need to be sequentially close
    but can form clusters through transitive proximity relationships.
    """
    if len(outputs) == 0:
        return []
    
    # Create adjacency list for proximity graph
    n = len(outputs)
    adjacency = [[] for _ in range(n)]
    
    # Build proximity graph - connect outputs within threshold
    for i in range(n):
        for j in range(i + 1, n):
            if abs(outputs[i]['Output'] - outputs[j]['Output']) <= proximity_threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)
    
    # Find connected components using DFS
    visited = [False] * n
    groups = []
     
    def dfs(node, current_group):
        visited[node] = True
        current_group.append(outputs[node])
        for neighbor in adjacency[node]:
            if not visited[neighbor]:
                dfs(neighbor, current_group)
    
    for i in range(n):
        if not visited[i]:
            current_group = []
            dfs(i, current_group)
            if len(current_group) >= 3:  # Only keep groups with 3+ items
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
        
        # Prepare data for download
        all_classified_data = []
        
        # Collect all classified sequences
        for seq in results['G.05.o1[0]']:
            for item in seq['group']:
                row = item.copy()
                row['Model'] = 'G.05.o1[0]'
                row['Classification'] = 'Today'
                all_classified_data.append(row)
        
        for seq in results['G.05.o2[≠0]']:
            for item in seq['group']:
                row = item.copy()
                row['Model'] = 'G.05.o2[≠0]'
                row['Classification'] = 'Other Days'
                all_classified_data.append(row)
        
        # Create download DataFrame if we have data
        if all_classified_data:
            
            download_df = pd.DataFrame(all_classified_data)
            # Reorder columns for better readability
            column_order = ['Model', 'Classification', 'Output', 'Origin', 'M #', 'Day', 'Arrival', 'Feed', 'M Name', 'R #', 'Tag', 'Family']
            download_columns = [col for col in column_order if col in download_df.columns]
            download_df = download_df[download_columns]
            
            # Sort by Model then by Output (descending)
            download_df = download_df.sort_values(['Model', 'Output'], ascending=[True, False])
            
            # Create Excel file
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                download_df.to_excel(writer, sheet_name='Model G Results', index=False)
                
                # Get workbook and worksheet for formatting
                workbook = writer.book
                worksheet = writer.sheets['Model G Results']
                
                # Add some basic formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                # Write headers with formatting
                for col_num, value in enumerate(download_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Auto-adjust column widths
                for i, col in enumerate(download_df.columns):
                    max_length = max(
                        download_df[col].astype(str).map(len).max(),
                        len(str(col))
                    ) + 2
                    worksheet.set_column(i, i, min(max_length, 50))
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%y-%m-%d_%H-%M")
            filename = f"model_g_results_{timestamp}.xlsx"
            
            # Download button
            st.download_button(
                label="📥 Download Model G Results (Excel)",
                data=excel_buffer.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
          
        # Today sequences
        if results['G.05.o1[0]']:
            with st.expander(f"🟢 G.05.o1[0] - Today Sequences ({len(results['G.05.o1[0]'])})"):
                for i, seq in enumerate(results['G.05.o1[0]']):
                    st.markdown(f"**Sequence {i+1}:**")
                    seq_df = pd.DataFrame(seq['group'])
                    seq_df = seq_df.sort_values('Output', ascending=False)  # Descending order by Output
                    st.dataframe(seq_df[['Output', 'Origin', 'M #', 'Day', 'Arrival']], use_container_width=True)
                    
                    info = seq['info']
                    # Sort outputs in descending order for display
                    sorted_outputs = sorted(info['outputs'], reverse=True)
                    st.markdown(f"- Outputs: {[f'{x:.3f}' for x in sorted_outputs]}")
                    st.markdown(f"- M# sequence: {info['m_numbers']}")
                    st.markdown(f"- Origins: {info['origins']}")
                    st.markdown("---")
        
        # Other day sequences
        if results['G.05.o2[≠0]']:
            with st.expander(f"🔵 G.05.o2[≠0] - Other Day Sequences ({len(results['G.05.o2[≠0]'])})"):
                for i, seq in enumerate(results['G.05.o2[≠0]']):
                    st.markdown(f"**Sequence {i+1}:**")
                    seq_df = pd.DataFrame(seq['group'])
                    seq_df = seq_df.sort_values('Output', ascending=False)  # Descending order by Output
                    st.dataframe(seq_df[['Output', 'Origin', 'M #', 'Day', 'Arrival']], use_container_width=True)
                    
                    info = seq['info']
                    # Sort outputs in descending order for display
                    sorted_outputs = sorted(info['outputs'], reverse=True)
                    st.markdown(f"- Outputs: {[f'{x:.3f}' for x in sorted_outputs]}")
                    st.markdown(f"- M# sequence: {info['m_numbers']}")
                    st.markdown(f"- Origins: {info['origins']}")
                    st.markdown("---")
    
    # Display debugging information
    if results['rejected_groups']:
        with st.expander(f"❌ Rejected Groups ({len(results['rejected_groups'])}) - Debug Info"):
            for i, group_info in enumerate(results['rejected_groups']):
                st.markdown(f"**Rejected Group {i+1}:**")
                st.markdown(f"- Size: {group_info['size']} travelers")
                # Sort outputs in descending order for display
                sorted_outputs = sorted(group_info['outputs'], reverse=True)
                st.markdown(f"- Outputs: {[f'{x:.3f}' for x in sorted_outputs]}")
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
                 
                # Show both arrival time order and output order
                group_by_arrival = sorted(group, key=lambda x: x['Arrival'])
                group_by_output = sorted(group, key=lambda x: x['Output'])
                
                st.markdown("**Ordered by Arrival Time:**")
                arrival_df = pd.DataFrame(group_by_arrival)
                st.dataframe(arrival_df[['Arrival', 'Output', 'Origin', 'M #', 'Day']], use_container_width=True)
                
                st.markdown("**Ordered by Output Value:**")
                output_df = pd.DataFrame(group_by_output)
                st.dataframe(output_df[['Output', 'Arrival', 'Origin', 'M #', 'Day']], use_container_width=True)
                
                # Show output range and M# sequence
                outputs = [item['Output'] for item in group]
                m_sequence_by_arrival = [item['M #'] for item in group_by_arrival]
                abs_m_sequence = [abs(float(m)) for m in m_sequence_by_arrival]
                
                st.markdown(f"- Output range: {min(outputs):.3f} to {max(outputs):.3f} (spread: {max(outputs) - min(outputs):.3f})")
                st.markdown(f"- M# sequence by arrival: {m_sequence_by_arrival}")
                st.markdown(f"- Absolute M# sequence: {abs_m_sequence}")
                st.markdown(f"- Is descending: {all(abs_m_sequence[j] > abs_m_sequence[j+1] for j in range(len(abs_m_sequence)-1))}")
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
