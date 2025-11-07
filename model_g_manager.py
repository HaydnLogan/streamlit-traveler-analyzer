# 11.6.2025 Model G.11 added. 
"""  8.16.25 Model G toggles FIXED
G Model Manager - Unified interface for all G model detection
Coordinates execution of G.05, G.06, G.08 and future G models
"""

import streamlit as st
import pandas as pd
from model_g_05_06 import run_g05_g06_detection
from model_g_08 import run_g08_detection
from model_g_09 import run_g09_detection
from model_g_10 import run_g10_detection
from model_g_11 import run_g11_detection

def run_model_g_detection(df, proximity_threshold=0.10, report_time=None, key_suffix="", 
                         run_g05_g06=True, run_g08=True, run_g09=True, run_g10=False, run_g11=False,
                         g10_group_0=True, g10_group_1=True, g10_group_2=True, g10_group_3=False, g10_group_4=False,
                         g11_threshold=3.0, g11_group_0=True, g11_group_1=True, g11_group_2=True, g11_group_3=True, g11_group_4=True,
                         g11_display_recipes=True, g11_display_others=True):
    """
    Unified G Model Detection Entry Point
    Runs all available G model detectors and consolidates results
    Returns format expected by main app: {'success': bool, 'summary': dict, 'results_df': DataFrame, 'error': str}
    """

    if df.empty:
        st.warning("⚠️ No data available for Model G detection")
        return {
            'success': False,
            'error': 'No data available for Model G detection',
            'summary': {'total_o1': 0, 'total_o2': 0, 'total_sequences': 0},
            'results_df': pd.DataFrame()
        }

    # 🔧 normalize types once
    norm = df.copy()
    if "Arrival_datetime" in norm.columns:
        norm["Arrival_datetime"] = pd.to_datetime(norm["Arrival_datetime"], errors="coerce")
    elif "Arrival" in norm.columns:
        norm["Arrival_datetime"] = pd.to_datetime(norm["Arrival"], errors="coerce")
    for col in ("Output", "M #"):
        if col in norm.columns:
            norm[col] = pd.to_numeric(norm[col], errors="coerce")

    st.write("### 🔍 Model G Detection Results")

    # Initialize consolidated results
    all_results = {
        'g05_g06': {},
        'g08': {},
        'g09': {},
        'g10': {},
        'g11': {},
        'g12': {}
    }

    # Track totals for main app summary
    total_today_sequences = 0
    total_other_sequences = 0
    total_sequences = 0
    results_list = []

    try:
        # Run G.05/G.06 Detection
        if run_g05_g06:
            with st.expander("G.05 & G.06 Detection", expanded=True):
                st.write("**Standard proximity grouping with descending sequences**")
                try:
                    g05_g06_results = run_g05_g06_detection(df, proximity_threshold)
                    all_results['g05_g06'] = g05_g06_results

                    # Display results summary
                    today_count = len(g05_g06_results.get('today_sequences', []))
                    other_count = len(g05_g06_results.get('other_day_sequences', []))
                    rejected_count = len(g05_g06_results.get('rejected_groups', []))

                    total_today_sequences += today_count
                    total_other_sequences += other_count

                    st.write(f"- **Today sequences:** {today_count}")
                    st.write(f"- **Other day sequences:** {other_count}")
                    st.write(f"- **Rejected groups:** {rejected_count}")

                    # Add to results list for DataFrame
                    for seq in g05_g06_results.get('today_sequences', []):
                        outputs = seq.get('outputs', [])
                        arrival_output = max(outputs) if outputs else None
                        results_list.append({
                            'Arrival_Output': arrival_output,
                            'Model': 'G.05/G.06',
                            'Type': 'Today',
                            'Category': seq.get('category', 'Unknown'),
                            'Origins': ', '.join(seq.get('origins', [])),
                            'M_#s': ', '.join(map(str, seq.get('m_values', []))),
                            'Outputs': ', '.join([f"{x:.2f}" for x in seq.get('outputs', [])])
                        })

                    for seq in g05_g06_results.get('other_day_sequences', []):
                        outputs = seq.get('outputs', [])
                        arrival_output = max(outputs) if outputs else None
                        results_list.append({
                            'Arrival_Output': arrival_output,
                            'Model': 'G.05/G.06',
                            'Type': 'Other Day',
                            'Category': seq.get('category', 'Unknown'),
                            'Origins': ', '.join(seq.get('origins', [])),
                            'M_#s': ', '.join(map(str, seq.get('m_values', []))),
                            'Outputs': ', '.join([f"{x:.2f}" for x in seq.get('outputs', [])])
                        })

                except Exception as e:
                    st.error(f"G.05/G.06 Detection Error: {str(e)}")
                    all_results['g05_g06'] = {'error': str(e)}
        else:
            st.info("G.05/G.06 Detection disabled")

        # Run G.08 Detection
        if run_g08:
            with st.expander("G.08 Detection", expanded=True):
                st.write("**x0Pd.w pattern recognition with Group 1B filtering**")
                try:
                    g08_results = run_g08_detection(df, proximity_threshold)
                    all_results['g08'] = g08_results

                    # Display results summary
                    today_count = len(g08_results.get('today_sequences', []))
                    other_count = len(g08_results.get('other_day_sequences', []))

                    total_today_sequences += today_count
                    total_other_sequences += other_count

                    # Count by category
                    today_by_category = {}
                    other_by_category = {}

                    for seq in g08_results.get('today_sequences', []):
                        cat = seq.get('category', 'Unknown')
                        today_by_category[cat] = today_by_category.get(cat, 0) + 1

                    for seq in g08_results.get('other_day_sequences', []):
                        cat = seq.get('category', 'Unknown')
                        other_by_category[cat] = other_by_category.get(cat, 0) + 1

                    st.write(f"- **Today sequences:** {today_count}")
                    if today_by_category:
                        for cat, count in sorted(today_by_category.items()):
                            st.write(f"  - {cat}: {count}")

                    st.write(f"- **Other day sequences:** {other_count}")
                    if other_by_category:
                        for cat, count in sorted(other_by_category.items()):
                            st.write(f"  - {cat}: {count}")

                    # Add to results list for DataFrame
                    for seq in g08_results.get('today_sequences', []):
                        outputs = seq.get('outputs', [])
                        arrival_output = max(outputs) if outputs else None
                        results_list.append({
                            'Arrival_Output': arrival_output,
                            'Model': 'G.08',
                            'Type': 'Today',
                            'Category': seq.get('category', 'Unknown'),
                            'Origins': ', '.join(map(str, seq.get('origins', []))),
                            'M_#s': ', '.join(map(str, seq.get('m_values', []))),
                            'Outputs': ', '.join([f"{x:.2f}" for x in seq.get('outputs', [])]),
                            'End_Type': seq.get('end_type', 'Unknown'),
                            'Arrivals': ', '.join(map(str, seq.get('arrivals', [])))
                        })

                    for seq in g08_results.get('other_day_sequences', []):
                        outputs = seq.get('outputs', [])
                        arrival_output = max(outputs) if outputs else None
                        results_list.append({
                            'Arrival_Output': arrival_output,
                            'Model': 'G.08',
                            'Type': 'Other Day',
                            'Category': seq.get('category', 'Unknown'),
                            'Origins': ', '.join(map(str, seq.get('origins', []))),
                            'M_#s': ', '.join(map(str, seq.get('m_values', []))),
                            'Outputs': ', '.join([f"{x:.2f}" for x in seq.get('outputs', [])]),
                            'End_Type': seq.get('end_type', 'Unknown'),
                            'Arrivals': ', '.join(map(str, seq.get('arrivals', [])))
                        })

                    # Show detailed sequences immediately if any found
                    if today_count > 0 or other_count > 0:
                        st.write("### 📋 **G.08 Sequences Found:**")

                        if g08_results.get('today_sequences'):
                            st.write("**Today Sequences:**")
                            for i, seq in enumerate(g08_results['today_sequences']):
                                with st.expander(f"G.08 {seq.get('category', 'Unknown')} - Sequence {i+1}", expanded=True):
                                    st.write(f"**Origins:** {seq.get('origins', [])}")
                                    st.write(f"**M# Values:** {seq.get('m_values', [])}")
                                    st.write(f"**Outputs:** {seq.get('outputs', [])}")
                                    st.write(f"**End Type:** {seq.get('end_type', 'Unknown')}")
                                    st.write(f"**Arrivals:** {seq.get('arrivals', [])}")

                        if g08_results.get('other_day_sequences'):
                            st.write("**Other Day Sequences:**")
                            for i, seq in enumerate(g08_results['other_day_sequences']):
                                with st.expander(f"G.08 {seq.get('category', 'Unknown')} - Sequence {i+1}", expanded=True):
                                    st.write(f"**Origins:** {seq.get('origins', [])}")
                                    st.write(f"**M# Values:** {seq.get('m_values', [])}")
                                    st.write(f"**Outputs:** {seq.get('outputs', [])}")
                                    st.write(f"**End Type:** {seq.get('end_type', 'Unknown')}")
                                    st.write(f"**Arrivals:** {seq.get('arrivals', [])}")

                except Exception as e:
                    st.error(f"G.08 Detection Error: {str(e)}")
                    all_results['g08'] = {'error': str(e)}
        else:
            st.info("G.08 Detection disabled")

        # Run G.09 Detection
        if run_g09:
            with st.expander("G.09 Detection", expanded=True):
                st.write("**x0Pd.w descending patterns with flip endings**")
                try:
                    g09_results = run_g09_detection(df, proximity_threshold)
                    all_results['g09'] = g09_results

                    # Display results summary
                    today_count = len(g09_results.get('today_sequences', []))
                    other_count = len(g09_results.get('other_day_sequences', []))

                    total_today_sequences += today_count
                    total_other_sequences += other_count

                    # Count by category
                    today_by_category = {}
                    other_by_category = {}

                    for seq in g09_results.get('today_sequences', []):
                        cat = seq.get('category', 'Unknown')
                        today_by_category[cat] = today_by_category.get(cat, 0) + 1

                    for seq in g09_results.get('other_day_sequences', []):
                        cat = seq.get('category', 'Unknown')
                        other_by_category[cat] = other_by_category.get(cat, 0) + 1

                    st.write(f"- **Today sequences:** {today_count}")
                    if today_by_category:
                        for cat, count in sorted(today_by_category.items()):
                            st.write(f"  - {cat}: {count}")

                    st.write(f"- **Other day sequences:** {other_count}")
                    if other_by_category:
                        for cat, count in sorted(other_by_category.items()):
                            st.write(f"  - {cat}: {count}")

                    # Add to results list for DataFrame
                    for seq in g09_results.get('today_sequences', []):
                        outputs = seq.get('outputs', [])
                        arrival_output = max(outputs) if outputs else None
                        results_list.append({
                            'Arrival_Output': arrival_output,
                            'Model': 'G.09',
                            'Type': 'Today',
                            'Category': seq.get('category', 'Unknown'),
                            'Origins': ', '.join(map(str, seq.get('origins', []))),
                            'M_#s': ', '.join(map(str, seq.get('m_values', []))),
                            'Outputs': ', '.join([f"{x:.2f}" for x in seq.get('outputs', [])]),
                            'Star_Origins': seq.get('star_origin_count', 0),
                            'Opposite_Flip': 'Yes' if seq.get('has_opposite_flip', False) else 'No',
                            'Excludes_Small': 'Yes' if seq.get('excludes_small', False) else 'No'
                        })

                    for seq in g09_results.get('other_day_sequences', []):
                        outputs = seq.get('outputs', [])
                        arrival_output = max(outputs) if outputs else None
                        results_list.append({
                            'Arrival_Output': arrival_output,
                            'Model': 'G.09',
                            'Type': 'Other Day',
                            'Category': seq.get('category', 'Unknown'),
                            'Origins': ', '.join(map(str, seq.get('origins', []))),
                            'M_#s': ', '.join(map(str, seq.get('m_values', []))),
                            'Outputs': ', '.join([f"{x:.2f}" for x in seq.get('outputs', [])]),
                            'Star_Origins': seq.get('star_origin_count', 0),
                            'Opposite_Flip': 'Yes' if seq.get('has_opposite_flip', False) else 'No',
                            'Excludes_Small': 'Yes' if seq.get('excludes_small', False) else 'No'
                        })

                    # Show detailed sequences if any found
                    if today_count > 0 or other_count > 0:
                        if st.button("Show G.09 Sequence Details", key="g09_details"):
                            display_g_model_details(all_results, "g09")

                except Exception as e:
                    st.error(f"G.09 Detection Error: {str(e)}")
                    all_results['g09'] = {'error': str(e)}
        else:
            st.info("G.09 Detection disabled")

        # Run G.10 Detection  
        if run_g10:
            with st.expander("G.10 Detection", expanded=True):
                st.write("**Pair Detection with Neighbor Scoring (GR, x0, x1, Fogz, Zero, Premium, DD patterns)**")
                try:
                    # Create group filter based on enabled groups
                    enabled_groups = []
                    if g10_group_0: enabled_groups.append(0)
                    if g10_group_1: enabled_groups.append(1)
                    if g10_group_2: enabled_groups.append(2)
                    if g10_group_3: enabled_groups.append(3)
                    if g10_group_4: enabled_groups.append(4)
                    
                    st.write(f"**Enabled Groups:** {enabled_groups}")
                    g10_results = run_g10_detection(df, proximity_threshold, enabled_groups=enabled_groups)
                    all_results['g10'] = g10_results

                    # Display results summary
                    today_count = len(g10_results.get('today_sequences', []))
                    other_count = len(g10_results.get('other_day_sequences', []))

                    total_today_sequences += today_count
                    total_other_sequences += other_count

                    # Count by category
                    today_by_category = {}
                    other_by_category = {}

                    for seq in g10_results.get('today_sequences', []):
                        cat = seq.get('classification', 'Unknown')
                        today_by_category[cat] = today_by_category.get(cat, 0) + 1

                    for seq in g10_results.get('other_day_sequences', []):
                        cat = seq.get('classification', 'Unknown')
                        other_by_category[cat] = other_by_category.get(cat, 0) + 1

                    st.write(f"- **Today sequences:** {today_count}")
                    if today_by_category:
                        for cat, count in sorted(today_by_category.items()):
                            st.write(f"  - {cat}: {count}")

                    st.write(f"- **Other day sequences:** {other_count}")
                    if other_by_category:
                        for cat, count in sorted(other_by_category.items()):
                            st.write(f"  - {cat}: {count}")

                    # Add to results list for DataFrame
                    for seq in g10_results.get('today_sequences', []):
                        outputs = seq.get('outputs', [])
                        arrival_output = max(outputs) if outputs else None
                        results_list.append({
                            'Arrival_Output': arrival_output,
                            'Model': 'G.10 (Pair Detection)',
                            'Type': 'Today',
                            'Category': seq.get('classification', 'Unknown'),
                            'Origins': seq.get('origins', ''),  # Already a string
                            'M_#s': ', '.join(map(str, seq.get('m_values', []))),
                            'Outputs': ', '.join([f"{x:.2f}" for x in seq.get('outputs', [])]),
                            'Pattern_Type': seq.get('type', 'Unknown'),
                            'Group': seq.get('group', 'Unknown'),
                            'Base_Score': seq.get('base_score', 0),
                            'Neighbor_Boost': seq.get('neighbor_boost', 0),
                            'Total_Score': seq.get('total_score', 0),
                            'Is_Recipe': 'Yes' if seq.get('is_recipe', False) else 'No'
                        })

                    for seq in g10_results.get('other_day_sequences', []):
                        outputs = seq.get('outputs', [])
                        arrival_output = max(outputs) if outputs else None
                        results_list.append({
                            'Arrival_Output': arrival_output,
                            'Model': 'G.10 (Pair Detection)',
                            'Type': 'Other Day',
                            'Category': seq.get('classification', 'Unknown'),
                            'Origins': seq.get('origins', ''),  # Already a string
                            'M_#s': ', '.join(map(str, seq.get('m_values', []))),
                            'Outputs': ', '.join([f"{x:.2f}" for x in seq.get('outputs', [])]),
                            'Pattern_Type': seq.get('type', 'Unknown'),
                            'Group': seq.get('group', 'Unknown'),
                            'Base_Score': seq.get('base_score', 0),
                            'Neighbor_Boost': seq.get('neighbor_boost', 0),
                            'Total_Score': seq.get('total_score', 0),
                            'Is_Recipe': 'Yes' if seq.get('is_recipe', False) else 'No'
                        })

                    # Show detailed sequences if any found
                    if today_count > 0 or other_count > 0:
                        if st.button("Show G.10 Sequence Details", key="g10_details"):
                            display_g_model_details(all_results, "g10")

                except Exception as e:
                    st.error(f"G.10 Detection Error: {str(e)}")
                    all_results['g10'] = {'error': str(e)}
        else:
            st.info("G.10 Detection disabled")

        # Run G.11 Detection  
        if run_g11:
            with st.expander("G.11 Detection (Pair Detection sTF)", expanded=True):
                st.write("**Pair Detection with Same Origin Requirement (GR, x0, x1, Fogz & Ps, Zero, Premiums, DD Fogz & D patterns)**")
                st.write(f"**Proximity Threshold:** {g11_threshold} hours")
                try:
                    # Create group filter based on enabled groups
                    enabled_groups = []
                    if g11_group_0: enabled_groups.append(0)
                    if g11_group_1: enabled_groups.append(1)
                    if g11_group_2: enabled_groups.append(2)
                    if g11_group_3: enabled_groups.append(3)
                    if g11_group_4: enabled_groups.append(4)
                    
                    st.write(f"**Enabled Groups:** {['Grp 0 TA', 'Grp 1 sAA', 'Grp 2 AA', 'Grp 3 oA', 'Grp 4 Ao'][i] for i in enabled_groups}")
                    st.write(f"**Display Recipes:** {g11_display_recipes}, **Display Others:** {g11_display_others}")
                    
                    g11_results = run_g11_detection(
                        df, 
                        proximity_threshold=g11_threshold, 
                        enabled_groups=enabled_groups,
                        display_recipes=g11_display_recipes,
                        display_others=g11_display_others
                    )
                    all_results['g11'] = g11_results

                    # Display results summary
                    today_count = len(g11_results.get('today_sequences', []))
                    other_count = len(g11_results.get('other_day_sequences', []))

                    total_today_sequences += today_count
                    total_other_sequences += other_count

                    # Count by category
                    today_by_category = {}
                    other_by_category = {}

                    for seq in g11_results.get('today_sequences', []):
                        cat = seq.get('classification', 'Unknown')
                        today_by_category[cat] = today_by_category.get(cat, 0) + 1

                    for seq in g11_results.get('other_day_sequences', []):
                        cat = seq.get('classification', 'Unknown')
                        other_by_category[cat] = other_by_category.get(cat, 0) + 1

                    st.write(f"- **Today sequences:** {today_count}")
                    if today_by_category:
                        for cat, count in sorted(today_by_category.items()):
                            st.write(f"  - {cat}: {count}")

                    st.write(f"- **Other day sequences:** {other_count}")
                    if other_by_category:
                        for cat, count in sorted(other_by_category.items()):
                            st.write(f"  - {cat}: {count}")

                    # Add to results list for DataFrame
                    for seq in g11_results.get('today_sequences', []):
                        outputs = seq.get('outputs', [])
                        arrival_output = max(outputs) if outputs else None
                        results_list.append({
                            'Arrival_Output': arrival_output,
                            'Model': 'G.11 (Pair Detection sTF)',
                            'Type': 'Today',
                            'Category': seq.get('classification', 'Unknown'),
                            'Origins': seq.get('origins', ''),  # Already a string
                            'M_#s': ', '.join(map(str, seq.get('m_values', []))),
                            'Outputs': ', '.join([f"{x:.2f}" for x in seq.get('outputs', [])]),
                            'Pattern_Type': seq.get('type', 'Unknown'),
                            'Group': seq.get('group', 'Unknown'),
                            'Base_Score': seq.get('base_score', 0),
                            'Neighbor_Boost': seq.get('neighbor_boost', 0),
                            'Total_Score': seq.get('total_score', 0),
                            'Is_Recipe': 'Yes' if seq.get('is_recipe', False) else 'No'
                        })

                    for seq in g11_results.get('other_day_sequences', []):
                        outputs = seq.get('outputs', [])
                        arrival_output = max(outputs) if outputs else None
                        results_list.append({
                            'Arrival_Output': arrival_output,
                            'Model': 'G.11 (Pair Detection sTF)',
                            'Type': 'Other Day',
                            'Category': seq.get('classification', 'Unknown'),
                            'Origins': seq.get('origins', ''),  # Already a string
                            'M_#s': ', '.join(map(str, seq.get('m_values', []))),
                            'Outputs': ', '.join([f"{x:.2f}" for x in seq.get('outputs', [])]),
                            'Pattern_Type': seq.get('type', 'Unknown'),
                            'Group': seq.get('group', 'Unknown'),
                            'Base_Score': seq.get('base_score', 0),
                            'Neighbor_Boost': seq.get('neighbor_boost', 0),
                            'Total_Score': seq.get('total_score', 0),
                            'Is_Recipe': 'Yes' if seq.get('is_recipe', False) else 'No'
                        })

                    # Show detailed sequences if any found
                    if today_count > 0 or other_count > 0:
                        if st.button("Show G.11 Sequence Details", key="g11_details"):
                            display_g_model_details(all_results, "g11")

                except Exception as e:
                    st.error(f"G.11 Detection Error: {str(e)}")
                    all_results['g11'] = {'error': str(e)}
        else:
            st.info("G.11 Detection disabled")

        # Placeholder sections for future G models
        with st.expander("Future G Models (G.12+)", expanded=False):
            st.write("**Placeholder for additional G model implementations**")
            st.write("- G.12: TBD")

        # Calculate totals
        total_sequences = total_today_sequences + total_other_sequences

        # Create results DataFrame
        results_df = pd.DataFrame(results_list) if results_list else pd.DataFrame()
        
        # Add Prox column - absolute difference between max and min outputs
        if not results_df.empty and 'Outputs' in results_df.columns:
            def calculate_prox(outputs_str):
                try:
                    # Parse the outputs string to get individual values
                    outputs = [float(x.strip()) for x in outputs_str.split(',')]
                    if len(outputs) >= 2:
                        return abs(max(outputs) - min(outputs))
                    return 0.0
                except:
                    return 0.0
            
            results_df['Prox'] = results_df['Outputs'].apply(calculate_prox)
        
        # Update Type column to show "Today", "Recent", or "Old" based on Day column logic
        if not results_df.empty and 'Type' in results_df.columns and not df.empty:
            def determine_type(row):
                if row['Type'] == 'Today':
                    return 'Today'
                
                # For "Other Day", check the Day column in the original df
                # Match by Arrival_Output to find corresponding row(s) in df
                if 'Arrival_Output' in row and row['Arrival_Output'] is not None:
                    # Find matching records in df
                    if 'Output' in df.columns and 'Day' in df.columns:
                        matching_rows = df[abs(df['Output'] - row['Arrival_Output']) < 0.01]
                        if not matching_rows.empty:
                            # Get unique Day values for this sequence
                            day_values = matching_rows['Day'].unique()
                            
                            # Check if we have [0], [-1], [-2] pattern or gaps
                            if 0 in day_values:
                                return 'Today'
                            elif -1 in day_values:
                                return 'Recent'
                            elif any(d < -1 and d >= -2 for d in day_values):
                                return 'Recent'
                            elif any(d == -3 for d in day_values):
                                # Check if there's a gap (no -1, -2)
                                if -1 not in day_values and -2 not in day_values:
                                    return 'Recent'
                                else:
                                    return 'Old'
                            else:
                                return 'Old'
                
                # Default to Old for anything that wasn't Today and couldn't be determined
                return 'Old' if row['Type'] == 'Other Day' else row['Type']
            
            results_df['Type'] = results_df.apply(determine_type, axis=1)
        
        # Sort by Arrival_Output descending if the column exists
        if not results_df.empty and 'Arrival_Output' in results_df.columns:
            # Fill NaN values with -999999 so they appear at the bottom when sorting descending
            results_df['Arrival_Output_Sort'] = results_df['Arrival_Output'].fillna(-999999)
            results_df = results_df.sort_values('Arrival_Output_Sort', ascending=False)
            results_df = results_df.drop('Arrival_Output_Sort', axis=1)

        # Return format expected by main app
        return {
            'success': True,
            'summary': {
                'total_o1': total_today_sequences,
                'total_o2': total_other_sequences, 
                'total_sequences': total_sequences
            },
            'results_df': results_df,
            'raw_results': all_results  # Include raw results for detailed access
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'summary': {'total_o1': 0, 'total_o2': 0, 'total_sequences': 0},
            'results_df': pd.DataFrame()
        }

def display_g_model_details(results, model_type="all"):
    """
    Display detailed results for specific G model types
    """

    if model_type == "all" or model_type == "g05_g06":
        if 'g05_g06' in results and 'error' not in results['g05_g06']:
            st.subheader("G.05 & G.06 Detailed Results")
            g05_g06 = results['g05_g06']

            # Display today sequences
            if g05_g06.get('today_sequences'):
                st.write("**Today Sequences:**")
                for i, seq in enumerate(g05_g06['today_sequences']):
                    with st.expander(f"{seq['category']} - Sequence {i+1}"):
                        st.write(f"**Origins:** {', '.join(seq['origins'])}")
                        st.write(f"**M# Values:** {seq['m_values']}")
                        st.write(f"**Outputs:** {seq['outputs']}")

            # Display other day sequences  
            if g05_g06.get('other_day_sequences'):
                st.write("**Other Day Sequences:**")
                for i, seq in enumerate(g05_g06['other_day_sequences']):
                    with st.expander(f"{seq['category']} - Sequence {i+1}"):
                        st.write(f"**Origins:** {', '.join(seq['origins'])}")
                        st.write(f"**M# Values:** {seq['m_values']}")
                        st.write(f"**Outputs:** {seq['outputs']}")

    if model_type == "all" or model_type == "g08":
        if 'g08' in results and 'error' not in results['g08']:
            st.subheader("G.08 Detailed Results")
            g08 = results['g08']

            # Display today sequences
            if g08.get('today_sequences'):
                st.write("**Today Sequences:**")
                for i, seq in enumerate(g08['today_sequences']):
                    with st.expander(f"{seq['category']} - Sequence {i+1}"):
                        st.write(f"**Origins:** {seq['origins']}")
                        st.write(f"**M# Values:** {seq['m_values']}")
                        st.write(f"**Outputs:** {seq['outputs']}")
                        st.write(f"**End Type:** {seq.get('end_type', 'Unknown')}")

            # Display other day sequences
            if g08.get('other_day_sequences'):
                st.write("**Other Day Sequences:**")
                for i, seq in enumerate(g08['other_day_sequences']):
                    with st.expander(f"{seq['category']} - Sequence {i+1}"):
                        st.write(f"**Origins:** {seq['origins']}")
                        st.write(f"**M# Values:** {seq['m_values']}")
                        st.write(f"**Outputs:** {seq['outputs']}")
                        st.write(f"**End Type:** {seq.get('end_type', 'Unknown')}")

    if model_type == "all" or model_type == "g09":
        if 'g09' in results and 'error' not in results['g09']:
            st.subheader("G.09 Detailed Results")
            g09 = results['g09']

            # Display today sequences
            if g09.get('today_sequences'):
                st.write("**Today Sequences:**")
                for i, seq in enumerate(g09['today_sequences']):
                    with st.expander(f"{seq['category']} - Sequence {i+1}"):
                        st.write(f"**Origins:** {seq['origins']}")
                        st.write(f"**M# Values:** {seq['m_values']}")
                        st.write(f"**Outputs:** {seq['outputs']}")
                        st.write(f"**Star Origins:** {seq.get('star_origin_count', 0)}")
                        st.write(f"**Opposite Flip:** {'Yes' if seq.get('has_opposite_flip', False) else 'No'}")
                        st.write(f"**Excludes Small:** {'Yes' if seq.get('excludes_small', False) else 'No'}")

            # Display other day sequences
            if g09.get('other_day_sequences'):
                st.write("**Other Day Sequences:**")
                for i, seq in enumerate(g09['other_day_sequences']):
                    with st.expander(f"{seq['category']} - Sequence {i+1}"):
                        st.write(f"**Origins:** {seq['origins']}")
                        st.write(f"**M# Values:** {seq['m_values']}")
                        st.write(f"**Outputs:** {seq['outputs']}")
                        st.write(f"**Star Origins:** {seq.get('star_origin_count', 0)}")
                        st.write(f"**Opposite Flip:** {'Yes' if seq.get('has_opposite_flip', False) else 'No'}")
                        st.write(f"**Excludes Small:** {'Yes' if seq.get('excludes_small', False) else 'No'}")

    if model_type == "all" or model_type == "g10":
        if 'g10' in results and 'error' not in results['g10']:
            st.subheader("G.10 Detailed Results")
            g10 = results['g10']

            # Display today sequences
            if g10.get('today_sequences'):
                st.write("**Today Sequences:**")
                for i, seq in enumerate(g10['today_sequences']):
                    with st.expander(f"{seq['classification']} - Sequence {i+1}"):
                        st.write(f"**Pattern Type:** {seq.get('type', 'Unknown')}")
                        st.write(f"**Origins:** {seq['origins']}")
                        st.write(f"**M# Values:** {seq['m_values']}")
                        st.write(f"**Outputs:** {seq['outputs']}")
                        st.write(f"**Group:** {seq.get('group', 'Unknown')}")
                        st.write(f"**Base Score:** {seq.get('base_score', 0)}")
                        st.write(f"**Neighbor Boost:** {seq.get('neighbor_boost', 0)}")
                        st.write(f"**Total Score:** {seq.get('total_score', 0)}")
                        st.write(f"**Is Recipe:** {'Yes' if seq.get('is_recipe', False) else 'No'}")

            # Display other day sequences
            if g10.get('other_day_sequences'):
                st.write("**Other Day Sequences:**")
                for i, seq in enumerate(g10['other_day_sequences']):
                    with st.expander(f"{seq['classification']} - Sequence {i+1}"):
                        st.write(f"**Pattern Type:** {seq.get('type', 'Unknown')}")
                        st.write(f"**Origins:** {seq['origins']}")
                        st.write(f"**M# Values:** {seq['m_values']}")
                        st.write(f"**Outputs:** {seq['outputs']}")
                        st.write(f"**Group:** {seq.get('group', 'Unknown')}")
                        st.write(f"**Base Score:** {seq.get('base_score', 0)}")
                        st.write(f"**Neighbor Boost:** {seq.get('neighbor_boost', 0)}")
                        st.write(f"**Total Score:** {seq.get('total_score', 0)}")
                        st.write(f"**Is Recipe:** {'Yes' if seq.get('is_recipe', False) else 'No'}")

    if model_type == "all" or model_type == "g11":
        if 'g11' in results and 'error' not in results['g11']:
            st.subheader("G.11 Detailed Results (Same Origin Pairs)")
            g11 = results['g11']

            # Display today sequences
            if g11.get('today_sequences'):
                st.write("**Today Sequences:**")
                for i, seq in enumerate(g11['today_sequences']):
                    with st.expander(f"{seq['classification']} - Sequence {i+1}"):
                        st.write(f"**Pattern Type:** {seq.get('type', 'Unknown')}")
                        st.write(f"**Origins:** {seq['origins']}")
                        st.write(f"**M# Values:** {seq['m_values']}")
                        st.write(f"**Outputs:** {seq['outputs']}")
                        st.write(f"**Group:** {seq.get('group', 'Unknown')}")
                        st.write(f"**Base Score:** {seq.get('base_score', 0)}")
                        st.write(f"**Neighbor Boost:** {seq.get('neighbor_boost', 0)}")
                        st.write(f"**Total Score:** {seq.get('total_score', 0)}")
                        st.write(f"**Is Recipe:** {'Yes' if seq.get('is_recipe', False) else 'No'}")

            # Display other day sequences
            if g11.get('other_day_sequences'):
                st.write("**Other Day Sequences:**")
                for i, seq in enumerate(g11['other_day_sequences']):
                    with st.expander(f"{seq['classification']} - Sequence {i+1}"):
                        st.write(f"**Pattern Type:** {seq.get('type', 'Unknown')}")
                        st.write(f"**Origins:** {seq['origins']}")
                        st.write(f"**M# Values:** {seq['m_values']}")
                        st.write(f"**Outputs:** {seq['outputs']}")
                        st.write(f"**Group:** {seq.get('group', 'Unknown')}")
                        st.write(f"**Base Score:** {seq.get('base_score', 0)}")
                        st.write(f"**Neighbor Boost:** {seq.get('neighbor_boost', 0)}")
                        st.write(f"**Total Score:** {seq.get('total_score', 0)}")
                        st.write(f"**Is Recipe:** {'Yes' if seq.get('is_recipe', False) else 'No'}")
