"""
G Model Manager - Unified interface for all G model detection
Coordinates execution of G.05, G.06, G.08 and future G models
"""

import streamlit as st
import pandas as pd
from model_g_05_06 import run_g05_g06_detection
from model_g_08 import run_g08_detection
from model_g_09 import run_g09_detection

def run_model_g_detection(df, proximity_threshold=0.10, report_time=None, key_suffix=""):
    """
    Unified G Model Detection Entry Point
    Runs all available G model detectors and consolidates results
    """
    
    if df.empty:
        st.warning("⚠️ No data available for Model G detection")
        return {}
    
    st.write("### 🔍 Model G Detection Results")
    
    # Initialize consolidated results
    all_results = {
        'g05_g06': {},
        'g08': {},
        # Placeholder for future models
        'g09': {},
        'g10': {},
        'g11': {},
        'g12': {}
    }
    
    # Run G.05/G.06 Detection
    with st.expander("G.05 & G.06 Detection", expanded=True):
        st.write("**Standard proximity grouping with descending sequences**")
        try:
            g05_g06_results = run_g05_g06_detection(df, proximity_threshold)
            all_results['g05_g06'] = g05_g06_results
            
            # Display results summary
            total_today = len(g05_g06_results.get('today_sequences', []))
            total_other = len(g05_g06_results.get('other_day_sequences', []))
            total_rejected = len(g05_g06_results.get('rejected_groups', []))
            
            st.write(f"- **Today sequences:** {total_today}")
            st.write(f"- **Other day sequences:** {total_other}")
            st.write(f"- **Rejected groups:** {total_rejected}")
            
        except Exception as e:
            st.error(f"G.05/G.06 Detection Error: {str(e)}")
            all_results['g05_g06'] = {'error': str(e)}
    
    # Run G.08 Detection
    with st.expander("G.08 Detection", expanded=True):
        st.write("**x0Pd.w pattern recognition with Group 1B filtering**")
        try:
            g08_results = run_g08_detection(df, proximity_threshold)
            all_results['g08'] = g08_results
            
            # Display results summary
            total_today = len(g08_results.get('today_sequences', []))
            total_other = len(g08_results.get('other_day_sequences', []))
            
            # Count by category
            today_by_category = {}
            other_by_category = {}
            
            for seq in g08_results.get('today_sequences', []):
                cat = seq.get('category', 'Unknown')
                today_by_category[cat] = today_by_category.get(cat, 0) + 1
            
            for seq in g08_results.get('other_day_sequences', []):
                cat = seq.get('category', 'Unknown')
                other_by_category[cat] = other_by_category.get(cat, 0) + 1
            
            st.write(f"- **Today sequences:** {total_today}")
            if today_by_category:
                for cat, count in sorted(today_by_category.items()):
                    st.write(f"  - {cat}: {count}")
            
            st.write(f"- **Other day sequences:** {total_other}")
            if other_by_category:
                for cat, count in sorted(other_by_category.items()):
                    st.write(f"  - {cat}: {count}")
                    
        except Exception as e:
            st.error(f"G.08 Detection Error: {str(e)}")
            all_results['g08'] = {'error': str(e)}
    
    # Run G.09 Detection
    with st.expander("G.09 Detection", expanded=True):
        st.write("**x0Pd.w descending patterns with flip endings**")
        try:
            g09_results = run_g09_detection(df, proximity_threshold)
            all_results['g09'] = g09_results
            
            # Display results summary
            total_today = len(g09_results.get('today_sequences', []))
            total_other = len(g09_results.get('other_day_sequences', []))
            
            # Count by category
            today_by_category = {}
            other_by_category = {}
            
            for seq in g09_results.get('today_sequences', []):
                cat = seq.get('category', 'Unknown')
                today_by_category[cat] = today_by_category.get(cat, 0) + 1
            
            for seq in g09_results.get('other_day_sequences', []):
                cat = seq.get('category', 'Unknown')
                other_by_category[cat] = other_by_category.get(cat, 0) + 1
            
            st.write(f"- **Today sequences:** {total_today}")
            if today_by_category:
                for cat, count in sorted(today_by_category.items()):
                    st.write(f"  - {cat}: {count}")
            
            st.write(f"- **Other day sequences:** {total_other}")
            if other_by_category:
                for cat, count in sorted(other_by_category.items()):
                    st.write(f"  - {cat}: {count}")
                    
        except Exception as e:
            st.error(f"G.09 Detection Error: {str(e)}")
            all_results['g09'] = {'error': str(e)}
    
    # Placeholder sections for future G models
    with st.expander("Future G Models (G.10 - G.12)", expanded=False):
        st.write("**Placeholder for additional G model implementations**")
        st.write("- G.10: TBD") 
        st.write("- G.11: TBD")
        st.write("- G.12: TBD")
    
    return all_results

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
