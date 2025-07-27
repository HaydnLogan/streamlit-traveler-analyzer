# Single Line Mega Report - Consolidates all model detections per output
# Creates one row per output showing all found patterns/sequences
# More efficient than running full model detection on entire dataset

import streamlit as st
import pandas as pd
from collections import defaultdict
import importlib.util
import sys
import os

# Import model detection functions
def import_model_functions():
    """Import all model detection functions from the attached files"""
    functions = {}
    
    # Import A model functions
    try:
        spec = importlib.util.spec_from_file_location("models_a", "attached_assets/models_a_today_1753623650438.py")
        models_a = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models_a)
        functions['detect_A_models'] = models_a.detect_A_models_today_only
        functions['classify_A_model'] = models_a.classify_A_model
        functions['find_flexible_descents'] = models_a.find_flexible_descents
    except Exception as e:
        st.warning(f"Could not import A model functions: {e}")
    
    # Import B model functions  
    try:
        spec = importlib.util.spec_from_file_location("mod_b", "attached_assets/mod_b_05pg1_1753623650437.py")
        mod_b = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod_b)
        functions['detect_B_models'] = mod_b.detect_B_models
        functions['classify_b_sequence'] = mod_b.classify_b_sequence
        functions['find_descending_sequences_b'] = mod_b.find_descending_sequences
    except Exception as e:
        st.warning(f"Could not import B model functions: {e}")
    
    # Import C model functions
    try:
        spec = importlib.util.spec_from_file_location("mod_c", "attached_assets/mod_c_04gpr3_1753623650438.py")
        mod_c = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod_c)
        functions['detect_C_models'] = mod_c.detect_C_models
        functions['classify_c01_sequence'] = mod_c.classify_c01_sequence
        functions['classify_c02_sequence'] = mod_c.classify_c02_sequence
        functions['classify_c04_sequence'] = mod_c.classify_c04_sequence
        functions['find_descending_sequences_c'] = mod_c.find_descending_sequences
    except Exception as e:
        st.warning(f"Could not import C model functions: {e}")
    
    # Import X model functions
    try:
        spec = importlib.util.spec_from_file_location("mod_x", "attached_assets/mod_x_03g_1753623650438.py")
        mod_x = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod_x)
        functions['detect_X_models'] = mod_x.detect_X_models
        functions['classify_x00_at_40'] = mod_x.classify_x00_at_40
        functions['classify_vip_01'] = mod_x.classify_vip_01
    except Exception as e:
        st.warning(f"Could not import X model functions: {e}")
        
    return functions

def analyze_output_patterns(output_rows, model_functions):
    """
    Analyze all rows for a specific output and find all model patterns
    This is much more efficient than running full model detection
    """
    patterns_found = []
    
    # Sort rows by arrival time
    rows = output_rows.sort_values("Arrival").reset_index(drop=True)
    
    # Check A model patterns
    if 'classify_A_model' in model_functions and 'find_flexible_descents' in model_functions:
        try:
            sequences = model_functions['find_flexible_descents'](rows)
            for seq in sequences:
                if seq.shape[0] >= 2:  # At least 2 rows for A models
                    last = seq.iloc[-1]
                    prior = seq.iloc[:-1] if seq.shape[0] > 1 else pd.DataFrame()
                    
                    # Only process if ends on 'today' ([0])
                    final_day = str(last["Day"]).strip()
                    if final_day == "[0]":
                        model, label = model_functions['classify_A_model'](last, prior)
                        if model:
                            m_path = " → ".join([f"|{row['M #']}|" for _, row in seq.iterrows()])
                            patterns_found.append(f"A:{model}({m_path})")
        except Exception as e:
            st.warning(f"Error in A model detection: {e}")
    
    # Check B model patterns
    if 'classify_b_sequence' in model_functions and 'find_descending_sequences_b' in model_functions:
        try:
            # Create a mini dataframe for this output to use existing B model logic
            mini_df = rows.copy()
            mini_df['Output'] = output_rows['Output'].iloc[0]  # Ensure consistent output value
            sequences = model_functions['find_descending_sequences_b'](pd.DataFrame({'Output': [mini_df['Output'].iloc[0]]}))
            
            for output_val, seq in sequences:
                if seq.shape[0] >= 3:
                    label_code, label_text = model_functions['classify_b_sequence'](seq)
                    if label_code:
                        m_path = " → ".join([f"|{row['M #']}|" for _, row in seq.iterrows()])
                        patterns_found.append(f"B:{label_code}({m_path})")
        except Exception as e:
            st.warning(f"Error in B model detection: {e}")
    
    # Check C model patterns
    if 'classify_c01_sequence' in model_functions and 'classify_c02_sequence' in model_functions and 'classify_c04_sequence' in model_functions:
        try:
            # Test all possible 3-point combinations for C models
            for i in range(len(rows)):
                for j in range(i+1, len(rows)):
                    for k in range(j+1, len(rows)):
                        three_points = pd.DataFrame([rows.loc[i], rows.loc[j], rows.loc[k]]).reset_index(drop=True)
                        
                        # Try each C model classifier
                        for classifier_name in ['classify_c01_sequence', 'classify_c02_sequence', 'classify_c04_sequence']:
                            try:
                                tag, label = model_functions[classifier_name](three_points)
                                if tag:
                                    m_path = " → ".join([f"|{row['M #']}|" for _, row in three_points.iterrows()])
                                    patterns_found.append(f"C:{tag}({m_path})")
                                    break  # Only classify with first matching classifier
                            except Exception:
                                continue
                                
            # Also test descending sequences for C models
            if 'find_descending_sequences_c' in model_functions:
                mini_df = rows.copy()
                mini_df['Output'] = output_rows['Output'].iloc[0]
                sequences = model_functions['find_descending_sequences_c'](pd.DataFrame({'Output': [mini_df['Output'].iloc[0]]}))
                
                for output_val, seq in sequences:
                    for classifier_name in ['classify_c01_sequence', 'classify_c02_sequence', 'classify_c04_sequence']:
                        try:
                            tag, label = model_functions[classifier_name](seq)
                            if tag:
                                m_path = " → ".join([f"|{row['M #']}|" for _, row in seq.iterrows()])
                                patterns_found.append(f"C:{tag}({m_path})")
                                break
                        except Exception:
                            continue
                            
        except Exception as e:
            st.warning(f"Error in C model detection: {e}")
    
    # Check X model patterns
    if 'classify_x00_at_40' in model_functions and 'classify_vip_01' in model_functions:
        try:
            # Test descending sequences for X models
            for i in range(len(rows)):
                path = []
                abs_seen = set()
                for j in range(i, len(rows)):
                    m = rows.loc[j, "M #"]
                    abs_m = abs(m)
                    if abs_m in abs_seen:
                        continue
                    if path and abs_m >= abs(path[-1]["M #"]):
                        continue
                    abs_seen.add(abs_m)
                    path.append(rows.loc[j])
                    if len(path) >= 3:
                        seq_df = pd.DataFrame(path).reset_index(drop=True)
                        
                        # Try X model classifiers
                        for classifier in [model_functions['classify_x00_at_40'], model_functions['classify_vip_01']]:
                            try:
                                tag, label = classifier(seq_df)
                                if tag:
                                    m_path = " → ".join([f"|{row['M #']}|" for _, row in seq_df.iterrows()])
                                    patterns_found.append(f"X:{tag}({m_path})")
                                    break
                            except Exception:
                                continue
                                
        except Exception as e:
            st.warning(f"Error in X model detection: {e}")
    
    return patterns_found

def generate_single_line_mega_report(df):
    """
    Generate single-line report showing all patterns found per output
    Much more efficient than running full model detection on entire dataset
    """
    st.subheader("🎯 Single Line Mega Report - All Patterns Per Output")
    
    # Import model functions
    model_functions = import_model_functions()
    
    if not model_functions:
        st.error("Could not import model detection functions")
        return
    
    # Group by output and analyze patterns
    mega_report_data = []
    outputs = df["Output"].unique()
    
    progress_bar = st.progress(0)
    for i, output in enumerate(outputs):
        output_rows = df[df["Output"] == output]
        
        # Skip if only one row (no patterns possible)
        if len(output_rows) < 2:
            continue
            
        patterns = analyze_output_patterns(output_rows, model_functions)
        
        if patterns:  # Only include outputs with found patterns
            # Get metadata
            latest_arrival = output_rows["Arrival"].max()
            row_count = len(output_rows)
            origins = ", ".join(output_rows["Origin"].unique())
            feeds = ", ".join(output_rows["Feed"].unique())
            
            mega_report_data.append({
                "Output": f"{output:,.3f}",
                "Rows": row_count,
                "Pattern Count": len(patterns),
                "Pattern Sequences": " | ".join(patterns),
                "Origins": origins,
                "Feeds": feeds,
                "Latest Arrival": latest_arrival.strftime('%a %y-%m-%d %H:%M')
            })
        
        progress_bar.progress((i + 1) / len(outputs))
    
    if not mega_report_data:
        st.warning("No patterns found in any outputs")
        return
    
    # Create DataFrame and sort by output descending
    mega_df = pd.DataFrame(mega_report_data)
    mega_df_sorted = mega_df.sort_values(
        by="Output", 
        key=lambda x: x.str.replace(",", "").astype(float), 
        ascending=False
    )
    
    st.write(f"Found patterns in {len(mega_df_sorted)} outputs")
    st.dataframe(mega_df_sorted, use_container_width=True)
    
    # Download button
    csv_data = mega_df_sorted.to_csv(index=False)
    st.download_button(
        "📥 Download Single Line Mega Report (CSV)",
        data=csv_data,
        file_name="single_line_mega_report.csv",
        mime="text/csv"
    )
    
    return mega_df_sorted

def run_single_line_analysis(df):
    """Main function to run single line mega report analysis"""
    try:
        return generate_single_line_mega_report(df)
    except Exception as e:
        st.error(f"Error generating mega report: {e}")
        st.write("Traceback:", str(e))
        return None
