# Simple Single Line Mega Report - Uses direct imports from existing files
# More reliable approach using the existing model files

import streamlit as st
import pandas as pd
from collections import defaultdict

# Direct imports from existing files
try:
    from models.models_a_today import detect_A_models_today_only, classify_A_model, find_flexible_descents
    A_MODELS_AVAILABLE = True
except ImportError:
    A_MODELS_AVAILABLE = False

try:
    from models.mod_c_04gpr3 import detect_C_models, classify_c01_sequence, classify_c02_sequence, classify_c04_sequence, find_descending_sequences
    C_MODELS_AVAILABLE = True
except ImportError:
    C_MODELS_AVAILABLE = False

def analyze_output_patterns_simple(output_rows):
    """
    Simplified pattern analysis using existing functions
    """
    patterns_found = []
    
    # Sort rows by arrival time
    rows = output_rows.sort_values("Arrival").reset_index(drop=True)
    
    # Check A model patterns if available
    if A_MODELS_AVAILABLE:
        try:
            sequences = find_flexible_descents(rows)
            for seq in sequences:
                if seq.shape[0] >= 2:
                    last = seq.iloc[-1]
                    prior = seq.iloc[:-1] if seq.shape[0] > 1 else pd.DataFrame()
                    
                    # Only process if ends on 'today' ([0])
                    final_day = str(last["Day"]).strip()
                    if final_day == "[0]":
                        model, label = classify_A_model(last, prior)
                        if model:
                            m_path = " → ".join([f"|{row['M #']}|" for _, row in seq.iterrows()])
                            patterns_found.append(f"A:{model}({m_path})")
        except Exception as e:
            st.warning(f"Error in A model detection: {e}")
    
    # Check C model patterns if available
    if C_MODELS_AVAILABLE:
        try:
            # Create mini dataframe for this output
            mini_df = rows.copy()
            mini_df['Output'] = output_rows['Output'].iloc[0]
            
            # Use existing C model detection
            c_results = detect_C_models(mini_df, run_c01=True, run_c02=True, run_c04=True)
            
            for tag, items in c_results.items():
                for item in items:
                    seq = item['sequence']
                    m_path = " → ".join([f"|{row['M #']}|" for _, row in seq.iterrows()])
                    patterns_found.append(f"C:{tag}({m_path})")
                    
        except Exception as e:
            st.warning(f"Error in C model detection: {e}")
    
    return patterns_found

def generate_simple_mega_report(df):
    """
    Generate single-line report using existing model functions
    """
    st.subheader("🎯 Single Line Mega Report - All Patterns Per Output")
    
    if not A_MODELS_AVAILABLE and not C_MODELS_AVAILABLE:
        st.error("No model detection functions available")
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
            
        patterns = analyze_output_patterns_simple(output_rows)
        
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

def run_simple_single_line_analysis(df):
    """Main function to run simple single line mega report analysis"""
    try:
        return generate_simple_mega_report(df)
    except Exception as e:
        st.error(f"Error generating mega report: {e}")
        st.write("Traceback:", str(e))
        return None
