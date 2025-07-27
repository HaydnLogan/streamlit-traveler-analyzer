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

def calculate_comprehensive_metrics(output_rows, report_time):
    """
    Calculate all metrics from the Excel template
    """
    rows = output_rows.sort_values("Arrival").reset_index(drop=True)
    
    # Basic metrics
    total_m_count = len(rows)
    m_numbers = rows["M #"].tolist()
    m_ids = ", ".join([str(m) for m in m_numbers])
    m_start_end = f"{m_numbers[0]} → {m_numbers[-1]}" if len(m_numbers) > 1 else str(m_numbers[0])
    
    # Arrival order
    arrival_order = " → ".join([str(m) for m in m_numbers])
    
    # Strength travelers (M# values of 0, 40, -40, 54, -54)
    strength_values = {0, 40, -40, 54, -54}
    strength_travelers = [m for m in m_numbers if m in strength_values]
    strength_count = len(strength_travelers)
    strength_ids = ", ".join([str(m) for m in strength_travelers])
    
    # Tag B travelers (assuming non-strength travelers)
    tag_b_travelers = [m for m in m_numbers if m not in strength_values]
    tag_b_count = len(tag_b_travelers)
    tag_b_ids = ", ".join([str(m) for m in tag_b_travelers])
    
    # Sequence analysis
    origins = rows["Origin"].tolist()
    families = rows.get("Family", [""] * len(rows)).tolist()
    feeds = rows["Feed"].tolist()
    
    # Sequence characteristics
    same_origin = len(set(origins)) == 1
    same_family = len(set(families)) == 1 if all(f for f in families) else False
    same_feed = len(set(feeds)) == 1
    
    # Time calculations
    latest_arrival = rows["Arrival"].max()
    hours_ago = int((report_time - latest_arrival).total_seconds() / 3600)
    
    # Input values (if available in data)
    input_18 = rows.get("Input @ 18:00", [""]).iloc[-1] if "Input @ 18:00" in rows.columns else ""
    input_arrival = rows.get("Input @ Arrival", [""]).iloc[-1] if "Input @ Arrival" in rows.columns else ""
    input_report = rows.get("Input @ Report", [""]).iloc[-1] if "Input @ Report" in rows.columns else ""
    
    # Differences (if input columns exist)
    diff_18 = ""
    diff_arrival = ""
    diff_report = ""
    
    return {
        "total_m_count": total_m_count,
        "m_ids": m_ids,
        "m_start_end": m_start_end,
        "arrival_order": arrival_order,
        "strength_count": strength_count,
        "strength_ids": strength_ids,
        "tag_b_count": tag_b_count,
        "tag_b_ids": tag_b_ids,
        "same_origin": same_origin,
        "same_family": same_family,
        "same_feed": same_feed,
        "hours_ago": hours_ago,
        "latest_arrival": latest_arrival,
        "input_18": input_18,
        "input_arrival": input_arrival,
        "input_report": input_report,
        "diff_18": diff_18,
        "diff_arrival": diff_arrival,
        "diff_report": diff_report,
        "origins": origins,
        "families": families,
        "feeds": feeds
    }
def analyze_output_patterns_comprehensive(output_rows, report_time):
    """
    Comprehensive pattern analysis with all Excel template metrics
    """
    patterns_found = []
    sequences_found = []
    
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
                            sequences_found.append({
                                "type": "A",
                                "model": model,
                                "sequence": seq,
                                "length": len(seq)
                            })                          
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
                    sequences_found.append({
                        "type": "C",
                        "model": tag,
                        "sequence": seq,
                        "length": len(seq)
                    })                  
                    
        except Exception as e:
            st.warning(f"Error in C model detection: {e}")
    
    # Calculate additional sequence metrics
    unique_sequences = len(sequences_found)
    largest_seq_length = max([s["length"] for s in sequences_found]) if sequences_found else 0
    
    # Get largest sequence start/end
    largest_seq = None
    if sequences_found:
        largest_seq = max(sequences_found, key=lambda x: x["length"])
        seq_start_end = f"{largest_seq['sequence'].iloc[0]['M #']} → {largest_seq['sequence'].iloc[-1]['M #']}"
    else:
        seq_start_end = ""
    
    return patterns_found, {
        "unique_sequences": unique_sequences,
        "largest_seq_length": largest_seq_length,
        "seq_start_end": seq_start_end,
        "sequences_found": sequences_found
    }

def generate_comprehensive_mega_report(df):
    """
    Generate comprehensive single-line report with all Excel template columns
    """
    st.subheader("🎯 Single Line Mega Report - All Patterns Per Output")
    
    if not A_MODELS_AVAILABLE and not C_MODELS_AVAILABLE:
        st.error("No model detection functions available")
        return

    # Get report time
    report_time = df["Arrival"].max()  
  
    # Group by output and analyze patterns
    mega_report_data = []
    outputs = df["Output"].unique()
    
    progress_bar = st.progress(0)
    for i, output in enumerate(outputs):
        output_rows = df[df["Output"] == output]
        
        # Skip if only one row (no patterns possible)
        if len(output_rows) < 2:
            continue
            
        # Get comprehensive analysis
        patterns, sequence_metrics = analyze_output_patterns_comprehensive(output_rows, report_time)
      
        if patterns:  # Only include outputs with found patterns
            # Calculate all metrics from Excel template
            metrics = calculate_comprehensive_metrics(output_rows, report_time)
            
            # Calculate star rating (placeholder - can be customized)
            stars = len(patterns)  # Simple rating based on pattern count
            
            # Calculate booster score (placeholder - can be customized)
            booster_score = sequence_metrics["largest_seq_length"] * 10
            
            mega_report_data.append({
                "Output": f"{output:,.3f}",
                "Stars": "⭐" * min(stars, 5),  # Max 5 stars
                "Booster Score": booster_score,
                "Input @ 18:00": metrics["input_18"],
                "18:00 Diff": metrics["diff_18"],
                "Input @ Arrival": metrics["input_arrival"], 
                "Arrival Diff": metrics["diff_arrival"],
                "Input @ Report": metrics["input_report"],
                "Report Diff": metrics["diff_report"],
                "Arrival": metrics["latest_arrival"].strftime('%a %y-%m-%d %H:%M'),
                "Day": "[0]" if "[0]" in str(output_rows["Day"].iloc[-1]) else "Other",
                "Hours ago": metrics["hours_ago"],
                "Total M# Count": metrics["total_m_count"],
                "Total M# ID": metrics["m_ids"],
                "Total M#s Start/End": metrics["m_start_end"],
                "M # Arrival Order": metrics["arrival_order"],
                "M # Strength Traveler Count": metrics["strength_count"],
                "M # Strength Traveler ID": metrics["strength_ids"],
                "Tag B Traveler Count": metrics["tag_b_count"],
                "Tag B Traveler ID": metrics["tag_b_ids"],
                "Pattern Sequences": " | ".join(patterns),
                "Unique Sequences": sequence_metrics["unique_sequences"],
                "Largest sequence length": sequence_metrics["largest_seq_length"],
                "Sequence Start/End": sequence_metrics["seq_start_end"],
                "Sequence Same Origin": "Yes" if metrics["same_origin"] else "No",
                "Sequence Same Family": "Yes" if metrics["same_family"] else "No",
                "Sequence Same Feed": "Yes" if metrics["same_feed"] else "No",
                "Sequence Indigo Families": "",  # Placeholder for future implementation
                "Bucket indigo families": "",    # Placeholder for future implementation
                "Models found": len(patterns)
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
    
    st.write(f"Found patterns in {len(mega_df_sorted)} outputs with comprehensive metrics")
    st.dataframe(mega_df_sorted, use_container_width=True)
    
    # Download button
    csv_data = mega_df_sorted.to_csv(index=False)
    st.download_button(
        "📥 Download Comprehensive Single Line Mega Report (CSV)",
        data=csv_data,
        file_name="comprehensive_single_line_mega_report.csv",
        mime="text/csv"
    )
    
    return mega_df_sorted

def run_simple_single_line_analysis(df):
    """Main function to run comprehensive single line mega report analysis"""
    try:
        return generate_comprehensive_mega_report(df)
    except Exception as e:
        st.error(f"Error generating mega report: {e}")
        st.write("Traceback:", str(e))
        return None
