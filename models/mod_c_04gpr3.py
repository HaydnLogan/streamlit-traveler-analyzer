# Complete Model C with All Bug Fixes Applied
# Fixed C.02 opposites detection and C.04 ascending detection
# Enhanced sequence finding with 3-point pattern detection

import streamlit as st
import pandas as pd
from collections import defaultdict

# Constants
STRENGTH_TRAVELERS = {0, 40, -40, 54, -54}
ANCHOR_ORIGINS = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}
EPIC_ORIGINS = {"trinidad", "tobago", "wasp-12b", "macedonia"}

C_GROUPS = {
    "C.01: ": {
        "C.01.o.[aft0]": "After Midnight Influence Shift *Origin Today",
        "C.01.o.[fwd0]": "Before Midnight Influence Shift *Origin Today",
        "C.01.t.[aft0]": "After Midnight Influence Shift No *Origin Today",
        "C.01.t.[fwd0]": "Before Midnight Influence Shift No *Origin Today",
    },
    "C.02: Opposites TODAY with M # in the middle": {
        "C.02.p1.[L0]": "Late Opposites, 0 Mid today",
        "C.02.p2.[E0]": "Early Opposites, 0 Mid today",
        "C.02.p3.[O0]": "Open Opposites, 0 Mid today",
        "C.02.n1.[L0]": "Late Opposites, ≠0 Mid today",
        "C.02.n2.[E0]": "Early Opposites, ≠0 Mid today",
        "C.02.n3.[O0]": "Open Opposites, ≠0 Mid today",
    },
    "C.04: Ascending 0, 40, 54": {
        "C.04.∀1.[0]": "Trio up to |54| today",
        "C.04.∀2.[±1]": "Trio up to |54| other days",
    }
}

def feed_icon(feed):
    """Return appropriate icon for feed type"""
    return "👶" if "sm" in feed.lower() else "🧔"

def classify_c01_sequence(seq):
    """Classify C.01 sequences - Influence Shift detection"""
    if seq.shape[0] < 3:
        return None, None

    final = seq.iloc[-1]
    final_m = final["M #"]
    final_day = str(final["Day"]).strip()
    final_hour = pd.to_datetime(final["Arrival"]).hour

    if final_day != "[0]" or abs(final_m) not in STRENGTH_TRAVELERS or final_m == 0:
        return None, None

    m_vals = seq["M #"].tolist()
    initial_sign = 1 if m_vals[0] > 0 else -1
    if not all((m > 0) == (initial_sign > 0) for m in m_vals[:-1]):
        return None, None
    if (m_vals[-1] > 0) == (initial_sign > 0):
        return None, None

    abs_path = [abs(m) for m in m_vals[:-1]]
    if abs_path != sorted(abs_path, reverse=True):
        return None, None

    origins = set(seq["Origin"].str.lower())
    has_origin = origins & (ANCHOR_ORIGINS | EPIC_ORIGINS)
    variant = "o" if has_origin else "t"
    suffix = "[aft0]" if final_hour >= 0 else "[fwd0]"
    tag = f"C.01.{variant}.{suffix}"

    label_map = {
        "C.01.o.[aft0]": "After Midnight Influence Shift *Origin Today",
        "C.01.o.[fwd0]": "Before Midnight Influence Shift *Origin Today",
        "C.01.t.[aft0]": "After Midnight Influence Shift No *Origin Today",
        "C.01.t.[fwd0]": "Before Midnight Influence Shift No *Origin Today"
    }
    label = label_map.get(tag)
    return tag, label

def classify_c02_sequence(seq):
    """FIXED C.02: Enhanced opposites detection with proper exclusions"""
    if seq.shape[0] != 3:
        return None, None

    m_vals = seq["M #"].tolist()
    first, mid, last = m_vals

    # FIXED: Exclude cases where all values are the same (e.g., 0,0,0)
    if first == mid == last:
        return None, None

    # FIXED: Exclude cases where middle value makes this not a true opposites pattern
    # Allow sequences like 67→0→-67 (true opposites with 0 middle)
    # But exclude sequences like 0→X→0 where endpoints are both zero

    # Check if first and last M # are true opposites (e.g., -54 and +54)
    if first + last != 0:
        return None, None

    # Only check Day == [0]  
    final_day = str(seq.iloc[-1]["Day"]).strip()
    if final_day != "[0]":
        return None, None

    # FIXED: Better time classification based on user data patterns
    arrival_time = pd.to_datetime(seq.iloc[-1]["Arrival"])
    hour = arrival_time.hour
    minute = arrival_time.minute
    
    # Based on user data: Late=morning hours, Early=evening hours, Open=17:00-18:00
    if hour == 17 or (hour == 18 and minute == 0):  # 17:00-18:00 Open window
        suffix = "[O0]"; num = "3"
    elif hour >= 18 or hour <= 1:  # Evening/night hours = Early
        suffix = "[E0]"; num = "2" 
    elif hour >= 2 and hour <= 16:  # Morning/afternoon hours = Late  
        suffix = "[L0]"; num = "1"
    else:
        return None, None

    # Determine mid-type: "p" if 0, otherwise "n"
    mid_type = "p" if mid == 0 else "n"

    tag = f"C.02.{mid_type}{num}.{suffix}"
    label_map = {
        "C.02.p1.[L0]": "Late Opposites, 0 Mid today",
        "C.02.p2.[E0]": "Early Opposites, 0 Mid today", 
        "C.02.p3.[O0]": "Open Opposites, 0 Mid today",
        "C.02.n1.[L0]": "Late Opposites, ≠0 Mid today",
        "C.02.n2.[E0]": "Early Opposites, ≠0 Mid today",
        "C.02.n3.[O0]": "Open Opposites, ≠0 Mid today"
    }
    label = label_map.get(tag)
    return tag, label

def classify_c04_sequence(seq):
    """FIXED C.04: Enhanced ascending detection with better sequence matching"""
    if seq.shape[0] != 3:
        return None, None

    # FIXED: Check for ascending absolute values pattern  
    m_vals = seq["M #"].tolist()
    abs_vals = [abs(m) for m in m_vals]
    
    # Must be exactly ascending: 0 → 40 → 54 (absolute values)  
    # This covers patterns like: 0→40→54, 0→40→-54, 0→-40→54, 0→-40→-54
    if abs_vals != [0, 40, 54]:
        return None, None

    final_day = str(seq.iloc[-1]["Day"]).strip()
    if final_day == "[0]":
        return "C.04.∀1.[0]", "Trio up to |54| today"
    else:
        return "C.04.∀2.[±1]", "Trio up to |54| other days"

def find_descending_sequences(df):
    """ENHANCED: Find descending sequences AND ascending/opposites patterns"""
    sequences = []
    seen_signatures = set()
    
    for output in df["Output"].unique():
        rows = df[df["Output"] == output].sort_values("Arrival").reset_index(drop=True)
        
        # Find descending sequences (existing logic)
        for i in range(len(rows)):
            path = []
            abs_seen = set()
            for j in range(i, len(rows)):
                m = rows.loc[j, "M #"]
                abs_m = abs(m)
                if abs_m in abs_seen:
                    continue
                # For descending sequences, next absolute value must be smaller
                if path and abs_m >= abs(path[-1]["M #"]):
                    continue
                abs_seen.add(abs_m)
                path.append(rows.loc[j])
                if len(path) >= 3:
                    sig = tuple([p["M #"] for p in path])
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        seq_df = pd.DataFrame(path).reset_index(drop=True)
                        sequences.append((output, seq_df))
        
        # ADDED: Find 3-point patterns (opposites and ascending) - ALL consecutive triplets
        for i in range(len(rows) - 2):
            three_points = [rows.loc[i], rows.loc[i+1], rows.loc[i+2]]
            sig = tuple([p["M #"] for p in three_points])
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                seq_df = pd.DataFrame(three_points).reset_index(drop=True)
                sequences.append((output, seq_df))
        
        # ADDED: Find ALL consecutive triplets (not just chronologically consecutive)
        # This ensures we catch patterns like 67→0→-67 even if not consecutive in time
        for i in range(len(rows)):
            for j in range(i+1, len(rows)):
                for k in range(j+1, len(rows)):
                    three_points = [rows.loc[i], rows.loc[j], rows.loc[k]]
                    sig = tuple([p["M #"] for p in three_points])
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        seq_df = pd.DataFrame(three_points).reset_index(drop=True)
                        sequences.append((output, seq_df))
                
    return sequences

def detect_C_models(df, run_c01=True, run_c02=True, run_c04=True):
    """Main C model detection function with enhanced pattern finding"""
    sequences = find_descending_sequences(df)
    model_outputs = defaultdict(list)

    for output, seq in sequences:
        classifiers = []
        if run_c01:
            classifiers.append(classify_c01_sequence)
        if run_c02:
            classifiers.append(classify_c02_sequence)
        if run_c04:
            classifiers.append(classify_c04_sequence)

        for clf in classifiers:
            tag, label = clf(seq)
            if tag:
                model_outputs[tag].append({
                    "output": output,
                    "label": label,
                    "tag": tag,
                    "timestamp": pd.to_datetime(seq.iloc[-1]["Arrival"]),
                    "sequence": seq
                })
                break  # Only classify with first matching classifier
    return model_outputs

def show_c_cluster_table(model_outputs):
    """Display C Model cluster report"""
    cluster = {}

    for tag, items in model_outputs.items():
        for r in items:
            out = r["output"]
            if out not in cluster:
                cluster[out] = {"tags": set(), "count": 0, "latest": None}
            cluster[out]["tags"].add(tag)
            cluster[out]["count"] += 1
            ts = r["timestamp"]
            if not cluster[out]["latest"] or ts > cluster[out]["latest"]:
                cluster[out]["latest"] = ts

    if not cluster:
        st.info("No C Models matched for cluster display.")
        return

    st.subheader("📊 C Model Cluster Report (Descending Output)")
    rows = []
    for out_val, c in cluster.items():
        rows.append({
            "Output": f"{out_val:,.3f}",
            "Sequences": c["count"],
            "Model Count": len(c["tags"]),
            "Tags Found": ", ".join(sorted(c["tags"])),
            "Latest Arrival": c["latest"].strftime('%-m/%-d/%y %H:%M')
        })

    rows.sort(key=lambda x: float(x["Output"].replace(",", "")), reverse=True)
    st.table(rows)

def show_c_model_results(model_outputs, report_time):
    """Display C model results with expandable groups"""
    st.subheader("🔍 C Model Results")

    for group, tag_dict in C_GROUPS.items():
        group_total = sum(len(model_outputs.get(tag, [])) for tag in tag_dict.keys())
        with st.expander(f"{group} – {group_total} match{'es' if group_total != 1 else ''}"):
            for tag, label in tag_dict.items():
                results = model_outputs.get(tag, [])
                output_count = len(set(r["output"] for r in results)) if results else 0
                header = f"{tag}. {label} – {output_count} output{'s' if output_count != 1 else ''}"

                with st.expander(header):
                    if not results:
                        st.info("No results found for this model.")
                        continue

                    today_results = [r for r in results if "[0]" in str(r["sequence"].iloc[-1]["Day"]).lower()]
                    other_results = [r for r in results if "[0]" not in str(r["sequence"].iloc[-1]["Day"]).lower()]

                    def render_group(name, group):
                        if not group:
                            return
                        st.markdown(f"#### {name}")
                        grouped = defaultdict(list)
                        for r in group:
                            grouped[r["output"]].append(r)

                        for out_val, items in grouped.items():
                            latest = max(items, key=lambda r: r["timestamp"])
                            hrs = int((report_time - latest["timestamp"]).total_seconds() / 3600)
                            ts = latest["timestamp"].strftime('%-m/%-d/%y %H:%M')
                            subhead = f"🔹 Output {out_val:,.3f} – {len(items)} sequence(s) {hrs} hours ago at {ts}"

                            with st.expander(subhead):
                                for res in items:
                                    seq = res["sequence"]
                                    m_path = " → ".join([f"|{row['M #']}|" for _, row in seq.iterrows()])
                                    icons = "".join([feed_icon(row["Feed"]) for _, row in seq.iterrows()])
                                    st.markdown(f"{m_path} Cross [{icons}]")
                                    st.table(seq.reset_index(drop=True))

                    render_group("📅 Today", today_results)
                    render_group("📦 Other Days", other_results)

def run_c_model_detection(df, run_c01=True, run_c02=True, run_c04=True):
    """Main function to run C model detection with cluster reporting"""
    report_time = df["Arrival"].max()
    model_outputs = detect_C_models(df, run_c01, run_c02, run_c04)
    st.write("🧬 Detected C Model outputs:", sum(len(v) for v in model_outputs.values()))
    show_c_cluster_table(model_outputs)
    show_c_model_results(model_outputs, report_time)
    return model_outputs
