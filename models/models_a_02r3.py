import streamlit as st
import pandas as pd
from collections import defaultdict

# Constants: Origin Classifications   🌌🪐💫☄️🌍
EPIC_ORIGINS = {"trinidad", "tobago", "wasp-12b", "macedonia"}
ANCHOR_ORIGINS = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}

# -----------------------
# Helper functions
# -----------------------
def feed_icon(feed):
    """Return appropriate icon based on feed type"""
    return "👶" if "sm" in feed.lower() else "🧔"

def sequence_signature(seq):
    """Create unique signature for a sequence based on M # values"""
    return tuple(seq["M #"].tolist())

def classify_A_model(row_0, prior_rows):
    """
    Classify A model based on the final row and prior sequence rows
    Updated to handle M# endings of 0, 40, and 54
    """
    m_val = abs(row_0["M #"])
    if m_val not in {0, 40, 54}:
        return None, None  # Only classify if final M # is 0, 40, or 54

    m_tag = f"|{m_val}|" if m_val != 0 else "0"
    
    t0 = row_0["Arrival"]
    o0 = row_0["Origin"].lower()

    # ⌚ Time classification - Fixed to properly handle 17:00-18:00 as "open"
    if (t0.hour == 17) or (t0.hour == 18 and t0.minute == 0):
        time = "open"
    elif 18 < t0.hour <= 23 or 0 <= t0.hour <= 1:
        time = "early"  
    else:
        time = "late"

    # 🌍 Determine origin category  🌌🪐💫☄️
    is_epic = o0 in EPIC_ORIGINS
    is_anchor = o0 in ANCHOR_ORIGINS
    prior = set(prior_rows["Origin"].str.lower()) if len(prior_rows) > 0 else set()
    strong = bool(prior & EPIC_ORIGINS) or bool(prior & ANCHOR_ORIGINS)

    # 🗃️ Classifier logic with updated naming for multiple endings
    if is_epic and time == "open":
        return "A01", f"Open Epic to {m_tag}"
    if is_anchor and time == "open":
        return "A02", f"Open Anchor to {m_tag}"
    if not is_anchor and time == "open" and strong:
        return "A03", f"Open non-Anchor to {m_tag}"
    if not is_anchor and time == "early" and strong:
        return "A04", f"Early non-Anchor to {m_tag}"
    if is_anchor and time == "late":
        return "A05", f"Late Anchor to {m_tag}"
    if not is_anchor and time == "late" and strong:
        return "A06", f"Late non-Anchor to {m_tag}"
    if not is_anchor and time == "open" and not strong:
        return "A07", f"Open general to {m_tag}"
    if not is_anchor and time == "early" and not strong:
        return "A08", f"Early general to {m_tag}"
    if not is_anchor and time == "late" and not strong:
        return "A09", f"Late general to {m_tag}"

    return None, None

def find_flexible_descents(rows):
    """
    Find sequences that descend by absolute M# value and end at 0, 40, or 54
    Enhanced to find ALL valid sequences with different ending origins
    """
    raw_sequences = []
    for i in range(len(rows)):
        path = []
        seen = set()
        last_abs = float("inf")
        
        # Build the descending sequence
        for j in range(i, len(rows)):
            m = rows.loc[j, "M #"]
            abs_m = abs(m)
            
            # Check if this could be an intermediate point
            if m not in {0, 40, -40, 54, -54}:
                if abs_m in seen or abs_m >= last_abs:
                    continue
                path.append(j)
                seen.add(abs_m)
                last_abs = abs_m
            else:
                # Found a potential ending - find ALL possible endings with this M# value
                if len(path) >= 1:  # Must have at least one prior point
                    current_path = path[:]
                    
                    # Find all rows with the same ending M# value and create separate sequences
                    for k in range(j, len(rows)):
                        if rows.loc[k, "M #"] == m:
                            ending_path = current_path + [k]
                            raw_sequences.append(rows.loc[ending_path])
                break
                
    # Remove embedded shorter sequences (but keep sequences with different endings)
    filtered = []
    all_signatures = [tuple(seq["M #"].tolist()) for seq in raw_sequences]
    all_origins = [tuple(seq["Origin"].tolist()) for seq in raw_sequences]
    
    for i, (sig, origins) in enumerate(zip(all_signatures, all_origins)):
        # Check if this is a shorter subsequence of another sequence with same signature AND same origins
        longer = any(
            set(sig).issubset(set(other_sig)) and 
            len(sig) < len(other_sig) and
            origins == other_origins
            for j, (other_sig, other_origins) in enumerate(zip(all_signatures, all_origins)) 
            if i != j
        )
        if not longer:
            filtered.append(raw_sequences[i])
    return filtered

def find_pairs(rows, seen_signatures):
    """Find 2-member pairs ending at valid M# values"""
    pairs = []
    for i in range(len(rows) - 1):
        m1 = rows.iloc[i]["M #"]
        m2 = rows.iloc[i + 1]["M #"]
        if abs(m2) >= abs(m1):
            continue
        if m2 not in {0, 40, -40, 54, -54}:
            continue
        pair = rows.iloc[[i, i + 1]]
        sig = tuple(pair["M #"].tolist())
        if sig not in seen_signatures:
            pairs.append(pair)
    return pairs

def detect_A_models(df):
    """
    Enhanced detection function with proper separation of 3+ sequences and unique pairs
    """
    report_time = df["Arrival"].max()
    model_outputs = defaultdict(list)
    all_sequence_signatures = set()  # Track all sequences (3+) with origins
    all_pair_signatures = set()      # Track all pairs with origins

    for output in df["Output"].unique():
        subset = df[df["Output"] == output].sort_values("Arrival").reset_index(drop=True)
        full_matches = find_flexible_descents(subset)

        # Process sequences of length 3 or more for "3+" categories
        for seq in full_matches:
            # Must have at least 3 rows and end with valid M# for "3+" categories
            if seq.shape[0] < 3 or abs(seq.iloc[-1]["M #"]) not in {0, 40, 54}:
                continue
                
            # Track both M# signature and origins to allow different ending origins
            sig = sequence_signature(seq)
            sig_with_origins = (sig, tuple(seq["Origin"].tolist()))
            if sig_with_origins in all_sequence_signatures:
                continue
            all_sequence_signatures.add(sig_with_origins)
            
            last = seq.iloc[-1]
            prior = seq.iloc[:-1]
            
            model, label = classify_A_model(last, prior)
            if model:
                model_outputs[model].append({
                    "label": label,
                    "output": output,
                    "timestamp": last["Arrival"],
                    "sequence": seq,
                    "feeds": seq["Feed"].nunique()
                })

        # Find 2-member pairs that are NOT part of any 3+ sequence
        pairs = find_pairs(subset, {sig for sig, _ in all_sequence_signatures})
        for seq in pairs:
            if seq.shape[0] != 2:
                continue
                
            sig = sequence_signature(seq)
            sig_with_origins = (sig, tuple(seq["Origin"].tolist()))
            
            # Check if this pair is part of any 3+ sequence by checking if its signature
            # is a subset of any longer sequence
            is_part_of_longer = any(
                set(sig).issubset(set(longer_sig)) and len(sig) < len(longer_sig)
                for longer_sig, _ in all_sequence_signatures
            )
            
            if is_part_of_longer or sig_with_origins in all_pair_signatures:
                continue
                
            all_pair_signatures.add(sig_with_origins)
            
            last = seq.iloc[-1]
            if abs(last["M #"]) not in {0, 40, 54}:
                continue
                
            prior = seq.iloc[:-1]
            model, label = classify_A_model(last, prior)
            if model:
                pr_model = model + "pr"
                model_outputs[pr_model].append({
                    "label": f"Pair to {label}",
                    "output": output,
                    "timestamp": last["Arrival"],
                    "sequence": seq,
                    "feeds": seq["Feed"].nunique()
                })

    return model_outputs, report_time

def show_a_model_results(model_outputs, report_time):
    """Display the A model results in Streamlit interface"""
    base_labels = {
        "A01": "Open Epic", "A02": "Open Anchor", "A03": "Open non-Anchor",
        "A04": "Early non-Anchor", "A05": "Late Anchor", "A06": "Late non-Anchor",
        "A07": "Open general", "A08": "Early general", "A09": "Late general"
    }

    st.subheader("🔍 A Model Results")
    
    for code, base_label in base_labels.items():
        for suffix, prefix in [("", "3+ to"), ("pr", "Pair to")]:
            key = code + suffix
            results = model_outputs.get(key, [])
            output_count = len(set(r["output"] for r in results))
            header = f"{key}. {prefix} {base_label} – {output_count} output{'s' if output_count != 1 else ''}"

            with st.expander(header):
                if results:
                    # Group by day type
                    today_results = [r for r in results if "[0]" in str(r["sequence"].iloc[-1]["Day"]).lower()]
                    other_results = [r for r in results if "[0]" not in str(r["sequence"].iloc[-1]["Day"]).lower()]

                    def render_group(name, group):
                        st.markdown(f"#### {name}")
                        output_groups = defaultdict(list)
                        for r in group:
                            output_groups[r["output"]].append(r)

                        for out_val, items in output_groups.items():
                            latest = max(items, key=lambda r: r["timestamp"])
                            hrs = int((report_time - latest["timestamp"]).total_seconds() / 3600)
                            ts = latest["timestamp"].strftime('%-m/%-d/%y %H:%M')
                            subhead = f"🔹 Output {out_val:,.3f} – {len(items)} sequence{'s' if len(items) != 1 else ''} {hrs} hours ago at {ts}"

                            with st.expander(subhead):
                                for res in items:
                                    seq = res["sequence"]
                                    m_path = " → ".join([f"|{row['M #']}|" for _, row in seq.iterrows()])
                                    origins = " → ".join([f"{row['Origin']}" for _, row in seq.iterrows()])
                                    icons = "".join([feed_icon(row["Feed"]) for _, row in seq.iterrows()])
                                    st.markdown(f"**{m_path}** ({origins}) Cross [{icons}]")
                                    st.table(seq.reset_index(drop=True))

                    if today_results:
                        render_group("📅 Today", today_results)
                    if other_results:
                        render_group("📦 Other Days", other_results)
                        
                    if not today_results and not other_results:
                        st.markdown("No matching outputs.")
                else:
                    st.markdown("No matching outputs.")

def run_a_model_detection(df):
    """Main function to run A model detection and display results"""
    model_outputs, report_time = detect_A_models(df)
    show_a_model_results(model_outputs, report_time)
    return model_outputs
