# Final Enhanced Model A with Expandable Groups and Cluster Reporting
# Implements all requested modifications: 3+ minimum, unique pairs, Model C-style UI

import streamlit as st
import pandas as pd
from collections import defaultdict

# Constants: Origin Classifications   🌌🪐💫☄️🌍
EPIC_ORIGINS = {"trinidad", "tobago", "wasp-12b", "macedonia"}
ANCHOR_ORIGINS = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}

# A Model Groups Structure (like C_GROUPS in Model C)
A_GROUPS = {
    "A01: Open Epic": {
        "A01": "3+ to Open Epic",
        "A01pr": "Pair to Open Epic"
    },
    "A02: Open Anchor": {
        "A02": "3+ to Open Anchor", 
        "A02pr": "Pair to Open Anchor"
    },
    "A03: Open non-Anchor": {
        "A03": "3+ to Open non-Anchor",
        "A03pr": "Pair to Open non-Anchor"
    },
    "A04: Early non-Anchor": {
        "A04": "3+ to Early non-Anchor",
        "A04pr": "Pair to Early non-Anchor" 
    },
    "A05: Late Anchor": {
        "A05": "3+ to Late Anchor",
        "A05pr": "Pair to Late Anchor"
    },
    "A06: Late non-Anchor": {
        "A06": "3+ to Late non-Anchor",
        "A06pr": "Pair to Late non-Anchor"
    },
    "A07: Open general": {
        "A07": "3+ to Open general",
        "A07pr": "Pair to Open general"
    },
    "A08: Early general": {
        "A08": "3+ to Early general", 
        "A08pr": "Pair to Early general"
    },
    "A09: Late general": {
        "A09": "3+ to Late general",
        "A09pr": "Pair to Late general"
    }
}

# Helper functions
def feed_icon(feed):
    """Return appropriate icon based on feed type"""
    return "👶" if "sm" in feed.lower() else "🧔"

def sequence_signature(seq):
    """Create unique signature for a sequence based on M # values"""
    return tuple(seq["M #"].tolist())

def classify_A_model(row_0, prior_rows):
    """
    Classify A model based on the final row and prior sequence rows
    Updated to handle M# endings of 0, 40, and 54 with dynamic labeling
    """
    m_val = abs(row_0["M #"])
    if m_val not in {0, 40, 54}:
        return None, None  # Only classify if final M # is 0, 40, or 54

    m_tag = f"|{m_val}|" if m_val != 0 else "0"
    
    t0 = row_0["Arrival"]
    o0 = row_0["Origin"].lower()

    # Time classification - Fixed to properly handle 17:00-18:00 as "open"
    if (t0.hour == 17) or (t0.hour == 18 and t0.minute == 0):
        time = "open"
    elif 18 < t0.hour <= 23 or 0 <= t0.hour <= 1:
        time = "early"  
    else:
        time = "late"

    # Determine origin category
    is_epic = o0 in EPIC_ORIGINS
    is_anchor = o0 in ANCHOR_ORIGINS
    prior = set(prior_rows["Origin"].str.lower()) if len(prior_rows) > 0 else set()
    strong = bool(prior & EPIC_ORIGINS) or bool(prior & ANCHOR_ORIGINS)

    # Classifier logic with updated naming for multiple endings
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
            
            # Check if this pair is part of any 3+ sequence
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

def show_a_cluster_table(model_outputs, input_reference=None):
    """Display A Model cluster report with highlight and filtering"""
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
        st.info("No A Models matched for cluster display.")
        return

    st.subheader("📊 A Model Cluster Report (Descending Output)")
    
    # Build rows
    rows = []
    for out_val, c in cluster.items():
        tags_str = ", ".join(sorted(c["tags"]))
        rows.append({
            "Output": round(out_val, 3),
            "Input": r.get("Input", out_val),  # fallback if Input is missing
            "Sequences": c["count"],
            "Model Count": len(c["tags"]),
            "Tags Found": tags_str,
            "Latest Arrival": c["latest"].strftime('%-m/%-d/%y %H:%M')
        })
    df_cluster = pd.DataFrame(rows).sort_values("Output", ascending=False)

    # --- Optional filter ---
    show_only_a02_a03 = st.checkbox("🧽 Show only rows with A02 or A03 tags")
    if show_only_a02_a03:
        df_cluster = df_cluster[df_cluster["Tags Found"].str.contains("A02|A03")]

    # --- Highlight logic ---
    def highlight_row(row):
        output = float(row["Output"])
        input_val = float(row["Input"])
        tag_val = row["Tags Found"]
        styles = [""] * len(row)
    
        delta = abs(output - input_val)
        if delta < 1e-3:
            styles = ["background-color: lightgray"] * len(row)
        elif output > input_val:
            styles = ["background-color: #fdd"] * len(row)  # light red
        elif output < input_val:
            styles = ["background-color: #ddf"] * len(row)  # light blue
    
        if "A02" in tag_val or "A03" in tag_val:
            idx = row.index.get_loc("Tags Found")
            styles[idx] = "font-weight: bold; background-color: yellow"
    
        return styles

    styled = df_cluster.style.apply(highlight_row, axis=1)
    st.dataframe(styled, use_container_width=True)


def show_a_model_results(model_outputs, report_time):
    """Display A model results with expandable groups like Model C"""
    st.subheader("🔍 A Model Results")
    
    # Group display similar to Model C structure
    for group, tag_dict in A_GROUPS.items():
        group_total = sum(len(model_outputs.get(tag, [])) for tag in tag_dict.keys())
        with st.expander(f"{group} – {group_total} match{'es' if group_total != 1 else ''}"):
            for tag, base_label in tag_dict.items():
                results = model_outputs.get(tag, [])
                output_count = len(set(r["output"] for r in results)) if results else 0
                
                # Dynamic labeling based on actual results
                if results:
                    sample_label = results[0]["label"]
                    header = f"{tag}. {sample_label} – {output_count} output{'s' if output_count != 1 else ''}"
                else:
                    header = f"{tag}. {base_label} – 0 outputs"

                with st.expander(header):
                    if not results:
                        st.info("No results found for this model.")
                        continue

                    # Group by day type
                    today_results = [r for r in results if "[0]" in str(r["sequence"].iloc[-1]["Day"]).lower()]
                    other_results = [r for r in results if "[0]" not in str(r["sequence"].iloc[-1]["Day"]).lower()]

                    def render_group(name, group):
                        if not group:
                            return
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

                    render_group("📅 Today", today_results)
                    render_group("📦 Other Days", other_results)

def run_a_model_detection(df):
    """Main function to run A model detection with cluster reporting"""
    report_time = df["Arrival"].max()
    model_outputs, _ = detect_A_models(df)
    st.write("🧬 Detected A Model outputs:", sum(len(v) for v in model_outputs.values()))
    show_a_cluster_table(model_outputs)  # Add cluster report like Model C
    show_a_model_results(model_outputs, report_time)
    return model_outputs
