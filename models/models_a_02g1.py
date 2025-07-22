import streamlit as st
import pandas as pd
from collections import defaultdict

# -----------------------
# Helper functions
# -----------------------
def feed_icon(feed):
    return "👶" if "sm" in feed.lower() else "🧔"

def sequence_signature(seq):
    return tuple(seq["M #"].tolist())

def classify_A_model(row_0, prior_rows):
    epic = {"trinidad", "tobago", "wasp-12b", "macedonia"}
    anchor = {"Spain", "saturn", "jupiter", "kepler-62", "kepler-44"}

    m_val = abs(row_0["M #"])
    if m_val not in {0, 40, 54}:
        return None, None  # Only classify if final M # is 0, 40, or 54

    m_tag = f"|{m_val}|" if m_val != 0 else "0"
    
    t0 = row_0["Arrival"]
    o0 = row_0["Origin"].lower()

    # ⌚ Time classification 🕓
    if t0.hour == 17 or (t0.hour == 18 and t0.minute == 0):
        time = "open"
    elif 17 < t0.hour < 24 or t0.hour == 0 or (t0.hour == 1 and t0.minute < 59):
        time = "early"
    else:
        time = "late"

    # Determine time category  (old, being replaced)
    # time = "open" if t0.hour == 18 and t0.minute == 0 else \
    #        "early" if (t0.hour < 2 or (t0.hour == 1 and t0.minute < 59)) else "late"

    # 🌍 Determine origin category  🌌🪐💫☄️
    is_epic = o0 in epic
    is_anchor = o0 in anchor
    prior = set(prior_rows["Origin"].str.lower())
    strong = bool(prior & epic or prior & anchor)

    # 🗃️ Classifier logic with new naming  📚
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
    raw_sequences = []
    for i in range(len(rows)):
        path = []
        seen = set()
        last_abs = float("inf")
        for j in range(i, len(rows)):
            m = rows.loc[j, "M #"]
            abs_m = abs(m)
            if m in {0, 40, -40, 54, -54}:
                if len(path) >= 2:
                    path.append(j)
                    raw_sequences.append(rows.loc[path])
                break
            if abs_m in seen or abs_m >= last_abs:
                continue
            path.append(j)
            seen.add(abs_m)
            last_abs = abs_m
    # Remove embedded shorter sequences
    filtered = []
    all_signatures = [tuple(seq["M #"].tolist()) for seq in raw_sequences]
    for i, sig in enumerate(all_signatures):
        longer = any(set(sig).issubset(set(other)) and len(sig) < len(other) 
                     for j, other in enumerate(all_signatures) if i != j)
        if not longer:
            filtered.append(raw_sequences[i])
    return filtered

def find_pairs(rows, seen_signatures):
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
    report_time = df["Arrival"].max()
    model_outputs = defaultdict(list)
    all_signatures = set()

    for output in df["Output"].unique():
        subset = df[df["Output"] == output].sort_values("Arrival").reset_index(drop=True)
        full_matches = find_flexible_descents(subset)

        for seq in full_matches:
            if seq.shape[0] < 3 or abs(seq.iloc[-1]["M #"]) not in {0, 40, 54}:
                continue
            sig = sequence_signature(seq)
            if sig in all_signatures:
                continue
            all_signatures.add(sig)
            prior = seq.iloc[:-1]
            valid_endings = seq[seq["M #"].abs().isin([0, 40, 54])]
            valid_endings = valid_endings[valid_endings["Origin"].str.lower().isin(anchor | epic)]
            valid_endings = valid_endings[valid_endings["Arrival"].dt.hour.isin([17, 18])]  # Open hours
            
            if valid_endings.empty:
                continue  # Skip if no valid final traveler
            
            last = valid_endings.iloc[-1]
            prior = seq[seq.index < last.name]
            model, label = classify_A_model(last, prior)
            if model:
                model_outputs[model].append({
                    "label": label,
                    "output": output,
                    "timestamp": last["Arrival"],
                    "sequence": seq,
                    "feeds": seq["Feed"].nunique()
                })

        # Now find 2-member pairs not already used
        pairs = find_pairs(subset, all_signatures)
        for seq in pairs:
            sig = sequence_signature(seq)
            if sig in all_signatures:
                continue
            all_signatures.add(sig)
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
    base_labels = {
        "A01": "Open Epic 0", "A02": "Open Anchor 0", "A03": "Open non-Anchor 0",
        "A04": "Early non-Anchor 0", "A05": "Late Anchor 0", "A06": "Late non-Anchor 0",
        "A07": "Open general 0", "A08": "Early general 0", "A09": "Late general 0"
    }

    st.subheader("🔍 A Model Results")
    for code, label in base_labels.items():
        for suffix, title in [("", f"2+ to {label}"), ("pr", f"Pair to {label}")]:
            key = code + suffix
            results = model_outputs.get(key, [])
            output_count = len(set(r["output"] for r in results))
            header = f"{key}. {title} – {output_count} output{'s' if output_count != 1 else ''}"

            with st.expander(header):
                if results:
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
                            subhead = f"🔹 Output {out_val:,.3f} – {len(items)} descending {hrs} hours ago at {ts}"

                            with st.expander(subhead):
                                for res in items:
                                    seq = res["sequence"]
                                    m_path = " → ".join([f"|{row['M #']}|" for _, row in seq.iterrows()])
                                    icons = "".join([feed_icon(row["Feed"]) for _, row in seq.iterrows()])
                                    st.markdown(f"{m_path} Cross [{icons}]")
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
    model_outputs, report_time = detect_A_models(df)
    show_a_model_results(model_outputs, report_time)
    return model_outputs
