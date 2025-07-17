import streamlit as st
import pandas as pd
from collections import defaultdict
# CavAir. Broad descending-sequence scanning; Inclusive classifier logic; Flexible polarity and feed checks; Visual expanders grouped by label and day
# --- Constants ---
ANCHOR_ORIGINS = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}
EPIC_ORIGINS = {"trinidad", "tobago", "wasp-12b", "macedonia"}

# --- Helpers ---
def feed_icon(feed):
    return "👶" if "sm" in feed.lower() else "🧔"

def sequence_signature(seq):
    return tuple(seq["M #"].tolist())

# --- Broad Sequence Finder ---
def find_descending_sequences(df):
    sequences = []
    seen_signatures = set()

    for output in df["Output"].unique():
        rows = df[df["Output"] == output].sort_values("Arrival").reset_index(drop=True)
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
                    sig = tuple([p["M #"] for p in path])
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        seq_df = pd.DataFrame(path).reset_index(drop=True)
                        sequences.append((output, seq_df))
    return sequences

# --- Classifier Logic ---
def classify_b_sequence(seq):
    # --- Constants ---
    epic = {"trinidad", "tobago", "wasp-12b", "macedonia"}
    anchor = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}
    origins = set(seq["Origin"].str.lower())
    has_origin = bool(origins & (epic | anchor))

    # --- Required Checks ---
    if seq.shape[0] < 3:
        return None, None

    abs_m = [abs(m) for m in seq["M #"]]
    if abs_m != sorted(abs_m, reverse=True) or len(set(abs_m)) != len(abs_m):
        return None, None  # Must descend strictly by abs and be unique

    final_m = seq.iloc[-1]["M #"]
    final_day = str(seq.iloc[-1]["Day"]).lower()
    ends_today = "[0]" in final_day
    day_tags = seq["Day"].astype(str).str.lower()
    today_count = day_tags.str.contains(r"\[0\]").sum()

    # --- Feed + Polarity Detection ---
    feeds = seq["Feed"].nunique()
    sign_set = set(1 if m > 0 else -1 for m in seq["M #"] if m != 0)
    polarity_same = len(sign_set) == 1
    feed_same = feeds == 1

    # --- Day Tag Gate ---
    if ends_today and today_count < 2:
        return None, None  # [0] variants require ≥2

    # --- Tag Builder ---
    def build_tag(series_num, polarity_code, day_code):
        label = f"B{series_num}{polarity_code}[{day_code}]"
        return label

    # --- Classify B01 ---
    if has_origin and abs(final_m) == 40:
        if polarity_same and feed_same:
            return build_tag("01", "a", "0" if ends_today else "≠0"), "Same Polarity, *Origin to |40| " + ("today" if ends_today else "≠[0]")
        else:
            return build_tag("01", "b", "0" if ends_today else "≠0"), "Same or mixed polarities, *Origin to |40| " + ("today" if ends_today else "≠[0]")

    # --- Classify B02 ---
    if not has_origin and abs(final_m) == 40:
        if polarity_same and feed_same:
            return build_tag("02", "a", "0" if ends_today else "≠0"), "Same Polarity, no *Origin to |40| " + ("today" if ends_today else "≠[0]")
        else:
            return build_tag("02", "b", "0" if ends_today else "≠0"), "Same or mixed polarities, no *Origin to |40| " + ("today" if ends_today else "≠[0]")

    # --- Classify B03 ---
    if has_origin and abs(final_m) != 40:
        if polarity_same and feed_same:
            return build_tag("03", "a", "0" if ends_today else "≠0"), "Same Polarity, *Origin to ≠|40| " + ("today" if ends_today else "≠[0]")
        else:
            return build_tag("03", "b", "0" if ends_today else "≠0"), "Same or mixed polarities, *Origin to ≠|40| " + ("today" if ends_today else "≠[0]")

    return None, None  # No match found


# --- Detection Wrapper ---
def detect_B_models(df, report_time):
    model_outputs = defaultdict(list)
    sequences = find_descending_sequences(df)

    for output, seq in sequences:
        label_code, label_text = classify_b_sequence(seq)
        if label_code:
            model_outputs[label_code].append({
                "label": label_text,
                "output": output,
                "timestamp": seq.iloc[-1]["Arrival"],
                "sequence": seq,
                "feeds": seq["Feed"].nunique()
            })

    return model_outputs

# --- Streamlit Display ---
def show_b_model_results(model_outputs, report_time):
    st.subheader("🔍 B Model Results")

    # Sort tags for predictable ordering (optional)
    sorted_tags = sorted(model_outputs.keys(), key=lambda x: (
        int(x[1:3]), x[3], x[4:]  # Series, polarity, day variant
    ))

    for code in sorted_tags:
        results = model_outputs.get(code, [])
        if not results:
            continue

        label_text = results[0]["label"] if results else "Unnamed pattern"
        output_count = len(set(r["output"] for r in results))
        header = f"{code}. {label_text} – {output_count} output{'s' if output_count != 1 else ''}"

        with st.expander(header):
            # Split into today vs non-today results
            today_results = [r for r in results if "[0]" in str(r["sequence"].iloc[-1]["Day"]).lower()]
            other_results = [r for r in results if "[0]" not in str(r["sequence"].iloc[-1]["Day"]).lower()]

            def render_group(name, group):
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
                            icons = "".join(["👶" if "sm" in row["Feed"].lower() else "🧔" for _, row in seq.iterrows()])
                            st.markdown(f"{m_path} Cross [{icons}]")
                            st.table(seq.reset_index(drop=True))

            if today_results:
                render_group("📅 Today", today_results)
            if other_results:
                render_group("📦 Other Days", other_results)
                
# --- App Entry Point ---
def run_b_model_detection(df):
    report_time = df["Arrival"].max()
    model_outputs = detect_B_models(df, report_time)

    st.write("🔬 Detected B Model outputs:", sum(len(v) for v in model_outputs.values()))
    show_b_model_results(model_outputs, report_time)
