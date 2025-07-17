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
    epic = EPIC_ORIGINS
    anchor = ANCHOR_ORIGINS
    origins = set(seq["Origin"].str.lower())
    has_anchor_or_epic = bool(origins & (epic | anchor))

    day_tags = seq["Day"].astype(str).str.lower()
    today_tags = [tag for tag in day_tags if "[0]" in tag]
    is_today = "[0]" in day_tags.iloc[-1]
    today_count = len(today_tags)

    if seq.shape[0] < 3 or not has_anchor_or_epic or not is_today or today_count < 2:
        return None, None

    final_m = seq.iloc[-1]["M #"]
    is_40 = final_m == 40

    sign_set = set(1 if m > 0 else -1 for m in seq["M #"] if m != 0)
    polarity = "a" if len(sign_set) == 1 else "b"

    if is_40:
        return f"B01{polarity}[0]", f"Polarity and Feed may vary to |40| Today w/ Anchor/EPIC"
    else:
        return f"B03{polarity}[0]", f"Polarity and Feed may vary to ≠|40| Today w/ Anchor/EPIC"

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
    base_names = {
        "B01a[0]": "Same Polarity Descenders, *Origin to |40| today",
        "B01b[0]": "Mixed Polarity Descenders, *Origin to |40| today",
        "B03a[0]": "Same Polarity Descenders to ≠|40| today",
        "B03b[0]": "Mixed Polarity Descenders to ≠|40| today"
    }

    st.subheader("🔍 B Model Results")
    for code, label in base_names.items():
        results = model_outputs.get(code, [])
        output_count = len(set(r["output"] for r in results))
        header = f"{code}. {label} – {output_count} output{'s' if output_count != 1 else ''}"

        with st.expander(header):
            if not results:
                st.markdown("No matching outputs.")
                continue

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
                            icons = "".join([feed_icon(row["Feed"]) for _, row in seq.iterrows()])
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
