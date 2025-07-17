import streamlit as st
import pandas as pd
from collections import defaultdict
# GroundTech. Detect same and mixed polarity sequences. Distinguish feed types (a = same feed, b = mixed feed). Require strictly descending absolute M #s (no duplicates).
# --- Constants ---
ANCHOR_ORIGINS = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}
EPIC_ORIGINS = {"trinidad", "tobago", "wasp-12b", "macedonia"}

# --- Helpers ---
def feed_icon(feed):
    return "👶" if "sm" in feed.lower() else "🧔"

def sequence_signature(seq):
    return tuple(seq["M #"].tolist())

def is_strict_descending_by_abs(seq):
    abs_values = [abs(m) for m in seq["M #"]]
    return all(abs_values[i] > abs_values[i+1] for i in range(len(abs_values)-1))

def has_anchor_or_epic(seq):
    origins = set(seq["Origin"].str.lower())
    return bool(origins & (ANCHOR_ORIGINS | EPIC_ORIGINS))

def count_today(seq):
    return sum("[0]" in str(d) for d in seq["Day"])

def is_same_polarity(seq):
    signs = set(1 if m > 0 else -1 for m in seq["M #"] if m != 0)
    return len(signs) == 1

def detect_B_models(df, report_time):
    model_outputs = defaultdict(list)
    seen_signatures = set()

    for output in df["Output"].unique():
        group = df[df["Output"] == output].sort_values("Arrival").reset_index(drop=True)
        n = len(group)
        for i in range(n):
            path = []
            for j in range(i, n):
                m = group.loc[j, "M #"]
                path.append(j)
                if len(path) < 3:
                    continue
                seq = group.loc[path].reset_index(drop=True)
                sig = sequence_signature(seq)
                if sig in seen_signatures:
                    continue
                if not is_strict_descending_by_abs(seq):
                    continue
                if count_today(seq) < 2 and "[0]" in str(seq.iloc[-1]["Day"]):
                    continue
                seen_signatures.add(sig)

                label = classify_b_model(seq)
                if label:
                    model_outputs[label].append({
                        "output": output,
                        "timestamp": seq.iloc[-1]["Arrival"],
                        "sequence": seq,
                        "feeds": seq["Feed"].nunique()
                    })
    return model_outputs

def classify_b_model(seq):
    has_special = has_anchor_or_epic(seq)
    is_today = "[0]" in str(seq.iloc[-1]["Day"])
    ends_at_40 = abs(seq.iloc[-1]["M #"]) == 40
    same_polarity = is_same_polarity(seq)

    feed_type = "a" if seq["Feed"].nunique() == 1 else "b"
    pol_type = "a" if same_polarity else "b"
    day_tag = "[0]" if is_today else "[≠0]"

    if has_special and ends_at_40:
        return f"B01{feed_type if same_polarity else pol_type}{day_tag}"
    if not has_special and ends_at_40:
        return f"B02{feed_type if same_polarity else pol_type}{day_tag}"
    if not ends_at_40:
        return f"B03{feed_type if same_polarity else pol_type}{day_tag}"
    return None

def show_b_model_results(model_outputs, report_time):
    base_names = {
        "B01a[0]": "Same Polarity, *Origin to |40| today",
        "B01a[≠0]": "Same Polarity, *Origin to |40| ≠[0]",
        "B01b[0]": "Same or mixed polarities, *Origin to |40| today",
        "B01b[≠0]": "Same or mixed polarities, *Origin to |40| ≠[0]",
        "B02a[0]": "Same Polarity, no *Origin to |40| today",
        "B02a[≠0]": "Same Polarity, no *Origin to |40| ≠[0]",
        "B02b[0]": "Same or mixed polarities, no *Origin to |40| today",
        "B02b[≠0]": "Same or mixed polarities, no *Origin to |40| ≠[0]",
        "B03a[0]": "Same Polarity, no *Origin to ≠|40| today",
        "B03a[≠0]": "Same Polarity, no *Origin to ≠|40| ≠[0]",
        "B03b[0]": "Same or mixed polarities, no *Origin to ≠|40| today",
        "B03b[≠0]": "Same or mixed polarities, no *Origin to ≠|40| ≠[0]"
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

def run_b_model_detection(df):
    report_time = df["Arrival"].max()
    model_outputs = detect_B_models(df, report_time)
    show_b_model_results(model_outputs, report_time)
