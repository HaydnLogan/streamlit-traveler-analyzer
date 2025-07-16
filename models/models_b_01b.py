import streamlit as st
import pandas as pd
from collections import defaultdict
# this one works, both keplers were wrong. now fixed.
# --- Constants ---
ANCHOR_ORIGINS = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}
EPIC_ORIGINS = {"trinidad", "tobago", "wasp-12b", "macedonia"}

# --- Helpers ---
def feed_icon(feed):
    return "👶" if "sm" in feed.lower() else "🧔"

def sequence_signature(seq):
    return tuple(seq["M #"].tolist())

def is_same_polarity(seq):
    signs = set([1 if m > 0 else -1 for m in seq["M #"] if m != 0])
    return len(signs) == 1

def has_anchor_origin(seq):
    return any(origin.lower() in ANCHOR_ORIGINS for origin in seq["Origin"])

def has_epic_origin(seq):
    return any(origin.lower() in EPIC_ORIGINS for origin in seq["Origin"])

def is_strict_descending_by_abs(seq):
    abs_values = [abs(m) for m in seq["M #"]]
    return abs_values == sorted(abs_values, reverse=True) and len(abs_values) == len(set(abs_values))

def find_descending_sequences(df):
    from collections import defaultdict
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
def classify_b_sequence(seq):
    epic = {"trinidad", "tobago", "wasp-12b", "macedonia"}
    anchor = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}
    origins = set(seq["Origin"].str.lower())
    has_anchor_or_epic = bool(origins & (epic | anchor))

    day_tags = seq["Day"].astype(str).str.lower()
    today_count = sum("[0]" in d for d in day_tags)
    last_day = day_tags.iloc[-1]
    is_today = "[0]" in last_day

    last_m = seq.iloc[-1]["M #"]
    is_40 = last_m == 40

    signs = set(1 if m > 0 else -1 for m in seq["M #"] if m != 0)
    polarity = "a" if len(signs) == 1 else "b"
    feeds = seq["Feed"].nunique()
    same_feed = feeds == 1

    if seq.shape[0] < 3:
        return None, None

    if has_anchor_or_epic:
        if is_40:
            if polarity == "a" and same_feed and today_count >= 2 and is_today:
                return "B01a[0]", "Same Polarity to |40| Today w/ Anchor/EPIC"
            if polarity == "a" and same_feed and not is_today:
                return "B01a[≠0]", "Same Polarity to |40| Not Today w/ Anchor/EPIC"
            if polarity == "b" and today_count >= 2 and is_today:
                return "B01b[0]", "Mixed Polarity to |40| Today w/ Anchor/EPIC"
            if polarity == "b" and not is_today:
                return "B01b[≠0]", "Mixed Polarity to |40| Not Today w/ Anchor/EPIC"
        else:
            if polarity == "a" and same_feed and today_count >= 2 and is_today:
                return "B03a[0]", "Same Polarity to ≠|40| Today w/ Anchor/EPIC"
            if polarity == "a" and same_feed and not is_today:
                return "B03a[≠0]", "Same Polarity to ≠|40| Not Today w/ Anchor/EPIC"
            if polarity == "b" and today_count >= 2 and is_today:
                return "B03b[0]", "Mixed Polarity to ≠|40| Today w/ Anchor/EPIC"
            if polarity == "b" and not is_today:
                return "B03b[≠0]", "Mixed Polarity to ≠|40| Not Today w/ Anchor/EPIC"
    else:
        if is_40:
            if polarity == "a" and same_feed and today_count >= 2 and is_today:
                return "B02a[0]", "Same Polarity to |40| Today no Anchor/EPIC"
            if polarity == "a" and same_feed and not is_today:
                return "B02a[≠0]", "Same Polarity to |40| Not Today no Anchor/EPIC"
            if polarity == "b" and today_count >= 2 and is_today:
                return "B02b[0]", "Mixed Polarity to |40| Today no Anchor/EPIC"
            if polarity == "b" and not is_today:
                return "B02b[≠0]", "Mixed Polarity to |40| Not Today no Anchor/EPIC"

    return None, None


# def find_valid_sequences(df):
#     sequences = []
#     seen_signatures = set()

#     for output in df["Output"].unique():
#         rows = df[df["Output"] == output].sort_values("Arrival").reset_index(drop=True)
#         n = len(rows)
#         for i in range(n):
#             for j in range(i + 3, n + 1):  # start with at least 3-length window
#                 seq = rows.iloc[i:j]
#                 sig = sequence_signature(seq)
#                 if sig in seen_signatures:
#                     continue
#                 if not is_strict_descending_by_abs(seq):
#                     continue
#                 if not is_same_polarity(seq):
#                     continue
#                 if seq["Feed"].nunique() > 1:
#                     continue
#                 if seq.iloc[-1]["M #"] != 40:
#                     continue
#                 if seq["Day"].astype(str).str.contains("\\[0\\]").sum() < 2:
#                     continue
#                 if not (has_anchor_origin(seq) or has_epic_origin(seq)):
#                     continue
#                 seen_signatures.add(sig)
#                 sequences.append(seq)
#     return sequences

# def detect_B_models(df, report_time):
#     model_outputs = defaultdict(list)
#     sequences = find_valid_sequences(df)

#     for seq in sequences:
#         output = seq.iloc[-1]["Output"]
#         day_label = str(seq.iloc[-1].get("Day", ""))
#         is_today = "[0]" in day_label
#         polarity_type = "a" if is_same_polarity(seq) else "b"
#         anchor_present = has_anchor_origin(seq)
#         epic_present = has_epic_origin(seq)
#         has_special_origin = anchor_present or epic_present
#         ends_at_40 = seq.iloc[-1]["M #"] == 40

#         if has_special_origin and ends_at_40:
#             model = f"B01{polarity_type}[0]" if is_today else f"B01{polarity_type}[≠0]"
#         elif not has_special_origin and ends_at_40:
#             model = f"B02{polarity_type}[0]" if is_today else f"B02{polarity_type}[≠0]"
#         elif not ends_at_40:
#             model = f"B03{polarity_type}[0]" if is_today else f"B03{polarity_type}[≠0]"
#         else:
#             continue

#         model_outputs[model].append({
#             "output": output,
#             "timestamp": seq.iloc[-1]["Arrival"],
#             "sequence": seq,
#             "feeds": seq["Feed"].nunique()
#         })

#     return model_outputs

def detect_B_models(df, report_time):
    from collections import defaultdict
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


def show_b_model_results(model_outputs, report_time):
    base_names = {
        "B01a[0]": "Same Polarity Descenders, *Origin to |40| today",
        "B01a[≠0]": "Same Polarity Descenders, *Origin to |40| ≠[0]",
        "B01b[0]": "Mixed Polarity Descenders, *Origin to |40| today",
        "B01b[≠0]": "Mixed Polarity Descenders, *Origin to |40| ≠[0]",
        "B02a[0]": "Same Polarity Descenders, no *Origin to |40| today",
        "B02a[≠0]": "Same Polarity Descenders, no *Origin to |40| ≠[0]",
        "B02b[0]": "Mixed Polarity Descenders, no *Origin to |40| today",
        "B02b[≠0]": "Mixed Polarity Descenders, no *Origin to |40| ≠[0]",
        "B03a[0]": "Same Polarity Descenders to ≠|40| today",
        "B03a[≠0]": "Same Polarity Descenders to ≠|40| ≠[0]",
        "B03b[0]": "Mixed Polarity Descenders to ≠|40| today",
        "B03b[≠0]": "Mixed Polarity Descenders to ≠|40| ≠[0]"
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

    st.write("🔬 Detected B Model outputs:", sum(len(v) for v in model_outputs.values()))
    
    show_b_model_results(model_outputs, report_time)
