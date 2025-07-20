# mod_x_02.py — Experimental Model X Detector with formatting like B/C. Ground Tech 7.20.25 0920
# mod_x_03.py — Experimental Model X Detector with formatting like B/C. Ground Tech 7.20.25 1800

import streamlit as st
from collections import defaultdict
import pandas as pd

# --- Classifier for multiple C.00 variants ---
def classify_x00_at_40(seq):
    if seq.shape[0] < 3:
        return None, None

    m_vals = seq["M #"].tolist()
    if len(set(abs(m) for m in m_vals)) != len(m_vals):
        return None, None

    abs_desc = all(abs(m_vals[i]) > abs(m_vals[i+1]) for i in range(len(m_vals)-1))
    if not abs_desc:
        return None, None

    signs = [1 if m > 0 else -1 for m in m_vals]
    if signs[-1] == signs[-2]:
        return None, None

    if abs(m_vals[-1]) != 40 or abs(m_vals[-2]) != 54:
        return None, None

    day = str(seq.iloc[-1]["Day"]).strip()
    is_today = (day == "[0]")

    arrival_dt = pd.to_datetime(seq.iloc[-1]["Arrival"])
    t_min = arrival_dt.hour * 60 + arrival_dt.minute
    arrival_tag = "[aft0]" if t_min >= 120 else "[fwd0]"
    if not is_today:
        arrival_tag = arrival_tag.replace("0", "≠[0]")

    origins = set(seq["Origin"].str.lower())
    has_origin = origins & {"spain", "saturn", "jupiter", "kepler-62", "kepler-44", "trinidad", "tobago", "wasp-12b", "macedonia"}
    origin_code = "o" if has_origin else "t"

    feeds = seq["Feed"].nunique()
    feed_code = "a" if feeds == 1 else "b"

    tag = f"C.00.{feed_code}{origin_code}.{arrival_tag}40"

    label_map = {
        "C.00.at.[fwd0]40": "Before 02:00 Influence Shift, same feed, No *Origin to |54|, |40|, Today",
        "C.00.at.[aft0]40": "After 02:00 Influence Shift, same feed, No *Origin to |54|, |40|, Today",
        "C.00.at.[fwd≠[0]]40": "Before 02:00 Influence Shift, same feed, No *Origin to |54|, |40|, Other days",
        "C.00.at.[aft≠[0]]40": "After 02:00 Influence Shift, same feed, No *Origin to |54|, |40|, Other days",
        "C.00.bt.[fwd0]40": "Before 02:00 Influence Shift, mixed feed, No *Origin to |54|, |40|, Today",
        "C.00.bt.[aft0]40": "After 02:00 Influence Shift, mixed feed, No *Origin to |54|, |40|, Today",
        "C.00.bt.[fwd≠[0]]40": "Before 02:00 Influence Shift, mixed feed, No *Origin to |54|, |40|, Other days",
        "C.00.bt.[aft≠[0]]40": "After 02:00 Influence Shift, mixed feed, No *Origin to |54|, |40|, Other days"
    }

    label = label_map.get(tag, "Model X Influence Shift")
    return tag, label

def classify_vip_01(seq):
    if seq.shape[0] < 3:
        return None, None

    m_vals = seq["M #"].tolist()
    if len(set(abs(m) for m in m_vals)) != len(m_vals):
        return None, None
    if not all(abs(m_vals[i]) > abs(m_vals[i+1]) for i in range(len(m_vals)-1)):
        return None, None
    if abs(m_vals[-1]) not in {40, 54}:
        return None, None

    output_vals = seq["Output"].unique()
    if len(output_vals) != 1:
        return None, None

    origin = seq.iloc[-1]["Origin"].strip().lower()
    if origin not in {"spain", "saturn", "jupiter", "kepler-62", "kepler-44", "trinidad", "tobago", "wasp-12b", "macedonia"}:
        return None, None

    feeds = seq["Feed"].nunique()
    feed_code = "a" if feeds == 1 else "b"

    arrival_dt = pd.to_datetime(seq.iloc[-1]["Arrival"])
    day_tag = str(seq.iloc[-1]["Day"]).strip()

    if day_tag == "[0]":
        if arrival_dt.hour in [17, 18]:
            tag = f"VIP.01.{feed_code}o.[O0].|{abs(m_vals[-1])}|"
            label = f"{'Same' if feed_code == 'a' else 'Mixed'} Feed descends to *Origin |{abs(m_vals[-1])}| at Open today"
        else:
            return None, None
    else:
        tag = f"VIP.01.{feed_code}o.[∀≠0].|{abs(m_vals[-1])}|"
        label = f"{'Same' if feed_code == 'a' else 'Mixed'} Feed descends to *Origin |{abs(m_vals[-1])}| any time other days"

    return tag, label


# --- Detector ---
def detect_X_models(df):
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

    model_outputs = defaultdict(list)
    for output, seq in sequences:
        for classifier in [classify_x00_at_40, classify_vip_01]:
            tag, label = classifier(seq)
            if tag:
                model_outputs[tag].append({
                    "output": output,
                    "label": label,
                    "tag": tag,
                    "timestamp": pd.to_datetime(seq.iloc[-1]["Arrival"]),
                    "sequence": seq
                })
                break
    return model_outputs


# --- Display ---
def run_x_model_detection(df):
    report_time = df["Arrival"].max()
    model_outputs = detect_X_models(df)

    st.subheader("🧪 Model X Results")

    if not model_outputs:
        st.info("No Model X results found.")
        return

    # Cluster Table
    cluster = defaultdict(lambda: {"tags": set(), "count": 0, "latest": None})
    for tag, items in model_outputs.items():
        for r in items:
            out = r["output"]
            cluster[out]["tags"].add(tag)
            cluster[out]["count"] += 1
            ts = r["timestamp"]
            if not cluster[out]["latest"] or ts > cluster[out]["latest"]:
                cluster[out]["latest"] = ts

    st.subheader("📊 Model X Cluster Report")
    rows = []
    for out_val, c in cluster.items():
        rows.append({
            "Output": f"{out_val:,.3f}",
            "Sequences": c["count"],
            "Model Count": len(c["tags"]),
            "Tags Found": ", ".join(sorted(c["tags"])),
            "Latest Arrival": c["latest"].strftime('%-m/%-d/%y %H:%M')
        })
    rows.sort(key=lambda x: x["Output"], reverse=True)
    st.table(rows)

    # Display Results grouped by model family
    family_groups = {
        "C.00.a": "Same Feed Sequences",
        "C.00.b": "Mixed Feed Sequences",
        "VIP.01.": "⭐ VIP Sequences"
    }

    for prefix, title in family_groups.items():
        subset = {k: v for k, v in model_outputs.items() if k.startswith(prefix)}
        if not subset:
            continue
        with st.expander(f"📂 {title} ({len(subset)} tags)", expanded=False):
            for tag, results in sorted(subset.items()):
                header = f"{tag}. {results[0]['label']} – {len(results)} output{'s' if len(results) != 1 else ''}"
                with st.expander(header):
                    grouped = defaultdict(list)
                    for r in results:
                        grouped[r["output"]].append(r)
                    for out_val, items in grouped.items():
                        latest = max(items, key=lambda r: r["timestamp"])
                        hrs = int((report_time - latest["timestamp"]).total_seconds() / 3600)
                        ts = latest["timestamp"].strftime('%-m/%-d/%y %H:%M')
                        with st.expander(f"🔹 Output {out_val:,.3f} – {len(items)} sequence(s) {hrs} hours ago at {ts}"):
                            for res in items:
                                seq = res["sequence"]
                                m_path = " → ".join([f"|{row['M #']}|" for _, row in seq.iterrows()])
                                icons = "".join(["👶" if "sm" in row["Feed"].lower() else "🧔" for _, row in seq.iterrows()])
                                st.markdown(f"{m_path} Cross [{icons}]")
                                st.table(seq.reset_index(drop=True))

    # Largest sequence per query tag
    st.subheader("🏆 Largest Sequence Per Tag")
    tag_largest = []
    for tag, items in model_outputs.items():
        top = max(items, key=lambda x: x["sequence"].shape[0])
        tag_largest.append({
            "Tag": tag,
            "Description": top["label"],
            "Output": f"{top['output']:,.3f}",
            "Length": top["sequence"].shape[0],
            "Time": top["timestamp"].strftime('%-m/%-d/%y %H:%M'),
            "Path": " → ".join([f"|{row['M #']}|" for _, row in top["sequence"].iterrows()]),
            "Sequence": top["sequence"]
        })

    tag_largest.sort(key=lambda x: x["Tag"])
    for entry in tag_largest:
        with st.expander(f"🏷️ {entry['Tag']} | {entry['Length']} travelers to {entry['Output']} at {entry['Time']}"):
            st.markdown(f"**{entry['Description']}**")
            st.markdown(f"**Traveler Path:** {entry['Path']}")
            st.table(entry["Sequence"].reset_index(drop=True))
