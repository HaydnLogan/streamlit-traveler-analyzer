# mod_x_01.py — Experimental Model X Detector with formatting like B/C. GroundTech 7.20.25 0740, 0820

import streamlit as st
from collections import defaultdict
import pandas as pd

# --- Classifier for C.00.at.[fwd0]40 and C.00.at.[aft0]40 ---
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
    if day != "[0]":
        return None, None

    arrival_dt = pd.to_datetime(seq.iloc[-1]["Arrival"])
    t_min = arrival_dt.hour * 60 + arrival_dt.minute
    arrival_tag = "[aft0]" if t_min >= 120 else "[fwd0]"

    origins = set(seq["Origin"].str.lower())
    has_origin = origins & {"spain", "saturn", "jupiter", "kepler-62", "kepler-44", "trinidad", "tobago", "wasp-12b", "macedonia"}
    origin_code = "o" if has_origin else "t"

    feeds = seq["Feed"].nunique()
    feed_code = "a" if feeds == 1 else "b"

    tag = f"C.00.{feed_code}{origin_code}.{arrival_tag}40"
    label = "Before or After Midnight Influence Shift to |40|"
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
        tag, label = classify_x00_at_40(seq)
        if tag:
            model_outputs[tag].append({
                "output": output,
                "label": label,
                "tag": tag,
                "timestamp": pd.to_datetime(seq.iloc[-1]["Arrival"]),
                "sequence": seq
            })
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

    # Display Results grouped by tag and by output
    for tag, results in model_outputs.items():
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
