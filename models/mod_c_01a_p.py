import streamlit as st
import pandas as pd
from collections import defaultdict

# CavAir C Models. Logical expansion for polarity transitions, M # symmetry, and strength traveler sequencing
# Includes reverse-sorted cluster table and grouped visual output

# --- Constants ---
STRENGTH_TRAVELERS = {0, 40, -40, 54, -54}
STAR_ORIGINS = {"anchor", "epic"}

# --- Classifiers ---
def classify_c01_sequence(seq):
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

    origin_set = set(str(o).lower() for o in seq["Origin"])
    variant = "o" if origin_set & STAR_ORIGINS else "t"
    suffix = "[aft0]" if final_hour >= 0 else "[fwd0]"
    tag = f"C.01.{variant}.{suffix}"
    label = f"C.01 {variant.upper()} polarity shift ending in {final_m}"
    return tag, label

def classify_c02_sequence(seq):
    if seq.shape[0] != 3:
        return None, None

    m_vals = seq["M #"].tolist()
    signs = [1 if m > 0 else -1 if m < 0 else 0 for m in m_vals]
    if signs[0] * signs[2] != -1:
        return None, None

    mid_m = m_vals[1]
    mid_type = "p" if mid_m == 0 else "n"

    final = seq.iloc[-1]
    final_day = str(final["Day"]).strip()
    if final_day != "[0]":
        return None, None

    time = pd.to_datetime(final["Arrival"]).time()
    h, m = time.hour, time.minute
    t_min = h * 60 + m

    if t_min in {1020, 1080}:
        suffix = "[O0]"; num = "3"
    elif 1021 <= t_min < 120:
        suffix = "[E0]"; num = "2"
    elif t_min >= 120:
        suffix = "[L0]"; num = "1"
    else:
        return None, None

    tag = f"C.02.{mid_type}{num}.{suffix}"
    label = f"C.02 {mid_type.upper()}-type, {'0' if mid_type=='p' else '≠0'} mid, end M #{final['M #']}"
    return tag, label

def classify_c04_sequence(seq):
    if seq.shape[0] != 3:
        return None, None

    m_vals = [abs(m) for m in seq["M #"]]
    if m_vals != [0, 40, 54]:
        return None, None

    final_day = str(seq.iloc[-1]["Day"]).strip()
    if final_day == "[0]":
        tag = "C.04.∀1.[0]"
        label = "Trio up to |54| today"
    else:
        tag = "C.04.∀2.[±1]"
        label = "Trio up to |54| other days"
    return tag, label

# --- Detection Wrapper ---
def detect_C_models(sequences):
    model_outputs = defaultdict(list)
    for output, seq in sequences:
        for clf in [classify_c01_sequence, classify_c02_sequence, classify_c04_sequence]:
            tag, label = clf(seq)
            if tag:
                model_outputs[tag].append({
                    "output": output,
                    "tag": tag,
                    "label": label,
                    "timestamp": pd.to_datetime(seq.iloc[-1]["Arrival"]),
                    "sequence": seq
                })
                break
    return model_outputs

# --- Cluster Table ---
def show_c_model_cluster(model_outputs):
    cluster = defaultdict(lambda: {"tags": set(), "count": 0, "latest": None})

    for tag, items in model_outputs.items():
        for r in items:
            out = r["output"]
            cluster[out]["tags"].add(tag)
            cluster[out]["count"] += 1
            ts = r["timestamp"]
            if not cluster[out]["latest"] or ts > cluster[out]["latest"]:
                cluster[out]["latest"] = ts

    if not cluster:
        st.info("No C Models matched for cluster display.")
        return

    st.subheader("🔄 C Model Cluster Report (Descending Output)")
    rows = []
    for out_val, c in cluster.items():
        rows.append({
            "Output": out_val,
            "Sequences": c["count"],
            "Model Count": len(c["tags"]),
            "Tags Found": ", ".join(sorted(c["tags"])),
            "Latest Arrival": c["latest"].strftime('%-m/%-d/%y %H:%M")
        })
    rows.sort(key=lambda x: x["Output"], reverse=True)
    st.table(rows)

# --- Streamlit Display ---
def show_c_model_results(model_outputs, report_time):
    st.subheader("🔬 C Model Results")

    sorted_tags = sorted(model_outputs.keys())
    for code in sorted_tags:
        results = model_outputs.get(code, [])
        if not results:
            continue

        label = results[0]["label"]
        output_count = len(set(r["output"] for r in results))
        header = f"{code}. {label} – {output_count} output{'s' if output_count != 1 else ''}"

        with st.expander(header):
            grouped = defaultdict(list)
            for r in results:
                grouped[r["output"]].append(r)

            for out_val, items in grouped.items():
                latest = max(items, key=lambda r: r["timestamp"])
                hrs = int((report_time - latest["timestamp"]).total_seconds() / 3600)
                ts = latest["timestamp"].strftime('%-m/%-d/%y %H:%M')
                subhead = f"🔹 Output {out_val:,.3f} – {len(items)} sequence(s), {hrs} hrs ago at {ts}"

                with st.expander(subhead):
                    for res in items:
                        seq = res["sequence"]
                        m_path = " → ".join([f"|{row['M #']}|" for _, row in seq.iterrows()])
                        st.markdown(f"{m_path}")
                        st.table(seq.reset_index(drop=True))

# --- App Entry Point ---
def run_c_model_detection(df, sequences):
    report_time = df["Arrival"].max()
    model_outputs = detect_C_models(sequences, report_time)
    st.write("🧬 Detected C Model outputs:", sum(len(v) for v in model_outputs.values()))
    show_c_model_cluster(model_outputs)
    show_c_model_results(model_outputs, report_time)
