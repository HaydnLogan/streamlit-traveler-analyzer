import streamlit as st
import pandas as pd
from collections import defaultdict

# --- Constants ---
STRENGTH_TRAVELERS = {0, 40, -40, 54, -54}
ANCHOR_ORIGINS = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}
EPIC_ORIGINS = {"trinidad", "tobago", "wasp-12b", "macedonia"}

# --- Helpers ---
def feed_icon(feed):
    return "👶" if "sm" in feed.lower() else "🧔"

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

    origins = set(seq["Origin"].str.lower())
    has_origin = origins & (ANCHOR_ORIGINS | EPIC_ORIGINS)
    variant = "o" if has_origin else "t"
    suffix = "[aft0]" if final_hour >= 0 else "[fwd0]"
    tag = f"C.01.{variant}.{suffix}"

    label_map = {
        "C.01.o.[aft0]": "After Midnight Influence Shift *Origin Today",
        "C.01.o.[fwd0]": "Before Midnight Influence Shift *Origin Today",
        "C.01.t.[aft0]": "After Midnight Influence Shift No *Origin Today",
        "C.01.t.[fwd0]": "Before Midnight Influence Shift No *Origin Today"
    }
    label = label_map.get(tag)
    return tag, label

def classify_c02_sequence(seq):
    if seq.shape[0] != 3:
        return None, None

    m_vals = seq["M #"].tolist()
    if m_vals[0] != -m_vals[2]:  # True opposite check
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
    elif t_min < 120:
        suffix = "[E0]"; num = "2"
    elif t_min >= 120:
        suffix = "[L0]"; num = "1"
    else:
        return None, None

    tag = f"C.02.{mid_type}{num}.{suffix}"
    label_map = {
        "C.02.p1.[L0]": "Late Opposites, 0 Mid today",
        "C.02.p2.[E0]": "Early Opposites, 0 Mid today",
        "C.02.p3.[O0]": "Open Opposites, 0 Mid today",
        "C.02.n1.[L0]": "Late Opposites, ≠0 Mid today",
        "C.02.n2.[E0]": "Early Opposites, ≠0 Mid today",
        "C.02.n3.[O0]": "Open Opposites, ≠0 Mid today"
    }
    label = label_map.get(tag)
    return tag, label


def classify_c04_sequence(seq):
    if seq.shape[0] != 3:
        return None, None

    m_vals = [abs(m) for m in seq["M #"]]
    if m_vals != [0, 40, 54]:
        return None, None

    final_day = str(seq.iloc[-1]["Day"]).strip()
    if final_day == "[0]":
        return "C.04.∀1.[0]", "Trio up to |54| today"
    else:
        return "C.04.∀2.[±1]", "Trio up to |54| other days"

# --- Detection ---
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

def detect_C_models(df, run_c01=True, run_c02=True, run_c04=True):
    sequences = find_descending_sequences(df)
    model_outputs = defaultdict(list)

    for output, seq in sequences:
        classifiers = []
        if run_c01:
            classifiers.append(classify_c01_sequence)
        if run_c02:
            classifiers.append(classify_c02_sequence)
        if run_c04:
            classifiers.append(classify_c04_sequence)

        for clf in classifiers:
            tag, label = clf(seq)
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

# --- Cluster Table ---
def show_c_cluster_table(model_outputs):
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
            "Output": f"{out_val:,.3f}",
            "Sequences": c["count"],
            "Model Count": len(c["tags"]),
            "Tags Found": ", ".join(sorted(c["tags"])),
            "Latest Arrival": c["latest"].strftime('%-m/%-d/%y %H:%M')
        })

    rows.sort(key=lambda x: x["Output"], reverse=True)
    st.table(rows)

# --- Streamlit Display ---
def show_c_model_results(model_outputs, report_time):
    st.subheader("🔬 C Model Results")

    expected_tags = {
        "C.01.o.[aft0]": "After Midnight Influence Shift *Origin Today",
        "C.01.o.[fwd0]": "Before Midnight Influence Shift *Origin Today",
        "C.01.t.[aft0]": "After Midnight Influence Shift No *Origin Today",
        "C.01.t.[fwd0]": "Before Midnight Influence Shift No *Origin Today",
        "C.02.p1.[L0]": "Late Opposites, 0 Mid today",
        "C.02.p2.[E0]": "Early Opposites, 0 Mid today",
        "C.02.p3.[O0]": "Open Opposites, 0 Mid today",
        "C.02.n1.[L0]": "Late Opposites, ≠0 Mid today",
        "C.02.n2.[E0]": "Early Opposites, ≠0 Mid today",
        "C.02.n3.[O0]": "Open Opposites, ≠0 Mid today",
        "C.04.∀1.[0]": "Trio up to |54| today",
        "C.04.∀2.[±1]": "Trio up to |54| other days"
    }

    for tag, label in expected_tags.items():
        results = model_outputs.get(tag, [])
        output_count = len(set(r["output"] for r in results)) if results else 0
        header = f"{tag}. {label} – {output_count} output{'s' if output_count != 1 else ''}"

        with st.expander(header):
            if not results:
                st.markdown("No matching sequences.")
                continue

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

# --- Entry Point ---
def run_c_model_detection(df, run_c01=True, run_c02=True, run_c04=True):
    report_time = df["Arrival"].max()
    model_outputs = detect_C_models(df, run_c01, run_c02, run_c04)
    st.write("🧬 Detected C Model outputs:", sum(len(v) for v in model_outputs.values()))
    show_c_cluster_table(model_outputs)
    show_c_model_results(model_outputs, report_time)
