import streamlit as st
import pandas as pd
from collections import defaultdict

# --- Constants ---
ANCHOR_ORIGINS = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}
EPIC_ORIGINS = {"trinidad", "tobago", "wasp-12b", "macedonia"}
STRENGTH_SET = {0, 40, -40, 54, -54}

# --- Helpers ---
def feed_icon(feed):
    return "👶" if "sm" in feed.lower() else "🧔"

def has_origin(seq):
    origins = set(seq["Origin"].str.lower())
    return bool(origins & (ANCHOR_ORIGINS | EPIC_ORIGINS))

def classify_time_tag(arrival, day_val):
    hour = arrival.hour
    if day_val != "[0]":
        return "[±1]"
    if hour in {17, 18}:
        return "[O0]"
    elif hour < 2:
        return "[fwd0]"
    else:
        return "[aft0]"

def classify_c_model(seq):
    m_list = seq["M #"].tolist()
    abs_list = [abs(m) for m in m_list]
    day_val = str(seq.iloc[-1]["Day"])
    tag = classify_time_tag(seq.iloc[-1]["Arrival"], day_val)
    has_special = has_origin(seq)

    # C.01
    if len(m_list) >= 4 and abs_list[:-1] == sorted(abs_list[:-1], reverse=True):
        signs = [1 if m > 0 else -1 for m in m_list]
        if all(s == signs[0] for s in signs[:-1]) and signs[-1] != signs[0] and m_list[-1] != 0:
            return (f"C.01.{'o' if has_special else 't'}.{tag}",
                    f"{'Late' if tag=='[aft0]' else 'Early'} Influence Shift {'*Origin' if has_special else 'No *Origin'} Today")

    # C.02
    if len(m_list) == 3 and (m_list[0] * m_list[-1] < 0):
        mid = m_list[1]
        code = "p" if mid == 0 else "n"
        if tag == "[aft0]":
            suffix = "1"
        elif tag == "[fwd0]":
            suffix = "2"
        else:
            suffix = "3"
        return (f"C.02.{code}{suffix}.{tag}",
                f"{'Late' if suffix=='1' else 'Early' if suffix=='2' else 'Open'} Opposites, {'0' if code=='p' else '≠0'} Mid today")

    # C.04
    if len(m_list) == 3 and m_list[0] == 0 and abs(m_list[1]) == 40 and abs(m_list[2]) == 54:
        return (f"C.04.∀{'1' if day_val == '[0]' else '2'}.{day_val if day_val == '[0]' else '[±1]'}",
                f"Trio up to |54| {'today' if day_val == '[0]' else 'other days'}")

    return None, None

def detect_C_models(df, report_time):
    model_outputs = defaultdict(list)

    for output_val in df["Output"].unique():
        group = df[df["Output"] == output_val].sort_values("Arrival").reset_index(drop=True)
        n = len(group)
        for i in range(n):
            for j in range(i + 2, min(i + 7, n)):  # Try up to 6-length sequences
                seq = group.iloc[i:j+1].reset_index(drop=True)
                label, description = classify_c_model(seq)
                if label:
                    model_outputs[label].append({
                        "label": description,
                        "output": output_val,
                        "timestamp": seq.iloc[-1]["Arrival"],
                        "sequence": seq,
                        "feeds": seq["Feed"].nunique()
                    })
    return model_outputs

def show_c_model_results(model_outputs, report_time):
    st.subheader("🔍 C Model Results")

    sorted_tags = sorted(model_outputs.keys())

    for code in sorted_tags:
        results = model_outputs.get(code, [])
        if not results:
            continue

        label_text = results[0]["label"]
        output_count = len(set(r["output"] for r in results))
        header = f"{code}. {label_text} – {output_count} output{'s' if output_count != 1 else ''}"

        with st.expander(header):
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

def run_c_model_detection(df):
    report_time = df["Arrival"].max()
    model_outputs = detect_C_models(df, report_time)
    show_c_model_results(model_outputs, report_time)
