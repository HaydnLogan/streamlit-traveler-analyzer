import streamlit as st
import pandas as pd
from collections import defaultdict

# -----------------------
# Helper functions
# -----------------------

def is_valid_terminal(m_val):
    return m_val in [0, 40, 54]

def feed_icon(feed):
    return "👶" if "sm" in feed.lower() else "🧔"

def sequence_signature(seq):
    return tuple(seq["M #"].tolist())

def classify_A_model(row_0, prior_rows):
    if not is_valid_terminal(row_0["M #"]):
        return None, None

    epic = {"trinidad", "tobago", "wasp-12b", "macedonia"}
    anchor = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}
    t0 = row_0["Arrival"]
    o0 = row_0["Origin"].lower()
    time = "open" if t0.hour == 18 and t0.minute == 0 else \
           "early" if (18 < t0.hour < 2 or (t0.hour == 1 and t0.minute < 59)) else "late"
    is_epic = o0 in epic
    is_anchor = o0 in anchor
    prior = set(prior_rows["Origin"].str.lower())
    strong = bool(prior & epic) or bool(prior & anchor)

    if is_epic and time == "open": return "A01", "Open Epic 0 or |40| or |54|"
    if is_anchor and time == "open": return "A02", "Open Anchor 0 or |40| or |54|"
    if not is_anchor and time == "open" and strong: return "A03", "Open non-Anchor 0 or |40| or |54|"
    if not is_anchor and time == "early" and strong: return "A04", "Early non-Anchor 0 or |40| or |54|"
    if is_anchor and time == "late": return "A05", "Late Anchor 0 or |40| or |54|"
    if not is_anchor and time == "late" and strong: return "A06", "Late non-Anchor 0 or |40| or |54|"
    if not is_anchor and time == "open" and not strong: return "A07", "Open general 0 or |40| or |54|"
    if not is_anchor and time == "early" and not strong: return "A08", "Early general 0 or |40| or |54|"
    if not is_anchor and time == "late" and not strong: return "A09", "Late general 0 or |40| or |54|"
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
            if is_valid_terminal(m):
                if len(path) >= 2:
                    path.append(j)
                    raw_sequences.append(rows.loc[path])
                break
            if abs_m in seen or abs_m >= last_abs:
                continue
            path.append(j)
            seen.add(abs_m)
            last_abs = abs_m
    # Filter out embedded sequences
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
        if not is_valid_terminal(m2):
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
            if seq.shape[0] < 2 or not is_valid_terminal(seq.iloc[-1]["M #"]):
                continue
            sig = sequence_signature(seq)
            if sig in all_signatures:
                continue
            all_signatures.add(sig)
            prior = seq.iloc[:-1]
            last = seq.iloc[-1]
            model, label = classify_A_model(last, prior)
            if model:
                model_outputs[model].append({
                    "label": label,
                    "output": output,
                    "timestamp": last["Arrival"],
                    "sequence": seq,
                    "feeds": seq["Feed"].nunique(),
                    "terminal": last["M #"]
                })

        # Now find 2-member pairs not already used
        pairs = find_pairs(subset, all_signatures)
        for seq in pairs:
            sig = sequence_signature(seq)
            if sig in all_signatures:
                continue
            all_signatures.add(sig)
            prior = seq.iloc[:-1]
            last = seq.iloc[-1]
            model, label = classify_A_model(last, prior)
            if model:
                pr_model = model + "pr"
                model_outputs[pr_model].append({
                    "label": f"Pair to {label}",
                    "output": output,
                    "timestamp": last["Arrival"],
                    "sequence": seq,
                    "feeds": seq["Feed"].nunique(),
                    "terminal": last["M #"]
                })

    return model_outputs, report_time

def show_a_model_results(model_outputs, report_time):
    # --- Label Map ---
    raw_labels = {
        "A01": "Open Epic", "A02": "Open Anchor", "A03": "Open non-Anchor",
        "A04": "Early non-Anchor", "A05": "Late Anchor", "A06": "Late non-Anchor",
        "A07": "Open general", "A08": "Early general", "A09": "Late general"
    }

    def label_with_terminals(base):
        return f"{base} 0 or |40| or |54|"

    base_labels = {k: label_with_terminals(v) for k, v in raw_labels.items()}

    # --- Display Block ---
    st.subheader("🔍 A Model Results")

    for code, base_label in base_labels.items():
        for suffix, sublabel in [("", f"2+ to {base_label}"), ("pr", f"Pair to {base_label}")]:
            key = code + suffix
            results = model_outputs.get(key, [])
            output_count = len(set(r["output"] for r in results))
            header = f"{key}. {sublabel} – {output_count} output{'s' if output_count != 1 else ''}"

            with st.expander(header):
                if not results:
                    st.markdown("No matching outputs.")
                    continue

                # --- Grouping ---
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
                                terminal = res.get("terminal", "‽")
                                st.markdown(f"{m_path} Cross [{icons}] → Terminal: |{terminal}|")
                                st.table(seq.reset_index(drop=True))

                # --- Render Results ---
                if today_results:
                    render_group("📅 Today", today_results)
                if other_results:
                    render_group("📦 Other Days", other_results)
                if not today_results and not other_results:
                    st.markdown("No matching outputs.")

    # --- A02 Diagnostic Tester ---
    try:
        if st.checkbox("Run A02 sequence test"):
            test_data = pd.DataFrame([
                {"Feed": "Bg", "Arrival": "7/14/2025 9:00", "Origin": "hawaii", "M #": 92, "Output": 23305.33, "Day": "[0]"},
                {"Feed": "Bg", "Arrival": "7/17/2025 7:00", "Origin": "mercury", "M #": 80, "Output": 23305.33, "Day": "[0]"},
                {"Feed": "Bg", "Arrival": "7/17/2025 18:00", "Origin": "spain", "M #": 40, "Output": 23305.33, "Day": "[0]"}
            ])
            test_data["Arrival"] = pd.to_datetime(test_data["Arrival"])
            validate_expected_A_model("A02", test_data)
            st.info("🧪 Test Sequence: |92| → |80| → |40| ➡️ Origin: [spain], Time: Open (18:00)")

    except Exception as e:
        st.error(f"A02 test failed: {e}")


def run_a_model_detection(df):
    model_outputs, report_time = detect_A_models(df)
    show_a_model_results(model_outputs, report_time)
    return model_outputs


# --- Unit Test Harness ---
def validate_expected_A_model(tag_expected, df_input):
    result, _ = detect_A_models(df_input)
    if tag_expected in result:
        st.success(f"✅ Found {tag_expected}: {result[tag_expected][0]['label']}")
    else:
        st.error(f"❌ {tag_expected} not found.")

test_data = pd.DataFrame([
    {"Feed": "Bg", "Arrival": "7/14/2025 9:00", "Origin": "hawaii", "M #": 92},
    {"Feed": "Bg", "Arrival": "7/17/2025 7:00", "Origin": "mercury", "M #": 80},
    {"Feed": "Bg", "Arrival": "7/17/2025 18:00", "Origin": "spain", "M #": 40}
])

test_data["Arrival"] = pd.to_datetime(test_data["Arrival"])
test_data["Output"] = 23305.333333  # Simulated output ID
test_data["Day"] = "[0]"  # Indicates today for classification logic
