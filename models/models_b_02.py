


# Detection Function
def detect_B_models(df):
    from collections import defaultdict

    def sequence_signature(seq):
        return tuple(seq["M #"].tolist())

    def is_descending_by_abs(seq):
        m_vals = seq["M #"].tolist()
        abs_vals = [abs(m) for m in m_vals]
        return all(abs_vals[i] > abs_vals[i + 1] for i in range(len(abs_vals) - 1))

    def get_polarity(seq):
        signs = [m > 0 for m in seq["M #"]]
        return "same" if all(sign == signs[0] for sign in signs) else "mixed"

    def anchor_or_epic_present(seq):
        key = set(["spain", "saturn", "jupiter", "kepler-62", "kepler-44", 
                   "trinidad", "tobago", "wasp-12b", "macedonia"])
        return any(origin.lower() in key for origin in seq["Origin"])

    def classify_b_sequence(seq, report_time):
        # This will be the logic where we match to one of 12 B types
        # Will return something like: ("B01b[0]", "Mixed Polarity to |40| Today")

        # Stub for now, until we wire the rule logic next
        return None, None

    report_time = df["Arrival"].max()
    model_outputs = defaultdict(list)
    all_signatures = set()

    for output_val in df["Output"].unique():
        subset = df[df["Output"] == output_val].sort_values("Arrival").reset_index(drop=True)

        for i in range(len(subset)):
            path = []
            for j in range(i, len(subset)):
                m = subset.iloc[j]["M #"]
                if m == 0:
                    break
                path.append(j)
                if len(path) >= 3:
                    seq = subset.iloc[path + [j]].reset_index(drop=True)
                    if not is_descending_by_abs(seq):
                        continue
                    sig = sequence_signature(seq)
                    if sig in all_signatures:
                        continue
                    all_signatures.add(sig)

                    label, label_text = classify_b_sequence(seq, report_time)
                    if label:
                        model_outputs[label].append({
                            "label": label_text,
                            "output": output_val,
                            "timestamp": seq.iloc[-1]["Arrival"],
                            "sequence": seq,
                            "feeds": seq["Feed"].nunique()
                        })
    return model_outputs, report_time


# Classifier Logic
def classify_b_sequence(seq, report_time):
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
    polarity = "same" if all(m > 0 for m in seq["M #"]) or all(m < 0 for m in seq["M #"]) else "mixed"
    feeds = seq["Feed"].nunique()
    same_feed = feeds == 1

    if seq.shape[0] < 3:
        return None, None

    # B01 group: *Origin present
    if has_anchor_or_epic:
        if is_40:
            if polarity == "same" and same_feed and today_count >= 2 and is_today:
                return "B01a[0]", "Same Polarity to |40| Today w/ Anchor/EPIC"
            if polarity == "same" and same_feed and not is_today:
                return "B01a[≠0]", "Same Polarity to |40| Not Today w/ Anchor/EPIC"
            if polarity == "mixed" and today_count >= 2 and is_today:
                return "B01b[0]", "Mixed Polarity to |40| Today w/ Anchor/EPIC"
            if polarity == "mixed" and not is_today:
                return "B01b[≠0]", "Mixed Polarity to |40| Not Today w/ Anchor/EPIC"
        else:
            if polarity == "same" and same_feed and today_count >= 2 and is_today:
                return "B03a[0]", "Same Polarity to ≠|40| Today w/ Anchor/EPIC"
            if polarity == "same" and same_feed and not is_today:
                return "B03a[≠0]", "Same Polarity to ≠|40| Not Today w/ Anchor/EPIC"
            if polarity == "mixed" and today_count >= 2 and is_today:
                return "B03b[0]", "Mixed Polarity to ≠|40| Today w/ Anchor/EPIC"
            if polarity == "mixed" and not is_today:
                return "B03b[≠0]", "Mixed Polarity to ≠|40| Not Today w/ Anchor/EPIC"

    # B02 group: no *Origin
    else:
        if is_40:
            if polarity == "same" and same_feed and today_count >= 2 and is_today:
                return "B02a[0]", "Same Polarity to |40| Today no Anchor/EPIC"
            if polarity == "same" and same_feed and not is_today:
                return "B02a[≠0]", "Same Polarity to |40| Not Today no Anchor/EPIC"
            if polarity == "mixed" and today_count >= 2 and is_today:
                return "B02b[0]", "Mixed Polarity to |40| Today no Anchor/EPIC"
            if polarity == "mixed" and not is_today:
                return "B02b[≠0]", "Mixed Polarity to |40| Not Today no Anchor/EPIC"
        else:
            # No sequence defined for B03 no anchor/epic, correct?
            return None, None

    return None, None


# Display Function
def show_b_model_results(model_outputs, report_time):
    import streamlit as st
    from collections import defaultdict

    st.subheader("🔵 B Model Results")

    for key, results in model_outputs.items():
        output_count = len(set(r["output"] for r in results))
        header = f"{key}. {results[0]['label']} – {output_count} output{'s' if output_count != 1 else ''}"

        with st.expander(header):
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
                            icons = "".join(["👶" if "sm" in row["Feed"].lower() else "🧔" for _, row in seq.iterrows()])
                            st.markdown(f"{m_path} Cross [{icons}]")
                            st.table(seq.reset_index(drop=True))

            if today_results:
                render_group("📅 Today", today_results)
            if other_results:
                render_group("📦 Other Days", other_results)
            if not today_results and not other_results:
                st.markdown("No matching outputs.")


# Wrapper Function
def run_b_model_detection(df):
    model_outputs, report_time = detect_B_models(df)
    show_b_model_results(model_outputs, report_time)
    return model_outputs

