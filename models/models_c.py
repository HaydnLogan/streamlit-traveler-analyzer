import streamlit as st
import pandas as pd

# 📦 Reusable UI
def display_query_results(label, results, is_trio=True):
    count = len(results)
    type_label = "trio" if is_trio else "pair"
    label_with_count = f"{label}: {count} {type_label}{'' if count == 1 else 's'}"
    with st.expander(label_with_count, expanded=True):
        if count == 0:
            st.write("No results found.")
        else:
            for idx, group in enumerate(results, 1):
                st.markdown(f"**{type_label.capitalize()} {idx}**")
                st.dataframe(pd.DataFrame(group))


def run_c_model_detection(df):
    st.markdown("## 🌒 C Model Detection Results")

    if df.empty:
        st.warning("No traveler data available.")
        return

    queries = []

    # 📍Strength Travelers
    strength_set = {0, 40, -40, 54, -54}
    anchor_origins = ["Saturn", "Jupiter", "Kepler-62f", "Kepler-442b"]
    epic_origins = ["Trinidad", "Tobago", "WASP-12b", "Macedonia"]

    # 🔍 C01: Influence shift after midnight w/ anchor or strength
    def query_C01(df):
        results = []
        df["Arrival"] = pd.to_datetime(df["Arrival"], errors="coerce")
        df = df[df["Arrival"].dt.hour >= 0]  # after midnight

        for output, group in df.groupby("Output"):
            if len(group) < 3:
                continue

            group_sorted = group.sort_values(by="Arrival")

            # First N travelers = opposite polarity of most recent
            polarities = group_sorted["M #"].apply(lambda x: "+" if x > 0 else "-" if x < 0 else "0").tolist()
            if len(set(polarities)) < 2 or "0" in polarities:
                continue

            # Polarity shift: initial vs last
            if polarities[0] == polarities[-1]:
                continue

            # Strength or anchor present
            m_numbers = set(group_sorted["M #"])
            origins = set(group_sorted["Origin"])
            has_strength = any(m in strength_set for m in m_numbers)
            has_anchor = any(o in anchor_origins for o in origins)

            if has_strength or has_anchor:
                results.append(group_sorted)

        return results

    # 🔍 C02: Exact 3 travelers, first vs last polarity flip, post-midnight
    def query_C02(df):
        results = []
        df["Arrival"] = pd.to_datetime(df["Arrival"], errors="coerce")
        df = df[df["Arrival"].dt.hour >= 0]  # post-midnight only

        for output, group in df.groupby("Output"):
            if len(group) != 3:
                continue

            group_sorted = group.sort_values(by="Arrival")
            polarities = group_sorted["M #"].apply(lambda x: "+" if x > 0 else "-" if x < 0 else "0").tolist()

            if "0" in polarities:
                continue

            if polarities[0] != polarities[-1]:
                results.append(group_sorted)

        return results

    queries = [
        ("Query C01 - Late Night Influence Shift w/ Anchor or Strength", query_C01, True),
        ("Query C02 - 3 Travelers, Polarity Flip, Post-Midnight", query_C02, True),
    ]

    for label, func, is_trio in queries:
        results = func(df)
        display_query_results(label, results, is_trio)
