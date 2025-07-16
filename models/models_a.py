import streamlit as st
import pandas as pd

# Utility to show results in Streamlit with expanders
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


# Core detection function
def run_a_model_detection(df):
    st.markdown("## 🤖 A Model Detection Results")

    if df.empty:
        st.warning("No traveler data to analyze.")
        return

    queries = []

    # Placeholder functions for each A model (replace with your actual logic)
    def query_A01a(df):
        results = []
        for output, group in df.groupby("Output"):
            group_today = group[
                (group["Arrival"].dt.hour == 18) & 
                (group["M #"].isin([0, 40, -40, 54, -54])) &
                (group["Arrival"].dt.date == pd.Timestamp.now().date())
            ]
            if not group_today.empty:
                results.append(group_today)
        return results

    def query_A01b(df):
        anchor_origins = ["Saturn", "Jupiter", "Kepler-62f", "Kepler-442b"]
        results = []
        for output, group in df.groupby("Output"):
            group_today = group[
                (group["Origin"].isin(anchor_origins)) &
                (group["Arrival"].dt.date == pd.Timestamp.now().date())
            ]
            if not group_today.empty:
                results.append(group_today)
        return results

    # Stub queries for A02–A09 and pair variants
    def query_A02(df): return []
    def query_A03(df): return []
    def query_A04(df): return []
    def query_A05(df): return []
    def query_A06(df): return []
    def query_A07(df): return []
    def query_A08(df): return []
    def query_A09(df): return []

    def query_A01a_pairs(df): return []
    def query_A01b_pairs(df): return []
    def query_A02_pairs(df): return []
    def query_A03_pairs(df): return []

    # Register all queries
    queries = [
        ("Query A01a - Today start @18:00 w/ Strength Traveler", query_A01a, True),
        ("Query A01b - Today w/ Anchor Origin", query_A01b, True),
        ("Query A02 - Same polarity descending trio", query_A02, True),
        ("Query A03 - Cross polarity trio starting with Anchor", query_A03, True),
        ("Query A04", query_A04, True),
        ("Query A05", query_A05, True),
        ("Query A06", query_A06, True),
        ("Query A07", query_A07, True),
        ("Query A08", query_A08, True),
        ("Query A09", query_A09, True),

        # Pair variants
        ("Query A01a-p - Pair variant", query_A01a_pairs, False),
        ("Query A01b-p - Pair variant", query_A01b_pairs, False),
        ("Query A02-p - Pair variant", query_A02_pairs, False),
        ("Query A03-p - Pair variant", query_A03_pairs, False),
    ]

    # Run each query and display
    for label, func, is_trio in queries:
        results = func(df)
        display_query_results(label, results, is_trio=is_trio)
