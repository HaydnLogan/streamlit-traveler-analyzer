import streamlit as st
import pandas as pd

# 👶 Small feed = 'Sm', 🧔 Big feed = 'Bg'
FEED_ICONS = {"Sm": "👶", "Bg": "🧔"}

# 📦 UI utility: display results for a query
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


def run_b_model_detection(df):
    st.markdown("## 🧠 B Model Detection Results")

    if df.empty:
        st.warning("No traveler data available.")
        return

    queries = []

    # 🔍 B01: Same polarity, descending abs(M #), same feed, 2+ today, 1 anchor/epic
    def query_B01(df):
        results = []
        anchor_origins = ["Saturn", "Jupiter", "Kepler-62f", "Kepler-442b"]
        epic_origins = ["Trinidad", "Tobago", "WASP-12b", "Macedonia"]

        for output, group in df.groupby("Output"):
            if len(group) < 3:
                continue

            sorted_group = group.copy()
            sorted_group["abs_m"] = sorted_group["M #"].abs()
            sorted_group = sorted_group.sort_values(by="abs_m", ascending=False)

            # Check same polarity
            polarities = set(sorted_group["M #"].apply(lambda x: "+" if x > 0 else "-" if x < 0 else "0"))
            if len(polarities) != 1 or "0" in polarities:
                continue

            # Same feed
            feeds = set(sorted_group["Feed"])
            if len(feeds) > 1:
                continue

            # At least 2 "Today"
            sorted_group["Arrival"] = pd.to_datetime(sorted_group["Arrival"], errors="coerce")
            today = pd.Timestamp.now().date()
            count_today = (sorted_group["Arrival"].dt.date == today).sum()
            if count_today < 2:
                continue

            # Anchor or EPIC origin
            origins = set(sorted_group["Origin"])
            if not any(o in anchor_origins + epic_origins for o in origins):
                continue

            results.append(sorted_group)

        return results

    # 🔍 B02: Mixed polarity, descending abs(M #), any feed, same output, 2+ today, 1 anchor/epic
    def query_B02(df):
        results = []
        anchor_origins = ["Saturn", "Jupiter", "Kepler-62f", "Kepler-442b"]
        epic_origins = ["Trinidad", "Tobago", "WASP-12b", "Macedonia"]

        for output, group in df.groupby("Output"):
            if len(group) < 3:
                continue

            sorted_group = group.copy()
            sorted_group["abs_m"] = sorted_group["M #"].abs()
            sorted_group = sorted_group.sort_values(by="abs_m", ascending=False)

            # Must have mixed polarity
            polarities = set(sorted_group["M #"].apply(lambda x: "+" if x > 0 else "-" if x < 0 else "0"))
            if len(polarities) < 2 or "0" in polarities:
                continue

            # At least 2 "Today"
            sorted_group["Arrival"] = pd.to_datetime(sorted_group["Arrival"], errors="coerce")
            today = pd.Timestamp.now().date()
            count_today = (sorted_group["Arrival"].dt.date == today).sum()
            if count_today < 2:
                continue

            # Anchor or EPIC origin
            origins = set(sorted_group["Origin"])
            if not any(o in anchor_origins + epic_origins for o in origins):
                continue

            results.append(sorted_group)

        return results

    # Register and run queries
    queries = [
        ("Query B01 - Same Polarity Descending, Same Feed, 2+ Today, Anchor/EPIC", query_B01, True),
        ("Query B02 - Mixed Polarity Descending, Any Feed, 2+ Today, Anchor/EPIC", query_B02, True),
    ]

    for label, func, is_trio in queries:
        results = func(df)
        display_query_results(label, results, is_trio)
