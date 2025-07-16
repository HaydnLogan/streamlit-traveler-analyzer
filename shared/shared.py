import pandas as pd
import numpy as np
import datetime as dt

def clean_timestamp(value):
    """Parse timestamp or return NaT if invalid."""
    try:
        return pd.to_datetime(value)
    except Exception:
        return pd.NaT

def get_most_recent_time(df, col="time"):
    """Return the most recent non-null timestamp in the specified column."""
    return pd.to_datetime(df[col], errors='coerce').max()

def get_input_value(df, report_time):
    """Find the last known measurement before report_time."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    valid = df[df["time"] <= report_time]
    return valid["value"].iloc[-1] if not valid.empty else None

def process_feed(df, feed_label, report_time, scope_type, scope_value, day_start_hour, measurements, input_value):
    """Process and structure a feed for analysis."""
    results = []
    df = df.copy()
    df["Feed"] = feed_label
    df["Arrival"] = pd.to_datetime(df["time"], errors="coerce")

    # Apply day partitioning
    df["Traveler Day"] = df["Arrival"].apply(lambda t: (
        t - pd.Timedelta(days=1) if t.hour < day_start_hour else t
    ).strftime("%Y-%m-%d"))

    recent = df.sort_values(by="Arrival", ascending=False)

    if scope_type == "Rows":
        trimmed = recent.head(scope_value)
    elif scope_type == "Days":
        unique_days = recent["Traveler Day"].unique()[:scope_value]
        trimmed = recent[recent["Traveler Day"].isin(unique_days)]
    else:
        trimmed = recent

    for _, row in trimmed.iterrows():
        result = {
            "Feed": feed_label,
            "Row": int(row.get("row", -1)),
            "Arrival": row["Arrival"],
            "M #": int(row.get("m #", 0)),
            "Origin": row.get("origin", "Unknown"),
            "Output": float(row.get("output", np.nan)),
            "Type": row.get("type", "N/A"),
        }
        results.append(result)

    return results
