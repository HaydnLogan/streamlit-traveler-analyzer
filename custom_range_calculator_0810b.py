"""
Custom Range Calculator (fixed)

- Robust tz-naive datetime handling (no ambiguous Series truth checks).
- Always include WASP-12b (weekly) and Macedonia (monthly) with correct synthetic Arrival:
    * WASP-12b: this week / last week / two weeks ago = most recent Sunday at day-start (17:00 or 18:00)
    * Macedonia: this month / last month / two months ago = first day of month at day-start
- Compute "Input @ 18:00" and "Input @ Report" ONCE per feed (small/big) and paste constants by feed.
- Compute "Input @ Arrival" per row (vectorized) and fill Diff columns.
- Output Traveler Report with exact column order:
  Feed, ddd, Arrival, Day, Origin, M Name, M #, R #, Tag, Family,
  Input @ 18:00, Diff @ 18:00, Input @ Arrival, Diff @ Arrival,
  Input @ Report, Diff @ Report, Output
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None  # allow running without Streamlit


# --------------------------
# Datetime / parsing helpers
# --------------------------

def safe_to_datetime(x, errors="coerce"):
    """Convert Series or scalar to pandas datetime, drop tz, return tz-naive."""
    try:
        ts = pd.to_datetime(x, errors=errors)
        if isinstance(ts, pd.Series):
            if pd.api.types.is_datetime64tz_dtype(ts):
                return ts.dt.tz_localize(None)
            return ts
        # scalar
        if isinstance(ts, pd.Timestamp) and ts.tz is not None:
            return ts.tz_localize(None)
        return ts
    except Exception:
        return pd.NaT if not isinstance(x, pd.Series) else pd.Series([pd.NaT] * len(x))


def ensure_timezone_naive(x):
    """Return tz-naive datetime/Series. Handles Series first, scalars second."""
    try:
        if isinstance(x, pd.Series):
            if not pd.api.types.is_datetime64_any_dtype(x):
                x = safe_to_datetime(x)
            if pd.api.types.is_datetime64tz_dtype(x):
                return x.dt.tz_localize(None)
            return x
        if x is None:
            return None
        if isinstance(x, pd.Timestamp):
            return x.tz_localize(None) if x.tz is not None else x
        ts = safe_to_datetime(x)
        if isinstance(ts, pd.Timestamp):
            return ts.tz_localize(None) if ts.tz is not None else ts
        return ts
    except Exception:
        return x


def day_start_anchor(report_dt: datetime, start_hour: int) -> datetime:
    """Return the most recent day-start anchor at start_hour relative to report_dt."""
    a = report_dt.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    return a if report_dt >= a else (a - timedelta(days=1))


def sunday_start(ts: datetime, start_hour: int) -> datetime:
    """Most-recent Sunday at selected start hour (17 or 18)."""
    anchor = day_start_anchor(ts, start_hour)
    days_back = (anchor.weekday() + 1) % 7  # Mon=0..Sun=6
    sunday = anchor - timedelta(days=days_back)
    return sunday.replace(hour=start_hour, minute=0, second=0, microsecond=0)


def month_start(ts: datetime, start_hour: int) -> datetime:
    """Most-recent month start (day=1) at start hour."""
    anchor = day_start_anchor(ts, start_hour)
    return anchor.replace(day=1, hour=start_hour, minute=0, second=0, microsecond=0)


def shift_months(dt0: datetime, months_back: int, start_hour: int) -> datetime:
    """Return the first day of the month 'months_back' ago at start_hour."""
    y, m = dt0.year, dt0.month
    m -= months_back
    while m <= 0:
        m += 12
        y -= 1
    return datetime(y, m, 1, start_hour, 0, 0)


def ensure_time_dt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create df['time_dt'] tz-naive, robust to strings with 'Z' and +/-HH:MM or +/-HHMM suffixes,
    or datetimes already present. Leaves other columns unchanged.
    """
    out = df.copy()
    if "time" not in out.columns:
        out["time_dt"] = pd.NaT
        return out

    t = out["time"]
    # Already datetime?
    if pd.api.types.is_datetime64_any_dtype(t):
        out["time_dt"] = t.dt.tz_localize(None) if pd.api.types.is_datetime64tz_dtype(t) else t
        return out

    s = t.astype(str).str.strip()
    s = s.str.replace("Z", "", regex=False)                          # drop trailing Z
    s = s.str.replace(r"([+-]\d{2}):?(\d{2})$", "", regex=True)      # drop +HH:MM or +HHMM
    s = s.str.replace("T", " ", regex=False)                         # ISO T -> space
    out["time_dt"] = pd.to_datetime(s, errors="coerce")
    return out


# -----------------------------------------
# Feed "Open" lookup and constant pasting
# -----------------------------------------

_TIME_CANDIDATES = [
    "time", "Time", "timestamp", "Timestamp", "datetime", "Datetime",
    "date", "Date", "ts", "Ts"
]
_OPEN_CANDIDATES = [
    "open", "Open", "OPEN", "o", "O", "Open Price", "open_price", "openPrice"
]


def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _prep_feed_df(feed_df: pd.DataFrame):
    """
    Prepare a compact df with time_dt (tz-naive, sorted asc) and Open (float).
    Works with lowercase 'time'/'open' as in typical CSVs.
    """
    if feed_df is None or len(feed_df) == 0:
        return None

    tcol = _pick_col(feed_df, _TIME_CANDIDATES)
    ocol = _pick_col(feed_df, _OPEN_CANDIDATES)
    if tcol is None or ocol is None:
        if st is not None:
            st.warning(f"Input prep: could not find time/open cols. Have={list(feed_df.columns)} picked time={tcol} open={ocol}")
        return None

    out = feed_df[[tcol, ocol]].copy()
    out = out.rename(columns={tcol: "time", ocol: "open"})
    out = ensure_time_dt(out)
    out = out.dropna(subset=["time_dt"])
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out = out.dropna(subset=["open"])
    out = out.sort_values("time_dt", kind="mergesort")
    out = out.rename(columns={"open": "Open"})

    if st is not None and len(out):
        st.info(f"Feed prep: time='{tcol}', open='{ocol}'. Range {out['time_dt'].iloc[0]} -> {out['time_dt'].iloc[-1]} ({len(out)} rows)")
    return out[["time_dt", "Open"]]


def _open_at_or_before_fast(prepped: pd.DataFrame, when_dt: datetime):
    """Return the last Open at or before when_dt using searchsorted (O(log n))."""
    if prepped is None or len(prepped) == 0 or when_dt is None:
        return None
    if not isinstance(when_dt, datetime):
        when_dt = safe_to_datetime(when_dt)
        if isinstance(when_dt, pd.Series) or pd.isna(when_dt):
            return None
    if not prepped["time_dt"].is_monotonic_increasing:
        prepped = prepped.sort_values("time_dt", kind="mergesort")
    idx = prepped["time_dt"].searchsorted(when_dt, side="right") - 1
    if idx < 0:
        return None
    return float(prepped["Open"].iloc[idx])


def _compute_feed_inputs(small_df, big_df, report_time, day_start_hour):
    """Compute the four constants: small@start, small@report, big@start, big@report."""
    rpt_dt = safe_to_datetime(report_time)
    if isinstance(rpt_dt, pd.Series) or pd.isna(rpt_dt):
        if st is not None:
            st.warning(f"Report time not parseable: {report_time}")
        return (None, None, None, None)

    start_dt = day_start_anchor(rpt_dt, day_start_hour)

    sm_p = _prep_feed_df(small_df) if isinstance(small_df, pd.DataFrame) else None
    bg_p = _prep_feed_df(big_df)   if isinstance(big_df, pd.DataFrame)   else None

    sm18 = _open_at_or_before_fast(sm_p, start_dt)
    smrp = _open_at_or_before_fast(sm_p, rpt_dt)
    bg18 = _open_at_or_before_fast(bg_p, start_dt)
    bgrp = _open_at_or_before_fast(bg_p, rpt_dt)

    if st is not None:
        st.info(
            "INPUT DEBUG — anchors: start={} | report={}  •  Small: Open@{:02d}:00={} | Open@report={}  •  Big: Open@{:02d}:00={} | Open@report={}"
            .format(start_dt, rpt_dt, day_start_hour, sm18, smrp, day_start_hour, bg18, bgrp)
        )
        if sm_p is None:
            st.warning("Small feed: prep returned None (missing time/open?)")
        else:
            st.info(f"Small feed range: {sm_p['time_dt'].min()} -> {sm_p['time_dt'].max()}")
            if sm18 is None or smrp is None:
                st.warning("Small feed: no Open at/before one or both anchors (anchors outside range?)")
        if big_df is not None:
            if bg_p is None:
                st.warning("Big feed: prep returned None (missing time/open?)")
            else:
                st.info(f"Big feed range: {bg_p['time_dt'].min()} -> {bg_p['time_dt'].max()}")
                if bg18 is None or bgrp is None:
                    st.warning("Big feed: no Open at/before one or both anchors (anchors outside range?)")

    return (sm18, smrp, bg18, bgrp)


def _apply_inputs_columns_and_debug(result_df, small_df, big_df, report_time, day_start_hour=18, show_debug=True):
    """
    Paste constant 'Input @ 18:00' and 'Input @ Report' values by feed into result_df.
    Feed detection accepts: sm/small/s/small feed and bg/big/b/big feed.
    """
    if result_df is None or len(result_df) == 0:
        return result_df

    sm18, smrp, bg18, bgrp = _compute_feed_inputs(small_df, big_df, report_time, day_start_hour)

    # Feed masks
    feed = None
    if "Feed" in result_df.columns:
        feed = result_df["Feed"]
    elif "data_source" in result_df.columns:
        feed = result_df["data_source"]
    if feed is None:
        return result_df

    f_lower = feed.astype(str).str.strip().str.lower()
    sm_mask = f_lower.isin(["sm", "small", "s", "small feed"])
    bg_mask = f_lower.isin(["bg", "big", "b", "big feed"])

    if "Input @ 18:00" not in result_df.columns:
        result_df["Input @ 18:00"] = np.nan
    if "Input @ Report" not in result_df.columns:
        result_df["Input @ Report"] = np.nan

    if sm18 is not None:
        result_df.loc[sm_mask, "Input @ 18:00"] = sm18
    if smrp is not None:
        result_df.loc[sm_mask, "Input @ Report"] = smrp
    if bg18 is not None:
        result_df.loc[bg_mask, "Input @ 18:00"] = bg18
    if bgrp is not None:
        result_df.loc[bg_mask, "Input @ Report"] = bgrp

    return result_df


# -----------------------------
# H/L/C scanning and utilities
# -----------------------------

def find_new_data_changes(df_source, report_time, origin_name, scope_days=20):
    """
    Detect points where origin's H/L/C changed in the last scope_days.
    Returns a list of dicts with keys: datetime, origin, H, L, C, avg, spread.
    """
    if df_source is None or origin_name is None:
        return []
    h_col, l_col, c_col = f"{origin_name} H", f"{origin_name} L", f"{origin_name} C"
    if not all(col in df_source.columns for col in [h_col, l_col, c_col]):
        return []

    df = ensure_time_dt(df_source)
    df = df.dropna(subset=["time_dt"]).sort_values("time_dt", ascending=False)

    rpt_dt = safe_to_datetime(report_time)
    if not isinstance(rpt_dt, pd.Timestamp):
        return []
    cutoff = rpt_dt - timedelta(days=scope_days)
    recent = df[df["time_dt"] >= cutoff]
    if recent.empty:
        return []

    changes = []
    last = None
    for _, row in recent.sort_values("time_dt").iterrows():
        cur = (row[h_col], row[l_col], row[c_col])
        if last is None or cur != last:
            avg = (row[h_col] + row[l_col] + row[c_col]) / 3.0
            spread = (row[h_col] - row[l_col])
            changes.append({
                "datetime": row["time_dt"],
                "origin": origin_name,
                "H": row[h_col], "L": row[l_col], "C": row[c_col],
                "avg": avg, "spread": spread
            })
            last = cur
    return changes[::-1]  # newest last


def find_most_current_data(df_source, report_time, origin_name, scope_days=20):
    """Return the latest H/L/C available at/before report_time for origin; None if missing."""
    rows = find_new_data_changes(df_source, report_time, origin_name, scope_days)
    return rows[-1] if rows else None


def calculate_raw_m_values(hlc_data, range_low, range_high):
    """Compute raw m bounds given desired output range (inclusive)."""
    spread = hlc_data.get("spread", 0.0)
    avg = hlc_data.get("avg", None)
    if spread is None or avg is None or spread == 0:
        return None
    low_m = (range_low - avg) / spread
    high_m = (range_high - avg) / spread
    return {
        "raw_m_low": float(min(low_m, high_m)),
        "raw_m_high": float(max(low_m, high_m)),
    }


# -----------------------------
# Traveler report schema helpers
# -----------------------------

def _map_feed_type(data_source: str) -> str:
    s = (data_source or "").strip().lower()
    if s in ("sm", "small", "s", "small feed"):
        return "Small"
    if s in ("bg", "big", "b", "big feed"):
        return "Big"
    return data_source or "Small"


_M_VALUE_CANDIDATES = [
    "M #", "M", "M value", "M Value", "M_Value", "M_value", "m value", "m_value"
]
def _detect_m_value_col(measurement_df):
    for c in _M_VALUE_CANDIDATES:
        if c in measurement_df.columns:
            return c
    return None


def _compute_day_index_series(arrival_series, report_time, day_start_hour: int):
    rpt = safe_to_datetime(report_time)
    if not isinstance(rpt, pd.Timestamp):
        return pd.Series(["[0]"] * len(arrival_series), index=arrival_series.index)
    rpt_anchor = day_start_anchor(rpt, day_start_hour)
    arr = safe_to_datetime(arrival_series)
    arr_anchor = arr.apply(lambda x: day_start_anchor(x, day_start_hour) if isinstance(x, pd.Timestamp) else pd.NaT)
    d = (arr_anchor - rpt_anchor).dt.days.fillna(0).astype(int)
    return d.map(lambda k: "[0]" if k == 0 else (f"[{k}]" if k < 0 else f"[+{k}]"))


def _open_at_or_before_many(prepped_df, when_series):
    """
    prepped_df: DataFrame with ascending 'time_dt' and 'Open'
    when_series: Series of datetimes (tz-naive)
    Returns numpy array of floats/np.nan
    """
    if prepped_df is None or len(prepped_df) == 0 or when_series is None or len(when_series) == 0:
        return np.array([np.nan] * len(when_series)) if hasattr(when_series, "__len__") else np.array([])
    times = prepped_df["time_dt"].to_numpy()
    vals  = prepped_df["Open"].to_numpy()
    w = safe_to_datetime(when_series).to_numpy()
    idx = np.searchsorted(times, w, side="right") - 1
    idx[idx < 0] = -1
    out = np.where(idx == -1, np.nan, vals[idx])
    return out


def _apply_input_at_arrival_and_diffs(result_df, small_df, big_df, report_time, day_start_hour: int):
    if result_df is None or len(result_df) == 0:
        return result_df

    result_df["Arrival"] = safe_to_datetime(result_df["Arrival"])
    sm_p = _prep_feed_df(small_df) if isinstance(small_df, pd.DataFrame) else None
    bg_p = _prep_feed_df(big_df)   if isinstance(big_df, pd.DataFrame)   else None

    for c in ["Input @ Arrival", "Input @ 18:00", "Input @ Report"]:
        if c not in result_df.columns:
            result_df[c] = np.nan

    f = result_df.get("Feed")
    f_lower = f.astype(str).str.lower()
    sm_mask = f_lower.isin(["small", "sm", "s", "small feed"])
    bg_mask = f_lower.isin(["big", "bg", "b", "big feed"])

    if sm_mask.any():
        result_df.loc[sm_mask, "Input @ Arrival"] = _open_at_or_before_many(sm_p, result_df.loc[sm_mask, "Arrival"])
    if bg_mask.any():
        result_df.loc[bg_mask, "Input @ Arrival"] = _open_at_or_before_many(bg_p, result_df.loc[bg_mask, "Arrival"])

    if "Output" in result_df.columns:
        result_df["Diff @ 18:00"]   = result_df["Output"] - pd.to_numeric(result_df["Input @ 18:00"], errors="coerce")
        result_df["Diff @ Arrival"] = result_df["Output"] - pd.to_numeric(result_df["Input @ Arrival"], errors="coerce")
        result_df["Diff @ Report"]  = result_df["Output"] - pd.to_numeric(result_df["Input @ Report"], errors="coerce")

    result_df["ddd"] = result_df["Arrival"].dt.strftime("%a")
    result_df["Day"] = _compute_day_index_series(result_df["Arrival"], report_time, day_start_hour)

    return result_df


_DESIRED_COLS = [
    "Feed", "ddd", "Arrival", "Day", "Origin", "M Name", "M #", "R #", "Tag", "Family",
    "Input @ 18:00", "Diff @ 18:00", "Input @ Arrival", "Diff @ Arrival",
    "Input @ Report", "Diff @ Report", "Output"
]
def _finalize_columns_order(df):
    for c in _DESIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[_DESIRED_COLS]


# -----------------------------
# Row builder for valid M's
# -----------------------------

def find_valid_m_values(
    measurement_df, raw_m_low, raw_m_high, hlc_data,
    range_low, range_high, is_high_range=False,
    data_source="Unknown", report_time=None, small_df=None, batch_inputs=None,
    day_start_hour: int = 18
):
    """
    Filter measurement_df by raw-m window and origin's HLC into final rows,
    producing the Traveler Report schema.
    """
    if measurement_df is None or measurement_df.empty:
        return pd.DataFrame()

    m_col = _detect_m_value_col(measurement_df)
    if m_col is None:
        return pd.DataFrame()

    m_series = pd.to_numeric(measurement_df[m_col], errors="coerce")
    mask = m_series.between(raw_m_low, raw_m_high, inclusive="both")
    subset = measurement_df.loc[mask].copy()
    if subset.empty:
        return subset

    feed_type = _map_feed_type(data_source)
    arrival_dt = hlc_data["datetime"]
    origin     = hlc_data["origin"]
    avg        = hlc_data["avg"]
    spread     = hlc_data["spread"]

    m_name_col = next((c for c in ["M Name", "m name", "M_Name"] if c in subset.columns), None)
    m_num_col  = next((c for c in ["M #", "M"] if c in subset.columns), None)
    r_num_col  = next((c for c in ["R #", "r #"] if c in subset.columns), None)
    tag_col    = next((c for c in ["Tag", "tag"] if c in subset.columns), None)
    fam_col    = next((c for c in ["Family", "family"] if c in subset.columns), None)

    out = pd.DataFrame({
        "Feed": feed_type,
        "ddd": "",
        "Arrival": arrival_dt,
        "Day": "",
        "Origin": origin,
        "M Name": subset[m_name_col] if m_name_col else subset[m_col].astype(str).map(lambda x: f"M{x}"),
        "M #": subset[m_num_col] if m_num_col else subset[m_col],
        "R #": subset[r_num_col] if r_num_col else "",
        "Tag": subset[tag_col] if tag_col else "",
        "Family": subset[fam_col] if fam_col else "",
        "Input @ 18:00": np.nan,
        "Diff @ 18:00":  np.nan,
        "Input @ Arrival": np.nan,
        "Diff @ Arrival":  np.nan,
        "Input @ Report": np.nan,
        "Diff @ Report":  np.nan,
        "Output": avg + pd.to_numeric(subset[m_col], errors="coerce") * spread,
    })

    out["Arrival"] = safe_to_datetime(out["Arrival"])
    return out


# -----------------------------
# Core processors and public API
# -----------------------------

def _always_include_special_origins(origins):
    base = {o.strip() for o in origins}
    base.update(["WASP-12b", "Macedonia"])
    return sorted(base)


def process_custom_ranges_advanced(
    measurement_df, small_df, report_time, custom_ranges, scope_days=20, big_df=None, run_model_g=False,
    day_start_hour=18  # fixed at 18 for custom, unless you prefer threading the UI choice
):
    """
    Process custom ranges; ensures WASP-12b and Macedonia are synthesized correctly.
    Returns a DataFrame of results with the Traveler Report schema.
    """
    if measurement_df is None or measurement_df.empty or small_df is None:
        return pd.DataFrame()

    origins = [c[:-2] for c in small_df.columns if c.endswith(" H")]
    origins = _always_include_special_origins(origins)

    rpt_dt = safe_to_datetime(report_time)
    if not isinstance(rpt_dt, pd.Timestamp):
        return pd.DataFrame()

    rows = []

    def _push_for_hlc(hlc, feed_tag):
        for (rng_low, rng_high, is_high) in custom_ranges:
            m_bounds = calculate_raw_m_values(hlc, rng_low, rng_high)
            if not m_bounds:
                continue
            df_add = find_valid_m_values(
                measurement_df,
                m_bounds["raw_m_low"], m_bounds["raw_m_high"],
                hlc, rng_low, rng_high, is_high_range=is_high,
                data_source=feed_tag, report_time=rpt_dt, small_df=small_df, day_start_hour=day_start_hour
            )
            if df_add is not None and not df_add.empty:
                rows.append(df_add)

    for origin in origins:
        name_l = origin.lower()
        if name_l in ("wasp-12b", "wasp"):
            base_sm = find_most_current_data(small_df, rpt_dt, "WASP-12b", scope_days)
            if base_sm:
                for w in (0, 1, 2):
                    when = sunday_start(rpt_dt, day_start_hour) - timedelta(weeks=w)
                    hlc = dict(base_sm)
                    hlc["datetime"] = when
                    hlc["origin"] = "Wasp-12b" if w == 0 else f"Wasp-12b[-{w}]"
                    _push_for_hlc(hlc, "sm")
            if big_df is not None:
                base_bg = find_most_current_data(big_df, rpt_dt, "WASP-12b", scope_days)
                if base_bg:
                    for w in (0, 1, 2):
                        when = sunday_start(rpt_dt, day_start_hour) - timedelta(weeks=w)
                        hlc = dict(base_bg)
                        hlc["datetime"] = when
                        hlc["origin"] = "Wasp-12b" if w == 0 else f"Wasp-12b[-{w}]"
                        _push_for_hlc(hlc, "bg")

        elif name_l == "macedonia":
            base_sm = find_most_current_data(small_df, rpt_dt, "Macedonia", scope_days)
            if base_sm:
                m0 = month_start(rpt_dt, day_start_hour)
                for mback in (0, 1, 2):
                    when = shift_months(m0, mback, day_start_hour)
                    hlc = dict(base_sm)
                    hlc["datetime"] = when
                    hlc["origin"] = "Macedonia" if mback == 0 else f"Macedonia[-{mback}]"
                    _push_for_hlc(hlc, "sm")
            if big_df is not None:
                base_bg = find_most_current_data(big_df, rpt_dt, "Macedonia", scope_days)
                if base_bg:
                    m0 = month_start(rpt_dt, day_start_hour)
                    for mback in (0, 1, 2):
                        when = shift_months(m0, mback, day_start_hour)
                        hlc = dict(base_bg)
                        hlc["datetime"] = when
                        hlc["origin"] = "Macedonia" if mback == 0 else f"Macedonia[-{mback}]"
                        _push_for_hlc(hlc, "bg")

        else:
            changes_sm = find_new_data_changes(small_df, rpt_dt, origin, scope_days)
            for hlc in changes_sm:
                _push_for_hlc(hlc, "sm")
            if big_df is not None:
                changes_bg = find_new_data_changes(big_df, rpt_dt, origin, scope_days)
                for hlc in changes_bg:
                    _push_for_hlc(hlc, "bg")

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # Paste Input constants, compute Input @ Arrival and diffs, enforce final columns.
    out = _apply_inputs_columns_and_debug(out, small_df, big_df, rpt_dt, day_start_hour, show_debug=True)
    out = _apply_input_at_arrival_and_diffs(out, small_df, big_df, rpt_dt, day_start_hour)
    out = _finalize_columns_order(out)
    return out


def apply_custom_ranges_advanced(
    df, small_df, report_time, high1, high2, low1, low2,
    use_high1, use_high2, use_low1, use_low2,
    big_df=None, run_model_g=False
):
    """
    Public entry for custom ranges. Signature matches existing app calls.
    Uses day_start_hour=18 for the "Input @ 18:00" column by default.
    """
    custom_ranges = []
    if use_high1: custom_ranges.append((high1[0], high1[1], True))
    if use_high2: custom_ranges.append((high2[0], high2[1], True))
    if use_low1:  custom_ranges.append((low1[0],  low1[1],  False))
    if use_low2:  custom_ranges.append((low2[0],  low2[1],  False))

    return process_custom_ranges_advanced(
        df, small_df, report_time, custom_ranges, scope_days=20, big_df=big_df, run_model_g=run_model_g,
        day_start_hour=18  # fixed 18:00 for custom runs
    )


def process_full_range_advanced(
    measurement_df, small_df, report_time, center, window_radius, scope_days=20, big_df=None, run_model_g=False,
    day_start_hour=18
):
    """
    Process a symmetric window around center, scanning all origins (including specials).
    Returns Traveler Report schema.
    """
    if measurement_df is None or measurement_df.empty or small_df is None:
        return pd.DataFrame()

    origins = [c[:-2] for c in small_df.columns if c.endswith(" H")]
    origins = _always_include_special_origins(origins)

    rpt_dt = safe_to_datetime(report_time)
    if not isinstance(rpt_dt, pd.Timestamp):
        return pd.DataFrame()

    rng_low = center - window_radius
    rng_high = center + window_radius

    rows = []

    def _push_for_hlc(hlc, feed_tag):
        m_bounds = calculate_raw_m_values(hlc, rng_low, rng_high)
        if not m_bounds:
            return
        df_add = find_valid_m_values(
            measurement_df,
            m_bounds["raw_m_low"], m_bounds["raw_m_high"],
            hlc, rng_low, rng_high, is_high_range=(center >= 0),
            data_source=feed_tag, report_time=rpt_dt, small_df=small_df, day_start_hour=day_start_hour
        )
        if df_add is not None and not df_add.empty:
            rows.append(df_add)

    for origin in origins:
        name_l = origin.lower()
        if name_l in ("wasp-12b", "wasp"):
            base_sm = find_most_current_data(small_df, rpt_dt, "WASP-12b", scope_days)
            if base_sm:
                for w in (0, 1, 2):
                    when = sunday_start(rpt_dt, day_start_hour) - timedelta(weeks=w)
                    hlc = dict(base_sm); hlc["datetime"] = when; hlc["origin"] = "Wasp-12b" if w == 0 else f"Wasp-12b[-{w}]"
                    _push_for_hlc(hlc, "sm")
            if big_df is not None:
                base_bg = find_most_current_data(big_df, rpt_dt, "WASP-12b", scope_days)
                if base_bg:
                    for w in (0, 1, 2):
                        when = sunday_start(rpt_dt, day_start_hour) - timedelta(weeks=w)
                        hlc = dict(base_bg); hlc["datetime"] = when; hlc["origin"] = "Wasp-12b" if w == 0 else f"Wasp-12b[-{w}]"
                        _push_for_hlc(hlc, "bg")

        elif name_l == "macedonia":
            base_sm = find_most_current_data(small_df, rpt_dt, "Macedonia", scope_days)
            if base_sm:
                m0 = month_start(rpt_dt, day_start_hour)
                for mback in (0, 1, 2):
                    when = shift_months(m0, mback, day_start_hour)
                    hlc = dict(base_sm); hlc["datetime"] = when; hlc["origin"] = "Macedonia" if mback == 0 else f"Macedonia[-{mback}]"
                    _push_for_hlc(hlc, "sm")
            if big_df is not None:
                base_bg = find_most_current_data(big_df, rpt_dt, "Macedonia", scope_days)
                if base_bg:
                    m0 = month_start(rpt_dt, day_start_hour)
                    for mback in (0, 1, 2):
                        when = shift_months(m0, mback, day_start_hour)
                        hlc = dict(base_bg); hlc["datetime"] = when; hlc["origin"] = "Macedonia" if mback == 0 else f"Macedonia[-{mback}]"
                        _push_for_hlc(hlc, "bg")

        else:
            changes_sm = find_new_data_changes(small_df, rpt_dt, origin, scope_days)
            for hlc in changes_sm:
                _push_for_hlc(hlc, "sm")
            if big_df is not None:
                changes_bg = find_new_data_changes(big_df, rpt_dt, origin, scope_days)
                for hlc in changes_bg:
                    _push_for_hlc(hlc, "bg")

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    out = _apply_inputs_columns_and_debug(out, small_df, big_df, rpt_dt, day_start_hour, show_debug=True)
    out = _apply_input_at_arrival_and_diffs(out, small_df, big_df, rpt_dt, day_start_hour)
    out = _finalize_columns_order(out)
    return out


def apply_full_range_advanced(
    df, small_df, report_time, window_radius, day_start_hour=18, input_value_at_start=None, big_df=None, run_model_g=False
):
    """
    Public entry for full-range scanning (symmetric about the chosen center).
    The 'center' is the user-selected input value (at or before selected time).
    """
    center = input_value_at_start if input_value_at_start is not None else 0.0
    return process_full_range_advanced(
        df, small_df, report_time, center=center, window_radius=window_radius,
        scope_days=20, big_df=big_df, run_model_g=run_model_g, day_start_hour=day_start_hour
    )
