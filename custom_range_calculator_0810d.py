"""
Custom Range Calculator for Market Data Analysis
Implements sophisticated range calculation based on H/L/C data from small CSV files.
This 0810 produces data fast! 500 point spread in under 10 seconds, but there is incorrect data! :-(
Macedonia[-1], Macedonia[-2], Wasp-12b[-1], Wasp-12b[-2] are all missing and Macedonia[0] prints the wrong date.
0810d, ChatGPT5 help
"""

import datetime as dt
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
try:
    import streamlit as st
except Exception:
    st = None  # allow running outside Streamlit

# --- tz-safe datetime coercion used across the calculator ---
def safe_to_datetime(x, errors="coerce"):
    """
    Convert Series or scalar to pandas datetime and drop timezone → tz-naive.
    - Series: returns dtype datetime64[ns], tz removed if present
    - Scalar: returns pandas.Timestamp (tz-naive) or NaT
    """
    ts = pd.to_datetime(x, errors=errors)
    # Series path
    if isinstance(ts, pd.Series):
        if pd.api.types.is_datetime64tz_dtype(ts):
            return ts.dt.tz_localize(None)
        return ts
    # Scalar path
    if isinstance(ts, pd.Timestamp) and ts.tz is not None:
        return ts.tz_localize(None)
    return ts

# --- ensure we have a tz-naive 'time_dt' column from 'time' ---
def ensure_time_dt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'time' not in out.columns:
        out['time_dt'] = pd.NaT
        return out
    t = out['time']
    if pd.api.types.is_datetime64_any_dtype(t):
        out['time_dt'] = t.dt.tz_localize(None) if pd.api.types.is_datetime64tz_dtype(t) else t
        return out
    s = t.astype(str).str.strip()
    s = s.str.replace('Z', '', regex=False)                        # drop trailing Z
    s = s.str.replace(r'([+-]\d{2}):?(\d{2})$', '', regex=True)    # drop +HH:MM or +HHMM
    s = s.str.replace('T', ' ', regex=False)                       # ISO T -> space
    out['time_dt'] = pd.to_datetime(s, errors='coerce')
    return out

# --- flexible column detection for feed prep ---
_TIME_CANDIDATES = ["time","Time","timestamp","Timestamp","datetime","Datetime","date","Date","ts","Ts"]
_OPEN_CANDIDATES = ["open","Open","OPEN","o","O","Open Price","open_price","openPrice"]

def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# --- prepare a compact (time_dt, Open) df for binary searches ---
def _prep_feed_df(feed_df: pd.DataFrame):
    if feed_df is None or len(feed_df) == 0:
        return None
    tcol = _pick_col(feed_df, _TIME_CANDIDATES)
    ocol = _pick_col(feed_df, _OPEN_CANDIDATES)
    if tcol is None or ocol is None:
        if st is not None:
            st.warning(f"Input prep: couldn't find time/open. cols={list(feed_df.columns)} picked time={tcol} open={ocol}")
        return None
    out = feed_df[[tcol, ocol]].copy().rename(columns={tcol: "time", ocol: "open"})
    out = ensure_time_dt(out).dropna(subset=["time_dt"])
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out = out.dropna(subset=["open"]).sort_values("time_dt", kind="mergesort")
    out = out.rename(columns={"open": "Open"})
    if st is not None and len(out):
        st.info(f"Feed prep: time='{tcol}', open='{ocol}'. Range {out['time_dt'].iloc[0]} → {out['time_dt'].iloc[-1]} ({len(out)} rows)")
    return out[["time_dt", "Open"]]

# --- last Open at/before a single timestamp (O(log n)) ---
def _open_at_or_before_fast(prepped: pd.DataFrame, when_dt: datetime):
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

# --- compute the 4 constants: small/big @ start and @ report ---
def _compute_feed_inputs(small_df, big_df, report_time, day_start_hour):
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
    # Rich debug
    if st is not None:
        st.info(
            f"INPUT DEBUG — anchors: start={start_dt} | report={rpt_dt}  •  "
            f"Small: Open@{day_start_hour:02d}:00={sm18} | Open@report={smrp}  •  "
            f"Big: Open@{day_start_hour:02d}:00={bg18} | Open@report={bgrp}"
        )
        if sm_p is None:
            st.warning("Small feed: prep returned None (missing 'time'/'open'?)")
        else:
            st.info(f"Small feed range: {sm_p['time_dt'].min()} → {sm_p['time_dt'].max()}")
            if sm18 is None or smrp is None:
                st.warning("Small feed: no Open at/before one or both anchors (anchors outside range?)")
        if big_df is not None:
            if bg_p is None:
                st.warning("Big feed: prep returned None (missing 'time'/'open'?)")
            else:
                st.info(f"Big feed range: {bg_p['time_dt'].min()} → {bg_p['time_dt'].max()}")
                if bg18 is None or bgrp is None:
                    st.warning("Big feed: no Open at/before one or both anchors (anchors outside range?)")
    return (sm18, smrp, bg18, bgrp)

# --- PASTE the 4 constants by feed into the result df + optional debug ---
def _apply_inputs_columns_and_debug(result_df, small_df, big_df, report_time, day_start_hour=18, show_debug=True):
    if result_df is None or len(result_df) == 0:
        return result_df
    sm18, smrp, bg18, bgrp = _compute_feed_inputs(small_df, big_df, report_time, day_start_hour)
    if show_debug and st is not None:
        st.info(
            "INPUT DEBUG (final) - Small: Open@{:02d}:00={} | Open@report={} • Big: Open@{:02d}:00={} | Open@report={}".format(
                day_start_hour, sm18, smrp, day_start_hour, bg18, bgrp
            )
        )
    feed = result_df.get("Feed") if "Feed" in result_df.columns else result_df.get("data_source")
    if feed is None:
        return result_df
    f_lower = feed.astype(str).str.strip().str.lower()
    sm_mask = f_lower.isin(["sm","small","s","small feed"])
    bg_mask = f_lower.isin(["bg","big","b","big feed"])
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

# --- vectorized "open at/before" for many arrival timestamps (fast) ---
def _open_at_or_before_many(prepped_df, when_series):
    if prepped_df is None or len(prepped_df) == 0 or when_series is None or len(when_series) == 0:
        return np.array([np.nan] * len(when_series)) if hasattr(when_series, "__len__") else np.array([])
    times = prepped_df["time_dt"].to_numpy()
    vals  = prepped_df["Open"].to_numpy()
    w = safe_to_datetime(when_series).to_numpy()
    idx = np.searchsorted(times, w, side="right") - 1
    idx[idx < 0] = -1
    out = np.where(idx == -1, np.nan, vals[idx])
    return out

# --- compute Day like [0], [-1], [+1] using day-start anchors ---
def _compute_day_index_series(arrival_series, report_time, day_start_hour: int):
    rpt = safe_to_datetime(report_time)
    if not isinstance(rpt, pd.Timestamp):
        return pd.Series(["[0]"] * len(arrival_series), index=arrival_series.index)
    rpt_anchor = day_start_anchor(rpt, day_start_hour)
    arr = safe_to_datetime(arrival_series)
    arr_anchor = arr.apply(lambda x: day_start_anchor(x, day_start_hour) if isinstance(x, pd.Timestamp) else pd.NaT)
    d = (arr_anchor - rpt_anchor).dt.days.fillna(0).astype(int)
    return d.map(lambda k: "[0]" if k == 0 else (f"[{k}]" if k < 0 else f"[+{k}]"))

# --- per-row Input @ Arrival + all 3 Diff columns + ddd/Day ---
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
    sm_mask = f_lower.isin(["small","sm","s","small feed"])
    bg_mask = f_lower.isin(["big","bg","b","big feed"])
    if sm_mask.any():
        result_df.loc[sm_mask, "Input @ Arrival"] = _open_at_or_before_many(sm_p, result_df.loc[sm_mask, "Arrival"])
    if bg_mask.any():
        result_df.loc[bg_mask, "Input @ Arrival"] = _open_at_or_before_many(bg_p, result_df.loc[bg_mask, "Arrival"])
    if "Output" in result_df.columns:
        result_df["Diff @ 18:00"]  = result_df["Output"] - pd.to_numeric(result_df["Input @ 18:00"], errors="coerce")
        result_df["Diff @ Arrival"] = result_df["Output"] - pd.to_numeric(result_df["Input @ Arrival"], errors="coerce")
        result_df["Diff @ Report"] = result_df["Output"] - pd.to_numeric(result_df["Input @ Report"], errors="coerce")
    result_df["ddd"] = result_df["Arrival"].dt.strftime("%a")
    result_df["Day"] = _compute_day_index_series(result_df["Arrival"], report_time, day_start_hour)
    return result_df

# --- enforce your exact Traveler Report column order ---
_DESIRED_COLS = [
    "Feed","ddd","Arrival","Day","Origin","M Name","M #","R #","Tag","Family",
    "Input @ 18:00","Diff @ 18:00","Input @ Arrival","Diff @ Arrival",
    "Input @ Report","Diff @ Report","Output"
]
def _finalize_columns_order(df):
    for c in _DESIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[_DESIRED_COLS]


# --- anchor helpers ---
def day_start_anchor(report_dt: datetime, start_hour: int) -> datetime:
    a = report_dt.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    return a if report_dt >= a else (a - timedelta(days=1))

def _sunday_start(ts: datetime, start_hour: int) -> datetime:
    anchor = _day_start_anchor(ts, start_hour)
    # Mon=0..Sun=6; distance back to Sunday:
    days_back = (anchor.weekday() + 1) % 7
    return (anchor - timedelta(days=days_back)).replace(hour=start_hour, minute=0, second=0, microsecond=0)

def _month_start(ts: datetime, start_hour: int) -> datetime:
    anchor = _day_start_anchor(ts, start_hour)
    return anchor.replace(day=1, hour=start_hour, minute=0, second=0, microsecond=0)

def _shift_months(dt0: datetime, months_back: int, start_hour: int) -> datetime:
    y, m = dt0.year, dt0.month
    m -= months_back
    while m <= 0:
        m += 12
        y -= 1
    return datetime(y, m, 1, start_hour, 0, 0)

def _special_cols(base: str, lag: int):
    """
    Build the *actual* column names that exist in your CSVs.
    Examples:
      base='WASP-12b', lag=0 -> 'WASP-12b H', 'WASP-12b L', 'WASP-12b C'
      base='WASP-12b', lag=1 -> 'WASP-12b H[1]', 'WASP-12b L[1]', 'WASP-12b C[1]'
    """
    suffix = '' if lag == 0 else f'[{lag}]'
    return (f'{base} H{suffix}', f'{base} L{suffix}', f'{base} C{suffix}')

def _current_hlc_from_cols(df: pd.DataFrame, report_time, h_col: str, l_col: str, c_col: str):
    """
    Return last non-null H/L/C at or before report_time from the given *explicit* columns.
    """
    if not all(col in df.columns for col in (h_col, l_col, c_col)):
        return None
    tmp = _ensure_time_dt(df)
    rpt = pd.to_datetime(report_time) if isinstance(report_time, str) else report_time
    cand = tmp.loc[tmp['time_dt'] <= rpt, [h_col, l_col, c_col, 'time_dt']].dropna()
    if cand.empty:
        return None
    row = cand.sort_values('time_dt').iloc[-1]
    H, L, C = float(row[h_col]), float(row[l_col]), float(row[c_col])
    spread = H - L
    if spread == 0:
        return None
    return {
        'H': H, 'L': L, 'C': C,
        'avg': (H + L + C) / 3.0,
        'spread': spread,
        'asof_time': row['time_dt'],  # data timestamp (not Arrival)
    }

def _wasp_entries_for_feed(df: pd.DataFrame, report_time, day_start_hour: int):
    """
    Build 3 entries for WASP-12b using correct columns and Arrival anchors.
    """
    rpt = pd.to_datetime(report_time) if isinstance(report_time, str) else report_time
    out = []
    for lag in (0, 1, 2):
        h, l, c = _special_cols('WASP-12b', lag)
        cur = _current_hlc_from_cols(df, rpt, h, l, c)
        if not cur:
            continue
        arrival = _sunday_start(rpt, day_start_hour) - timedelta(weeks=lag)
        cur['datetime'] = arrival
        cur['origin'] = 'Wasp-12b' if lag == 0 else f'Wasp-12b[-{lag}]'
        out.append(cur)
    return out

def _macedonia_entries_for_feed(df: pd.DataFrame, report_time, day_start_hour: int):
    """
    Build 3 entries for Macedonia using correct columns and Arrival anchors.
    """
    rpt = pd.to_datetime(report_time) if isinstance(report_time, str) else report_time
    out = []
    m0 = _month_start(rpt, day_start_hour)
    for lag in (0, 1, 2):
        h, l, c = _special_cols('Macedonia', lag)
        cur = _current_hlc_from_cols(df, rpt, h, l, c)
        if not cur:
            continue
        arrival = _shift_months(m0, lag, day_start_hour)
        cur['datetime'] = arrival
        cur['origin'] = 'Macedonia' if lag == 0 else f'Macedonia[-{lag}]'
        out.append(cur)
    return out



def find_new_data_changes(small_df, report_time, origin_name, scope_days=20):
    """
    Find the first time new data appears for an origin by detecting changes in H/L/C values.
    Starts from the bottom (most recent) and works backwards to find when data changed.
    
    Args:
        small_df: Small CSV dataframe
        report_time: Target datetime for analysis
        origin_name: Name of origin (e.g., 'Venus', 'Mercury', 'Fiji')
        scope_days: Maximum days to look back
        
    Returns:
        List of dictionaries with H, L, C values and datetime for each data change
    """
    try:
        # Convert report_time to pandas datetime if needed
        if isinstance(report_time, str):
            report_time = pd.to_datetime(report_time)
        
        # Look for columns ending with H, L, C for this origin
        h_col = f"{origin_name} H"
        l_col = f"{origin_name} L" 
        c_col = f"{origin_name} C"
        
        # Check if these columns exist
        if not all(col in small_df.columns for col in [h_col, l_col, c_col]):
            return []
        
        # Simple timezone stripping - no conversion
        small_df_copy = small_df.copy()
        # Strip timezone from ISO format: 2025-08-06T18:45:00-04:00 → 2025-08-06 18:45:00
        if 'time' in small_df_copy.columns:
            time_strings = small_df_copy['time'].astype(str)
            # Remove timezone offset (everything after + or - in time)
            clean_times = time_strings.str.replace(r'[+-]\d{2}:?\d{2}$', '', regex=True)
            # Replace T with space for standard datetime format
            clean_times = clean_times.str.replace('T', ' ')
            small_df_copy['time_dt'] = pd.to_datetime(clean_times, errors='coerce')
        else:
            small_df_copy['time_dt'] = pd.to_datetime(small_df_copy.iloc[:, 0], errors='coerce')
        
        # Ensure report_time is timezone-naive datetime
        if isinstance(report_time, str):
            # Strip timezone from string format
            clean_report_time = report_time.replace('T', ' ')
            clean_report_time = pd.Series([clean_report_time]).str.replace(r'[+-]\d{2}:?\d{2}$', '', regex=True).iloc[0]
            report_time = pd.to_datetime(clean_report_time)
        elif hasattr(report_time, 'tz') and report_time.tz is not None:
            report_time = report_time.replace(tzinfo=None)
        
        # Get data within scope and at or before report time
        scope_start = report_time - timedelta(days=scope_days)
        scoped_df = small_df_copy[
            (small_df_copy['time_dt'] >= scope_start) & 
            (small_df_copy['time_dt'] <= report_time)
        ].copy()
        
        if scoped_df.empty:
            return []
        
        # Sort by time ASCENDING (oldest first) to analyze changes properly
        scoped_df = scoped_df.sort_values('time_dt', ascending=True).reset_index(drop=True)
        
        new_data_entries = []
        previous_h, previous_l, previous_c = None, None, None
        
        # Go through rows chronologically to detect when data changes
        for idx, row in scoped_df.iterrows():
            h_val = row[h_col]
            l_val = row[l_col]
            c_val = row[c_col]
            
            # Skip if any values are null
            if pd.isna(h_val) or pd.isna(l_val) or pd.isna(c_val):
                continue
            
            current_h, current_l, current_c = float(h_val), float(l_val), float(c_val)
            
            # Check if this is new data (different from previous row)
            if (previous_h is None or 
                current_h != previous_h or 
                current_l != previous_l or 
                current_c != previous_c):
                
                # This is new data!
                new_data_entries.append({
                    'H': current_h,
                    'L': current_l,
                    'C': current_c,
                    'datetime': row['time_dt'],
                    'origin': origin_name
                })
                
                # Update previous values
                previous_h, previous_l, previous_c = current_h, current_l, current_c
        
        return new_data_entries
        
    except Exception as e:
        st.error(f"Error finding new data changes for {origin_name}: {e}")
        return []

def find_most_current_data(small_df, report_time, origin_name, scope_days=20):
    """
    Find the most current data for an origin at report time.
    
    Args:
        small_df: Small CSV dataframe
        report_time: Target datetime for analysis
        origin_name: Name of origin (e.g., 'Venus', 'Mercury', 'Fiji')
        scope_days: Maximum days to look back
        
    Returns:
        Dictionary with H, L, C values and datetime, or None if not found
    """
    try:
        # Convert report_time to pandas datetime if needed
        if isinstance(report_time, str):
            report_time = pd.to_datetime(report_time)
        
        # Look for columns ending with H, L, C for this origin
        h_col = f"{origin_name} H"
        l_col = f"{origin_name} L" 
        c_col = f"{origin_name} C"
        
        # Check if these columns exist
        if not all(col in small_df.columns for col in [h_col, l_col, c_col]):
            return None
        
        # Simple timezone stripping - no conversion
        small_df_copy = small_df.copy()
        # Strip timezone from ISO format: 2025-08-06T18:45:00-04:00 → 2025-08-06 18:45:00
        if 'time' in small_df_copy.columns:
            time_strings = small_df_copy['time'].astype(str)
            # Remove timezone offset (everything after + or - in time)
            clean_times = time_strings.str.replace(r'[+-]\d{2}:?\d{2}$', '', regex=True)
            # Replace T with space for standard datetime format
            clean_times = clean_times.str.replace('T', ' ')
            small_df_copy['time_dt'] = pd.to_datetime(clean_times, errors='coerce')
        else:
            small_df_copy['time_dt'] = pd.to_datetime(small_df_copy.iloc[:, 0], errors='coerce')
        
        # Ensure report_time is timezone-naive datetime
        if isinstance(report_time, str):
            # Strip timezone from string format
            clean_report_time = report_time.replace('T', ' ')
            clean_report_time = pd.Series([clean_report_time]).str.replace(r'[+-]\d{2}:?\d{2}$', '', regex=True).iloc[0]
            report_time = pd.to_datetime(clean_report_time)
        elif hasattr(report_time, 'tz') and report_time.tz is not None:
            report_time = report_time.replace(tzinfo=None)
        
        # Priority 1: Look for data from the same day as report_time
        report_date = report_time.date()
        same_day_df = small_df_copy[small_df_copy['time_dt'].dt.date == report_date].copy()
        
        if not same_day_df.empty:
            # Sort by time descending to get most recent first
            same_day_df = same_day_df.sort_values('time_dt', ascending=False)
            
            # Find data at or before report_time on the same day
            valid_same_day = same_day_df[same_day_df['time_dt'] <= report_time]
            
            if not valid_same_day.empty:
                for _, row in valid_same_day.iterrows():
                    h_val = row[h_col]
                    l_val = row[l_col]
                    c_val = row[c_col]
                    
                    if not (pd.isna(h_val) or pd.isna(l_val) or pd.isna(c_val)):
                        return {
                            'H': float(h_val),
                            'L': float(l_val),
                            'C': float(c_val),
                            'datetime': row['time_dt'],
                            'origin': origin_name
                        }
        
        # Priority 2: If no same-day data, look within scope_days (as fallback)
        scope_start = report_time - timedelta(days=scope_days)
        scoped_df = small_df_copy[
            (small_df_copy['time_dt'] >= scope_start) & 
            (small_df_copy['time_dt'] <= report_time)
        ].copy()
        
        if scoped_df.empty:
            return None
        
        # Sort by time descending to start with most recent
        scoped_df = scoped_df.sort_values('time_dt', ascending=False)
        
        # Find the most current data by checking for different values
        for i, row in scoped_df.iterrows():
            h_val = row[h_col]
            l_val = row[l_col]
            c_val = row[c_col]
            
            # Skip if any values are null
            if pd.isna(h_val) or pd.isna(l_val) or pd.isna(c_val):
                continue
            
            # Check if this row has different data from previous rows
            is_current = True
            if i > 0:  # Not the first row
                # Look at next newer rows to see if data changed
                newer_rows = scoped_df[scoped_df['time_dt'] > row['time_dt']]
                if not newer_rows.empty:
                    for _, newer_row in newer_rows.iterrows():
                        if (newer_row[h_col] == h_val and 
                            newer_row[l_col] == l_val and 
                            newer_row[c_col] == c_val):
                            is_current = False
                            break
            
            if is_current:
                return {
                    'H': float(h_val),
                    'L': float(l_val),
                    'C': float(c_val),
                    'datetime': row['time_dt'],
                    'origin': origin_name
                }
        
        # If no "current" data found, use the most recent row with valid data
        for _, row in scoped_df.iterrows():
            h_val = row[h_col]
            l_val = row[l_col]
            c_val = row[c_col]
            
            if not (pd.isna(h_val) or pd.isna(l_val) or pd.isna(c_val)):
                return {
                    'H': float(h_val),
                    'L': float(l_val),
                    'C': float(c_val),
                    'datetime': row['time_dt'],
                    'origin': origin_name
                }
        
        return None
        
    except Exception as e:
        st.error(f"Error finding current data for {origin_name}: {e}")
        return None

def calculate_raw_m_values(hlc_data, range_low, range_high):
    """
    Calculate raw M values for a price range.
    
    Args:
        hlc_data: Dictionary with H, L, C values
        range_low: Lower bound of price range
        range_high: Upper bound of price range
        
    Returns:
        Dictionary with raw_m_low and raw_m_high
    """
    try:
        H = hlc_data['H']
        L = hlc_data['L']
        C = hlc_data['C']
        
        # Calculate average
        avg = (H + L + C) / 3
        
        # Calculate spread
        spread = H - L
        
        if spread == 0:
            return None  # Cannot calculate with zero spread
        
        # Calculate raw M values
        raw_m_low = (range_low - avg) / spread
        raw_m_high = (range_high - avg) / spread
        
        return {
            'raw_m_low': raw_m_low,
            'raw_m_high': raw_m_high,
            'avg': avg,
            'spread': spread
        }
        
    except Exception as e:
        st.error(f"Error calculating raw M values: {e}")
        return None

def find_valid_m_values(measurement_df, raw_m_low, raw_m_high, hlc_data, range_low, range_high, is_high_range=False, data_source="Unknown", report_time=None):
    """
    Find valid M values from measurement file within the raw M range.
    
    Args:
        measurement_df: Measurement dataframe
        raw_m_low: Lower raw M boundary
        raw_m_high: Upper raw M boundary  
        hlc_data: H/L/C data for output calculation
        range_low: Lower price boundary
        range_high: Upper price boundary
        is_high_range: True if this is a High range, False for Low range
        
    Returns:
        Dictionary with 'valid_entries' and 'valid_m_list'
    """
    try:
        valid_entries = []
        valid_m_values = []  # Track valid M values
        rejected_m_values = []  # Track rejected M values with reasons
        
        # Get M values from measurement file - use flexible column detection
        m_value_col = None
        for col in ['M value', 'M Value', 'M_Value', 'M_value', 'm value', 'm_value']:
            if col in measurement_df.columns:
                m_value_col = col
                break
        
        if m_value_col is None:
            return {
                'valid_entries': valid_entries,
                'valid_m_list': valid_m_values
            }
        
        m_values = measurement_df[m_value_col].dropna().unique()
        
        # Filter M values within raw M range
        debug_count = 0
        zone_summary = {"0 to 6": 0, "6 to 12": 0, "12 to 18": 0, "18 to 24": 0, "Out of Range": 0}
        for m_val in m_values:
            try:
                m_float = float(m_val)
                debug_count += 1
                
                # Check if M value is within range
                if raw_m_low <= m_float <= raw_m_high:
                    # Calculate output for this M value
                    output = hlc_data['avg'] + m_float * hlc_data['spread']
                    
                    
                    # This M value is valid (within raw M range)
                    valid_m_values.append(m_float)
                    # Get all rows with this M value (match by detected column, not 'M #')
                    matching_rows = measurement_df[measurement_df[m_value_col] == m_val]
                    

                    
                    for _, row in matching_rows.iterrows():
                        # Determine zone based on output value
                        zone_value = ""
                        if range_low <= output <= range_high:
                            if is_high_range:
                                # High range: zones measured from the top (range_high)
                                distance_from_top = range_high - output
                                if distance_from_top <= 6:
                                    zone_value = "0 to 6"
                                elif distance_from_top <= 12:
                                    zone_value = "6 to 12"
                                elif distance_from_top <= 18:
                                    zone_value = "12 to 18"
                                else:
                                    zone_value = "18 to 24"
                            else:
                                # Low range: zones measured from the bottom (range_low)
                                distance_from_bottom = output - range_low
                                if distance_from_bottom <= 6:
                                    zone_value = "0 to 6"
                                elif distance_from_bottom <= 12:
                                    zone_value = "6 to 12"
                                elif distance_from_bottom <= 18:
                                    zone_value = "12 to 18"
                                else:
                                    zone_value = "18 to 24"
                        else:
                            zone_value = "Out of Range"
                        
                        # Track zone distribution
                        zone_summary[zone_value] += 1
                        

                        
                        # Format arrival time from H/L/C data
                        try:
                            arrival_dt = hlc_data['datetime']
                            day_abbrev = arrival_dt.strftime('%a')  # Mon, Tue, Wed, etc.
                            arrival_excel = arrival_dt.strftime('%Y-%m-%d %H:%M')  # Excel-friendly format
                            
                            # Calculate day index using proper get_day_index function
                            try:
                                from a_helpers import get_day_index
                                if isinstance(arrival_dt, str):
                                    arrival_dt = pd.to_datetime(arrival_dt)
                                # Use the proper get_day_index function with start_hour (defaulting to 18)
                                day_index = get_day_index(arrival_dt, report_time, 18)

                            except Exception as e:

                                day_index = "[0]"
                        except:
                            day_abbrev = ""
                            arrival_excel = str(hlc_data['datetime'])
                            day_index = "[0]"
                        
                        # Determine feed type from data source name
                        feed_type = "Small" if data_source == "Small CSV" else "Big"
                        
                        # Calculate input values (basic implementation for now)
                        # These would normally come from small_df at specific times
                        input_18 = hlc_data.get('H', 0)  # Placeholder
                        input_arrival = hlc_data.get('C', 0)  # Placeholder  
                        input_report = hlc_data.get('L', 0)  # Placeholder
                        
                        valid_entries.append({
                            'Feed': feed_type,
                            'ddd': day_abbrev,
                            'Arrival': arrival_excel,

                            'Day': day_index,
                            'Origin': hlc_data['origin'],
                            'M Name': row.get('M Name', row.get('m name', f"M{m_val}")),
                            'M #': row.get('M #', row.get('m #', m_val)),
                            'M Value': m_val,
                            'R #': row.get('R #', row.get('r #', '')),
                            'Tag': row.get('Tag', row.get('tag', '')),
                            'Family': row.get('Family', row.get('family', '')),
                            'Input @ 18:00': input_18,
                            'Diff @ 18:00': output - input_18,
                            'Input @ Arrival': input_arrival,
                            'Diff @ Arrival': output - input_arrival,
                            'Input @ Report': input_report,
                            'Diff @ Report': output - input_report,
                            'Output': output,
                            'Range': f"{range_low:.1f}-{range_high:.1f}",
                            'Zone': zone_value
                        })
                else:
                    # M value outside raw range
                    rejected_m_values.append({
                        'm_value': m_float,
                        'reason': 'outside_raw_range',
                        'distance_from_range': min(abs(m_float - raw_m_low), abs(m_float - raw_m_high))
                    })
                
            except (ValueError, TypeError):
                continue  # Skip invalid M values
        

        
        return {
            'valid_entries': valid_entries,
            'valid_m_list': valid_m_values
        }
        
    except Exception as e:
        st.error(f"Error finding valid M values: {e}")
        return {
            'valid_entries': [],
            'valid_m_list': []
        }

def process_custom_ranges_advanced(
    measurement_df,
    small_df,
    report_time,
    custom_ranges,
    scope_days: int = 20,
    big_df=None,
    run_model_g: bool = False,
    day_start_hour: int = 18,
):
    """
    Process custom ranges with special handling for WASP-12b and Macedonia:
      - WASP-12b H/L/C        -> 'Wasp-12b'       at most-recent Sunday @ start
      - WASP-12b H[1]/L[1]/C[1] -> 'Wasp-12b[-1]' at 1 Sunday ago @ start
      - WASP-12b H[2]/L[2]/C[2] -> 'Wasp-12b[-2]' at 2 Sundays ago @ start
      - Macedonia H/L/C       -> 'Macedonia'      at month 1st @ start
      - Macedonia H[1]/L[1]/C[1] -> 'Macedonia[-1]' at 1 month ago @ start
      - Macedonia H[2]/L[2]/C[2] -> 'Macedonia[-2]' at 2 months ago @ start

    Other origins continue to use find_new_data_changes(...).
    """
    import pandas as pd

    if measurement_df is None or measurement_df.empty or small_df is None:
        return pd.DataFrame()

    rpt_dt = safe_to_datetime(report_time)
    if not isinstance(rpt_dt, pd.Timestamp):
        return pd.DataFrame()

    # Gather "normal" origins from "... H" columns; exclude special bases
    def _list_origins(df):
        return sorted({c[:-2] for c in df.columns if c.endswith(" H")})

    origins_sm = _list_origins(small_df)
    origins_sm = [o for o in origins_sm if o not in ("WASP-12b", "Macedonia")]

    origins_bg = []
    if isinstance(big_df, pd.DataFrame) and not big_df.empty:
        origins_bg = _list_origins(big_df)
        origins_bg = [o for o in origins_bg if o not in ("WASP-12b", "Macedonia")]

    results = []

    def _append_validation(val):
        # Accept either a DataFrame or the old dict {'valid_entries': ..., 'valid_m_list': ...}
        if val is None:
            return
        if isinstance(val, pd.DataFrame):
            if not val.empty:
                results.append(val)
        elif isinstance(val, dict):
            rows = val.get("valid_entries", [])
            if rows:
                results.append(pd.DataFrame(rows))

    # Helper to push for one HLC snapshot across all custom ranges
    def _push_for_hlc(hlc, feed_tag):
        for (range_low, range_high, is_high_range) in custom_ranges:
            m_bounds = calculate_raw_m_values(hlc, range_low, range_high)
            if not m_bounds:
                continue
            enhanced = dict(hlc)
            enhanced.update(m_bounds)
            val = find_valid_m_values(
                measurement_df,
                m_bounds["raw_m_low"],
                m_bounds["raw_m_high"],
                enhanced,
                range_low,
                range_high,
                is_high_range=is_high_range,
                data_source=feed_tag,
                report_time=rpt_dt,
            )
            _append_validation(val)

    # ---- Special: WASP-12b (small feed) ----
    for hlc in _wasp_entries_for_feed(small_df, rpt_dt, day_start_hour):
        _push_for_hlc(hlc, "sm")

    # ---- Special: Macedonia (small feed) ----
    for hlc in _macedonia_entries_for_feed(small_df, rpt_dt, day_start_hour):
        _push_for_hlc(hlc, "sm")

    # ---- Normal origins (small feed) ----
    for origin in origins_sm:
        changes = find_new_data_changes(small_df, rpt_dt, origin, scope_days=scope_days)
        if not changes:
            continue
        for hlc in changes:
            _push_for_hlc(hlc, "sm")

    # ---- Big feed, if provided ----
    if isinstance(big_df, pd.DataFrame) and not big_df.empty:
        # Special WASP-12b (big)
        for hlc in _wasp_entries_for_feed(big_df, rpt_dt, day_start_hour):
            _push_for_hlc(hlc, "bg")
        # Special Macedonia (big)
        for hlc in _macedonia_entries_for_feed(big_df, rpt_dt, day_start_hour):
            _push_for_hlc(hlc, "bg")
        # Normal origins (big)
        for origin in origins_bg:
            changes = find_new_data_changes(big_df, rpt_dt, origin, scope_days=scope_days)
            if not changes:
                continue
            for hlc in changes:
                _push_for_hlc(hlc, "bg")

    out = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    # Paste constants (Input @ 18:00 / @ Report), then per-row Input @ Arrival, diffs, and final column order
    out = _apply_inputs_columns_and_debug(out, small_df, big_df, rpt_dt, day_start_hour, show_debug=True)
    out = _apply_input_at_arrival_and_diffs(out, small_df, big_df, rpt_dt, day_start_hour)
    out = _finalize_columns_order(out)
    return out


def apply_custom_ranges_advanced(df, small_df, report_time, high1, high2, low1, low2, use_high1, use_high2, use_low1, use_low2, big_df=None, run_model_g=False):
    """
    Apply advanced custom ranges to dataframe.
    
    Returns:
        Filtered dataframe with Range and Zone columns
    """
    # DEBUG: Confirm new module is being used
    st.error("🔥 MODULE RELOADED - NEW VERSION ACTIVE 🔥")
    st.info(f"🧮 Advanced Custom Range Processing Started - {len(df)} measurements to process")
    # Prepare custom ranges configuration
    custom_ranges = {}
    
    if use_high1 and high1 > 0:
        custom_ranges['High 1'] = {'enabled': True, 'value': high1}
    if use_high2 and high2 > 0:
        custom_ranges['High 2'] = {'enabled': True, 'value': high2}
    if use_low1 and low1 > 0:
        custom_ranges['Low 1'] = {'enabled': True, 'value': low1}
    if use_low2 and low2 > 0:
        custom_ranges['Low 2'] = {'enabled': True, 'value': low2}
    
    if not custom_ranges:
        return df
    
    # Process ranges using advanced method
    valid_entries = process_custom_ranges_advanced(df, small_df, report_time, custom_ranges, big_df=big_df, run_model_g=run_model_g)
    
    if not valid_entries:
        st.warning("No valid entries found using advanced custom range calculation")
        return pd.DataFrame()  # Return empty dataframe
    
    # Convert to dataframe
    filtered_df = pd.DataFrame(valid_entries)
    
    # Add Range and Zone columns
    def get_range_name(output_val):
        for range_name, range_config in custom_ranges.items():
            range_value = range_config['value']
            if range_name.startswith('High'):
                range_low = range_value - 24
                range_high = range_value
            else:
                range_low = range_value
                range_high = range_value + 24
            
            if range_low <= output_val <= range_high:
                return range_name
        return 'Other'
    
    def get_zone(output_val):
        for range_name, range_config in custom_ranges.items():
            range_value = range_config['value']
            if range_name.startswith('High'):
                range_low = range_value - 24
                range_high = range_value
                if range_low <= output_val <= range_high:
                    distance = range_high - output_val
                    if distance <= 6:
                        return "0-6"
                    elif distance <= 12:
                        return "6-12"
                    elif distance <= 18:
                        return "12-18"
                    else:
                        return "18-24"
            else:
                range_low = range_value
                range_high = range_value + 24
                if range_low <= output_val <= range_high:
                    distance = output_val - range_low
                    if distance <= 6:
                        return "0-6"
                    elif distance <= 12:
                        return "6-12"
                    elif distance <= 18:
                        return "12-18"
                    else:
                        return "18-24"
        return ""
    
    # Don't overwrite the Zone column - it's already correctly calculated in find_valid_m_values
    filtered_df['Range'] = filtered_df['Output'].apply(get_range_name)
    # Keep the existing Zone column from find_valid_m_values function
    
    return filtered_df

def process_full_range_advanced(
    measurement_df,
    small_df,
    report_time,
    center,
    window_radius,
    scope_days: int = 20,
    big_df=None,
    run_model_g: bool = False,
    day_start_hour: int = 18,
):
    """
    Full-range processor centered on `center` with +/- `window_radius`,
    with special handling for WASP-12b and Macedonia as described in
    process_custom_ranges_advanced docstring.
    """
    import pandas as pd

    if measurement_df is None or measurement_df.empty or small_df is None:
        return pd.DataFrame()

    rpt_dt = safe_to_datetime(report_time)
    if not isinstance(rpt_dt, pd.Timestamp):
        return pd.DataFrame()

    range_low = center - window_radius
    range_high = center + window_radius

    # Gather "normal" origins from "... H" columns; exclude special bases
    def _list_origins(df):
        return sorted({c[:-2] for c in df.columns if c.endswith(" H")})

    origins_sm = _list_origins(small_df)
    origins_sm = [o for o in origins_sm if o not in ("WASP-12b", "Macedonia")]

    origins_bg = []
    if isinstance(big_df, pd.DataFrame) and not big_df.empty:
        origins_bg = _list_origins(big_df)
        origins_bg = [o for o in origins_bg if o not in ("WASP-12b", "Macedonia")]

    results = []

    def _append_validation(val):
        if val is None:
            return
        if isinstance(val, pd.DataFrame):
            if not val.empty:
                results.append(val)
        elif isinstance(val, dict):
            rows = val.get("valid_entries", [])
            if rows:
                results.append(pd.DataFrame(rows))

    def _push_for_hlc(hlc, feed_tag):
        m_bounds = calculate_raw_m_values(hlc, range_low, range_high)
        if not m_bounds:
            return
        enhanced = dict(hlc)
        enhanced.update(m_bounds)
        val = find_valid_m_values(
            measurement_df,
            m_bounds["raw_m_low"],
            m_bounds["raw_m_high"],
            enhanced,
            range_low,
            range_high,
            is_high_range=(center >= 0),
            data_source=feed_tag,
            report_time=rpt_dt,
        )
        _append_validation(val)

    # ---- Special: WASP-12b (small feed) ----
    for hlc in _wasp_entries_for_feed(small_df, rpt_dt, day_start_hour):
        _push_for_hlc(hlc, "sm")

    # ---- Special: Macedonia (small feed) ----
    for hlc in _macedonia_entries_for_feed(small_df, rpt_dt, day_start_hour):
        _push_for_hlc(hlc, "sm")

    # ---- Normal origins (small feed) ----
    for origin in origins_sm:
        changes = find_new_data_changes(small_df, rpt_dt, origin, scope_days=scope_days)
        if not changes:
            continue
        for hlc in changes:
            _push_for_hlc(hlc, "sm")

    # ---- Big feed, if provided ----
    if isinstance(big_df, pd.DataFrame) and not big_df.empty:
        # Special WASP-12b (big)
        for hlc in _wasp_entries_for_feed(big_df, rpt_dt, day_start_hour):
            _push_for_hlc(hlc, "bg")
        # Special Macedonia (big)
        for hlc in _macedonia_entries_for_feed(big_df, rpt_dt, day_start_hour):
            _push_for_hlc(hlc, "bg")
        # Normal origins (big)
        for origin in origins_bg:
            changes = find_new_data_changes(big_df, rpt_dt, origin, scope_days=scope_days)
            if not changes:
                continue
            for hlc in changes:
                _push_for_hlc(hlc, "bg")

    out = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    out = _apply_inputs_columns_and_debug(out, small_df, big_df, rpt_dt, day_start_hour, show_debug=True)
    out = _apply_input_at_arrival_and_diffs(out, small_df, big_df, rpt_dt, day_start_hour)
    out = _finalize_columns_order(out)
    return out



def apply_full_range_advanced(
    df: pd.DataFrame,
    small_df: pd.DataFrame,
    report_time: dt.datetime,
    window_radius: float,
    day_start_hour: int = 18,
    input_value_at_start: float | None = None,
    big_df: pd.DataFrame | None = None,
    run_model_g: bool = False,
):
    """
    Apply the advanced Full Range flow (mirrors apply_custom_ranges_advanced):
      - Determine center (prefer input_value_at_start; else derive from small_df).
      - Call process_full_range_advanced to collect valid entries via raw-M windows.
      - Return a dataframe; drop 'Range'/'Zone' columns if present.
    """
    # 1) Center
    center = None
    if input_value_at_start is not None and not pd.isna(input_value_at_start):
        center = float(input_value_at_start)
    else:
        # Derive from small_df: prefer Open @ day start, else last Open/close <= report_time
        try:
            sdf = small_df.copy()
            if 'time' in sdf.columns:
                # Use your existing clean_timestamp
                sdf['time'] = sdf['time'].apply(clean_timestamp)
                if report_time is not None:
                    sdf = sdf[sdf['time'] <= report_time]

            if not sdf.empty:
                # compute day start
                base = dt.datetime(report_time.year, report_time.month, report_time.day, day_start_hour, 0, 0)
                if report_time < base:
                    base = base - dt.timedelta(days=1)

                center_row = sdf[sdf['time'] == base]
                if not center_row.empty:
                    center_row = center_row.iloc[-1]
                else:
                    center_row = sdf.iloc[-1]

                for cand in ('Open', 'open', 'close'):
                    if cand in center_row.index:
                        center = pd.to_numeric(pd.Series(center_row[cand]), errors='coerce').iloc[0]
                        break
        except Exception:
            center = None

    if center is None or pd.isna(center):
        st.error("Full Range (Advanced): could not determine center. Provide input @ day start or ensure small feed has time/Open/close.")
        return pd.DataFrame()

    # 2) Process
    valid_entries = process_full_range_advanced(
        measurement_df=df,
        small_df=small_df,
        report_time=report_time,
        center=center,
        window_radius=window_radius,
        scope_days=20,
        big_df=big_df,
        run_model_g=run_model_g
    )

    if not valid_entries:
        st.warning("Full Range (Advanced): no valid entries found.")
        return pd.DataFrame()

    # 3) Convert to DF and drop Range/Zone if present
    out_df = pd.DataFrame(valid_entries)
    out_df = out_df.drop(columns=['Range', 'Zone'], errors='ignore')

    # Nice to have: order consistently
    preferred_cols = [
        'Feed','ddd','Arrival','Day','Origin',
        'M Name','M #','M Value','R #','Tag','Family',
        'Input @ 18:00','Diff @ 18:00','Input @ Arrival','Diff @ Arrival',
        'Input @ Report','Diff @ Report','Output'
    ]
    ordered = [c for c in preferred_cols if c in out_df.columns]
    remaining = [c for c in out_df.columns if c not in ordered]
    out_df = out_df[ordered + remaining]

    st.success(f"✅ Full Range (Advanced): {len(out_df)} entries")
    return out_df
