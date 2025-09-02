
"""
Custom Range Calculator (revised)
- Fixes ambiguous Series truth checks in datetime handling
- Always includes WASP-12b (weekly) and Macedonia (monthly) with correct synthetic Arrival timestamps
- Computes 'Input @ 18:00' and 'Input @ Report' ONCE per feed, then pastes constants into the result
- Adds a compact debug line for the four input values (small/big at day-start and report)
- Keeps existing function names/signatures used by the app. `apply_custom_ranges_advanced` keeps its original signature;
  `apply_full_range_advanced` keeps `day_start_hour` (default 18). If the app passes 17:00 in full range, it will honor it.
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# ---------------------------------------------------------------------
# Datetime helpers (robust & vectorized)
# ---------------------------------------------------------------------

def safe_to_datetime(x, errors='coerce'):
    """ Safely convert a Series or scalar to pandas datetime (tz-naive)."""
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
        return pd.NaT if not isinstance(x, pd.Series) else pd.Series([pd.NaT]*len(x))

def ensure_timezone_naive(x):
    """ Return tz-naive datetime/Series. Handles Series first, scalars second. """
    try:
        if isinstance(x, pd.Series):
            if not pd.api.types.is_datetime64_any_dtype(x):
                x = safe_to_datetime(x)
            if pd.api.types.is_datetime64tz_dtype(x):
                return x.dt.tz_localize(None)
            return x
        # scalar
        if x is None:
            return None
        if isinstance(x, pd.Timestamp):
            return x.tz_localize(None) if x.tz is not None else x
        ts = safe_to_datetime(x)
        if isinstance(ts, pd.Timestamp):
            return ts.tz_localize(None) if ts.tz is not None else ts
        return ts
    except Exception as e:
        # Quiet fallback (avoid spamming warnings)
        return x

def _day_start_anchor(report_dt: datetime, start_hour: int) -> datetime:
    a = report_dt.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    return a if report_dt >= a else (a - timedelta(days=1))

def _sunday_start(ts: datetime, start_hour: int) -> datetime:
    """ Most-recent Sunday at selected start hour (17 or 18). """
    anchor = _day_start_anchor(ts, start_hour)
    # Monday=0..Sunday=6 -> distance back to Sunday
    days_back = (anchor.weekday() + 1) % 7
    sunday = anchor - timedelta(days=days_back)
    return sunday.replace(hour=start_hour, minute=0, second=0, microsecond=0)

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

# ---------------------------------------------------------------------
# Feed 'Open' lookup (fast, single-pass) + debug + assignment
# ---------------------------------------------------------------------

def _to_naive_dt_series(series):
    s = safe_to_datetime(series)
    if isinstance(s, pd.Series) and pd.api.types.is_datetime64tz_dtype(s):
        s = s.dt.tz_localize(None)
    return s

def _prep_feed_df(feed_df):
    """ Prepare DataFrame with time_dt (tz-naive, sorted asc) and Open (float). """
    if feed_df is None or len(feed_df) == 0:
        return None
    if 'time' not in feed_df.columns or 'Open' not in feed_df.columns:
        return None
    out = feed_df[['time', 'Open']].copy()
    out['time_dt'] = _to_naive_dt_series(out['time'])
    out = out.dropna(subset=['time_dt'])
    out['Open'] = pd.to_numeric(out['Open'], errors='coerce')
    out = out.dropna(subset=['Open'])
    out = out.sort_values('time_dt', kind='mergesort')
    return out[['time_dt', 'Open']]

def _open_at_or_before_fast(prepped, when_dt):
    if prepped is None or len(prepped) == 0 or when_dt is None:
        return None
    if not isinstance(when_dt, datetime):
        when_dt = safe_to_datetime(when_dt)
        if isinstance(when_dt, pd.Series) or pd.isna(when_dt):
            return None
    idx = prepped['time_dt'].searchsorted(when_dt, side='right') - 1
    if idx < 0:
        return None
    return float(prepped['Open'].iloc[idx])

def _compute_feed_inputs(small_df, big_df, report_time, day_start_hour):
    rpt_dt = safe_to_datetime(report_time)
    if isinstance(rpt_dt, pd.Series) or pd.isna(rpt_dt):
        return (None, None, None, None)
    start_dt = _day_start_anchor(rpt_dt, day_start_hour)
    sm_p = _prep_feed_df(small_df) if small_df is not None else None
    bg_p = _prep_feed_df(big_df) if big_df is not None else None
    sm18 = _open_at_or_before_fast(sm_p, start_dt)
    smrp = _open_at_or_before_fast(sm_p, rpt_dt)
    bg18 = _open_at_or_before_fast(bg_p, start_dt)
    bgrp = _open_at_or_before_fast(bg_p, rpt_dt)
    return (sm18, smrp, bg18, bgrp)

def _apply_inputs_columns_and_debug(
    result_df, small_df, big_df, report_time, day_start_hour=18, show_debug=True
):
    """ Paste constant 'Input @ 18:00' and 'Input @ Report' values by feed. """
    if result_df is None or len(result_df) == 0:
        return result_df
    sm18, smrp, bg18, bgrp = _compute_feed_inputs(small_df, big_df, report_time, day_start_hour)

    if show_debug:
        try:
            st.info(
                f"🔎 INPUT DEBUG — Small: Open@{day_start_hour:02d}:00={sm18} | Open@report={smrp} | "
                f"Big: Open@{day_start_hour:02d}:00={bg18} | Open@report={bgrp}" 
            )
        except Exception:
            pass

    # feed mask
    feed = result_df.get('Feed') if 'Feed' in result_df.columns else result_df.get('data_source')
    if feed is None:
        return result_df
    f_lower = feed.astype(str).str.strip().str.lower()
    sm_mask = f_lower.isin(['sm','small','s','small feed'])
    bg_mask = f_lower.isin(['bg','big','b','big feed'])

    if 'Input @ 18:00' not in result_df.columns:
        result_df['Input @ 18:00'] = None
    if 'Input @ Report' not in result_df.columns:
        result_df['Input @ Report'] = None

    if sm18 is not None:
        result_df.loc[sm_mask, 'Input @ 18:00'] = sm18
    if smrp is not None:
        result_df.loc[sm_mask, 'Input @ Report'] = smrp
    if bg18 is not None:
        result_df.loc[bg_mask, 'Input @ 18:00'] = bg18
    if bgrp is not None:
        result_df.loc[bg_mask, 'Input @ Report'] = bgrp
    return result_df

# ---------------------------------------------------------------------
# H/L/C scanning utilities
# ---------------------------------------------------------------------

def _ensure_time_dt(df):
    out = df.copy()
    if 'time' in out.columns:
        s = out['time'].astype(str).str.replace(r'[+-]\d{2}:?\d{2}$', '', regex=True).str.replace('T',' ')
        out['time_dt'] = pd.to_datetime(s, errors='coerce')
    else:
        out['time_dt'] = pd.NaT
    return out

def find_new_data_changes(small_df, report_time, origin_name, scope_days=20):
    """ Detect rows where origin's H/L/C changed (first appearance of new data). """
    if small_df is None or origin_name is None:
        return []
    h_col, l_col, c_col = f"{origin_name} H", f"{origin_name} L", f"{origin_name} C"
    if not all(col in small_df.columns for col in [h_col, l_col, c_col]):
        return []

    df = _ensure_time_dt(small_df)
    df = df.dropna(subset=['time_dt']).sort_values('time_dt', ascending=False)
    rpt_dt = safe_to_datetime(report_time)
    if not isinstance(rpt_dt, pd.Timestamp):
        return []
    cutoff = rpt_dt - timedelta(days=scope_days)

    recent = df[df['time_dt'] >= cutoff]
    if recent.empty:
        return []

    changes = []
    last = None
    for _, row in recent.sort_values('time_dt').iterrows():
        cur = (row[h_col], row[l_col], row[c_col])
        if last is None or cur != last:
            changes.append({
                'datetime': row['time_dt'],
                'origin': origin_name,
                'H': row[h_col], 'L': row[l_col], 'C': row[c_col],
                'avg': (row[h_col] + row[l_col] + row[c_col]) / 3.0,
                'spread': (row[h_col] - row[l_col])
            })
            last = cur
    return changes[::-1]  # newest last

def find_most_current_data(small_df, report_time, origin_name, scope_days=20):
    """ Return the latest H/L/C available at/before report_time for origin. """
    rows = find_new_data_changes(small_df, report_time, origin_name, scope_days)
    if not rows:
        return None
    return rows[-1]

def calculate_raw_m_values(hlc_data, range_low, range_high):
    """ Compute raw-m bounds given desired output range (inclusive). """
    spread = hlc_data.get('spread', 0.0)
    avg = hlc_data.get('avg', None)
    if spread is None or avg is None or spread == 0:
        return None
    low_m  = (range_low  - avg) / spread
    high_m = (range_high - avg) / spread
    return {'raw_m_low': float(min(low_m, high_m)), 'raw_m_high': float(max(low_m, high_m))}

def find_valid_m_values(
    measurement_df, raw_m_low, raw_m_high, hlc_data,
    range_low, range_high, is_high_range=False,
    data_source= "Unknown", report_time=None, small_df=None, batch_inputs=None
):
    """ Filter measurement_df by raw-m window and origin's HLC into final rows. """
    if measurement_df is None or measurement_df.empty:
        return pd.DataFrame()

    # Raw m within bounds
    m_col = 'M #' if 'M #' in measurement_df.columns else 'M'
    if m_col not in measurement_df.columns:
        return pd.DataFrame()

    mask = measurement_df[m_col].astype(float).between(raw_m_low, raw_m_high, inclusive='both')
    subset = measurement_df.loc[mask].copy()
    if subset.empty:
        return subset

    # Enrich with origin/output columns
    subset['Output Low Requested']  = range_low
    subset['Output High Requested'] = range_high
    subset['Origin'] = hlc_data['origin']
    subset['Arrival'] = hlc_data['datetime']
    subset['Output']  = hlc_data['avg'] + subset[m_col].astype(float) * hlc_data['spread']
    subset['Feed']    = data_source  # expected: 'sm' or 'bg' by caller
    return subset

# ---------------------------------------------------------------------
# Core processors (Custom & Full)
# ---------------------------------------------------------------------

def _always_include_special_origins(origins: list):
    base = {o.strip() for o in origins}
    base.update(['WASP-12b', 'Macedonia'])
    return sorted(base)

def process_custom_ranges_advanced(
    measurement_df, small_df, report_time, custom_ranges, scope_days=20, big_df=None, run_model_g=False,
    *, day_start_hour: int = 18  # default 18; app doesn't pass it for custom ranges
):
    \"\"\"Process custom ranges; ensures WASP-12b and Macedonia are synthesized correctly.\"\"\"
    if measurement_df is None or measurement_df.empty:
        return pd.DataFrame()

    # Derive origins from H columns + force include specials
    origins = [c[:-2] for c in small_df.columns if c.endswith(' H')]
    origins = _always_include_special_origins(origins)

    rpt_dt = safe_to_datetime(report_time)
    if not isinstance(rpt_dt, pd.Timestamp):
        return pd.DataFrame()

    rows = []

    # Helper to push results for a single HLC 'event'
    def _push_for_hlc(hlc, feed_tag):
        for (rng_low, rng_high, is_high) in custom_ranges:
            m_bounds = calculate_raw_m_values(hlc, rng_low, rng_high)
            if not m_bounds:
                continue
            df_add = find_valid_m_values(
                measurement_df,
                m_bounds['raw_m_low'], m_bounds['raw_m_high'],
                hlc, rng_low, rng_high, is_high_range=is_high,
                data_source=feed_tag, report_time=rpt_dt, small_df=small_df
            )
            if df_add is not None and not df_add.empty:
                rows.append(df_add)

    # Walk origins
    for origin in origins:
        name_l = origin.lower()
        if name_l in ('wasp-12b', 'wasp'):
            base = find_most_current_data(small_df, rpt_dt, 'WASP-12b', scope_days)
            if base:
                for w in (0, 1, 2):
                    when = _sunday_start(rpt_dt, day_start_hour) - timedelta(weeks=w)
                    hlc = dict(base)
                    hlc['datetime'] = when
                    hlc['origin']   = 'Wasp-12b' if w == 0 else f'Wasp-12b[-{w}]'
                    _push_for_hlc(hlc, 'sm')
        elif name_l == 'macedonia':
            base = find_most_current_data(small_df, rpt_dt, 'Macedonia', scope_days)
            if base:
                month0 = _month_start(rpt_dt, day_start_hour)
                for mback in (0, 1, 2):
                    when = _shift_months(month0, mback, day_start_hour)
                    hlc = dict(base)
                    hlc['datetime'] = when
                    hlc['origin']   = 'Macedonia' if mback == 0 else f'Macedonia[-{mback}]'
                    _push_for_hlc(hlc, 'sm')
        else:
            changes = find_new_data_changes(small_df, rpt_dt, origin, scope_days)
            for hlc in changes:
                _push_for_hlc(hlc, 'sm')

        # If big_df is provided and contains matching H/L/C, mirror behavior for 'bg'
        if big_df is not None and isinstance(big_df, pd.DataFrame):
            if name_l in ('wasp-12b', 'wasp'):
                base = find_most_current_data(big_df, rpt_dt, 'WASP-12b', scope_days)
                if base:
                    for w in (0, 1, 2):
                        when = _sunday_start(rpt_dt, day_start_hour) - timedelta(weeks=w)
                        hlc = dict(base)
                        hlc['datetime'] = when
                        hlc['origin']   = 'Wasp-12b' if w == 0 else f'Wasp-12b[-{w}]'
                        _push_for_hlc(hlc, 'bg')
            elif name_l == 'macedonia':
                base = find_most_current_data(big_df, rpt_dt, 'Macedonia', scope_days)
                if base:
                    month0 = _month_start(rpt_dt, day_start_hour)
                    for mback in (0, 1, 2):
                        when = _shift_months(month0, mback, day_start_hour)
                        hlc = dict(base)
                        hlc['datetime'] = when
                        hlc['origin']   = 'Macedonia' if mback == 0 else f'Macedonia[-{mback}]'
                        _push_for_hlc(hlc, 'bg')
            else:
                changes = find_new_data_changes(big_df, rpt_dt, origin, scope_days)
                for hlc in changes:
                    _push_for_hlc(hlc, 'bg')

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    # Paste 'Input @ 18:00' and 'Input @ Report' constants + debug
    out = _apply_inputs_columns_and_debug(out, small_df, big_df, rpt_dt, day_start_hour, show_debug=True)
    return out

def apply_custom_ranges_advanced(
    df, small_df, report_time, high1, high2, low1, low2,
    use_high1, use_high2, use_low1, use_low2,
    big_df=None, run_model_g=False
):
    \"\"\"Public entry for custom ranges. Signature kept for app compatibility.\"\"\"
    custom_ranges = []
    if use_high1: custom_ranges.append((high1[0],  high1[1],  True))
    if use_high2: custom_ranges.append((high2[0],  high2[1],  True))
    if use_low1:  custom_ranges.append((low1[0],   low1[1],   False))
    if use_low2:  custom_ranges.append((low2[0],   low2[1],   False))

    return process_custom_ranges_advanced(
        df, small_df, report_time, custom_ranges, scope_days=20, big_df=big_df, run_model_g=run_model_g,
        day_start_hour=18  # fixed to 18:00 for custom per your request
    )

def process_full_range_advanced(
    measurement_df, small_df, report_time, center, window_radius, scope_days=20, big_df=None, run_model_g=False,
    *, day_start_hour: int = 18
):
    \"\"\"Process a symmetric window around center, scanning all origins (incl. specials).\"\"\"
    if measurement_df is None or measurement_df.empty:
        return pd.DataFrame()

    origins = [c[:-2] for c in small_df.columns if c.endswith(' H')]
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
            m_bounds['raw_m_low'], m_bounds['raw_m_high'],
            hlc, rng_low, rng_high, is_high_range=(center >= 0),
            data_source=feed_tag, report_time=rpt_dt, small_df=small_df
        )
        if df_add is not None and not df_add.empty:
            rows.append(df_add)

    for origin in origins:
        name_l = origin.lower()
        if name_l in ('wasp-12b', 'wasp'):
            base_sm = find_most_current_data(small_df, rpt_dt, 'WASP-12b', scope_days)
            if base_sm:
                for w in (0, 1, 2):
                    when = _sunday_start(rpt_dt, day_start_hour) - timedelta(weeks=w)
                    hlc = dict(base_sm); hlc['datetime'] = when; hlc['origin'] = 'Wasp-12b' if w == 0 else f'Wasp-12b[-{w}]'
                    _push_for_hlc(hlc, 'sm')
            if big_df is not None:
                base_bg = find_most_current_data(big_df, rpt_dt, 'WASP-12b', scope_days)
                if base_bg:
                    for w in (0, 1, 2):
                        when = _sunday_start(rpt_dt, day_start_hour) - timedelta(weeks=w)
                        hlc = dict(base_bg); hlc['datetime'] = when; hlc['origin'] = 'Wasp-12b' if w == 0 else f'Wasp-12b[-{w}]'
                        _push_for_hlc(hlc, 'bg')
        elif name_l == 'macedonia':
            base_sm = find_most_current_data(small_df, rpt_dt, 'Macedonia', scope_days)
            if base_sm:
                month0 = _month_start(rpt_dt, day_start_hour)
                for mback in (0, 1, 2):
                    when = _shift_months(month0, mback, day_start_hour)
                    hlc = dict(base_sm); hlc['datetime'] = when; hlc['origin'] = 'Macedonia' if mback == 0 else f'Macedonia[-{mback}]'
                    _push_for_hlc(hlc, 'sm')
            if big_df is not None:
                base_bg = find_most_current_data(big_df, rpt_dt, 'Macedonia', scope_days)
                if base_bg:
                    month0 = _month_start(rpt_dt, day_start_hour)
                    for mback in (0, 1, 2):
                        when = _shift_months(month0, mback, day_start_hour)
                        hlc = dict(base_bg); hlc['datetime'] = when; hlc['origin'] = 'Macedonia' if mback == 0 else f'Macedonia[-{mback}]'
                        _push_for_hlc(hlc, 'bg')
        else:
            changes_sm = find_new_data_changes(small_df, rpt_dt, origin, scope_days)
            for hlc in changes_sm:
                _push_for_hlc(hlc, 'sm')
            if big_df is not None:
                changes_bg = find_new_data_changes(big_df, rpt_dt, origin, scope_days)
                for hlc in changes_bg:
                    _push_for_hlc(hlc, 'bg')

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = _apply_inputs_columns_and_debug(out, small_df, big_df, rpt_dt, day_start_hour, show_debug=True)
    return out

def apply_full_range_advanced(
    df, small_df, report_time, window_radius, day_start_hour=18, input_value_at_start=None, big_df=None, run_model_g=False
):
    \"\"\"Public entry for full-range scanning (symmetric about the chosen center).\"\"\"
    # 'center' is the user-selected input value (at or before selected time)
    center = input_value_at_start if input_value_at_start is not None else 0.0
    return process_full_range_advanced(
        df, small_df, report_time, center=center, window_radius=window_radius,
        scope_days=20, big_df=big_df, run_model_g=run_model_g, day_start_hour=day_start_hour
    )
