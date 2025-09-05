"""
This is the update to version 0810.  It is still fast, prints 16,000 lines in under 17 seconds, was under 10 seconds with errors.
it fixed the missing Macedonia[-1], Macedonia[-2], Wasp-12b[-1], Wasp-12b[-2]. (under 10 seconds)
Now addresses Input @ 1800 value for each csv, and Input @ Report for each csv..  Input @ 18:00 renamed to Input @ Start.
Input @ start should show 2 values, but instead it shows 4 values.  
Input @ report should show 2 values, but instead it shows only 1 value.
This current version runs in 17 seconds instead of 8 seconds.  
"""

import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st

# ------------------------------
# Utilities
# ------------------------------

def _clean_ts_series(s):
    """Parse a pandas Series of timestamps (strings or datetimes). Strip timezone offsets and 'T'."""
    s = s.astype(str).str.replace(r"[+-]\d{2}:?\d{2}$", "", regex=True).str.replace("T", " ")
    return pd.to_datetime(s, errors="coerce")

def clean_timestamp(x):
    """Public helper used elsewhere in app; tolerant of strings/naive/aware datetimes."""
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.replace(tzinfo=None) if getattr(x, "tzinfo", None) is not None else x
    if isinstance(x, str):
        x = x.replace("T", " ")
        x = pd.Series([x]).str.replace(r"[+-]\d{2}:?\d{2}$", "", regex=True).iloc[0]
        try:
            return pd.to_datetime(x, errors="coerce").to_pydatetime()
        except Exception:
            return pd.NaT
    return pd.NaT

def _ensure_time_dt(df):
    """Ensure df has a 'time_dt' column derived from 'time' (lowercase) or first column."""
    df = df.copy()
    time_col = 'time' if 'time' in df.columns else df.columns[0]
    df['time_dt'] = _clean_ts_series(df[time_col])
    return df

def _row_at_or_before_report(df, report_time):
    """Return the row at report_time or the last row <= report_time."""
    rpt = clean_timestamp(report_time)
    df2 = _ensure_time_dt(df)
    df2 = df2.sort_values('time_dt')
    df2 = df2[df2['time_dt'] <= rpt]
    if df2.empty:
        return None
    return df2.iloc[-1]

def _most_recent_sunday_anchor(report_time, day_start_hour=18, offset_weeks=0):
    """Most recent Sunday @ hour; offset_weeks=1 => one Sunday earlier, etc."""
    rt = clean_timestamp(report_time)
    # Monday=0..Sunday=6
    days_since_sun = (rt.weekday() + 1) % 7  # Sunday -> 0
    anchor = rt - timedelta(days=days_since_sun) - timedelta(weeks=offset_weeks)
    return anchor.replace(hour=day_start_hour, minute=0, second=0, microsecond=0)

def _first_of_month_anchor(report_time, day_start_hour=18, offset_months=0):
    """Most recent first-of-month @ hour; offset_months=1 => previous month, etc."""
    rt = clean_timestamp(report_time)
    # Compute target month/year
    y = rt.year
    m = rt.month - offset_months
    while m <= 0:
        m += 12
        y -= 1
    return datetime(y, m, 1, day_start_hour, 0, 0)

def _get_hlc_from_row(row, base_name, idx=None):
    """Fetch H/L/C from the given row for base_name; idx=None for base, 1 or 2 for bracketed set."""
    suf = "" if idx is None else f"[{idx}]"
    cols = (f"{base_name} H{suf}", f"{base_name} L{suf}", f"{base_name} C{suf}")
    try:
        H, L, C = (float(row.get(cols[0], np.nan)), float(row.get(cols[1], np.nan)), float(row.get(cols[2], np.nan)))
    except Exception:
        H, L, C = (np.nan, np.nan, np.nan)
    if np.isnan(H) or np.isnan(L) or np.isnan(C):
        return None
    return {"H": H, "L": L, "C": C}

# ------------------------------
# Helpers for open values
# ------------------------------

def get_open_at(df, target_time):
    df2 = _ensure_time_dt(df)
    df2 = df2.sort_values("time_dt")
    row = df2[df2["time_dt"] <= target_time].iloc[-1:]
    if row.empty:
        return None
    for col in ("open", "Open"):
        if col in row.columns:
            return float(row.iloc[0][col])
    return None

# ------------------------------
# Generic HLC discovery (unchanged behavior for non-special origins)
# ------------------------------

def find_new_data_changes(small_df, report_time, origin_name, scope_days=20):
    """
    For MOST origins: detect changes over time.
    For WASP-12b/Macedonia: this function is bypassed elsewhere.
    """
    try:
        h_col = f"{origin_name} H"
        l_col = f"{origin_name} L"
        c_col = f"{origin_name} C"
        if not all(col in small_df.columns for col in (h_col, l_col, c_col)):
            return []
        sdf = _ensure_time_dt(small_df)
        rpt = clean_timestamp(report_time)
        scope_start = rpt - timedelta(days=scope_days)
        scoped = sdf[(sdf['time_dt'] >= scope_start) & (sdf['time_dt'] <= rpt)].copy()
        if scoped.empty:
            return []
        scoped = scoped.sort_values('time_dt')
        out, prev = [], (None, None, None)
        for _, r in scoped.iterrows():
            vals = (r[h_col], r[l_col], r[c_col])
            if any(pd.isna(v) for v in vals):
                continue
            cur = tuple(float(v) for v in vals)
            if prev == (None, None, None) or cur != prev:
                out.append({"H": cur[0], "L": cur[1], "C": cur[2], "datetime": r['time_dt'], "origin": origin_name})
                prev = cur
        return out
    except Exception as e:
        st.error(f"Error in find_new_data_changes({origin_name}): {e}")
        return []

def find_most_current_data(small_df, report_time, origin_name, scope_days=20):
    """Most recent valid H/L/C at/before report_time for generic origins."""
    try:
        h_col = f"{origin_name} H"
        l_col = f"{origin_name} L"
        c_col = f"{origin_name} C"
        if not all(col in small_df.columns for col in (h_col, l_col, c_col)):
            return None
        sdf = _ensure_time_dt(small_df)
        rpt = clean_timestamp(report_time)
        same_day = sdf[sdf['time_dt'].dt.date == rpt.date()]
        if not same_day.empty:
            same_day = same_day[same_day['time_dt'] <= rpt].sort_values('time_dt', ascending=False)
            for _, r in same_day.iterrows():
                if not any(pd.isna(r[c] for c in (h_col, l_col, c_col))):
                    return {"H": float(r[h_col]), "L": float(r[l_col]), "C": float(r[c_col]), "datetime": r['time_dt'], "origin": origin_name}
        # fallback in scope
        scope_start = rpt - timedelta(days=scope_days)
        scoped = sdf[(sdf['time_dt'] >= scope_start) & (sdf['time_dt'] <= rpt)].sort_values('time_dt', ascending=False)
        for _, r in scoped.iterrows():
            if not any(pd.isna(r[c] for c in (h_col, l_col, c_col))):
                return {"H": float(r[h_col]), "L": float(r[l_col]), "C": float(r[c_col]), "datetime": r['time_dt'], "origin": origin_name}
        return None
    except Exception as e:
        st.error(f"Error in find_most_current_data({origin_name}): {e}")
        return None

# ------------------------------
# Special one-shot grab for WASP-12b and Macedonia
# ------------------------------

def _build_wasp_macedonia_hlc_list(hlc_df, report_time, base_name, day_start_hour=18):
    """Grab a *single* report-time row and return three HLC dicts with correct Arrival and origin label.
       base_name: 'WASP-12b' or 'Macedonia'
    """
    row = _row_at_or_before_report(hlc_df, report_time)
    if row is None:
        return []

    entries = []

    if base_name == "WASP-12b":
        # This week (idx=None)
        for idx, label, weeks_back in [(None, "Wasp-12b", 0), (1, "Wasp-12b[-1]", 1), (2, "Wasp-12b[-2]", 2)]:
            hlc = _get_hlc_from_row(row, "WASP-12b", idx)
            if hlc:
                hlc['datetime'] = _most_recent_sunday_anchor(report_time, day_start_hour, weeks_back)
                hlc['origin'] = label
                entries.append(hlc)

    elif base_name == "Macedonia":
        for idx, label, months_back in [(None, "Macedonia", 0), (1, "Macedonia[-1]", 1), (2, "Macedonia[-2]", 2)]:
            hlc = _get_hlc_from_row(row, "Macedonia", idx)
            if hlc:
                hlc['datetime'] = _first_of_month_anchor(report_time, day_start_hour, months_back)
                hlc['origin'] = label
                entries.append(hlc)

    return entries

# ------------------------------
# Core math
# ------------------------------

def calculate_raw_m_values(hlc_data, range_low, range_high):
    try:
        H, L, C = hlc_data['H'], hlc_data['L'], hlc_data['C']
        avg = (H + L + C) / 3.0
        spread = H - L
        if spread == 0:
            return None
        return {'raw_m_low': (range_low - avg) / spread, 'raw_m_high': (range_high - avg) / spread, 'avg': avg, 'spread': spread}
    except Exception as e:
        st.error(f"Error calculating raw M: {e}")
        return None

def find_valid_m_values(measurement_df, raw_m_low, raw_m_high, hlc_data, range_low, range_high, is_high_range=False, data_source="Unknown", report_time=None):
    try:
        valid_entries, valid_m_values = [], []

        # Flexible M value column
        m_value_col = next((c for c in ['M value','M Value','M_Value','M_value','m value','m_value'] if c in measurement_df.columns), None)
        if m_value_col is None:
            return {'valid_entries': [], 'valid_m_list': []}

        m_values = measurement_df[m_value_col].dropna().unique()

        for m_val in m_values:
            try:
                m_float = float(m_val)
            except Exception:
                continue
            if not (raw_m_low <= m_float <= raw_m_high):
                continue

            output = hlc_data['avg'] + m_float * hlc_data['spread']
            valid_m_values.append(m_float)

            # rows matching this M value
            rows = measurement_df[measurement_df[m_value_col] == m_val]

            # Zone
            for _, row in rows.iterrows():
                if range_low <= output <= range_high:
                    if is_high_range:
                        d = range_high - output
                    else:
                        d = output - range_low
                    if d <= 6: zone = "0 to 6"
                    elif d <= 12: zone = "6 to 12"
                    elif d <= 18: zone = "12 to 18"
                    else: zone = "18 to 24"
                else:
                    zone = "Out of Range"

                # Arrival format & day index
                try:
                    arrival_dt = hlc_data['datetime']
                    ddd = arrival_dt.strftime('%a')
                    arrival_excel = arrival_dt.strftime('%Y-%m-%d %H:%M')
                    try:
                        from a_helpers import get_day_index
                        day_index = get_day_index(arrival_dt, report_time, 18)
                    except Exception:
                        day_index = "[0]"
                except Exception:
                    ddd, arrival_excel, day_index = "", str(hlc_data.get('datetime','')), "[0]"

                feed_type = "Small" if data_source == "Small CSV" else "Big"
                input_start = hlc_data.get("input_start", 0)
                input_arrival = hlc_data.get('C', 0)
                input_report = hlc_data.get("input_report", 0)

                valid_entries.append({
                    'Feed': feed_type,
                    'ddd': ddd,
                    'Arrival': arrival_excel,
                    'Day': day_index,
                    'Origin': hlc_data['origin'],
                    'M Name': row.get('M Name', row.get('m name', f"M{m_val}")),
                    'M #': row.get('M #', row.get('m #', m_val)),
                    'M Value': m_val,
                    'R #': row.get('R #', row.get('r #', '')),
                    'Tag': row.get('Tag', row.get('tag', '')),
                    'Family': row.get('Family', row.get('family', '')),
                    'Input @ start': input_start,
                    'Diff @ start': output - input_start,
                    'Input @ Arrival': input_arrival,
                    'Diff @ Arrival': output - input_arrival,
                    'Input @ Report': input_report,
                    'Diff @ Report': output - input_report,
                    'Output': output,
                    'Range': f"{range_low:.1f}-{range_high:.1f}",
                    'Zone': zone
                })

        return {'valid_entries': valid_entries, 'valid_m_list': valid_m_values}
    except Exception as e:
        st.error(f"Error finding valid M values: {e}")
        return {'valid_entries': [], 'valid_m_list': []}

# ------------------------------
# Processing flows
# ------------------------------

def _collect_origins(hlc_df_list):
    all_origins = set()
    for hlc_df, _ in hlc_df_list:
        for col in hlc_df.columns:
            if col.endswith(" H"):
                all_origins.add(col[:-2])
    return list(all_origins)

def process_custom_ranges_advanced(measurement_df, small_df, report_time, custom_ranges, scope_days=20, big_df=None, run_model_g=False, day_start_hour=18):
    all_valid_entries, processing_summary = [], []
    data_sources = []
    if small_df is not None and not small_df.empty:
        data_sources.append((small_df.copy(), "Small CSV"))
    if big_df is not None and not big_df.empty:
        data_sources.append((big_df.copy(), "Big CSV"))

    origins = _collect_origins(data_sources)

    # Iterate ranges
    for range_name, cfg in custom_ranges.items():
        if not cfg.get('enabled', False):
            continue
        val = cfg.get('value', 0)
        if val == 0:
            continue

        if range_name.startswith("High"):
            range_low, range_high, is_high_range = val - 24, val, True
        else:
            range_low, range_high, is_high_range = val, val + 24, False

        range_entries = []

        for hlc_df, data_source in data_sources:
            for origin in origins:
                hlc_sets = []

                # SPECIAL: WASP-12b and Macedonia -> one-shot grab from report row
                if origin == "WASP-12b":
                    hlc_sets = _build_wasp_macedonia_hlc_list(hlc_df, report_time, "WASP-12b", day_start_hour)
                elif origin == "Macedonia":
                    hlc_sets = _build_wasp_macedonia_hlc_list(hlc_df, report_time, "Macedonia", day_start_hour)
                else:
                    # all other origins: changes-based discovery
                    hlc_sets = find_new_data_changes(hlc_df, report_time, origin, scope_days)

                if not hlc_sets:
                    continue
                    
                for hlc in hlc_sets:
                    # Attach open values once per feed
                    if origin == "Macedonia":
                        start_anchor = _first_of_month_anchor(report_time, day_start_hour)
                    else:
                        start_anchor = _most_recent_sunday_anchor(report_time, day_start_hour)
                    # start_anchor = _most_recent_sunday_anchor(report_time, day_start_hour) if origin != "Macedonia" else _first_of_month_anchor(report_time, day_start_hour)
                    
                    hlc['input_start'] = get_open_at(hlc_df, start_anchor)
                    hlc['input_report'] = get_open_at(hlc_df, clean_timestamp(report_time))
                    
                    calc = calculate_raw_m_values(hlc, range_low, range_high)
                    if not calc:
                        continue
                    h2 = {**hlc, **calc}
                    res = find_valid_m_values(measurement_df, calc['raw_m_low'], calc['raw_m_high'], h2, range_low, range_high, is_high_range, data_source, report_time)
                    range_entries.extend(res['valid_entries'])

                    # summary (compact)
                    dt_str = hlc['datetime'].strftime('%m/%d/%Y %H:%M') if hasattr(hlc['datetime'], 'strftime') else str(hlc['datetime'])
                    processing_summary.append({
                        'Range': f"{range_low:.1f}-{range_high:.1f}",
                        'Feed': data_source.replace(" CSV",""),
                        'DateTime': dt_str,
                        'Origin': hlc['origin'],
                        'H': hlc['H'], 'L': hlc['L'], 'C': hlc['C'],
                        'Raw M Low': calc['raw_m_low'], 'Raw M High': calc['raw_m_high'],
                        'Valid M Values': len(res['valid_m_list']),
                        'Valid list': ', '.join([f"{m:.1f}" for m in res['valid_m_list']]) or 'None'
                    })

        all_valid_entries.extend(range_entries)
        st.info(f"{range_name}: Found {len(range_entries)} valid entries")

    # (Optional) show summary
    if processing_summary:
        st.markdown("### Processing Summary")
        st.dataframe(pd.DataFrame(processing_summary), use_container_width=True)

    # Model G (unchanged behavior)
    if all_valid_entries and run_model_g:
        try:
            from a_helpers import GROUP_1A_TRAVELERS
            custom_df = pd.DataFrame(all_valid_entries)
            grp_1a_df = custom_df[custom_df['M #'].isin(GROUP_1A_TRAVELERS)].copy()
            if not grp_1a_df.empty:
                try:
                    from models_g_updated import run_model_g_detection
                except ImportError:
                    try:
                        from model_g import run_model_g_detection
                    except ImportError:
                        from model_g_detector import run_model_g_detection
                run_model_g_detection(grp_1a_df, report_time, key_suffix="_custom")
        except Exception as e:
            st.warning(f"Model G detection not available: {e}")

    return all_valid_entries

def apply_custom_ranges_advanced(df, small_df, report_time, high1, high2, low1, low2, use_high1, use_high2, use_low1, use_low2, big_df=None, run_model_g=False, day_start_hour=18):
    st.info(f"🧮 Advanced Custom Range Processing Started - {len(df)} measurements to process")

    custom_ranges = {}
    if use_high1 and high1 > 0: custom_ranges['High 1'] = {'enabled': True, 'value': high1}
    if use_high2 and high2 > 0: custom_ranges['High 2'] = {'enabled': True, 'value': high2}
    if use_low1 and low1 > 0:   custom_ranges['Low 1']  = {'enabled': True, 'value': low1}
    if use_low2 and low2 > 0:   custom_ranges['Low 2']  = {'enabled': True, 'value': low2}
    if not custom_ranges:
        return df

    valid_entries = process_custom_ranges_advanced(
        measurement_df=df,
        small_df=small_df,
        report_time=report_time,
        custom_ranges=custom_ranges,
        big_df=big_df,
        run_model_g=run_model_g,
        day_start_hour=day_start_hour
    )

    if not valid_entries:
        st.warning("No entries found using advanced H/L/C calculation")
        return pd.DataFrame()

    out = pd.DataFrame(valid_entries)

    # Preserve Zone calculated earlier; just set Range label for display completeness
    def _range_name(output_val):
        for rname, rcfg in custom_ranges.items():
            v = rcfg['value']
            if rname.startswith('High'):
                lo, hi = v - 24, v
            else:
                lo, hi = v, v + 24
            if lo <= output_val <= hi:
                return rname
        return 'Other'

    out['Range'] = out['Output'].apply(_range_name)
    return out

def process_full_range_advanced(measurement_df, small_df, report_time, center, window_radius, scope_days=20, big_df=None, run_model_g=False, day_start_hour=18):
    lo, hi = center - window_radius, center + window_radius
    st.info(f"🧮 Full Range (Advanced) window: [{lo}, {hi}] around center={center}")

    all_valid_entries, processing_summary = [], []
    data_sources = []
    if small_df is not None and not small_df.empty:
        data_sources.append((small_df.copy(), "Small CSV"))
    if big_df is not None and not big_df.empty:
        data_sources.append((big_df.copy(), "Big CSV"))

    origins = _collect_origins(data_sources)

    for hlc_df, data_source in data_sources:
        for origin in origins:
            if origin == "WASP-12b":
                hlc_sets = _build_wasp_macedonia_hlc_list(hlc_df, report_time, "WASP-12b", day_start_hour)
            elif origin == "Macedonia":
                hlc_sets = _build_wasp_macedonia_hlc_list(hlc_df, report_time, "Macedonia", day_start_hour)
            else:
                hlc_sets = find_new_data_changes(hlc_df, report_time, origin, scope_days)

            if not hlc_sets:
                continue

            for hlc in hlc_sets:
                # Attach open values once per feed
                if origin == "Macedonia":
                    start_anchor = _first_of_month_anchor(report_time, day_start_hour)
                else:
                    start_anchor = _most_recent_sunday_anchor(report_time, day_start_hour)
                # start_anchor = _most_recent_sunday_anchor(report_time, day_start_hour) if origin != "Macedonia" else _first_of_month_anchor(report_time, day_start_hour)
                
                hlc['input_start'] = get_open_at(hlc_df, start_anchor)
                hlc['input_report'] = get_open_at(hlc_df, clean_timestamp(report_time))
                
                calc = calculate_raw_m_values(hlc, lo, hi)
                if not calc:
                    continue
                h2 = {**hlc, **calc}
                res = find_valid_m_values(measurement_df, calc['raw_m_low'], calc['raw_m_high'], h2, lo, hi, False, data_source, report_time)
                all_valid_entries.extend(res['valid_entries'])

                dt_str = hlc['datetime'].strftime('%m/%d/%Y %H:%M') if hasattr(hlc['datetime'], 'strftime') else str(hlc['datetime'])
                processing_summary.append({
                    'Range': f"{lo:.1f}-{hi:.1f}",
                    'Feed': data_source.replace(' CSV',''),
                    'DateTime': dt_str,
                    'Origin': hlc['origin'],
                    'H': hlc['H'],'L': hlc['L'],'C': hlc['C'],
                    'Raw M Low': calc['raw_m_low'],'Raw M High': calc['raw_m_high'],
                    'Valid M Values': len(res['valid_m_list']),
                    'Valid list': ', '.join([f"{m:.1f}" for m in res['valid_m_list']]) or 'None'
                })

    if processing_summary:
        st.markdown("### Full Range – Processing Summary")
        st.dataframe(pd.DataFrame(processing_summary), use_container_width=True)

    return all_valid_entries

def apply_full_range_advanced(df, small_df, report_time, window_radius, day_start_hour=18, input_value_at_start=None, big_df=None, run_model_g=False):
    # Center selection
    center = None
    if input_value_at_start is not None and not pd.isna(input_value_at_start):
        center = float(input_value_at_start)
    else:
        try:
            sdf = _ensure_time_dt(small_df.copy())
            rpt = clean_timestamp(report_time)
            sdf = sdf[sdf['time_dt'] <= rpt]
            if not sdf.empty:
                # compute day start boundary
                base = dt.datetime(rpt.year, rpt.month, rpt.day, day_start_hour, 0, 0)
                if rpt < base:
                    base = base - dt.timedelta(days=1)
                # Prefer exact match at start hour
                exact = sdf[sdf['time_dt'] == base]
                if not exact.empty:
                    row = exact.iloc[-1]
                else:
                    row = sdf.iloc[-1]
                for cand in ('open','Open','close'):
                    if cand in small_df.columns:
                        try:
                            center = float(row[cand])
                            break
                        except Exception:
                            continue
        except Exception:
            center = None

    if center is None or pd.isna(center):
        st.error("Full Range (Advanced): could not determine center. Provide input @ start or ensure small feed has time/open/close.")
        return pd.DataFrame()

    valid_entries = process_full_range_advanced(
        measurement_df=df,
        small_df=small_df,
        report_time=report_time,
        center=center,
        window_radius=window_radius,
        scope_days=20,
        big_df=big_df,
        run_model_g=run_model_g,
        day_start_hour=day_start_hour
    )

    if not valid_entries:
        st.warning("Full Range (Advanced): no valid entries found.")
        return pd.DataFrame()

    out_df = pd.DataFrame(valid_entries).drop(columns=['Range','Zone'], errors='ignore')

    preferred_cols = [
        'Feed','ddd','Arrival','Day','Origin',
        'M Name','M #','M Value','R #','Tag','Family',
        'Input @ start','Diff @ start','Input @ Arrival','Diff @ Arrival',
        'Input @ Report','Diff @ Report','Output'
    ]
    ordered = [c for c in preferred_cols if c in out_df.columns]
    remaining = [c for c in out_df.columns if c not in ordered]
    out_df = out_df[ordered + remaining]

    st.success(f"✅ Full Range (Advanced): {len(out_df)} entries")
    return out_df
