"""
v1125_4. FIXES two-pass processing type mismatch and adds processing summaries.
- Fixed M # comparison by converting to int (was failing due to string/int mismatch)
- Added 'Pass' column to results to identify Pass1 vs Pass2 entries
- Returns tuple (dataframe, summary_dict) instead of just dataframe
- Summary includes counts, M numbers, origins, and feeds for both passes

v1125_3.  Table generation process completes but shows 0 results. 
v1125_2.  This First iteration does not work.  Produces error:  Error in process_cluster_tables_two_pass: too many values to unpack (expected 2).
v1125 update to version 0813.  Adds first and second pass for Swing Analysys Tool. 

v0813 update to version v0810.  It is still fast, prints 16,000 lines in under 17 seconds, was under 10 seconds with errors.
This fixes the missing Macedonia[-1], Macedonia[-2], Wasp-12b[-1], Wasp-12b[-2]. (under 10 seconds)
Now addresses Input @ 1800 value for each csv, and Input @ Report for each csv.  Input @ 18:00 renamed to Input @ Start.
Input @ start shows 2 values, one for each csv, which is correct..  
Input @ report shows 1 value, which is incorrect, should show one for each csv.  Fix this later.
This version runs in 17 seconds instead of 8 seconds.  
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
    rt = clean_timestamp(report_time)
    days_since_sun = (rt.weekday() + 1) % 7
    anchor = rt - timedelta(days=days_since_sun) - timedelta(weeks=offset_weeks)
    return anchor.replace(hour=day_start_hour, minute=0, second=0, microsecond=0)

def _first_of_month_anchor(report_time, day_start_hour=18, offset_months=0):
    """Most recent Sunday @ hour; offset_weeks=1 => one Sunday earlier, etc."""
    rt = clean_timestamp(report_time)
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
    df2 = df2[df2['time_dt'] <= target_time]
    if df2.empty:
        return None
    row = df2.iloc[-1]
    for col in ("open", "Open"):
        if col in row.index:
            try:
                return float(row[col])
            except Exception:
                continue
    return None

# ------------------------------
# Generic HLC discovery (unchanged behavior for non-Wasp/Macedonia origins)
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

            rows = measurement_df[measurement_df[m_value_col] == m_val]
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
                input_start = hlc_data.get('input_start', 0)
                input_arrival = hlc_data.get('C', 0)
                input_report = hlc_data.get('input_report', 0)

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

    for hlc_df, data_source in data_sources:
        feed_type = "Small" if data_source == "Small CSV" else "Big"
        # capture once per feed
        start_anchor = _most_recent_sunday_anchor(report_time, day_start_hour)
        feed_start = get_open_at(hlc_df, start_anchor)
        feed_report = get_open_at(hlc_df, clean_timestamp(report_time))

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
                    hlc['input_start'] = feed_start
                    hlc['input_report'] = feed_report

                    calc = calculate_raw_m_values(hlc, range_low, range_high)
                    if not calc:
                        continue
                    h2 = {**hlc, **calc}
                    res = find_valid_m_values(measurement_df, calc['raw_m_low'], calc['raw_m_high'], h2, range_low, range_high, is_high_range, data_source, report_time)
                    range_entries.extend(res['valid_entries'])

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
    """Process the full range around a center (center +/- window_radius).

    Captures open values once per feed (small/big) and attaches them to each HLC set before
    computing raw M values and finding valid M entries.
    """
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
        feed_type = "Small" if data_source == "Small CSV" else "Big"
        # capture once per feed (use the usual start anchor)
        start_anchor = _most_recent_sunday_anchor(report_time, day_start_hour)
        feed_start = get_open_at(hlc_df, start_anchor)
        feed_report = get_open_at(hlc_df, clean_timestamp(report_time))

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
                # attach feed-level opens
                hlc['input_start'] = feed_start
                hlc['input_report'] = feed_report

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


def process_cluster_tables_two_pass(
    measurement_df,
    small_df,
    big_df,
    report_time,
    scope_days=20,
    day_start_hour=18,
    valid_list_pass1=None,
    valid_list_pass2=None,
    max_output_spread=3.0
):
    """
    Two-pass processing for cluster tables (FOGZ, Large Discounts, Recips PD).
    
    Pass 1: Get M#s from first 2 most recent days, filtered by valid_list_pass1
    Pass 2: Get M#s from all data within scope, filtered by valid_list_pass2
    
    Returns combined dataframe with all M#s from both passes.
    """
    if valid_list_pass1 is None:
        valid_list_pass1 = set()
    if valid_list_pass2 is None:
        valid_list_pass2 = set()
    
    try:
        # Collect all origins from both feeds
        # _collect_origins expects tuples of (df, name)
        hlc_df_list = [(small_df, 'Small')]
        if big_df is not None:
            hlc_df_list.append((big_df, 'Big'))
        origins = _collect_origins(hlc_df_list)
        
        all_pass1_results = []
        all_pass2_results = []
        
        # Process each feed separately
        for hlc_df, feed_name in hlc_df_list:
            # Get origins from this specific feed
            feed_origins = []
            for col in hlc_df.columns:
                if col.endswith(" H"):
                    origin = col[:-2]
                    if origin not in feed_origins:
                        feed_origins.append(origin)
            
            for origin_name in feed_origins:
                # Get HLC data changes from this feed
                hlc_data = find_new_data_changes(hlc_df, report_time, origin_name, scope_days)
                
                if not hlc_data:
                    continue
                
                # Sort by datetime descending to get most recent first
                hlc_data_sorted = sorted(hlc_data, key=lambda x: x['datetime'], reverse=True)
                
                # PASS 1: Get first 2 most recent days only
                if hlc_data_sorted:
                    # Get unique dates from the data
                    unique_dates = []
                    for item in hlc_data_sorted:
                        item_date = item['datetime'].date()
                        if item_date not in unique_dates:
                            unique_dates.append(item_date)
                        if len(unique_dates) >= 2:
                            break
                    
                    # Filter to only include data from first 2 days
                    if len(unique_dates) > 0:
                        pass1_data = [item for item in hlc_data_sorted 
                                     if item['datetime'].date() in unique_dates[:2]]
                        
                        # Process each HLC point for pass 1
                        for hlc_item in pass1_data:
                            range_low = hlc_item['L']
                            range_high = hlc_item['H']
                            
                            # Find valid M values within this range
                            valid_results = find_valid_m_values(
                                measurement_df,
                                raw_m_low=range_low,
                                raw_m_high=range_high,
                                hlc_data=[hlc_item],
                                range_low=range_low,
                                range_high=range_high,
                                is_high_range=False,
                                data_source=f"Pass1_{feed_name}_{origin_name}",
                                report_time=report_time
                            )
                            
                            # Filter by valid list pass 1 (using M #, not M Value)
                            if valid_results and valid_results.get('valid_entries'):
                                for result in valid_results['valid_entries']:
                                    try:
                                        m_num = int(float(result['M #']))
                                        if m_num in valid_list_pass1:
                                            # Add feed identifier and pass info
                                            result['Feed'] = feed_name
                                            result['Pass'] = 'Pass1'
                                            all_pass1_results.append(result)
                                    except (ValueError, TypeError):
                                        continue
                
                # PASS 2: Get all data within scope (no date restriction)
                for hlc_item in hlc_data_sorted:
                    range_low = hlc_item['L']
                    range_high = hlc_item['H']
                    
                    # Find valid M values within this range
                    valid_results = find_valid_m_values(
                        measurement_df,
                        raw_m_low=range_low,
                        raw_m_high=range_high,
                        hlc_data=[hlc_item],
                        range_low=range_low,
                        range_high=range_high,
                        is_high_range=False,
                        data_source=f"Pass2_{feed_name}_{origin_name}",
                        report_time=report_time
                    )
                    
                    # Filter by valid list pass 2 (using M #, not M Value)
                    if valid_results and valid_results.get('valid_entries'):
                        for result in valid_results['valid_entries']:
                            try:
                                m_num = int(float(result['M #']))
                                if m_num in valid_list_pass2:
                                    # Add feed identifier and pass info
                                    result['Feed'] = feed_name
                                    result['Pass'] = 'Pass2'
                                    all_pass2_results.append(result)
                            except (ValueError, TypeError):
                                continue
        
        # Combine both passes
        combined_results = all_pass1_results + all_pass2_results
        
        # Create processing summary
        processing_summary = {
            'pass1_count': len(all_pass1_results),
            'pass2_count': len(all_pass2_results),
            'total_count': len(combined_results),
            'pass1_m_numbers': sorted(set([int(float(r['M #'])) for r in all_pass1_results if r.get('M #')])),
            'pass2_m_numbers': sorted(set([int(float(r['M #'])) for r in all_pass2_results if r.get('M #')])),
            'pass1_origins': sorted(set([r['Origin'] for r in all_pass1_results if r.get('Origin')])),
            'pass2_origins': sorted(set([r['Origin'] for r in all_pass2_results if r.get('Origin')])),
            'pass1_feeds': sorted(set([r['Feed'] for r in all_pass1_results if r.get('Feed')])),
            'pass2_feeds': sorted(set([r['Feed'] for r in all_pass2_results if r.get('Feed')]))
        }
        
        if not combined_results:
            return pd.DataFrame(), processing_summary
        
        # Convert to DataFrame
        prep_df = pd.DataFrame(combined_results)
        
        # Remove duplicates (same M#, Origin, Arrival)
        if not prep_df.empty:
            prep_df = prep_df.drop_duplicates(subset=['M #', 'Origin', 'Arrival'], keep='first')
            prep_df = prep_df.sort_values(['Arrival', 'Output'], ascending=[False, False])
        
        return prep_df, processing_summary
        
    except Exception as e:
        st.error(f"Error in process_cluster_tables_two_pass: {e}")
        import traceback
        st.code(traceback.format_exc())
        empty_summary = {
            'pass1_count': 0, 'pass2_count': 0, 'total_count': 0,
            'pass1_m_numbers': [], 'pass2_m_numbers': [],
            'pass1_origins': [], 'pass2_origins': [],
            'pass1_feeds': [], 'pass2_feeds': []
        }
        return pd.DataFrame(), empty_summary


def match_cluster_table_entries(prep_df, valid_list_pass1, valid_list_pass2, max_output_spread=3.0, feed_filter=None):
    """
    Find matches between Pass1 M#s (recent 2 days) and Pass2 M#s (within scope).
    
    Parameters:
    - prep_df: Combined dataframe from both passes
    - valid_list_pass1: Set of M#s from pass 1 (recent data)
    - valid_list_pass2: Set of M#s from pass 2 (all data)
    - max_output_spread: Maximum allowed output difference
    - feed_filter: Optional feed filter ('Small' or 'Big')
    
    Returns:
    - DataFrame with matched cluster table entries in G.11 format
    """
    if prep_df.empty:
        return pd.DataFrame()
    
    try:
        # Filter by feed if specified
        if feed_filter:
            prep_df = prep_df[prep_df['Feed'] == feed_filter].copy()
        
        if prep_df.empty:
            return pd.DataFrame()
        
        # Separate pass1 and pass2 entries
        pass1_df = prep_df[prep_df['M #'].isin(valid_list_pass1)].copy()
        pass2_df = prep_df[prep_df['M #'].isin(valid_list_pass2)].copy()
        
        matches = []
        
        # For each pass1 entry, find matching pass2 entries
        for _, row1 in pass1_df.iterrows():
            m1 = row1['M #']
            output1 = row1['Output']
            origin1 = row1['Origin']
            arrival1 = row1.get('Arrival', '')
            feed1 = row1.get('Feed', '')
            
            # Find pass2 entries within output spread
            for _, row2 in pass2_df.iterrows():
                m2 = row2['M #']
                output2 = row2['Output']
                origin2 = row2['Origin']
                arrival2 = row2.get('Arrival', '')
                feed2 = row2.get('Feed', '')
                
                # Skip if same M# (shouldn't happen but safety check)
                if m1 == m2:
                    continue
                
                # Check output spread
                output_spread = abs(output1 - output2)
                if output_spread > max_output_spread:
                    continue
                
                # Calculate zone price (average)
                zone_price = (output1 + output2) / 2
                
                # Determine match type based on arrival dates
                match_type = 'Older'
                if arrival1 and arrival2:
                    try:
                        date1 = pd.to_datetime(arrival1).date() if not isinstance(arrival1, pd.Timestamp) else arrival1.date()
                        date2 = pd.to_datetime(arrival2).date() if not isinstance(arrival2, pd.Timestamp) else arrival2.date()
                        report_date = pd.to_datetime(arrival1).date()  # Use arrival1 as reference
                        
                        # Check if either is from today (most recent day)
                        if date1 == report_date or date2 == report_date:
                            match_type = 'Today'
                        elif date1 >= report_date - pd.Timedelta(days=2) or date2 >= report_date - pd.Timedelta(days=2):
                            match_type = 'Recent'
                    except:
                        pass
                
                # Create match entry
                matches.append({
                    'Arrival_Output': zone_price,
                    'Arrival_DateTime': arrival1,
                    'Arrival_Bracket': row1.get('Day', ''),
                    'Model': 'Cluster Match',
                    'Type': match_type,
                    'Category': f"M#{int(m1)} ↔ M#{int(m2)}",
                    'Origins': f"{origin1}, {origin2}",
                    'Feed': feed1,
                    'M_#s': f"{int(m1)}, {int(m2)}",
                    'Outputs': f"{output1:.2f}, {output2:.2f}",
                    'Prox': output_spread,
                    'Pattern_Type': 'Cluster',
                    'Group': 'N/A',
                    'Is_Match': 'Yes'
                })
        
        if not matches:
            return pd.DataFrame()
        
        # Convert to DataFrame
        matches_df = pd.DataFrame(matches)
        
        # Sort by output descending
        matches_df = matches_df.sort_values('Arrival_Output', ascending=False)
        
        return matches_df
        
    except Exception as e:
        st.error(f"Error in match_cluster_table_entries: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()
