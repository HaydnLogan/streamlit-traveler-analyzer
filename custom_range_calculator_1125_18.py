"""
v1125_18. UPDATED PAIR COUNTING LOGIC for Confluence column.

NEW MULTI-TIER PAIR COUNTING ALGORITHM:
- Groups matches by: Feed, Origin (from base M#), base M# value, and Arrival_DateTime
- Counts UNIQUE matching M#s for each grouping
- Example: Small Feed, Origin Kepler-62, M# -5, at 2025-11-23 18:00
  * If matching M#s are: -60, -50, -41, -40, +40 (5 unique matches)
  * Count each unique match's Group classification
  * If 3 matches are Group '1. SAA' and 2 are Group '4. AA':
    - Lines with '1. SAA' get labeled: "3 Pr SAA"
    - Lines with '4. AA' get labeled: "5 Pr AA" (total count)
- HIERARCHY RULE: Report highest Group with 2+ matches
  * If no Group has 2+ matches, all lines report at their own Group level
  * Minimum 2 lines required to qualify for a Group label

GROUP HIERARCHY (highest to lowest):
1. SAA (Same Anchor, Same)
2. STT (Same Trinidad/Tobago)
3. TA/AT (Trinidad/Tobago with Anchor)
4. AA (Anchors, varying)
5. oA (other to Anchor)
6. Ao (Anchor to other)
7. oo (other to other)

v1125_17. TWO ACTION ITEMS: Open/Zone/Confluence columns and multiple pairs detection.

1. ADDED THREE NEW COLUMNS (at beginning of matched tables):
   - 'Open' column: Marks the output closest to each feed's Open value
     * Finds which Arrival_Output is nearest to the feed's start/open
     * Marks it with "Open" text
     * One per feed (Small, Big)
   - 'Zone' column: Empty for now (placeholder for future use)
   - 'Confluence' column: Shows multiple pair information

2. CONFLUENCE DETECTION (multiple pairs):
   - Detects when multiple pairs share same:
     * Arrival_Output (e.g., 24404.915)
     * Arrival_DateTime (e.g., 2025-11-23 18:00)
     * Feed (Feed1)
   - Labels with number of pairs: "2 pr", "3 pr", "4 pr", etc.
   - Adds group classification:
     * "SAA" if all pairs are Same Anchor
     * "AA" if varying Anchors
     * Other group codes as appropriate
   - Example: "3 pr AA" = 3 pairs with varying Anchors
   - Applied as final step after matching completes

3. IMPLEMENTATION DETAILS:
   - feed_opens parameter added to match_cluster_table_entries()
   - feed_opens dictionary tracked in process_cluster_tables_two_pass()
   - Feed open values captured for each feed: {'Small': value, 'Big': value}
   - Open marker: Calculates distance_to_open for each feed's matches
   - Confluence: Groups by (Arrival_Output, Arrival_DateTime, Feed1)
   - Column order: Open, Zone, Confluence, then all other columns

4. CONFLUENCE GROUP LOGIC:
   - Single group: Uses that group code (e.g., "3 pr SAA")
   - Mixed groups:
     * All SAA → "X pr SAA"
     * All AA → "X pr AA"
     * Mixed Anchor-related (SAA, AA, TA, AT, oA, Ao) → "X pr AA"
     * Otherwise → "X pr"

v1125_16. SEVEN ACTION ITEMS: Tags/Families, column renames, Category, Pattern_Type, Group, timing, Excel export.

1. TAGS AND FAMILIES COLUMNS ADDED:
   - Added 'Tags' column: Shows Pass1 tag, then Pass2 tag
   - Added 'Families' column: Shows Pass1 family, then Pass2 family
   - Copied from prep table entries
   - Format: "tag1, tag2" and "family1, family2"

2. COLUMN RENAMES:
   - 'Category' → 'Match' (shows M# pair, e.g., "M#6 ↔ M#87")
   - 'Model' → 'Category' (shows table name + arrival order)

3. CATEGORY COLUMN UPDATED:
   - Now shows abbreviated table name + arrival order
   - Format: "Fogz PD", "Lrg Disc PD", "Recips PD"
   - PD = Premium before Discount (Pass1 < |40|, Pass2 >= |40|)
   - DP = Discount before Premium (Pass1 >= |40|, Pass2 < |40|)
   - DD = Both Discounts, PP = Both Premiums

4. PATTERN_TYPE COLUMN UPDATED:
   - Uses G.11 file patterns
   - X0: (30,50), (22,60), (14,68), (10,77), (6,87), (5,96), (3,103), (2,107), (1,111)
   - XD0: (27,54), (15,67)
   - X1: (36,43), (26,55)
   - XD1: (33,45)
   - X2: (39,41)
   - XD2: (38,42)
   - "undef" for undefined patterns
   - Handles both positive and negative M#s

5. GROUP COLUMN UPDATED:
   - Uses G.11 classification
   - Group 1 (1. SAA): Same Anchor (both origins same Anchor)
   - Group 2 (2. STT): Same Trinidad/Tobago
   - Group 3 (3. TA/AT): Trinidad/Tobago + Anchor
   - Group 4 (4. AA): Different Anchors
   - Group 5 (5. oA): Other + Anchor (later is Anchor)
   - Group 6 (6. Ao): Anchor + Other (earlier is Anchor)
   - Group 7 (7. oo): Neither Anchor nor TT

6. TIMING DISPLAY UPDATED (in swing tool):
   - Replaced "[OK]" with "⏱️"
   - Changed "All cluster tables generated" to "Timing Summary: All cluster tables generated"
   - Changed background from green (success) to orange (warning)

7. EXCEL EXPORT ADDED (in swing tool):
   - Download button: "📥 Download All Matched Tables (Excel)"
   - 4 tabs: FOGZ, Large Discounts, Recips PD, Combined
   - Combined tab: All tables sorted by Arrival_Output
   - Filename: cluster_tables_YYYYMMDD_HHMMSS.xlsx

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

def filter_measurement_by_m_numbers(measurement_df, m_number_set):
    """
    Filter measurement DataFrame to only include rows with M # in the given set.
    
    Parameters:
    - measurement_df: Full measurement DataFrame
    - m_number_set: Set of M # values to keep (e.g., {0, 1, -1, 2, -2})
    
    Returns:
    - Filtered DataFrame
    """
    if measurement_df is None or measurement_df.empty or not m_number_set:
        return measurement_df
    
    try:
        # Find M # column (case insensitive variations)
        m_num_col = next((c for c in ['M #', 'm #', 'M Number', 'm number'] if c in measurement_df.columns), None)
        if m_num_col is None:
            st.warning("⚠️ No 'M #' column found in measurement file")
            return measurement_df
        
        # Convert M # values to integers and filter
        filtered_df = measurement_df[
            measurement_df[m_num_col].apply(
                lambda x: int(float(x)) in m_number_set if pd.notna(x) else False
            )
        ].copy()
        
        return filtered_df
    except Exception as e:
        st.error(f"Error filtering measurement by M #: {e}")
        return measurement_df

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
                    'Range': f"{range_low:.1f}-{range_high:.1f}"
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
    max_output_spread=3.0,
    window_radius=150,
    allowed_origins=None,
    segment_size=None,
    combine_segments=True,
    feed_selection="Both feeds"
):
    """
    Two-pass processing for cluster tables (FOGZ, Large Discounts, Recips PD).
    
    Pass 1: Get M#s from first 2 most recent days, filtered by valid_list_pass1
    Pass 2: Get M#s from all data within scope, filtered by valid_list_pass2
    
    Uses window_radius around each feed's Open (Input @ start) to calculate raw M ranges.
    
    allowed_origins: Optional set of origin names to process (e.g., {'Spain', 'Saturn', 'Jupiter'})
                     If None, processes all origins.
    
    feed_selection: Which feed(s) to process: "Both feeds", "Small feed only", or "Big feed only"
    
    segment_size: If specified, breaks the window into segments of this size (e.g., 75 units)
                  This reduces memory usage and processing time for large datasets.
                  Example: window_radius=150 (300 unit range) with segment_size=75 creates 4 segments
    
    combine_segments: If True, combines all segment results into single dataframe
                      If False, returns list of dataframes (one per segment)
    
    Returns combined dataframe with all M#s from both passes, and processing summary.
    """
    if valid_list_pass1 is None:
        valid_list_pass1 = set()
    if valid_list_pass2 is None:
        valid_list_pass2 = set()
    
    try:
        # Diagnostic: Check what M #s exist in measurement file
        m_num_col = next((c for c in ['M #', 'm #', 'M Number', 'm number'] if c in measurement_df.columns), None)
        if m_num_col:
            available_m_nums = set()
            for val in measurement_df[m_num_col].dropna().unique():
                try:
                    available_m_nums.add(int(float(val)))
                except:
                    pass
            
            pass1_overlap = available_m_nums.intersection(valid_list_pass1)
            pass2_overlap = available_m_nums.intersection(valid_list_pass2)
            
            st.info(f"📊 Measurement file has {len(available_m_nums)} unique M #s")
            st.info(f"✓ Pass 1 M #s available: {sorted(list(pass1_overlap)) if pass1_overlap else 'None'}")
            st.info(f"✓ Pass 2 M #s available: {sorted(list(pass2_overlap)[:10]) if pass2_overlap else 'None'} {'...' if len(pass2_overlap) > 10 else ''}")
        else:
            st.error("❌ No 'M #' column found in measurement file!")
            return pd.DataFrame(), {'pass1_count': 0, 'pass2_count': 0, 'total_count': 0, 
                                   'pass1_m_numbers': [], 'pass2_m_numbers': [],
                                   'pass1_origins': [], 'pass2_origins': [],
                                   'pass1_feeds': [], 'pass2_feeds': []}
        
        # Collect feeds based on feed_selection
        # _collect_origins expects tuples of (df, name)
        hlc_df_list = []
        if feed_selection in ["Both feeds", "Small feed only"]:
            hlc_df_list.append((small_df, 'Small'))
        if feed_selection in ["Both feeds", "Big feed only"] and big_df is not None:
            hlc_df_list.append((big_df, 'Big'))
        
        if not hlc_df_list:
            st.warning("⚠️ No feeds selected for processing")
            return pd.DataFrame(), {'pass1_count': 0, 'pass2_count': 0, 'total_count': 0, 
                                   'pass1_m_numbers': [], 'pass2_m_numbers': [],
                                   'pass1_origins': [], 'pass2_origins': [],
                                   'pass1_feeds': [], 'pass2_feeds': []}
        
        st.info(f"📋 Processing {len(hlc_df_list)} feed(s): {', '.join([name for _, name in hlc_df_list])}")
        origins = _collect_origins(hlc_df_list)
        
        all_pass1_results = []
        all_pass2_results = []
        all_pass1_processing = []  # Track all HLCs examined
        all_pass2_processing = []  # Track all HLCs examined
        
        # Process each feed separately
        feed_opens = {}  # Track feed open values
        for hlc_df, feed_name in hlc_df_list:
            # Capture feed-level open values (Input @ start and Input @ report)
            start_anchor = _most_recent_sunday_anchor(report_time, day_start_hour)
            feed_start = get_open_at(hlc_df, start_anchor)
            feed_report = get_open_at(hlc_df, clean_timestamp(report_time))
            
            # Store feed open for later use
            feed_opens[feed_name] = feed_start
            
            # Calculate window range based on this feed's Open
            full_window_low = feed_start - window_radius
            full_window_high = feed_start + window_radius
            
            # Create segments if segmentation is enabled
            if segment_size is not None:
                # Break window into segments
                segments = []
                current_low = full_window_low
                segment_num = 1
                while current_low < full_window_high:
                    current_high = min(current_low + segment_size, full_window_high)
                    segments.append({
                        'num': segment_num,
                        'low': current_low,
                        'high': current_high,
                        'size': current_high - current_low
                    })
                    current_low = current_high
                    segment_num += 1
                
                st.info(f"🔷 {feed_name} Feed: Processing {len(segments)} segments (full window [{full_window_low:.2f}, {full_window_high:.2f}])")
                for seg in segments:
                    st.caption(f"  Segment {seg['num']}: [{seg['low']:.2f}, {seg['high']:.2f}] ({seg['size']:.1f} units)")
            else:
                # Single segment covering full window
                segments = [{'num': 1, 'low': full_window_low, 'high': full_window_high, 'size': window_radius * 2}]
                st.info(f"🧮 {feed_name} Feed Window: [{full_window_low:.2f}, {full_window_high:.2f}] around Open = {feed_start:.2f}")
            
            # Get origins from this specific feed
            feed_origins = []
            for col in hlc_df.columns:
                if col.endswith(" H"):
                    origin = col[:-2]
                    if origin not in feed_origins:
                        feed_origins.append(origin)
            
            # Show all origins found
            st.caption(f"📋 {feed_name} Feed: Found {len(feed_origins)} origins: {', '.join(sorted(feed_origins))}")
            
            # Apply origin filtering if specified
            if allowed_origins is not None:
                original_count = len(feed_origins)
                feed_origins = [o for o in feed_origins if o in allowed_origins]
                if not feed_origins:
                    st.warning(f"⚠️ {feed_name} Feed: No matching origins after filtering!")
                    st.caption(f"Available: {', '.join(sorted([o for o in hlc_df.columns if o.endswith(' H')]))}")
                    st.caption(f"Allowed: {', '.join(sorted(allowed_origins))}")
                    continue
                st.info(f"✓ {feed_name} Feed: Processing {len(feed_origins)}/{original_count} filtered origins: {', '.join(feed_origins)}")
            else:
                st.info(f"✓ {feed_name} Feed: Processing all {len(feed_origins)} origins")
            
            # Process each segment
            for segment in segments:
                window_low = segment['low']
                window_high = segment['high']
                
                if len(segments) > 1:
                    st.caption(f"▶️ Processing Segment {segment['num']}/{len(segments)}")
                
                for origin_name in feed_origins:
                    # Get HLC data changes from this feed
                    hlc_data = find_new_data_changes(hlc_df, report_time, origin_name, scope_days)
                
                    if not hlc_data:
                        continue
                    
                    # Sort by datetime descending to get most recent first
                    hlc_data_sorted = sorted(hlc_data, key=lambda x: x['datetime'], reverse=True)
                    
                    # Pre-filter measurement files for both passes
                    measurement_df_pass1 = filter_measurement_by_m_numbers(measurement_df, valid_list_pass1)
                    measurement_df_pass2 = filter_measurement_by_m_numbers(measurement_df, valid_list_pass2)
                    
                    # PASS 1: Get first 2 most recent days only
                    pass1_processing = []  # Track all HLCs examined in Pass 1
                    
                    if hlc_data_sorted and not measurement_df_pass1.empty:
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
                                # Attach feed-level opens
                                hlc_item['input_start'] = feed_start
                                hlc_item['input_report'] = feed_report
                                
                                # CRITICAL: Calculate raw M values using WINDOW range, not HLC range
                                calc = calculate_raw_m_values(hlc_item, window_low, window_high)
                                if not calc:
                                    continue
                                
                                # Merge hlc_item with calculated values
                                hlc_with_calc = {**hlc_item, **calc}
                                
                                # Find valid M values within this range (using filtered measurement file)
                                valid_results = find_valid_m_values(
                                    measurement_df_pass1,  # Use filtered measurement file
                                    raw_m_low=calc['raw_m_low'],  # Calculated using window range
                                    raw_m_high=calc['raw_m_high'],  # Calculated using window range
                                    hlc_data=hlc_with_calc,  # Pass merged dict (not list!)
                                    range_low=window_low,  # Use window range
                                    range_high=window_high,  # Use window range
                                    is_high_range=False,
                                    data_source=f"Pass1_{feed_name}_{origin_name}",
                                    report_time=report_time
                                )
                                
                                # Track processing (even if no results)
                                dt_str = hlc_item['datetime'].strftime('%Y-%m-%d %H:%M') if hasattr(hlc_item['datetime'], 'strftime') else str(hlc_item['datetime'])
                                pass1_processing.append({
                                    'Feed': feed_name,
                                    'Origin': origin_name,
                                    'DateTime': dt_str,
                                    'Date': hlc_item['datetime'].date() if hasattr(hlc_item['datetime'], 'date') else 'Unknown',
                                    'H': hlc_item['H'],
                                    'L': hlc_item['L'],
                                    'C': hlc_item['C'],
                                    'Raw M Low': calc['raw_m_low'],
                                    'Raw M High': calc['raw_m_high'],
                                    'Window Range': f"{window_low:.2f}-{window_high:.2f}",
                                    'Feed Open': feed_start,
                                    'Results Found': len(valid_results.get('valid_entries', []))
                                })
                                
                                # All results are already filtered by M # due to pre-filtering
                                if valid_results and valid_results.get('valid_entries'):
                                    for result in valid_results['valid_entries']:
                                        # Add feed identifier and pass info
                                        result['Feed'] = feed_name
                                        result['Pass'] = 'Pass1'
                                        all_pass1_results.append(result)
                    
                    # Collect Pass 1 processing details
                    all_pass1_processing.extend(pass1_processing)
                    
                    # PASS 2: Get all data within scope (no date restriction)
                    pass2_processing = []  # Track all HLCs examined in Pass 2
                    
                    if not measurement_df_pass2.empty:
                        for hlc_item in hlc_data_sorted:
                            # Attach feed-level opens
                            hlc_item['input_start'] = feed_start
                            hlc_item['input_report'] = feed_report
                            
                            # CRITICAL: Calculate raw M values using WINDOW range, not HLC range
                            calc = calculate_raw_m_values(hlc_item, window_low, window_high)
                            if not calc:
                                continue
                            
                            # Merge hlc_item with calculated values
                            hlc_with_calc = {**hlc_item, **calc}
                            
                            # Find valid M values within this range (using filtered measurement file)
                            valid_results = find_valid_m_values(
                                measurement_df_pass2,  # Use filtered measurement file
                                raw_m_low=calc['raw_m_low'],  # Calculated using window range
                                raw_m_high=calc['raw_m_high'],  # Calculated using window range
                                hlc_data=hlc_with_calc,  # Pass merged dict (not list!)
                                range_low=window_low,  # Use window range
                                range_high=window_high,  # Use window range
                                is_high_range=False,
                                data_source=f"Pass2_{feed_name}_{origin_name}",
                                report_time=report_time
                            )
                            
                            # Track processing (even if no results)
                            dt_str = hlc_item['datetime'].strftime('%Y-%m-%d %H:%M') if hasattr(hlc_item['datetime'], 'strftime') else str(hlc_item['datetime'])
                            pass2_processing.append({
                                'Feed': feed_name,
                                'Origin': origin_name,
                                'DateTime': dt_str,
                                'Date': hlc_item['datetime'].date() if hasattr(hlc_item['datetime'], 'date') else 'Unknown',
                                'H': hlc_item['H'],
                                'L': hlc_item['L'],
                                'C': hlc_item['C'],
                                'Raw M Low': calc['raw_m_low'],
                                'Raw M High': calc['raw_m_high'],
                                'Window Range': f"{window_low:.2f}-{window_high:.2f}",
                                'Feed Open': feed_start,
                                'Results Found': len(valid_results.get('valid_entries', []))
                            })
                            
                            # All results are already filtered by M # due to pre-filtering
                            if valid_results and valid_results.get('valid_entries'):
                                for result in valid_results['valid_entries']:
                                    # Add feed identifier and pass info
                                    result['Feed'] = feed_name
                                    result['Pass'] = 'Pass2'
                                    all_pass2_results.append(result)
                    
                    # Collect Pass 2 processing details
                    all_pass2_processing.extend(pass2_processing)
        
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
            'pass2_feeds': sorted(set([r['Feed'] for r in all_pass2_results if r.get('Feed')])),
            # NEW: Detailed processing information
            'pass1_hlcs_examined': len(all_pass1_processing),
            'pass2_hlcs_examined': len(all_pass2_processing),
            'pass1_processing_details': pd.DataFrame(all_pass1_processing) if all_pass1_processing else pd.DataFrame(),
            'pass2_processing_details': pd.DataFrame(all_pass2_processing) if all_pass2_processing else pd.DataFrame(),
            'pass1_dates': sorted(set([p['Date'] for p in all_pass1_processing if 'Date' in p])),
            'pass2_dates': sorted(set([p['Date'] for p in all_pass2_processing if 'Date' in p])),
            'feed_opens': feed_opens  # NEW: Feed open values
        }
        
        if not combined_results:
            return pd.DataFrame(), processing_summary
        
        # Convert to DataFrame
        prep_df = pd.DataFrame(combined_results)
        
        # Remove duplicates (same M#, Origin, Arrival, Feed)
        if not prep_df.empty:
            prep_df = prep_df.drop_duplicates(subset=['M #', 'Origin', 'Arrival', 'Feed'], keep='first')
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
            'pass1_feeds': [], 'pass2_feeds': [],
            'pass1_hlcs_examined': 0, 'pass2_hlcs_examined': 0,
            'pass1_processing_details': pd.DataFrame(),
            'pass2_processing_details': pd.DataFrame(),
            'pass1_dates': [], 'pass2_dates': []
        }
        return pd.DataFrame(), empty_summary


def match_cluster_table_entries(prep_df, valid_list_pass1, valid_list_pass2, max_output_spread=3.0, feed_filter=None, measurement_df=None, check_recip=False, allow_mixed_feed=True, table_name="Cluster", feed_opens=None):
    """
    Find matches between Pass1 M#s (recent 2 days) and Pass2 M#s (within scope).
    
    Parameters:
    - prep_df: Combined dataframe from both passes
    - valid_list_pass1: Set of M#s from pass 1 (recent data)
    - valid_list_pass2: Set of M#s from pass 2 (all data)
    - max_output_spread: Maximum allowed output difference
    - feed_filter: Optional feed filter ('Small' or 'Big')
    - feed_opens: Dict of feed open values {'Small': value, 'Big': value}
    - measurement_df: Measurement dataframe (required if check_recip=True)
    - check_recip: If True, only match M#s that are reciprocals based on 'R #' column
    - allow_mixed_feed: If False, only match entries from the same feed
    
    Returns:
    - DataFrame with matched cluster table entries in G.11 format
    """
    if prep_df.empty:
        return pd.DataFrame()
    
    # Build reciprocal lookup if needed
    recip_lookup = {}
    if check_recip and measurement_df is not None:
        try:
            # Find the M# and Recip R# columns (case-insensitive)
            m_col = next((c for c in measurement_df.columns if c.lower().replace(' ', '') in ['m#', 'm', 'mnumber']), None)
            # Check for various reciprocal column names: "R #", "Recip R #", "Reciprocal", etc.
            recip_col = next((c for c in measurement_df.columns if c.lower().replace(' ', '') in ['r#', 'r', 'recipr#', 'recipr', 'reciprocal', 'reciprocal#']), None)
            
            if m_col and recip_col:
                st.info(f"🔍 Building reciprocal lookup from columns: '{m_col}' → '{recip_col}'")
                for _, row in measurement_df.iterrows():
                    m_num = row[m_col]
                    recip_num = row[recip_col]
                    if pd.notna(m_num) and pd.notna(recip_num):
                        try:
                            m_int = int(float(m_num))
                            r_int = int(float(recip_num))
                            recip_lookup[m_int] = r_int
                        except:
                            pass
                
                st.success(f"✓ Built reciprocal lookup with {len(recip_lookup)} entries")
                # Show some examples
                example_pairs = [(k, v) for k, v in list(recip_lookup.items())[:5]]
                st.caption(f"Examples: {example_pairs}")
            else:
                st.warning(f"⚠️ Could not find reciprocal columns. M# column: {m_col}, Recip column: {recip_col}")
                st.caption(f"Available columns: {list(measurement_df.columns)}")
        except Exception as e:
            st.error(f"❌ Error building reciprocal lookup: {e}")
            import traceback
            st.code(traceback.format_exc())
    
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
                
                # Check feed matching if mixed feed not allowed
                if not allow_mixed_feed:
                    if feed1 != feed2:
                        continue  # Skip mixed feed matches
                
                # Check reciprocal relationship if required
                if check_recip and recip_lookup:
                    # m2 must be the reciprocal of m1 OR m1 must be the reciprocal of m2
                    m1_int = int(float(m1))
                    m2_int = int(float(m2))
                    expected_recip_of_m1 = recip_lookup.get(m1_int)
                    expected_recip_of_m2 = recip_lookup.get(m2_int)
                    
                    # Also check absolute values (since 6 and -87 are reciprocals if 6 and 87 are)
                    m1_abs = abs(m1_int)
                    m2_abs = abs(m2_int)
                    expected_recip_of_m1_abs = recip_lookup.get(m1_abs)
                    expected_recip_of_m2_abs = recip_lookup.get(m2_abs)
                    
                    # Check if they're reciprocals of each other (exact or absolute values)
                    is_recip = (expected_recip_of_m1 == m2_int or 
                               expected_recip_of_m2 == m1_int or
                               expected_recip_of_m1_abs == m2_abs or
                               expected_recip_of_m2_abs == m1_abs)
                    
                    if not is_recip:
                        continue  # Not reciprocals, skip this pair
                
                # Check output spread
                output_spread = abs(output1 - output2)
                if output_spread > max_output_spread:
                    continue
                
                # Use the Pass1 output as the arrival output (not average)
                arrival_output = output1
                
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
                # Combine feeds if different
                if feed1 == feed2:
                    feed_display = feed1
                else:
                    feed_display = f"{feed1} + {feed2}"
                
                # Get Tag and Family from both entries (Pass1 first, then Pass2)
                tag1 = row1.get('Tag', '')
                tag2 = row2.get('Tag', '')
                family1 = row1.get('Family', '')
                family2 = row2.get('Family', '')
                
                # Determine PD or DP based on M# absolute values
                # Pass1 M#s < |40| are Discounts (D), Pass2 M#s >= |40| are Premiums (P)
                m1_abs = abs(int(float(m1)))
                m2_abs = abs(int(float(m2)))
                
                if m1_abs < 40 and m2_abs >= 40:
                    arrival_order = "PD"  # Premium first, then Discount
                elif m1_abs >= 40 and m2_abs < 40:
                    arrival_order = "DP"  # Discount first, then Premium
                elif m1_abs < 40 and m2_abs < 40:
                    arrival_order = "DD"  # Both Discounts
                else:
                    arrival_order = "PP"  # Both Premiums
                
                # Determine table-specific category using table name and arrival order
                # e.g., "Fogz PD", "Recips DP", "Lrg Disc PD", etc.
                table_category = f"{table_name} {arrival_order}"
                
                # Determine Pattern_Type using G.11 classification
                # Based on M# pairs and relationships
                pattern_type = "undef"  # Default
                
                # Check for known patterns from G.11
                m1_val = int(float(m1))
                m2_val = int(float(m2))
                
                # X0 patterns: (30,50), (22,60), (14,68), (10,77), (6,87), (5,96), (3,103), (2,107), (1,111)
                x0_pairs = [(30,50), (22,60), (14,68), (10,77), (6,87), (5,96), (3,103), (2,107), (1,111),
                           (-30,-50), (-22,-60), (-14,-68), (-10,-77), (-6,-87), (-5,-96), (-3,-103), (-2,-107), (-1,-111)]
                # XD0 patterns: (27,54), (15,67)
                xd0_pairs = [(27,54), (15,67), (-27,-54), (-15,-67)]
                # X1 patterns: (36,43), (26,55)
                x1_pairs = [(36,43), (26,55), (-36,-43), (-26,-55)]
                # XD1 patterns: (33,45)
                xd1_pairs = [(33,45), (-33,-45)]
                # X2 patterns: (39,41)
                x2_pairs = [(39,41), (-39,-41)]
                # XD2 patterns: (38,42)
                xd2_pairs = [(38,42), (-38,-42)]
                
                pair = (m1_val, m2_val)
                pair_abs = (abs(m1_val), abs(m2_val))
                
                if pair in x0_pairs or pair_abs in x0_pairs or (pair[1], pair[0]) in x0_pairs or (pair_abs[1], pair_abs[0]) in x0_pairs:
                    pattern_type = "X0"
                elif pair in xd0_pairs or pair_abs in xd0_pairs or (pair[1], pair[0]) in xd0_pairs or (pair_abs[1], pair_abs[0]) in xd0_pairs:
                    pattern_type = "XD0"
                elif pair in x1_pairs or pair_abs in x1_pairs or (pair[1], pair[0]) in x1_pairs or (pair_abs[1], pair_abs[0]) in x1_pairs:
                    pattern_type = "X1"
                elif pair in xd1_pairs or pair_abs in xd1_pairs or (pair[1], pair[0]) in xd1_pairs or (pair_abs[1], pair_abs[0]) in xd1_pairs:
                    pattern_type = "XD1"
                elif pair in x2_pairs or pair_abs in x2_pairs or (pair[1], pair[0]) in x2_pairs or (pair_abs[1], pair_abs[0]) in x2_pairs:
                    pattern_type = "X2"
                elif pair in xd2_pairs or pair_abs in xd2_pairs or (pair[1], pair[0]) in xd2_pairs or (pair_abs[1], pair_abs[0]) in xd2_pairs:
                    pattern_type = "XD2"
                
                # Determine Group using G.11 classification
                # Anchor origins: Spain, Saturn, Jupiter, Kepler-62, Kepler-44
                ANCHOR_ORIGINS = {"spain", "saturn", "jupiter", "kepler-62", "kepler-44"}
                # Trinidad/Tobago
                TT_ORIGINS = {"trinidad", "tobago"}
                
                origin1_lower = origin1.lower()
                origin2_lower = origin2.lower()
                
                earlier_is_anchor = origin1_lower in ANCHOR_ORIGINS
                later_is_anchor = origin2_lower in ANCHOR_ORIGINS
                earlier_is_tt = origin1_lower in TT_ORIGINS
                later_is_tt = origin2_lower in TT_ORIGINS
                
                # Group 1: Both Anchors, same Anchor
                if earlier_is_anchor and later_is_anchor and origin1_lower == origin2_lower:
                    group = "1. SAA"
                # Group 2: Both Trinidad/Tobago
                elif earlier_is_tt and later_is_tt:
                    group = "2. STT"
                # Group 3: One Trinidad/Tobago, one Anchor
                elif earlier_is_tt and later_is_anchor:
                    group = "3. TA"
                elif earlier_is_anchor and later_is_tt:
                    group = "3. AT"
                # Group 4: Both Anchors, different Anchors
                elif earlier_is_anchor and later_is_anchor and origin1_lower != origin2_lower:
                    group = "4. AA"
                # Group 5: Later is Anchor, earlier is not
                elif later_is_anchor and not earlier_is_anchor:
                    group = "5. oA"
                # Group 6: Earlier is Anchor, later is not
                elif earlier_is_anchor and not later_is_anchor:
                    group = "6. Ao"
                else:
                    group = "7. oo"  # Neither is Anchor or TT
                
                matches.append({
                    'Arrival_Output': arrival_output,
                    'Arrival_DateTime': arrival1,
                    'Arrival_Bracket': row1.get('Day', ''),
                    'Category': table_category,  # Will be updated by caller
                    'Type': match_type,
                    'Match': f"M#{int(m1)} ↔ M#{int(m2)}",
                    'Origins': f"{origin1}, {origin2}",
                    'Tags': f"{tag1}, {tag2}",
                    'Families': f"{family1}, {family2}",
                    'Feed': feed_display,
                    'Feed1': feed1,
                    'Feed2': feed2,
                    'M_#s': f"{int(m1)}, {int(m2)}",
                    'Outputs': f"{output1:.2f}, {output2:.2f}",
                    'Prox': output_spread,
                    'Pattern_Type': pattern_type,
                    'Group': group,
                    'Arrival_Order': arrival_order,  # PD, DP, DD, PP
                    'Is_Match': 'Yes'
                })
        
        if not matches:
            if check_recip:
                st.info("ℹ️ No reciprocal pairs found matching criteria")
            return pd.DataFrame()
        
        # Convert to DataFrame
        matches_df = pd.DataFrame(matches)
        
        # ACTION ITEM 2: Detect multiple pairs (confluence) - UPDATED LOGIC v18
        # Group by Feed, Origin (from first M#), base M# value, and Arrival_DateTime
        # to count unique matches per grouping
        matches_df['Confluence'] = ''
        
        if not matches_df.empty:
            # Extract the base M# (first M# in the M_#s field)
            def extract_base_m(m_str):
                """Extract first M# from string like '-5, -60'"""
                try:
                    return int(m_str.split(',')[0].strip())
                except:
                    return None
            
            # Extract the matching M# (second M# in the M_#s field)
            def extract_match_m(m_str):
                """Extract second M# from string like '-5, -60'"""
                try:
                    return int(m_str.split(',')[1].strip())
                except:
                    return None
            
            # Extract first origin from Origins field
            def extract_base_origin(origins_str):
                """Extract first origin from string like 'Kepler-62, Kepler-62'"""
                try:
                    return origins_str.split(',')[0].strip()
                except:
                    return ''
            
            matches_df['_base_m'] = matches_df['M_#s'].apply(extract_base_m)
            matches_df['_match_m'] = matches_df['M_#s'].apply(extract_match_m)
            matches_df['_base_origin'] = matches_df['Origins'].apply(extract_base_origin)
            
            # Group by Feed1, base origin, base M#, and Arrival_DateTime
            grouped = matches_df.groupby(['Feed1', '_base_origin', '_base_m', 'Arrival_DateTime'])
            
            for (feed, base_origin, base_m, dt), group_df in grouped:
                if len(group_df) > 1:
                    # Multiple pairs found - count UNIQUE matching M#s
                    unique_match_ms = group_df['_match_m'].unique()
                    num_unique_matches = len(unique_match_ms)
                    
                    # Count how many of each Group classification among the unique matches
                    # Build a dict: {group_code: count}
                    group_counts = {}
                    
                    for match_m in unique_match_ms:
                        # Find the row(s) with this match_m in this group
                        match_rows = group_df[group_df['_match_m'] == match_m]
                        if not match_rows.empty:
                            # Get the Group from this match (take first if multiple)
                            group_full = match_rows.iloc[0]['Group']
                            group_code = group_full.split('.')[1].strip() if '.' in group_full else group_full
                            group_counts[group_code] = group_counts.get(group_code, 0) + 1
                    
                    # Define group hierarchy (1. SAA is highest, 7. oo is lowest)
                    group_hierarchy = ['SAA', 'STT', 'TA', 'AT', 'AA', 'oA', 'Ao', 'oo']
                    
                    # Check if the TOP group in hierarchy (SAA) has at least 2 matches
                    # Only SAA can truly "win" as the highest ranked group
                    top_group = group_hierarchy[0]  # 'SAA'
                    top_group_count = group_counts.get(top_group, 0)
                    
                    if top_group_count >= 2:
                        # SAA (top group) qualifies! Apply split labeling
                        # Lines with SAA: get SAA count
                        # Lines with other groups: get TOTAL count at their level
                        for idx in group_df.index:
                            row_group_full = matches_df.loc[idx, 'Group']
                            row_group_code = row_group_full.split('.')[1].strip() if '.' in row_group_full else row_group_full
                            
                            if row_group_code == top_group:
                                # This line has the winning top group
                                matches_df.loc[idx, 'Confluence'] = f"{top_group_count} Pr {top_group}"
                            else:
                                # This line has a different group - report TOTAL at their level
                                matches_df.loc[idx, 'Confluence'] = f"{num_unique_matches} Pr {row_group_code}"
                    else:
                        # Top group (SAA) doesn't qualify (< 2 matches)
                        # Find ANY group with ≥2 matches for uniform reporting
                        fallback_group = None
                        for group_code in group_hierarchy:
                            if group_counts.get(group_code, 0) >= 2:
                                fallback_group = group_code
                                break
                        
                        if fallback_group:
                            # Found a group with ≥2 - ALL lines report uniformly at this level
                            for idx in group_df.index:
                                matches_df.loc[idx, 'Confluence'] = f"{num_unique_matches} Pr {fallback_group}"
                        else:
                            # No group has ≥2 at all - find first group with any matches
                            fallback_group = None
                            for group_code in group_hierarchy:
                                if group_counts.get(group_code, 0) >= 1:
                                    fallback_group = group_code
                                    break
                            
                            if not fallback_group:
                                fallback_group = 'AA'  # Ultimate fallback
                            
                            # ALL lines report at this fallback level
                            for idx in group_df.index:
                                matches_df.loc[idx, 'Confluence'] = f"{num_unique_matches} Pr {fallback_group}"
            
            # Clean up temporary columns
            matches_df.drop(columns=['_base_m', '_match_m', '_base_origin'], inplace=True)
        
        # ACTION ITEM 1: Add Open, Zone columns at the beginning
        matches_df['Open'] = ''
        matches_df['Zone'] = ''
        
        # Mark outputs closest to feed opens
        if feed_opens and not matches_df.empty:
            for feed_name, feed_open in feed_opens.items():
                # Find matches from this feed
                feed_matches = matches_df[matches_df['Feed1'] == feed_name]
                
                if not feed_matches.empty:
                    # Find the output closest to feed open
                    feed_matches_copy = feed_matches.copy()
                    feed_matches_copy['distance_to_open'] = abs(feed_matches_copy['Arrival_Output'] - feed_open)
                    
                    # Get the index of the closest output
                    closest_idx = feed_matches_copy['distance_to_open'].idxmin()
                    
                    # Mark it as Open
                    matches_df.loc[closest_idx, 'Open'] = 'Open'
        
        # Show matched pairs if reciprocal checking was done
        if check_recip:
            unique_pairs = set()
            for _, row in matches_df.iterrows():
                m_nums = row['M_#s']
                unique_pairs.add(m_nums)
            st.success(f"✓ Found {len(unique_pairs)} unique reciprocal pairs")
            st.caption(f"Pairs: {', '.join(sorted(unique_pairs))}")
        
        # Reorder columns to put Open, Zone, Confluence first
        cols = ['Open', 'Zone', 'Confluence'] + [col for col in matches_df.columns if col not in ['Open', 'Zone', 'Confluence']]
        matches_df = matches_df[cols]
        
        # Sort by output descending
        matches_df = matches_df.sort_values('Arrival_Output', ascending=False)
        
        return matches_df
        
    except Exception as e:
        st.error(f"Error in match_cluster_table_entries: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()
