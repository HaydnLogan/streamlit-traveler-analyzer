# models_g_updated.py
# ---------------------------------------------------------------------
# Model G detection utilities & UI
# Adds G.08 (x0Pd.w descending) with [0]/[≠0] buckets and
# optional enhancements:
# - robust Arrival parsing
# - day-start aware "today" detection (default 18:00)
# - per-bucket toggles & expanders
# - summary metrics and results table
# ---------------------------------------------------------------------

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# Import origin/group constants from your helpers
# (lower-case origin names in sets)
from a_helpers import (
    EPIC_ORIGINS,
    ANCHOR_ORIGINS,
    GROUP_1B_TRAVELERS,
)

# ==========================================================
# Config / constants
# ==========================================================

# x0Pd.w descending (positive and negative sides)
_G08_POS_TRACK = [111, 107, 103, 96, 87, 77, 68, 60, 50, 40, 39, 36, 30, 22, 14, 10, 6, 5, 3, 2, 1, 0]
_G08_NEG_TRACK = (
    [-111, -107, -103, -96, -87, -77, -68, -60, -50, -40, -39, -36, -30, -22, -14, -10, -6, -5, -3, -2, -1]
    + [0]
)

_G08_TRACKS = {1: _G08_POS_TRACK, -1: _G08_NEG_TRACK}
_G08_INDEX = {1: {v: i for i, v in enumerate(_G08_POS_TRACK)},
              -1: {v: i for i, v in enumerate(_G08_NEG_TRACK)}}

# Category labels for G.08
_G08_CAT_ANCHOR = "G.08.01"  # ends in Anchor
_G08_CAT_EPIC   = "G.08.02"  # ends in Epic
_G08_CAT_BOTH   = "G.08.03"  # ends in both Anchor & Epic (same end timestamp)


# ==========================================================
# Parsing / helpers
# ==========================================================

def _parse_arrival_to_ts(val) -> pd.Timestamp:
    """
    Parse various Arrival formats to pandas Timestamp (naive).
    Expected common app format: 'mm/dd/YYYY HH:MM' string,
    or an existing Timestamp/datetime.
    """
    if isinstance(val, pd.Timestamp):
        return val.tz_localize(None) if getattr(val, "tz", None) is not None else val
    if isinstance(val, datetime):
        return pd.Timestamp(val).tz_localize(None)
    try:
        # Try common format first for speed/intent
        ts = pd.to_datetime(val, errors="coerce", infer_datetime_format=True)
        if isinstance(ts, pd.Timestamp):
            return ts.tz_localize(None) if getattr(ts, "tz", None) is not None else ts
        return pd.NaT
    except Exception:
        return pd.NaT


def _origin_kind(origin_text: str) -> str:
    """
    Classify origin as 'Epic', 'Anchor', or 'Other'
    using your EPIC_ORIGINS / ANCHOR_ORIGINS sets (lower-case names).
    """
    o = str(origin_text).strip().lower()
    # match by containment for variants like "wasp-12b (Special)"
    if any(ep in o for ep in EPIC_ORIGINS):
        return "Epic"
    if any(an in o for an in ANCHOR_ORIGINS):
        return "Anchor"
    return "Other"


def _final_arrival_ts(sequence_rows: list[dict]) -> pd.Timestamp | None:
    """Return the maximum (latest) parsed Arrival in a sequence."""
    if not sequence_rows:
        return None
    times = [_parse_arrival_to_ts(x.get("Arrival")) for x in sequence_rows]
    times = [t for t in times if pd.notna(t)]
    return max(times) if times else None


def _is_today_sequence(sequence_rows: list[dict],
                       report_time: datetime | pd.Timestamp | None,
                       start_hour: int = 18) -> bool:
    """
    True if the sequence ends on [0] (i.e., same "report day" per start_hour).
    Priority 1: use 'Day' value on last item if present (e.g., '[0]').
    Priority 2: compute from final Arrival vs the 'report day start'.
    """
    if not sequence_rows:
        return False

    # 1) Try explicit Day label on last chronological item
    seq = sorted(sequence_rows, key=lambda x: _parse_arrival_to_ts(x.get("Arrival")))
    last = seq[-1]

    day_str = str(last.get("Day", "")).strip().lower()
    if day_str in ("[0]", "0", "[0] today"):
        return True
    if day_str.startswith("[") and day_str.endswith("]"):
        try:
            return int(day_str[1:-1]) == 0
        except Exception:
            pass

    # 2) Compute fallback
    end_ts = _final_arrival_ts(sequence_rows)
    if end_ts is None or report_time is None:
        return False

    rpt = pd.to_datetime(report_time)
    rpt = rpt.tz_localize(None) if getattr(rpt, "tz", None) is not None else rpt

    day_start = rpt.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    if rpt.hour < start_hour:
        day_start -= pd.Timedelta(days=1)

    days_diff = int((end_ts - day_start) // pd.Timedelta(days=1))
    return days_diff == 0


def _no_interior_duplicate_arrivals(sequence_rows: list[dict]) -> bool:
    """
    Ensure no duplicate Arrival timestamps in the *interior* of the sequence.
    (Duplicates at the *end* are allowed for 'Both' classification check.)
    """
    if len(sequence_rows) <= 2:
        return True
    # by time order
    seq = sorted(sequence_rows, key=lambda x: _parse_arrival_to_ts(x.get("Arrival")))
    *interior, last = seq
    interior_ts = [_parse_arrival_to_ts(x.get("Arrival")) for x in interior]
    counts = pd.Series(interior_ts).value_counts(dropna=False)
    return (counts <= 1).all()


def _g08_end_category(sequence_rows: list[dict], df_all: pd.DataFrame) -> str | None:
    """
    Determine G.08 category based on the *ending* Arrival timestamp.
    G.08.01 = ends in Anchor
    G.08.02 = ends in Epic
    G.08.03 = ends in BOTH (two rows at end timestamp: one Anchor, one Epic)
    """
    if not sequence_rows:
        return None

    # End timestamp of the sequence
    end_ts = _final_arrival_ts(sequence_rows)
    if end_ts is None:
        return None

    # Find all rows in the entire dataset with this same end timestamp
    # (Arrival may be string; ensure consistent parsing)
    if "Arrival_datetime" in df_all.columns:
        same_end = df_all[pd.to_datetime(df_all["Arrival_datetime"], errors="coerce").dt.floor("min") ==
                          pd.to_datetime(end_ts).floor("min")]
    else:
        same_end = df_all[_parse_arrival_to_ts(df_all["Arrival"]) == end_ts]

    # Determine classes present at the end
    classes = set(_origin_kind(x) for x in same_end["Origin"].tolist())
    # If we couldn't find via joining, at least check the last row in the sequence
    if not classes:
        classes.add(_origin_kind(sequence_rows[-1].get("Origin", "")))

    has_anchor = "Anchor" in classes
    has_epic = "Epic" in classes

    if has_anchor and has_epic:
        return _G08_CAT_BOTH
    if has_anchor:
        return _G08_CAT_ANCHOR
    if has_epic:
        return _G08_CAT_EPIC
    return None


def _prep_df_for_g08(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanity-clean the input df; ensure needed columns exist and have usable types.
    Returns a filtered dataframe with added 'Arrival_ts' column.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df2 = df.copy()

    # Normalize core columns
    # M # as float numeric
    if "M #" not in df2.columns:
        return pd.DataFrame()
    df2["M #"] = pd.to_numeric(df2["M #"], errors="coerce")

    # Arrival usable timestamp
    if "Arrival_datetime" in df2.columns:
        df2["Arrival_ts"] = pd.to_datetime(df2["Arrival_datetime"], errors="coerce")
    else:
        df2["Arrival_ts"] = df2["Arrival"].apply(_parse_arrival_to_ts) if "Arrival" in df2.columns else pd.NaT
    df2["Arrival_ts"] = df2["Arrival_ts"].dt.tz_localize(None)

    # Drop missing core values
    df2 = df2.dropna(subset=["M #", "Arrival_ts"])

    # Group 1b filter (as specified)
    df2 = df2[df2["M #"].isin(GROUP_1B_TRAVELERS)]

    # Limit to values on either positive or negative track
    allowed = set(_G08_POS_TRACK) | set(_G08_NEG_TRACK)
    df2 = df2[df2["M #"].isin(allowed)]

    # Keep essential columns only (avoid bloating sequence dicts)
    keep_cols = [c for c in df2.columns if c in (
        "Feed", "ddd", "Arrival", "Arrival_datetime", "Arrival_ts", "Day", "Origin",
        "M Name", "M #", "R #", "Tag", "Family", "Output"
    )]
    return df2[keep_cols].sort_values(["Arrival_ts"]).reset_index(drop=True)


def _build_g08_sequences(df: pd.DataFrame) -> list[list[dict]]:
    """
    Greedy scan to build strictly descending (in track order) sequences with
    strictly increasing Arrival times. Min length = 3.
    We build positive-side and negative-side sequences independently.

    Note: duplicates at the *final* timestamp aren't included in the path; they
    still get recognized for G.08.03 via end-of-sequence classification.
    """
    sequences = []

    for sign in (1, -1):
        track = _G08_TRACKS[sign]
        index_map = _G08_INDEX[sign]

        side_df = df[df["M #"].isin(track)].copy()
        if side_df.empty:
            continue

        # index on the track
        side_df["t_idx"] = side_df["M #"].map(index_map).astype(int)

        # Sort by time, then by track index (just to stabilize)
        side_df = side_df.sort_values(["Arrival_ts", "t_idx"]).reset_index(drop=True)

        # Greedy collector
        curr_seq = []
        last_idx = -1
        last_time = pd.Timestamp.min

        for _, row in side_df.iterrows():
            t_idx = int(row["t_idx"])
            t_arr = row["Arrival_ts"]

            # must be strictly later in time and strictly further down the track
            if t_idx > last_idx and t_arr > last_time:
                curr_seq.append(row.to_dict())
                last_idx = t_idx
                last_time = t_arr
            else:
                # sequence break → save if long enough
                if len(curr_seq) >= 3:
                    sequences.append(curr_seq)
                # start a new one with this row
                curr_seq = [row.to_dict()]
                last_idx = t_idx
                last_time = t_arr

        # flush tail
        if len(curr_seq) >= 3:
            sequences.append(curr_seq)

    return sequences


def _sequence_summary_fields(sequence: list[dict]) -> dict:
    """Return a few consistent summary fields for a sequence row used in the results table."""
    arrs = [x.get("Arrival") for x in sequence]
    arr_ts = [_parse_arrival_to_ts(a) for a in arrs]
    start_ts = min(arr_ts) if arr_ts else None
    end_ts = max(arr_ts) if arr_ts else None

    m_path = " → ".join(str(int(round(x.get("M #", 0)))) for x in sequence)
    outputs = [x.get("Output") for x in sequence if pd.notna(x.get("Output"))]
    if outputs:
        out_min, out_max = (np.nanmin(outputs), np.nanmax(outputs))
        out_rng = f"{out_min:.2f} – {out_max:.2f}"
    else:
        out_rng = ""

    return {
        "length": len(sequence),
        "start_arrival": start_ts.strftime("%m/%d/%Y %H:%M") if pd.notna(start_ts) else "",
        "end_arrival": end_ts.strftime("%m/%d/%Y %H:%M") if pd.notna(end_ts) else "",
        "m_path": m_path,
        "output_range": out_rng,
    }


def _results_table(enabled_sequences: list[dict]) -> pd.DataFrame:
    """Flatten sequences into a concise DataFrame for display/export."""
    rows = []
    for s in enabled_sequences:
        base = {
            "Category": s.get("category", ""),
            "Length": s.get("length", 0),
            "End Arrival": s.get("end_arrival", ""),
            "End Day": s.get("end_day", ""),
            "Ends In": s.get("ends_in", ""),  # Anchor/Epic/Both
            "Track Side": s.get("track_side", ""),
            "M Path": s.get("m_path", ""),
            "Output Range": s.get("output_range", ""),
        }
        rows.append(base)
    df = pd.DataFrame(rows)
    # Nice default sort: newest end first, then category
    with pd.option_context("mode.use_inf_as_na", True):
        try:
            df["_sort_end"] = pd.to_datetime(df["End Arrival"], errors="coerce")
            df = df.sort_values(["_sort_end", "Category"], ascending=[False, True]).drop(columns=["_sort_end"])
        except Exception:
            pass
    return df


def display_sequence_details(seq_info: dict):
    """Streamlit expander: show the raw steps for a sequence."""
    seq = seq_info.get("sequence", [])
    if not seq:
        st.info("No details to display.")
        return
    df = pd.DataFrame(seq)
    # Order columns for readability if present
    preferred = ["Arrival", "Day", "Origin", "M #", "M Name", "Output", "Feed", "R #", "Tag", "Family"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    st.dataframe(df[cols], use_container_width=True)


# ==========================================================
# Main entry point
# ==========================================================

def run_model_g_detection(df: pd.DataFrame,
                          report_time: datetime | pd.Timestamp | None,
                          key_suffix: str = "") -> dict:
    """
    Model G detection entry point. Currently implements G.08:
      x0Pd.w descending sequences, min length=3, using Grp 1b data.
    Produces [0] vs [≠0] buckets based on end timestamp relative to report day.

    Returns:
        dict(success: bool, summary: dict, results_df: DataFrame)
    """
    try:
        st.markdown("## 🟢 Model G Detection")

        if df is None or df.empty:
            st.warning("No data for Model G.")
            return {"success": True, "summary": {}, "results_df": pd.DataFrame()}

        # Day start (UI or default)
        col_a, col_b = st.columns(2)
        with col_a:
            start_hour = st.selectbox("Day start hour (for [0]/[≠0] calc)", [18, 17], index=0,
                                      key=f"g_daystart{key_suffix}")
        with col_b:
            min_len = st.slider("Minimum sequence length", 3, 8, 3, 1, key=f"g_minlen{key_suffix}")

        # Prepare data
        work = _prep_df_for_g08(df)
        if work.empty:
            st.warning("Model G: required columns missing or no eligible rows after filtering.")
            return {"success": True, "summary": {}, "results_df": pd.DataFrame()}

        # Build G.08 sequences (greedy)
        sequences = _build_g08_sequences(work)

        # Filter by minimum length and interior-duplicate rule
        sequences = [s for s in sequences if len(s) >= min_len and _no_interior_duplicate_arrivals(s)]

        # Bucketize
        results = {
            "G.08.01[0]":  [],  # Anchor today
            "G.08.01[≠0]": [],
            "G.08.02[0]":  [],  # Epic today
            "G.08.02[≠0]": [],
            "G.08.03[0]":  [],  # Both today
            "G.08.03[≠0]": [],
        }

        # Build per-sequence metadata
        for seq in sequences:
            # End class: Anchor/Epic/Both/None
            cat_base = _g08_end_category(seq, work)
            if not cat_base:
                continue  # skip if we cannot classify end

            # 'today' bucketing
            is_today = _is_today_sequence(seq, report_time, start_hour=start_hour)

            # Track side (+ or -) inferred from first non-zero M#
            first_nonzero = next((x for x in seq if float(x.get("M #", 0)) != 0), None)
            side = "+" if (first_nonzero and float(first_nonzero.get("M #", 0)) > 0) else "-"

            # Summary fields
            smry = _sequence_summary_fields(seq)
            end_day = seq[-1].get("Day", "")
            ends_in = {"G.08.01": "Anchor", "G.08.02": "Epic", "G.08.03": "Both"}.get(cat_base, "")

            seq_info = {
                "category": cat_base + ("[0]" if is_today else "[≠0]"),
                "sequence": seq,
                "length": smry["length"],
                "start_arrival": smry["start_arrival"],
                "end_arrival": smry["end_arrival"],
                "end_day": end_day,
                "m_path": smry["m_path"],
                "output_range": smry["output_range"],
                "ends_in": ends_in,
                "track_side": side,
            }

            bucket_key = seq_info["category"]
            if bucket_key in results:
                results[bucket_key].append(seq_info)

        # --- UI toggles for G.08 buckets ---
        st.markdown("### G.08 — x0Pd.w descending (Grp 1b)")
        c1, c2 = st.columns(2)
        with c1:
            show_g08_01_o1 = st.checkbox("G.08.01[0]  Anchor today",  True, key=f"g080101{key_suffix}")
            show_g08_02_o1 = st.checkbox("G.08.02[0]  Epic today",    True, key=f"g080201{key_suffix}")
            show_g08_03_o1 = st.checkbox("G.08.03[0]  Both today",    True, key=f"g080301{key_suffix}")
        with c2:
            show_g08_01_o2 = st.checkbox("G.08.01[≠0] Anchor other", True, key=f"g080102{key_suffix}")
            show_g08_02_o2 = st.checkbox("G.08.02[≠0] Epic other",   True, key=f"g080202{key_suffix}")
            show_g08_03_o2 = st.checkbox("G.08.03[≠0] Both other",   True, key=f"g080302{key_suffix}")

        # --- Metrics ---
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("G.08.01[0]",  len(results["G.08.01[0]"]))
            st.metric("G.08.02[0]",  len(results["G.08.02[0]"]))
        with mcol2:
            st.metric("G.08.01[≠0]", len(results["G.08.01[≠0]"]))
            st.metric("G.08.02[≠0]", len(results["G.08.02[≠0]"]))
        with mcol3:
            st.metric("G.08.03[0]",  len(results["G.08.03[0]"]))
            st.metric("G.08.03[≠0]", len(results["G.08.03[≠0]"]))

        # --- Enabled sequences collector ---
        enabled_sequences = []
        if show_g08_01_o1: enabled_sequences.extend(results["G.08.01[0]"])
        if show_g08_01_o2: enabled_sequences.extend(results["G.08.01[≠0]"])
        if show_g08_02_o1: enabled_sequences.extend(results["G.08.02[0]"])
        if show_g08_02_o2: enabled_sequences.extend(results["G.08.02[≠0]"])
        if show_g08_03_o1: enabled_sequences.extend(results["G.08.03[0]"])
        if show_g08_03_o2: enabled_sequences.extend(results["G.08.03[≠0]"])

        # --- Optional expanders by bucket ---
        def _expander_block(title: str, key: str):
            if results[key]:
                st.markdown(f"#### {title}")
                for i, seq in enumerate(results[key], start=1):
                    with st.expander(f"{key} • Seq {i}: {seq['length']} steps • {seq['m_path']}"):
                        display_sequence_details(seq)

        if show_g08_01_o1: _expander_block("🧭 G.08.01[0] — ends Anchor (Today)", "G.08.01[0]")
        if show_g08_01_o2: _expander_block("🧭 G.08.01[≠0] — ends Anchor (Other days)", "G.08.01[≠0]")
        if show_g08_02_o1: _expander_block("🌋 G.08.02[0] — ends Epic (Today)", "G.08.02[0]")
        if show_g08_02_o2: _expander_block("🌋 G.08.02[≠0] — ends Epic (Other days)", "G.08.02[≠0]")
        if show_g08_03_o1: _expander_block("⚓🌋 G.08.03[0] — ends BOTH (Today)", "G.08.03[0]")
        if show_g08_03_o2: _expander_block("⚓🌋 G.08.03[≠0] — ends BOTH (Other days)", "G.08.03[≠0]")

        # --- Flatten results for table / export ---
        results_df = _results_table(enabled_sequences)

        if not results_df.empty:
            st.markdown("### Detection Results (enabled categories)")
            st.dataframe(results_df, use_container_width=True)
        else:
            st.info("No enabled G.08 sequences to show.")

        # Summary like your app expects
        total_o1 = sum(len(results[k]) for k in ("G.08.01[0]", "G.08.02[0]", "G.08.03[0]"))
        total_o2 = sum(len(results[k]) for k in ("G.08.01[≠0]", "G.08.02[≠0]", "G.08.03[≠0]"))
        summary = {
            "total_o1": total_o1,
            "total_o2": total_o2,
            "total_sequences": total_o1 + total_o2,
            # Optional: per-bucket counts
            "g08_01_o1": len(results["G.08.01[0]"]),
            "g08_01_o2": len(results["G.08.01[≠0]"]),
            "g08_02_o1": len(results["G.08.02[0]"]),
            "g08_02_o2": len(results["G.08.02[≠0]"]),
            "g08_03_o1": len(results["G.08.03[0]"]),
            "g08_03_o2": len(results["G.08.03[≠0]"]),
        }

        return {"success": True, "summary": summary, "results_df": results_df}

    except Exception as e:
        st.error(f"Model G detection error: {e}")
        return {"success": False, "error": str(e), "summary": {}, "results_df": pd.DataFrame()}
