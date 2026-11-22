#!/usr/bin/env python3
"""
run_session.py — run the Unified Traveler Pipeline, then the Recommendations, in one step.

Each step can still be run on its own with its usual command; this script only chains them.

  Both steps:
    python run_session.py -dt "2026-11-15 18:00"

  Pipeline only / Recommendations only:
    python run_session.py -dt "2026-11-15 18:00" --pipeline-only
    python run_session.py -dt "2026-11-12 18:00" --recs-only --bars "my_1m_pipe_MA.csv"

  Anything after "--" is passed to the pipeline unchanged:
    python run_session.py -dt "2026-11-15 18:00" -- --signal-matrix "550_Signal_Matrix.xlsx"

Defaults match your usual command:
  -m "Meas 4_8 all.xlsx" --lookback 27 --assets NQ --reports matched sequence --reporter-frames --no-verify
"""
import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="Pipeline + Recommendations for one report time.")
    ap.add_argument("-dt", "--dt", required=True, help='Report time, "YYYY-MM-DD HH:MM"')
    ap.add_argument("-m", "--measurement", default="Meas 4_8 all.xlsx")
    ap.add_argument("--lookback", default="27")
    ap.add_argument("--asset", default="NQ", help="NQ, ES or RTY (Reporter / Recommendations asset)")
    ap.add_argument("--pipeline", default="unified_traveler_pipeline_6_2_5.py")
    ap.add_argument("--output-dir", "-o", default=".", help="Where the pipeline writes and Recommendations reads")
    ap.add_argument("--daily-goal", default="60")
    ap.add_argument("--bars", default=None, help="1-minute Gasser CSV → outcome columns (past sessions only)")
    ap.add_argument("--reporter", default=None, help="Use this Reporter file instead of finding it by date")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pipeline-only", action="store_true")
    g.add_argument("--recs-only", action="store_true")
    a, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    ts = datetime.strptime(a.dt.strip(), "%Y-%m-%d %H:%M").strftime("%Y%m%d_%H%M")
    py = sys.executable

    # ---- step 1: pipeline (Matched → Reporter → Signal Matrix with the Recommendations tab)
    if not a.recs_only:
        cmd = [py, os.path.join(HERE, a.pipeline), "-m", a.measurement, "-dt", a.dt,
               "--lookback", a.lookback, "--assets", a.asset, "--reports", "matched", "sequence",
               "--reporter-frames", "--output-dir", a.output_dir, "--no-verify"] + extra
        print("\n▶ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            sys.exit(f"Pipeline stopped with code {rc}; Recommendations not run.")
        if a.pipeline_only:
            return

    # ---- step 2: Recommendations workbook
    rep = a.reporter
    if not rep:
        hits = sorted(glob.glob(os.path.join(a.output_dir, f"Reporter_{a.asset}*_{ts}.xlsx")), key=os.path.getmtime)
        if not hits:
            sys.exit(f"No Reporter_{a.asset}*_{ts}.xlsx in {os.path.abspath(a.output_dir)} — "
                     "run the pipeline first or pass --reporter.")
        rep = hits[-1]
    cid = os.path.basename(rep).split("_")[1]           # e.g. NQZ (changes to NQH after the roll)
    out = os.path.join(a.output_dir, f"{cid}_Recommendations_{ts}.xlsx")
    cmd = [py, os.path.join(HERE, "reporter_views.py"), "--reporter", rep, "--out", out, "--daily-goal", a.daily_goal]
    if a.bars:
        cmd += ["--bars", a.bars]
    print("\n▶ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"Recommendations stopped with code {rc}.")
    print(f"\n✅ Done: {out}")


if __name__ == "__main__":
    main()
