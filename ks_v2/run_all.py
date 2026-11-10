"""One-command entry point for the v2 pipeline.

Usage:
  python run_all.py --stage all
  python run_all.py --stage data,eda,text,models,eval,dual,ts,explain
  python run_all.py --stage models        # rerun only the model stage

Stages are idempotent and can be re-run in any order that respects the
dependencies:  data -> {eda, text} -> models -> {eval, dual, ts, explain}
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

STAGES = ["data", "eda", "text", "models", "eval", "dual", "ts", "explain"]
MODULE = {
    "data": "src.data_v2",
    "eda": "src.eda_numbers",
    "text": "src.text_model",
    "models": "src.models",
    "eval": "src.evaluate",
    "dual": "src.dual",
    "ts": "src.tseries",
    "explain": "src.explain",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    help="comma separated stage names or 'all'")
    args = ap.parse_args()
    stages = STAGES if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}; valid: {STAGES}")

    from src import config as C
    C.log(f"=== run_all stages={stages} ===")
    for s in stages:
        t0 = time.time()
        C.log(f"--- stage {s} start ---")
        mod = importlib.import_module(MODULE[s])
        mod.main()
        C.log(f"--- stage {s} done in {time.time() - t0:.1f}s ---")
    C.log("=== run_all finished ===")


if __name__ == "__main__":
    main()
