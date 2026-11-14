# -*- coding: utf-8 -*-
"""run_all.py — 串行跑完整链路并汇总日志。

  python run_all.py                 # 全链路
  python run_all.py --from matrix   # 从某一步开始
  python run_all.py --only evaluate
步骤：gate → baseline → matrix → evaluate → figures
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import cfg

STEPS = ["gate", "baseline", "matrix", "evaluate", "figures"]
SCRIPTS = {"gate": "data_gate.py", "baseline": "run_baseline.py",
           "matrix": "run_matrix.py", "evaluate": "evaluate.py", "figures": "figures.py"}
HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="gate", choices=STEPS)
    ap.add_argument("--only", default=None, choices=STEPS)
    args = ap.parse_args()
    steps = [args.only] if args.only else STEPS[STEPS.index(args.start):]

    log_path = os.path.join(cfg.RES, "runs.log")
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "4"
    env["PYTHONIOENCODING"] = "utf-8"

    t_all = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 70}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] run_all steps={steps}\n")
        for step in steps:
            t0 = time.time()
            print(f"\n>>> {step} ({SCRIPTS[step]})", flush=True)
            log.write(f"\n>>> {step} start {time.strftime('%H:%M:%S')}\n")
            log.flush()
            proc = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, SCRIPTS[step])],
                                    cwd=HERE, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace", bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            code = proc.wait()
            dur = time.time() - t0
            log.write(f"<<< {step} exit={code} {dur:.1f}s\n")
            print(f"<<< {step} exit={code} {dur:.1f}s", flush=True)
            if code != 0:
                raise SystemExit(f"step {step} failed with code {code}")
    print(f"\nALL DONE in {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
