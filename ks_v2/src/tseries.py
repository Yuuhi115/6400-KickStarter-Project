"""Stage 7 - platform-level monthly success rate: Holt-Winters vs naive benchmarks."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def series_from(df: pd.DataFrame) -> pd.Series:
    g = df.groupby(["yr", "mon"], observed=True)["succ"].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= 50].copy()
    g["date"] = pd.to_datetime(dict(year=g.yr, month=g.mon, day=1))
    g = g.sort_values("date").set_index("date")
    s = g["mean"]
    # make the series regular: drop nothing, fill the (rare) internal gaps
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(full).interpolate(limit_direction="both")


def main() -> None:
    df = C.load_v2()
    s = series_from(df)
    C.log(f"ts: monthly series {s.index.min():%Y-%m} .. {s.index.max():%Y-%m} (n={len(s)})")

    cut = pd.Timestamp(f"{C.TRAIN_MAX_YEAR}-12-01")
    train = s[s.index <= cut]
    test = s[s.index > cut]
    C.log(f"ts: train months={len(train)} test months={len(test)}")

    fit = ExponentialSmoothing(train, trend="add", seasonal="add",
                               seasonal_periods=12,
                               initialization_method="estimated").fit()
    preds = {"HW_additive": np.empty(len(test)), "HW_multiplicative": np.empty(len(test)),
             "seasonal_naive": np.empty(len(test)), "last_value": np.empty(len(test))}
    mult_ok = True
    for i, ts in enumerate(test.index):
        hist = s[s.index < ts]
        try:
            fm = ExponentialSmoothing(hist, trend="add", seasonal="add",
                                      seasonal_periods=12,
                                      initialization_method="estimated").fit()
            preds["HW_additive"][i] = float(np.ravel(fm.forecast(1))[0])
        except Exception:  # noqa: BLE001
            preds["HW_additive"][i] = np.nan
        try:
            fm2 = ExponentialSmoothing(hist, trend="add", seasonal="mul",
                                       seasonal_periods=12,
                                       initialization_method="estimated").fit()
            preds["HW_multiplicative"][i] = float(np.ravel(fm2.forecast(1))[0])
        except Exception:  # noqa: BLE001
            mult_ok = False
            preds["HW_multiplicative"][i] = np.nan
        preds["seasonal_naive"][i] = float(hist.iloc[-12])
        preds["last_value"][i] = float(hist.iloc[-1])

    rows = []
    y = test.to_numpy()
    for name, p in preds.items():
        m = np.isfinite(p)
        if not m.any():
            continue
        err = y[m] - p[m]
        rows.append({"method": name, "n": int(m.sum()),
                     "mae": float(np.mean(np.abs(err))),
                     "rmse": float(np.sqrt(np.mean(err ** 2))),
                     "mape": float(np.mean(np.abs(err) / np.maximum(y[m], 1e-9)))})
    tb = pd.DataFrame(rows).sort_values("mae")
    tb["mae_vs_seasonal_naive"] = tb["mae"] / tb.loc[tb.method == "seasonal_naive", "mae"].iloc[0]
    tb.to_csv(C.TABLE_DIR / "timeseries_metrics.csv", index=False)
    C.log("\n" + tb.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.plot(train.index, train.values, color="#1f4e79", lw=1.2, label="train months")
    ax.plot(test.index, test.values, "o-", color="black", ms=2.5, lw=1.0, label="actual (out-of-time)")
    for name, color in [("HW_additive", "#c00000"), ("seasonal_naive", "#7f7f7f")]:
        ax.plot(test.index, preds[name], "-", color=color, lw=1.0, label=name)
    ax.set_ylabel("monthly success rate"); ax.set_xlabel("month")
    ax.set_title("Monthly success rate: Holt-Winters vs seasonal naive")
    ax.legend(fontsize=7, loc="upper left")
    C.savefig(fig, "fig_holtwinters.png")
    C.log("stage tseries done")


if __name__ == "__main__":
    main()
