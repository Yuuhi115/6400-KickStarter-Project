"""Redraw figt7 only, reusing the artifacts written by build_text_layer.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd                                    # noqa: E402
import build_text_layer as B                           # noqa: E402

resp = pd.read_csv(B.PKG / "11_结果与报告" / "text_length_response.csv")
df = B.build_frame()
B.fig_length(df, resp)
print("figt7 redrawn:", B.FIG / "figt7_length_response.png")
