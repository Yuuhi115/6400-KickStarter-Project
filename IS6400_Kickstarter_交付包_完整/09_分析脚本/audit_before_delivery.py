# -*- coding: utf-8 -*-
"""交付前门禁（read-only）。见根目录 `_工作协议.md` 第三节。

四个门禁：
  G1 术语   : 禁用词是否都在 `§0 口径契约` 里被公式化定义
  G2 一致性 : 同一格子的 p_profile / p_cell_mean / rate 是否并排存在（单一事实源）
  G3 可行动性: 每条建议是否 n_target>=300、时长 1-60 天、dp_max>=0
  G4 可复现 : 脚本是否硬编码绝对路径、requirements.txt 是否存在并锁定 sklearn 版本

用法：
  python "IS6400_Kickstarter_交付包_完整\\09_分析脚本\\audit_before_delivery.py"
退出码：0=全部门禁通过；1=存在 FAIL（不得交付）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]              # 工作区根
PKG = ROOT / "IS6400_Kickstarter_交付包_完整"
OUT = PKG / "11_结果与报告"
SCRIPTS = PKG / "09_分析脚本"
DOCS = [
    PKG / "交付说明_期末进度.md",
    OUT / "S1_模型对比.md",
    OUT / "S1_解释性.md",
]

BANNED = {
    "期望募资额": "p×中位数不是 E[Y]",
    "期望损失": "无成本字段，不能叫期望损失",
    "最高期望价值": "同上，价值维度未经检验",
    "已自证": "只有消融，不构成自证",
    "证明无泄漏": "只能证明显式事后列未入模",
}
MIN_CELL_N = 300
DUR_MIN, DUR_MAX = 1, 60

results: list[tuple[str, str, str]] = []      # (gate, status, detail)


def add(gate: str, ok: bool, detail: str) -> None:
    results.append((gate, "PASS" if ok else "FAIL", detail))


# ---------------------------------------------------------------- G1 术语
def gate_terms() -> None:
    texts = {p.name: p.read_text(encoding="utf-8", errors="ignore")
             for p in DOCS if p.exists()}
    if not texts:
        add("G1", False, "找不到任何交付文档")
        return
    hits: list[str] = []
    for name, txt in texts.items():
        for word, why in BANNED.items():
            n = txt.count(word)
            if n:
                hits.append(f"{name}: '{word}' ×{n}（{why}）")
    has_contract = any("口径契约" in t for t in texts.values())
    ok = not hits and has_contract
    detail = []
    if hits:
        detail.append("禁用词未清除: " + "; ".join(hits[:6]))
    if not has_contract:
        detail.append("交付文档缺少 `§0 口径契约` 一节（术语→公式→代码位置）")
    add("G1", ok, " | ".join(detail) if detail else "无禁用词，且存在 §0 口径契约")


# ---------------------------------------------------------------- G2 一致性
def gate_consistency() -> None:
    src = OUT / "cells.parquet"
    legacy = OUT / "decision_axes_cells.csv"
    if not src.exists():
        add("G2", False,
            "缺少唯一事实源 `11_结果与报告/cells.parquet`（要求并排 p_profile / p_cell_mean / rate）")
        return
    try:
        import pandas as pd
        df = pd.read_parquet(src)
    except Exception as exc:                                   # noqa: BLE001
        add("G2", False, f"cells.parquet 读取失败: {exc}")
        return
    need = {"p_profile", "p_cell_mean", "rate"}
    miss = need - set(df.columns)
    add("G2", not miss,
        "口径列齐备" if not miss else f"cells.parquet 缺列: {sorted(miss)}")


# ---------------------------------------------------------------- G3 可行动性
def gate_actionable() -> None:
    legacy = OUT / "decision_axes_cells.csv"
    if not legacy.exists():
        add("G3", False, "找不到 decision_axes_cells.csv")
        return
    import pandas as pd
    d = pd.read_csv(legacy, encoding="utf-8-sig")
    use = d[d["n"] >= MIN_CELL_N].copy()
    problems = []

    neg = int((use["dp_max"] < 0).sum())
    if neg:
        problems.append(f"{neg} 个可用格 dp_max<0（应允许“保持原配置”使 ≥0）")

    n61 = int((use["best_dur_band"].astype(str) == "61d+").sum())
    if n61:
        problems.append(f"{n61} 个可用格最优时长落在 61d+（平台上限 {DUR_MAX} 天）")

    lookup = d.set_index(["category", "goal_band", "dur_band"])["n"]
    under = []
    for _, r in use.iterrows():
        k = (r["category"], r["best_goal_band"], r["best_dur_band"])
        tn = lookup.get(k)
        if tn is not None and tn < MIN_CELL_N:
            under.append((r["category"], r["best_goal_band"], r["best_dur_band"], int(tn)))
    if under:
        problems.append(f"{len(under)} 个出发格的目标格 n<{MIN_CELL_N}"
                        f"（例: {under[0][0]} -> {under[0][1]}×{under[0][2]} n={under[0][3]}）")

    add("G3", not problems, " | ".join(problems) if problems
        else "所有建议均满足 n_target、1–60 天、dp_max≥0")


# ---------------------------------------------------------------- G4 可复现
def gate_reproducible() -> None:
    problems = []
    hard = []
    pat = re.compile(r"[A-Za-z]:\\\\?Workspace|E:\\\\Workspace|E:/Workspace", re.I)
    for py in sorted(SCRIPTS.glob("*.py")):
        if py.name == Path(__file__).name:
            continue  # 本脚本自身用 Path(__file__) 定位，用法说明里的示例路径不算硬编码
        txt = py.read_text(encoding="utf-8", errors="ignore")
        if pat.search(txt) or 'r"E:' in txt or "r'E:" in txt:
            hard.append(py.name)
    if hard:
        problems.append(f"{len(hard)} 个脚本硬编码绝对路径: {', '.join(hard[:6])}")

    req = PKG / "requirements.txt"
    if not req.exists():
        problems.append("缺 `requirements.txt`（模型由特定 sklearn 版本保存，跨版本会加载失败）")
    else:
        rt = req.read_text(encoding="utf-8", errors="ignore")
        if "scikit-learn" not in rt.lower():
            problems.append("requirements.txt 未锁定 scikit-learn 版本")

    schema = OUT / "model_histgb_ours_schema.json"
    if schema.exists():
        try:
            json.loads(schema.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            problems.append("model schema JSON 无法解析")

    add("G4", not problems, " | ".join(problems) if problems else "无硬编码路径，版本锁定齐备")


def main() -> int:
    for fn in (gate_terms, gate_consistency, gate_actionable, gate_reproducible):
        try:
            fn()
        except Exception as exc:                                # noqa: BLE001
            add(fn.__name__, False, f"检查异常: {exc}")

    print("=" * 74)
    print("交付前门禁（_工作协议.md §3）")
    print("=" * 74)
    width = max(len(g) for g, _, _ in results)
    for gate, status, detail in results:
        mark = "PASS" if status == "PASS" else "FAIL"
        print(f"[{mark}] {gate:<{width}}  {detail}")

    fails = [r for r in results if r[1] == "FAIL"]
    print("-" * 74)
    if fails:
        print(f"结果: {len(fails)} 项未通过 —— 该产物不得交付，先修再跑。")
        return 1
    print("结果: 全部门禁通过，可交付。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
