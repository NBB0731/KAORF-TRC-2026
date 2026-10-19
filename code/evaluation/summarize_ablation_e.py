"""Summarise the E-stage (F5) mechanism-ablation table.

Validation-side, paired across seeds. Each run is reduced to the epoch selected
by the training protocol (minimum validation joint loss, matching the locked
pipeline's --selection-metric joint), then arms are compared against the `full`
reference arm seed by seed.

    python scripts/summarize_ablation_e.py --runs runs/ablation_e_20260923 \
        --reference full --out runs/ablation_e_20260923/SUMMARY_ablation_e.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from scipy import stats

ARMS = [
    "full",
    "abl_no_anchor",
    "abl_no_bounded_residual",
    "abl_no_stop_gradient",
    "abl_no_ordinal",
    "abl_single_threshold",
    "abl_fuzzy_only",
]

ARM_LABEL = {
    "full": "full (reference)",
    "abl_no_anchor": "no kinematic anchor",
    "abl_no_bounded_residual": "no bounded residual",
    "abl_no_stop_gradient": "no stop-gradient",
    "abl_no_ordinal": "no ordinal constraint",
    "abl_single_threshold": "single 0.25 nm threshold",
    "abl_fuzzy_only": "fuzzy proxy target only",
}

METRICS = [
    ("ade_nm", "lower"),
    ("fde_nm", "lower"),
    ("risk_auprc", "higher"),
    ("risk_brier", "lower"),
    ("risk_ece", "lower"),
    ("moderate_risk_auprc", "higher"),
    ("moderate_risk_brier", "lower"),
    ("moderate_risk_ece", "lower"),
]

SEEDS = [17, 29, 42, 71, 101, 131, 211, 307, 509, 887]


def load(runs: Path, seed: int, arm: str) -> dict | None:
    path = runs / f"seed_{seed}" / arm / "metrics.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return None
    best = min(rows, key=lambda r: r["val"]["loss"])
    return {"selected_epoch": int(best["epoch"]), **best["val"]}


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def fmt(value: float, digits: int = 4) -> str:
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def fmt_p(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.2e}" if value < 1e-3 else f"{value:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/ablation_e_20260923")
    ap.add_argument("--reference", default="full")
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    runs = Path(args.runs)
    table: dict[str, dict[int, dict]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            got = load(runs, seed, arm)
            if got is not None:
                table[arm][seed] = got

    lines: list[str] = []
    rows_csv: list[tuple] = []

    present = [arm for arm in ARMS if table[arm]]
    missing = [arm for arm in ARMS if not table[arm]]

    lines.append("# E 阶段：F5 机制消融（验证侧，跨种子配对）")
    lines.append("")
    lines.append(f"- 结果目录：`{runs}`")
    lines.append("- 协议：train = NY 2025-01..08（32 日），val = NY 2025-09..10（8 日）；")
    lines.append("  **blind_test（2025-11..12）从未传入训练器**（launcher 的 leakage guard 断言三者不相交）。")
    lines.append("- 每次 run 取「验证集 joint loss 最小的 epoch」（与原锁定流程 `--selection-metric joint` 一致）。")
    lines.append("- 消融臂相对 `full` 均为**单因子**改动；`full` 与 `constrained_feedback` 完全等价。")
    lines.append(f"- 参考臂：`{args.reference}`；配对种子：{', '.join(str(s) for s in SEEDS)}")
    lines.append("")
    if missing:
        lines.append(f"> 注意：以下臂尚无结果（未跑完）：{', '.join(missing)}")
        lines.append("")

    # ---------------------------------------------------------------- means
    lines.append("## 1. 各臂均值（验证集，选中 epoch）")
    lines.append("")
    header = "| 消融臂 | " + " | ".join(
        f"{m}{'↓' if d == 'lower' else '↑'}" for m, d in METRICS
    ) + " | n |"
    lines.append(header)
    lines.append("|" + "---|" * (len(METRICS) + 2))
    for arm in present:
        n = len(table[arm])
        cells = []
        for metric, _ in METRICS:
            values = [
                table[arm][s][metric]
                for s in table[arm]
                if metric in table[arm][s] and not math.isnan(table[arm][s][metric])
            ]
            mean, sd = mean_sd(values)
            cells.append(f"{fmt(mean)} ± {fmt(sd)}")
        lines.append(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + f" | {n} |")
    lines.append("")

    # -------------------------------------------------------- paired tests
    lines.append(f"## 2. 相对参考臂 `{args.reference}` 的配对检验（10 种子）")
    lines.append("")
    for metric, direction in METRICS:
        lines.append(f"### {metric} ({'越低越好' if direction == 'lower' else '越高越好'})")
        lines.append("")
        lines.append(f"| 消融臂 | 参考 mean±sd | 该臂 mean±sd | Δ(臂−参考) | 该臂胜 | 配对 t | p(t) | Wilcoxon W | p(W) | Cohen's dz |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        ref_map = {s: table[args.reference][s].get(metric) for s in table[args.reference]}
        for arm in present:
            if arm == args.reference:
                continue
            pa, pb = [], []
            for seed in sorted(set(table[arm]) & set(table[args.reference])):
                va, vb = table[arm][seed].get(metric), ref_map.get(seed)
                if va is None or vb is None:
                    continue
                if isinstance(va, float) and math.isnan(va):
                    continue
                if isinstance(vb, float) and math.isnan(vb):
                    continue
                pa.append(va)
                pb.append(vb)
            if len(pa) < 2:
                lines.append(f"| {ARM_LABEL[arm]} | - | - | - | - | - | - | - | - | - |")
                continue
            diffs = [a - b for a, b in zip(pa, pb)]
            md = statistics.fmean(diffs)
            sd = statistics.stdev(diffs)
            t_stat, p_t = stats.ttest_1samp(diffs, 0.0)
            try:
                w_stat, p_w = stats.wilcoxon(pa, pb)
            except ValueError:
                w_stat, p_w = float("nan"), float("nan")
            dz = md / sd if sd > 0 else float("inf")
            wins = sum(1 for d in diffs if d < 0) if direction == "lower" else sum(1 for d in diffs if d > 0)
            ma, sa = mean_sd(pb)
            mb, sb = mean_sd(pa)
            lines.append(
                f"| {ARM_LABEL[arm]} | {fmt(ma)} ± {fmt(sa)} | {fmt(mb)} ± {fmt(sb)} | "
                f"{md:+.4f} | {wins}/{len(diffs)} | {t_stat:.3f} | {fmt_p(p_t)} | "
                f"{'-' if math.isnan(w_stat) else f'{w_stat:.1f}'} | {fmt_p(p_w)} | {dz:+.2f} |"
            )
            rows_csv.append(
                (
                    arm, metric, len(diffs), ma, sa, mb, sb, md, t_stat, p_t,
                    w_stat, p_w, dz, wins, len(diffs) - wins,
                )
            )
        lines.append("")

    # ---------------------------------------------------------------- notes
    lines.append("## 3. 判读注意")
    lines.append("")
    lines.append("- **单因子定义**：")
    lines.append("  - `no kinematic anchor`：可学习的运动学锚头换成**非学习的几何模糊先验**（`anchor_from_fuzzy`）。")
    lines.append("    这不是「删掉锚 + 删掉锚相关损失」——anchor BCE 与 calibration KL 仍然作用，")
    lines.append("    只是参照物退化为模糊先验，因此仍是单因子对比。")
    lines.append("  - `no bounded residual`：同时去掉 `residual_bound` 缩放与 `tanh` 饱和（`unbounded_residual`）。")
    lines.append("  - `no stop-gradient`：`constrained_feedback` 不传 `--detach-risk-feedback`。")
    lines.append("  - `no ordinal constraint`：`moderate = anchor + softplus(delta)` → `anchor + delta`（`free_ordinal`）。")
    lines.append("  - `single 0.25 nm threshold`：moderate 目标阈值由 0.5 改为与 severe 相同的 0.25（`--moderate-threshold-nm 0.25`）。")
    lines.append("  - `fuzzy proxy target only`：两档目标都改用模糊代理（不传 `--close-event-target`）。")
    lines.append("- ⚠️ **`fuzzy proxy target only` 的 AUPRC/Brier 与其余臂目标定义不同**，")
    lines.append("  其风险指标不可与其它臂直接横向比较，只能作为「用模糊代理替代近遇硬标签」的整体效果参考。")
    lines.append("- 本表是**验证侧**结果。`blind_test` 的一次性口径不受影响，也不得被本阶段的任何选择所污染。")
    lines.append("")

    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["arm", "metric", "n_paired", "ref_mean", "ref_sd", "arm_mean", "arm_sd",
                 "mean_diff_arm_minus_ref", "t", "p_t", "wilcoxon_W", "p_w", "cohens_dz",
                 "arm_wins", "arm_losses"]
            )
            writer.writerows(rows_csv)
        print(f"wrote {args.csv}")
    print("\n".join(lines[:200]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
