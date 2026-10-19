"""Summarise the D stage (F3) zero-shot cross-port transfer table.

Inputs
------
``--runs``    tag root produced by ``scripts/eval_external_zeroshot.py``
              (expects ``<tag>/<port>/seed_<s>/<variant>/metrics.json``)
``--ny-runs`` the locked New York run root, used only to print the NY
              blind-test reference next to the external numbers

Outputs
-------
``SUMMARY_external_zeroshot.md`` / ``.csv``

Statistics
----------
* Headline numbers are the pooled pass (all selected days at once), averaged over
  the 10 paired seeds -- the same aggregation the blind-test summary uses.
* Uncertainty is a **day-clustered bootstrap** (the pre-registered sampling unit
  is the calendar day): days are resampled with replacement inside each seed and
  the seed-level statistic is averaged, so both day sampling and seed spread are
  represented. This matters most for Chesapeake Bay, where a day carries ~127
  scenes instead of ~1 400-2 300.
* The feedback / no-feedback comparison uses the ``sub12`` tag, where both
  variants are evaluated on the *same* 12 stratified days, so the pairs are
  comparable seed by seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

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

LABEL = {
    "ade_nm": "ADE (nm)",
    "fde_nm": "FDE (nm)",
    "risk_auprc": "severe AUPRC",
    "risk_brier": "severe Brier",
    "risk_ece": "severe ECE",
    "moderate_risk_auprc": "moderate AUPRC",
    "moderate_risk_brier": "moderate Brier",
    "moderate_risk_ece": "moderate ECE",
}

PORT_LABEL = {
    "houston_galveston": "Houston-Galveston",
    "chesapeake": "Chesapeake Bay",
}


def fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def fmt_p(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.2e}" if value < 1e-3 else f"{value:.4f}"


def mean_sd(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def load_tag(runs: Path, tag: str, port: str, seed: int, variant: str) -> dict | None:
    path = runs / tag / port / f"seed_{seed}" / variant / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_ny(ny_runs: Path, seed: int, variant: str) -> dict | None:
    path = ny_runs / f"seed_{seed}" / variant / "final_test_metrics.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = dict(payload.get("test_metrics", {}))
    metrics["_selected_epoch"] = payload.get("selected_epoch")
    return metrics


def bootstrap_ci(
    records: list[dict],
    key: str,
    weighting: str,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Day-clustered bootstrap over the paired seeds.

    weighting = "node" -> node-weighted day mean (exact for ADE/FDE)
    weighting = "day"  -> unweighted mean of finite day values (risk metrics)
    """
    usable = [r for r in records if r is not None and r.get("per_day")]
    if not usable:
        return float("nan"), float("nan"), float("nan")
    n_days = len(usable[0]["per_day"])
    weights = np.array([[d["node_samples"] for d in r["per_day"]] for r in usable], dtype=float)
    values = np.array(
        [[d["metrics"].get(key, float("nan")) for d in r["per_day"]] for r in usable],
        dtype=float,
    )

    def statistic(idx: np.ndarray) -> float:
        w = weights[:, idx]
        v = values[:, idx]
        if weighting == "node":
            totals = w.sum(axis=1)
            per_seed = np.where(totals > 0, (w * v).sum(axis=1) / np.maximum(totals, 1e-12), np.nan)
        else:
            finite = np.isfinite(v)
            counts = finite.sum(axis=1)
            per_seed = np.where(
                counts > 0, np.where(finite, v, 0.0).sum(axis=1) / np.maximum(counts, 1), np.nan
            )
        per_seed = per_seed[np.isfinite(per_seed)]
        return float(per_seed.mean()) if per_seed.size else float("nan")

    point = statistic(np.arange(n_days))
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = statistic(rng.integers(0, n_days, size=n_days))
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/external_zeroshot_20260923")
    parser.add_argument("--ny-runs", default="runs/final_ny_temporal_10seed_20260921")
    parser.add_argument("--headline-tag", default="full48")
    parser.add_argument("--paired-tag", default="sub12")
    parser.add_argument("--reference-variant", default="constrained_feedback")
    parser.add_argument("--baseline-variant", default="constrained_no_feedback")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 42, 71, 101, 131, 211, 307, 509, 887])
    parser.add_argument("--ports", nargs="+", default=["houston_galveston", "chesapeake"])
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    parser.add_argument("--out", default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    runs, ny_runs = Path(args.runs), Path(args.ny_runs)
    rng = np.random.default_rng(args.bootstrap_seed)
    lines: list[str] = []
    csv_rows: list[tuple] = []

    headline: dict[str, dict[str, list[dict]]] = {}
    for port in args.ports:
        recs = [
            load_tag(runs, args.headline_tag, port, seed, args.reference_variant)
            for seed in args.seeds
        ]
        headline[port] = {"records": [r for r in recs if r]}

    lines.append("# D 阶段：外港零微调迁移（F3）——锁定 NY 检查点直接跨港评估")
    lines.append("")
    lines.append("- 协议：**零训练、零微调、零再选择**。检查点取自 NY 验证集选择（`--selection-metric joint`）的锁定盲测 run；")
    lines.append("  外港场景在预注册清单中标记为 `external_only` / `allowed_for_tuning=no`，仅用于最终评估。")
    lines.append("- **未读取任何 NY 场景文件**（评估器对每个路径断言非 new_york），`blind_test` 的一次性口径不受影响。")
    lines.append("- 指标计算**逐字复用** `scripts/train_rfign_dev.py` 的 `run_epoch`，与 NY 盲测同一代码路径；")
    lines.append("  另有守卫断言「逐日节点加权均值 == 整批 pooled 值」（相对差 < 1e-6，实测约 6e-9，纯浮点累加序差异）。")
    lines.append(f"- 结果目录：`{runs}/{args.headline_tag}`；配对子集：`{runs}/{args.paired_tag}`；NY 参照：`{ny_runs}`")
    lines.append(f"- 配对种子（{len(args.seeds)} 个）：{', '.join(str(s) for s in args.seeds)}")
    lines.append("")
    lines.append("> 不确定度用**按日聚类的 bootstrap**（预注册的抽样单位是日）：在每个种子内对日重抽样，再对种子取均值，")
    lines.append("> 同时反映日的抽样波动与种子间波动。Chesapeake Bay 日均仅约 127 个场景（NY 的 1/11），CI 尤其必要。")
    lines.append("")

    # ------------------------------------------------------- 1. headline table
    lines.append(f"## 1. 外港零微调迁移（`{args.reference_variant}`，全 {headline[args.ports[0]]['records'][0]['n_days'] if headline[args.ports[0]]['records'] else '-'} 日）")
    lines.append("")
    lines.append("| 港口 | " + " | ".join(f"{LABEL[m]}{'↓' if d == 'lower' else '↑'}" for m, d in METRICS) + " | 日数 | 场景数 | 节点样本 | 严重/中危正例率 |")
    lines.append("|" + "---|" * (len(METRICS) + 5))
    for port in args.ports:
        recs = headline[port]["records"]
        if not recs:
            lines.append(f"| {PORT_LABEL.get(port, port)} |" + " - |" * (len(METRICS) + 4))
            continue
        cells = []
        for metric, _ in METRICS:
            mean, sd = mean_sd([r["pooled_metrics"].get(metric, float("nan")) for r in recs])
            cells.append(f"{fmt(mean)} ± {fmt(sd)}")
        days = recs[0]["n_days"]
        scenes = int(statistics.fmean([r["n_scenes"] for r in recs]))
        nodes = int(statistics.fmean([r["n_node_samples"] for r in recs]))
        sev = statistics.fmean(
            [r["pooled_metrics"].get("risk_positive_rate", float("nan")) for r in recs]
        )
        lines.append(
            f"| {PORT_LABEL.get(port, port)} | " + " | ".join(cells)
            + f" | {days} | {scenes:,} | {nodes:,} | {sev:.4f} |"
        )
    lines.append("")

    # ------------------------------------------------------- 2. bootstrap CI
    lines.append("## 2. 按日聚类 bootstrap 95% CI（同一 10 个种子）")
    lines.append("")
    lines.append("| 港口 | 指标 | 点估计 | 95% CI 下 | 95% CI 上 | 权重口径 |")
    lines.append("|---|---|---|---|---|---|")
    for port in args.ports:
        recs = headline[port]["records"]
        if not recs:
            continue
        for metric, _ in METRICS:
            weighting = "node" if metric in ("ade_nm", "fde_nm") else "day"
            point, lo, hi = bootstrap_ci(recs, metric, weighting, args.bootstrap, rng)
            lines.append(
                f"| {PORT_LABEL.get(port, port)} | {LABEL[metric]} | {fmt(point)} | "
                f"{fmt(lo)} | {fmt(hi)} | {'节点加权' if weighting == 'node' else '日均值'} |"
            )
            csv_rows.append((port, metric, point, lo, hi, weighting, len(recs)))
    lines.append("")

    # -------------------------------------------------- 3. NY reference + gap
    ny = {variant: {seed: load_ny(ny_runs, seed, variant) for seed in args.seeds}
          for variant in (args.reference_variant, args.baseline_variant)}
    lines.append("## 3. 与 NY 盲测参照的落差（域偏移敏感性）")
    lines.append("")
    ny_ref = [m for m in ny[args.reference_variant].values() if m]
    if ny_ref:
        cells = []
        for metric, _ in METRICS:
            mean, sd = mean_sd([m.get(metric, float("nan")) for m in ny_ref])
            cells.append(f"{fmt(mean)} ± {fmt(sd)}")
        lines.append("| 域 | " + " | ".join(f"{LABEL[m]}" for m, _ in METRICS) + " |")
        lines.append("|" + "---|" * (len(METRICS) + 1))
        lines.append("| **New York（盲测，8 日）** | " + " | ".join(cells) + " |")
        for port in args.ports:
            recs = headline[port]["records"]
            if not recs:
                continue
            cells = [
                fmt(mean_sd([r["pooled_metrics"].get(metric, float("nan")) for r in recs])[0])
                for metric, _ in METRICS
            ]
            lines.append(f"| {PORT_LABEL.get(port, port)}（零微调） | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("| 港口 | 指标 | NY | 外港 | Δ(外港−NY) | 相对变化 |")
        lines.append("|---|---|---|---|---|---|")
        for port in args.ports:
            recs = headline[port]["records"]
            if not recs:
                continue
            for metric, _ in METRICS:
                ny_mean = mean_sd([m.get(metric, float("nan")) for m in ny_ref])[0]
                ext_mean = mean_sd([r["pooled_metrics"].get(metric, float("nan")) for r in recs])[0]
                if math.isnan(ny_mean) or math.isnan(ext_mean) or ny_mean == 0:
                    continue
                delta = ext_mean - ny_mean
                pct = 100.0 * delta / abs(ny_mean)
                lines.append(
                    f"| {PORT_LABEL.get(port, port)} | {LABEL[metric]} | {fmt(ny_mean)} | "
                    f"{fmt(ext_mean)} | {delta:+.4f} | {pct:+.2f}% |"
                )
        lines.append("")
        lines.append("> 口径提醒：NY 是 **8 日**时间盲测（抽样单位 8 天），外港是 **48 日**评估（抽样单位 48 天），")
        lines.append("> 两者的 CI 不可直接比较；本表只作描述性的域偏移读数。")
        lines.append("")
    else:
        lines.append("> 未找到 NY 盲测参照文件，跳过本表。")
        lines.append("")

    # --------------------------------------------------- 4. paired mechanism
    paired_probe = load_tag(runs, args.paired_tag, args.ports[0], args.seeds[0], args.reference_variant)
    paired_days = paired_probe["n_days"] if paired_probe else "?"
    lines.append(f"## 4. 共享 {paired_days} 日分层子集上的配对机制对比")
    lines.append("")
    lines.append(
        f"- 两个变体在**完全相同的日期**上评估（每月 1 日、层轮转），因此可种子级配对："
        f"`{args.reference_variant}` vs `{args.baseline_variant}`。"
    )
    lines.append("- 这是「机制在域偏移下是否仍然有用」的检查；NY 域内的机制结论以 E 阶段消融为准。")
    lines.append("")
    lines.append("| 港口 | 指标 | 参考 mean±sd | 基线 mean±sd | Δ(参考−基线) | 参考胜 | 配对 t | p(t) | Wilcoxon | p(W) | dz |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for port in args.ports:
        refs = [load_tag(runs, args.paired_tag, port, s, args.reference_variant) for s in args.seeds]
        bases = [load_tag(runs, args.paired_tag, port, s, args.baseline_variant) for s in args.seeds]
        for metric, direction in METRICS:
            pa, pb = [], []
            for r, b in zip(refs, bases):
                if not r or not b:
                    continue
                va = r["pooled_metrics"].get(metric, float("nan"))
                vb = b["pooled_metrics"].get(metric, float("nan"))
                if math.isnan(va) or math.isnan(vb):
                    continue
                pa.append(va)
                pb.append(vb)
            if len(pa) < 2:
                lines.append(f"| {PORT_LABEL.get(port, port)} | {LABEL[metric]} | - | - | - | - | - | - | - | - | - |")
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
            if direction == "lower":
                wins = sum(1 for d in diffs if d < 0)
            else:
                wins = sum(1 for d in diffs if d > 0)
            ma, sa = mean_sd(pb)
            mb, sb = mean_sd(pa)
            lines.append(
                f"| {PORT_LABEL.get(port, port)} | {LABEL[metric]} | {fmt(mb)} ± {fmt(sb)} | "
                f"{fmt(ma)} ± {fmt(sa)} | {md:+.4f} | {wins}/{len(diffs)} | {t_stat:.3f} | "
                f"{fmt_p(p_t)} | {'-' if math.isnan(w_stat) else f'{w_stat:.1f}'} | {fmt_p(p_w)} | {dz:+.2f} |"
            )
            csv_rows.append((port, f"paired::{metric}", md, float("nan"), float("nan"), "pooled", len(diffs)))
    lines.append("")

    # ---------------------------------------------------------------- notes
    lines.append("## 5. 判读注意")
    lines.append("")
    lines.append("- **外港从未参与任何选择**：没有早期停止、没有阈值调节、没有超参搜索；检查点与其阈值全部来自 NY 训练流程。")
    lines.append("- **Chesapeake Bay 极稀疏**（日均约 127 场景）：单日波动可主导均值，因此上面给了按日聚类 CI 与正例率；")
    lines.append("  引用时须同时报场景数/节点样本数与正例率，不能只报均值。")
    lines.append("- **Houston-Galveston 体量约为 NY 的 1.6 倍**（2 272 场景/日）：本表的 pooled 值按节点样本自然加权，")
    lines.append("  与 NY 盲测的加权方式一致。")
    lines.append("- 风险指标的分级阈值沿用锁定流程（严重 0.25 nm、中危 0.5 nm），未针对外港调整。")
    lines.append("- 本阶段**不触碰 blind_test**，也不改变任何已有汇总；NY 结论仍以 `SUMMARY_blind_test.md` 为准。")
    lines.append("")

    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["port", "metric", "value_or_delta", "ci_lo", "ci_hi", "weighting", "n"])
            writer.writerows(csv_rows)
        print(f"wrote {args.csv}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
