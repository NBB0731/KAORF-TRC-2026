# -*- coding: utf-8 -*-
"""F2-C3 graph baseline vs KAORF and vs the F2-C1 plain Transformer.

Everything is paired by locked seed (10 seeds), using the same statistics that
the E-stage ablation and the D-stage external-port summary use: paired t,
Wilcoxon signed-rank, Cohen's dz = mean(diff)/sd(diff), and a win/loss count
over seeds.

Inputs (all archived locally, nothing is recomputed here)
---------------------------------------------------------
* graph    : _cloud_results/kaorf_c1graph_20260925/seed_*/graph_baseline/metrics.json
* KAORF    : _cloud_results/kaorf_blind_20260923/seed_*/constrained_feedback/final_test_metrics.json
* C1       : _cloud_results/kaorf_f6_20260924/runs/baseline_c1_20260924/seed_*/plain_transformer/metrics.json

Output
------
* C3_GRAPH_SUMMARY.md   -- markdown tables ready to paste into the manuscript
* C3_GRAPH_SUMMARY.json -- the same numbers, machine readable

Usage
-----
    python summarize_c1graph.py
"""
from __future__ import annotations

import json
import math
import pathlib

from scipy import stats

SEEDS = [17, 29, 42, 71, 101, 131, 211, 307, 509, 887]

GRAPH = pathlib.Path(__file__).resolve().parents[2] / "results/f4_f6/runs/baseline_c3graph_20260925"
KAORF = pathlib.Path(__file__).resolve().parents[2] / "results/blind"
C1 = pathlib.Path(__file__).resolve().parents[2] / "results/f4_f6/runs/baseline_c1_20260924"
OUT = pathlib.Path(__file__).resolve().parents[2] / "results/c3_graph_summary"

METRICS = [
    ("ade_nm", "ADE (nm)", "lower"),
    ("fde_nm", "FDE (nm)", "lower"),
    ("risk_auprc", "severe AUPRC", "higher"),
    ("risk_brier", "severe Brier", "lower"),
    ("risk_ece", "severe ECE", "lower"),
]


def load_graph(seed: int) -> dict:
    path = GRAPH / f"seed_{seed}" / "graph_baseline" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))["test_metrics"]


def load_kaorf(seed: int) -> dict:
    path = KAORF / f"seed_{seed}" / "constrained_feedback" / "final_test_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))["test_metrics"]


def load_c1(seed: int) -> dict:
    path = C1 / f"seed_{seed}" / "plain_transformer" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))["test_metrics"]


def mean_sd(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def fmt_p(value: float) -> str:
    if value < 1e-4:
        return f"{value:.1e}"
    return f"{value:.4f}"


def compare(a_vals: list[float], b_vals: list[float], direction: str,
            a_name: str, b_name: str) -> dict:
    """a is the challenger, b is the reference; direction is what is better."""
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    n = len(diffs)
    md = sum(diffs) / n
    sd = math.sqrt(sum((d - md) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    t_stat, p_t = stats.ttest_rel(a_vals, b_vals)
    try:
        w_stat, p_w = stats.wilcoxon(a_vals, b_vals)
    except ValueError:
        w_stat, p_w = float("nan"), float("nan")
    dz = md / sd if sd > 0 else float("inf")
    a_wins = sum(1 for d in diffs if d < 0) if direction == "lower" \
        else sum(1 for d in diffs if d > 0)
    return {
        "a": a_name, "b": b_name,
        "a_mean": mean_sd(a_vals)[0], "a_sd": mean_sd(a_vals)[1],
        "b_mean": mean_sd(b_vals)[0], "b_sd": mean_sd(b_vals)[1],
        "mean_diff": md, "t": float(t_stat), "p_t": float(p_t),
        "wilcoxon_W": float(w_stat), "p_w": float(p_w),
        "cohens_dz": dz,
        "a_wins": a_wins, "a_losses": n - a_wins, "n": n,
    }


def main() -> int:
    graph = {s: load_graph(s) for s in SEEDS}
    kaorf = {s: load_kaorf(s) for s in SEEDS}
    c1 = {s: load_c1(s) for s in SEEDS}

    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    payload: dict = {"seeds": SEEDS, "comparisons": {}}

    lines.append("# F2-C3 graph-interaction baseline: blind-test comparison")
    lines.append("")
    lines.append("All numbers: New York 2025-11..12 blind partition, 10 locked seeds, "
                 "paired by seed.")
    lines.append("")

    # --- per-seed table -------------------------------------------------
    header = ["seed", "graph ADE", "graph FDE", "graph AUPRC", "graph Brier", "graph ECE",
              "KAORF ADE", "KAORF FDE", "KAORF AUPRC", "KAORF Brier", "KAORF ECE",
              "C1 ADE", "C1 FDE", "C1 AUPRC", "C1 Brier", "C1 ECE"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for seed in SEEDS:
        row = [str(seed)]
        for source in (graph, kaorf, c1):
            for key, _, _ in METRICS:
                row.append(f"{source[seed][key]:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- pairwise comparisons -------------------------------------------
    pairs = [("graph", graph, "KAORF", kaorf), ("graph", graph, "C1", c1),
             ("C1", c1, "KAORF", kaorf)]
    for a_name, a_src, b_name, b_src in pairs:
        lines.append(f"## {a_name} vs {b_name} (positive Δ = {a_name} worse "
                     f"for lower-is-better metrics)")
        lines.append("")
        lines.append("| metric | " + f"{a_name} mean±sd | {b_name} mean±sd | "
                     "Δ(a−b) | a wins | t | p(t) | W | p(W) | dz |")
        lines.append("|" + "---|" * 10)
        payload["comparisons"][f"{a_name}_vs_{b_name}"] = {}
        for key, label, direction in METRICS:
            a_vals = [a_src[s][key] for s in SEEDS]
            b_vals = [b_src[s][key] for s in SEEDS]
            res = compare(a_vals, b_vals, direction, a_name, b_name)
            res["metric"] = label
            res["direction"] = direction
            payload["comparisons"][f"{a_name}_vs_{b_name}"][key] = res
            w_text = "-" if math.isnan(res["wilcoxon_W"]) else f"{res['wilcoxon_W']:.1f}"
            lines.append(
                f"| {label} | {res['a_mean']:.4f}±{res['a_sd']:.4f} | "
                f"{res['b_mean']:.4f}±{res['b_sd']:.4f} | {res['mean_diff']:+.4f} | "
                f"{res['a_wins']}/{res['n']} | {res['t']:.3f} | {fmt_p(res['p_t'])} | "
                f"{w_text} | {fmt_p(res['p_w'])} | {res['cohens_dz']:+.2f} |"
            )
        lines.append("")

    (OUT / "C3_GRAPH_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "C3_GRAPH_SUMMARY.json").write_text(json.dumps(payload, indent=2),
                                               encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT/'C3_GRAPH_SUMMARY.md'} and {OUT/'C3_GRAPH_SUMMARY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
