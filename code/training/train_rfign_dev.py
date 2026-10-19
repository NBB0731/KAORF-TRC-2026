"""Train the development RFIGN prototype on NPZ interaction scenes.

Ablation switches added 2026-09-23 for the F5 mechanism study (E stage):
  --moderate-threshold-nm   default 0.5 == the hard-coded pre-ablation value
  --anchor-from-fuzzy       default off == pre-ablation behaviour
  --unbounded-residual      default off == pre-ablation behaviour
  --free-ordinal            default off == pre-ablation behaviour
With no new flag passed, this trainer reproduces the locked blind-test pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Permit direct execution from any working directory, including remote jobs
# launched as ``python scripts/train_rfign_dev.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Resolve the package layout (code/, code/model, code/training, code/evaluation)
# so the same import lines work from any working directory.
_CODE_ROOT = next(_p for _p in Path(__file__).resolve().parents if _p.name == "code")
for _extra in ("", "model", "training", "evaluation"):
    _cand = str(_CODE_ROOT / _extra) if _extra else str(_CODE_ROOT)
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from rfign_model import RFIGN



class SceneDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        loaded = [np.load(path) for path in paths]
        self.history = torch.from_numpy(np.concatenate([data["history"] for data in loaded]))
        self.history_mask = torch.from_numpy(np.concatenate([data["history_mask"] for data in loaded]))
        self.future = torch.from_numpy(np.concatenate([data["future"] for data in loaded]))
        self.node_mask = torch.from_numpy(np.concatenate([data["node_mask"] for data in loaded]))
        self.static = torch.from_numpy(np.concatenate([data["static"] for data in loaded]))

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, index: int):
        return (
            self.history[index],
            self.history_mask[index],
            self.future[index],
            self.node_mask[index],
            self.static[index],
        )


def ground_truth_risk(model: RFIGN, future_m: torch.Tensor, static: torch.Tensor, node_mask: torch.Tensor):
    return model.fuzzy_risk(future_m / model.position_scale_m, static, node_mask)


def ground_truth_close_event(
    future_m: torch.Tensor,
    node_mask: torch.Tensor,
    threshold_nm: float,
) -> torch.Tensor:
    difference = future_m[:, :, None] - future_m[:, None, :]
    minimum_distance_m = torch.linalg.vector_norm(difference, dim=-1).amin(dim=-1)
    valid = node_mask[:, :, None] & node_mask[:, None, :]
    eye = torch.eye(valid.shape[1], dtype=torch.bool, device=future_m.device).unsqueeze(0)
    return ((minimum_distance_m <= threshold_nm * 1852.0) & valid & ~eye).float()


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(scores), 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (scores >= lower) & (scores < upper if upper < 1.0 else scores <= upper)
        if selected.any():
            ece += selected.sum() / total * abs(scores[selected].mean() - labels[selected].mean())
    return float(ece)


def bernoulli_kl(probability: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-5, 1.0 - 1e-5)
    reference = reference.clamp(1e-5, 1.0 - 1e-5)
    return probability * torch.log(probability / reference) + (1.0 - probability) * torch.log(
        (1.0 - probability) / (1.0 - reference)
    )


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    close_event_target: bool = False,
    close_threshold_nm: float = 0.25,
    moderate_threshold_nm: float = 0.5,
    anchor_loss_weight: float = 0.2,
    moderate_loss_weight: float = 0.1,
    calibration_weight: float = 0.1,
):
    model.train(train)
    total_loss = total_ade = total_fde = total_nodes = 0.0
    all_scores, all_labels = [], []
    all_moderate_scores, all_moderate_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for history, history_mask, future, node_mask, static in loader:
            history, history_mask, future = history.to(device), history_mask.to(device), future.to(device)
            node_mask, static = node_mask.to(device), static.to(device)
            output = model(history, history_mask, node_mask, static)
            target = future / model.position_scale_m
            valid = node_mask[:, :, None, None]
            position_error = nn.functional.smooth_l1_loss(output["prediction"], target, reduction="none")
            position_loss = (position_error * valid).sum() / (valid.sum() * target.shape[2] * 2)
            if close_event_target:
                target_risk = ground_truth_close_event(
                    future, node_mask, close_threshold_nm
                ).detach()
                target_moderate_risk = ground_truth_close_event(
                    future, node_mask, moderate_threshold_nm
                ).detach()
            else:
                target_risk = ground_truth_risk(model, future, static, node_mask).detach()
                target_moderate_risk = target_risk
            pair_valid = node_mask[:, :, None] & node_mask[:, None, :]
            eye = torch.eye(pair_valid.shape[1], dtype=torch.bool, device=device).unsqueeze(0)
            pair_valid = pair_valid & ~eye
            risk_loss = nn.functional.binary_cross_entropy(output["risk"][pair_valid], target_risk[pair_valid])
            loss = position_loss + 0.2 * risk_loss
            if model.constrained_risk_feedback:
                anchor_loss = nn.functional.binary_cross_entropy(
                    output["anchor_risk"][pair_valid], target_risk[pair_valid]
                )
                calibration_loss = bernoulli_kl(
                    output["risk"][pair_valid], output["anchor_risk"][pair_valid].detach()
                ).mean()
                moderate_loss = nn.functional.binary_cross_entropy(
                    output["moderate_risk"][pair_valid],
                    target_moderate_risk[pair_valid],
                )
                loss = (
                    loss
                    + anchor_loss_weight * anchor_loss
                    + moderate_loss_weight * moderate_loss
                    + calibration_weight * calibration_loss
                )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

            distance_m = torch.linalg.vector_norm(output["prediction"] * model.position_scale_m - future, dim=-1)
            nodes = node_mask.sum().item()
            total_loss += loss.item() * nodes
            total_ade += (distance_m * node_mask.unsqueeze(-1)).sum().item() / distance_m.shape[-1]
            total_fde += (distance_m[:, :, -1] * node_mask).sum().item()
            total_nodes += nodes
            if not train:
                upper = torch.triu(pair_valid, diagonal=1)
                all_scores.append(output["risk"][upper].detach().cpu().numpy())
                all_labels.append((target_risk[upper] >= 0.5).cpu().numpy())
                if model.constrained_risk_feedback:
                    all_moderate_scores.append(
                        output["moderate_risk"][upper].detach().cpu().numpy()
                    )
                    all_moderate_labels.append(
                        (target_moderate_risk[upper] >= 0.5).cpu().numpy()
                    )
    metrics = {
        "loss": total_loss / max(total_nodes, 1),
        "ade_nm": total_ade / max(total_nodes, 1) / 1852.0,
        "fde_nm": total_fde / max(total_nodes, 1) / 1852.0,
    }
    if all_scores:
        scores, labels = np.concatenate(all_scores), np.concatenate(all_labels)
        metrics["risk_auprc"] = float(average_precision_score(labels, scores)) if labels.any() else float("nan")
        metrics["risk_positive_rate"] = float(labels.mean())
        metrics["risk_brier"] = float(np.mean((scores - labels) ** 2))
        metrics["risk_ece"] = expected_calibration_error(labels.astype(float), scores)
    if all_moderate_scores:
        moderate_scores = np.concatenate(all_moderate_scores)
        moderate_labels = np.concatenate(all_moderate_labels)
        metrics["moderate_risk_auprc"] = float(
            average_precision_score(moderate_labels, moderate_scores)
        )
        metrics["moderate_risk_brier"] = float(
            np.mean((moderate_scores - moderate_labels) ** 2)
        )
        metrics["moderate_risk_ece"] = expected_calibration_error(
            moderate_labels.astype(float), moderate_scores
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--val", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--test",
        type=Path,
        nargs="+",
        help="Locked test scenes. These are evaluated only once after validation-based selection.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--detach-risk-feedback", action="store_true")
    parser.add_argument("--isolated-risk-head", action="store_true")
    parser.add_argument("--constrained-risk-feedback", action="store_true")
    parser.add_argument("--close-event-target", action="store_true")
    parser.add_argument("--close-threshold-nm", type=float, default=0.25)
    parser.add_argument("--moderate-threshold-nm", type=float, default=0.5)
    parser.add_argument("--residual-bound", type=float, default=1.0)
    parser.add_argument("--anchor-loss-weight", type=float, default=0.2)
    parser.add_argument("--moderate-loss-weight", type=float, default=0.1)
    parser.add_argument("--calibration-weight", type=float, default=0.1)
    parser.add_argument("--selection-metric", choices=["ade", "joint"], default="ade")
    parser.add_argument("--seed", type=int, default=42)
    # --- ablation switches (F5 / E stage); all default to pre-ablation behaviour
    parser.add_argument("--anchor-from-fuzzy", action="store_true")
    parser.add_argument("--unbounded-residual", action="store_true")
    parser.add_argument("--free-ordinal", action="store_true")
    # ---
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    train_data, val_data = SceneDataset(args.train), SceneDataset(args.val)
    test_data = SceneDataset(args.test) if args.test else None
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = (
        DataLoader(test_data, args.batch_size, shuffle=False, num_workers=args.workers)
        if test_data is not None
        else None
    )
    if args.no_feedback and args.detach_risk_feedback:
        parser.error("--no-feedback and --detach-risk-feedback are mutually exclusive")
    model = RFIGN(
        feedback=not args.no_feedback,
        detach_risk_feedback=args.detach_risk_feedback,
        isolated_risk_head=args.isolated_risk_head,
        constrained_risk_feedback=args.constrained_risk_feedback,
        residual_bound=args.residual_bound,
        anchor_from_fuzzy=args.anchor_from_fuzzy,
        unbounded_residual=args.unbounded_residual,
        free_ordinal=args.free_ordinal,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    args.out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_epoch = None
    log = []
    for epoch in range(1, args.epochs + 1):
        epoch_options = dict(
            close_event_target=args.close_event_target,
            close_threshold_nm=args.close_threshold_nm,
            moderate_threshold_nm=args.moderate_threshold_nm,
            anchor_loss_weight=args.anchor_loss_weight,
            moderate_loss_weight=args.moderate_loss_weight,
            calibration_weight=args.calibration_weight,
        )
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, train=True, **epoch_options
        )
        val_metrics = run_epoch(
            model, val_loader, optimizer, device, train=False, **epoch_options
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        print(json.dumps(row), flush=True)
        log.append(row)
        selection_value = (
            val_metrics["ade_nm"] if args.selection_metric == "ade" else val_metrics["loss"]
        )
        if selection_value < best:
            best = selection_value
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "metrics": val_metrics, "epoch": epoch},
                args.out / "best.pt",
            )
    (args.out / "metrics.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    if test_loader is not None:
        checkpoint = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        test_metrics = run_epoch(
            model, test_loader, optimizer=None, device=device, train=False, **epoch_options
        )
        final = {
            "selected_epoch": int(checkpoint["epoch"]),
            "selection_metric": args.selection_metric,
            "validation_metrics_at_selected_epoch": checkpoint["metrics"],
            "test_metrics": test_metrics,
        }
        (args.out / "final_test_metrics.json").write_text(
            json.dumps(final, indent=2), encoding="utf-8"
        )
        print(json.dumps({"final_test": final}), flush=True)


if __name__ == "__main__":
    main()
