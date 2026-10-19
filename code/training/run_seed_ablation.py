"""Run paired RFIGN feedback/no-feedback experiments over multiple seeds.

The two variants use the same seed, split, optimizer settings, and epoch count,
so their best validation metrics form paired observations for inference.

E-stage ablation arms added 2026-09-23 (F5 mechanism study). Every new arm is a
single-factor change against the `full` reference arm:
  full                    -- the complete constrained model (== constrained_feedback)
  abl_no_anchor           -- learned kinematic anchor replaced by the fuzzy prior
  abl_no_bounded_residual -- unbounded, unsaturated log-odds residual
  abl_no_stop_gradient    -- risk message no longer detached (no stop-gradient)
  abl_no_ordinal          -- non-negative ordinal increment removed
  abl_single_threshold    -- moderate target uses the severe 0.25 nm threshold
  abl_fuzzy_only          -- fuzzy-domain proxy targets instead of close-event labels
Ablation arms are evaluated on train/val ONLY; --test is never passed for them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ABLATION_VARIANTS = [
    "full",
    "abl_no_anchor",
    "abl_no_bounded_residual",
    "abl_no_stop_gradient",
    "abl_no_ordinal",
    "abl_single_threshold",
    "abl_fuzzy_only",
]


def constrained_flags(
    args: argparse.Namespace,
    *,
    detach: bool = True,
    close_event: bool = True,
    moderate_nm: float | None = None,
    extra: tuple[str, ...] = (),
) -> list[str]:
    """Flags shared by the constrained-risk-feedback family."""
    flags = ["--constrained-risk-feedback"]
    if close_event:
        flags.append("--close-event-target")
    if detach:
        flags.append("--detach-risk-feedback")
    flags.extend(
        (
            "--selection-metric",
            "joint",
            "--residual-bound",
            str(args.residual_bound),
            "--anchor-loss-weight",
            str(args.anchor_loss_weight),
            "--moderate-loss-weight",
            str(args.moderate_loss_weight),
            "--calibration-weight",
            str(args.calibration_weight),
            "--close-threshold-nm",
            str(args.close_threshold_nm),
            "--moderate-threshold-nm",
            str(args.moderate_threshold_nm if moderate_nm is None else moderate_nm),
        )
    )
    flags.extend(extra)
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--val", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 42, 71, 101])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual-bound", type=float, default=1.0)
    parser.add_argument("--anchor-loss-weight", type=float, default=0.2)
    parser.add_argument("--moderate-loss-weight", type=float, default=0.1)
    parser.add_argument("--calibration-weight", type=float, default=0.1)
    parser.add_argument("--close-threshold-nm", type=float, default=0.25)
    parser.add_argument("--moderate-threshold-nm", type=float, default=0.5)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "feedback",
            "detached_feedback",
            "isolated_feedback",
            "isolated_no_feedback",
            "constrained_feedback",
            "constrained_no_feedback",
            "no_feedback",
        ]
        + ABLATION_VARIANTS,
        default=["feedback", "no_feedback"],
    )
    args = parser.parse_args()

    trainer = Path(__file__).with_name("train_rfign_dev.py")
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = args.out / f"seed_{seed}" / variant
            metrics = run_dir / "metrics.json"
            final_test = run_dir / "final_test_metrics.json"
            if metrics.exists() and (not args.test or final_test.exists()):
                print(f"skip completed run: {run_dir}", flush=True)
                continue
            command = [
                sys.executable,
                str(trainer),
                "--train",
                *map(str, args.train),
                "--val",
                *map(str, args.val),
                "--out",
                str(run_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
                "--device",
                args.device,
                "--seed",
                str(seed),
            ]
            if args.test:
                command.extend(("--test", *map(str, args.test)))
            if variant == "no_feedback":
                command.append("--no-feedback")
            elif variant == "detached_feedback":
                command.append("--detach-risk-feedback")
            elif variant == "isolated_feedback":
                command.extend(("--isolated-risk-head", "--detach-risk-feedback"))
            elif variant == "isolated_no_feedback":
                command.extend(("--isolated-risk-head", "--no-feedback"))
            elif variant in ("constrained_feedback", "full"):
                command.extend(constrained_flags(args))
            elif variant == "constrained_no_feedback":
                command.extend(constrained_flags(args, detach=False))
                command.append("--no-feedback")
            # ---------------------------------------------------- E-stage ablations
            elif variant == "abl_no_anchor":
                command.extend(constrained_flags(args, extra=("--anchor-from-fuzzy",)))
            elif variant == "abl_no_bounded_residual":
                command.extend(constrained_flags(args, extra=("--unbounded-residual",)))
            elif variant == "abl_no_stop_gradient":
                command.extend(constrained_flags(args, detach=False))
            elif variant == "abl_no_ordinal":
                command.extend(constrained_flags(args, extra=("--free-ordinal",)))
            elif variant == "abl_single_threshold":
                # moderate collapses onto the severe 0.25 nm threshold
                command.extend(constrained_flags(args, moderate_nm=args.close_threshold_nm))
            elif variant == "abl_fuzzy_only":
                command.extend(constrained_flags(args, close_event=False))
            # ----------------------------------------------------------------------
            print(f"start seed={seed} variant={variant}", flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
