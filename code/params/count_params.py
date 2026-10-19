# -*- coding: utf-8 -*-
"""Parameter count for the three compared architectures (Table IX material).

Instantiates the exact locked configurations:
  * KAORF (constrained_feedback): feedback + detached + bounded residual 0.25
  * No-feedback baseline:         feedback=False (otherwise identical defaults)
  * C1 plain Transformer:         PlainTransformer defaults (capacity-matched trunk)

All three share hidden_dim=96, heads=4, layers=2, feature_dim=8,
history_steps=20, future_steps=6.  Counting is definition-level (module tree),
independent of loss switches such as --close-event-target, which do not add
parameters.  A random-tensor forward pass smoke-test confirms each instance
constructs and runs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rf = load_module(BASE / "kaorf_ablation" / "rfign_model.py", "rfign_model_count")
pt = load_module(BASE / "kaorf_f2c1" / "plain_transformer.py", "plain_transformer_count")


def count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    return {"total": total, "trainable": trainable, "buffers": buffers}


def per_module(model):
    out = {name: sum(p.numel() for p in child.parameters()) for name, child in model.named_children()}
    direct = sum(p.numel() for n, p in model.named_parameters() if "." not in n)
    if direct:
        out["time_embedding(direct)"] = direct
    return out


models = {
    "KAORF (constrained_feedback)": rf.RFIGN(
        feedback=True, detach_risk_feedback=True, constrained_risk_feedback=True, residual_bound=0.25
    ),
    "No-feedback baseline": rf.RFIGN(feedback=False),
    "C1 plain Transformer": pt.PlainTransformer(),
}

report = {}
for name, m in models.items():
    c = count(m)
    c["per_module"] = per_module(m)
    report[name] = c
    print(f"{name}: total={c['total']:,} trainable={c['trainable']:,}")

# Forward smoke test on random tensors (values are meaningless; shapes are not).
B, N = 2, 8
hist = torch.randn(B, N, 20, 8)
hm = torch.ones(B, N, 20, dtype=torch.bool)
nm = torch.ones(B, N, dtype=torch.bool)
st = torch.rand(B, N, 2) * 100
with torch.no_grad():
    outs = [m(hist, hm, nm, st) for m in models.values()]
shapes = {n: tuple(o["prediction"].shape) for n, o in zip(models, outs)}
print("forward OK:", shapes)
report["forward_shapes"] = shapes

out_path = Path(__file__).with_name("param_report.json")
out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
print("written:", out_path)
