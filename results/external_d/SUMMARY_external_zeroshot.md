# D 阶段：外港零微调迁移（F3）——锁定 NY 检查点直接跨港评估

- 协议：**零训练、零微调、零再选择**。检查点取自 NY 验证集选择（`--selection-metric joint`）的锁定盲测 run；
  外港场景在预注册清单中标记为 `external_only` / `allowed_for_tuning=no`，仅用于最终评估。
- **未读取任何 NY 场景文件**（评估器对每个路径断言非 new_york），`blind_test` 的一次性口径不受影响。
- 指标计算**逐字复用** `scripts/train_rfign_dev.py` 的 `run_epoch`，与 NY 盲测同一代码路径；
  另有守卫断言「逐日节点加权均值 == 整批 pooled 值」（相对差 < 1e-6，实测约 6e-9，纯浮点累加序差异）。
- 结果目录：`runs/external_zeroshot_20260923/full48`；配对子集：`runs/external_zeroshot_20260923/sub12`；NY 参照：`runs/final_ny_temporal_10seed_20260921`
- 配对种子（10 个）：17, 29, 42, 71, 101, 131, 211, 307, 509, 887

> 不确定度用**按日聚类的 bootstrap**（预注册的抽样单位是日）：在每个种子内对日重抽样，再对种子取均值，
> 同时反映日的抽样波动与种子间波动。Chesapeake Bay 日均仅约 127 个场景（NY 的 1/11），CI 尤其必要。

## 1. 外港零微调迁移（`constrained_feedback`，全 48 日）

| 港口 | ADE (nm)↓ | FDE (nm)↓ | severe AUPRC↑ | severe Brier↓ | severe ECE↓ | moderate AUPRC↑ | moderate Brier↓ | moderate ECE↓ | 日数 | 场景数 | 节点样本 | 严重/中危正例率 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Houston-Galveston | 0.8740 ± 0.0170 | 1.5329 ± 0.0271 | 0.5029 ± 0.0021 | 0.0629 ± 0.0002 | 0.0138 ± 0.0014 | 0.7022 ± 0.0018 | 0.0853 ± 0.0004 | 0.0214 ± 0.0020 | 48 | 85,974 | 522,151 | 0.0938 |
| Chesapeake Bay | 1.1290 ± 0.0172 | 2.0560 ± 0.0303 | 0.6161 ± 0.0012 | 0.0539 ± 0.0001 | 0.0080 ± 0.0012 | 0.6956 ± 0.0021 | 0.0852 ± 0.0004 | 0.0117 ± 0.0007 | 48 | 19,517 | 56,518 | 0.0947 |

## 2. 按日聚类 bootstrap 95% CI（同一 10 个种子）

| 港口 | 指标 | 点估计 | 95% CI 下 | 95% CI 上 | 权重口径 |
|---|---|---|---|---|---|
| Houston-Galveston | ADE (nm) | 0.8740 | 0.8658 | 0.8826 | 节点加权 |
| Houston-Galveston | FDE (nm) | 1.5329 | 1.5175 | 1.5477 | 节点加权 |
| Houston-Galveston | severe AUPRC | 0.5099 | 0.5015 | 0.5180 | 日均值 |
| Houston-Galveston | severe Brier | 0.0613 | 0.0594 | 0.0632 | 日均值 |
| Houston-Galveston | severe ECE | 0.0161 | 0.0148 | 0.0176 | 日均值 |
| Houston-Galveston | moderate AUPRC | 0.6975 | 0.6903 | 0.7042 | 日均值 |
| Houston-Galveston | moderate Brier | 0.0835 | 0.0812 | 0.0856 | 日均值 |
| Houston-Galveston | moderate ECE | 0.0232 | 0.0219 | 0.0245 | 日均值 |
| Chesapeake Bay | ADE (nm) | 1.1290 | 1.1109 | 1.1493 | 节点加权 |
| Chesapeake Bay | FDE (nm) | 2.0560 | 2.0181 | 2.0977 | 节点加权 |
| Chesapeake Bay | severe AUPRC | 0.6458 | 0.6159 | 0.6760 | 日均值 |
| Chesapeake Bay | severe Brier | 0.0614 | 0.0559 | 0.0676 | 日均值 |
| Chesapeake Bay | severe ECE | 0.0299 | 0.0238 | 0.0371 | 日均值 |
| Chesapeake Bay | moderate AUPRC | 0.7153 | 0.6933 | 0.7364 | 日均值 |
| Chesapeake Bay | moderate Brier | 0.0902 | 0.0860 | 0.0948 | 日均值 |
| Chesapeake Bay | moderate ECE | 0.0363 | 0.0307 | 0.0429 | 日均值 |

## 3. 与 NY 盲测参照的落差（域偏移敏感性）

| 域 | ADE (nm) | FDE (nm) | severe AUPRC | severe Brier | severe ECE | moderate AUPRC | moderate Brier | moderate ECE |
|---|---|---|---|---|---|---|---|---|
| **New York（盲测，8 日）** | 0.8753 ± 0.0090 | 1.4212 ± 0.0132 | 0.4503 ± 0.0014 | 0.0663 ± 0.0001 | 0.0072 ± 0.0011 | 0.5966 ± 0.0012 | 0.1099 ± 0.0001 | 0.0160 ± 0.0027 |
| Houston-Galveston（零微调） | 0.8740 | 1.5329 | 0.5029 | 0.0629 | 0.0138 | 0.7022 | 0.0853 | 0.0214 |
| Chesapeake Bay（零微调） | 1.1290 | 2.0560 | 0.6161 | 0.0539 | 0.0080 | 0.6956 | 0.0852 | 0.0117 |

| 港口 | 指标 | NY | 外港 | Δ(外港−NY) | 相对变化 |
|---|---|---|---|---|---|
| Houston-Galveston | ADE (nm) | 0.8753 | 0.8740 | -0.0013 | -0.15% |
| Houston-Galveston | FDE (nm) | 1.4212 | 1.5329 | +0.1117 | +7.86% |
| Houston-Galveston | severe AUPRC | 0.4503 | 0.5029 | +0.0527 | +11.70% |
| Houston-Galveston | severe Brier | 0.0663 | 0.0629 | -0.0034 | -5.13% |
| Houston-Galveston | severe ECE | 0.0072 | 0.0138 | +0.0066 | +92.27% |
| Houston-Galveston | moderate AUPRC | 0.5966 | 0.7022 | +0.1056 | +17.69% |
| Houston-Galveston | moderate Brier | 0.1099 | 0.0853 | -0.0246 | -22.36% |
| Houston-Galveston | moderate ECE | 0.0160 | 0.0214 | +0.0054 | +33.69% |
| Chesapeake Bay | ADE (nm) | 0.8753 | 1.1290 | +0.2537 | +28.98% |
| Chesapeake Bay | FDE (nm) | 1.4212 | 2.0560 | +0.6349 | +44.67% |
| Chesapeake Bay | severe AUPRC | 0.4503 | 0.6161 | +0.1659 | +36.84% |
| Chesapeake Bay | severe Brier | 0.0663 | 0.0539 | -0.0123 | -18.62% |
| Chesapeake Bay | severe ECE | 0.0072 | 0.0080 | +0.0008 | +11.78% |
| Chesapeake Bay | moderate AUPRC | 0.5966 | 0.6956 | +0.0990 | +16.59% |
| Chesapeake Bay | moderate Brier | 0.1099 | 0.0852 | -0.0247 | -22.49% |
| Chesapeake Bay | moderate ECE | 0.0160 | 0.0117 | -0.0043 | -26.89% |

> 口径提醒：NY 是 **8 日**时间盲测（抽样单位 8 天），外港是 **48 日**评估（抽样单位 48 天），
> 两者的 CI 不可直接比较；本表只作描述性的域偏移读数。

## 4. 共享 12 日分层子集上的配对机制对比

- 两个变体在**完全相同的日期**上评估（每月 1 日、层轮转），因此可种子级配对：`constrained_feedback` vs `constrained_no_feedback`。
- 这是「机制在域偏移下是否仍然有用」的检查；NY 域内的机制结论以 E 阶段消融为准。

| 港口 | 指标 | 参考 mean±sd | 基线 mean±sd | Δ(参考−基线) | 参考胜 | 配对 t | p(t) | Wilcoxon | p(W) | dz |
|---|---|---|---|---|---|---|---|---|---|---|
| Houston-Galveston | ADE (nm) | 0.8759 ± 0.0182 | 0.8836 ± 0.0140 | -0.0077 | 7/10 | -1.750 | 0.1141 | 12.0 | 0.1309 | -0.55 |
| Houston-Galveston | FDE (nm) | 1.5396 ± 0.0308 | 1.5521 ± 0.0237 | -0.0125 | 8/10 | -1.761 | 0.1120 | 10.0 | 0.0840 | -0.56 |
| Houston-Galveston | severe AUPRC | 0.5018 ± 0.0025 | 0.5029 ± 0.0030 | -0.0011 | 3/10 | -3.226 | 0.0104 | 6.0 | 0.0273 | -1.02 |
| Houston-Galveston | severe Brier | 0.0616 ± 0.0002 | 0.0617 ± 0.0002 | -0.0000 | 5/10 | -0.533 | 0.6068 | 27.0 | 1.0000 | -0.17 |
| Houston-Galveston | severe ECE | 0.0137 ± 0.0014 | 0.0146 ± 0.0016 | -0.0009 | 8/10 | -2.464 | 0.0359 | 8.0 | 0.0488 | -0.78 |
| Houston-Galveston | moderate AUPRC | 0.7029 ± 0.0018 | 0.7044 ± 0.0025 | -0.0015 | 0/10 | -4.601 | 0.0013 | 0.0 | 0.0020 | -1.45 |
| Houston-Galveston | moderate Brier | 0.0840 ± 0.0004 | 0.0839 ± 0.0004 | +0.0001 | 4/10 | 1.889 | 0.0915 | 12.0 | 0.1309 | +0.60 |
| Houston-Galveston | moderate ECE | 0.0209 ± 0.0020 | 0.0220 ± 0.0021 | -0.0011 | 7/10 | -2.491 | 0.0344 | 8.0 | 0.0488 | -0.79 |
| Chesapeake Bay | ADE (nm) | 1.1107 ± 0.0180 | 1.1140 ± 0.0153 | -0.0033 | 5/10 | -0.731 | 0.4836 | 21.0 | 0.5566 | -0.23 |
| Chesapeake Bay | FDE (nm) | 2.0224 ± 0.0286 | 2.0313 ± 0.0315 | -0.0089 | 5/10 | -0.906 | 0.3887 | 21.0 | 0.5566 | -0.29 |
| Chesapeake Bay | severe AUPRC | 0.6176 ± 0.0024 | 0.6189 ± 0.0025 | -0.0013 | 1/10 | -4.218 | 0.0022 | 1.0 | 0.0039 | -1.33 |
| Chesapeake Bay | severe Brier | 0.0548 ± 0.0002 | 0.0548 ± 0.0001 | +0.0000 | 4/10 | 1.185 | 0.2663 | 17.0 | 0.3223 | +0.37 |
| Chesapeake Bay | severe ECE | 0.0062 ± 0.0009 | 0.0055 ± 0.0012 | +0.0006 | 3/10 | 2.165 | 0.0586 | 10.0 | 0.0840 | +0.68 |
| Chesapeake Bay | moderate AUPRC | 0.6992 ± 0.0025 | 0.7019 ± 0.0025 | -0.0027 | 0/10 | -7.588 | 3.37e-05 | 0.0 | 0.0020 | -2.40 |
| Chesapeake Bay | moderate Brier | 0.0850 ± 0.0004 | 0.0847 ± 0.0004 | +0.0004 | 0/10 | 6.345 | 1.34e-04 | 0.0 | 0.0020 | +2.01 |
| Chesapeake Bay | moderate ECE | 0.0130 ± 0.0031 | 0.0124 ± 0.0028 | +0.0006 | 5/10 | 1.199 | 0.2612 | 19.0 | 0.4316 | +0.38 |

## 5. 判读注意

- **外港从未参与任何选择**：没有早期停止、没有阈值调节、没有超参搜索；检查点与其阈值全部来自 NY 训练流程。
- **Chesapeake Bay 极稀疏**（日均约 127 场景）：单日波动可主导均值，因此上面给了按日聚类 CI 与正例率；
  引用时须同时报场景数/节点样本数与正例率，不能只报均值。
- **Houston-Galveston 体量约为 NY 的 1.6 倍**（2 272 场景/日）：本表的 pooled 值按节点样本自然加权，
  与 NY 盲测的加权方式一致。
- 风险指标的分级阈值沿用锁定流程（严重 0.25 nm、中危 0.5 nm），未针对外港调整。
- 本阶段**不触碰 blind_test**，也不改变任何已有汇总；NY 结论仍以 `SUMMARY_blind_test.md` 为准。

