# KAORF — 锁定独立时间盲测结果（New York, locked independent temporal blind test）

> **性质说明**：本表全部数字来自 `final_test_metrics.json → test_metrics`，
> 即 2025 年 11–12 月的 8 个纽约港日期，在**由验证集选出的 checkpoint 上评估一次**所得。
> 这 8 个日期未参与任何 epoch 选择、超参搜索或早停。
> **本表不是开发集（validation）结果**，两者必须严格分开报告、不可混用。

- 配对种子（两变体齐全）：[17, 29, 42, 71, 101, 131, 211, 307, 509, 887]
- 仅 constrained_feedback：无
- 仅 constrained_no_feedback：无
- 完全缺失：无

## 1. 分组描述统计（mean ± sd）

| 指标 | constrained_feedback | constrained_no_feedback | 差值 (A−B) |
|---|---|---|---|
| ADE (nm) | 0.8753 ± 0.0090 | 0.8944 ± 0.0062 | -0.0191 |
| FDE (nm) | 1.4212 ± 0.0132 | 1.4460 ± 0.0100 | -0.0249 |
| Severe risk AUPRC | 0.4503 ± 0.0014 | 0.4503 ± 0.0013 | +0.0000 |
| Severe risk Brier | 0.0663 ± 0.0001 | 0.0663 ± 0.0001 | -0.0000 |
| Severe risk ECE | 0.0072 ± 0.0011 | 0.0074 ± 0.0010 | -0.0002 |
| Moderate risk AUPRC | 0.5966 ± 0.0012 | 0.5982 ± 0.0011 | -0.0015 |
| Moderate risk Brier | 0.1099 ± 0.0001 | 0.1098 ± 0.0001 | +0.0002 |
| Moderate risk ECE | 0.0160 ± 0.0027 | 0.0164 ± 0.0029 | -0.0004 |

## 2. 逐指标配对检验（10 个配对种子）

| 指标 | 指标方向 | A 胜 | B 胜 | 配对 t | p(t) | Wilcoxon W | p(W) | Cohen's dz |
|---|---|---|---|---|---|---|---|---|
| ADE | 越低越好 | 10 | 0 | 6.779 | 8.091e-05 | 0.0 | 0.001953 | 2.14 |
| FDE | 越低越好 | 10 | 0 | 6.296 | 0.0001416 | 0.0 | 0.001953 | 1.99 |
| Severe risk AUPRC | 越高越好 | 5 | 5 | 0.058 | 0.955 | 25.0 | 0.8457 | 0.02 |
| Severe risk Brier | 越低越好 | 6 | 4 | 0.084 | 0.9349 | 26.0 | 0.9219 | 0.03 |
| Severe risk ECE | 越低越好 | 7 | 3 | 1.352 | 0.2093 | 14.0 | 0.1934 | 0.43 |
| Moderate risk AUPRC | 越高越好 | 0 | 10 | -7.173 | 5.233e-05 | 0.0 | 0.001953 | -2.27 |
| Moderate risk Brier | 越低越好 | 0 | 10 | -5.982 | 0.0002071 | 0.0 | 0.001953 | -1.89 |
| Moderate risk ECE | 越低越好 | 8 | 2 | 2.049 | 0.07078 | 9.0 | 0.06445 | 0.65 |

- 方向列说明：本例中 ADE/FDE/Brier/ECE 越小越好，AUPRC 越大越好；
  **差值统一取「A 优于 B」为正**，故差值为正即 `constrained_feedback` 更优。
- `p(t)` 为配对 t 检验（双侧），`p(W)` 为 Wilcoxon 符号秩检验（双侧，正态性不满足时以它为准）。
- `dz` 为配对 Cohen's dz = mean(diff) / sd(diff)。

## 3. 判读

以上全部结论仅适用于**锁定独立时间盲测**；与开发集（validation）上的结论分别陈述，
不得相互引用作为证据。盲测结果**不得**再用于任何调参或模型修改——本轮结束后若改动模型，
该盲测即失效，必须重新走一轮完整的 train/val/blind_test。
