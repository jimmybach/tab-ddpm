# Concept Drift Benchmark Datasets / 概念漂移基准数据集

本文件夹包含用于评估 LLM 表格数据合成能力的概念漂移基准数据集，共 **15 个数据集**，覆盖三种漂移类型。

This folder contains concept drift benchmark datasets for evaluating LLM-based tabular data synthesis. It includes **15 datasets** covering three types of drift.

---

## 漂移类型定义 / Drift Type Definitions

| 符号 / Symbol | 名称 / Name | 数学定义 / Mathematical Definition |
|---|---|---|
| **P(Y)** | 先验概率漂移 Prior Probability Shift | P(X) 和 P(Y\|X) 不变，仅 P(Y) 变化 / Only P(Y) changes; P(X) and P(Y\|X) remain constant |
| **P(X)** | 协变量漂移 Covariate Shift | P(Y\|X) 不变，仅 P(X) 变化 / Only P(X) changes; P(Y\|X) remains constant |
| **P(Y\|X)** | 真实概念漂移 Real Concept Drift | P(Y\|X) 发生变化，对 P(X) 无约束 / P(Y\|X) changes; no strict constraint on P(X) |

---

## 数据集总览 / Dataset Overview

### 规格 / Specifications

- **实例数 / Instances:** 4,000（合成数据集及可缩减真实数据集）| 13,910（Gas Sensor）| 20,000（Electricity）
- **窗口 / Window:** w = 100，共 40 个窗口（适用于 4,000 实例数据集）
- **漂移点 / Drift point:** t = 2,000（双段结构数据集）

---

### P(Y) — 先验概率漂移 / Prior Probability Shift

P(X) 和 P(Y|X) 全程不变，仅通过池化重采样（pool-based resampling）控制正例比例，保证数学纯净性。

P(X) and P(Y|X) remain constant throughout. Class proportions are controlled via pool-based resampling, guaranteeing mathematical purity.

| # | 数据集 / Dataset | 来源 / Source | 实例数 / Instances | 漂移模式 / Drift Mode | 漂移幅度 / Drift Range |
|---|---|---|---|---|---|
| 1 | **Creditcard** | OpenML `creditcard` v1 | 4,000 | Abrupt | 0.15% → 4.75%（×31） |
| 2 | **Bank Marketing** | OpenML data\_id=1461 | 4,000 | Abrupt | 10% → 40%（×4） |
| 3 | **Adult** | UCI Adult (Census Income) | 4,000 | Abrupt + Gradual | 10% → 70% → 30% |
| 4 | **Agrawal** | River `synth.Agrawal` fn=0 | 4,000 | Abrupt | 20% → 80%（×4） |
| 5 | **SEA** | River `synth.SEA` variant=0 | 4,000 | Gradual (sigmoid) | 20% → 80% |

**各数据集说明 / Dataset Notes:**

- **Creditcard:** 正例为欺诈交易，V1–V28 为 PCA 匿名特征（原始设计，不可还原），仅 `Amount` 为原始特征。`duration` 字段不存在于此数据集。
  *Positive class = fraud. V1–V28 are PCA-anonymised by design (privacy-preserving). Only `Amount` is original.*

- **Bank Marketing:** 正例为客户订购定期存款。已删除 `duration`（通话时长）特征——该字段仅通话结束后可知，保留会造成数据泄漏。特征名已还原为 UCI 原始命名。
  *Positive class = subscribed. `duration` removed to prevent data leakage. Feature names restored to original UCI names.*

- **Adult:** 正例为年收入超过 50K 美元。双段漂移：t=2,000 处突变至 70%，t=2,500 起线性下降至 30%（每窗口约 −2.86%）。
  *Positive class = income >50K. Two-phase drift: abrupt jump to 70% at t=2,000, then linear decline to 30% starting t=2,500.*

- **Agrawal (P(Y)):** River fn=0 全程固定（y=1 当且仅当 age<40 或 age≥60），P(Y|X) 由生成器严格保证不变。
  *River fn=0 fixed throughout. P(Y|X) strictly guaranteed constant by the generator.*

- **SEA:** sigmoid 平滑过渡，steepness=300（等比缩放自原始 1,500，保持过渡区占比约 45%）。
  *Sigmoid transition with steepness=300 (proportionally scaled from 1,500, preserving ~45% transition coverage).*

---

### P(X) — 协变量漂移 / Covariate Shift

P(Y|X) 全程不变，仅特征分布发生变化。

P(Y|X) remains constant throughout. Only the feature distribution shifts.

| # | 数据集 / Dataset | 来源 / Source | 实例数 / Instances | 漂移模式 / Drift Mode | 漂移机制 / Drift Mechanism |
|---|---|---|---|---|---|
| 6 | **Gas Sensor** | OpenML data\_id=1476 | 13,910 | Gradual (real) | 传感器随时间退化，6 类气体，10 个批次 |
| 7 | **Agrawal** | River `synth.Agrawal` fn=0 | 4,000 | Abrupt | salary +40,000; loan +100,000 at t=2,000 |
| 8 | **Gaussian Mixture** | Custom (numpy) | 4,000 | Gradual (sigmoid) | x2 均值从 0 漂至 5，平行于决策边界 |

**各数据集说明 / Dataset Notes:**

- **Gas Sensor:** 真实传感器退化轨迹，保留原始大小（13,910 条），不做人工截取或重采样。包含 128 个传感器特征（V1–V128）。
  *Authentic sensor degradation trajectory. Retained at original size (13,910). Contains 128 sensor features (V1–V128).*

- **Agrawal (P(X)):** fn=0 标签规则仅依赖 age（y=1 iff age<40 或 age≥60），因此 salary/loan 的位移对 P(Y|X) 完全无影响，数学纯净性由构造保证。
  *fn=0 rule depends only on age, so salary/loan shifts have zero effect on P(Y|X) by construction.*

- **Gaussian Mixture:** 决策边界固定为 x1=0；两个类别的均值沿 x2 方向（平行于边界）协同漂移，边界不被触及，P(Y|X) 严格不变。
  *Decision boundary fixed at x1=0. Both class means drift along x2 (parallel to boundary), so P(Y|X) is strictly unchanged.*

---

### P(Y|X) — 真实概念漂移 / Real Concept Drift

P(Y|X) 随时间变化，合成数据集的 P(X) 由构造保证稳定，真实数据集的 P(X) 可能协同漂移。

P(Y|X) changes over time. Synthetic datasets keep P(X) stationary by construction; real datasets may exhibit P(X) co-drift.

| # | 数据集 / Dataset | 来源 / Source | 实例数 / Instances | P(X) 稳定 | 漂移模式 / Drift Mode | 漂移机制 / Drift Mechanism |
|---|---|---|---|---|---|---|
| 9  | **STAGGER** | River `synth.STAGGER` | 4,000 | ✓ | Abrupt ×2 | fn0→fn1 at t=1,333; fn1→fn2 at t=2,666 |
| 10 | **Hyperplane** | River `synth.Hyperplane` | 4,000 | ✓ | Incremental | 10 个权重中 4 个以 Δ=0.001/步持续旋转 |
| 11 | **Mixed** | River `synth.Mixed` | 4,000 | ✓ | Abrupt ×1 | AND→OR 规则于 t=2,000 |
| 12 | **SINE** | River `synth.Sine` | 4,000 | ✓ | Abrupt ×1 | 正弦曲线极性翻转于 t=2,000 |
| 13 | **Electricity** | OpenML data\_id=151 | 20,000 | ✗ | Gradual | NSW 电力市场 1996–1998 年渐进演化 |
| 14 | **Covertype** | OpenML data\_id=1596 | 4,000 | ✗ | Incremental | 地形→植被映射随海拔梯度持续偏移 |
| 15 | **MAGIC Telescope** | OpenML data\_id=1120 | 4,000 | ✗ | Incremental | 伽马/强子判别边界随入射角 fAlpha 偏移 |

**各数据集说明 / Dataset Notes:**

- **STAGGER:** 三个 Boolean 规则在 3 等分时间点切换，P(X) 全程均匀，正例率约 11%→56%→67%。
  *Three Boolean rules switch at two time points. Uniform P(X). Positive rates ~11%→56%→67%.*

- **Hyperplane:** 持续增量漂移，无突变点，适合评估算法对缓慢变化的追踪能力。
  *Continuous incremental drift, no abrupt point. Tests algorithm's ability to track slow change.*

- **Mixed:** 正弦曲线边界不变，仅 AND/OR 组合逻辑翻转，正例率约 13%→67%。
  *Sinusoidal boundary unchanged; only AND/OR logic flips. Positive rates ~13%→67%.*

- **SINE:** 决策边界完全不变，仅极性翻转。由于边界对称，P(Y) 约保持 50%，P(X) 完全静止——P(Y|X) 的翻转在 P(X)/P(Y) 中完全不可见，是最纯净的 P(Y|X) 测试场景。
  *Boundary unchanged, only polarity flips. P(Y)≈50%, P(X) stationary — the flip is invisible in both marginals. Purest P(Y|X) test.*

- **Electricity:** 保留原始大小（20,000）以保证渐变信号充分可见；时间顺序不作任何改变。
  *Retained at 20,000 to ensure gradual drift signal is sufficiently visible. Chronological order unchanged.*

- **Covertype:** 对 581,012 条原始记录按海拔排序后分层采样：20 个海拔带 × 200 条 = 4,000，覆盖完整海拔轨迹（~1,921–3,836m）。
  *Stratified sampling from 581,012 records sorted by elevation: 20 bands × 200 = 4,000, covering full range (~1,921–3,836m).*

- **MAGIC Telescope:** 对 19,020 条原始记录按 fAlpha 排序后分层采样：20 个角度区间 × 200 条 = 4,000，覆盖完整角度范围 0°–90°（不作简单截断）。
  *Stratified sampling from 19,020 records sorted by fAlpha: 20 bins × 200 = 4,000, covering full range 0°–90° (not naive truncation).*

---

## 实验设计原则 / Experimental Design Principles

### 数学纯净性 / Mathematical Purity

| 漂移类型 | P(X) | P(Y\|X) | P(Y) |
|---|---|---|---|
| P(Y) 数据集 | **不变** | **不变** | 变化 |
| P(X) 数据集 | 变化 | **不变** | 稳定（约） |
| P(Y\|X) 数据集 | 合成：不变；真实：可变 | 变化 | 随之变化 |

### 窗口设计 / Window Design

所有 4,000 实例数据集统一使用 **40 个窗口 × 100 条**，与原始 20,000 实例版本（40 个窗口 × 500 条）的窗口数量完全一致，保证评估粒度不变。

All 4,000-instance datasets use **40 windows × 100 instances**, matching the window count of the original 20,000-instance version (40 × 500), preserving evaluation granularity.

### 数据集规模选择 / Dataset Size Rationale

| 数据集 | 实例数 | 原因 |
|---|---|---|
| 合成数据集及可重采样真实数据集 | 4,000 | 40 窗口 × 100，LLM 实验验证足够 |
| Gas Sensor | 13,910 | 真实传感器退化轨迹，不可截取 |
| Electricity | 20,000 | 渐变漂移，需足够长度保证信号可见 |

---

## 文件结构 / File Structure

```
data_concept_drift/
├── README.md                          ← 本文件 / this file
│
├── P(Y)/                              ← 先验概率漂移 / Prior Probability Shift
│   ├── credit/
│   │   ├── generate_prior_shift.py
│   │   ├── creditcard_prior_shift.csv
│   │   └── prior_shift_class_proportion.png
│   ├── bank_marketing/
│   │   ├── generate_bank_prior_shift.py
│   │   ├── bank_prior_shift.csv
│   │   └── bank_prior_shift_class_proportion.png
│   ├── adult/
│   │   ├── generate_adult_prior_shift.py
│   │   ├── adult_prior_shift.csv
│   │   └── adult_prior_shift_class_proportion.png
│   ├── agrawal/
│   │   ├── generate_agrawal_prior_shift.py
│   │   ├── agrawal_prior_shift.csv
│   │   └── agrawal_prior_shift_class_proportion.png
│   └── sea/
│       ├── generate_sea_prior_shift.py
│       ├── sea_prior_shift.csv
│       └── sea_prior_shift_class_proportion.png
│
├── P(X)/                              ← 协变量漂移 / Covariate Shift
│   ├── gas_sensor/
│   │   ├── generate_gas_sensor.py
│   │   ├── gas_sensor_covariate_shift.csv
│   │   └── gas_sensor_feature_drift.png
│   ├── agrawal/
│   │   ├── generate_agrawal_covariate_shift.py
│   │   ├── agrawal_covariate_shift.csv
│   │   └── agrawal_covariate_shift_feature_drift.png
│   └── gaussian_mixture/
│       ├── generate_gaussian_mixture_covariate_shift.py
│       ├── gaussian_mixture_covariate_shift.csv
│       └── gaussian_mixture_feature_drift.png
│
└── P(YX)/                             ← 真实概念漂移 / Real Concept Drift
    ├── stagger/
    │   ├── generate_stagger.py
    │   ├── stagger_concept_drift.csv
    │   └── stagger_concept_drift.png
    ├── hyperplane/
    │   ├── generate_hyperplane.py
    │   ├── hyperplane_concept_drift.csv
    │   └── hyperplane_concept_drift.png
    ├── mixed/
    │   ├── generate_mixed.py
    │   ├── mixed_concept_drift.csv
    │   └── mixed_concept_drift.png
    ├── sine/
    │   ├── generate_sine.py
    │   ├── sine_concept_drift.csv
    │   └── sine_concept_drift.png
    ├── electricity/
    │   ├── generate_electricity.py
    │   ├── electricity_concept_drift.csv
    │   └── electricity_concept_drift.png
    ├── covertype/
    │   ├── generate_covertype.py
    │   ├── covertype_concept_drift.csv
    │   └── covertype_concept_drift.png
    └── magic_telescope/
        ├── generate_magic_telescope.py
        ├── magic_telescope_concept_drift.csv
        └── magic_telescope_concept_drift.png
```

---

## CSV 格式 / CSV Format

所有数据集均包含时间索引列 `t` 和标签列 `y`，其余列为特征。

All datasets include a time index column `t` and a label column `y`; remaining columns are features.

```
t, feature_1, feature_2, ..., feature_n, y
0, ...
1, ...
```

- `t`: 时间索引，从 0 开始连续递增 / Time index, starting from 0
- `y`: 标签，二分类为 {0, 1}，Covertype 为 {1–7} / Label; binary {0,1} or multiclass {1–7} for Covertype
- 特征列：各数据集具体说明见下表 / Feature columns: see details below

---

## 参考文献 / References

- Harries, M. (1999). *Splice-2 comparative evaluation: Electricity pricing.* UNSW Technical Report.
- Blackard, J.A. & Dean, D.J. (1999). *Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types.* Computers and Electronics in Agriculture, 24(3), 131–151.
- Bock, R.K. et al. (2004). *Methods for multidimensional event classification: A case study using images from a Cherenkov gamma-ray telescope.* Nuclear Instruments and Methods, 516(2–3), 511–528.
- Bifet, A. et al. (2010). *MOA: Massive online analysis.* JMLR, 11, 1601–1604.
- Gama, J. et al. (2004). *Learning with drift detection.* SBIA 2004, LNAI 3171, 286–295.
