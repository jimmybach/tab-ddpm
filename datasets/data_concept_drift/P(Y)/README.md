# P(Y) 漂移数据集汇总

## 漂移类型定义

**Prior Probability Shift（先验概率漂移）**指数据流中只有类别先验 P(Y) 发生变化，而特征分布 P(X) 和条件概率 P(Y|X) 保持不变：

| 分量 | 变化情况 |
|------|---------|
| P(X) | **不变** — 各类别样本的特征分布全程一致 |
| P(Y\|X) | **不变** — 同一特征组合对应的标签逻辑不改变 |
| P(Y) | **变化** — 各类别在流中出现的比例随时间漂移 |

---

## 数据集总览

| # | 数据集 | 来源类型 | 漂移模式 | 漂移幅度 | 特征类型 | 总实例数 | 窗口大小 |
|---|--------|---------|---------|---------|---------|---------|---------|
| 1 | credit | 真实 | Abrupt | 0.17% → 4.75%（×28） | 全数值（PCA 匿名） | 4,000 | 100 |
| 2 | bank_marketing | 真实 | Abrupt | 10% → 40%（×4） | 混合（数值+类别） | 4,000 | 100 |
| 3 | adult | 真实 | Abrupt + Gradual | 10% → 70% → 30% | 混合（数值+类别） | 4,000 | 100 |
| 4 | agrawal | 合成 | Abrupt | 20% → 80%（×4） | 混合（有语义） | 4,000 | 100 |
| 5 | sea | 合成 | Gradual (sigmoid) | 20% → 80% | 全数值 | 4,000 | 100 |

---

## 数据集详情

### 1. credit（真实）

- **原始来源**：OpenML `creditcard` version 1，信用卡欺诈检测
- **正例定义**：欺诈交易（Class=1）
- **漂移设计**：t=2,000 处单次突变，欺诈率从 0.17% 跳升至 4.75%
- **特征**：V1–V28（PCA 匿名，不可还原）+ Amount，共 29 列
- **类别不平衡**：极度不平衡（原始欺诈率仅 0.17%）
- **特殊说明**：欺诈样本总量有限，pre/post 完全不重叠使用
- **子目录**：[credit/](credit/)

---

### 2. bank_marketing（真实）

- **原始来源**：OpenML `data_id=1461`，葡萄牙银行电话营销活动
- **正例定义**：客户订购定期存款（y=yes）
- **漂移设计**：t=2,000 处单次突变，订购率从 10% 跳升至 40%
- **特征**：age、job、marital、education、default、balance、housing、loan、contact、day、month、campaign、pdays、previous、poutcome，共 15 列
- **特殊说明**：`duration`（通话时长）已删除——该字段仅通话结束后可知，保留会造成数据泄漏，且掩盖 LLM 对深层特征的学习
- **子目录**：[bank_marketing/](bank_marketing/)

---

### 3. adult（真实）

- **原始来源**：UCI Adult (Census Income)，1994 年美国人口普查
- **正例定义**：年收入超过 50K 美元（income=>50K）
- **漂移设计**：双段混合漂移
  - t=2,000（窗口 21）：abrupt 突变，10% → 70%
  - t=2,500（窗口 26）：gradual 线性下降，70% → 30%（每窗口 −2.67%）
- **特征**：age、workclass、fnlwgt、education、education-num、marital-status、occupation、relationship、race、sex、capital-gain、capital-loss、hours-per-week、native-country，共 14 列
- **子目录**：[adult/](adult/)

---

### 4. agrawal（合成）

- **生成器**：`river.datasets.synth.Agrawal`，classification_function=0 全程固定
- **正例定义**：满足 fn=0 贷款审批条件
- **漂移设计**：t=2,000 处单次突变，正例率从 20% 跳升至 80%
- **特征**：salary、commission、age、elevel、car、zipcode、hvalue、hyears、loan，共 9 列
- **特殊说明**：P(Y|X) 由生成器严格保证不变，ground truth 绝对干净；特征具有真实语义，LLM 可利用领域知识辅助合成
- **子目录**：[agrawal/](agrawal/)

---

### 5. sea（合成）

- **生成器**：`river.datasets.synth.SEA`，variant=0 全程固定
- **正例定义**：x1 + x2 ≤ 8
- **漂移设计**：sigmoid 平滑过渡，正例率从 20% 渐变至 80%

$$\text{rate}(t) = 0.20 + 0.60 \times \sigma\!\left(\frac{t - 2{,}000}{300}\right)$$

- **有效过渡区**：t ≈ 800–3,200（约 24 个窗口）
- **特征**：x1、x2、x3，共 3 列（x3 为噪声特征）
- **特殊说明**：五个数据集中唯一的纯 gradual 漂移，用于评估 LLM 对连续变化先验的建模能力
- **子目录**：[sea/](sea/)

---

## 实验设计说明

### 为何使用 Prior Probability Shift 作为评估维度

在 LLM 合成表格数据的评估中，P(Y) 漂移是最直接可量化的维度：

- P(X) 漂移需要比较高维特征分布，评估复杂
- P(Y|X) 漂移需要训练分类器才能检测
- **P(Y) 漂移只需统计正例比例**，评估简单且客观

通过对比合成数据与原始数据的滚动正例比例曲线，可以直观判断 LLM 是否捕捉到了先验分布的变化。

### 与 data_cd 版本的区别

本目录（`data_concept_drift/P(Y)/`）是 `data_cd/P(Y)/` 的缩小版：

| 参数 | data_cd | data_concept_drift |
|------|---------|-------------------|
| 总实例数 | 20,000 | **4,000** |
| 窗口大小 | 500 | **100** |
| 窗口数量 | 40 | 40 |
| 漂移边界 | t=10,000 | **t=2,000** |
| sigmoid 陡度 | 1,500 | **300** |

实例数缩减至 1/5，保持相同的 40 窗口结构和相同比例的覆盖范围，以降低 LLM 合成实验的时间开销。

### 数据集选择的覆盖逻辑

| 维度 | 覆盖情况 |
|------|---------|
| 漂移模式 | Abrupt（×3）、Abrupt+Gradual（×1）、纯 Gradual（×1） |
| 数据来源 | 真实数据（×3）、合成数据（×2） |
| 特征类型 | 全数值（×2）、混合型（×3） |
| 漂移幅度 | 极端（×28）、强（×4）、中等（sigmoid） |
| 类别不平衡 | 极度（×1）、中度（×4） |

---

## 文件结构

```
P(Y)/
├── README.md                          ← 本文件
├── credit/
│   ├── generate_prior_shift.py
│   ├── credit_prior_shift.csv
│   └── credit_prior_shift_class_proportion.png
├── bank_marketing/
│   ├── generate_bank_prior_shift.py
│   ├── bank_prior_shift.csv
│   └── bank_prior_shift_class_proportion.png
├── adult/
│   ├── generate_adult_prior_shift.py
│   ├── adult_prior_shift.csv
│   └── adult_prior_shift_class_proportion.png
├── agrawal/
│   ├── generate_agrawal_prior_shift.py
│   ├── agrawal_prior_shift.csv
│   └── agrawal_prior_shift_class_proportion.png
└── sea/
    ├── generate_sea_prior_shift.py
    ├── sea_prior_shift.csv
    └── sea_prior_shift_class_proportion.png
```
