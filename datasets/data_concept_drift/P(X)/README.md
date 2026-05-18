# P(X) 漂移数据集汇总

## 漂移类型定义

**Covariate Shift（协变量漂移）**指数据流中只有特征边际分布 P(X) 发生变化，而条件概率 P(Y|X) 和（由此推导的）类别先验 P(Y) 保持不变：

| 分量 | 变化情况 |
|------|---------|
| P(X) | **变化** — 特征的均值、方差或分布形状随时间漂移 |
| P(Y\|X) | **不变** — 同一特征组合对应的标签逻辑全程固定 |
| P(Y) | **稳定** — 由于 P(Y\|X) 不变且类别采样比例受控，先验分布不变 |

### 数学纯净性保证原则

本目录中所有数据集均采用以下方式严格保证 P(Y|X) 不变：

| 数据集 | 保证方式 |
|--------|---------|
| Gas Sensor | 真实物理成因：传感器老化改变观测值，气体本身的化学属性不变 |
| Agrawal | 漂移特征（salary、loan）不进入决策函数，标签由 age 重新计算 |
| Gaussian Mixture | 漂移维度（x2）与决策维度（x1）正交，决策规则 $y=\mathbf{1}[x_1 \geq 0]$ 固定 |

---

## 数据集总览

| # | 数据集 | 来源类型 | 漂移模式 | 漂移特征 | 实例数 | 特征数 | 类别数 |
|---|--------|---------|---------|---------|--------|--------|--------|
| 1 | Gas Sensor | 真实 (OpenML) | Gradual（传感器老化） | V1–V128 全部 | 4,000 | 128 | 6 |
| 2 | Agrawal | 合成 (river) | Abrupt | salary, loan | 4,000 | 9 | 2 |
| 3 | Gaussian Mixture | 合成 (numpy) | Gradual (sigmoid) | x2 | 4,000 | 2 | 2 |

---

## 数据集详情

### 1. Gas Sensor（真实，多分类，Gradual）

- **原始来源**：OpenML `gas-drift`，16 个金属氧化物传感器，36 个月，10 个时间批次
- **漂移机制**：传感器随时间发生氧化和污染，导致响应值持续偏移——这是唯一有**真实物理成因**的 P(X) 漂移数据集
- **标签**：6 种气体（Ethanol、Ethylene、Ammonia、Acetaldehyde、Acetone、Toluene）
- **特殊说明**：从原始数据中均匀采样 4,000 条，保留原始时序结构
- **子目录**：[gas_sensor/](gas_sensor/)

---

### 2. Agrawal（合成，二分类，Abrupt）

- **生成器**：`river.datasets.synth.Agrawal`，`classification_function=0`
- **漂移机制**：在 t=2,000 处对 salary 加 +40,000、loan 加 +100,000
- **P(Y|X) 保证**：fn=0 的决策规则为 $y=1 \iff (\text{age}<40 \lor \text{age} \geq 60)$，仅依赖 age，与 salary/loan 完全无关；标签对所有样本重新计算
- **P(Y) 验证**：pre/post 正例率差值 < 0.02 ✓
- **子目录**：[agrawal/](agrawal/)

---

### 3. Gaussian Mixture（合成，二分类，Gradual）

- **生成方式**：numpy 自定义，2D 高斯混合
- **决策边界**：$x_1 = 0$，全程固定
- **漂移机制**：$x_2$ 均值按 sigmoid 从 0 漂移至 5，平行于决策边界方向

$$\mu_2(t) = 5 \cdot \sigma\!\left(\frac{t - 2{,}000}{300}\right)$$

- **有效过渡区**：$t \approx [800,\, 3{,}200]$（约 24 个窗口）
- **P(Y|X) 保证**：漂移维度 $x_2$ 与决策维度 $x_1$ 正交，P(Y=1) ≈ 0.5 全程（pre≈0.50，post≈0.50）✓
- **评估指标**：以 $x_2$ 的窗口均值轨迹作为主要漂移信号（$x_1$ 无漂移，不参与评估）
- **子目录**：[gaussian_mixture/](gaussian_mixture/)

---

## 与 P(Y) 数据集的对比

| 维度 | P(Y) 数据集 | P(X) 数据集 |
|------|------------|------------|
| 变化的分量 | 类别先验 P(Y) | 特征分布 P(X) |
| 不变的分量 | P(X)、P(Y\|X) | P(Y\|X)、P(Y) |
| 可观测信号 | 滚动类别比例变化 | 特征滚动均值/方差变化 |
| 检测难度 | 较易（直接统计标签） | 较难（需监控特征分布） |
| LLM 合成挑战 | 能否重现标签比例变化 | 能否重现特征分布的时序演变 |

---

## 与 data_cd 版本的区别

本目录（`data_concept_drift/P(X)/`）是 `data_cd/P(X)/` 的缩小版：

| 参数 | data_cd | data_concept_drift |
|------|---------|-------------------|
| 总实例数（合成数据集） | 20,000 | **4,000** |
| 窗口大小 | 500 | **100** |
| 窗口数量 | 40 | 40 |
| 漂移边界（合成） | t=10,000 | **t=2,000** |
| sigmoid 陡度 | 1,500 | **300** |

Gas Sensor 数据集实例数也从 13,910 缩减至 4,000（均匀采样）。

---

## 文件结构

```
P(X)/
├── README.md                                    ← 本文件
├── gas_sensor/
│   ├── generate_gas_sensor.py
│   ├── gas_sensor_covariate_shift.csv
│   └── gas_sensor_feature_drift.png
├── agrawal/
│   ├── generate_agrawal_covariate_shift.py
│   ├── agrawal_covariate_shift.csv
│   └── agrawal_covariate_shift_feature_drift.png
└── gaussian_mixture/
    ├── generate_gaussian_mixture_covariate_shift.py
    ├── gaussian_mixture_covariate_shift.csv
    └── gaussian_mixture_feature_drift.png
```
