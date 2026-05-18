# P(Y|X) 真实概念漂移数据集汇总

## 漂移类型定义

**真实概念漂移（Real Concept Drift）**指在给定特征 X 的条件下，标签的条件分布 P(Y|X) 随时间发生变化，而 P(X) 不一定保持稳定。与协变量漂移不同，同一输入向量 **x** 在漂移前后系统性地对应不同标签。

形式化定义：存在 t₁ < t₂，使得：

```
P_{t1}(Y | X = x) ≠ P_{t2}(Y | X = x)  对 x 的非零测度子集成立
```

本集合对 P(X) 或 P(Y) 不施加额外约束。合成生成器通过设计保证 P(X) 稳定，提供干净的隔离；真实数据集的 P(X) 可能与 P(Y|X) 协同漂移，反映真实的生态或市场复杂性。

---

## 数据集总览

| # | 数据集 | 来源 | 实例数 | 特征数 | 类别数 | P(X) 稳定 | 漂移模式 | 漂移机制 |
|---|---|---|---|---|---|---|---|---|
| 1 | STAGGER | River 合成 | 4,000 | 3 类别型 | 2 | ✓ | Abrupt ×2 | fn0→fn1 于 t=1,333；fn1→fn2 于 t=2,666 |
| 2 | Hyperplane | River 合成 | 4,000 | 10 连续型 | 2 | ✓ | Incremental | 4/10 权重以 Δ=0.001/步持续旋转 |
| 3 | Mixed | River 合成 | 4,000 | 4（连续+布尔） | 2 | ✓ | Abrupt ×1 | AND→OR 规则于 t=2,000 |
| 4 | SINE | River 合成 | 4,000 | 2 连续型 | 2 | ✓ | Abrupt ×1 | 正弦曲线极性翻转于 t=2,000 |
| 5 | Electricity | OpenML 151 | 4,000 | 8 连续型 | 2 | ✗ | Gradual | NSW 市场动态 1996–1998 年演化 |
| 6 | Covertype | OpenML 1596 | 4,000 | 54 混合型 | 7 | ✗ | Incremental | 地形→植被映射随海拔持续偏移 |
| 7 | MAGIC Telescope | OpenML 1120 | 4,000 | 10 连续型 | 2 | ✗ | Incremental | 伽马/强子判别边界随入射角 fAlpha 偏移 |

---

## 数据集详情

### 1. STAGGER

**子目录：** `stagger/`

三个 Boolean 标签函数在两个时间点突变切换，特征为 {size, color, shape} 的类别型数据：

- **fn0**（t = 0–1,332）：y=1 当且仅当 size=small 且 color=red
- **fn1**（t = 1,333–2,665）：y=1 当且仅当 color=green 或 shape=circular
- **fn2**（t = 2,666–3,999）：y=1 当且仅当 size=medium 或 size=large

P(X) 完全静止（均匀分布于各类别域）。由于三个规则覆盖 27 格特征空间中不同比例的区域（约 11%、56%、67%），P(Y) 在切换点发生大幅跳变。

**漂移模式：** Abrupt | **P(X) 稳定：** 是

---

### 2. Hyperplane

**子目录：** `hyperplane/`

10 维连续特征空间中的旋转超平面。标签由 **w**·**x** 的符号决定，其中 10 个权重中有 4 个以每步 0.001 的速率持续旋转。边界从不跳变，在整个 4,000 步流中缓慢持续漂移。

这是专门用于评估能够追踪渐变持续 P(Y|X) 变化（而不依赖 P(X) 信号）的算法的经典基准。

**漂移模式：** Incremental | **P(X) 稳定：** 是

---

### 3. Mixed

**子目录：** `mixed/`

异构特征集（2 个连续型 + 2 个布尔型）结合正弦曲线边界。标签规则在 t=2,000 处从 AND 逻辑（严格，正例率约 13%）切换为 OR 逻辑（宽松，正例率约 67%）。正弦边界曲线本身不移动，仅布尔组合规则发生改变。

**漂移模式：** Abrupt | **P(X) 稳定：** 是

---

### 4. SINE

**子目录：** `sine/`

二维连续特征 x1, x2 ∈ [0, 1]。决策边界为正弦曲线 x2=sin(x1)。t=2,000 处正负区域互换：

- **fn0：** y=1 当且仅当 x2 ≥ sin(x1)（曲线上方）
- **fn1：** y=1 当且仅当 x2 < sin(x1)（曲线下方）

边界几何形状相同；仅极性翻转。由于边界近似对称，P(Y) 在整个流中约保持 50%，P(X) 完全稳定——P(Y|X) 的完全翻转不会在 P(X) 或 P(Y) 中产生任何信号，对依赖分布变化的检测方法构成极大挑战。

**漂移模式：** Abrupt | **P(X) 稳定：** 是

---

### 5. Electricity

**子目录：** `electricity/`

澳大利亚 NSW 电力市场数据，1996 年 5 月至 1998 年 12 月。任务：预测当前价格相对 24 小时移动平均线是 UP 还是 DOWN。从原始数据中取前 4,000 条按时间顺序排列的记录。

这是概念漂移领域最广泛引用的真实世界基准数据集。随着电力市场成熟、参与者策略演变和季节性规律偏移，供需特征与价格方向之间的映射关系渐变。P(X) 与 P(Y|X) 均会漂移，反映真实市场复杂性。

**漂移模式：** Gradual | **P(X) 稳定：** 否（真实世界协同漂移）

---

### 6. Covertype

**子目录：** `covertype/`

科罗拉多州罗斯福国家森林 30×30 米地块的 54 个地图特征，用于预测森林覆盖类型。从 581,012 条记录中跨 40 个海拔带均匀采样 4,000 条（每带 100 条，按海拔排序）。

随着海拔从约 1,859m 上升至 3,858m，主导覆盖类型依次过渡：黄松 → 扭叶松 → 云杉/冷杉 → 矮曲林。相同地形特征（坡度、日照、土壤类型）在不同海拔对应不同植被，沿生态梯度产生连续的增量式 P(Y|X) 漂移。

**漂移模式：** Incremental | **P(X) 稳定：** 否（海拔驱动协同漂移）

---

### 7. MAGIC Telescope

**子目录：** `magic_telescope/`

来自 MAGIC 实验的伽马射线望远镜数据。任务：根据 10 个图像形状参数区分伽马射线事件（信号）和强子簇射事件（背景）。从全部 19,020 条记录中按 `fAlpha`（入射角 0°–90°）排序后均匀采样 4,000 条。

随着观测角度增大，切伦科夫簇射图像的投影几何关系改变，特征空间中最优判别边界随之偏移，产生由物理成像机制驱动的增量式 P(Y|X) 漂移。

**漂移模式：** Incremental | **P(X) 稳定：** 否（fAlpha 驱动协同漂移）

---

## 漂移模式覆盖情况

| 漂移模式 | 数据集 |
|---|---|
| Abrupt（单次突变） | SINE、Mixed |
| Abrupt（多次突变） | STAGGER |
| Incremental（连续渐变） | Hyperplane、Covertype、MAGIC Telescope |
| Gradual（真实世界） | Electricity |

---

## P(X) 稳定性汇总

| P(X) 状态 | 数据集 |
|---|---|
| 稳定（由构造保证） | STAGGER、Hyperplane、Mixed、SINE |
| 不稳定（真实世界协同漂移） | Electricity、Covertype、MAGIC Telescope |

合成生成器提供纯粹的 P(Y|X) 漂移，P(X) 受控。真实数据集则提供特征分布与标签规则协同演化的生态或经济现实场景。

---

## 与 data_cd 版本的区别

本目录（`data_concept_drift/P(YX)/`）是 `data_cd/P(YX)/` 的缩小版：

| 参数 | data_cd | data_concept_drift |
|------|---------|-------------------|
| 总实例数 | 20,000 | **4,000** |
| 窗口大小 | 500 | **100** |
| 窗口数量 | 40 | 40 |
| 漂移边界（合成） | t=6,667 / t=10,000 / t=13,333 | **t=1,333 / t=2,000 / t=2,666** |

实例数缩减至 1/5，漂移边界按比例缩放，保持相同的相对漂移位置。

---

## 文件结构

```
P(YX)/
├── README.md                       ← 本文件
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
