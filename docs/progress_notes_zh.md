# 论文项目进度笔记

> 最后更新：2026-03-16

---

## 一、项目目标

用 Conv1D VAE 学习触觉振动信号（haptic vibrotactile signals）的潜在空间，
再通过 PCA 发现 8 个可解释的**复合控制维度**（PC1–PC8），
最终实现 **"给 8 个旋钮值 → 生成对应触觉信号"** 的可控生成系统。

---

## 二、已完成的里程碑

### 1. 代码仓库搭建（重构）

把原来分散的 6 个 Jupyter notebook（Conv1D_AE, Conv1d_VAE, min_AE 等）
重构成了干净的模块化项目结构：

```
src/
  data/          数据集加载 + 预处理
  models/        ConvVAE, ConvAE, 共用组件(common.py)
  training/      训练器, 损失函数, KL调度
  eval/          评估, 27种信号度量, 验证框架, 可视化
  pipelines/     PCA控制, 潜在提取, 控制规范
  utils/         种子, 配置加载

scripts/
  train.py             训练入口
  extract_and_pca.py   提取潜在向量 + PCA
  build_controls.py    构建控制规范 + sweep gallery
  validate_extended.py 完整验证（27指标, 双参考, 跨种子对齐）
  cross_seed_stability.py  跨种子稳定性分析
  eval.py              独立评估入口

configs/
  vae_balanced.yaml       主配置（latent_dim=24, beta_max=0.003）
  vae_balanced_s123.yaml  种子123变体
  vae_balanced_s456.yaml  种子456变体
  ae_matched.yaml         匹配架构的AE基线（无KL，重建上限）
  vae_default.yaml        原始配置（latent_dim=64）
  vae_compact.yaml        紧凑配置（latent_dim=16）
```

Colab 笔记本 (`colab/train_colab.ipynb`) 提供端到端流程，
共 22 个 cell，顺序编号 1–21。

### 2. VAE 模型迭代

| 阶段 | 配置 | latent_dim | 数据 | 问题/结果 |
|------|------|-----------|------|-----------|
| 初始 | vae_default | 64 | 585 (Initial) | 重建不错，但PCA 8个PC只覆盖36.8%方差 |
| 压缩 | vae_compact | 16 | 585 (Initial) | PCA覆盖68%，但重建质量大幅下降 |
| 平衡v1 | vae_balanced | 24 | 585 (Initial) | PCA覆盖58.9%，重建STD ratio ~70% |
| 混合（失败） | vae_balanced | 24 | 1378 (Initial+HapticGen) | PCA降至48.8%，PC语义重叠严重 |
| **平衡v2** | **vae_balanced** | **24** | **793 (HapticGen)** | **PCA 56.9%，重建81.3%，可控性代价仅3%** |

> **教训**：混合两个不同生成模型的数据会破坏 PCA 结构。单一模型来源 + 足够样本量 = 最佳结果。

**当前使用的模型：`vae_balanced`（v2，HapticGen 数据）**
- 训练数据：793 个 HapticGen（微调模型）用户点赞样本
- latent_dim = 24
- channels = [32, 64, 128, 128]
- beta_max = 0.003（KL正则化强度）
- 损失 = MSE + L1(×0.2) + 多尺度频谱(×0.15) + 振幅(×0.5) + KL(free-bits)

### 3. PCA 控制维度发现

从 793 个触觉信号中提取 24 维潜在向量 → StandardScaler + PCA → 8 个复合控制维度

**PCA结果（vae_balanced v2, seed=42, HapticGen 数据）：**

| PC | 方差% | 累计% | 复合标签 | 通俗解读 |
|----|-------|-------|----------|---------|
| PC1 | 20.2% | 20.2% | **Energy–spectral warmth composite** | 能量 + 低频主导 |
| PC2 | 7.7% | 27.9% | **Temporal irregularity & onset softness** | 节奏不规则 + 起攻变柔 |
| PC3 | 5.5% | 33.4% | **Modulation rate–spectral balance** | 调制频率 + 频谱明暗 |
| PC4 | 5.1% | 38.5% | **Onset-weighted decay** | 能量集中在开头 |
| PC5 | 4.8% | 43.3% | **Continuity–rhythm composite** | 连续但不规则 |
| PC6 | 4.7% | 48.0% | **Soft attack with rhythmic regularity** | 起攻柔 + 节奏规律 |
| PC7 | 4.6% | 52.6% | **Sustained energy & late emphasis** | 能量持续/后段加强 |
| PC8 | 4.3% | 56.9% | **Soft attack with temporal jitter** | 起攻柔 + 节奏抖动 |

> **注意**：PC 轴是复合轴（composite axes），不应简化为单一指标标签。
> 看似重叠的 PC4/PC7（都含 decay slope）和 PC6/PC8（都含 attack time）
> 在第二、第三指标上方向相反，实际描述不同的复合感知维度。

### 4. 控制系统接口

已实现稳定的解码接口：
```
controls = [c1, c2, ..., c8]  （默认值0，范围 P5–P95）
decode(controls) → haptic signal (4000 samples @ 8000Hz)
```

每个控制维度有明确的范围（从训练数据的第5到第95百分位）。
产出了 `controls_spec.json`（机器可读）和 `controls_table.md`（论文表格）。

### 5. 统计验证

#### 5.1 信号度量体系（27个指标）

| 类别 | 指标数 | 示例 |
|------|--------|------|
| 强度 | 3 | RMS能量, 峰值振幅, 峰均比 |
| 频谱 | 8 | 频谱质心, 滚降频率, 斜率, 平坦度, 高频比, 3个频段能量 |
| 包络/时域 | 7 | 衰减斜率, 攻击时间, 瞬态能量比, 有效持续时间, 包络面积/熵 |
| 节奏 | 4 | 起始密度, IOI熵, 起始间隔变异系数, 调制频谱峰 |
| 连续性 | 2 | 过零率, 间隙比 |
| 纹理 | 2 | 短时方差, AM调制指数 |

#### 5.2 单调性验证（Spearman ρ）

对每个 PC 做 21 步 sweep（P5 到 P95），计算每个 PC × 每个指标 的 Spearman 秩相关：

- **PC1**：AM调制指数 ρ=−0.85, 衰减斜率 ρ=+0.78, RMS ρ=+0.74
- **PC2**：频谱质心 ρ=+0.97, 短时方差 ρ=−0.97, AM调制 ρ=−0.91

产出热力图 `monotonicity_extended_heatmap.png`

#### 5.3 正交性（选择性分析）

每个 PC 的"目标指标"相关性 vs "非目标指标"相关性 → 选择性比值
- 平均选择性 3.1×（>2× 为良好正交性）

#### 5.4 效应量（主要 vs 次要）

PC1–PC4 在关键指标上的效应量显著大于 PC5–PC8 → 验证了"主要/次要"划分

#### 5.5 跨种子稳定性

用 3 个不同随机种子（42, 123, 456）训练 vae_balanced：
- 方差解释比稳定：PC1 = 25.0% ± 1.6%
- PC1/PC2 的指标方向一致性 87%
- 总方差覆盖 60.6% ± 4.3%

#### 5.6 双参考 sweep

除了从 PCA 原点（全0）出发做 sweep，还从数据集均值出发做 sweep，
比较两种参考的指标方向一致性 → 确认控制效果不依赖于参考点选择。

#### 5.7 PCA 轴对齐（余弦相似度）

跨种子比较 PCA 主成分方向的余弦相似度，评估控制轴的稳定性。
产出 `pca_axis_alignment.json` 和 `pca_axis_alignment.png`。

### 6. VAE vs AE 重建质量对比

训练了一个与 VAE 完全相同架构的确定性 AE（`ae_matched.yaml`，无 KL 损失）。

**最新结果（HapticGen 793 样本）：**

| 模型 | 平均 STD ratio | 说明 |
|------|---------------|------|
| VAE (vae_balanced) | **81.3%** | 主模型 |
| AE (ae_matched) | 84.3% | 重建上限（无 KL） |
| **可控性代价** | **3.0%** | VAE 与 AE 的差值 |

> 可控性代价从之前的 12% 降至 3%，说明 HapticGen 数据的潜在空间更容易被结构化。

### 7. 指标分组体系

27 个指标组织为 7 个感知组，每组 1-2 个代表指标（共 12 个）：

| 组 | 标签 | 代表指标 |
|----|------|---------|
| G1 | 强度/能量 | rms_energy, crest_factor |
| G2 | 频谱形状 | spectral_centroid, spectral_flatness |
| G3 | 时域包络 | envelope_decay_slope, envelope_entropy |
| G4 | 起攻/瞬态 | attack_time |
| G5 | 节奏 | onset_density, ioi_entropy |
| G6 | 连续性 | gap_ratio |
| G7 | 纹理/调制 | am_modulation_index, short_term_variance |

### 8. 代码清理

- 删除死代码：`simple_ae.py`, `SyntheticVibrationDataset`, `make_vibration`, 未使用的配置文件
- 合并重复：3个 sweep 函数统一为 `sweep_axis()`，`_group_norm` 提取到 `common.py`
- 删除冗余脚本：`validate_controls.py` 合并进 `validate_extended.py`
- 重写 Colab 笔记本：顺序编号、一致的变量插值语法、清晰的分区结构
- 净减少约 2100 行代码

---

## 三、当前状态

**模型已训练完成（v2, HapticGen 793样本），控制系统已建立并通过统计验证。**

已有产出物：
- `controls_spec.json` — 机器可读的控制规范（含复合标签）
- `controls_table.md` — 论文级别的控制维度表格（复合命名 + 27指标）
- `monotonicity_extended_heatmap.png` — 27指标单调性热力图
- `pca_axis_alignment.json/png` — 跨种子 PCA 轴稳定性
- `metric_binding_extended.json` — 完整的指标绑定和验证证据
- `selectivity_extended.png` — 正交性柱状图
- `pc_sweep_gallery/` — 每个 PC 的波形 + 频谱图 sweep

---

## 四、待做 / 可能的下一步

1. **论文写作**：用已有的验证数据和图表撰写 Method + Results 章节
2. **LLM 接口**：在控制系统上接入 LLM，实现 "文本描述 → 控制参数 → 触觉信号"
3. **用户实验**：设计感知实验验证控制维度是否与人类感知对齐
4. **更多种子/数据集**：如果审稿人质疑泛化性，可增加实验

---

## 五、关键决策记录

| 决策 | 理由 |
|------|------|
| latent_dim=24 | 16太小（重建差），64太大（PCA只覆盖37%） |
| beta_max=0.003 | 太大会后验崩塌，太小潜在空间无结构 |
| 8个PC | PC1-PC6有独立语义，PC7/PC8用复合命名消除表面重叠 |
| P5–P95 范围 | 避免极端值，比固定±2更稳健 |
| 只用 HapticGen 数据 | 混合 Initial+HapticGen 破坏 PCA 结构（48.8%→56.9%） |
| 复合标签命名 | PCA 轴是复合轴，单指标命名是误导 |
| 用 AE 作重建上限 | 相同架构无 KL → 公平比较可控性代价 |
| 27指标→7组12代表 | 减少组间冗余，论文正文用代表指标，附录放完整27指标 |

---

## 六、GitHub 仓库

https://github.com/cindy-77jiayi/thesis_hapticAE

- `main` 分支：代码开发
- `colab-runs` 分支：Colab 带输出的笔记本存档
