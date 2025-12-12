# SATCON系统完整工作流程分析

## 目录
1. [整体架构](#整体架构)
2. [核心流程详解](#核心流程详解)
3. [关键数据流示例](#关键数据流示例)
4. [基线 vs 增强系统对比](#基线-vs-增强系统对比)
5. [总结](#总结)

---

## 整体架构

### SATCONSystem 类结构

```
SATCONSystem 类
    ├── 初始化模块 (__init__)
    │   ├── SatelliteNOMA (卫星NOMA传输)
    │   ├── A2GChannel (空地信道)
    │   ├── S2AChannel (卫星到ABS回传)
    │   ├── ABSPlacement (ABS位置优化)
    │   └── NOMAAllocator (功率分配器)
    │
    ├── 核心计算模块
    │   ├── compute_abs_noma_rates() - ABS NOMA速率
    │   ├── compute_abs_oma_rates() - ABS OMA速率
    │   └── hybrid_decision() - 混合决策规则
    │
    └── 仿真入口
        ├── simulate_single_realization() - 单次仿真
        └── simulate_performance() - Monte Carlo仿真
```

---

## 核心流程详解

### 1. 单次仿真流程 (simulate_single_realization)

这是整个系统的核心，按照7个步骤执行：

```python
def simulate_single_realization(self, snr_db, elevation_deg, seed):
    """
    输入：SNR (dB), 卫星仰角 (度), 随机种子
    输出：总系统速率 (bps), 模式统计
    """
```

#### 步骤1：生成用户分布

```python
# 1. 生成用户分布
dist = UserDistribution(self.config.N, self.config.coverage_radius, seed=seed)
user_positions = dist.generate_uniform_circle()  # 返回 [N, 2] 用户坐标
```

**物理意义**：在半径500m的圆形区域内均匀分布32个用户

---

#### 步骤2：优化ABS位置

```python
# 2. 优化ABS位置
abs_position, _ = self.abs_placement.optimize_position_complete(
    user_positions, self.a2g_channel
)
# 返回 [x, y, h] - ABS的3D坐标
```

**使用的方法**：k-means聚类 + 高度优化（几何中心）

**关键区别**：
- **基线方法（src/satcon_system.py）**：`optimize_position_complete()` → k-means
- **增强方法（src_enhanced/）**：`GradientPositionOptimizer.optimize()` → L-BFGS-B梯度优化

---

#### 步骤3：计算卫星NOMA速率

```python
# 3. 卫星NOMA速率
snr_linear = 10 ** (snr_db / 10)  # dB转线性
sat_channel_gains = self.sat_noma.compute_channel_gains_with_pathloss(elevation_deg)
sat_rates, _ = self.sat_noma.compute_achievable_rates(sat_channel_gains, snr_linear)
```

**计算链条**：
1. 计算卫星信道增益（Loo衰落模型 + 自由空间路径损耗）
2. 用户配对（弱-强配对）
3. NOMA功率分配（使用 `compute_power_factors`）
4. 计算每个用户的可达速率（Shannon公式）

**输出**：`sat_rates` [N] - 每个用户通过卫星直传的速率

---

#### 步骤4：计算A2G信道增益

```python
# 4. A2G信道增益
h_abs = abs_position[2]  # ABS高度
distances_2d = dist.compute_distances_from_point(user_positions, abs_position)

fading_a2g = self.a2g_channel.generate_fading(self.config.N, seed=seed+1)
channel_gains_a2g = np.array([
    self.a2g_channel.compute_channel_gain(
        h_abs, r, fading,
        self.config.Gd_t_dB, self.config.Gd_r_dB, self.Nd
    )
    for r, fading in zip(distances_2d, fading_a2g)
])
```

**物理模型**：
- **路径损耗**：基于高度和距离的A2G路径损耗模型
- **小尺度衰落**：Rayleigh衰落
- **LOS概率**：取决于高度和仰角

**输出**：`channel_gains_a2g` [N] - 归一化信道增益 (G/(L×Nd))

---

#### 步骤5：计算ABS NOMA速率

```python
# 5. ABS NOMA速率
abs_noma_rates, pairs = self.compute_abs_noma_rates(
    user_positions, abs_position, channel_gains_a2g, self.config.Pd
)
```

**详细实现**：

```python
def compute_abs_noma_rates(self, ..., channel_gains_a2g, total_power):
    K = len(channel_gains_a2g) // 2  # 配对数 = 16
    bandwidth_per_pair = self.Bd / K  # 每对分配 1.2MHz / 16 = 75kHz

    # 1. 用户配对（基于A2G增益，独立于卫星配对）
    pairs, paired_gains = self.allocator.optimal_user_pairing(channel_gains_a2g)

    rates_noma = np.zeros(len(channel_gains_a2g))

    for k in range(K):
        weak_idx, strong_idx = pairs[k]
        gamma_weak, gamma_strong = paired_gains[k]

        # 2. NOMA功率分配（确保弱用户能解码）
        beta_strong, beta_weak = self.allocator.compute_power_factors(
            gamma_strong, gamma_weak, total_power
        )

        # 3. 速率计算
        # 强用户速率（直接检测自己的信号）
        rate_strong = bandwidth_per_pair * log2(
            1 + beta_strong * P * gamma_strong
        )

        # 弱用户速率（SIC后，干扰来自强用户残留信号）
        rate_weak = bandwidth_per_pair * log2(
            1 + (beta_weak * P * gamma_weak) / (beta_strong * P * gamma_weak + 1)
        )

        rates_noma[weak_idx] = rate_weak
        rates_noma[strong_idx] = rate_strong

    return rates_noma, pairs
```

**关键点**：
- **配对策略**：弱-强配对（最大化和速率）
- **功率分配**：更多功率给弱用户
- **SIC（连续干扰消除）**：强用户先解码弱用户信号并消除，弱用户受强用户干扰

---

#### 步骤6：计算ABS OMA速率

```python
abs_oma_rates = self.compute_abs_oma_rates(channel_gains_a2g, self.config.Pd)
```

```python
def compute_abs_oma_rates(self, channel_gains_a2g, total_power):
    K = len(channel_gains_a2g) // 2
    bandwidth_per_user = self.Bd / K  # 每用户独占带宽

    # OMA：全功率 + 独占带宽
    rates_oma = bandwidth_per_user * log2(1 + total_power * channel_gains_a2g)

    return rates_oma
```

**OMA vs NOMA 对比**：

| 维度 | OMA | NOMA |
|------|-----|------|
| **带宽分配** | 每用户独占 Bd/K | 每对共享 Bd/K |
| **功率分配** | 全功率给单用户 | 按比例分配 |
| **干扰** | 无干扰 | 弱用户受强用户干扰 |
| **适用场景** | 信道差异小 | 信道差异大 |

---

#### 步骤7：混合决策 (Hybrid Decision)

这是论文的核心创新！

```python
# 7. 混合决策
final_rates, modes = self.hybrid_decision(
    sat_rates, abs_noma_rates, abs_oma_rates, s2a_rates, pairs
)
```

**决策规则（论文Section III.C）**：

```
对于每对 k: (弱用户i, 强用户j)

┌─────────────────────────────────────────────┐
│  R^s_i < R^{dn}_i AND R^s_j < R^{dn}_j ?   │
└──────────────┬──────────────────────────────┘
               │
       YES ────┼──── NO
               │
               ↓
    ┌──────────────────┐
    │  情况1: NOMA      │
    │  i → ABS NOMA    │
    │  j → ABS NOMA    │
    └──────────────────┘

               │ (NO)
               ↓
┌─────────────────────────────────────────┐
│  R^s_i < R^{do}_i AND R^s_j >= R^{dn}_j │
└──────────────┬──────────────────────────┘
               │
       YES ────┼──── NO
               │
               ↓
    ┌──────────────────┐
    │  情况2: OMA Weak  │
    │  i → ABS OMA     │
    │  j → Satellite   │
    └──────────────────┘

               │ (NO)
               ↓
┌─────────────────────────────────────────┐
│  R^s_i >= R^{dn}_i AND R^s_j < R^{do}_j │
└──────────────┬──────────────────────────┘
               │
       YES ────┼──── NO
               │
               ↓
    ┌──────────────────┐
    │ 情况3: OMA Strong│
    │  i → Satellite   │
    │  j → ABS OMA     │
    └──────────────────┘

               │ (NO)
               ↓
    ┌──────────────────┐
    │  情况4: None     │
    │  i → Satellite   │
    │  j → Satellite   │
    └──────────────────┘
```

**决策逻辑的物理意义**：

| 情况 | 条件 | 决策 | 原因 |
|------|------|------|------|
| **1** | 两用户ABS都优于卫星 | 使用ABS NOMA | 频谱效率最高 |
| **2** | 弱用户受益，强用户不受益 | 仅弱用户用ABS OMA | 避免强用户性能下降 |
| **3** | 强用户受益，弱用户不受益 | 仅强用户用ABS OMA | 增强强用户速率 |
| **4** | 两用户都不受益 | 都用卫星直传 | ABS无法提供增益 |

---

### 2. Monte Carlo仿真流程 (simulate_performance)

```python
def simulate_performance(self, snr_db_range, elevation_deg=10,
                        n_realizations=100, verbose=True):
    """
    输入：
        snr_db_range: [0, 5, 10, ..., 30] dB
        n_realizations: 1000次 (论文标准)

    输出：
        mean_sum_rates: 平均总速率 [n_snr]
        mean_se: 平均频谱效率 [n_snr]
        std_sum_rates: 标准差 [n_snr]
        mode_statistics: 模式统计
    """
```

**执行流程**：

```
对于每个SNR点 (例如 10dB):
    对于每次实现 r = 1 到 1000:
        seed = random_seed + i * n_realizations + r
        ├── 生成新的用户分布
        ├── 优化ABS位置
        ├── 计算卫星速率
        ├── 计算ABS速率
        ├── 混合决策
        └── 记录结果 sum_rates_all[i, r]

    计算平均值 mean_sum_rates[i] = mean(sum_rates_all[i, :])
    计算标准差 std_sum_rates[i] = std(sum_rates_all[i, :])
    计算频谱效率 mean_se[i] = mean_sum_rates[i] / Bd

返回统计结果（绘制曲线）
```

**随机种子策略**：

```python
seed = self.config.random_seed + i * n_realizations + r
```

- `i`：SNR点索引（0, 1, 2, ...）
- `r`：实现次数（0, 1, ..., 999）
- 确保每个（SNR, 实现）组合使用**唯一且可重复**的随机种子

---

## 关键数据流示例

### 假设场景：
- **用户数**：N = 32
- **配对数**：K = 16
- **SNR**：20 dB
- **ABS带宽**：Bd = 1.2 MHz
- **ABS功率**：Pd = 10 W

### 数据流：

```python
# 步骤1：用户分布
user_positions = [
    [x1, y1],  # 用户1位置
    [x2, y2],  # 用户2位置
    ...
    [x32, y32]  # 用户32位置
]  # shape: [32, 2]

# 步骤2：ABS位置
abs_position = [x_abs, y_abs, h_abs]  # 例如: [50, -100, 300m]

# 步骤3：卫星速率
sat_gains = [γ_1^s, γ_2^s, ..., γ_32^s]  # 卫星信道增益
sat_rates = [R_1^s, R_2^s, ..., R_32^s]  # 卫星速率 (bps)
# 例如: [2.5e6, 3.1e6, ..., 2.8e6] bps

# 步骤4：A2G信道增益
distances_2d = [d1, d2, ..., d32]  # 用户到ABS的水平距离
a2g_gains = [γ_1^a, γ_2^a, ..., γ_32^a]  # A2G信道增益

# 步骤5：ABS配对（独立于卫星配对！）
abs_pairs = [
    (2, 15),   # 对1: 用户2(弱), 用户15(强)
    (7, 23),   # 对2: 用户7(弱), 用户23(强)
    ...
]  # shape: [16, 2]

# 步骤6：ABS NOMA速率
abs_noma_rates = [
    0,          # 用户1 (未配对或其他对)
    3.2e6,      # 用户2 (弱用户, 对1)
    ...
    4.5e6,      # 用户15 (强用户, 对1)
    ...
]  # shape: [32]

# 步骤7：ABS OMA速率
abs_oma_rates = [
    2.8e6,      # 用户1 (独占模式)
    3.8e6,      # 用户2
    ...
]  # shape: [32]

# 步骤8：混合决策（对于对1: 用户2和用户15）
# 对1决策：
if sat_rates[2] < abs_noma_rates[2] and sat_rates[15] < abs_noma_rates[15]:
    # 2.5e6 < 3.2e6 AND 3.1e6 < 4.5e6 → TRUE
    mode = 'noma'
    final_rates[2] = 3.2e6
    final_rates[15] = 4.5e6

# 总速率
sum_rate = sum(final_rates) = 95.2e6 bps  # 95.2 Mbps
spectral_efficiency = sum_rate / Bd = 95.2e6 / 1.2e6 = 79.3 bits/s/Hz
```

---

## 基线 vs 增强系统对比

| 模块 | **基线系统 (satcon_system.py)** | **增强系统 (src_enhanced/)** |
|------|----------------------------------|------------------------------|
| **ABS位置** | k-means聚类 (几何中心) | 梯度优化 (性能驱动) |
| **用户配对** | 独立配对 (卫星和ABS各自配对) | 联合配对 (考虑协同) |
| **混合决策** | 贪婪规则 (每对局部最优) | 整数规划 (全局最优) |
| **优化框架** | 顺序执行 | 交替迭代 |
| **性能增益** | 基线 | +3-4% |

### 详细对比

#### 1. ABS位置优化

**基线方法（k-means）**：
```python
# 最小化几何距离
min sum ||u_i - p||^2
```

**增强方法（梯度优化）**：
```python
# 最大化系统速率
max sum R_i(p, h)
```

**关键区别**：
- k-means：纯几何优化，不考虑信道质量
- 梯度优化：直接优化系统性能，考虑路径损耗、LOS概率等物理特性

---

#### 2. 用户配对优化

**基线方法（独立配对）**：
```python
sat_pairs = optimal_pairing(sat_gains)   # 独立优化
abs_pairs = optimal_pairing(a2g_gains)   # 独立优化
```

**增强方法（联合配对）**：
```python
# 联合优化考虑协同效应
sat_pairs, abs_pairs = joint_optimize(sat_gains, a2g_gains)

# 使用局部搜索：
for iteration in range(max_iterations):
    # 交换卫星配对
    try_swap_sat_pairs()
    # 交换ABS配对
    try_swap_abs_pairs()
    # 计算联合收益
    joint_benefit = compute_joint_benefit(sat_pairs, abs_pairs)
```

**关键区别**：
- 独立配对：无法利用卫星和ABS之间的协同效应
- 联合配对：通过局部搜索找到更好的配对组合

---

#### 3. 混合决策优化

**基线方法（贪婪决策）**：
```python
# 每对独立选择最佳模式
for k in range(K):
    options = [sat_rate, noma_rate, oma_weak_rate, oma_strong_rate]
    mode[k] = argmax(options)  # 局部最优
```

**增强方法（整数规划）**：
```python
# 全局优化
# 决策变量：x[k], y_weak[k], y_strong[k] ∈ {0,1}
# 目标函数：
max sum(x[k]*R_noma[k] + y_weak[k]*R_oma_weak[k] +
        y_strong[k]*R_oma_strong[k] + (1-x[k]-y_weak[k]-y_strong[k])*R_sat[k])
# 约束：
x[k] + y_weak[k] + y_strong[k] <= 1  # 互斥性
```

**关键区别**：
- 贪婪决策：每对独立决策，可能错失全局最优
- 整数规划：联合优化所有对的决策，达到全局最优

---

## 总结

### `satcon_system.py` 实现了论文的**基线方法**：

1. **流程清晰**：7步完整仿真流程
2. **模块化设计**：每个功能独立模块
3. **符合论文**：严格遵循Section III的算法
4. **可扩展性**：为增强优化提供基础

### 核心创新点（相比纯卫星NOMA）：
- 引入ABS作为中继
- 混合NOMA/OMA传输
- 基于规则的智能决策

### 改进空间（src_enhanced实现）：

| 改进点 | 方法 | 预期增益 |
|--------|------|----------|
| **位置优化** | 从几何到性能驱动 | ~1-2% |
| **配对优化** | 从独立到联合 | ~0.5-1% |
| **决策优化** | 从贪婪到全局最优 | ~0.5-1% |
| **交替优化** | 迭代优化框架 | ~0.5% |
| **总计** | 三个模块协同 | **~3-4%** |

### 关键技术指标：

- **用户数**：N = 32
- **配对数**：K = 16
- **ABS带宽**：Bd = 1.2 MHz（主实验）
- **ABS功率**：Pd = 10 W
- **卫星仰角**：10° (主实验)
- **SNR范围**：0-30 dB
- **Monte Carlo次数**：1000次（论文标准）

### 仿真时间估算：

- **单次仿真**：~0.01秒
- **单个SNR点（1000次）**：~10秒
- **完整曲线（7个SNR点）**：~70秒
- **带进度条和实时反馈**

---

## 附录：关键公式

### 1. Shannon容量公式
```
R = B * log2(1 + SNR)
```

### 2. NOMA速率公式

**强用户**：
```
R_strong = B * log2(1 + β_strong * P * γ_strong)
```

**弱用户**：
```
R_weak = B * log2(1 + (β_weak * P * γ_weak) / (β_strong * P * γ_weak + 1))
```

### 3. OMA速率公式
```
R_oma = B * log2(1 + P * γ)
```

### 4. 功率分配因子

根据NOMA SIC可解码条件：
```
β_weak + β_strong = 1
β_weak > β_strong
```

### 5. 路径损耗模型

**自由空间路径损耗**：
```
L_fs = (4πd/λ)^2
```

**A2G路径损耗**：
```
L_a2g = L_0 + 10 * α * log10(d) + X_σ
```

其中：
- `L_0`：参考距离处的路径损耗
- `α`：路径损耗指数
- `X_σ`：阴影衰落

---

**文档版本**：v1.0
**创建日期**：2025-12-12
**作者**：SATCON Enhancement Project
**相关代码**：`src/satcon_system.py`, `src_enhanced/joint_satcon_system.py`
