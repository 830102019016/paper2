# Monte Carlo仿真与参数配置说明

## 目录
1. [Monte Carlo仿真流程详解](#monte-carlo仿真流程详解)
2. [参数固定与变化策略](#参数固定与变化策略)
3. [实验类型总览](#实验类型总览)
4. [随机种子策略](#随机种子策略)
5. [时间复杂度分析](#时间复杂度分析)

---

## 问题1：Monte Carlo仿真的1000次是什么意思？

### 简短回答

**不是1000个不重复的SNR点，而是对每个SNR点重复1000次独立仿真。**

### 详细解释

#### 实际运行过程

```python
def simulate_performance(self, snr_db_range, elevation_deg=10,
                        n_realizations=1000, verbose=True):
    """
    输入：
        snr_db_range: [0, 1, 2, ..., 30]  # 31个SNR点
        n_realizations: 1000  # 每个SNR点重复1000次

    输出：
        mean_sum_rates: 平均总速率 [31]
        mean_se: 平均频谱效率 [31]
        std_sum_rates: 标准差 [31]
    """

    # 对于每个SNR点
    for i, snr_db in enumerate(snr_db_range):  # 循环31次
        # 对于每次独立实现
        for r in range(n_realizations):  # 循环1000次
            seed = random_seed + i * n_realizations + r  # 唯一随机种子

            # 生成新的随机用户分布
            user_positions = generate_uniform_circle(seed=seed)

            # 计算此次仿真的系统速率
            sum_rate = simulate_single_realization(snr_db, seed)

            # 保存结果
            sum_rates_all[i, r] = sum_rate

        # 对1000次结果取平均和标准差
        mean_sum_rates[i] = mean(sum_rates_all[i, :])  # 平均值
        std_sum_rates[i] = std(sum_rates_all[i, :])    # 标准差
```

---

### 具体示例：SNR = 20dB

```python
# SNR点：20 dB (i=20)
# 配置：N=32用户，Bd=1.2MHz，Pd=1W

# 实现1（seed = 42 + 20*1000 + 0 = 20042）：
#   - 随机生成用户位置1
#   - 随机生成信道衰落1
#   → 系统速率 = 95.2 Mbps

# 实现2（seed = 42 + 20*1000 + 1 = 20043）：
#   - 随机生成用户位置2（完全不同）
#   - 随机生成信道衰落2（完全不同）
#   → 系统速率 = 97.5 Mbps

# ...

# 实现1000（seed = 42 + 20*1000 + 999 = 21041）：
#   - 随机生成用户位置1000
#   - 随机生成信道衰落1000
#   → 系统速率 = 94.8 Mbps

# 最终统计（SNR=20dB这一点）：
mean_rate_20dB = mean([95.2, 97.5, ..., 94.8]) = 96.1 Mbps
std_rate_20dB = std([95.2, 97.5, ..., 94.8]) = 2.3 Mbps
spectral_efficiency_20dB = 96.1 / 1.2 = 80.1 bits/s/Hz
```

---

### 为什么需要1000次Monte Carlo？

#### 仿真目的

**消除随机性的影响，获得统计上可靠的结果。**

#### 每次仿真的随机因素

1. **用户位置随机**
   - 32个用户在500m半径圆内均匀随机分布
   - 每次仿真位置完全不同

2. **信道衰落随机**
   - **卫星链路**：Loo衰落模型（阴影 + 多径）
   - **A2G链路**：Rayleigh衰落（小尺度衰落）
   - 每次仿真衰落系数完全不同

#### 统计意义

```
单次仿真： 速率 = 95.2 Mbps  ← 特定场景的结果
1000次仿真： 平均速率 = 96.1 ± 2.3 Mbps  ← 统计平均结果（可靠）
```

**95%置信区间**：
```
置信区间 = 96.1 ± 1.96 * (2.3 / √1000)
         = 96.1 ± 0.14 Mbps
         = [95.96, 96.24] Mbps
```

---

### 论文标准

| 场景 | Monte Carlo次数 | 用途 |
|------|----------------|------|
| **快速测试** | 10-100次 | 代码调试、快速验证 |
| **论文标准** | **1000次** | IEEE会议/期刊标准 |
| **高精度验证** | 5000-10000次 | 关键结果验证 |

**本项目采用**：1000次（符合IEEE通信领域标准）

---

### 总仿真次数计算

```python
# 主实验配置
SNR_points = [0, 1, 2, 3, ..., 30]  # 31个点
n_realizations = 1000                # 每点1000次

# 总仿真次数
total_simulations = 31 × 1000 = 31,000 次独立仿真
```

---

### 时间估算

```python
# 性能指标（基于实际测试）
单次仿真时间 ≈ 0.01 秒

# 完整主实验时间
总时间 = 31,000 × 0.01 秒
       = 310 秒
       ≈ 5 分钟

# 带进度条显示：
仿真中: 100%|████████| 31/31 [05:10<00:00, 10.0s/it]
        频谱效率: 80.1 bits/s/Hz
```

---

## 问题2：除了SNR，其他参数是固定的吗？

### 简短回答

**在主实验中，用户数、ABS带宽、ABS功率是固定的。但在扩展实验中，这些参数也会变化。**

---

## 参数固定与变化策略

### 主实验配置（Main Experiment）

**目的**：生成论文主曲线（Figure 2），研究SNR对系统性能的影响

| 参数 | 值 | 状态 | 说明 |
|------|-----|------|------|
| **SNR** | [0, 1, 2, ..., 30] dB | ❌ **变化（X轴）** | 主要变量 |
| **用户数** | N = 32 | ✅ 固定 | 论文标准配置 |
| **ABS带宽** | Bd = 1.2 MHz | ✅ 固定 | 中等带宽 |
| **ABS功率** | Pd = 1 W (30 dBm) | ✅ 固定 | 典型UAV功率 |
| **卫星带宽** | Bs = 5 MHz | ✅ 固定 | 卫星频谱资源 |
| **卫星仰角** | 10° | ✅ 固定 | 低仰角场景 |
| **覆盖半径** | 500m | ✅ 固定 | 典型城市场景 |
| **Monte Carlo次数** | 1000 | ✅ 固定 | 统计可靠性 |

**Python代码示例**：

```python
# 主实验（生成Figure 2）
satcon = SATCONSystem(config, abs_bandwidth=1.2e6)

mean_rates, mean_se, std_rates, mode_stats = satcon.simulate_performance(
    snr_db_range=np.arange(0, 31),  # 0-30 dB（变化）
    elevation_deg=10,                # 固定
    n_realizations=1000,             # 固定
    verbose=True
)
```

**输出结果**：

```
X轴: SNR [0, 1, 2, ..., 30] dB
Y轴: 频谱效率 (bits/s/Hz)
曲线:
  - SAT-NOMA (仅卫星)
  - Original-SATCON (基线)
  - SATCON+M1 (位置优化)
  - SATCON+M2 (配对优化)
  - SATCON+M3 (决策优化)
  - Proposed-Full (完整方案)
```

---

## 实验类型总览

### 1. 主实验（Main Experiment）- SNR扫描

```yaml
实验代码: simulations/fig2_complete.py
实验类型: snr_sweep
目的: 生成论文主曲线（Figure 2）

固定参数:
  - 用户数: N = 32
  - ABS带宽: Bd = 1.2 MHz
  - ABS功率: Pd = 1 W
  - 卫星仰角: 10°
  - 覆盖半径: 500m

变化参数:
  - SNR: [0, 1, 2, ..., 30] dB  ← 唯一变量

Monte Carlo: 1000次/SNR点

输出:
  - 文件: results/figures/fig2_complete.png
  - 数据: results/data/fig2_complete.npz
  - X轴: SNR (dB)
  - Y轴: 频谱效率 (bits/s/Hz)
```

---

### 2. 消融实验（Ablation Study）

```yaml
实验代码: experiments/run_comparison.py
实验类型: ablation_study
目的: 验证每个模块的独立贡献

固定参数:
  - 用户数: N = 32
  - ABS带宽: Bd = 1.2 MHz
  - ABS功率: Pd = 1 W
  - 卫星仰角: 10°
  - SNR: 20 dB（典型值）

测试系统（6个）:
  1. SAT-NOMA（无ABS）
  2. Original-SATCON（基线，全部禁用）
  3. SATCON+M1（仅梯度位置优化）
  4. SATCON+M2（仅联合配对优化）
  5. SATCON+M3（仅整数规划决策）
  6. Proposed-Full（全部启用）

Monte Carlo: 500次（消融实验可用较少次数）

输出:
  - 每个模块的性能增益
  - 模块间的协同效应
  - 柱状图对比
```

**预期结果**：

| 系统 | 频谱效率 (bits/s/Hz) | 相对增益 |
|------|---------------------|---------|
| SAT-NOMA | 75.2 | 基线1 |
| Original-SATCON | 78.5 | 基线2 |
| SATCON+M1 | 79.8 | +1.7% |
| SATCON+M2 | 79.2 | +0.9% |
| SATCON+M3 | 79.0 | +0.6% |
| **Proposed-Full** | **81.2** | **+3.4%** |

---

### 3. 带宽影响实验（Bandwidth Sweep）

```yaml
实验代码: experiments/run_comparison.py --bandwidth-sweep
实验类型: bandwidth_sweep
目的: 研究ABS带宽对性能的影响

固定参数:
  - 用户数: N = 32
  - ABS功率: Pd = 1 W
  - 卫星仰角: 10°
  - SNR: 20 dB

变化参数:
  - ABS带宽: [0.4, 1.2, 2.0, 3.0] MHz  ← 4个点

Monte Carlo: 1000次/带宽点

输出:
  - X轴: ABS带宽 (MHz)
  - Y轴: 系统总速率 (Mbps)
  - 分析: 带宽利用效率
```

**物理意义**：

```
Bd = 0.4 MHz → 窄带 → 每对带宽少 → ABS吸引力低
Bd = 1.2 MHz → 中等 → 平衡点（论文选择）
Bd = 3.0 MHz → 宽带 → 每对带宽多 → ABS吸引力高
```

---

### 4. 用户可扩展性实验（Scalability）

```yaml
实验代码: experiments/run_comparison.py --scalability
实验类型: user_number_sweep
目的: 验证系统在不同用户规模下的表现

固定参数:
  - ABS带宽: Bd = 1.2 MHz
  - ABS功率: Pd = 1 W
  - 卫星仰角: 10°
  - SNR: 20 dB

变化参数:
  - 用户数: [16, 32, 64]  ← 3个点
  - 配对数: [8, 16, 32]

Monte Carlo: 1000次/用户数

输出:
  - X轴: 用户数
  - Y轴: 平均用户速率 (Mbps)
  - 分析: 系统可扩展性、公平性
```

**关键观察**：

```
N=16:  每对带宽 = 1.2MHz/8 = 150kHz  → 速率高
N=32:  每对带宽 = 1.2MHz/16 = 75kHz  → 速率中等（基线）
N=64:  每对带宽 = 1.2MHz/32 = 37.5kHz → 速率低
```

---

### 5. 卫星仰角实验（Elevation Angle）

```yaml
实验代码: experiments/run_comparison.py --elevation
实验类型: elevation_sweep
目的: 研究卫星仰角对性能的影响

固定参数:
  - 用户数: N = 32
  - ABS带宽: Bd = 1.2 MHz
  - ABS功率: Pd = 1 W
  - SNR: 20 dB

变化参数:
  - 卫星仰角: [10°, 20°, 40°]  ← 3个点

Monte Carlo: 1000次/仰角

输出:
  - X轴: 卫星仰角 (度)
  - Y轴: 系统总速率 (Mbps)
  - 分析: 仰角对卫星链路质量的影响
```

**物理意义**：

```
E = 10° (低仰角):
  - 路径长 → 路径损耗大
  - Loo阴影严重
  → 卫星链路质量差 → ABS重要性高

E = 40° (高仰角):
  - 路径短 → 路径损耗小
  - Loo阴影轻
  → 卫星链路质量好 → ABS重要性降低
```

---

## 参数固定 vs 变化总结表

| 参数 | 主实验 | 消融实验 | 带宽实验 | 用户数实验 | 仰角实验 |
|------|--------|----------|----------|------------|----------|
| **SNR (dB)** | **变化** [0-30] | 固定 (20) | 固定 (20) | 固定 (20) | **变化** [0-30] |
| **用户数** | 固定 (32) | 固定 (32) | 固定 (32) | **变化** [16,32,64] | 固定 (32) |
| **ABS带宽 (MHz)** | 固定 (1.2) | 固定 (1.2) | **变化** [0.4,1.2,2.0,3.0] | 固定 (1.2) | 固定 (1.2) |
| **ABS功率 (W)** | 固定 (1) | 固定 (1) | 固定 (1) | 固定 (1) | 固定 (1) |
| **卫星仰角 (°)** | 固定 (10) | 固定 (10) | 固定 (10) | 固定 (10) | **变化** [10,20,40] |
| **Monte Carlo** | 1000 | 500 | 1000 | 1000 | 1000 |

---

## 为什么ABS功率始终固定在1W？

### 实际约束

1. **监管限制**
   - 无人机发射功率受国家法规严格限制
   - 大多数国家：UAV功率 ≤ 1W (30 dBm)

2. **能源约束**
   - UAV电池容量有限（通常2000-5000 mAh）
   - 功率越大 → 续航时间越短
   - 1W是续航与性能的平衡点

3. **实际部署**
   - 商用小型UAV（DJI、Parrot等）：0.5-2W
   - 1W是典型中等功率值

4. **研究焦点**
   - 论文关注：**算法优化**（位置、配对、决策）
   - 不是关注：硬件优化（功率放大器设计）
   - 固定功率可以公平比较算法性能

### 如果需要研究功率影响

可以设计额外实验：

```yaml
实验类型: power_sweep
变化参数: Pd = [0.5, 1.0, 2.0, 5.0] W
目的: 研究功率与性能的权衡
```

但这不是本项目的重点。

---

## 随机种子策略

### 种子生成公式

```python
seed = base_seed + i * n_realizations + r

其中：
  - base_seed: 配置文件中的随机种子（默认42）
  - i: SNR点索引（0, 1, 2, ..., 30）
  - n_realizations: 每点实现次数（1000）
  - r: 当前实现索引（0, 1, ..., 999）
```

### 具体示例

```python
# 配置
base_seed = 42
n_realizations = 1000

# SNR = 0dB, 第1次仿真
seed = 42 + 0*1000 + 0 = 42

# SNR = 0dB, 第2次仿真
seed = 42 + 0*1000 + 1 = 43

# SNR = 0dB, 第1000次仿真
seed = 42 + 0*1000 + 999 = 1041

# SNR = 10dB, 第1次仿真
seed = 42 + 10*1000 + 0 = 10042

# SNR = 30dB, 第1000次仿真
seed = 42 + 30*1000 + 999 = 31041
```

### 种子用途

每个种子控制：

1. **用户位置生成**
   ```python
   dist = UserDistribution(N=32, radius=500, seed=seed)
   user_positions = dist.generate_uniform_circle()
   ```

2. **卫星信道衰落**
   ```python
   sat_fading = loo_model.generate_fading(N=32, seed=seed)
   ```

3. **A2G信道衰落**
   ```python
   a2g_fading = rayleigh_model.generate_fading(N=32, seed=seed+1)
   ```

### 可重复性保证

```python
# 运行1
result1 = simulate_performance(snr_db_range, random_seed=42)

# 运行2（相同种子）
result2 = simulate_performance(snr_db_range, random_seed=42)

# 结果完全一致
assert np.allclose(result1, result2)  # ✓ 通过
```

---

## 时间复杂度分析

### 单次仿真时间分解

```python
# 步骤1：生成用户分布
user_positions = generate_uniform_circle()  # ~0.0001s

# 步骤2：优化ABS位置
abs_position = optimize_position()
  - k-means（基线）: ~0.001s
  - 梯度优化（增强）: ~0.01s

# 步骤3：计算卫星速率
sat_rates = compute_sat_rates()  # ~0.001s

# 步骤4：计算A2G速率
a2g_rates = compute_a2g_rates()  # ~0.001s

# 步骤5：用户配对
sat_pairs, abs_pairs = pairing()
  - 独立配对（基线）: ~0.001s
  - 联合配对（增强）: ~0.005s

# 步骤6：混合决策
decisions = hybrid_decision()
  - 贪婪决策（基线）: ~0.0005s
  - 整数规划（增强）: ~0.002s

# 总计
基线系统: ~0.005s/次
增强系统: ~0.02s/次
```

### 完整实验时间估算

| 实验类型 | SNR点数 | Monte Carlo | 单次时间 | 总时间 |
|---------|---------|-------------|---------|--------|
| **主实验（基线）** | 31 | 1000 | 0.005s | 155s ≈ 2.6分钟 |
| **主实验（增强）** | 31 | 1000 | 0.02s | 620s ≈ 10分钟 |
| **消融实验** | 1 | 500 | 0.02s | 10s |
| **带宽实验** | 1 | 1000 | 0.02s | 20s |
| **用户数实验** | 1 | 1000 | 0.02s | 20s |

**完整论文实验（全部）**：
```
总时间 = 主实验×6系统 + 消融实验 + 带宽实验 + 用户数实验
       ≈ 10分钟×6 + 10s + 20s + 20s
       ≈ 60-70分钟
```

### 并行加速（可选）

```python
# 使用多进程并行
from multiprocessing import Pool

def simulate_one_snr(snr_db):
    return simulate_single_realization(snr_db, ...)

# 并行执行（8核CPU）
with Pool(8) as pool:
    results = pool.map(simulate_one_snr, snr_db_range)

# 加速比：~6-7x（理想8x，考虑开销）
# 时间：10分钟 → 1.5分钟
```

---

## 配置文件示例

### config.py（核心参数）

```python
class Config:
    # 系统参数
    N = 32                      # 用户数（固定）
    K = 16                      # 配对数 = N/2
    coverage_radius = 500       # 覆盖半径 (m)

    # ABS参数
    Bd_options = [0.4e6, 1.2e6, 2.0e6, 3.0e6]  # 带宽选项 (Hz)
    Pd = 1.0                    # 功率 (W)（固定）
    Pd_dBm = 30                 # 功率 (dBm)

    # 卫星参数
    Bs = 5e6                    # 带宽 (Hz)（固定）
    Ps_dB = np.arange(0, 31)    # SNR范围 [0-30] dB

    # Monte Carlo
    n_monte_carlo = 1000        # 默认次数
    random_seed = 42            # 基础随机种子
```

### experiments/config_comparison.yaml（实验配置）

```yaml
# 场景参数
scenarios:
  # 主实验
  snr_range: [0, 5, 10, 15, 20, 25, 30]  # 7个点（简化版）
  elevation_angle: 10
  abs_bandwidth: 1.2e6
  user_number: 32

  # 扩展实验
  elevation_angles: [10, 20, 40]
  abs_bandwidths: [0.4e6, 1.2e6, 2.0e6, 3.0e6]
  user_numbers: [16, 32, 64]

# Monte Carlo参数
monte_carlo:
  n_realizations: 1000          # 主实验
  n_realizations_ablation: 500  # 消融实验
  random_seed: 42

# 对比基线
baselines:
  - name: "Original-SATCON"
    use_module1: false
    use_module2: false
    use_module3: false

  - name: "Proposed-Full"
    use_module1: true
    use_module2: true
    use_module3: true
```

---

## 总结

### 问题1总结：Monte Carlo仿真

- **不是**：1000个不同的SNR点
- **而是**：每个SNR点重复1000次独立仿真
- **总次数**：31个SNR点 × 1000次 = 31,000次仿真
- **目的**：消除随机性，获得统计可靠的平均结果
- **标准**：符合IEEE通信领域论文标准（1000次）

### 问题2总结：参数固定策略

| 实验类型 | 变化参数 | 固定参数 |
|---------|---------|---------|
| **主实验** | SNR [0-30dB] | 用户数(32)、带宽(1.2MHz)、功率(1W)、仰角(10°) |
| **消融实验** | 系统配置(6种) | SNR(20dB)、用户数(32)、带宽(1.2MHz) |
| **带宽实验** | 带宽[0.4-3.0MHz] | SNR(20dB)、用户数(32)、功率(1W) |
| **用户数实验** | 用户数[16,32,64] | SNR(20dB)、带宽(1.2MHz)、功率(1W) |
| **仰角实验** | 仰角[10,20,40°] | SNR(20dB)、用户数(32)、带宽(1.2MHz) |

**关键原则**：
- 每次实验只变化1个参数（控制变量法）
- ABS功率始终固定（实际约束）
- Monte Carlo次数根据实验重要性调整（1000或500次）

---

**文档版本**：v1.0
**创建日期**：2025-12-12
**作者**：SATCON Enhancement Project
**相关文档**：[SATCON系统完整工作流程分析.md](./SATCON系统完整工作流程分析.md)
