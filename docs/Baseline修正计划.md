# Baseline修正计划：按SATCON论文正确逻辑重构

## 问题诊断

### 当前错误实现
```python
# 现有代码（错误）
sat_pairs = optimal_pairing(sat_gains)    # 卫星配对
abs_pairs = optimal_pairing(a2g_gains)    # ❌ ABS不应该做配对！

# 计算速率时使用两个不同的配对
for k in range(K):
    sat_i, sat_j = sat_pairs[k]
    abs_m, abs_n = abs_pairs[k]  # ❌ 这是错误的
    ...
```

### 正确逻辑（论文）
```python
# 正确实现
sat_pairs = optimal_pairing(sat_gains)    # 唯一的配对

# ABS基于卫星的配对来决策
for k in range(K):
    i, j = sat_pairs[k]  # 只用一个配对

    # 计算3种速率
    R_s_i, R_s_j = 卫星直达(sat_pairs[k], sat_gains)
    R_dn_i, R_dn_j = ABS_NOMA(sat_pairs[k], a2g_gains)  # 用同一对！
    R_do_i, R_do_j = ABS_OMA(sat_pairs[k], a2g_gains)

    # 4条规则决策
    ...
```

---

## 修正步骤

### 第1步：修正`src/satcon_system.py`

#### 需要修改的函数

**1. `compute_abs_noma_rates()` - 核心修改**

```python
# 当前签名（错误）
def compute_abs_noma_rates(self, user_positions, abs_position,
                           channel_gains_a2g, total_power):
    # 错误：内部调用了 optimal_user_pairing(channel_gains_a2g)
    pairs, paired_gains = self.allocator.optimal_user_pairing(channel_gains_a2g)
    ...

# 修正后签名
def compute_abs_noma_rates(self, sat_pairs, channel_gains_a2g, total_power):
    """
    基于卫星配对计算ABS NOMA转发速率

    参数：
        sat_pairs: 卫星配对（必须使用！）
        channel_gains_a2g: A2G信道增益
        total_power: ABS功率

    返回：
        rates_noma: NOMA速率 [N]
    """
    K = len(sat_pairs)
    bandwidth_per_pair = self.Bd / K
    rates_noma = np.zeros(2*K)

    for k in range(K):
        # 使用卫星的配对！
        weak_idx, strong_idx = sat_pairs[k]
        gamma_weak = channel_gains_a2g[weak_idx]
        gamma_strong = channel_gains_a2g[strong_idx]

        # 确保弱-强顺序（基于A2G信道）
        if gamma_weak > gamma_strong:
            weak_idx, strong_idx = strong_idx, weak_idx
            gamma_weak, gamma_strong = gamma_strong, gamma_weak

        # ABS重新编码NOMA（Decode-and-Forward）
        beta_strong, beta_weak = self.allocator.compute_power_factors(
            gamma_strong, gamma_weak, total_power
        )

        # 速率计算
        rate_strong = bandwidth_per_pair * np.log2(
            1 + beta_strong * total_power * gamma_strong
        )
        rate_weak = bandwidth_per_pair * np.log2(
            1 + beta_weak * total_power * gamma_weak /
            (beta_strong * total_power * gamma_weak + 1)
        )

        rates_noma[weak_idx] = rate_weak
        rates_noma[strong_idx] = rate_strong

    return rates_noma
```

**2. `compute_abs_oma_rates()` - 需要修改**

```python
# 当前签名（缺少配对信息）
def compute_abs_oma_rates(self, channel_gains_a2g, total_power):
    ...

# 修正后
def compute_abs_oma_rates(self, sat_pairs, channel_gains_a2g, total_power):
    """
    基于卫星配对计算ABS OMA速率

    注意：OMA是逐用户的，但需要知道配对来正确分配带宽
    """
    K = len(sat_pairs)
    bandwidth_per_user = self.Bd / K  # K对，每对一个时隙，每用户半个时隙

    rates_oma = bandwidth_per_user * np.log2(
        1 + total_power * channel_gains_a2g
    )

    return rates_oma
```

**3. `hybrid_decision()` - 保持不变**

这个函数逻辑是对的，不需要修改。

**4. `simulate_single_realization()` - 修改调用**

```python
# 修正后的流程
def simulate_single_realization(self, snr_db, elevation_deg, seed):
    # 1. 生成用户
    dist = UserDistribution(...)
    user_positions = dist.generate_uniform_circle()

    # 2. 优化ABS位置
    abs_position, _ = self.abs_placement.optimize_position_complete(...)

    # 3. 计算卫星信道增益
    sat_channel_gains = self.sat_noma.compute_channel_gains_with_pathloss(elevation_deg)

    # 4. 卫星配对（唯一的配对）
    sat_pairs, _ = self.allocator.optimal_user_pairing(sat_channel_gains)

    # 5. 计算卫星NOMA速率
    snr_linear = 10 ** (snr_db / 10)
    sat_rates, _ = self.sat_noma.compute_achievable_rates(sat_channel_gains, snr_linear)

    # 6. 计算A2G信道增益
    channel_gains_a2g = ...

    # 7. ABS速率（基于卫星配对！）
    abs_noma_rates = self.compute_abs_noma_rates(
        sat_pairs, channel_gains_a2g, self.config.Pd
    )
    abs_oma_rates = self.compute_abs_oma_rates(
        sat_pairs, channel_gains_a2g, self.config.Pd
    )

    # 8. 混合决策
    final_rates, modes = self.hybrid_decision(
        sat_rates, abs_noma_rates, abs_oma_rates, s2a_rates, sat_pairs
    )

    return np.sum(final_rates), modes
```

---

### 第2步：验证修正后的Baseline

**测试脚本**：创建 `experiments/test_corrected_baseline.py`

```python
"""
验证修正后的Baseline逻辑

目标：
1. 确认只有一个配对（sat_pairs）
2. ABS基于sat_pairs做决策
3. 性能合理
"""

def test_single_pairing():
    """测试只有一个配对"""
    system = SATCONSystem(config, 1.2e6)

    # 运行一次仿真
    sum_rate, modes = system.simulate_single_realization(
        snr_db=20, elevation_deg=10, seed=42
    )

    print(f"总速率: {sum_rate/1e6:.2f} Mbps")
    print(f"模式统计: {Counter(modes)}")

    # 验证：不应该有abs_pairs的计算
    # (通过代码审查确认)

def test_performance():
    """对比修正前后的性能"""
    # 修正前：错误的双配对逻辑
    # 修正后：正确的单配对逻辑

    # 运行50次MC
    ...
```

---

### 第3步：重新设计增强方案

修正Baseline后，需要重新思考增强方向。

#### 可能的增强方向

**方向1：优化卫星配对策略**
```
问题：optimal_user_pairing只考虑卫星信道
改进：在配对时同时考虑A2G信道

思路：
- 目标：最大化"ABS能提供的增益"
- 配对时不仅看sat_gains，也看a2g_gains
- 找到一个配对，使得"ABS转发后的总速率"最大
```

**方向2：增强ABS决策逻辑**
```
问题：4条规则是启发式的，可能不是最优
改进：用ILP全局优化16对的决策

当前：逐对独立决策（规则1-4）
改进：联合优化所有对的决策
- 考虑ABS资源约束
- 考虑用户公平性
- 全局最优
```

**方向3：ABS位置与配对联合优化**
```
问题：ABS位置和配对是分步优化的
改进：迭代优化位置和配对

流程：
1. 初始位置 → 配对
2. 固定配对 → 优化位置
3. 固定位置 → 优化配对
4. 迭代直到收敛
```

---

## 实施优先级

### 立即执行（修正Baseline）
1. ✅ 修改`src/satcon_system.py`的相关函数
2. ✅ 创建测试脚本验证
3. ✅ 运行50次MC确认性能

### 后续执行（重新设计增强）
4. 🔄 分析哪个增强方向最有潜力
5. 🔄 实现新的增强模块
6. 🔄 对比性能

---

## 预期结果

### Baseline修正后
- 代码逻辑与论文一致
- 性能可能略有变化（因为之前的"错误"可能偶然带来了提升）
- 建立正确的基准线

### 重新设计增强后
- 基于正确理解的改进
- 性能提升更有说服力
- 论文逻辑更清晰

---

## 需要你确认的问题

1. **是否立即开始修正Baseline？**
   - 修改`src/satcon_system.py`
   - 测试验证

2. **修正后如何处理现有的增强代码？**
   - `src_enhanced/`目录下的代码暂时保留？
   - 还是删除重写？

3. **重新设计增强方案的方向？**
   - 优先考虑哪个方向？
   - 目标性能提升是多少？

请告诉我下一步该做什么！
