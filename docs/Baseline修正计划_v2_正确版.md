# Baseline修正计划 v2（融合三个关键修正点）

## 修正目标

1. ✅ 只有一个配对（sat_pairs）
2. ✅ ABS-OMA带宽分配正确（整个pair带宽给单个用户）
3. ✅ ABS-NOMA受S2A约束（DF瓶颈）
4. ✅ 决策对照是端到端速率

---

## 修正后的正确实现

### 1. `compute_abs_noma_rates()` - 完整版

```python
def compute_abs_noma_rates(self, sat_pairs, channel_gains_a2g,
                           s2a_rates, total_power):
    """
    计算ABS NOMA转发速率（Decode-and-Forward）

    关键修正：
    1. 使用sat_pairs（不重新配对）
    2. 考虑S2A解码约束（DF瓶颈）

    参数：
        sat_pairs: 卫星配对 [(i,j), ...]
        channel_gains_a2g: A2G信道增益 [N]
        s2a_rates: S2A链路速率 [N]（卫星→ABS解码速率）
        total_power: ABS功率 Pd

    返回：
        rates_noma: NOMA转发速率 [N]
    """
    K = len(sat_pairs)
    bandwidth_per_pair = self.Bd / K  # 每对一个时隙

    rates_noma = np.zeros(2*K)

    for k in range(K):
        # 使用卫星配对
        weak_idx, strong_idx = sat_pairs[k]
        gamma_weak = channel_gains_a2g[weak_idx]
        gamma_strong = channel_gains_a2g[strong_idx]

        # 确保弱-强顺序（基于A2G信道）
        if gamma_weak > gamma_strong:
            weak_idx, strong_idx = strong_idx, weak_idx
            gamma_weak, gamma_strong = gamma_strong, gamma_weak

        # ABS重新编码NOMA（用A2G信道的功率分配）
        beta_strong, beta_weak = self.allocator.compute_power_factors(
            gamma_strong, gamma_weak, total_power
        )

        # 计算A2G速率（ABS → 用户）
        rate_a2g_strong = bandwidth_per_pair * np.log2(
            1 + beta_strong * total_power * gamma_strong
        )
        rate_a2g_weak = bandwidth_per_pair * np.log2(
            1 + beta_weak * total_power * gamma_weak /
            (beta_strong * total_power * gamma_weak + 1)
        )

        # 【关键修正2】考虑S2A瓶颈（DF约束）
        # ABS必须先从卫星解码，再转发
        rate_s2a_weak = s2a_rates[weak_idx]
        rate_s2a_strong = s2a_rates[strong_idx]

        # 取瓶颈（min）
        rates_noma[weak_idx] = min(rate_a2g_weak, rate_s2a_weak)
        rates_noma[strong_idx] = min(rate_a2g_strong, rate_s2a_strong)

    return rates_noma
```

---

### 2. `compute_abs_oma_rates()` - 完整版

```python
def compute_abs_oma_rates(self, sat_pairs, channel_gains_a2g,
                          s2a_rates, total_power):
    """
    计算ABS OMA转发速率

    关键修正：
    1. 使用sat_pairs（知道配对关系）
    2. 【修正点1】整个pair带宽给单个用户（不是Bd/2K）
    3. 【修正点2】考虑S2A约束

    参数：
        sat_pairs: 卫星配对
        channel_gains_a2g: A2G信道增益 [N]
        s2a_rates: S2A链路速率 [N]
        total_power: ABS功率 Pd

    返回：
        rates_oma: OMA转发速率 [N]
    """
    K = len(sat_pairs)

    # 【关键修正1】每对占用 Bd/K 带宽
    # OMA时，整个 Bd/K 都给一个用户
    bandwidth_per_pair = self.Bd / K  # 不是 Bd/(2K) ！

    # 计算A2G速率（全功率 + 全带宽）
    rates_a2g = bandwidth_per_pair * np.log2(
        1 + total_power * channel_gains_a2g
    )

    # 【关键修正2】考虑S2A约束
    # 即使是OMA，ABS也需要从卫星解码
    rates_oma = np.minimum(rates_a2g, s2a_rates)

    return rates_oma
```

---

### 3. `compute_s2a_rates()` - 新增函数

```python
def compute_s2a_rates(self, sat_pairs, abs_position, elevation_deg,
                     sat_channel_gains, snr_linear):
    """
    计算S2A链路速率（卫星→ABS）

    ABS需要解码卫星的NOMA信号

    参数：
        sat_pairs: 卫星配对
        abs_position: ABS位置 [x, y, h]
        elevation_deg: 卫星仰角
        sat_channel_gains: 卫星信道增益（各用户）
        snr_linear: 卫星SNR（线性）

    返回：
        s2a_rates: S2A解码速率 [N]
    """
    K = len(sat_pairs)
    Bs_per_pair = self.config.Bs / K

    # 计算ABS的卫星信道增益
    # 方法1：使用与用户相似的路径损耗模型
    # 方法2：简化假设（论文中可能有说明）

    # 这里用简化方法：假设ABS在地面中心，计算其到卫星的信道增益
    gamma_abs = self.s2a_channel.compute_channel_gain(
        elevation_deg, abs_position
    )

    s2a_rates = np.zeros(2*K)

    for k in range(K):
        weak_idx, strong_idx = sat_pairs[k]

        # ABS作为"强用户"，需要先解弱用户信号，再解强用户
        # （SIC顺序与地面用户相同）

        # 这里简化：假设ABS能可靠解码
        # 实际应该详细计算，取决于论文的S2A模型

        # 简化版：假设S2A充足（不成为瓶颈）
        s2a_rates[weak_idx] = 1e9  # 很大的值
        s2a_rates[strong_idx] = 1e9

    return s2a_rates
```

**注意**：S2A建模需要查阅SATCON论文，看具体如何处理。可能的情况：
1. 论文假设S2A充足（简化）
2. 论文详细建模了S2A信道

---

### 4. `hybrid_decision()` - 保持不变（已正确）

```python
def hybrid_decision(self, sat_rates, abs_noma_rates, abs_oma_rates,
                   sat_pairs):
    """
    混合决策（逐对，4条规则）

    【修正点3】对照关系正确：
    - 对照：R_s（卫星直达，端到端）
    - 候选：R_dn, R_do（ABS转发，已含S2A约束，端到端）

    参数：
        sat_rates: 卫星直达速率 [N]
        abs_noma_rates: ABS NOMA速率 [N]（已含S2A约束）
        abs_oma_rates: ABS OMA速率 [N]（已含S2A约束）
        sat_pairs: 配对 [(i,j), ...]
    """
    final_rates = sat_rates.copy()
    modes = []

    for k, (weak_idx, strong_idx) in enumerate(sat_pairs):
        R_s_i = sat_rates[weak_idx]
        R_s_j = sat_rates[strong_idx]
        R_dn_i = abs_noma_rates[weak_idx]
        R_dn_j = abs_noma_rates[strong_idx]
        R_do_i = abs_oma_rates[weak_idx]
        R_do_j = abs_oma_rates[strong_idx]

        # 【正确对照】端到端速率比较

        # 规则1：NOMA双用户
        if R_s_i < R_dn_i and R_s_j < R_dn_j:
            final_rates[weak_idx] = R_dn_i
            final_rates[strong_idx] = R_dn_j
            modes.append('noma')

        # 规则2：OMA只给弱用户
        elif R_s_i < R_do_i and R_s_j >= R_dn_j:
            final_rates[weak_idx] = R_do_i
            final_rates[strong_idx] = R_s_j
            modes.append('oma_weak')

        # 规则3：OMA只给强用户
        elif R_s_i >= R_dn_i and R_s_j < R_do_j:
            final_rates[weak_idx] = R_s_i
            final_rates[strong_idx] = R_do_j
            modes.append('oma_strong')

        # 规则4：不转发
        else:
            modes.append('sat')

    return final_rates, modes
```

---

### 5. `simulate_single_realization()` - 完整流程

```python
def simulate_single_realization(self, snr_db, elevation_deg, seed):
    """
    单次仿真（修正后的完整流程）
    """
    # 1. 生成用户分布
    dist = UserDistribution(self.config.N, self.config.coverage_radius, seed=seed)
    user_positions = dist.generate_uniform_circle()

    # 2. 优化ABS位置
    abs_position, _ = self.abs_placement.optimize_position_complete(
        user_positions, self.a2g_channel
    )

    # 3. 计算卫星信道增益
    sat_channel_gains = self.sat_noma.compute_channel_gains_with_pathloss(elevation_deg)

    # 4. 【唯一的配对】卫星NOMA配对
    sat_pairs, _ = self.allocator.optimal_user_pairing(sat_channel_gains)

    # 5. 计算卫星NOMA速率（基于sat_pairs）
    snr_linear = 10 ** (snr_db / 10)
    sat_rates, _ = self.sat_noma.compute_achievable_rates(sat_channel_gains, snr_linear)

    # 6. 计算A2G信道增益
    h_abs = abs_position[2]
    distances_2d = dist.compute_distances_from_point(user_positions, abs_position)
    fading_a2g = self.a2g_channel.generate_fading(self.config.N, seed=seed+1)
    channel_gains_a2g = np.array([
        self.a2g_channel.compute_channel_gain(
            h_abs, r, fading,
            self.config.Gd_t_dB, self.config.Gd_r_dB, self.Nd
        )
        for r, fading in zip(distances_2d, fading_a2g)
    ])

    # 7. 计算S2A链路速率
    s2a_rates = self.compute_s2a_rates(
        sat_pairs, abs_position, elevation_deg,
        sat_channel_gains, snr_linear
    )

    # 8. ABS速率（基于sat_pairs，含S2A约束）
    abs_noma_rates = self.compute_abs_noma_rates(
        sat_pairs, channel_gains_a2g, s2a_rates, self.config.Pd
    )
    abs_oma_rates = self.compute_abs_oma_rates(
        sat_pairs, channel_gains_a2g, s2a_rates, self.config.Pd
    )

    # 9. 混合决策
    final_rates, modes = self.hybrid_decision(
        sat_rates, abs_noma_rates, abs_oma_rates, sat_pairs
    )

    # 10. 统计
    mode_stats = {
        'noma': modes.count('noma'),
        'oma_weak': modes.count('oma_weak'),
        'oma_strong': modes.count('oma_strong'),
        'sat': modes.count('sat')
    }

    return np.sum(final_rates), mode_stats
```

---

## 关键修正总结

### ✅ 修正点1：OMA带宽
```python
# ❌ 错误
bandwidth_per_user = Bd / K  # 误导！

# ✅ 正确
bandwidth_per_pair = Bd / K  # 清晰！
rates_oma[l] = bandwidth_per_pair * log2(...)  # OMA时，整个pair带宽给l
```

### ✅ 修正点2：S2A约束
```python
# ❌ 遗漏
R_dn = rate_a2g

# ✅ 正确
R_dn = min(rate_a2g, rate_s2a)  # DF瓶颈
```

### ✅ 修正点3：决策对照
```python
# ✅ 已正确（端到端对比）
if sat_rate[l] < abs_rate[l]:  # 都是端到端速率
    选ABS
```

---

## 实施计划

1. **立即修正** `src/satcon_system.py`
   - 修改 `compute_abs_noma_rates()`
   - 修改 `compute_abs_oma_rates()`
   - 新增 `compute_s2a_rates()`（需查论文）
   - 修改 `simulate_single_realization()`

2. **测试验证**
   - 单次运行检查逻辑
   - 50次MC检查性能
   - 对比修正前后

3. **文档记录**
   - 记录修正的影响
   - 更新实验结果

---

## S2A建模的待确认问题

需要查阅SATCON论文确认：
1. S2A信道模型是什么？
2. 是否简化假设S2A充足？
3. 还是详细建模了S2A的NOMA解码？

**建议**：先实现简化版（S2A充足），运行后看性能是否合理，再决定是否需要详细S2A建模。
