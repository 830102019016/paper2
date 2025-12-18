# Fig.2 复现差异：最可能的原因与修正建议（基于你当前实现）

你现在的曲线出现两个“典型症状”：
- **Bd 越小，SATCON 的“谱效率”越大**（Bd=0.4MHz 反而最高）；
- 高 SNR 区间 SATCON 曲线 **飙到 50–60 bit/s/Hz**，明显比论文 Fig.2 的量级大很多。

结合你上传的 `satcon_system.py`，最可能的问题集中在 **“谱效率归一化口径”** 与 **“S2A 链路带宽/噪声的口径”** 两处。

---

## 1) 你现在把谱效率除以 **Bd** 了（导致 Bd 越小，SE 越大）

在 `SATCONSystem.simulate_performance()` 里，进度条和最终输出都用：

- `current_se = mean_sum_rate / self.Bd`
- `mean_se = mean_sum_rates / self.Bd`

这会直接导致：**同一个 sum-rate，Bd 越小，除出来的 SE 越大**，所以 Bd=0.4MHz 曲线最高是必然的（这也解释了你图里的排序）。  
证据：`mean_se = mean_sum_rates / self.Bd` fileciteturn6file0L435-L443

✅ **建议修正（更符合论文 Fig.2 的常见定义）**  
把 SATCON 的谱效率统一按 **卫星下行带宽 Bs** 归一化（和 SAT–NOMA 一致），例如：

- `mean_se = mean_sum_rates / self.config.Bs`

如果你希望“系统总带宽口径”，也可以尝试 `(Bs + Bd)`，但从你现有 SAT–NOMA 复现口径看，**优先用 Bs** 更可能对齐 Fig.2。

---

## 2) 你把 S2A 解码链路的带宽/噪声也绑定到了 **Bd**（会系统性抬高 SATCON）

### 2.1 S2A 的噪声功率用的是 Bd
初始化时：

- `self.Nsd = config_obj.get_s2a_noise_power(abs_bandwidth)`  
并且注释明确写了“基于 Bd”。 fileciteturn6file0L78-L82

这会造成一个很不合理的现象：**当 Bd 变小，S2A 噪声也变小** → S2A 信道增益 `h_s2a`（本质上是“(G_t G_r)/(L N)”）会被放大 → S2A 速率变大 → DF 瓶颈更不容易卡住 → SATCON 曲线被整体抬高。

> 注意：论文里 S2A 是“卫星→ABS 解码卫星下行 NOMA 复合信号”。直觉上它应使用**卫星的下行带宽 Bs**（或者 Bs/K 的分配），而不是 ABS 自己的 Bd。

### 2.2 S2A 速率公式也在用 Bd/K
在 `compute_s2a_rates()` 里：

- `Bd_per_pair = self.Bd / K`
- `rate_s2a_* = Bd_per_pair * log2(1 + ...)`

证据：fileciteturn6file0L259-L309

但你同一段注释里又写了“强用户：R_sd_j = (Bs/K) * log2(...)”  
（注释写 Bs/K，但代码用 Bd/K）。fileciteturn6file0L299-L306

✅ **建议修正（更贴近论文物理含义）**  
- S2A 带宽：用 `Bs/K`（不是 Bd/K）
- S2A 噪声功率：用 `get_s2a_noise_power(self.config.Bs)`（或按 `Bs/K` 计算噪声，关键是**不要跟 Bd 绑定**）

这样做的效果通常是：
- S2A 约束更“真实”，DF 的 `min(R_a2g, R_s2a)` 更可能在高 SNR 也产生瓶颈（压住 SATCON 的爆炸式增长）；
- Bd 改变时，曲线不会出现“Bd 越小越夸张”的反常趋势。

---

## 3) “S2A 侧 β 功率分配”你是重新算的，可能与卫星侧不一致

你在 S2A SIC 解码时，β 是这样来的：

```python
beta_strong, beta_weak = allocator.compute_power_factors(gamma_sat_strong, gamma_sat_weak, 1.0)
```

证据：fileciteturn6file0L294-L297

这有两个风险：
1. `compute_power_factors()` 的设计目标可能是“ABS→用户(A2G) 的功分”或带保护机制，与论文卫星侧 β 的生成规则不完全一致；
2. 即使同一个函数，卫星侧和 ABS 侧如果用了不同的输入/约束，也会让“ABS 解码卫星 NOMA”变得不匹配。

✅ **建议修正**
- 最稳妥：让 `SatelliteNOMA.compute_achievable_rates()` 在计算卫星直达速率时**把每对的 β 输出并缓存**，S2A 解码直接复用这一组 β（因为 S2A 解码的信号就是卫星发出的那份叠加信号）。

---

## 推荐的最小改动顺序（先改就最可能立刻对齐 Fig.2）

1) **把 SATCON 的 SE 归一化从 `/Bd` 改成 `/Bs`**（立刻解决 “Bd 越小 SE 越大” 的反常排序）  
2) **把 S2A 的带宽与噪声从 Bd 口径改为 Bs 口径**（压住高 SNR 区间的过高增益，且消除 Bd→S2A 的耦合）  
3) **S2A 的 β 直接复用卫星侧 β**（减少实现与论文之间的“隐性偏差”）

---

## 你现在的图为何会“看起来差很多”（一句话总结）
当前实现里，**谱效率和 S2A 解码能力都被 Bd 强绑定**（`mean_se /= Bd`，`R_s2a ∝ Bd/K`，`Nsd ∝ Bd`），导致 **Bd 越小越占便宜**，并在高 SNR 时把 DF 瓶颈“抹平”，从而把 SATCON 曲线整体抬到远高于论文的量级。 fileciteturn6file0L78-L82 fileciteturn6file0L259-L309 fileciteturn6file0L435-L443
