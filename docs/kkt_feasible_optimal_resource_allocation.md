# KKT 可行最优 S2A 资源分配问题总结

本文档系统性总结了 KKT 基于约束的 S2A 带宽分配问题的数学模型、物理含义、
可行最优性（feasible optimality）定义，以及其在 Greedy–KKT–L-BFGS-B 框架中的角色。
本文档可直接用于论文写作、审稿回复及后续算法扩展。

---

## 1. 问题背景：为什么需要 KKT 资源分配？

在 UAV/ABS 作为 Decode-and-Forward（DF）中继的系统中，
任意经 ABS 转发的用户，其端到端下行速率满足：

\[
R_u^{\mathrm{E2E}} = \min\!\left(
R_u^{\mathrm{A2G}},
R_u^{\mathrm{S2A}}(b_k)
\right)
\]

其中：

- \(R_u^{\mathrm{A2G}}\)：ABS → 用户（A2G）链路的可达速率  
- \(R_u^{\mathrm{S2A}}(b_k)\)：卫星 → ABS（S2A）链路的可达速率  
- \(b_k\)：分配给第 \(k\) 个卫星侧用户对（SAT-pair）的 S2A 带宽  

**核心问题在于：**

- A2G 速率由 ABS 位置、配对和模式决定
- S2A 带宽是全系统共享、受限的关键资源
- 若 S2A 带宽不足，则 A2G 的潜在性能会被“压低”

因此，S2A 资源分配的目标不是提升 A2G 上限，
而是**消除 DF 回程瓶颈，使端到端速率能够真正达到 A2G 的可达水平**。

---

## 2. KKT 资源分配的优化目标（核心定义）

### 2.1 设计目标

对于每一个需要 ABS 转发的用户 \(u\)，希望满足：

\[
R_u^{\mathrm{S2A}}(b_k) \ge R_u^{\mathrm{A2G}}
\quad \Rightarrow \quad
R_u^{\mathrm{E2E}} = R_u^{\mathrm{A2G}}
\]

即：

> **S2A 不再成为瓶颈，端到端速率完全由 A2G 决定**

---

### 2.2 为什么这是“可行最优（Feasible Optimal）”而不是“全局最优”？

- KKT 分配并不试图：
  - 改变用户的传输模式
  - 改变 A2G 速率本身
- 而是：
  - 在给定模式与 A2G 速率的前提下
  - 找到**最小的 S2A 带宽分配**
  - 使系统满足 DF 可行性约束

因此，KKT 解的本质是：

> **在给定结构下，使系统“刚好可行”的最优解**

---

## 3. 数学模型与逐项解释

### 3.1 SAT 模式（无需 S2A）

若第 \(k\) 个用户对采用 `sat` 模式：

\[
b_k = 0
\]

含义：
- 用户直接由卫星服务
- ABS 不参与解码与转发
- 分配任何 S2A 带宽都是浪费

---

### 3.2 NOMA / OMA 模式（需要 S2A）

当第 \(k\) 个用户对需要 ABS 转发时，
S2A 链路上传输的是卫星侧形成的 NOMA 叠加信号。

#### 3.2.1 S2A 速率模型（NOMA）

对于 SAT-pair 中的强、弱用户：

- **强用户（SIC 后）**：
\[
R_{k,\mathrm{strong}}^{\mathrm{S2A}}
=
b_k \log_2\!\left(
1 + \beta_{\mathrm{strong}} \cdot \mathrm{SNR} \cdot h_{s2a}
\right)
\]

- **弱用户（受强用户干扰）**：
\[
R_{k,\mathrm{weak}}^{\mathrm{S2A}}
=
b_k \log_2\!\left(
1 +
\frac{\beta_{\mathrm{weak}} \cdot \mathrm{SNR} \cdot h_{s2a}}
{1 + \beta_{\mathrm{strong}} \cdot \mathrm{SNR} \cdot h_{s2a}}
\right)
\]

其中：

- \(b_k\)：S2A 带宽（Hz）
- \(\beta_{\mathrm{strong}}, \beta_{\mathrm{weak}}\)：卫星侧功率分配系数
- \(\mathrm{SNR}\)：卫星发射 SNR（线性值）
- \(h_{s2a}\)：S2A 信道增益

---

#### 3.2.2 反解所需带宽（核心步骤）

由于：
\[
R_u^{\mathrm{S2A}}(b_k)
=
b_k \log_2(1+\gamma_u^{\mathrm{eff}})
\]

为了满足：
\[
R_u^{\mathrm{S2A}}(b_k) \ge R_u^{\mathrm{A2G}}
\]

可以反解得到最小带宽需求：
\[
b_u =
\frac{R_u^{\mathrm{A2G}}}
{\log_2(1+\gamma_u^{\mathrm{eff}})}
\]

---

### 3.2.3 为什么取 max？

在 NOMA 转发模式下：

- ABS 必须成功解码并转发 **两名用户的信号**
- 因此必须同时满足：
\[
b_k \ge b_{\mathrm{weak}}, \quad b_k \ge b_{\mathrm{strong}}
\]

最终：
\[
\boxed{
b_k = \max(b_{\mathrm{weak}}, b_{\mathrm{strong}})
}
\]

这是 DF 中继 + NOMA 解码的**硬约束**，不是启发式选择。

---

### 3.3 OMA 模式的特殊情况

- `oma_weak`：只约束弱用户的 \(R_{\mathrm{A2G}}\)
- `oma_strong`：只约束强用户的 \(R_{\mathrm{A2G}}\)

但注意：
> **S2A 侧仍然解码的是卫星 NOMA 叠加信号**，  
因此有效 SINR 的表达形式不变。

---

## 4. 总带宽约束与比例缩减

### 4.1 可行性约束

系统满足：
\[
\sum_{k=1}^{K} b_k \le B_s
\]

若：
\[
\sum_{k=1}^{K} b_k^\star > B_s
\]

则采用比例缩减：
\[
b_k \leftarrow
b_k^\star \cdot
\frac{B_s}{\sum_{k=1}^{K} b_k^\star}
\]

---

### 4.2 为什么不回溯 Greedy？

- 回溯意味着重新改变 transmission mode
- 将问题变为混合整数 + 连续联合优化
- 复杂度显著上升，稳定性下降

当前设计采用：

> **连续缩放保证可行性，而非离散结构回溯**

这是一个**工程上合理、论文中可防守的设计选择**。

---

## 5. KKT 解的性质总结（审稿人关心点）

### 5.1 它优化了什么？

- ❌ 不提升 A2G 上限
- ❌ 不改变传输模式
- ✅ 消除 DF 瓶颈
- ✅ 释放 A2G 潜力
- ✅ 保证系统可行

---

### 5.2 为什么称为“可行最优”？

因为：

- 在给定 transmission mode 和 A2G 速率的条件下
- 它使用 **最小的 S2A 带宽**
- 达成 **最大可能的端到端速率**

即：
\[
R_u^{\mathrm{E2E}} = R_u^{\mathrm{A2G}}
\quad (\text{在带宽允许时})
\]

---

## 6. 可直接用于论文的一句话总结

> The KKT-based S2A bandwidth allocation does not aim to improve the access-link capacity, but to eliminate the DF bottleneck with minimum backhaul resource consumption, thereby achieving a feasible-optimal end-to-end performance under the given transmission structure.

---

## 7. 后续算法扩展方向（基于本模型）

- 带宽感知 Greedy 模式选择
- 基于 S2A 紧张度的模式回溯
- 非比例的优先级/公平性缩放
- 联合 S2A–A2G 带宽显式优化

以上扩展均以当前 KKT 可行最优模型为理论基线。

---
