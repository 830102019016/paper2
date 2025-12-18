# SATCON 框架下的算法创新空间总结

> 本文档用于系统性总结：**在保持 SATCON 原有系统框架与优化目标不变的前提下，可引入的算法级创新空间**。  
> 该总结可直接用于论文的 *Motivation*、*Related Work*、或 *Contributions* 部分，也可作为后续算法设计的路线图。

---

## 一、基本判断（Baseline 定位）

SATCON（Satellite–Aerial Terrestrial Hybrid NOMA）工作的核心贡献在于：
- 提出了 **卫星 + ABS 协同的系统框架**；
- 给出了 **SAT / ABS-NOMA / ABS-OMA 的混合传输机制**。

然而，在该框架下：

> **几乎所有关键决策均由启发式规则或固定策略完成，而非优化算法驱动。**

这意味着：
- 系统模型是“完备的”；
- 算法层面仍然是“开放的”。

因此，**在不改变框架、不改变优化目标（如 sum-rate / spectral efficiency / fairness）的前提下，引入算法优化是完全正当且具有显著创新空间的**。

---

## 二、可引入算法创新的关键决策环节总览

| 决策环节 | SATCON Baseline 做法 | 算法创新空间 |
|--------|------------------|--------------|
| 模式选择（SAT / ABS-NOMA / ABS-OMA） | 基于 4 条 if-else 规则 | 离散优化 / ILP / DP |
| S2A 资源分配 | 固定 Bs/K 均分 | 连续优化 / Water-filling |
| ABS 位置 | k-means / k-medoids | 连续非凸优化 / 群智能 |
| 功率分配（NOMA） | 闭式公式 + safeguard | 单变量 / 联合优化 |
| 时变调度 | 单时隙 snapshot | MPC / RL / 预测控制 |

---

## 三、核心且最稳妥的算法创新方向（强烈推荐）

### 1️⃣ Pair-wise 模式选择优化（替代启发式规则）

**Baseline 问题**：
- SATCON 使用 4 条规则逐对判断传输模式；
- 无法保证局部或全局最优。

**算法化改进（不改框架、不改目标）**：

对每个卫星用户对 \((i,j)\)，在以下模式中选择最优者：
- 直接卫星传输（SAT）
- ABS-NOMA
- ABS-OMA

优化问题：

\[
\max_{m_k \in \{\text{SAT},\,\text{ABS-NOMA},\,\text{ABS-OMA}\}}
\sum_{l \in k} R_l(m_k)
\]

**可采用算法**：
- 穷举搜索（3 选 1，复杂度极低）
- 贪心 / 动态规划
- 小规模整数线性规划（ILP）

📌 *该改进是“零风险、强对比、最容易发表”的算法创新点。*

---

### 2️⃣ S2A 资源分配优化（消除 DF 瓶颈）

**Baseline 问题**：
- S2A 带宽固定均分 Bs/K；
- 与 ABS 实际服务用户集合无关。

**算法化改进**：

\[
\max_{\{b_k\}} \sum_k \sum_{l\in k} \min(R_l^{A2G},\,R_l^{S2A}(b_k))
\]

\[
\text{s.t. } \sum_k b_k \le B_s, \quad b_k \ge 0
\]

**可采用算法**：
- KKT 条件求解
- Water-filling
- 交替优化 / 凸近似（SCA）

📌 *这是通信系统论文中非常“正统”的优化问题。*

---

## 四、中等难度、但提升明显的算法创新方向

### 3️⃣ ABS 位置的连续优化（替代几何聚类）

**Baseline 问题**：
- k-means / k-medoids 仅基于几何距离；
- 未直接优化通信性能指标。

**算法化目标**：

\[
\max_{\mathbf{p}_d} \sum_{u\in\mathcal{U}} R_u(\mathbf{p}_d)
\]

**可采用算法**：
- L-BFGS-B
- 粒子群优化（PSO）
- 遗传算法（GA）
- 连续凸近似（SCA）

---

### 4️⃣ NOMA 功率分配的联合优化

**Baseline 问题**：
- 功率分配因子由闭式公式固定给出；
- 未与链路条件或系统目标联合优化。

**算法化改进（pair 内）**：

\[
\max_{0<\beta_j<1} \; R_i(\beta_j) + R_j(\beta_j)
\]

**特点**：
- 单变量连续优化；
- 可数值求解或解析近似；
- 改动小但有明确增益。

---

## 五、高阶算法扩展（保持框架，提升智能性）

### 5️⃣ 多时隙调度与预测控制

**Baseline 限制**：
- 单时隙 snapshot 决策；
- 未考虑链路时变性。

**算法扩展方向**：
- 基于 MPC 的多时隙优化
- 基于强化学习（DQN / PPO）的长期收益最大化
- LSTM / 可见性预测 + 贪心调度

📌 *系统结构不变，仅升级决策方式。*

---

## 六、推荐的研究与写作路线

**优先级建议**：

1. Pair-wise 模式选择优化（必做，最稳）  
2. S2A 资源分配优化（系统级提升）  
3. ABS 连续位置优化（空间维度增强）  
4. 功率分配或预测控制（锦上添花）

---

## 七、可直接用于论文的总结性表述

> *While SATCON establishes an effective satellite–aerial hybrid transmission framework, most of its key decisions rely on heuristic rules rather than optimization-based algorithms. This work demonstrates that significant performance gains can be achieved by introducing algorithmic optimization at the decision layer, without altering the underlying system architecture or optimization objectives.*

---

*本文件可作为后续算法设计与论文撰写的统一参考文档。*

