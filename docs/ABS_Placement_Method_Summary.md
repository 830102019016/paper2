# ABS Placement 方法总结（Baseline 与 Enhanced）

## 1. 研究背景与系统定位

在 SATCON 系统中，空中基站（ABS）的引入旨在通过 UAV-assisted transmission 提升地面用户的下行服务质量，并与卫星直传形成互补。  
系统的**最终目标**是：  
**最大化系统整体下行总速率（system sum rate）**，  
该目标通过“决策型混合下行链路（hybrid NOMA/OMA decision）”实现。

由于 ABS 的物理位置会直接决定 A2G 信道特性，ABS placement 被视为一个**先于混合决策的结构性优化步骤**。

---

## 2. Baseline ABS Placement（k-means / k-medoids）

### 2.1 方法描述

SATCON baseline 采用 k-means 与 k-medoids 无监督学习方法，对用户的空间分布进行聚类分析，并选取几何意义下的中心点作为 ABS 的部署位置。

其优化目标定义为最小化用户与 ABS 之间的几何距离平方和：

\[
\arg\min_{p \in W} \sum_{u_i \in U} \|u_i - p\|^2.
\]

该方法主要用于提供一个低复杂度、几何合理的 ABS 初始部署位置。

---

### 2.2 方法特性与局限性

**特性：**
- 仅依赖用户空间坐标；
- 不涉及信道模型；
- 计算复杂度低，易于实现；
- 适合作为基线或初始化策略。

**局限性：**
- 未显式考虑 A2G 路径损耗与 LoS 概率；
- 未直接关联通信速率；
- 与后续混合 NOMA/OMA 决策耦合较弱。

---

## 3. Enhanced ABS Placement（Gradient-based Optimization）

### 3.1 核心思想

增强型 ABS placement 采用**性能驱动（performance-driven）**的优化思想，将 ABS 的三维位置 \((x, y, h)\) 作为连续优化变量，并通过梯度优化方法直接改善 A2G 下行通信性能。

---

### 3.2 优化建模

**优化变量：**
\[
\mathbf{p} = (x, y, h)
\]

**代理优化目标：**
\[
\max_{\mathbf{p}} \sum_{i=1}^{N} B_d \log_2 \left(1 + P_d \, \Gamma_i^d(\mathbf{p}) \right)
\]

其中 \(\Gamma_i^d(\mathbf{p})\) 表示考虑路径损耗、LoS 概率与小尺度衰落后的 A2G 信道增益。

该方法采用 OMA 速率作为代理目标，以保证目标函数的连续性与可优化性。

---

### 3.3 优化算法

- 使用 L-BFGS-B 拟牛顿算法；
- 联合优化 ABS 的平面位置与高度；
- 在给定物理约束下快速收敛。

---

## 4. 与决策型混合下行链路的关系

增强型 ABS placement 并不直接执行 NOMA/OMA 或卫星直传决策，而是通过改善 A2G 信道质量：

- 提高 ABS-assisted transmission 的潜在速率上界；
- 使更多用户满足 UAV-assisted transmission 优于卫星直传的判决条件；
- 扩展混合下行链路决策的可行解空间。

因此，该模块在系统中承担**结构性支撑作用**。

---

## 5. 方法对比总结

| 维度 | Baseline（k-means） | Enhanced（Gradient-based） |
|---|---|---|
| 优化目标 | 几何距离最小化 | A2G 速率最大化（代理） |
| 是否考虑信道 | 否 | 是 |
| 优化变量 | 平面位置 | 三维位置 |
| 方法性质 | 启发式 | 性能驱动 |
| 与混合决策关系 | 间接 | 强耦合（结构性支撑） |

---

## 6. 总结

增强型 ABS placement 将 ABS 部署问题从几何启发式建模提升为通信性能驱动的连续优化问题。通过改善 A2G 信道结构，该方法为后续基于速率的混合 NOMA/OMA 下行链路决策创造了更有利的条件，从而有效提升系统整体性能。
