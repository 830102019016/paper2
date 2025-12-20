# 路线 A：基于 ABS 配对的模式选择（SATCON 忠实实现）

## 1. 背景与问题动机

在 SATCON 框架中，**卫星（SAT）与空中基站（ABS）分别基于不同信道条件进行用户排序与配对**：

- 卫星侧：根据卫星下行信道增益 \(\Gamma^s\) 进行 NOMA 配对（SAT-pairs）；
- ABS 侧：根据空地（A2G）信道增益 \(\Gamma^d\) 重新进行 NOMA 配对（ABS-pairs）。

因此，在一般情况下，SAT 配对与 ABS 配对**并不一致**。论文中的 toy example 已明确展示了这一点。

然而，当前实现中的 `GreedySelector` 是**基于 SAT-pairs 进行模式选择的**，这在 SAT-pairs ≠ ABS-pairs 的情况下会导致物理与逻辑不一致的问题。

---

## 2. 当前 GreedySelector 为何会被“击穿”

### 2.1 当前实现的隐含假设

现有 `GreedySelector` 的核心假设是：

> **ABS 的 NOMA/OMA 速率可以直接在 SAT-pairs 上进行比较和决策。**

即，它默认：

```
SAT-pairs == ABS-pairs
```

在该假设下，对每个 pair 比较四种模式（sat / noma / oma_weak / oma_strong）是自洽的。

### 2.2 反例：SAT 配对与 ABS 配对不一致

设有四个用户 \(i,j,k,l\)：

- SAT 配对：\((i,j), (k,l)\)
- ABS 配对：\((i,k), (j,l)\)

此时，如果在 SAT-pair \((i,j)\) 上比较：

- `abs_noma_rates[i] + abs_noma_rates[j]`

那么该“ABS-NOMA 收益”在物理上是**不可实现的**，因为：

- ABS 的 NOMA 发射对象并不是 \((i,j)\)，而是 \((i,k)\) 或 \((j,l)\)；
- 功率分配、SIC 顺序、可解码速率均基于 ABS-pairs 定义。

因此，**当前 GreedySelector 在 SAT-pairs ≠ ABS-pairs 时，会对一个不存在的物理动作进行优化**，从而被该反例“击穿”。

---

## 3. 路线 A 的核心思想（忠实 SATCON）

### 3.1 基本原则

**模式选择应以 ABS 实际形成的用户对（ABS-pairs）为决策单位。**

原因在于：

- NOMA / OMA 的选择发生在 ABS；
- ABS 的发射资源（时隙 / 频带 / 功率）是按 ABS-pairs 分配的；
- ABS 是否转发、如何转发，只对其服务的 ABS-pair 有物理意义。

SAT-pairs 的作用应当仅限于：

- 决定卫星 NOMA 下行速率（作为所有用户的 baseline）。

### 3.2 路线 A 的结构性修改

在路线 A 中，系统流程应调整为：

1. **卫星侧**：
   - 基于 \(\Gamma^s\) 形成 SAT-pairs；
   - 计算每个用户的卫星直达速率 \(R^s_l\)。

2. **ABS 侧**：
   - 基于 \(\Gamma^d\) 形成 ABS-pairs；
   - 基于 ABS-pairs 计算：
     - ABS-NOMA 速率 \(R^{dn}_l\)（含 S2A 约束）；
     - ABS-OMA 速率 \(R^{do}_l\)（含 S2A 约束）。

3. **模式选择（关键修改）**：
   - 遍历 **ABS-pairs**；
   - 对每个 ABS-pair \((u,v)\)，比较以下四种**物理可行**模式：
     - `sat`：\(R^s_u + R^s_v\)
     - `noma`：\(R^{dn}_u + R^{dn}_v\)
     - `oma_u`：\(R^{do}_u + R^s_v\)
     - `oma_v`：\(R^s_u + R^{do}_v\)
   - 选择最优模式。

---

## 4. 为什么路线 A 是“合理且必要的”

### 4.1 物理一致性

- 每一次 NOMA/OMA 发射都严格对应一个 ABS-pair；
- 不再对“未实际配对的用户组合”计算虚假的 NOMA 收益。

### 4.2 与 SATCON 论文完全一致

- SATCON 原文中，ABS 的混合 NOMA/OMA 决策正是针对 **ABS 形成的用户对**；
- SAT-pairs 与 ABS-pairs 不一致是论文明确允许的情形。

### 4.3 保留算法层创新空间

- 路线 A 仅修正“决策维度错误”的问题；
- 在此基础上仍可引入：
  - Greedy / Exhaustive / Learning-based 选择器；
  - sum-rate、fairness、energy-aware 等目标。

---

## 5. 小结

- 当前 GreedySelector 被“击穿”的根本原因在于：**在 SAT-pairs 上对 ABS 发射模式进行优化**；
- 路线 A 通过将模式选择严格绑定到 ABS-pairs，恢复了物理与协议层的一致性；
- 该修改不是引入新模型，而是对 SATCON 原始机制的忠实实现；
- 路线 A 是进行复现、对比与算法增强时最稳妥、最容易说服审稿人的选择。

