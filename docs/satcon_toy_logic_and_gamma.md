# SATCON（Toy Network）正确逻辑总结 + \Gamma^s / \Gamma^d 计算说明

> 本文档用于**按论文 D. Toy Network 场景**重新梳理 SATCON 的“正确决策逻辑”，并总结 **\(\Gamma^s\)** 与 **\(\Gamma^d\)** 的来源与计算方式（便于复现与写论文）。

---

## 1. Toy Network 场景下的正确端到端逻辑（以论文描述为准）

Toy Network 的关键点是：  
**卫星侧按 \(\Gamma^s\) 形成一套 NOMA 配对；ABS 侧按 \(\Gamma^d\) 形成另一套（可能不同的）NOMA/OMA 配对。**  
ABS 在转发阶段是 **Decode → Re-pair（按自身链路排序）→ Re-encode（重新功率分配）→ 下行** 的逻辑。

下面按“时间/流程”给出完整步骤。

### Step 0：网络与资源设定
- 总用户数 \(N=2K\)（Toy Network 示例：\(N=4, K=2\)）
- 卫星下行总带宽 \(B_s\)，每个卫星配对占用 \(B_s/K\)
- ABS 接入带宽 \(B_d\)，每个 ABS 侧配对占用 \(B_d/K\)
- 用户**始终接收卫星信号**；当 ABS 也发射时，用户在“卫星直达路径”和“ABS 路径”之间进行选择（常用 selection：取更大可达率；Toy 例子里“MT4 仍由 VLEO 服务”即体现该口径）。

---

### Step 1：计算卫星侧信道质量指标 \(\Gamma_i^s\) 并进行卫星 NOMA 配对
1. 对每个用户 \(i\)，计算 \(\Gamma_i^s\)（卫星→用户的等效信道质量指标，见第 2 节）。
2. 将用户按 \(\Gamma_i^s\) 升序排序：
   \[
   \Gamma_1^s \le \Gamma_2^s \le \cdots \le \Gamma_N^s
   \]
3. 采用“最弱–最强”最优配对策略形成 \(K\) 对：
   \[
   \{1,N\},\{2,N-1\},\ldots
   \]
4. 卫星对每一对用户发送 NOMA 叠加信号，并按卫星侧最优功率分配得到每对的 \(\beta^s\)（强用户功率系数）以及卫星直达速率 \(R_i^s,R_j^s\)。

> 这一套配对与 \(\beta^s\) **由卫星决定**，定义了“卫星实际空口发送的 NOMA 叠加信号”。

---

### Step 2：ABS 被动接收卫星下行，并在 S2A 上完成解码（DF 第一跳）
1. ABS 位于卫星覆盖内，能够接收卫星对所有用户广播的下行信号。
2. ABS 对卫星侧每个用户的流计算 S2A 解码速率 \(R_i^{sd}\)（注意带宽应使用 **\(B_s/K\)**，并复用卫星侧 \(\beta^s\)；S2A 常按论文公式忽略小尺度衰落）。
3. 形成 DF 约束：若某用户未来走 ABS 路径，其端到端速率将受 \(R_i^{sd}\) 限制：
   \[
   R_i^{(\cdot)} \le R_i^{sd}
   \]

> 直观理解：ABS 想转发谁，就必须先在 S2A 上“拿到谁的数据”。

---

### Step 3：计算 ABS 侧信道质量指标 \(\Gamma_i^d\) 并在 ABS 侧重新配对
Toy Network 的核心：ABS 侧使用 **自己的链路质量排序**进行配对。

1. 对每个用户 \(i\)，计算 \(\Gamma_i^d\)（ABS→用户 A2G 的等效信道质量指标，见第 2 节）。
2. 将用户按 \(\Gamma_i^d\) 升序排序：
   \[
   \Gamma_1^d \le \Gamma_2^d \le \cdots \le \Gamma_N^d
   \]
3. 采用同样的“最弱–最强”最优配对策略，在 ABS 侧形成 \(K\) 对：
   \[
   \{1,N\},\{2,N-1\},\ldots \quad (\text{基于 } \Gamma^d \text{ 的排序})
   \]
4. 对每个 ABS 配对，ABS 使用其自身链路条件（\(\Gamma^d\)）计算 ABS 侧的功率分配 \(\beta^d\)（注意：这是 **重新编码/重新功率分配**，不是沿用卫星 \(\beta^s\)）。

> 这一套 ABS 配对与 \(\beta^d\) 对应的是 **ABS→用户（A2G）阶段的重编码传输设计**。

---

### Step 4：对每个“ABS 侧配对”执行 hybrid NOMA/OMA 规则（模式选择）
对 ABS 侧每个用户对 \(\{i,j\}\)（基于 \(\Gamma^d\) 配对结果），ABS 做以下评估与选择：

#### 4.1 计算两类 ABS 可达速率（带 DF 约束）
- **ABS-NOMA**（两人都由 ABS 用 NOMA 服务）：
  \[
  R_i^{dn}=\min(R_i^{d,\mathrm{noma}},R_i^{sd}),\quad
  R_j^{dn}=\min(R_j^{d,\mathrm{noma}},R_j^{sd})
  \]
- **ABS-OMA**（只服务其中一人 \(u\in\{i,j\}\)）：
  \[
  R_u^{do}=\min(R_u^{d,\mathrm{oma}},R_u^{sd})
  \]
  另一人 \(v\) 仍走卫星直达。

#### 4.2 与卫星直达速率比较（“是否获益”）
对每个用户 \(l\)，比较：
- 走卫星：\(R_l^s\)
- 走 ABS：\(R_l^{dn}\)（若 ABS-NOMA）或 \(R_l^{do}\)（若 ABS-OMA）

#### 4.3 Toy Network 描述对应的规则结果
- 若 **两人都从 ABS-NOMA 获益**（例如 Toy Network 中 \(\{MT_1,MT_3\}\)）：  
  → 该对采用 **ABS-NOMA**
- 若 **只有一人从 ABS 获益**（例如 Toy Network 中 \(\{MT_2,MT_4\}\) 仅 \(MT_2\) 获益）：  
  → 该对采用 **ABS-OMA**，ABS 仅向获益者发射；另一人由卫星服务
- 若 **两人都不获益**：  
  → ABS 对该对不发射（等价于该对全由卫星服务）

---

### Step 5：用户最终“采用哪条链路”的口径（输出指标）
常见复现口径可写为：
\[
R_l = \max\big(R_l^s,\; R_l^{ABS}\big)
\]
其中 \(R_l^{ABS}\) 取决于该用户在其 ABS 配对上被选中的是 ABS-NOMA/ABS-OMA 还是不服务。

Toy Network 的一句话总结：  
- 卫星先按 \(\Gamma^s\) 配对并广播 NOMA 信号；  
- ABS 在收到并解码后，按 \(\Gamma^d\) **重新配对并重编码**；  
- 对每个 ABS 配对应用 hybrid 规则，可能 NOMA、可能 OMA（只发一人）；  
- 未获益者继续由卫星服务。

---

## 2. \(\Gamma^s\) 与 \(\Gamma^d\) 的计算方式（复现可用版本）

> 二者都是**用于排序/配对的等效信道质量指标**，通常可理解为“归一化信道增益”或“单位功率下的接收 SNR 系数”。

### 2.1 \(\Gamma_i^s\)：卫星→用户（用于卫星配对）
复现常用形式（与论文 \(\text{SNR}\times \Gamma\) 结构一致）：

\[
\Gamma_i^s = \frac{G_t^s\,G_r^{(u)}}{L_{s,i}\;N_{s}}
\cdot |h_{s,i}|^2
\]

- \(G_t^s\)：卫星发射天线增益  
- \(G_r^{(u)}\)：用户接收天线增益  
- \(L_{s,i}\)：卫星到用户的大尺度路径损耗（FSPL + LMS/阴影项等）  
- \(N_s\)：卫星下行噪声功率（常用 \(N_s=\kappa T_s (B_s/K)\)）  
- \(|h_{s,i}|^2\)：小尺度衰落功率（如 LMS/Loo 抽样；或简化为 1）

**用途**：按 \(\Gamma_i^s\) 排序得到卫星 NOMA 配对与卫星侧功率分配 \(\beta^s\)。

---

### 2.2 \(\Gamma_i^d\)：ABS→用户（用于 ABS 重新配对）
复现常用形式：

\[
\Gamma_i^d = \frac{G_t^d\,G_r^{(u)}}{L_{d,i}^{A2G}\;N_{d}}
\cdot |h_{d,i}|^2
\]

- \(G_t^d\)：ABS 发射天线增益  
- \(G_r^{(u)}\)：用户接收天线增益  
- \(L_{d,i}^{A2G}\)：A2G 大尺度路径损耗（依赖仰角与 LoS 概率）  
- \(N_d\)：ABS 下行噪声功率（常用 \(N_d=\kappa T_d (B_d/K)\)）  
- \(|h_{d,i}|^2\)：小尺度衰落功率（常用 Rayleigh：\(|h|^2\sim\exp(1)\)）

**A2G 路损的典型建模（复现常用写法）**：
- 3D 距离：\(d_{3D}=\sqrt{h^2+d_{2D}^2}\)
- 仰角：\(\theta=\arctan(h/d_{2D})\)
- LoS 概率：
  \[
  p_{\mathrm{LoS}}(\theta)=\frac{1}{1+a\exp(-b(\theta-a))}
  \]
- 等效路径损耗（dB）：
  \[
  L_{d,i}^{A2G}(\mathrm{dB}) = L_{FS}(d_{3D}) + p_{\mathrm{LoS}}\eta_{\mathrm{LoS}} + (1-p_{\mathrm{LoS}})\eta_{\mathrm{NLoS}}
  \]
  再从 dB 转线性代入 \(\Gamma_i^d\)。

**用途**：按 \(\Gamma_i^d\) 排序得到 ABS 侧配对与 ABS 侧功率分配 \(\beta^d\)。

---

## 3. 复现实现时的“防走偏要点”（强烈建议照做）

- **两套排序两套配对必须显式区分**：  
  - 卫星：`pairs_s = pair_by(Gamma_s)`  
  - ABS：`pairs_d = pair_by(Gamma_d)`
- **S2A 解码必须基于卫星真实发送的叠加信号**：使用 `pairs_s` 与 \(\beta^s\)，带宽用 \(B_s/K\)
- **A2G 下行必须基于 ABS 重编码的配对与功分**：使用 `pairs_d` 与 \(\beta^d\)，带宽用 \(B_d/K\)

---

*（完）*
