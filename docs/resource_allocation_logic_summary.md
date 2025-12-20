# Resource Allocation Logic and System Workflow Summary

This document summarizes the key design logic, assumptions, and workflow of the proposed
Greedy–KKT–L-BFGS-B framework, with a focus on S2A resource allocation, DF bottleneck handling,
and system-level optimization consistency.

It serves as a reference for manuscript writing, reviewer response, and future algorithm upgrades.

---

## 1. Relationship Between Greedy Mode Selection and A2G Rates

### 1.1 What is determined by the GreedySelector?

After the Greedy mode selector is executed:

- The transmission mode of **each ABS-formed user pair** is fully determined:
  - `sat`, `noma`, `oma_weak`, or `oma_strong`
- For each user that is scheduled to be served via the ABS, the system already knows:
  - Its **A2G achievable rate without considering the S2A constraint**

These rates are explicitly computed as:
- `a2g_rates_noma[u]` for NOMA-based A2G transmission
- `a2g_rates_oma[u]` for OMA-based A2G transmission

### 1.2 Interpretation

The GreedySelector operates at the **decision layer** and determines:
> How fast each user *would like* to be served over the A2G link, assuming the backhaul is not limiting.

Formally, it determines the target access-link rates:
\[
R_u^{\mathrm{A2G}}, \quad \forall u \in \mathcal{U}_{\mathrm{ABS}}
\]

These rates are **not** yet guaranteed to be feasible end-to-end.

---

## 2. Role of KKT-Based S2A Bandwidth Allocation

### 2.1 Purpose of S2A resource allocation

The S2A bandwidth allocation does **not** aim to improve the A2G capacity.
Instead, its goal is to:

> Eliminate the DF bottleneck so that the end-to-end rate can truly reach the A2G achievable rate.

For DF relaying, the end-to-end rate is:
\[
R_u^{\mathrm{E2E}} = \min \left( R_u^{\mathrm{A2G}},\; R_u^{\mathrm{S2A}}(b_k) \right)
\]

The KKT allocator chooses the minimum bandwidth \( b_k \) such that:
\[
R_u^{\mathrm{S2A}}(b_k) \ge R_u^{\mathrm{A2G}}
\]

### 2.2 Input A2G rates used in KKT

The correct input to the KKT allocator is:

- The **A2G achievable rate under the selected mode**
- **Before** applying any S2A constraint

Specifically:
- `noma` mode → `a2g_rates_noma[weak]` and `a2g_rates_noma[strong]`
- `oma_weak` mode → `a2g_rates_oma[weak]`
- `oma_strong` mode → `a2g_rates_oma[strong]`
- `sat` mode → no S2A bandwidth required (`b_k = 0`)

Using final end-to-end rates or S2A-limited rates as inputs would result in circular logic.

---

## 3. Proportional Scaling When Total S2A Demand Exceeds \( B_s \)

### 3.1 Current design choice

If the total required S2A bandwidth exceeds the satellite budget:
\[
\sum_{k} b_k^\star > B_s
\]

The system applies **proportional scaling**:
\[
b_k \leftarrow b_k^\star \cdot \frac{B_s}{\sum_k b_k^\star}
\]

### 3.2 Is there backtracking to Greedy mode selection?

**No.**  
The current system does **not** re-run or revise the Greedy mode selection after scaling.

This is a **deliberate design choice**, not a flaw.

### 3.3 Rationale

- Backtracking would require:
  - Reversing discrete transmission-mode decisions
  - Solving a coupled mixed-integer optimization problem
- This would significantly increase complexity and destabilize convergence

Instead, the framework adopts:
> Continuous feasibility enforcement (via scaling) rather than discrete structural re-optimization.

This ensures:
- Algorithmic stability
- Polynomial-time complexity
- Clear separation of decision and constraint layers

### 3.4 How to defend this in a paper or review response

A standard justification is:

> When the total S2A bandwidth demand exceeds the satellite budget, a proportional scaling is applied to guarantee feasibility. Although this may lead to a degradation from the ideal rates predicted by the greedy mode selection, the transmission structure remains unchanged, which avoids combinatorial re-optimization and ensures stable convergence of the overall framework.

---

## 4. Overall System Workflow and Optimization Order

### 4.1 Nested optimization structure

The proposed framework follows a **nested optimization hierarchy**:

#### Outer loop (continuous optimization)
- Variable: ABS 3D position \( (x, y, h) \)
- Algorithm: L-BFGS-B
- Objective: Maximize system sum rate

#### Inner procedures (re-evaluated at each position)
1. Channel gain computation (A2G and S2A)
2. User pairing:
   - SAT-pairs (based on satellite channel gains)
   - ABS-pairs (based on A2G channel gains)
3. Greedy mode selection (ABS-side)
4. KKT-based S2A bandwidth allocation
5. End-to-end rate computation

### 4.2 One iteration of the outer loop

For a given ABS position \( \mathbf{p}_d \):

1. Compute channel gains \( \Gamma^{d}(\mathbf{p}_d) \), \( h_{s2a}(\mathbf{p}_d) \)
2. Form SAT-pairs and ABS-pairs
3. Run GreedySelector → determine modes and \( R^{\mathrm{A2G}} \)
4. Run KKTAllocator → determine feasible \( b_k \)
5. Compute:
   \[
   R_u^{\mathrm{E2E}} = \min(R_u^{\mathrm{A2G}}, R_u^{\mathrm{S2A}})
   \]
6. Sum over all users → system throughput
7. Return throughput as the objective value to L-BFGS-B

### 4.3 Convergence behavior

- Greedy and KKT are **not iterative by themselves**
- They are re-executed whenever the ABS position changes
- The outer L-BFGS-B loop drives the system toward a locally optimal configuration

---

## 5. Key Takeaways (Design-Level Summary)

- Greedy mode selection determines the **desired A2G rates**
- KKT allocation ensures **S2A feasibility**, not A2G enhancement
- Proportional scaling guarantees feasibility without structural backtracking
- ABS position optimization is the only iterative component
- The framework is:
  - Logically self-consistent
  - Computationally tractable
  - Defendable in system-level research

---

## 6. Possible Future Extensions (Beyond Current Scope)

- Bandwidth-aware Greedy mode selection
- Iterative mode refinement under S2A constraints
- Priority- or fairness-aware scaling strategies

These extensions build upon the current framework but are intentionally left out to maintain algorithmic simplicity and stability.
