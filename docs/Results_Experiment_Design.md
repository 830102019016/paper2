# Experimental Design for Joint Transmission Decision Framework

This document summarizes the recommended experimental design for validating the
proposed joint transmission decision framework, with direct reference to the
baseline SATCON (Karavolos et al., 2022) and extensions tailored to highlight
system-level innovations.

---

## 1. Objectives of the Experimental Evaluation

The experiments aim to demonstrate that the proposed framework:
- Improves system-level throughput compared to baseline methods.
- Intelligently activates or deactivates ABS-assisted transmission.
- Converges efficiently under an iterative joint optimization framework.
- Provides performance gains without unnecessary ABS utilization.

---

## 2. Baseline Reference for Experimental Design

The experimental setup follows the structure of:
Karavolos et al., “Satellite–Aerial–Terrestrial Hybrid NOMA Scheme in 6G Networks,” 2022.

Key characteristics adopted from the baseline:
- SNR sweep as the primary independent variable.
- ABS bandwidth and satellite elevation angle as secondary parameters.
- Evaluation of spectral efficiency and sum rate as core metrics.

---

## 3. Simulation Scenario and Parameters

### 3.1 Network Topology
- Satellite: VLEO / LEO satellite with NOMA downlink.
- ABS: Single aerial base station with fixed altitude.
- Coverage area: Circular region (radius ≈ 500 m).
- UE deployment: Uniform random distribution within the coverage area.
- Number of UEs: N = 2K (e.g., N = 32).

### 3.2 Channel Models
- Satellite-to-ground: LMS channel.
- Air-to-ground: A2G channel with distance-dependent path loss.
- Perfect SIC is assumed for NOMA transmissions.

### 3.3 Key Parameters (Example)
- Satellite bandwidth Bs = 5 MHz
- ABS bandwidth Bd ∈ {0.4, 1.2, 2.0, 3.0} MHz
- Satellite SNR range: 0–30 dB
- Satellite elevation angle E ∈ {10°, 20°, 40°}
- ABS transmit power Pd = 30 dBm
- Number of Monte Carlo realizations: ≥ 10^4

---

## 4. Compared Schemes

The following schemes are evaluated:
1. **Satellite-only (SAT-NOMA)**  
   Direct satellite transmission without ABS assistance.
2. **SATCON (rule-based baseline)**  
   ABS-assisted transmission with heuristic, pair-wise decision rules.
3. **Proposed Joint Framework (Ours)**  
   Joint transmission decision with ILP formulation, iterative refinement,
   and intelligent fallback.

(Optional: Greedy variant as a low-complexity comparison.)

---

## 5. Core Performance Metrics

- **Spectral Efficiency (bps/Hz)**
- **System Sum Rate (Mbps)**
- **Transmission Mode Distribution**
- **Convergence Behavior**
- **Ablation Performance Loss**

---

## 6. Main Figures for Results Section

### Fig. 1: Spectral Efficiency vs SNR (Different ABS Bandwidths)
- Fixed satellite elevation angle (e.g., E = 10°).
- Compare all schemes.
- Purpose: Show ABS usefulness under different bandwidth constraints.

### Fig. 2: System Sum Rate vs SNR (Different ABS Bandwidths)
- Same setup as Fig. 1.
- Purpose: Validate throughput gain of the proposed framework.

### Fig. 3: Spectral Efficiency vs SNR (Different Satellite Elevation Angles)
- Fixed ABS bandwidth (e.g., Bd = 1.2 MHz).
- Purpose: Show sensitivity to satellite geometry.

---

## 7. Additional Figures Highlighting Novel Contributions

### Fig. 4: Transmission Mode Distribution vs SNR
- Fraction of UE pairs selecting:
  - ABS-NOMA (both UEs via ABS)
  - ABS-OMA (weak UE only)
  - ABS-OMA (strong UE only)
  - Satellite-only
- Purpose: Visualize intelligent fallback behavior.

### Fig. 5: Convergence of Iterative Framework
- System sum rate vs iteration index.
- Purpose: Demonstrate rapid and stable convergence.

### Fig. 6: Ablation Study
- Fixed representative scenario.
- Compare:
  - Full framework
  - Without iteration
  - Without ILP (heuristic decision)
  - Without fallback mechanism
- Purpose: Quantify contribution of each system component.

---

## 8. Recommended Minimum Experimental Set

To ensure sufficient rigor for SCI journals:
- 1 simulation parameter table
- 3 baseline-aligned figures (Fig. 1–3)
- 3 innovation-focused figures (Fig. 4–6)

---

## 9. Key Takeaway for Results Interpretation

Each figure should answer a single question:
- When is ABS beneficial?
- How much does joint decision improve performance?
- Why is fallback necessary?
- How stable is the iterative framework?

The narrative should consistently emphasize system-level coordination
rather than link-level optimization.

---
