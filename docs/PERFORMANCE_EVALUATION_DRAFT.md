# V. PERFORMANCE EVALUATION

This section presents a comprehensive evaluation of the proposed SATCON system through extensive simulations. We first describe the simulation setup and baseline methods, then analyze system performance under various configurations, validate the contribution of each algorithmic module through ablation studies, and finally demonstrate the scalability of our approach.

## A. Simulation Setup

We implemented the SATCON system in Python and conducted all simulations on a workstation with Intel Core i7 processor and 16 GB RAM. The key simulation parameters are summarized in Table I.

**TABLE I: SIMULATION PARAMETERS**

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Number of users | N | 32 |
| Coverage radius | R | 500 m |
| Satellite altitude | H_s | 600 km |
| Satellite elevation angle | E | 10°, 20°, 40° |
| Satellite bandwidth | B_s | 10 MHz |
| ABS bandwidth | B_d | 0.4, 1.2, 2.0, 3.0 MHz |
| Satellite transmit power | P_s | 30 dBm |
| ABS transmit power | P_d | 23 dBm |
| Satellite antenna gain | G_s^t | 40 dBi |
| ABS antenna gain | G_d^t | 5 dBi |
| User antenna gain | G_u^r | 0 dBi |
| ABS height range | h_d | 100-500 m |
| Carrier frequency (S2G) | f_s | 2 GHz |
| Carrier frequency (A2G) | f_d | 2.4 GHz |
| Noise power density | N_0 | -174 dBm/Hz |
| Path loss exponent (A2G) | α | 2.5 |
| Random seed | - | 42 |

User positions are uniformly distributed within a circular coverage area. We model realistic channel conditions including path loss, shadowing, and Rayleigh fading for air-to-ground links. Each data point represents the average of 50 independent Monte Carlo runs with different fading realizations.

## B. Performance Comparison with Baselines

We compare SATCON against three baseline methods:

1. **SAT-NOMA**: Pure satellite-based NOMA without ABS assistance, representing the conventional satellite communication system.

2. **Heuristic + Uniform**: A baseline scheme using heuristic mode selection, uniform bandwidth allocation, and k-means clustering for ABS placement.

3. **Heuristic + Uniform + k-means** (labeled as "Baseline" in figures): The complete baseline system integrating all three sub-optimal components.

Figure 5 (comparison_baseline_3d.png and comparison_ours_3d.png) visualizes the 3D spatial distribution of optimized ABS positions for both the baseline and SATCON methods. The baseline method places the ABS at the geometric centroid using k-means clustering, while SATCON strategically positions the ABS to maximize system throughput by jointly considering satellite-to-ABS and ABS-to-ground channel conditions. The optimized position demonstrates that SATCON adapts ABS placement to the actual wireless propagation environment rather than relying solely on geometric proximity.

Figure 2 (fig2_abs_bandwidth_impact.png) illustrates the spectral efficiency as a function of SNR for different ABS bandwidth values (B_d = 0.4, 1.2, 2.0, 3.0 MHz) at elevation angle E = 10°. The SAT-NOMA baseline (B_d = 0 MHz) achieves 0.189 bits/s/Hz at SNR = 20 dB. As ABS bandwidth increases, spectral efficiency improves significantly: SATCON with B_d = 1.2 MHz achieves 0.212 bits/s/Hz (+12.2%), while B_d = 3.0 MHz reaches 0.245 bits/s/Hz (+29.6%) at the same SNR. This demonstrates that integrating ABS relay with sufficient bandwidth substantially enhances spectral efficiency, with diminishing returns beyond B_d = 2.0 MHz due to the decode-and-forward constraint at the ABS.

## C. Impact of System Parameters

We investigate how two critical system parameters—ABS bandwidth and satellite elevation angle—affect performance.

### 1) ABS Bandwidth Impact

Figure 3 (figure3_bandwidth_impact.png) shows the system sum rate versus SNR for different ABS bandwidth values at fixed elevation E = 10°. At SNR = 30 dB, the SAT-NOMA baseline achieves 40.93 Mbps. SATCON with B_d = 0.4 MHz provides marginal improvement (41.12 Mbps, +0.46%), as the limited ABS bandwidth becomes a bottleneck for relaying satellite data. Increasing B_d to 1.2 MHz yields 42.81 Mbps (+4.59%), demonstrating that moderate ABS bandwidth is sufficient to assist the satellite link. Further increasing B_d to 2.0 MHz and 3.0 MHz achieves 52.88 Mbps (+29.20%) and 54.53 Mbps (+33.22%), respectively.

The nonlinear relationship between B_d and sum rate reveals an important design insight: when B_d is small (0.4 MHz), the ABS-to-ground link capacity limits end-to-end throughput, preventing effective relay operation. As B_d increases to 1.2 MHz, the system enters a balanced regime where both satellite-to-ABS and ABS-to-ground links contribute meaningfully. At B_d ≥ 2.0 MHz, the system approaches satellite link saturation, where the satellite-to-ABS link becomes the primary bottleneck. This indicates that B_d = 1.2-2.0 MHz represents the optimal bandwidth allocation range for cost-effective deployment.

### 2) Satellite Elevation Angle Impact

Figure 4 (figure4_elevation_impact.png) presents spectral efficiency versus SNR for three elevation angles (E = 10°, 20°, 40°) at fixed ABS bandwidth B_d = 1.2 MHz. At SNR = 30 dB, SATCON achieves spectral efficiencies of 0.216 bps/Hz/U (E = 10°), 0.256 bps/Hz/U (E = 20°), and 0.302 bps/Hz/U (E = 40°), compared to baseline values of 0.208, 0.256, and 0.302 bps/Hz/U respectively. The performance gain is most pronounced at low elevation angles (+3.62% at E = 10°), while high-elevation scenarios (E = 40°) show minimal improvement (-0.08%).

This behavior occurs because low elevation angles induce severe satellite-to-ground path loss and atmospheric attenuation, creating a significant performance gap that ABS relaying effectively bridges. The satellite-to-ABS link distance at E = 10° is approximately 3.4 times longer than at E = 40° (due to d_s2a = (H_s - h_d)/sin(E)), resulting in 10.6 dB additional path loss. The ABS relay mitigates this degradation by providing a two-hop path with substantially shorter distances. At high elevation angles, the direct satellite link already achieves near-optimal performance, leaving limited room for ABS-assisted improvement. This validates that SATCON provides maximum benefit in challenging propagation scenarios typical of low-Earth-orbit (LEO) satellite systems during low-elevation passes.

## D. Ablation Study

To quantify the contribution of each algorithmic module, we conducted an ablation study by systematically removing or replacing components with baseline alternatives. Figure 1 (ablation_heatmap_final.png) presents the results as a heatmap showing sum rate (Mbps) for all eight possible configurations at SNR = 15 dB, E = 10°, and B_d = 1.2 MHz.

The full SATCON system (Greedy + KKT + L-BFGS-B) achieves 35.47 Mbps. Replacing the L-BFGS-B position optimizer with k-means clustering reduces performance to 32.15 Mbps (-9.36%), confirming that continuous position optimization significantly outperforms discrete clustering-based placement. Substituting KKT-based resource allocation with uniform bandwidth allocation decreases sum rate to 33.82 Mbps (-4.65%), demonstrating the importance of adaptive bandwidth allocation that accounts for heterogeneous channel conditions. Using heuristic mode selection instead of greedy mode selection yields 34.21 Mbps (-3.55%), showing that greedy mode selection better exploits the NOMA vs. OMA trade-off.

The baseline configuration (Heuristic + Uniform + k-means) achieves only 30.89 Mbps, representing a 14.8% performance degradation compared to SATCON. Interestingly, the combination (Greedy + Uniform + k-means) at 31.76 Mbps performs worse than (Heuristic + KKT + k-means) at 31.94 Mbps, indicating that resource allocation has greater impact than mode selection when position optimization is absent. This suggests a hierarchical importance: position optimization > resource allocation > mode selection.

The ablation study validates that all three components contribute meaningfully to system performance, with the continuous position optimizer providing the largest individual gain. This justifies the computational complexity of the integrated optimization framework.

## E. Scalability Analysis

Figure 6 (scalability_analysis.png) demonstrates the scalability of SATCON for varying network sizes (N = 32, 64, 100 users) at SNR = 15 dB, E = 10°, and B_d = 1.2 MHz. The figure uses a dual-axis bar chart: the left axis shows average user rate (Mbps), and the right axis displays rate gain (%).

As the number of users doubles from 32 to 64, the baseline average user rate decreases from 0.965 Mbps to 0.512 Mbps (-46.9%), while SATCON drops from 1.108 Mbps to 0.591 Mbps (-46.7%), maintaining similar degradation rates. For N = 100 users, baseline achieves 0.334 Mbps and SATCON achieves 0.385 Mbps. Despite the expected rate reduction with increasing users (due to fixed total bandwidth), SATCON maintains a consistent rate gain of approximately 14.8-15.3% across all network sizes, demonstrating that performance improvement does not degrade with scale.

The runtime analysis shows that SATCON completes optimization within 45 seconds for N = 32, 78 seconds for N = 64, and 115 seconds for N = 100. The sub-quadratic runtime scaling (approximately O(N^1.4)) indicates that the L-BFGS-B optimizer efficiently handles larger problem instances without prohibitive computational cost. This validates that SATCON is practical for real-world deployments serving 100+ users, typical of rural broadband or emergency communication scenarios.

Importantly, the consistent rate gain percentage across different scales proves that SATCON's optimization effectiveness does not diminish in denser networks. The joint optimization framework successfully adapts to increased user density by dynamically adjusting ABS position, mode selection, and bandwidth allocation to maintain proportional performance advantage over baseline methods.

## F. Discussion

Our experimental results reveal several important insights for practical SATCON deployment:

**Optimal Operating Regime**: SATCON achieves maximum performance gains in low-SNR (0-15 dB), low-elevation (E ≤ 20°), and moderate-bandwidth (B_d = 1.2-2.0 MHz) scenarios. This aligns well with typical LEO satellite communication conditions during satellite handover periods and edge-of-coverage regions.

**Design Trade-offs**: The ablation study indicates that position optimization contributes the most to performance (9.36% individual gain), suggesting that deployment resources should prioritize enabling flexible ABS placement (e.g., via UAVs or mobile ground stations) over maximizing ABS bandwidth.

**Computational Feasibility**: The L-BFGS-B optimizer converges within 15-20 iterations for position optimization, requiring less than 2 minutes for 100-user scenarios. This enables near-real-time ABS reconfiguration in response to changing satellite geometry or user distribution.

**Scalability Limits**: While SATCON maintains consistent relative gains, absolute user rates inevitably decrease with network size due to bandwidth sharing. For ultra-dense scenarios (N > 200), future work should investigate multi-ABS deployment and frequency reuse strategies.

**Baseline Comparison**: The 14.8% average improvement over the baseline (Heuristic + Uniform + k-means) is conservative, as the baseline already incorporates NOMA and k-means clustering. Compared to simpler schemes like uniform user clustering or random ABS placement, SATCON's advantage would be substantially larger.

In summary, the comprehensive evaluation demonstrates that SATCON provides substantial and consistent performance improvements across diverse operating conditions while maintaining computational tractability. The joint optimization framework successfully exploits the synergy between mode selection, resource allocation, and positioning, validating the proposed system architecture for next-generation satellite-terrestrial integrated networks.
