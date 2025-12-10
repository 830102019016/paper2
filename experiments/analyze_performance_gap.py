# -*- coding: utf-8 -*-
"""
Performance Gap Analysis

Analyze why current performance gain is only 3-4% instead of target 40%

Investigation areas:
1. Current scenario characteristics
2. Module effectiveness in different conditions
3. Parameter sensitivity analysis
4. Bottleneck identification

Author: SATCON Enhancement Project
Date: 2025-12-10
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.user_distribution import UserDistribution
from src.a2g_channel import A2GChannel
from src.noma_transmission import SatelliteNOMA
from src_enhanced.joint_satcon_system import JointOptimizationSATCON


class PerformanceAnalyzer:
    """Analyze performance gap and identify optimization opportunities"""

    def __init__(self):
        self.config = config
        self.results = {}

    def analyze_baseline_bottleneck(self):
        """
        Analyze where baseline system is bottlenecked

        Key questions:
        - Is ABS placement suboptimal?
        - Is user pairing inefficient?
        - Is hybrid decision leaving opportunities?
        """
        print("=" * 80)
        print("ANALYSIS 1: Baseline Bottleneck Identification")
        print("=" * 80)

        Bd = 1.2e6
        snr_db = 20
        elevation = 10
        n_test = 20

        # Create systems
        baseline = JointOptimizationSATCON(
            config, Bd,
            use_module1=False, use_module2=False, use_module3=False
        )

        print("\nRunning baseline system analysis...")
        print(f"Testing {n_test} random user distributions\n")

        position_quality = []
        pairing_quality = []
        decision_quality = []

        for seed in range(n_test):
            # Generate user distribution
            dist = UserDistribution(config.N, config.coverage_radius, seed=seed)
            user_positions = dist.generate_uniform_circle()

            # Analyze position quality
            # Compare k-means position vs user centroid
            centroid = np.mean(user_positions[:, :2], axis=0)
            from src.abs_placement import ABSPlacement
            abs_placement = ABSPlacement()
            abs_xy, _, _ = abs_placement.optimize_xy_position(user_positions)
            position_deviation = np.linalg.norm(abs_xy - centroid)
            position_quality.append(position_deviation)

            # Analyze pairing quality
            # Check channel gain variance within pairs
            sat_noma = SatelliteNOMA(config)
            sat_gains = sat_noma.compute_channel_gains_with_pathloss(elevation)

            from src.power_allocation import NOMAAllocator
            allocator = NOMAAllocator()
            sat_pairs, _ = allocator.optimal_user_pairing(sat_gains)

            pair_variances = []
            for weak_idx, strong_idx in sat_pairs:
                variance = np.abs(sat_gains[weak_idx] - sat_gains[strong_idx])
                pair_variances.append(variance)

            pairing_quality.append(np.mean(pair_variances))

        # Print statistics
        print(f"\nPosition Quality (deviation from centroid):")
        print(f"  Mean: {np.mean(position_quality):.2f} m")
        print(f"  Std:  {np.std(position_quality):.2f} m")
        print(f"  Analysis: {'Good' if np.mean(position_quality) < 50 else 'Could improve'}")

        print(f"\nPairing Quality (avg channel gain difference):")
        print(f"  Mean: {np.mean(pairing_quality):.6f}")
        print(f"  Std:  {np.std(pairing_quality):.6f}")
        print(f"  Analysis: {'Well paired' if np.std(pairing_quality) < 0.001 else 'High variance'}")

        return {
            'position_quality': np.mean(position_quality),
            'pairing_quality': np.mean(pairing_quality)
        }

    def analyze_scenario_characteristics(self):
        """
        Analyze current scenario and identify challenging conditions

        Hypothesis: Current scenario might be too "easy" for optimization
        """
        print("\n" + "=" * 80)
        print("ANALYSIS 2: Scenario Characteristics")
        print("=" * 80)

        Bd = 1.2e6
        snr_db = 20
        elevation = 10
        n_test = 50

        print(f"\nAnalyzing {n_test} user distributions...")

        # Metrics to analyze
        user_spreads = []
        channel_variances = []
        abs_sat_ratios = []

        a2g_channel = A2GChannel()
        sat_noma = SatelliteNOMA(config)

        for seed in range(n_test):
            # User distribution
            dist = UserDistribution(config.N, config.coverage_radius, seed=seed)
            user_positions = dist.generate_uniform_circle()

            # User spread (how scattered are users?)
            centroid = np.mean(user_positions[:, :2], axis=0)
            distances = [np.linalg.norm(u[:2] - centroid) for u in user_positions]
            user_spreads.append(np.std(distances))

            # Channel variance (how diverse are channel conditions?)
            sat_gains = sat_noma.compute_channel_gains_with_pathloss(elevation)
            channel_variances.append(np.std(sat_gains) / np.mean(sat_gains))

            # ABS vs Satellite potential (how much better can ABS be?)
            snr_linear = 10 ** (snr_db / 10)
            sat_rates, _ = sat_noma.compute_achievable_rates(sat_gains, snr_linear)

            # Estimate ABS rates
            abs_pos = [0, 0, 100]  # Simplified
            distances_2d = [np.linalg.norm(u[:2] - abs_pos[:2]) for u in user_positions]
            fading = a2g_channel.generate_fading(config.N, seed=seed)

            Nd = config.get_abs_noise_power(Bd)
            a2g_gains = [a2g_channel.compute_channel_gain(
                abs_pos[2], d, f, config.Gd_t_dB, config.Gd_r_dB, Nd
            ) for d, f in zip(distances_2d, fading)]

            abs_rates_est = [Bd * np.log2(1 + config.Pd * g) for g in a2g_gains]

            ratio = np.mean(abs_rates_est) / np.mean(sat_rates)
            abs_sat_ratios.append(ratio)

        # Print analysis
        print(f"\nUser Spread (std of distances from centroid):")
        print(f"  Mean: {np.mean(user_spreads):.2f} m")
        print(f"  Range: [{np.min(user_spreads):.2f}, {np.max(user_spreads):.2f}] m")
        print(f"  Analysis: {'Low diversity' if np.mean(user_spreads) < 100 else 'High diversity'}")

        print(f"\nChannel Gain Variance (CoV):")
        print(f"  Mean: {np.mean(channel_variances):.3f}")
        print(f"  Range: [{np.min(channel_variances):.3f}, {np.max(channel_variances):.3f}]")
        print(f"  Analysis: {'Homogeneous' if np.mean(channel_variances) < 0.5 else 'Heterogeneous'}")

        print(f"\nABS/Satellite Rate Ratio:")
        print(f"  Mean: {np.mean(abs_sat_ratios):.2f}x")
        print(f"  Range: [{np.min(abs_sat_ratios):.2f}x, {np.max(abs_sat_ratios):.2f}x]")
        print(f"  Analysis: {'Limited ABS benefit' if np.mean(abs_sat_ratios) < 1.5 else 'Strong ABS benefit'}")

        return {
            'user_spread': np.mean(user_spreads),
            'channel_variance': np.mean(channel_variances),
            'abs_sat_ratio': np.mean(abs_sat_ratios)
        }

    def test_parameter_sensitivity(self):
        """
        Test how different parameters affect optimization gain

        Parameters to vary:
        - User number (N)
        - Coverage radius (R)
        - ABS bandwidth (Bd)
        - Satellite elevation (E)
        - SNR
        """
        print("\n" + "=" * 80)
        print("ANALYSIS 3: Parameter Sensitivity")
        print("=" * 80)

        print("\nTesting different parameter combinations...")
        print("(This will take a few minutes)\n")

        results = {}

        # Test 1: Different user numbers
        print("Test 1: User Number Impact")
        user_numbers = [16, 32, 64]
        for N in user_numbers:
            config_temp = config
            config_temp.N = N

            baseline = JointOptimizationSATCON(
                config_temp, 1.2e6,
                use_module1=False, use_module2=False, use_module3=False
            )
            full = JointOptimizationSATCON(
                config_temp, 1.2e6,
                use_module1=True, use_module2=True, use_module3=True
            )

            # Quick test
            rates_b = []
            rates_f = []
            for seed in range(5):
                rate_b, _ = baseline.simulate_single_realization(20, 10, seed)
                rate_f, _ = full.simulate_single_realization(20, 10, seed)
                rates_b.append(rate_b)
                rates_f.append(rate_f)

            gain = (np.mean(rates_f) - np.mean(rates_b)) / np.mean(rates_b) * 100
            print(f"  N={N:2d}: Gain = {gain:+5.2f}%")
            results[f'N={N}'] = gain

        config_temp.N = config.N  # Reset

        # Test 2: Different coverage radii
        print("\nTest 2: Coverage Radius Impact")
        radii = [300, 500, 800]
        for R in radii:
            config_temp = config
            config_temp.coverage_radius = R

            baseline = JointOptimizationSATCON(
                config_temp, 1.2e6,
                use_module1=False, use_module2=False, use_module3=False
            )
            full = JointOptimizationSATCON(
                config_temp, 1.2e6,
                use_module1=True, use_module2=True, use_module3=True
            )

            rates_b = []
            rates_f = []
            for seed in range(5):
                rate_b, _ = baseline.simulate_single_realization(20, 10, seed)
                rate_f, _ = full.simulate_single_realization(20, 10, seed)
                rates_b.append(rate_b)
                rates_f.append(rate_f)

            gain = (np.mean(rates_f) - np.mean(rates_b)) / np.mean(rates_b) * 100
            print(f"  R={R:3d}m: Gain = {gain:+5.2f}%")
            results[f'R={R}m'] = gain

        config_temp.coverage_radius = config.coverage_radius  # Reset

        # Test 3: Different bandwidths
        print("\nTest 3: ABS Bandwidth Impact")
        bandwidths = [0.4e6, 1.2e6, 3.0e6]
        for Bd in bandwidths:
            baseline = JointOptimizationSATCON(
                config, Bd,
                use_module1=False, use_module2=False, use_module3=False
            )
            full = JointOptimizationSATCON(
                config, Bd,
                use_module1=True, use_module2=True, use_module3=True
            )

            rates_b = []
            rates_f = []
            for seed in range(5):
                rate_b, _ = baseline.simulate_single_realization(20, 10, seed)
                rate_f, _ = full.simulate_single_realization(20, 10, seed)
                rates_b.append(rate_b)
                rates_f.append(rate_f)

            gain = (np.mean(rates_f) - np.mean(rates_b)) / np.mean(rates_b) * 100
            print(f"  Bd={Bd/1e6:.1f}MHz: Gain = {gain:+5.2f}%")
            results[f'Bd={Bd/1e6:.1f}MHz'] = gain

        # Test 4: Different elevations
        print("\nTest 4: Satellite Elevation Impact")
        elevations = [10, 30, 60]
        for E in elevations:
            baseline = JointOptimizationSATCON(
                config, 1.2e6,
                use_module1=False, use_module2=False, use_module3=False
            )
            full = JointOptimizationSATCON(
                config, 1.2e6,
                use_module1=True, use_module2=True, use_module3=True
            )

            rates_b = []
            rates_f = []
            for seed in range(5):
                rate_b, _ = baseline.simulate_single_realization(20, E, seed)
                rate_f, _ = full.simulate_single_realization(20, E, seed)
                rates_b.append(rate_b)
                rates_f.append(rate_f)

            gain = (np.mean(rates_f) - np.mean(rates_b)) / np.mean(rates_b) * 100
            print(f"  E={E:2d}deg: Gain = {gain:+5.2f}%")
            results[f'E={E}deg'] = gain

        return results

    def identify_optimal_scenarios(self):
        """
        Identify specific scenarios where optimization gains are largest

        Strategy: Test various challenging conditions
        """
        print("\n" + "=" * 80)
        print("ANALYSIS 4: Optimal Scenario Identification")
        print("=" * 80)

        print("\nSearching for high-gain scenarios...\n")

        scenarios = [
            {
                'name': 'Baseline (Current)',
                'N': 32, 'R': 500, 'Bd': 1.2e6, 'E': 10, 'SNR': 20
            },
            {
                'name': 'Large Coverage',
                'N': 32, 'R': 800, 'Bd': 1.2e6, 'E': 10, 'SNR': 20
            },
            {
                'name': 'Many Users',
                'N': 64, 'R': 500, 'Bd': 1.2e6, 'E': 10, 'SNR': 20
            },
            {
                'name': 'Low Elevation',
                'N': 32, 'R': 500, 'Bd': 1.2e6, 'E': 10, 'SNR': 20
            },
            {
                'name': 'High Bandwidth',
                'N': 32, 'R': 500, 'Bd': 3.0e6, 'E': 10, 'SNR': 20
            },
            {
                'name': 'Low SNR',
                'N': 32, 'R': 500, 'Bd': 1.2e6, 'E': 10, 'SNR': 10
            },
            {
                'name': 'Challenging (Combined)',
                'N': 64, 'R': 800, 'Bd': 3.0e6, 'E': 10, 'SNR': 15
            }
        ]

        results = []

        for scenario in scenarios:
            # Setup config
            config_temp = config
            config_temp.N = scenario['N']
            config_temp.coverage_radius = scenario['R']

            # Create systems
            baseline = JointOptimizationSATCON(
                config_temp, scenario['Bd'],
                use_module1=False, use_module2=False, use_module3=False
            )
            full = JointOptimizationSATCON(
                config_temp, scenario['Bd'],
                use_module1=True, use_module2=True, use_module3=True
            )

            # Test (fewer realizations for speed)
            rates_b = []
            rates_f = []
            for seed in range(10):
                rate_b, _ = baseline.simulate_single_realization(
                    scenario['SNR'], scenario['E'], seed
                )
                rate_f, _ = full.simulate_single_realization(
                    scenario['SNR'], scenario['E'], seed
                )
                rates_b.append(rate_b)
                rates_f.append(rate_f)

            mean_b = np.mean(rates_b)
            mean_f = np.mean(rates_f)
            gain = (mean_f - mean_b) / mean_b * 100

            results.append({
                'scenario': scenario['name'],
                'baseline_rate': mean_b / 1e6,
                'full_rate': mean_f / 1e6,
                'gain': gain
            })

            print(f"{scenario['name']:25s}: {mean_b/1e6:6.2f} -> {mean_f/1e6:6.2f} Mbps ({gain:+5.2f}%)")

        # Reset config
        config_temp.N = config.N
        config_temp.coverage_radius = config.coverage_radius

        # Find best scenario
        best_idx = np.argmax([r['gain'] for r in results])
        print(f"\n{'='*80}")
        print(f"BEST SCENARIO: {results[best_idx]['scenario']}")
        print(f"  Gain: {results[best_idx]['gain']:.2f}%")
        print(f"{'='*80}")

        return results

    def generate_recommendations(self):
        """Generate recommendations based on analysis"""
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        print("""
Based on the analysis, here are recommendations to increase optimization gains:

1. TEST CHALLENGING SCENARIOS:
   - Increase user number to N=64 or N=128
   - Increase coverage radius to R=800m or R=1000m
   - Use lower satellite elevation (E=10 deg is good)
   - Test with higher ABS bandwidth (Bd=3.0 MHz)

2. ADJUST MODULE PARAMETERS:
   - Module 1: Try multiple initial positions for gradient optimizer
   - Module 2: Increase local search iterations (max_iterations=50->100)
   - Module 3: Install Gurobi for better ILP solving

3. ENHANCE ALGORITHMS:
   - Module 1: Add height optimization in gradient descent
   - Module 2: Implement genetic algorithm for larger N
   - Module 3: Relax integer constraints for faster convergence

4. SCENARIO-SPECIFIC TUNING:
   - For heterogeneous channels: Module 2 should help more
   - For sparse users: Module 1 should help more
   - For resource-limited: Module 3 should help more

5. VALIDATION STRATEGY:
   - Run experiments with identified optimal scenarios
   - Compare against more baselines (random placement, etc.)
   - Add confidence intervals to results
        """)

    def run_full_analysis(self):
        """Run all analyses"""
        print("=" * 80)
        print("PERFORMANCE GAP ANALYSIS")
        print("=" * 80)
        print("\nAnalyzing why current gain is 3-4% instead of target 40%\n")

        # Analysis 1
        self.results['bottleneck'] = self.analyze_baseline_bottleneck()

        # Analysis 2
        self.results['scenario'] = self.analyze_scenario_characteristics()

        # Analysis 3
        self.results['sensitivity'] = self.test_parameter_sensitivity()

        # Analysis 4
        self.results['optimal_scenarios'] = self.identify_optimal_scenarios()

        # Recommendations
        self.generate_recommendations()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

        return self.results


# ==================== Main ====================
if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    results = analyzer.run_full_analysis()
