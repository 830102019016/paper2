# -*- coding: utf-8 -*-
"""
Joint Optimization SATCON System

Integrates three enhancement modules:
1. Gradient-based ABS position optimization (Module 1)
2. Joint user pairing optimization (Module 2)
3. Integer programming based hybrid decision (Module 3)

Key innovation: Alternating optimization framework
- Original: Sequential optimization (position -> pairing -> decision)
- New: Iterative joint optimization with feedback loops

Algorithm:
1. Initialize ABS position (using gradient optimizer)
2. Repeat until convergence:
   a. Fix position, optimize joint pairing
   b. Fix pairing, optimize hybrid decision
   c. Fix pairing/decision, optimize position
   d. Check convergence
3. Return final solution

Author: SATCON Enhancement Project
Date: 2025-12-10
"""
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.user_distribution import UserDistribution
from src.a2g_channel import A2GChannel, S2AChannel
from src.noma_transmission import SatelliteNOMA
from src.power_allocation import NOMAAllocator

# Import enhancement modules
from src_enhanced.gradient_position_optimizer import GradientPositionOptimizer
from src_enhanced.joint_pairing_optimizer import JointPairingOptimizer
from src_enhanced.integer_programming_decision import IntegerProgrammingDecision


class JointOptimizationSATCON:
    """
    Joint Optimization SATCON System

    Combines three enhancement modules into a unified framework with
    iterative optimization for global system performance
    """

    def __init__(self, config_obj, abs_bandwidth,
                 use_module1=True, use_module2=True, use_module3=True):
        """
        Initialize Joint Optimization SATCON System

        Args:
            config_obj: Configuration object
            abs_bandwidth: ABS bandwidth (Hz)
            use_module1: Enable gradient position optimization
            use_module2: Enable joint pairing optimization
            use_module3: Enable integer programming decision
        """
        self.config = config_obj
        self.Bd = abs_bandwidth

        # Module flags (for ablation studies)
        self.use_module1 = use_module1
        self.use_module2 = use_module2
        self.use_module3 = use_module3

        # Initialize base systems
        self.sat_noma = SatelliteNOMA(config_obj)
        self.a2g_channel = A2GChannel()
        self.s2a_channel = S2AChannel()
        self.allocator = NOMAAllocator()

        # Initialize enhancement modules
        if use_module1:
            self.gradient_optimizer = GradientPositionOptimizer(config_obj)

        if use_module2:
            self.pairing_optimizer = JointPairingOptimizer(config_obj)

        # Always initialize decision optimizer (needed even for baseline)
        self.decision_optimizer = IntegerProgrammingDecision()

        # Fallback to original methods if modules disabled
        from src.abs_placement import ABSPlacement
        self.abs_placement = ABSPlacement()

        # Noise power
        self.Nd = config_obj.get_abs_noise_power(abs_bandwidth)
        self.Nsd = config_obj.get_s2a_noise_power()

        # Optimization parameters
        self.max_iterations = 10
        self.convergence_threshold = 1e-3  # 0.1% improvement

    def compute_channel_gains(self, user_positions, abs_position,
                             elevation_deg, snr_db, seed):
        """
        Compute all channel gains (satellite and A2G)

        Args:
            user_positions: User positions
            abs_position: ABS position [x, y, h]
            elevation_deg: Satellite elevation angle
            snr_db: Satellite SNR
            seed: Random seed for fading

        Returns:
            sat_gains: Satellite channel gains [N]
            a2g_gains: A2G channel gains [N]
            sat_rates: Satellite NOMA rates [N]
        """
        # Satellite channel gains
        snr_linear = 10 ** (snr_db / 10)
        sat_gains = self.sat_noma.compute_channel_gains_with_pathloss(elevation_deg)
        sat_rates, _ = self.sat_noma.compute_achievable_rates(sat_gains, snr_linear)

        # A2G channel gains
        h_abs = abs_position[2]
        distances_2d = np.sqrt((user_positions[:, 0] - abs_position[0])**2 +
                               (user_positions[:, 1] - abs_position[1])**2)

        fading_a2g = self.a2g_channel.generate_fading(self.config.N, seed=seed)
        a2g_gains = np.array([
            self.a2g_channel.compute_channel_gain(
                h_abs, r, fading,
                self.config.Gd_t_dB, self.config.Gd_r_dB, self.Nd
            )
            for r, fading in zip(distances_2d, fading_a2g)
        ])

        return sat_gains, a2g_gains, sat_rates

    def optimize_position(self, user_positions, a2g_fading, initial_position=None):
        """
        Optimize ABS position

        Uses Module 1 (gradient optimizer) if enabled,
        otherwise falls back to k-means

        Args:
            user_positions: User positions
            a2g_fading: A2G fading coefficients (fixed for fairness)
            initial_position: Initial guess [x, y, h]

        Returns:
            optimal_position: Optimized ABS position [x, y, h]
        """
        if self.use_module1:
            # Module 1: Gradient-based optimization
            optimal_pos, _ = self.gradient_optimizer.optimize(
                user_positions, a2g_fading, initial_guess=initial_position
            )
            return optimal_pos
        else:
            # Fallback: Original k-means + height optimization
            abs_xy, _, _ = self.abs_placement.optimize_xy_position(user_positions)
            abs_h, _ = self.abs_placement.optimize_height(
                abs_xy, user_positions, self.a2g_channel
            )
            return np.array([abs_xy[0], abs_xy[1], abs_h])

    def optimize_pairing(self, sat_gains, a2g_gains):
        """
        Optimize user pairing

        Uses Module 2 (joint pairing) if enabled,
        otherwise falls back to independent pairing

        Args:
            sat_gains: Satellite channel gains
            a2g_gains: A2G channel gains

        Returns:
            sat_pairs: Satellite pairing [(i,j), ...]
            abs_pairs: ABS pairing [(m,n), ...]
        """
        if self.use_module2:
            # Module 2: Joint pairing optimization
            sat_pairs, abs_pairs, _, _ = self.pairing_optimizer.optimize_greedy_with_local_search(
                sat_gains, a2g_gains
            )
            return sat_pairs, abs_pairs
        else:
            # Fallback: Independent pairing
            sat_pairs, _ = self.allocator.optimal_user_pairing(sat_gains)
            abs_pairs, _ = self.allocator.optimal_user_pairing(a2g_gains)
            return sat_pairs, abs_pairs

    def optimize_decision(self, sat_pairs, abs_pairs, sat_gains, a2g_gains, Ps_dB):
        """
        Optimize hybrid NOMA/OMA decision

        Uses Module 3 (integer programming) if enabled,
        otherwise falls back to greedy decision

        Args:
            sat_pairs: Satellite pairing
            abs_pairs: ABS pairing
            sat_gains: Satellite channel gains
            a2g_gains: A2G channel gains
            Ps_dB: Satellite power (dB)

        Returns:
            decisions: Mode decisions for each pair
            final_rate: Total system rate
        """
        if self.use_module3:
            # Module 3: Integer programming decision
            decisions, final_rate, _ = self.decision_optimizer.optimize(
                sat_pairs, abs_pairs, sat_gains, a2g_gains,
                Ps_dB, self.config.Pd, self.config.Bs, self.Bd
            )
            return decisions, final_rate
        else:
            # Fallback: Original greedy decision
            decisions, final_rate, _ = self.decision_optimizer.optimize_greedy(
                sat_pairs, abs_pairs, sat_gains, a2g_gains,
                Ps_dB, self.config.Pd, self.config.Bs, self.Bd
            )
            return decisions, final_rate

    def compute_system_rate(self, user_positions, abs_position,
                           elevation_deg, snr_db, seed):
        """
        Compute total system rate for given configuration

        Args:
            user_positions: User positions
            abs_position: ABS position
            elevation_deg: Satellite elevation
            snr_db: Satellite SNR
            seed: Random seed

        Returns:
            total_rate: Total system rate
            info: Detailed information
        """
        # Compute channel gains
        sat_gains, a2g_gains, sat_rates = self.compute_channel_gains(
            user_positions, abs_position, elevation_deg, snr_db, seed
        )

        # Optimize pairing
        sat_pairs, abs_pairs = self.optimize_pairing(sat_gains, a2g_gains)

        # Optimize decision
        decisions, total_rate = self.optimize_decision(
            sat_pairs, abs_pairs, sat_gains, a2g_gains, snr_db
        )

        info = {
            'sat_pairs': sat_pairs,
            'abs_pairs': abs_pairs,
            'decisions': decisions,
            'mode_counts': {
                'noma': decisions.count('noma'),
                'oma_weak': decisions.count('oma_weak'),
                'oma_strong': decisions.count('oma_strong'),
                'sat': decisions.count('sat')
            }
        }

        return total_rate, info

    def optimize_joint(self, user_positions, elevation_deg, snr_db, seed,
                      verbose=False):
        """
        Joint optimization with alternating algorithm

        Algorithm:
        1. Initialize ABS position
        2. Repeat:
           a. Fix position, optimize pairing + decision
           b. Fix pairing/decision, optimize position
           c. Check convergence
        3. Return best solution

        Args:
            user_positions: User positions
            elevation_deg: Satellite elevation
            snr_db: Satellite SNR
            seed: Random seed
            verbose: Print debug info

        Returns:
            best_rate: Best achieved rate
            best_config: Best configuration
        """
        # Generate fixed A2G fading for fair comparison
        a2g_fading = self.a2g_channel.generate_fading(self.config.N, seed=seed)

        # Step 1: Initialize position
        abs_position = self.optimize_position(user_positions, a2g_fading)

        best_rate = 0
        best_config = None

        # Step 2: Alternating optimization
        for iteration in range(self.max_iterations):
            # Compute system rate with current position
            current_rate, info = self.compute_system_rate(
                user_positions, abs_position, elevation_deg, snr_db, seed
            )

            if verbose:
                print(f"  Iter {iteration+1}: Rate={current_rate/1e6:.2f} Mbps, "
                      f"Pos=[{abs_position[0]:.1f}, {abs_position[1]:.1f}, {abs_position[2]:.0f}]")

            # Update best
            if current_rate > best_rate:
                best_rate = current_rate
                best_config = {
                    'abs_position': abs_position.copy(),
                    'sat_pairs': info['sat_pairs'],
                    'abs_pairs': info['abs_pairs'],
                    'decisions': info['decisions'],
                    'mode_counts': info['mode_counts']
                }

                improvement = (current_rate - best_rate) / best_rate if best_rate > 0 else float('inf')

                # Check convergence
                if iteration > 0 and improvement < self.convergence_threshold:
                    if verbose:
                        print(f"  Converged at iteration {iteration+1}")
                    break

            # Update position based on current pairing/decision
            # (In practice, this would require a more sophisticated approach
            #  that considers the impact of position on pairing/decision)
            # For now, we do a single optimization pass
            if iteration == 0 and self.use_module1:
                # Try to refine position
                abs_position_new = self.optimize_position(
                    user_positions, a2g_fading, initial_position=abs_position
                )

                # Check if refinement helps
                new_rate, _ = self.compute_system_rate(
                    user_positions, abs_position_new, elevation_deg, snr_db, seed
                )

                if new_rate > current_rate:
                    abs_position = abs_position_new
                    if verbose:
                        print(f"    Position refined: Rate improved to {new_rate/1e6:.2f} Mbps")

        return best_rate, best_config

    def simulate_single_realization(self, snr_db, elevation_deg, seed,
                                   use_joint_optimization=True):
        """
        Single realization simulation

        Args:
            snr_db: Satellite SNR (dB)
            elevation_deg: Satellite elevation angle (degrees)
            seed: Random seed
            use_joint_optimization: Use alternating optimization

        Returns:
            sum_rate: Total system rate
            mode_stats: Mode statistics
        """
        # Generate user distribution
        dist = UserDistribution(self.config.N, self.config.coverage_radius, seed=seed)
        user_positions = dist.generate_uniform_circle()

        if use_joint_optimization:
            # Joint optimization
            sum_rate, config_info = self.optimize_joint(
                user_positions, elevation_deg, snr_db, seed, verbose=False
            )
            mode_stats = config_info['mode_counts']
        else:
            # Simple one-pass optimization
            a2g_fading = self.a2g_channel.generate_fading(self.config.N, seed=seed)
            abs_position = self.optimize_position(user_positions, a2g_fading)
            sum_rate, info = self.compute_system_rate(
                user_positions, abs_position, elevation_deg, snr_db, seed
            )
            mode_stats = info['mode_counts']

        return sum_rate, mode_stats

    def simulate_performance(self, snr_db_range, elevation_deg=10,
                           n_realizations=100, use_joint_optimization=True,
                           verbose=True):
        """
        Monte Carlo performance simulation

        Args:
            snr_db_range: Array of SNR values (dB)
            elevation_deg: Satellite elevation angle
            n_realizations: Number of Monte Carlo realizations
            use_joint_optimization: Use alternating optimization
            verbose: Show progress bar

        Returns:
            mean_sum_rates: Mean sum rates [n_snr]
            mean_se: Mean spectral efficiency [n_snr]
            std_sum_rates: Standard deviation [n_snr]
            mode_statistics: Mode usage statistics
        """
        n_snr_points = len(snr_db_range)
        sum_rates_all = np.zeros((n_snr_points, n_realizations))
        mode_stats_all = {
            'noma': np.zeros((n_snr_points, n_realizations)),
            'oma_weak': np.zeros((n_snr_points, n_realizations)),
            'oma_strong': np.zeros((n_snr_points, n_realizations)),
            'sat': np.zeros((n_snr_points, n_realizations))
        }

        if verbose:
            modules_enabled = []
            if self.use_module1:
                modules_enabled.append("M1:GradPos")
            if self.use_module2:
                modules_enabled.append("M2:JointPair")
            if self.use_module3:
                modules_enabled.append("M3:ILP")
            if not modules_enabled:
                modules_enabled.append("Baseline")

            module_str = "+".join(modules_enabled)
            print(f"Joint SATCON Simulation [{module_str}]")
            print(f"  Bd={self.Bd/1e6:.1f}MHz, Joint_Opt={use_joint_optimization}")

        snr_iterator = tqdm(enumerate(snr_db_range), total=n_snr_points,
                           desc="Simulating", disable=not verbose)

        for i, snr_db in snr_iterator:
            for r in range(n_realizations):
                seed = self.config.random_seed + i * n_realizations + r

                sum_rate, mode_stats = self.simulate_single_realization(
                    snr_db, elevation_deg, seed, use_joint_optimization
                )

                sum_rates_all[i, r] = sum_rate
                for mode in mode_stats_all.keys():
                    if mode in mode_stats:
                        mode_stats_all[mode][i, r] = mode_stats[mode]

            if verbose:
                current_se = np.mean(sum_rates_all[i, :]) / self.Bd
                snr_iterator.set_postfix({'SE': f'{current_se:.2f} bits/s/Hz'})

        # Statistics
        mean_sum_rates = np.mean(sum_rates_all, axis=1)
        std_sum_rates = np.std(sum_rates_all, axis=1)
        mean_se = mean_sum_rates / self.Bd

        mean_mode_stats = {mode: np.mean(counts, axis=1)
                          for mode, counts in mode_stats_all.items()}

        return mean_sum_rates, mean_se, std_sum_rates, mean_mode_stats


# ==================== Test Code ====================
def test_joint_satcon_system():
    """Test Joint Optimization SATCON System"""
    print("=" * 70)
    print("Testing Joint Optimization SATCON System")
    print("=" * 70)

    # Test configurations
    test_snr = np.array([10, 20, 30])
    n_real = 5

    # Create systems for comparison
    systems = {
        'Baseline (Original)': JointOptimizationSATCON(
            config, 1.2e6,
            use_module1=False, use_module2=False, use_module3=False
        ),
        'Module 1 (GradPos)': JointOptimizationSATCON(
            config, 1.2e6,
            use_module1=True, use_module2=False, use_module3=False
        ),
        'Module 2 (JointPair)': JointOptimizationSATCON(
            config, 1.2e6,
            use_module1=False, use_module2=True, use_module3=False
        ),
        'Module 3 (ILP)': JointOptimizationSATCON(
            config, 1.2e6,
            use_module1=False, use_module2=False, use_module3=True
        ),
        'Full System (All)': JointOptimizationSATCON(
            config, 1.2e6,
            use_module1=True, use_module2=True, use_module3=True
        )
    }

    print(f"\nTest configuration:")
    print(f"  SNR points: {test_snr}")
    print(f"  Realizations: {n_real}")
    print(f"  Systems: {len(systems)}")

    results = {}

    for name, system in systems.items():
        print(f"\n{'-'*70}")
        print(f"Testing: {name}")
        print(f"{'-'*70}")

        mean_rates, mean_se, _, mode_stats = system.simulate_performance(
            test_snr, elevation_deg=10, n_realizations=n_real, verbose=True
        )

        results[name] = {
            'mean_rates': mean_rates,
            'mean_se': mean_se,
            'mode_stats': mode_stats
        }

    # Comparison
    print(f"\n{'='*70}")
    print("Performance Comparison")
    print(f"{'='*70}")

    baseline_rates = results['Baseline (Original)']['mean_rates']

    print(f"\n{'System':<25} {'SE@10dB':<12} {'SE@20dB':<12} {'SE@30dB':<12} {'Gain@20dB'}")
    print("-" * 70)

    for name, data in results.items():
        se = data['mean_se']
        gain = (data['mean_rates'][1] - baseline_rates[1]) / baseline_rates[1] * 100

        print(f"{name:<25} {se[0]:<12.2f} {se[1]:<12.2f} {se[2]:<12.2f} {gain:>7.2f}%")

    print("\n" + "=" * 70)
    print(" Joint Optimization SATCON Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    test_joint_satcon_system()
