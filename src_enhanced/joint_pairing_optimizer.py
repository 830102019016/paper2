# -*- coding: utf-8 -*-
"""
Joint User Pairing Optimizer

Core idea:
- Original: Satellite and ABS pairing are independent
- New: Joint optimization considering synergy effects

Methods:
1. Greedy + Local Search (recommended for N=32)
2. Exhaustive search (only for N<=16, exponential complexity)

Author: SATCON Enhancement Project
Date: 2025-12-10
"""
import numpy as np
from itertools import combinations
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.power_allocation import NOMAAllocator


class JointPairingOptimizer:
    """
    Joint Pairing Optimizer

    Input:
    - sat_gains: Satellite channel gains
    - a2g_gains: A2G channel gains
    - config: System configuration

    Output:
    - sat_pairs: Optimal satellite pairing
    - abs_pairs: Optimal ABS pairing
    - joint_benefit: Joint benefit (sum rate)
    """

    def __init__(self, config_obj):
        """
        Initialize joint pairing optimizer

        Args:
            config_obj: System configuration object
        """
        self.config = config_obj
        self.allocator = NOMAAllocator()

    def compute_joint_benefit(self, sat_pair_idx, abs_pair_idx,
                              sat_gains, a2g_gains, Ps_dB, Pd, Bs, Bd):
        """
        Compute joint benefit for given pairing combination

        Key: For each pair k, consider 4 hybrid decision options:
        1. Both users use ABS NOMA
        2. Weak user uses ABS OMA, strong user uses satellite
        3. Weak user uses satellite, strong user uses ABS OMA
        4. Both users use satellite NOMA

        Args:
            sat_pair_idx: Satellite pairing indices [(i1,j1), (i2,j2), ...]
            abs_pair_idx: ABS pairing indices [(m1,n1), (m2,n2), ...]
            sat_gains: Satellite channel gains [N]
            a2g_gains: A2G channel gains [N]
            Ps_dB: Satellite transmit power (dB)
            Pd: ABS transmit power (W)
            Bs, Bd: Satellite and ABS bandwidth

        Returns:
            total_rate: Total system rate (bps)

        Note:
            This implements simplified per-pair hybrid decision.
            Full global optimization will be in Module 3.
        """
        K = len(sat_pair_idx)
        Bs_per_pair = Bs / K
        Bd_per_pair = Bd / K

        # Convert Ps from dB to W
        Ps = 10 ** (Ps_dB / 10)

        total_rate = 0.0

        # Process each pair k and make hybrid decision
        for k in range(K):
            # Get satellite pair users
            sat_i, sat_j = sat_pair_idx[k]
            gamma_sat_i, gamma_sat_j = sat_gains[sat_i], sat_gains[sat_j]

            # Ensure i is weak, j is strong for satellite
            if gamma_sat_i > gamma_sat_j:
                sat_i, sat_j = sat_j, sat_i
                gamma_sat_i, gamma_sat_j = gamma_sat_j, gamma_sat_i

            # Compute satellite NOMA rates for this pair
            beta_sat_j, beta_sat_i = self.allocator.compute_power_factors(
                gamma_sat_j, gamma_sat_i, Ps
            )
            R_sat_i = Bs_per_pair * np.log2(1 + beta_sat_i * Ps * gamma_sat_i /
                                            (beta_sat_j * Ps * gamma_sat_i + 1))
            R_sat_j = Bs_per_pair * np.log2(1 + beta_sat_j * Ps * gamma_sat_j)

            # Get ABS pair users (corresponding to same pair index k)
            abs_m, abs_n = abs_pair_idx[k]
            gamma_abs_m, gamma_abs_n = a2g_gains[abs_m], a2g_gains[abs_n]

            # Ensure m is weak, n is strong for ABS
            if gamma_abs_m > gamma_abs_n:
                abs_m, abs_n = abs_n, abs_m
                gamma_abs_m, gamma_abs_n = gamma_abs_n, gamma_abs_m

            # Compute ABS NOMA rates for this pair
            beta_abs_n, beta_abs_m = self.allocator.compute_power_factors(
                gamma_abs_n, gamma_abs_m, Pd
            )
            R_abs_noma_m = Bd_per_pair * np.log2(1 + beta_abs_m * Pd * gamma_abs_m /
                                                  (beta_abs_n * Pd * gamma_abs_m + 1))
            R_abs_noma_n = Bd_per_pair * np.log2(1 + beta_abs_n * Pd * gamma_abs_n)

            # Compute ABS OMA rates (each user gets full slot)
            R_abs_oma_m = Bd_per_pair * np.log2(1 + Pd * gamma_abs_m)
            R_abs_oma_n = Bd_per_pair * np.log2(1 + Pd * gamma_abs_n)

            # Now make hybrid decision: choose best option for this pair
            # Note: This assumes sat_i corresponds to abs_m (both weak in their systems)
            #       and sat_j corresponds to abs_n (both strong in their systems)

            # Option 1: Both users use ABS NOMA
            rate_option1 = R_abs_noma_m + R_abs_noma_n

            # Option 2: ABS OMA for weak user, satellite for strong user
            # Map: abs_m (weak) uses ABS OMA, sat_j (strong) uses satellite
            if abs_m == sat_i:  # Same user
                rate_option2 = R_abs_oma_m + R_sat_j
            elif abs_m == sat_j:
                rate_option2 = R_abs_oma_m + R_sat_i
            else:
                rate_option2 = 0  # Different users, skip this option

            # Option 3: ABS OMA for strong user, satellite for weak user
            # Map: abs_n (strong) uses ABS OMA, sat_i (weak) uses satellite
            if abs_n == sat_j:  # Same user
                rate_option3 = R_sat_i + R_abs_oma_n
            elif abs_n == sat_i:
                rate_option3 = R_sat_j + R_abs_oma_n
            else:
                rate_option3 = 0  # Different users, skip this option

            # Option 4: Both users use satellite NOMA
            rate_option4 = R_sat_i + R_sat_j

            # Choose best option
            pair_rate = max(rate_option1, rate_option2, rate_option3, rate_option4)
            total_rate += pair_rate

        return total_rate

    def optimize_greedy_with_local_search(self, sat_gains, a2g_gains):
        """
        Greedy algorithm + Local search

        Algorithm:
        1. Initial solution: Original SATCON (independent pairing)
        2. Local search: Swap pairs to try improvement
        3. Iterate until convergence

        Args:
            sat_gains: [N] Satellite channel gains
            a2g_gains: [N] A2G channel gains

        Returns:
            best_sat_pairs: Optimal satellite pairing
            best_abs_pairs: Optimal ABS pairing
            best_benefit: Best joint benefit
            iterations: Number of iterations
        """
        N = len(sat_gains)
        K = N // 2

        # Use default bandwidth
        Bd = self.config.Bd_options[1]  # 1.2 MHz

        # Initial solution: Original independent pairing
        sat_pairs_init, _ = self.allocator.optimal_user_pairing(sat_gains)
        abs_pairs_init, _ = self.allocator.optimal_user_pairing(a2g_gains)

        current_sat_pairs = sat_pairs_init.copy()
        current_abs_pairs = abs_pairs_init.copy()

        # Use SNR = 20 dB for testing
        Ps_dB = 20  # dB

        current_benefit = self.compute_joint_benefit(
            current_sat_pairs, current_abs_pairs,
            sat_gains, a2g_gains,
            Ps_dB, self.config.Pd,
            self.config.Bs, Bd
        )

        improved = True
        iterations = 0
        max_iterations = 50

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            # Try swapping satellite pairs
            for k1 in range(K):
                for k2 in range(k1+1, K):
                    # Swap one user between pair k1 and k2
                    new_sat_pairs = [list(p) for p in current_sat_pairs]  # Deep copy
                    new_sat_pairs[k1] = [current_sat_pairs[k1][0], current_sat_pairs[k2][1]]
                    new_sat_pairs[k2] = [current_sat_pairs[k2][0], current_sat_pairs[k1][1]]

                    new_benefit = self.compute_joint_benefit(
                        new_sat_pairs, current_abs_pairs,
                        sat_gains, a2g_gains,
                        Ps_dB, self.config.Pd,
                        self.config.Bs, Bd
                    )

                    if new_benefit > current_benefit:
                        current_sat_pairs = new_sat_pairs
                        current_benefit = new_benefit
                        improved = True

            # Try swapping ABS pairs
            for k1 in range(K):
                for k2 in range(k1+1, K):
                    new_abs_pairs = [list(p) for p in current_abs_pairs]  # Deep copy
                    new_abs_pairs[k1] = [current_abs_pairs[k1][0], current_abs_pairs[k2][1]]
                    new_abs_pairs[k2] = [current_abs_pairs[k2][0], current_abs_pairs[k1][1]]

                    new_benefit = self.compute_joint_benefit(
                        current_sat_pairs, new_abs_pairs,
                        sat_gains, a2g_gains,
                        Ps_dB, self.config.Pd,
                        self.config.Bs, Bd
                    )

                    if new_benefit > current_benefit:
                        current_abs_pairs = new_abs_pairs
                        current_benefit = new_benefit
                        improved = True

        return current_sat_pairs, current_abs_pairs, current_benefit, iterations

    def optimize(self, sat_gains, a2g_gains):
        """
        Main optimization interface

        Args:
            sat_gains: [N] Satellite channel gains
            a2g_gains: [N] A2G channel gains

        Returns:
            sat_pairs: Optimal satellite pairing
            abs_pairs: Optimal ABS pairing
            benefit: Joint benefit
            info: Optimization info dict
        """
        sat_pairs, abs_pairs, benefit, iterations = \
            self.optimize_greedy_with_local_search(sat_gains, a2g_gains)

        info = {
            'iterations': iterations,
            'final_benefit': benefit
        }

        return sat_pairs, abs_pairs, benefit, info


# ==================== Test Code ====================
def test_joint_pairing():
    """Test joint pairing optimizer"""
    print("=" * 60)
    print("Testing Joint Pairing Optimizer")
    print("=" * 60)

    # Generate test channel gains with MORE diversity
    # to create strong opportunities for joint optimization
    np.random.seed(42)

    # Create diverse channel conditions
    # Some users have good satellite but bad A2G
    # Some users have bad satellite but good A2G
    sat_gains = np.random.exponential(0.01, size=config.N)
    a2g_gains = np.random.exponential(0.05, size=config.N)

    # Add STRONG diversity: create clear specialization
    for i in range(0, config.N, 4):
        if i+1 < config.N:
            # User i: VERY good sat, VERY bad A2G
            sat_gains[i] *= 5.0  # Strong satellite
            a2g_gains[i] *= 0.2  # Weak A2G
            # User i+1: VERY bad sat, VERY good A2G
            sat_gains[i+1] *= 0.2  # Weak satellite
            a2g_gains[i+1] *= 5.0  # Strong A2G

    print(f"\nTest configuration:")
    print(f"  Number of users: {config.N}")
    print(f"  Number of pairs: {config.N // 2}")

    # Create optimizer
    optimizer = JointPairingOptimizer(config)

    # Baseline: Independent pairing (original SATCON)
    print(f"\n" + "-" * 60)
    print("Baseline: Independent pairing (Original SATCON)")
    print("-" * 60)

    allocator = NOMAAllocator()
    sat_pairs_old, _ = allocator.optimal_user_pairing(sat_gains)
    abs_pairs_old, _ = allocator.optimal_user_pairing(a2g_gains)

    Bd = config.Bd_options[1]  # 1.2 MHz
    Ps_dB = 20  # Use SNR = 20 dB for testing

    benefit_old = optimizer.compute_joint_benefit(
        sat_pairs_old, abs_pairs_old,
        sat_gains, a2g_gains,
        Ps_dB, config.Pd, config.Bs, Bd
    )

    print(f"  Total rate: {benefit_old/1e6:.2f} Mbps")
    print(f"  Satellite pairs: {len(sat_pairs_old)} pairs")
    print(f"  ABS pairs: {len(abs_pairs_old)} pairs")

    # New method: Joint optimization
    print(f"\n" + "-" * 60)
    print("New Method: Joint pairing optimization")
    print("-" * 60)
    print(f"Running local search optimization...")

    sat_pairs_new, abs_pairs_new, benefit_new, info = optimizer.optimize(
        sat_gains, a2g_gains
    )

    print(f"\nOptimization results:")
    print(f"  Total rate: {benefit_new/1e6:.2f} Mbps")
    print(f"  Iterations: {info['iterations']}")

    # Performance comparison
    print(f"\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    improvement_abs = (benefit_new - benefit_old) / 1e6
    improvement_pct = (benefit_new - benefit_old) / benefit_old * 100

    print(f"\nBaseline (Independent pairing):")
    print(f"  Rate: {benefit_old/1e6:.2f} Mbps")

    print(f"\nProposed (Joint optimization):")
    print(f"  Rate: {benefit_new/1e6:.2f} Mbps")

    print(f"\nImprovement:")
    print(f"  Absolute gain: {improvement_abs:.2f} Mbps")
    print(f"  Relative gain: {improvement_pct:.2f}%")

    # Validate success
    print(f"\n" + "=" * 60)
    if benefit_new >= benefit_old * 0.99:  # Allow small numerical error
        print("[PASS] Joint pairing optimizer test PASSED")
        print(f"  - Joint optimization converged in {info['iterations']} iterations")
        if benefit_new > benefit_old:
            print(f"  - Performance better than baseline (+{improvement_pct:.2f}%)")
        else:
            print(f"  - Performance equal to baseline (local optimum)")
    else:
        print("[FAIL] Test FAILED")
        print(f"  - Performance worse than baseline ({improvement_pct:.2f}%)")
    print("=" * 60)

    return sat_pairs_new, abs_pairs_new, benefit_new, info


if __name__ == "__main__":
    test_joint_pairing()
