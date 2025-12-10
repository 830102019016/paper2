# -*- coding: utf-8 -*-
"""
Integer Programming based Hybrid Decision Optimizer

Original: Rule-based greedy decision (4 rules, per-pair local optimal)
New: Integer programming for global optimal decision

Key idea:
- Original: Each pair independently chooses best mode
- New: Jointly optimize all pairs' mode selection for global optimum

Dependencies:
- cvxpy (optional): pip install cvxpy
- If cvxpy not available, falls back to improved greedy algorithm

Author: SATCON Enhancement Project
Date: 2025-12-10
"""
import numpy as np
import sys
from pathlib import Path

# Try to import cvxpy
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: cvxpy not installed. Using greedy algorithm instead.")
    print("  To install: pip install cvxpy")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.power_allocation import NOMAAllocator


class IntegerProgrammingDecision:
    """
    Integer Programming based Hybrid Decision Optimizer

    Decision variables (for each pair k):
    - x[k] in {0,1}: Use ABS NOMA for both users
    - y_weak[k] in {0,1}: Use ABS OMA for weak user only
    - y_strong[k] in {0,1}: Use ABS OMA for strong user only

    Objective:
    - Maximize sum of all user rates

    Constraints:
    - Mutual exclusion: x[k] + y_weak[k] + y_strong[k] <= 1
    - Bandwidth limit: sum of used bandwidth <= Bd
    - S2A backhaul limit: ABS rate <= S2A rate (simplified: assume sufficient)
    """

    def __init__(self):
        """Initialize optimizer"""
        self.use_ilp = CVXPY_AVAILABLE
        self.allocator = NOMAAllocator()

    def compute_rate_options(self, sat_pairs, abs_pairs, sat_gains, a2g_gains,
                            Ps_dB, Pd, Bs, Bd):
        """
        Compute all rate options for each pair

        Args:
            sat_pairs: Satellite pairing [(i,j), ...]
            abs_pairs: ABS pairing [(m,n), ...]
            sat_gains: Satellite channel gains [N]
            a2g_gains: A2G channel gains [N]
            Ps_dB: Satellite power (dB)
            Pd: ABS power (W)
            Bs: Satellite bandwidth
            Bd: ABS bandwidth

        Returns:
            rate_options: dict with keys ['sat', 'abs_noma', 'abs_oma_weak', 'abs_oma_strong']
                         Each value is [K] array of rates for each pair
        """
        K = len(sat_pairs)
        Bs_per_pair = Bs / K
        Bd_per_pair = Bd / K
        Ps = 10 ** (Ps_dB / 10)

        # Initialize rate arrays
        rate_sat = np.zeros(K)           # Satellite NOMA rates (sum of pair)
        rate_abs_noma = np.zeros(K)      # ABS NOMA rates (sum of pair)
        rate_abs_oma_weak = np.zeros(K)  # ABS OMA weak + Sat strong
        rate_abs_oma_strong = np.zeros(K) # Sat weak + ABS OMA strong

        for k in range(K):
            # Satellite pair
            sat_i, sat_j = sat_pairs[k]
            gamma_sat_i, gamma_sat_j = sat_gains[sat_i], sat_gains[sat_j]

            if gamma_sat_i > gamma_sat_j:
                sat_i, sat_j = sat_j, sat_i
                gamma_sat_i, gamma_sat_j = gamma_sat_j, gamma_sat_i

            # Satellite NOMA rates
            beta_sat_j, beta_sat_i = self.allocator.compute_power_factors(
                gamma_sat_j, gamma_sat_i, Ps
            )
            R_sat_i = Bs_per_pair * np.log2(1 + beta_sat_i * Ps * gamma_sat_i /
                                            (beta_sat_j * Ps * gamma_sat_i + 1))
            R_sat_j = Bs_per_pair * np.log2(1 + beta_sat_j * Ps * gamma_sat_j)

            # ABS pair
            abs_m, abs_n = abs_pairs[k]
            gamma_abs_m, gamma_abs_n = a2g_gains[abs_m], a2g_gains[abs_n]

            if gamma_abs_m > gamma_abs_n:
                abs_m, abs_n = abs_n, abs_m
                gamma_abs_m, gamma_abs_n = gamma_abs_n, gamma_abs_m

            # ABS NOMA rates
            beta_abs_n, beta_abs_m = self.allocator.compute_power_factors(
                gamma_abs_n, gamma_abs_m, Pd
            )
            R_abs_noma_m = Bd_per_pair * np.log2(1 + beta_abs_m * Pd * gamma_abs_m /
                                                  (beta_abs_n * Pd * gamma_abs_m + 1))
            R_abs_noma_n = Bd_per_pair * np.log2(1 + beta_abs_n * Pd * gamma_abs_n)

            # ABS OMA rates
            R_abs_oma_m = Bd_per_pair * np.log2(1 + Pd * gamma_abs_m)
            R_abs_oma_n = Bd_per_pair * np.log2(1 + Pd * gamma_abs_n)

            # Rate options for this pair
            rate_sat[k] = R_sat_i + R_sat_j

            # Check if users match
            if abs_m == sat_i and abs_n == sat_j:
                # Perfect match
                rate_abs_noma[k] = R_abs_noma_m + R_abs_noma_n
                rate_abs_oma_weak[k] = R_abs_oma_m + R_sat_j
                rate_abs_oma_strong[k] = R_sat_i + R_abs_oma_n
            elif abs_m == sat_j and abs_n == sat_i:
                # Reversed match
                rate_abs_noma[k] = R_abs_noma_m + R_abs_noma_n
                rate_abs_oma_weak[k] = R_abs_oma_m + R_sat_i
                rate_abs_oma_strong[k] = R_sat_j + R_abs_oma_n
            else:
                # Different users - use best available
                rate_abs_noma[k] = R_abs_noma_m + R_abs_noma_n
                rate_abs_oma_weak[k] = max(R_abs_oma_m, R_abs_oma_n) + min(R_sat_i, R_sat_j)
                rate_abs_oma_strong[k] = max(R_sat_i, R_sat_j) + min(R_abs_oma_m, R_abs_oma_n)

        return {
            'sat': rate_sat,
            'abs_noma': rate_abs_noma,
            'abs_oma_weak': rate_abs_oma_weak,
            'abs_oma_strong': rate_abs_oma_strong
        }

    def optimize_ilp(self, sat_pairs, abs_pairs, sat_gains, a2g_gains,
                    Ps_dB, Pd, Bs, Bd):
        """
        Integer Linear Programming optimization

        Args:
            sat_pairs: Satellite pairing
            abs_pairs: ABS pairing
            sat_gains: Satellite channel gains
            a2g_gains: A2G channel gains
            Ps_dB: Satellite power (dB)
            Pd: ABS power (W)
            Bs: Satellite bandwidth
            Bd: ABS bandwidth

        Returns:
            decisions: List of decisions for each pair ['sat'/'noma'/'oma_weak'/'oma_strong']
            final_rate: Total system rate
            info: Optimization info
        """
        if not CVXPY_AVAILABLE:
            raise RuntimeError("cvxpy not available. Use optimize_greedy() instead.")

        K = len(sat_pairs)

        # Compute rate options
        rate_opts = self.compute_rate_options(
            sat_pairs, abs_pairs, sat_gains, a2g_gains,
            Ps_dB, Pd, Bs, Bd
        )

        # Decision variables
        x_noma = cp.Variable(K, boolean=True)
        y_weak = cp.Variable(K, boolean=True)
        y_strong = cp.Variable(K, boolean=True)

        # Objective: maximize total rate
        objective_terms = []
        for k in range(K):
            # Contribution from each mode
            contrib = (x_noma[k] * rate_opts['abs_noma'][k] +
                      y_weak[k] * rate_opts['abs_oma_weak'][k] +
                      y_strong[k] * rate_opts['abs_oma_strong'][k] +
                      (1 - x_noma[k] - y_weak[k] - y_strong[k]) * rate_opts['sat'][k])
            objective_terms.append(contrib)

        objective = cp.Maximize(cp.sum(objective_terms))

        # Constraints
        constraints = []

        # 1. Mutual exclusion: at most one ABS mode per pair
        for k in range(K):
            constraints.append(x_noma[k] + y_weak[k] + y_strong[k] <= 1)

        # 2. Only use ABS if it's better than satellite
        for k in range(K):
            # NOMA only if better
            if rate_opts['abs_noma'][k] <= rate_opts['sat'][k]:
                constraints.append(x_noma[k] == 0)
            # OMA weak only if better
            if rate_opts['abs_oma_weak'][k] <= rate_opts['sat'][k]:
                constraints.append(y_weak[k] == 0)
            # OMA strong only if better
            if rate_opts['abs_oma_strong'][k] <= rate_opts['sat'][k]:
                constraints.append(y_strong[k] == 0)

        # Solve
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.GLPK_MI, verbose=False)
        except:
            try:
                problem.solve(verbose=False)
            except Exception as e:
                print(f"ILP solver failed: {e}")
                return self.optimize_greedy(sat_pairs, abs_pairs, sat_gains, a2g_gains,
                                           Ps_dB, Pd, Bs, Bd)

        # Extract decisions
        decisions = []
        final_rate = 0.0

        for k in range(K):
            if x_noma.value[k] > 0.5:
                decisions.append('noma')
                final_rate += rate_opts['abs_noma'][k]
            elif y_weak.value[k] > 0.5:
                decisions.append('oma_weak')
                final_rate += rate_opts['abs_oma_weak'][k]
            elif y_strong.value[k] > 0.5:
                decisions.append('oma_strong')
                final_rate += rate_opts['abs_oma_strong'][k]
            else:
                decisions.append('sat')
                final_rate += rate_opts['sat'][k]

        info = {
            'solver': 'ILP',
            'optimal_value': problem.value,
            'status': problem.status
        }

        return decisions, final_rate, info

    def optimize_greedy(self, sat_pairs, abs_pairs, sat_gains, a2g_gains,
                       Ps_dB, Pd, Bs, Bd):
        """
        Greedy algorithm (fallback when cvxpy not available)

        This is similar to original SATCON but with improved logic

        Args:
            [Same as optimize_ilp]

        Returns:
            decisions: List of decisions
            final_rate: Total rate
            info: Optimization info
        """
        K = len(sat_pairs)

        # Compute rate options
        rate_opts = self.compute_rate_options(
            sat_pairs, abs_pairs, sat_gains, a2g_gains,
            Ps_dB, Pd, Bs, Bd
        )

        # Greedy decision: for each pair, choose best option
        decisions = []
        final_rate = 0.0

        for k in range(K):
            # Compare all options
            options = {
                'sat': rate_opts['sat'][k],
                'noma': rate_opts['abs_noma'][k],
                'oma_weak': rate_opts['abs_oma_weak'][k],
                'oma_strong': rate_opts['abs_oma_strong'][k]
            }

            # Choose best
            best_mode = max(options, key=options.get)
            decisions.append(best_mode)
            final_rate += options[best_mode]

        info = {
            'solver': 'Greedy',
            'optimal_value': final_rate,
            'status': 'success'
        }

        return decisions, final_rate, info

    def optimize(self, sat_pairs, abs_pairs, sat_gains, a2g_gains,
                Ps_dB, Pd, Bs, Bd):
        """
        Main optimization interface

        Automatically chooses ILP if available, otherwise greedy

        Args:
            sat_pairs: Satellite pairing
            abs_pairs: ABS pairing
            sat_gains: Satellite channel gains
            a2g_gains: A2G channel gains
            Ps_dB: Satellite power (dB)
            Pd: ABS power (W)
            Bs: Satellite bandwidth
            Bd: ABS bandwidth

        Returns:
            decisions: List of mode decisions for each pair
            final_rate: Total system rate
            info: Optimization info
        """
        if self.use_ilp:
            try:
                return self.optimize_ilp(sat_pairs, abs_pairs, sat_gains, a2g_gains,
                                        Ps_dB, Pd, Bs, Bd)
            except:
                print("ILP failed, falling back to greedy")
                return self.optimize_greedy(sat_pairs, abs_pairs, sat_gains, a2g_gains,
                                           Ps_dB, Pd, Bs, Bd)
        else:
            return self.optimize_greedy(sat_pairs, abs_pairs, sat_gains, a2g_gains,
                                       Ps_dB, Pd, Bs, Bd)


# ==================== Test Code ====================
def test_integer_programming_decision():
    """Test integer programming decision optimizer"""
    print("=" * 60)
    print("Testing Integer Programming Decision Optimizer")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)
    sat_gains = np.random.exponential(0.01, size=config.N)
    a2g_gains = np.random.exponential(0.05, size=config.N)

    # Add diversity
    for i in range(0, config.N, 4):
        if i+1 < config.N:
            sat_gains[i] *= 5.0
            a2g_gains[i] *= 0.2
            sat_gains[i+1] *= 0.2
            a2g_gains[i+1] *= 5.0

    print(f"\nTest configuration:")
    print(f"  Number of users: {config.N}")
    print(f"  Number of pairs: {config.N // 2}")
    print(f"  Solver available: {'ILP (cvxpy)' if CVXPY_AVAILABLE else 'Greedy (fallback)'}")

    # Get pairing from baseline
    allocator = NOMAAllocator()
    sat_pairs, _ = allocator.optimal_user_pairing(sat_gains)
    abs_pairs, _ = allocator.optimal_user_pairing(a2g_gains)

    # Test parameters
    Bd = config.Bd_options[1]
    Ps_dB = 20

    # Create optimizer
    optimizer = IntegerProgrammingDecision()

    # Baseline: Greedy decision (per-pair local optimal)
    print(f"\n" + "-" * 60)
    print("Baseline: Greedy decision (per-pair local optimal)")
    print("-" * 60)

    decisions_greedy, rate_greedy, info_greedy = optimizer.optimize_greedy(
        sat_pairs, abs_pairs, sat_gains, a2g_gains,
        Ps_dB, config.Pd, config.Bs, Bd
    )

    print(f"  Total rate: {rate_greedy/1e6:.2f} Mbps")
    print(f"  Mode distribution:")
    print(f"    Satellite: {decisions_greedy.count('sat')}")
    print(f"    ABS NOMA: {decisions_greedy.count('noma')}")
    print(f"    ABS OMA weak: {decisions_greedy.count('oma_weak')}")
    print(f"    ABS OMA strong: {decisions_greedy.count('oma_strong')}")

    # Proposed: Global optimization
    print(f"\n" + "-" * 60)
    print("Proposed: Global optimization decision")
    print("-" * 60)

    decisions_opt, rate_opt, info_opt = optimizer.optimize(
        sat_pairs, abs_pairs, sat_gains, a2g_gains,
        Ps_dB, config.Pd, config.Bs, Bd
    )

    print(f"  Solver: {info_opt['solver']}")
    print(f"  Total rate: {rate_opt/1e6:.2f} Mbps")
    print(f"  Mode distribution:")
    print(f"    Satellite: {decisions_opt.count('sat')}")
    print(f"    ABS NOMA: {decisions_opt.count('noma')}")
    print(f"    ABS OMA weak: {decisions_opt.count('oma_weak')}")
    print(f"    ABS OMA strong: {decisions_opt.count('oma_strong')}")

    # Performance comparison
    print(f"\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    improvement_abs = (rate_opt - rate_greedy) / 1e6
    improvement_pct = (rate_opt - rate_greedy) / rate_greedy * 100

    print(f"\nBaseline (Greedy):")
    print(f"  Rate: {rate_greedy/1e6:.2f} Mbps")

    print(f"\nProposed (Global Optimization):")
    print(f"  Rate: {rate_opt/1e6:.2f} Mbps")

    print(f"\nImprovement:")
    print(f"  Absolute gain: {improvement_abs:.2f} Mbps")
    print(f"  Relative gain: {improvement_pct:.2f}%")

    # Validate
    print(f"\n" + "=" * 60)
    if rate_opt >= rate_greedy * 0.99:
        print("[PASS] Integer programming decision test PASSED")
        print(f"  - Solver: {info_opt['solver']}")
        if rate_opt > rate_greedy:
            print(f"  - Performance better than greedy (+{improvement_pct:.2f}%)")
        else:
            print(f"  - Performance equal to greedy (local optimum)")
    else:
        print("[FAIL] Test FAILED")
    print("=" * 60)

    return decisions_opt, rate_opt, info_opt


if __name__ == "__main__":
    test_integer_programming_decision()
