# -*- coding: utf-8 -*-
"""
Unit tests for Integer Programming Decision Optimizer

Tests:
1. Optimizer initialization
2. Rate options computation
3. Greedy decision optimization
4. ILP optimization (if cvxpy available)
5. Performance comparison
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src_enhanced.integer_programming_decision import IntegerProgrammingDecision, CVXPY_AVAILABLE
from src.power_allocation import NOMAAllocator


class TestIntegerProgrammingDecision:
    """Test suite for IntegerProgrammingDecision"""

    @pytest.fixture
    def setup(self):
        """Setup test environment"""
        np.random.seed(42)

        # Generate diverse channel gains
        sat_gains = np.random.exponential(0.01, size=config.N)
        a2g_gains = np.random.exponential(0.05, size=config.N)

        # Add diversity
        for i in range(0, config.N, 4):
            if i+1 < config.N:
                sat_gains[i] *= 5.0
                a2g_gains[i] *= 0.2
                sat_gains[i+1] *= 0.2
                a2g_gains[i+1] *= 5.0

        # Get pairing
        allocator = NOMAAllocator()
        sat_pairs, _ = allocator.optimal_user_pairing(sat_gains)
        abs_pairs, _ = allocator.optimal_user_pairing(a2g_gains)

        optimizer = IntegerProgrammingDecision()

        return {
            'optimizer': optimizer,
            'sat_pairs': sat_pairs,
            'abs_pairs': abs_pairs,
            'sat_gains': sat_gains,
            'a2g_gains': a2g_gains
        }

    def test_initialization(self, setup):
        """Test optimizer initialization"""
        optimizer = setup['optimizer']

        assert optimizer.allocator is not None
        assert isinstance(optimizer.use_ilp, bool)

    def test_compute_rate_options(self, setup):
        """Test rate options computation"""
        optimizer = setup['optimizer']
        sat_pairs = setup['sat_pairs']
        abs_pairs = setup['abs_pairs']
        sat_gains = setup['sat_gains']
        a2g_gains = setup['a2g_gains']

        Bd = config.Bd_options[1]
        Ps_dB = 20

        rate_opts = optimizer.compute_rate_options(
            sat_pairs, abs_pairs, sat_gains, a2g_gains,
            Ps_dB, config.Pd, config.Bs, Bd
        )

        # Check structure
        assert 'sat' in rate_opts
        assert 'abs_noma' in rate_opts
        assert 'abs_oma_weak' in rate_opts
        assert 'abs_oma_strong' in rate_opts

        # Check dimensions
        K = len(sat_pairs)
        assert len(rate_opts['sat']) == K
        assert len(rate_opts['abs_noma']) == K

        # Check values are positive
        assert np.all(rate_opts['sat'] > 0)
        assert np.all(rate_opts['abs_noma'] > 0)

    def test_greedy_optimization(self, setup):
        """Test greedy optimization"""
        optimizer = setup['optimizer']
        sat_pairs = setup['sat_pairs']
        abs_pairs = setup['abs_pairs']
        sat_gains = setup['sat_gains']
        a2g_gains = setup['a2g_gains']

        Bd = config.Bd_options[1]
        Ps_dB = 20

        decisions, final_rate, info = optimizer.optimize_greedy(
            sat_pairs, abs_pairs, sat_gains, a2g_gains,
            Ps_dB, config.Pd, config.Bs, Bd
        )

        # Check decisions
        assert len(decisions) == len(sat_pairs)
        assert all(d in ['sat', 'noma', 'oma_weak', 'oma_strong'] for d in decisions)

        # Check rate
        assert final_rate > 0
        assert 1e6 < final_rate < 100e6

        # Check info
        assert info['solver'] == 'Greedy'
        assert info['status'] == 'success'

    @pytest.mark.skipif(not CVXPY_AVAILABLE, reason="cvxpy not installed")
    def test_ilp_optimization(self, setup):
        """Test ILP optimization (only if cvxpy available)"""
        optimizer = setup['optimizer']
        sat_pairs = setup['sat_pairs']
        abs_pairs = setup['abs_pairs']
        sat_gains = setup['sat_gains']
        a2g_gains = setup['a2g_gains']

        Bd = config.Bd_options[1]
        Ps_dB = 20

        decisions, final_rate, info = optimizer.optimize_ilp(
            sat_pairs, abs_pairs, sat_gains, a2g_gains,
            Ps_dB, config.Pd, config.Bs, Bd
        )

        # Check decisions
        assert len(decisions) == len(sat_pairs)
        assert all(d in ['sat', 'noma', 'oma_weak', 'oma_strong'] for d in decisions)

        # Check rate
        assert final_rate > 0

        # Check info
        assert info['solver'] == 'ILP'

    def test_optimization_interface(self, setup):
        """Test main optimization interface"""
        optimizer = setup['optimizer']
        sat_pairs = setup['sat_pairs']
        abs_pairs = setup['abs_pairs']
        sat_gains = setup['sat_gains']
        a2g_gains = setup['a2g_gains']

        Bd = config.Bd_options[1]
        Ps_dB = 20

        decisions, final_rate, info = optimizer.optimize(
            sat_pairs, abs_pairs, sat_gains, a2g_gains,
            Ps_dB, config.Pd, config.Bs, Bd
        )

        # Should work regardless of cvxpy availability
        assert len(decisions) == len(sat_pairs)
        assert final_rate > 0

    def test_performance_vs_greedy(self, setup):
        """Test that optimization is at least as good as greedy"""
        optimizer = setup['optimizer']
        sat_pairs = setup['sat_pairs']
        abs_pairs = setup['abs_pairs']
        sat_gains = setup['sat_gains']
        a2g_gains = setup['a2g_gains']

        Bd = config.Bd_options[1]
        Ps_dB = 20

        # Greedy
        _, rate_greedy, _ = optimizer.optimize_greedy(
            sat_pairs, abs_pairs, sat_gains, a2g_gains,
            Ps_dB, config.Pd, config.Bs, Bd
        )

        # Optimal
        _, rate_opt, info = optimizer.optimize(
            sat_pairs, abs_pairs, sat_gains, a2g_gains,
            Ps_dB, config.Pd, config.Bs, Bd
        )

        # Optimal should be >= greedy
        assert rate_opt >= rate_greedy * 0.99

        improvement_pct = (rate_opt - rate_greedy) / rate_greedy * 100

        print(f"\nPerformance comparison:")
        print(f"  Solver: {info['solver']}")
        print(f"  Greedy: {rate_greedy/1e6:.2f} Mbps")
        print(f"  Optimal: {rate_opt/1e6:.2f} Mbps")
        print(f"  Improvement: {improvement_pct:.2f}%")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '-s'])
