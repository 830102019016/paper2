# -*- coding: utf-8 -*-
"""
Unit tests for Joint Pairing Optimizer

Tests:
1. Optimizer initialization
2. Joint benefit computation
3. Optimization convergence
4. Performance improvement over independent pairing
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src_enhanced.joint_pairing_optimizer import JointPairingOptimizer
from src.power_allocation import NOMAAllocator


class TestJointPairingOptimizer:
    """Test suite for JointPairingOptimizer"""

    @pytest.fixture
    def setup(self):
        """Setup test environment"""
        np.random.seed(42)

        # Generate diverse channel gains
        sat_gains = np.random.exponential(0.01, size=config.N)
        a2g_gains = np.random.exponential(0.05, size=config.N)

        # Add diversity to create optimization opportunities
        for i in range(0, config.N, 4):
            if i+1 < config.N:
                sat_gains[i] *= 5.0
                a2g_gains[i] *= 0.2
                sat_gains[i+1] *= 0.2
                a2g_gains[i+1] *= 5.0

        optimizer = JointPairingOptimizer(config)

        return {
            'optimizer': optimizer,
            'sat_gains': sat_gains,
            'a2g_gains': a2g_gains
        }

    def test_initialization(self, setup):
        """Test optimizer initialization"""
        optimizer = setup['optimizer']

        assert optimizer.config is not None
        assert optimizer.allocator is not None
        assert isinstance(optimizer.allocator, NOMAAllocator)

    def test_compute_joint_benefit(self, setup):
        """Test joint benefit computation"""
        optimizer = setup['optimizer']
        sat_gains = setup['sat_gains']
        a2g_gains = setup['a2g_gains']

        # Get initial pairing
        allocator = NOMAAllocator()
        sat_pairs, _ = allocator.optimal_user_pairing(sat_gains)
        abs_pairs, _ = allocator.optimal_user_pairing(a2g_gains)

        # Compute benefit
        Bd = config.Bd_options[1]
        Ps_dB = 20
        benefit = optimizer.compute_joint_benefit(
            sat_pairs, abs_pairs,
            sat_gains, a2g_gains,
            Ps_dB, config.Pd, config.Bs, Bd
        )

        # Benefit should be positive
        assert benefit > 0
        # Benefit should be reasonable (1-100 Mbps)
        assert 1e6 < benefit < 100e6

    def test_optimization_convergence(self, setup):
        """Test that optimization converges"""
        optimizer = setup['optimizer']
        sat_gains = setup['sat_gains']
        a2g_gains = setup['a2g_gains']

        sat_pairs, abs_pairs, benefit, info = optimizer.optimize(sat_gains, a2g_gains)

        # Should converge
        assert info['iterations'] > 0
        assert info['iterations'] <= 50  # Max iterations
        # Should return valid pairing
        assert len(sat_pairs) == config.N // 2
        assert len(abs_pairs) == config.N // 2
        # Benefit should be positive
        assert benefit > 0

    def test_performance_vs_independent(self, setup):
        """Test that joint optimization outperforms independent pairing"""
        optimizer = setup['optimizer']
        sat_gains = setup['sat_gains']
        a2g_gains = setup['a2g_gains']

        # Baseline: Independent pairing
        allocator = NOMAAllocator()
        sat_pairs_old, _ = allocator.optimal_user_pairing(sat_gains)
        abs_pairs_old, _ = allocator.optimal_user_pairing(a2g_gains)

        Bd = config.Bd_options[1]
        Ps_dB = 20
        benefit_old = optimizer.compute_joint_benefit(
            sat_pairs_old, abs_pairs_old,
            sat_gains, a2g_gains,
            Ps_dB, config.Pd, config.Bs, Bd
        )

        # Joint optimization
        sat_pairs_new, abs_pairs_new, benefit_new, info = optimizer.optimize(
            sat_gains, a2g_gains
        )

        # Joint should be at least as good as independent
        # (Allow small numerical tolerance)
        assert benefit_new >= benefit_old * 0.99

        improvement_pct = (benefit_new - benefit_old) / benefit_old * 100

        print(f"\nPerformance comparison:")
        print(f"  Independent pairing: {benefit_old/1e6:.2f} Mbps")
        print(f"  Joint optimization: {benefit_new/1e6:.2f} Mbps")
        print(f"  Improvement: {improvement_pct:.2f}%")

    def test_different_scenarios(self, setup):
        """Test optimization in different scenarios"""
        optimizer = setup['optimizer']

        # Scenario 1: Uniform channels (no diversity)
        np.random.seed(100)
        sat_gains_uniform = np.random.exponential(0.01, size=config.N)
        a2g_gains_uniform = np.random.exponential(0.01, size=config.N)

        _, _, benefit_uniform, _ = optimizer.optimize(sat_gains_uniform, a2g_gains_uniform)

        # Scenario 2: Diverse channels
        sat_gains_diverse = setup['sat_gains']
        a2g_gains_diverse = setup['a2g_gains']

        _, _, benefit_diverse, _ = optimizer.optimize(sat_gains_diverse, a2g_gains_diverse)

        # Both should produce valid results
        assert benefit_uniform > 0
        assert benefit_diverse > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '-s'])
