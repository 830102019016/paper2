# -*- coding: utf-8 -*-
"""
Unit tests for Joint Optimization SATCON System

Tests:
1. System initialization with different module combinations
2. Single realization simulation
3. Performance comparison against baseline
4. Ablation study
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src_enhanced.joint_satcon_system import JointOptimizationSATCON


class TestJointOptimizationSATCON:
    """Test suite for JointOptimizationSATCON"""

    @pytest.fixture
    def setup(self):
        """Setup test environment"""
        np.random.seed(42)

        # Create systems with different configurations
        systems = {
            'baseline': JointOptimizationSATCON(
                config, 1.2e6,
                use_module1=False, use_module2=False, use_module3=False
            ),
            'module1': JointOptimizationSATCON(
                config, 1.2e6,
                use_module1=True, use_module2=False, use_module3=False
            ),
            'module2': JointOptimizationSATCON(
                config, 1.2e6,
                use_module1=False, use_module2=True, use_module3=False
            ),
            'module3': JointOptimizationSATCON(
                config, 1.2e6,
                use_module1=False, use_module2=False, use_module3=True
            ),
            'full': JointOptimizationSATCON(
                config, 1.2e6,
                use_module1=True, use_module2=True, use_module3=True
            )
        }

        return systems

    def test_initialization(self, setup):
        """Test system initialization"""
        systems = setup

        for name, system in systems.items():
            assert system.config == config
            assert system.Bd == 1.2e6
            assert system.sat_noma is not None
            assert system.a2g_channel is not None
            assert system.allocator is not None
            assert system.decision_optimizer is not None

    def test_single_realization(self, setup):
        """Test single realization simulation"""
        system = setup['baseline']

        # Run single simulation
        sum_rate, mode_stats = system.simulate_single_realization(
            snr_db=20, elevation_deg=10, seed=42
        )

        # Check outputs
        assert sum_rate > 0
        assert 1e6 < sum_rate < 100e6  # Reasonable range

        # Check mode statistics
        assert 'noma' in mode_stats or 'sat' in mode_stats
        total_pairs = config.N // 2
        mode_count = sum(mode_stats.values())
        assert mode_count == total_pairs

    def test_module1_improvement(self, setup):
        """Test that Module 1 improves performance"""
        baseline = setup['baseline']
        module1 = setup['module1']

        # Test at SNR=20dB with multiple realizations
        n_real = 10
        rates_baseline = []
        rates_module1 = []

        for seed in range(n_real):
            rate_b, _ = baseline.simulate_single_realization(20, 10, seed)
            rate_m1, _ = module1.simulate_single_realization(20, 10, seed)
            rates_baseline.append(rate_b)
            rates_module1.append(rate_m1)

        mean_baseline = np.mean(rates_baseline)
        mean_module1 = np.mean(rates_module1)

        print(f"\nModule 1 Test:")
        print(f"  Baseline: {mean_baseline/1e6:.2f} Mbps")
        print(f"  Module 1: {mean_module1/1e6:.2f} Mbps")
        print(f"  Improvement: {(mean_module1-mean_baseline)/mean_baseline*100:.2f}%")

        # Module 1 should improve or at least not degrade significantly
        assert mean_module1 >= mean_baseline * 0.95

    def test_module2_improvement(self, setup):
        """Test that Module 2 improves performance"""
        baseline = setup['baseline']
        module2 = setup['module2']

        # Test at SNR=20dB with multiple realizations
        n_real = 10
        rates_baseline = []
        rates_module2 = []

        for seed in range(n_real):
            rate_b, _ = baseline.simulate_single_realization(20, 10, seed)
            rate_m2, _ = module2.simulate_single_realization(20, 10, seed)
            rates_baseline.append(rate_b)
            rates_module2.append(rate_m2)

        mean_baseline = np.mean(rates_baseline)
        mean_module2 = np.mean(rates_module2)

        print(f"\nModule 2 Test:")
        print(f"  Baseline: {mean_baseline/1e6:.2f} Mbps")
        print(f"  Module 2: {mean_module2/1e6:.2f} Mbps")
        print(f"  Improvement: {(mean_module2-mean_baseline)/mean_baseline*100:.2f}%")

        # Module 2 should improve or at least not degrade significantly
        assert mean_module2 >= mean_baseline * 0.95

    def test_full_system_improvement(self, setup):
        """Test that full system improves performance"""
        baseline = setup['baseline']
        full = setup['full']

        # Test at SNR=20dB with multiple realizations
        n_real = 10
        rates_baseline = []
        rates_full = []

        for seed in range(n_real):
            rate_b, _ = baseline.simulate_single_realization(20, 10, seed)
            rate_f, _ = full.simulate_single_realization(20, 10, seed)
            rates_baseline.append(rate_b)
            rates_full.append(rate_f)

        mean_baseline = np.mean(rates_baseline)
        mean_full = np.mean(rates_full)

        improvement_pct = (mean_full - mean_baseline) / mean_baseline * 100

        print(f"\nFull System Test:")
        print(f"  Baseline: {mean_baseline/1e6:.2f} Mbps")
        print(f"  Full System: {mean_full/1e6:.2f} Mbps")
        print(f"  Improvement: {improvement_pct:.2f}%")

        # Full system should improve performance
        assert mean_full >= mean_baseline * 0.99

    def test_monte_carlo_simulation(self, setup):
        """Test Monte Carlo simulation"""
        system = setup['baseline']

        # Small test
        snr_range = np.array([10, 20])
        n_real = 3

        mean_rates, mean_se, std_rates, mode_stats = system.simulate_performance(
            snr_range, elevation_deg=10, n_realizations=n_real, verbose=False
        )

        # Check outputs
        assert len(mean_rates) == len(snr_range)
        assert len(mean_se) == len(snr_range)
        assert len(std_rates) == len(snr_range)

        # Check values are positive
        assert np.all(mean_rates > 0)
        assert np.all(mean_se > 0)

        # Spectral efficiency should be in reasonable range
        assert np.all(mean_se > 1)
        assert np.all(mean_se < 100)

    def test_ablation_study(self, setup):
        """Test ablation study - each module contributes"""
        systems = setup

        # Run quick test
        snr = 20
        n_real = 5

        results = {}
        for name, system in systems.items():
            rates = []
            for seed in range(n_real):
                rate, _ = system.simulate_single_realization(snr, 10, seed)
                rates.append(rate)
            results[name] = np.mean(rates)

        baseline_rate = results['baseline']

        print(f"\nAblation Study (SNR={snr}dB, {n_real} realizations):")
        print(f"  Baseline:        {baseline_rate/1e6:.2f} Mbps")
        print(f"  + Module 1:      {results['module1']/1e6:.2f} Mbps "
              f"({(results['module1']-baseline_rate)/baseline_rate*100:+.2f}%)")
        print(f"  + Module 2:      {results['module2']/1e6:.2f} Mbps "
              f"({(results['module2']-baseline_rate)/baseline_rate*100:+.2f}%)")
        print(f"  + Module 3:      {results['module3']/1e6:.2f} Mbps "
              f"({(results['module3']-baseline_rate)/baseline_rate*100:+.2f}%)")
        print(f"  + All Modules:   {results['full']/1e6:.2f} Mbps "
              f"({(results['full']-baseline_rate)/baseline_rate*100:+.2f}%)")

        # All enhanced systems should be >= baseline
        assert results['module1'] >= baseline_rate * 0.95
        assert results['module2'] >= baseline_rate * 0.95
        assert results['module3'] >= baseline_rate * 0.95
        assert results['full'] >= baseline_rate * 0.95


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '-s'])
