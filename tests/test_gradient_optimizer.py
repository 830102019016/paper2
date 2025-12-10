# -*- coding: utf-8 -*-
"""
Unit tests for Gradient Position Optimizer

Tests:
1. Optimizer initialization
2. Rate computation correctness
3. Optimization convergence
4. Performance improvement over k-means
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src_enhanced.gradient_position_optimizer import GradientPositionOptimizer
from src.user_distribution import UserDistribution
from src.a2g_channel import A2GChannel
from src.abs_placement import ABSPlacement


class TestGradientPositionOptimizer:
    """Test suite for GradientPositionOptimizer"""

    @pytest.fixture
    def setup(self):
        """Setup test environment"""
        # Generate test users
        dist = UserDistribution(config.N, config.coverage_radius, seed=42)
        user_positions = dist.generate_uniform_circle()

        # Generate A2G fading
        a2g_channel = A2GChannel()
        a2g_fading = a2g_channel.generate_fading(config.N, seed=42)

        # Create optimizer
        optimizer = GradientPositionOptimizer(config)

        return {
            'optimizer': optimizer,
            'user_positions': user_positions,
            'a2g_fading': a2g_fading,
            'a2g_channel': a2g_channel
        }

    def test_initialization(self, setup):
        """Test optimizer initialization"""
        optimizer = setup['optimizer']

        assert optimizer.config is not None
        assert optimizer.a2g_channel is not None
        assert optimizer.bounds_xy == (-config.coverage_radius, config.coverage_radius)
        assert optimizer.bounds_h == (50, 500)

    def test_compute_total_rate(self, setup):
        """Test total rate computation"""
        optimizer = setup['optimizer']
        user_positions = setup['user_positions']
        a2g_fading = setup['a2g_fading']

        # Test at user centroid with height 100m
        center = np.mean(user_positions[:, :2], axis=0)
        pos_3d = [center[0], center[1], 100.0]

        rate = optimizer.compute_total_rate(pos_3d, user_positions, a2g_fading)

        # Rate should be positive
        assert rate > 0
        # Rate should be reasonable (between 100 Mbps and 2 Gbps)
        assert 100e6 < rate < 2e9

    def test_optimization_convergence(self, setup):
        """Test that optimization converges"""
        optimizer = setup['optimizer']
        user_positions = setup['user_positions']
        a2g_fading = setup['a2g_fading']

        optimal_pos, info = optimizer.optimize(user_positions, a2g_fading)

        # Should converge successfully
        assert info['success'] == True
        # Should have reasonable number of iterations (< 100)
        assert info['iterations'] < 100
        # Should improve from initial guess
        assert info['improvement'] > 0
        # Optimal position should be within bounds
        assert -config.coverage_radius <= optimal_pos[0] <= config.coverage_radius
        assert -config.coverage_radius <= optimal_pos[1] <= config.coverage_radius
        assert 50 <= optimal_pos[2] <= 500

    def test_performance_vs_kmeans(self, setup):
        """Test that gradient optimization outperforms k-means"""
        optimizer = setup['optimizer']
        user_positions = setup['user_positions']
        a2g_fading = setup['a2g_fading']
        a2g_channel = setup['a2g_channel']

        # Gradient optimization
        optimal_pos, info = optimizer.optimize(user_positions, a2g_fading)
        gradient_rate = info['final_rate']

        # k-means baseline
        kmeans_placement = ABSPlacement()
        kmeans_xy, _, _ = kmeans_placement.optimize_xy_position(user_positions)
        kmeans_h, _ = kmeans_placement.optimize_height(kmeans_xy, user_positions, a2g_channel)
        kmeans_pos = [kmeans_xy[0], kmeans_xy[1], kmeans_h]
        kmeans_rate = optimizer.compute_total_rate(kmeans_pos, user_positions, a2g_fading)

        # Gradient should be at least as good as k-means
        # (Allow small tolerance for numerical errors)
        assert gradient_rate >= kmeans_rate * 0.99

        # Compute relative improvement
        improvement_pct = (gradient_rate - kmeans_rate) / kmeans_rate * 100

        print(f"\nPerformance comparison:")
        print(f"  Gradient rate: {gradient_rate/1e6:.2f} Mbps")
        print(f"  k-means rate: {kmeans_rate/1e6:.2f} Mbps")
        print(f"  Improvement: {improvement_pct:.2f}%")

    def test_different_initial_guesses(self, setup):
        """Test optimization with different initial guesses"""
        optimizer = setup['optimizer']
        user_positions = setup['user_positions']
        a2g_fading = setup['a2g_fading']

        # Test multiple initial guesses
        initial_guesses = [
            [0, 0, 100],        # Center
            [100, 100, 150],    # Upper right
            [-100, -100, 200],  # Lower left
            [50, -50, 250]      # Random
        ]

        results = []
        for init_guess in initial_guesses:
            optimal_pos, info = optimizer.optimize(user_positions, a2g_fading,
                                                  initial_guess=init_guess)
            results.append(info['final_rate'])

        # All should converge to similar rates (within 5%)
        max_rate = max(results)
        min_rate = min(results)
        assert (max_rate - min_rate) / max_rate < 0.05


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '-s'])
