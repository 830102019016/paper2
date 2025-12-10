# -*- coding: utf-8 -*-
"""
Gradient-based ABS Position Optimizer

Optimization objective: Maximize system sum rate
Method: L-BFGS-B gradient optimization

Comparison with k-means:
- Original method: min sum ||u_i - p||^2 (geometric distance)
- New method: max sum R_i(p, h) (system rate)

Author: SATCON Enhancement Project
Date: 2025-12-10
"""
import numpy as np
from scipy.optimize import minimize
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.a2g_channel import A2GChannel


class GradientPositionOptimizer:
    """
    Gradient-based Position Optimizer for ABS

    Key innovation:
    - From geometric optimization (k-means) to performance-driven optimization
    - Joint optimization of 3D position [x, y, h]
    - Considers channel physical characteristics (path loss, LOS probability)
    """

    def __init__(self, config_obj):
        """
        Initialize optimizer

        Args:
            config_obj: System configuration object
        """
        self.config = config_obj
        self.a2g_channel = A2GChannel()

        # Optimization bounds
        self.bounds_xy = (-config_obj.coverage_radius, config_obj.coverage_radius)
        self.bounds_h = (50, 500)  # Height range: 50m - 500m

    def compute_total_rate(self, pos_3d, user_positions, a2g_gains_fading):
        """
        Compute system total rate at given ABS position

        This is the objective function for optimization

        Args:
            pos_3d: [x, y, h] ABS 3D position
            user_positions: [N, 2] User position array
            a2g_gains_fading: [N] A2G small-scale fading (pre-generated, fixed)

        Returns:
            total_rate: System total rate (bps)

        Physical meaning:
            - Place ABS at position (x, y, h)
            - Compute achievable rate for each user
            - Sum to get total rate
        """
        x, y, h = pos_3d
        total_rate = 0.0

        for i, u in enumerate(user_positions):
            # Compute 2D distance (horizontal distance)
            r = np.sqrt((x - u[0])**2 + (y - u[1])**2)

            # Use default bandwidth (1.2 MHz)
            Bd = self.config.Bd_options[1]  # 1.2 MHz

            # Compute A2G channel gain (including path loss and small-scale fading)
            gamma = self.a2g_channel.compute_channel_gain(
                h=h,
                r=r,
                fading_gain=a2g_gains_fading[i],
                G_tx_dB=self.config.Gd_t_dB,
                G_rx_dB=self.config.Gd_r_dB,
                noise_power=self.config.get_abs_noise_power(Bd)
            )

            # Compute OMA rate (simplified version, as proxy for position optimization)
            # Note: Actual system uses NOMA, but use OMA as proxy in position optimization
            rate = Bd * np.log2(1 + self.config.Pd * gamma)
            total_rate += rate

        return total_rate

    def optimize(self, user_positions, a2g_gains_fading, initial_guess=None):
        """
        Optimize ABS position

        Uses L-BFGS-B algorithm for gradient-based optimization

        Args:
            user_positions: [N, 2] User positions
            a2g_gains_fading: [N] A2G small-scale fading
            initial_guess: [3] Initial position guess (optional)
                          If not provided, use user centroid + default height

        Returns:
            optimal_position: [3] Optimal ABS position [x, y, h]
            optimization_info: dict Optimization information
                - success: Whether converged successfully
                - iterations: Number of iterations
                - final_rate: Optimal rate
                - initial_rate: Initial rate
                - improvement: Absolute improvement
        """
        # Initial guess: user centroid + default height 100m
        if initial_guess is None:
            center_xy = np.mean(user_positions[:, :2], axis=0)
            initial_guess = [center_xy[0], center_xy[1], 100.0]

        # Define objective function (minimize negative rate = maximize rate)
        def objective(pos_3d):
            rate = self.compute_total_rate(pos_3d, user_positions, a2g_gains_fading)
            return -rate  # Negative for minimization

        # Boundary constraints
        bounds = [
            (self.bounds_xy[0], self.bounds_xy[1]),  # x range
            (self.bounds_xy[0], self.bounds_xy[1]),  # y range
            (self.bounds_h[0], self.bounds_h[1])     # h range
        ]

        # L-BFGS-B optimization (quasi-Newton method with bounds)
        result = minimize(
            objective,
            x0=initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 100,      # Maximum iterations
                'ftol': 1e-6         # Function value convergence tolerance
            }
        )

        # Collect optimization info
        optimization_info = {
            'success': result.success,
            'iterations': result.nit,
            'final_rate': -result.fun,  # Restore positive value
            'initial_rate': -objective(initial_guess),
            'improvement': (-result.fun) - (-objective(initial_guess)),
            'message': result.message
        }

        return result.x, optimization_info


# ==================== Test Code ====================
def test_gradient_optimizer():
    """Test gradient position optimizer"""
    from src.user_distribution import UserDistribution
    from src.abs_placement import ABSPlacement

    print("=" * 60)
    print("Testing Gradient Position Optimizer")
    print("=" * 60)

    # Generate test users (32 users, uniformly distributed in 500m radius circle)
    dist = UserDistribution(config.N, config.coverage_radius, seed=42)
    user_positions = dist.generate_uniform_circle()

    print(f"\nUser distribution:")
    print(f"  Number of users: {config.N}")
    print(f"  Coverage radius: {config.coverage_radius}m")
    print(f"  Centroid: ({np.mean(user_positions[:, 0]):.2f}, {np.mean(user_positions[:, 1]):.2f})")

    # Generate A2G fading (fixed for fair comparison)
    a2g_channel = A2GChannel()
    a2g_fading = a2g_channel.generate_fading(config.N, seed=42)

    # Create optimizer
    optimizer = GradientPositionOptimizer(config)

    # Optimize
    print(f"\nStarting gradient optimization...")
    optimal_pos, info = optimizer.optimize(user_positions, a2g_fading)

    print(f"\nGradient optimization results:")
    print(f"  Optimal position: ({optimal_pos[0]:.2f}, {optimal_pos[1]:.2f}, {optimal_pos[2]:.0f}m)")
    print(f"  Convergence: {'Success' if info['success'] else 'Failed'}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Initial rate: {info['initial_rate']/1e6:.2f} Mbps")
    print(f"  Optimal rate: {info['final_rate']/1e6:.2f} Mbps")
    print(f"  Improvement: {info['improvement']/1e6:.2f} Mbps (+{info['improvement']/info['initial_rate']*100:.1f}%)")

    # Compare with k-means
    print(f"\n" + "=" * 60)
    print("Comparison with Original k-means")
    print("=" * 60)

    kmeans_placement = ABSPlacement()
    kmeans_xy, method, cost = kmeans_placement.optimize_xy_position(user_positions)
    kmeans_h, _ = kmeans_placement.optimize_height(kmeans_xy, user_positions, a2g_channel)
    kmeans_pos = [kmeans_xy[0], kmeans_xy[1], kmeans_h]

    # Compute rate at k-means position
    kmeans_rate = optimizer.compute_total_rate(kmeans_pos, user_positions, a2g_fading)

    print(f"\nk-means results:")
    print(f"  Position: ({kmeans_pos[0]:.2f}, {kmeans_pos[1]:.2f}, {kmeans_pos[2]:.0f}m)")
    print(f"  Rate: {kmeans_rate/1e6:.2f} Mbps")

    print(f"\nPerformance comparison:")
    print(f"  Gradient optimization vs k-means:")
    print(f"    Absolute gain: {(info['final_rate'] - kmeans_rate)/1e6:.2f} Mbps")
    print(f"    Relative gain: {(info['final_rate'] - kmeans_rate)/kmeans_rate*100:.1f}%")

    # Validate success criteria
    print(f"\n" + "=" * 60)
    if info['success'] and info['final_rate'] > kmeans_rate:
        print("[PASS] Gradient position optimizer test PASSED")
        print(f"  - Optimization converged successfully")
        print(f"  - Performance better than k-means (+{(info['final_rate'] - kmeans_rate)/kmeans_rate*100:.1f}%)")
    else:
        print("[FAIL] Test FAILED")
        if not info['success']:
            print(f"  - Optimization did not converge: {info['message']}")
        if info['final_rate'] <= kmeans_rate:
            print(f"  - Performance not better than k-means")
    print("=" * 60)

    return optimal_pos, info, kmeans_pos, kmeans_rate


if __name__ == "__main__":
    test_gradient_optimizer()
