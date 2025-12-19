"""
对比ABS位置优化方法：我们的算法 vs Baseline

生成两个用户分布和ABS位置图：
1. 我们的算法：Greedy+KKT+L-BFGS-B
2. Baseline：启发式+uniform+k-means

目的：展示我们算法的完整优势（模式选择+资源分配+位置优化）
预期提升：约8%（基于蒙特卡洛仿真平均结果）
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.user_distribution import UserDistribution
from src.abs_placement import ABSPlacement
from src.a2g_channel import A2GChannel, S2AChannel
from src.noma_transmission import SatelliteNOMA
from src.power_allocation import NOMAAllocator
from src.optimizers.mode_selector import GreedySelector, HeuristicSelector
from src.optimizers.resource_allocator import KKTAllocator, UniformAllocator
from src.optimizers.position_optimizer import ContinuousPositionOptimizer


def compute_system_sum_rate(abs_position, user_positions,
                            mode_selector, s2a_allocator,
                            snr_linear, elevation_deg,
                            sat_rates, sat_pairs, sat_power_factors,
                            Bd, Pd, Nd, Nsd, config_obj):
    """
    计算给定ABS位置下的系统总速率

    返回:
        sum_rate: 系统总速率 (bps)
        final_rates: 各用户最终速率 (bps)
    """
    a2g_channel = A2GChannel()
    s2a_channel = S2AChannel()
    allocator = NOMAAllocator()

    # 1. 计算A2G信道增益
    x_abs, y_abs, h_abs = abs_position
    distances_2d = np.linalg.norm(
        user_positions[:, :2] - np.array([x_abs, y_abs]), axis=1
    )

    # 生成固定的小尺度衰落
    np.random.seed(config_obj.random_seed)
    fading_gains = a2g_channel.generate_fading(len(user_positions),
                                                seed=config_obj.random_seed)

    channel_gains_a2g = np.array([
        a2g_channel.compute_channel_gain(
            h_abs, r, fading,
            config_obj.Gd_t_dB, config_obj.Gd_r_dB, Nd
        )
        for r, fading in zip(distances_2d, fading_gains)
    ])

    # 2. 计算S2A信道增益
    elevation_rad = np.deg2rad(elevation_deg)
    d_s2a = (config_obj.satellite_altitude - h_abs) / np.sin(elevation_rad)
    h_s2a = s2a_channel.compute_channel_gain(
        distance=d_s2a,
        fading_gain=1.0,
        G_tx_dB=config_obj.Gs_t_dB,
        G_rx_dB=config_obj.Gsd_r_dB,
        noise_power=Nsd
    )

    # 3. 计算A2G速率
    K = len(sat_pairs)
    bandwidth_per_pair = Bd / K

    # NOMA A2G速率
    a2g_rates_noma = np.zeros(2*K)
    for k in range(K):
        weak_idx, strong_idx = sat_pairs[k]
        gamma_weak = channel_gains_a2g[weak_idx]
        gamma_strong = channel_gains_a2g[strong_idx]

        if gamma_weak > gamma_strong:
            weak_idx, strong_idx = strong_idx, weak_idx
            gamma_weak, gamma_strong = gamma_strong, gamma_weak

        beta_strong, beta_weak = allocator.compute_power_factors(
            gamma_strong, gamma_weak, Pd
        )

        a2g_rates_noma[strong_idx] = bandwidth_per_pair * np.log2(
            1 + beta_strong * Pd * gamma_strong
        )
        a2g_rates_noma[weak_idx] = bandwidth_per_pair * np.log2(
            1 + beta_weak * Pd * gamma_weak /
            (beta_strong * Pd * gamma_weak + 1)
        )

    # OMA A2G速率
    a2g_rates_oma = bandwidth_per_pair * np.log2(
        1 + Pd * channel_gains_a2g
    )

    # 4. S2A资源分配
    _, modes_temp = mode_selector.select_modes(
        sat_rates, a2g_rates_noma, a2g_rates_oma, sat_pairs
    )

    b_allocated = s2a_allocator.allocate_bandwidth(
        sat_pairs, modes_temp, a2g_rates_noma, a2g_rates_oma,
        snr_linear, h_s2a, sat_power_factors
    )

    s2a_rates = s2a_allocator.compute_s2a_rates(
        sat_pairs, b_allocated, snr_linear, h_s2a, sat_power_factors
    )

    # 5. 计算最终速率
    abs_noma_rates = np.minimum(a2g_rates_noma, s2a_rates)
    abs_oma_rates = np.minimum(a2g_rates_oma, s2a_rates)

    final_rates, _ = mode_selector.select_modes(
        sat_rates, abs_noma_rates, abs_oma_rates, sat_pairs
    )

    sum_rate = np.sum(final_rates)

    return sum_rate, final_rates


def plot_user_distribution_with_abs(user_positions, abs_position,
                                     sum_rate, method_name,
                                     snr_db, elevation_deg, save_path):
    """
    绘制用户分布和ABS位置图

    参数:
        user_positions: 用户位置 [N, 3]
        abs_position: ABS位置 [x, y, h]
        sum_rate: 系统总速率 (bps)
        method_name: 方法名称
        snr_db: SNR (dB)
        elevation_deg: 仰角 (度)
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # 1. 绘制覆盖区域
    coverage_radius = config.coverage_radius
    circle = plt.Circle((0, 0), coverage_radius,
                       color='lightgray', fill=True, alpha=0.2,
                       label=f'Coverage Area (R={coverage_radius}m)')
    ax.add_patch(circle)

    # 2. 绘制用户位置
    ax.scatter(user_positions[:, 0], user_positions[:, 1],
              c='blue', marker='o', s=80, alpha=0.6,
              label=f'Users (N={len(user_positions)})', zorder=3)

    # 3. 绘制ABS位置
    ax.scatter(abs_position[0], abs_position[1],
              c='red', marker='^', s=500,
              edgecolors='black', linewidths=2.5,
              label=f'ABS (h={abs_position[2]:.0f}m)', zorder=5)

    # 4. 添加ABS位置坐标文本
    ax.text(abs_position[0], abs_position[1] - 80,
           f'({abs_position[0]:.1f}, {abs_position[1]:.1f})',
           fontsize=11, ha='center', va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    # 5. 图表设置
    ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold')

    # 标题包含方法名称
    title = f'{method_name}\nUser Distribution and ABS Position'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    # 添加参数信息文本框
    info_text = (f'SNR: {snr_db} dB\n'
                f'Elevation: {elevation_deg}°\n'
                f'Sum Rate: {sum_rate/1e6:.2f} Mbps\n'
                f'ABS Height: {abs_position[2]:.0f} m')

    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes,
           fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')
    ax.axis('equal')
    ax.set_xlim([-coverage_radius*1.2, coverage_radius*1.2])
    ax.set_ylim([-coverage_radius*1.2, coverage_radius*1.2])

    plt.tight_layout()

    # 保存图表
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 图表已保存: {save_path}")

    return fig, ax


def main():
    """主函数：生成两个对比图"""

    print("=" * 80)
    print("生成ABS位置优化对比图")
    print("=" * 80)

    # ==================== 配置参数 ====================
    snr_db = 20
    elevation_deg = 10
    abs_bandwidth = 1.2e6

    print(f"\n系统配置:")
    print(f"  用户数: {config.N}")
    print(f"  覆盖半径: {config.coverage_radius} m")
    print(f"  ABS带宽: {abs_bandwidth/1e6:.1f} MHz")
    print(f"  SNR: {snr_db} dB")
    print(f"  仰角: {elevation_deg}°")

    # ==================== 生成用户分布（两个图使用相同分布）====================
    print(f"\n{'='*80}")
    print(f"步骤1: 生成用户分布（固定随机种子确保一致性）")
    print(f"{'='*80}")

    dist = UserDistribution(config.N, config.coverage_radius,
                           seed=config.random_seed)
    user_positions = dist.generate_uniform_circle()

    print(f"[OK] 生成 {config.N} 个用户位置")
    print(f"  X范围: [{np.min(user_positions[:, 0]):.1f}, {np.max(user_positions[:, 0]):.1f}] m")
    print(f"  Y范围: [{np.min(user_positions[:, 1]):.1f}, {np.max(user_positions[:, 1]):.1f}] m")

    # ==================== 计算卫星信息（两个方法共用）====================
    print(f"\n{'='*80}")
    print(f"步骤2: 计算卫星传输参数")
    print(f"{'='*80}")

    sat_noma = SatelliteNOMA(config)
    sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)

    allocator = NOMAAllocator()
    sat_pairs, _ = allocator.optimal_user_pairing(sat_channel_gains)

    snr_linear = 10 ** (snr_db / 10)
    sat_rates, _, sat_power_factors = sat_noma.compute_achievable_rates(
        sat_channel_gains, snr_linear
    )

    print(f"[OK] 卫星配对数: {len(sat_pairs)}")
    print(f"[OK] 卫星平均速率: {np.mean(sat_rates)/1e6:.2f} Mbps")

    Pd = config.Pd
    Nd = config.get_abs_noise_power(abs_bandwidth)
    Nsd = config.get_s2a_noise_power(config.Bs)

    # ==================== 方法1: Baseline (启发式+uniform+k-means) ====================
    print(f"\n{'='*80}")
    print(f"步骤3: Baseline方法 (启发式 + uniform + k-means)")
    print(f"{'='*80}")

    # Baseline使用启发式模式选择器和均匀资源分配器
    baseline_mode_selector = HeuristicSelector()
    baseline_s2a_allocator = UniformAllocator(config)

    a2g_channel = A2GChannel()
    placement = ABSPlacement(
        abs_height_range=(config.abs_height_min, config.abs_height_max),
        height_step=config.abs_height_step
    )

    abs_position_baseline, baseline_info = placement.optimize_position_complete(
        user_positions, a2g_channel
    )

    print(f"[OK] k-means优化完成")
    print(f"  ABS位置: ({abs_position_baseline[0]:.2f}, "
          f"{abs_position_baseline[1]:.2f}, {abs_position_baseline[2]:.0f})")
    print(f"  优化方法: {baseline_info['method']}")

    # 计算baseline的系统总速率（使用启发式+uniform）
    baseline_sum_rate, baseline_rates = compute_system_sum_rate(
        abs_position_baseline, user_positions,
        baseline_mode_selector, baseline_s2a_allocator,
        snr_linear, elevation_deg,
        sat_rates, sat_pairs, sat_power_factors,
        abs_bandwidth, Pd, Nd, Nsd, config
    )

    print(f"  系统总速率: {baseline_sum_rate/1e6:.2f} Mbps")
    print(f"  配置: 启发式模式选择 + 均匀资源分配 + k-means位置优化")

    # ==================== 方法2: 我们的算法 (Greedy+KKT+L-BFGS-B) ====================
    print(f"\n{'='*80}")
    print(f"步骤4: 我们的算法 (Greedy + KKT + L-BFGS-B)")
    print(f"{'='*80}")

    # 我们的算法使用Greedy模式选择器和KKT资源分配器
    ours_mode_selector = GreedySelector()
    ours_s2a_allocator = KKTAllocator(config)

    optimizer = ContinuousPositionOptimizer(config, method='L-BFGS-B')

    abs_position_ours, ours_sum_rate, opt_info = optimizer.optimize_position_lbfgsb_pure(
        user_positions, ours_mode_selector, ours_s2a_allocator,
        snr_linear, elevation_deg,
        sat_channel_gains, sat_rates, sat_pairs, sat_power_factors,
        abs_bandwidth, Pd, Nd, Nsd,
        max_iter=20,
        verbose=True
    )

    print(f"[OK] L-BFGS-B优化完成")
    print(f"  ABS位置: ({abs_position_ours[0]:.2f}, "
          f"{abs_position_ours[1]:.2f}, {abs_position_ours[2]:.0f})")
    print(f"  系统总速率: {ours_sum_rate/1e6:.2f} Mbps")
    print(f"  配置: Greedy模式选择 + KKT资源分配 + L-BFGS-B位置优化")

    # ==================== 对比分析 ====================
    print(f"\n{'='*80}")
    print(f"步骤5: 性能对比")
    print(f"{'='*80}")

    improvement_abs = ours_sum_rate - baseline_sum_rate
    improvement_pct = 100 * (ours_sum_rate / baseline_sum_rate - 1)

    print(f"\nBaseline (启发式+uniform+k-means):")
    print(f"  位置: ({abs_position_baseline[0]:.2f}, "
          f"{abs_position_baseline[1]:.2f}, {abs_position_baseline[2]:.0f})")
    print(f"  总速率: {baseline_sum_rate/1e6:.2f} Mbps")

    print(f"\n我们的算法 (Greedy+KKT+L-BFGS-B):")
    print(f"  位置: ({abs_position_ours[0]:.2f}, "
          f"{abs_position_ours[1]:.2f}, {abs_position_ours[2]:.0f})")
    print(f"  总速率: {ours_sum_rate/1e6:.2f} Mbps")

    print(f"\n改进:")
    print(f"  绝对改进: +{improvement_abs/1e6:.3f} Mbps")
    print(f"  相对改进: +{improvement_pct:.2f}%")

    # ==================== 生成对比图 ====================
    print(f"\n{'='*80}")
    print(f"步骤6: 生成可视化图表")
    print(f"{'='*80}")

    # 图1: Baseline
    baseline_save_path = Path('results/figures/comparison_baseline_kmeans.png')
    plot_user_distribution_with_abs(
        user_positions, abs_position_baseline,
        baseline_sum_rate,
        'Baseline: Heuristic + Uniform + k-means',
        snr_db, elevation_deg,
        baseline_save_path
    )

    # 图2: 我们的算法
    ours_save_path = Path('results/figures/comparison_ours_lbfgsb.png')
    plot_user_distribution_with_abs(
        user_positions, abs_position_ours,
        ours_sum_rate,
        'Our Algorithm: Greedy + KKT + L-BFGS-B',
        snr_db, elevation_deg,
        ours_save_path
    )

    # ==================== 总结 ====================
    print(f"\n{'='*80}")
    print(f"[OK] 对比图生成完成")
    print(f"{'='*80}")
    print(f"\n生成的图表:")
    print(f"  1. Baseline: {baseline_save_path}")
    print(f"  2. 我们的算法: {ours_save_path}")
    print(f"\n关键发现:")
    print(f"  - 两个图使用相同的用户分布（随机种子: {config.random_seed}）")
    print(f"  - 我们的算法总速率提升: +{improvement_pct:.2f}%")
    print(f"  - ABS位置差异: Δx={abs(abs_position_ours[0]-abs_position_baseline[0]):.2f}m, "
          f"Δy={abs(abs_position_ours[1]-abs_position_baseline[1]):.2f}m, "
          f"Δh={abs(abs_position_ours[2]-abs_position_baseline[2]):.0f}m")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
