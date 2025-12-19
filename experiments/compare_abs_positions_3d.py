"""
对比ABS位置优化方法：我们的算法 vs Baseline (3D版本)

生成两个3D用户分布和ABS位置图：
1. 我们的算法：Greedy+KKT+L-BFGS-B
2. Baseline：启发式+uniform+k-means

目的：展示我们算法的完整优势（模式选择+资源分配+位置优化）
3D可视化能更直观地展示ABS的高度优化
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
        sum_rate: 系统总速率
        final_rates: 各用户最终速率
        user_modes: 用户模式列表 ['sat', 'abs_noma_weak', 'abs_noma_strong', 'abs_oma']
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

    # 2.5 【Toy Network】ABS侧配对（按Γ^d排序）
    abs_pairs, _ = allocator.optimal_user_pairing(channel_gains_a2g)

    # 3. 计算A2G速率
    K = len(sat_pairs)
    bandwidth_per_pair = Bd / K

    # NOMA A2G速率（基于abs_pairs）
    a2g_rates_noma = np.zeros(2*K)
    for k in range(K):
        weak_idx, strong_idx = abs_pairs[k]  # 【修正】使用ABS配对
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

    # 4. S2A资源分配（基于abs_pairs决策，sat_pairs解码）
    _, modes_temp = mode_selector.select_modes(
        sat_rates, a2g_rates_noma, a2g_rates_oma, abs_pairs
    )

    b_allocated = s2a_allocator.allocate_bandwidth(
        sat_pairs, modes_temp, a2g_rates_noma, a2g_rates_oma,
        snr_linear, h_s2a, sat_power_factors
    )

    s2a_rates = s2a_allocator.compute_s2a_rates(
        sat_pairs, b_allocated, snr_linear, h_s2a, sat_power_factors
    )

    # 5. 计算最终速率（含DF约束）
    abs_noma_rates = np.minimum(a2g_rates_noma, s2a_rates)
    abs_oma_rates = np.minimum(a2g_rates_oma, s2a_rates)

    # 6. 【Toy Network】最终模式选择（基于abs_pairs）
    final_rates, selected_modes = mode_selector.select_modes(
        sat_rates, abs_noma_rates, abs_oma_rates, abs_pairs
    )

    # 7. 确定每个用户的传输模式
    # selected_modes是一个列表，长度为pair数量，值为 'sat', 'noma', 'oma_weak', 'oma_strong'
    # 需要转换为用户级别的模式（基于abs_pairs）
    user_modes = ['sat'] * len(user_positions)  # 默认所有用户为卫星直达

    for k, (weak_idx, strong_idx) in enumerate(abs_pairs):  # 【修正】使用ABS配对
        mode = selected_modes[k]

        if mode == 'noma':
            # 两个用户都使用ABS NOMA中继
            user_modes[weak_idx] = 'abs_noma_weak'
            user_modes[strong_idx] = 'abs_noma_strong'
        elif mode == 'oma_weak':
            # 只有弱用户使用ABS OMA中继
            user_modes[weak_idx] = 'abs_oma'
            # 强用户保持卫星直达 (已默认)
        elif mode == 'oma_strong':
            # 只有强用户使用ABS OMA中继
            user_modes[strong_idx] = 'abs_oma'
            # 弱用户保持卫星直达 (已默认)
        # mode == 'sat': 两个用户都保持卫星直达 (已默认)

    sum_rate = np.sum(final_rates)

    return sum_rate, final_rates, user_modes


def plot_3d_user_distribution_with_abs(user_positions, abs_position,
                                        sum_rate, method_name,
                                        snr_db, elevation_deg, save_path,
                                        sat_pairs=None, channel_gains=None,
                                        user_modes=None):
    """
    绘制3D用户分布和ABS位置图

    参数:
        user_positions: 用户位置 [N, 3]
        abs_position: ABS位置 [x, y, h]
        sum_rate: 系统总速率 (bps)
        method_name: 方法名称
        snr_db: SNR (dB)
        elevation_deg: 仰角 (度)
        save_path: 保存路径
        sat_pairs: 卫星配对信息（可选）
        channel_gains: 信道增益（可选）
        user_modes: 用户模式（可选，'sat'表示卫星直达）
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    coverage_radius = config.coverage_radius

    # 1. 绘制地面覆盖区域（圆形）
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = coverage_radius * np.cos(theta)
    circle_y = coverage_radius * np.sin(theta)
    circle_z = np.zeros_like(circle_x)
    ax.plot(circle_x, circle_y, circle_z, 'gray', linewidth=2, alpha=0.5)

    # 填充地面圆盘
    theta_fill = np.linspace(0, 2*np.pi, 50)
    r_fill = np.linspace(0, coverage_radius, 30)
    Theta, R = np.meshgrid(theta_fill, r_fill)
    X_fill = R * np.cos(Theta)
    Y_fill = R * np.sin(Theta)
    Z_fill = np.zeros_like(X_fill)
    ax.plot_surface(X_fill, Y_fill, Z_fill, alpha=0.1, color='lightgray')

    # 2. 绘制用户位置（按最终传输模式区分颜色 - 5种类型）
    if user_modes is not None:
        # 根据user_modes直接分类为5种接收类型
        sat_weak = []          # 卫星直达 - 弱用户
        sat_strong = []        # 卫星直达 - 强用户
        abs_noma_weak = []     # ABS NOMA中继 - 弱用户
        abs_noma_strong = []   # ABS NOMA中继 - 强用户
        abs_oma = []           # ABS OMA中继

        for i, mode in enumerate(user_modes):
            if mode == 'sat':
                # 判断是强用户还是弱用户（基于sat_pairs）
                is_weak = False
                for weak_idx, strong_idx in sat_pairs:
                    if i == weak_idx or i == strong_idx:
                        if channel_gains[weak_idx] <= channel_gains[strong_idx]:
                            is_weak = (i == weak_idx)
                        else:
                            is_weak = (i == strong_idx)
                        break
                if is_weak:
                    sat_weak.append(i)
                else:
                    sat_strong.append(i)
            elif mode == 'abs_noma_weak':
                abs_noma_weak.append(i)
            elif mode == 'abs_noma_strong':
                abs_noma_strong.append(i)
            elif mode == 'abs_oma':
                abs_oma.append(i)

        # 【类型1】卫星直达NOMA弱用户（蓝色圆圈）
        if sat_weak:
            ax.scatter(user_positions[sat_weak, 0], user_positions[sat_weak, 1],
                      user_positions[sat_weak, 2],
                      c='blue', marker='o', s=100, alpha=0.8,
                      edgecolors='darkblue', linewidths=1.5,
                      label=f'Sat Direct Weak ({len(sat_weak)})', zorder=3)

        # 【类型2】卫星直达NOMA强用户（蓝色三角）
        if sat_strong:
            ax.scatter(user_positions[sat_strong, 0], user_positions[sat_strong, 1],
                      user_positions[sat_strong, 2],
                      c='blue', marker='^', s=100, alpha=0.8,
                      edgecolors='darkblue', linewidths=1.5,
                      label=f'Sat Direct Strong ({len(sat_strong)})', zorder=3)

        # 【类型3】ABS NOMA中继弱用户（红色圆圈）
        if abs_noma_weak:
            ax.scatter(user_positions[abs_noma_weak, 0], user_positions[abs_noma_weak, 1],
                      user_positions[abs_noma_weak, 2],
                      c='red', marker='o', s=100, alpha=0.8,
                      edgecolors='darkred', linewidths=1.5,
                      label=f'ABS NOMA Weak ({len(abs_noma_weak)})', zorder=3)

        # 【类型4】ABS NOMA中继强用户（红色三角）
        if abs_noma_strong:
            ax.scatter(user_positions[abs_noma_strong, 0], user_positions[abs_noma_strong, 1],
                      user_positions[abs_noma_strong, 2],
                      c='red', marker='^', s=100, alpha=0.8,
                      edgecolors='darkred', linewidths=1.5,
                      label=f'ABS NOMA Strong ({len(abs_noma_strong)})', zorder=3)

        # 【类型5】ABS OMA中继（绿色方块）
        if abs_oma:
            ax.scatter(user_positions[abs_oma, 0], user_positions[abs_oma, 1],
                      user_positions[abs_oma, 2],
                      c='green', marker='s', s=120, alpha=0.8,
                      edgecolors='darkgreen', linewidths=1.5,
                      label=f'ABS OMA ({len(abs_oma)})', zorder=3)
    else:
        # 没有模式信息，所有用户用灰色
        ax.scatter(user_positions[:, 0], user_positions[:, 1], user_positions[:, 2],
                  c='gray', marker='o', s=60, alpha=0.7,
                  label=f'Users (N={len(user_positions)})', zorder=3)

    # 3. 绘制ABS位置
    ax.scatter(abs_position[0], abs_position[1], abs_position[2],
              c='red', marker='^', s=500,
              edgecolors='black', linewidths=2.5,
              zorder=5)

    # 4. 绘制ABS到地面的投影线（虚线）
    ax.plot([abs_position[0], abs_position[0]],
           [abs_position[1], abs_position[1]],
           [0, abs_position[2]],
           'r--', linewidth=2, alpha=0.6)

    # 5. 绘制ABS在地面的投影点
    ax.scatter(abs_position[0], abs_position[1], 0,
              c='orange', marker='x', s=200, linewidths=3,
              zorder=4)

    # 6. 图表设置
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Height (m)', fontsize=12, fontweight='bold', labelpad=10)

    # 标题
    title = f'{method_name}\n3D User Distribution and ABS Position'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # 设置坐标轴范围
    ax.set_xlim([-coverage_radius*1.2, coverage_radius*1.2])
    ax.set_ylim([-coverage_radius*1.2, coverage_radius*1.2])
    ax.set_zlim([0, config.abs_height_max*1.2])

    # 添加参数信息文本
    info_text = (f'SNR: {snr_db} dB\n'
                f'Elevation: {elevation_deg} deg\n'
                f'Sum Rate: {sum_rate/1e6:.2f} Mbps\n'
                f'ABS Position:\n'
                f'  x={abs_position[0]:.1f}m\n'
                f'  y={abs_position[1]:.1f}m\n'
                f'  h={abs_position[2]:.0f}m')

    # 将文本放在图的右上角
    ax.text2D(0.98, 0.98, info_text,
             transform=ax.transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 图例
    ax.legend(fontsize=9, loc='upper left')

    # 设置视角
    ax.view_init(elev=25, azim=45)

    # 网格
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 3D图表已保存: {save_path}")

    return fig, ax


def main():
    """主函数：生成两个3D对比图"""

    print("=" * 80)
    print("生成3D ABS位置优化对比图")
    print("=" * 80)

    # ==================== 配置参数 ====================
    snr_db = 15
    elevation_deg = 10
    abs_bandwidth = 1.2e6

    print(f"\n系统配置:")
    print(f"  用户数: {config.N}")
    print(f"  覆盖半径: {config.coverage_radius} m")
    print(f"  ABS带宽: {abs_bandwidth/1e6:.1f} MHz")
    print(f"  SNR: {snr_db} dB")
    print(f"  仰角: {elevation_deg} deg")

    # ==================== 生成用户分布 ====================
    print(f"\n{'='*80}")
    print(f"步骤1: 生成用户分布（固定随机种子确保一致性）")
    print(f"{'='*80}")

    dist = UserDistribution(config.N, config.coverage_radius,
                           seed=config.random_seed)
    user_positions = dist.generate_uniform_circle()

    print(f"[OK] 生成 {config.N} 个用户位置")
    print(f"  X范围: [{np.min(user_positions[:, 0]):.1f}, {np.max(user_positions[:, 0]):.1f}] m")
    print(f"  Y范围: [{np.min(user_positions[:, 1]):.1f}, {np.max(user_positions[:, 1]):.1f}] m")

    # ==================== 计算卫星信息 ====================
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

    # ==================== 方法1: Baseline ====================
    print(f"\n{'='*80}")
    print(f"步骤3: Baseline方法 (启发式 + uniform + k-means)")
    print(f"{'='*80}")

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

    baseline_sum_rate, _, baseline_user_modes = compute_system_sum_rate(
        abs_position_baseline, user_positions,
        baseline_mode_selector, baseline_s2a_allocator,
        snr_linear, elevation_deg,
        sat_rates, sat_pairs, sat_power_factors,
        abs_bandwidth, Pd, Nd, Nsd, config
    )

    print(f"  系统总速率: {baseline_sum_rate/1e6:.2f} Mbps")

    # 统计模式分布
    sat_count_baseline = sum(1 for m in baseline_user_modes if m == 'sat')
    abs_count_baseline = sum(1 for m in baseline_user_modes if m != 'sat')
    print(f"  卫星直达用户: {sat_count_baseline}, ABS中继用户: {abs_count_baseline}")

    # ==================== 方法2: 我们的算法 ====================
    print(f"\n{'='*80}")
    print(f"步骤4: 我们的算法 (Greedy + KKT + L-BFGS-B)")
    print(f"{'='*80}")

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

    # 计算我们算法的用户模式分布
    ours_sum_rate_check, _, ours_user_modes = compute_system_sum_rate(
        abs_position_ours, user_positions,
        ours_mode_selector, ours_s2a_allocator,
        snr_linear, elevation_deg,
        sat_rates, sat_pairs, sat_power_factors,
        abs_bandwidth, Pd, Nd, Nsd, config
    )

    sat_count_ours = sum(1 for m in ours_user_modes if m == 'sat')
    abs_count_ours = sum(1 for m in ours_user_modes if m != 'sat')
    print(f"  卫星直达用户: {sat_count_ours}, ABS中继用户: {abs_count_ours}")

    # ==================== 对比分析 ====================
    print(f"\n{'='*80}")
    print(f"步骤5: 性能对比")
    print(f"{'='*80}")

    improvement_abs = ours_sum_rate - baseline_sum_rate
    improvement_pct = 100 * (ours_sum_rate / baseline_sum_rate - 1)

    print(f"\nBaseline: {baseline_sum_rate/1e6:.2f} Mbps")
    print(f"我们的算法: {ours_sum_rate/1e6:.2f} Mbps")
    print(f"改进: +{improvement_abs/1e6:.3f} Mbps (+{improvement_pct:.2f}%)")

    # ==================== 生成3D对比图 ====================
    print(f"\n{'='*80}")
    print(f"步骤6: 生成3D可视化图表")
    print(f"{'='*80}")

    # 图1: Baseline 3D
    baseline_save_path = Path('results/figures/comparison_baseline_3d.png')
    plot_3d_user_distribution_with_abs(
        user_positions, abs_position_baseline,
        baseline_sum_rate,
        'Baseline: Heuristic + Uniform + k-means',
        snr_db, elevation_deg,
        baseline_save_path,
        sat_pairs=sat_pairs,
        channel_gains=sat_channel_gains,
        user_modes=baseline_user_modes
    )

    # 图2: 我们的算法 3D
    ours_save_path = Path('results/figures/comparison_ours_3d.png')
    plot_3d_user_distribution_with_abs(
        user_positions, abs_position_ours,
        ours_sum_rate,
        'Our Algorithm: Greedy + KKT + L-BFGS-B',
        snr_db, elevation_deg,
        ours_save_path,
        sat_pairs=sat_pairs,
        channel_gains=sat_channel_gains,
        user_modes=ours_user_modes
    )

    # ==================== 总结 ====================
    print(f"\n{'='*80}")
    print(f"[OK] 3D对比图生成完成")
    print(f"{'='*80}")
    print(f"\n生成的图表:")
    print(f"  1. Baseline (3D): {baseline_save_path}")
    print(f"  2. 我们的算法 (3D): {ours_save_path}")
    print(f"\n关键优势:")
    print(f"  - 3D可视化更直观地展示ABS高度优化")
    print(f"  - Baseline高度: {abs_position_baseline[2]:.0f}m")
    print(f"  - 我们的算法高度: {abs_position_ours[2]:.0f}m")
    print(f"  - 高度差异: {abs(abs_position_baseline[2] - abs_position_ours[2]):.0f}m")
    print(f"  - 性能提升: +{improvement_pct:.2f}%")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
