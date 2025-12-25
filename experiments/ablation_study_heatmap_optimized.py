"""
消融实验 - 优化版经典增益热力图

优化点：
1. SNR采样密度可调（默认1dB步长）
2. 图表宽高比优化（格子接近正方形）
3. 添加关键SNR点的分隔线
4. 优化配色方案（多种可选）
5. 可选择性显示数值（避免拥挤）
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

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


def compute_system_performance(abs_position, user_positions,
                                mode_selector, s2a_allocator,
                                snr_linear, elevation_deg,
                                sat_rates, sat_pairs, sat_power_factors,
                                Bd, Pd, Nd, Nsd, config_obj):
    """计算系统性能（简化版）"""
    a2g_channel = A2GChannel()
    s2a_channel = S2AChannel()
    allocator = NOMAAllocator()

    x_abs, y_abs, h_abs = abs_position
    distances_2d = np.linalg.norm(
        user_positions[:, :2] - np.array([x_abs, y_abs]), axis=1
    )

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

    elevation_rad = np.deg2rad(elevation_deg)
    d_s2a = (config_obj.satellite_altitude - h_abs) / np.sin(elevation_rad)
    h_s2a = s2a_channel.compute_channel_gain(
        distance=d_s2a, fading_gain=1.0,
        G_tx_dB=config_obj.Gs_t_dB,
        G_rx_dB=config_obj.Gsd_r_dB,
        noise_power=Nsd
    )

    abs_pairs, _ = allocator.optimal_user_pairing(channel_gains_a2g)
    K = len(sat_pairs)
    bandwidth_per_pair = Bd / K

    a2g_rates_noma = np.zeros(2*K)
    for k in range(K):
        weak_idx, strong_idx = abs_pairs[k]
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

    a2g_rates_oma = bandwidth_per_pair * np.log2(1 + Pd * channel_gains_a2g)

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

    abs_noma_rates = np.minimum(a2g_rates_noma, s2a_rates)
    abs_oma_rates = np.minimum(a2g_rates_oma, s2a_rates)

    final_rates, _ = mode_selector.select_modes(
        sat_rates, abs_noma_rates, abs_oma_rates, abs_pairs
    )

    sum_rate = np.sum(final_rates)
    total_bandwidth = config_obj.Bs + Bd
    spectral_efficiency = sum_rate / total_bandwidth

    return spectral_efficiency


def run_scheme_quick(scheme_config, user_positions, snr_db_range,
                      elevation_deg, abs_bandwidth, config_obj):
    """快速运行单个方案"""
    mode_type, resource_type, position_type = scheme_config

    mode_selector = GreedySelector() if mode_type == 'Greedy' else HeuristicSelector()
    s2a_allocator = KKTAllocator(config_obj) if resource_type == 'KKT' else UniformAllocator(config_obj)

    if position_type == 'k-means':
        a2g_channel = A2GChannel()
        placement = ABSPlacement(
            abs_height_range=(config_obj.abs_height_min, config_obj.abs_height_max),
            height_step=config_obj.abs_height_step
        )
        abs_position, _ = placement.optimize_position_complete(user_positions, a2g_channel)
    else:
        sat_noma = SatelliteNOMA(config_obj)
        sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)
        allocator = NOMAAllocator()
        sat_pairs, _ = allocator.optimal_user_pairing(sat_channel_gains)

        snr_mid = 15
        snr_linear = 10 ** (snr_mid / 10)
        sat_rates, _, sat_power_factors = sat_noma.compute_achievable_rates(
            sat_channel_gains, snr_linear
        )

        Pd = config_obj.Pd
        Nd = config_obj.get_abs_noise_power(abs_bandwidth)
        Nsd = config_obj.get_s2a_noise_power(config_obj.Bs)

        optimizer = ContinuousPositionOptimizer(config_obj, method='L-BFGS-B')
        abs_position, _, _ = optimizer.optimize_position_lbfgsb_pure(
            user_positions, mode_selector, s2a_allocator,
            snr_linear, elevation_deg,
            sat_channel_gains, sat_rates, sat_pairs, sat_power_factors,
            abs_bandwidth, Pd, Nd, Nsd,
            max_iter=20, verbose=False
        )

    sat_noma = SatelliteNOMA(config_obj)
    sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)
    allocator = NOMAAllocator()
    sat_pairs, _ = allocator.optimal_user_pairing(sat_channel_gains)

    Pd = config_obj.Pd
    Nd = config_obj.get_abs_noise_power(abs_bandwidth)
    Nsd = config_obj.get_s2a_noise_power(config_obj.Bs)

    spectral_efficiencies = []
    for snr_db in snr_db_range:
        snr_linear = 10 ** (snr_db / 10)
        sat_rates, _, sat_power_factors = sat_noma.compute_achievable_rates(
            sat_channel_gains, snr_linear
        )

        se = compute_system_performance(
            abs_position, user_positions,
            mode_selector, s2a_allocator,
            snr_linear, elevation_deg,
            sat_rates, sat_pairs, sat_power_factors,
            abs_bandwidth, Pd, Nd, Nsd, config_obj
        )
        spectral_efficiencies.append(se)

    return np.array(spectral_efficiencies)


def plot_optimized_heatmap(gain_matrix, scheme_names, snr_labels,
                           save_path, colormap='RdYlGn',
                           show_values='auto', add_grid_lines=True):
    """
    优化版热力图

    参数:
        gain_matrix: 增益矩阵 [方案数 × SNR点数]
        scheme_names: 方案名称列表
        snr_labels: SNR标签列表
        save_path: 保存路径
        colormap: 配色方案 ('RdYlGn', 'coolwarm', 'viridis', 'RdBu_r')
        show_values: 显示数值 ('all', 'sparse', 'none', 'auto')
        add_grid_lines: 是否添加网格线
    """

    n_schemes = len(scheme_names)
    n_snr = len(snr_labels)

    # 动态计算最佳图表尺寸（保持格子接近正方形）
    cell_width = 0.45  # 每个格子的宽度（英寸）
    cell_height = 0.8  # 每个格子的高度（英寸）

    fig_width = max(12, n_snr * cell_width + 3)  # 最小12英寸
    fig_height = max(6, n_schemes * cell_height + 2)  # 最小6英寸

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 绘制热力图
    im = ax.imshow(gain_matrix, cmap=colormap, aspect='auto',
                   vmin=-5, vmax=50, interpolation='nearest')

    # 设置刻度
    ax.set_xticks(np.arange(n_snr))
    ax.set_yticks(np.arange(n_schemes))
    ax.set_xticklabels(snr_labels, fontsize=11, rotation=0)
    ax.set_yticklabels(scheme_names, fontsize=13, fontweight='bold')

    # 标签
    ax.set_xlabel('SNR [dB]', fontsize=15, fontweight='bold')
    ax.set_title('Ablation Study: Performance Gain Relative to Baseline',
                 fontsize=16, fontweight='bold', pad=20)

    # 添加关键SNR分隔线（每10dB）
    if add_grid_lines:
        for snr_val in [10, 20]:
            if snr_val in [int(s) for s in snr_labels]:
                idx = list(snr_labels).index(str(snr_val))
                ax.axvline(x=idx - 0.5, color='white', linestyle='-',
                          linewidth=2, alpha=0.7)

        # 添加方案分隔线
        for i in range(1, n_schemes):
            ax.axhline(y=i - 0.5, color='white', linestyle='-',
                      linewidth=1.5, alpha=0.5)

    # 智能显示数值
    if show_values == 'auto':
        # 根据格子数量自动决定
        if n_snr > 20:
            show_values = 'sparse'  # 太多格子，稀疏显示
        else:
            show_values = 'all'  # 少量格子，全部显示

    if show_values != 'none':
        for i in range(n_schemes):
            for j in range(n_snr):
                # 稀疏模式：只显示每5个或关键点
                if show_values == 'sparse':
                    if j % 5 != 0 and gain_matrix[i, j] < 25:
                        continue  # 跳过非关键点

                # 根据背景颜色选择文字颜色
                text_color = 'white' if (gain_matrix[i, j] > 25 or gain_matrix[i, j] < -2) else 'black'

                # 字体大小根据格子数量调整
                fontsize = 11 if n_snr <= 20 else 9 if n_snr <= 31 else 7

                ax.text(j, i, f'{gain_matrix[i, j]:.1f}',
                       ha="center", va="center",
                       color=text_color, fontsize=fontsize, fontweight='bold')

    # 优化的颜色条
    cbar = plt.colorbar(im, ax=ax, label='Performance Gain [%]',
                       fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

    # 添加性能等级标注
    cbar.ax.text(1.5, -5, 'Low', transform=cbar.ax.transData,
                fontsize=9, color='gray', style='italic')
    cbar.ax.text(1.5, 25, 'Medium', transform=cbar.ax.transData,
                fontsize=9, color='gray', style='italic')
    cbar.ax.text(1.5, 50, 'High', transform=cbar.ax.transData,
                fontsize=9, color='gray', style='italic')

    plt.tight_layout()

    # 保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 优化版热力图已保存: {save_path}")

    # 显示
    plt.show()

    return fig


def main():
    """主函数"""

    print("=" * 100)
    print("消融实验 - 优化版经典增益热力图")
    print("=" * 100)

    # ==================== 可调参数 ====================
    # 选项1：密集采样（1dB步长，0-30dB）- 推荐
    snr_db_range = np.arange(0, 31, 1)

    # 选项2：中等采样（2dB步长，0-30dB）
    # snr_db_range = np.arange(0, 31, 2)

    # 选项3：聚焦关键区间（1dB步长，5-25dB）
    # snr_db_range = np.arange(5, 26, 1)

    # 选项4：超密集采样（0.5dB步长，10-20dB）- 用于局部分析
    # snr_db_range = np.arange(10, 21, 0.5)

    elevation_deg = 10
    abs_bandwidth = 1.2e6

    print(f"\n实验配置:")
    print(f"  用户数: {config.N}")
    print(f"  ABS带宽: {abs_bandwidth/1e6:.1f} MHz")
    print(f"  SNR范围: {snr_db_range[0]} ~ {snr_db_range[-1]} dB")
    print(f"  SNR步长: {snr_db_range[1]-snr_db_range[0]} dB")
    print(f"  SNR采样点数: {len(snr_db_range)}")
    print(f"  仰角: {elevation_deg}°")

    # 生成用户分布
    dist = UserDistribution(config.N, config.coverage_radius, seed=config.random_seed)
    user_positions = dist.generate_uniform_circle()
    print(f"[OK] 生成 {config.N} 个用户位置")

    # 运行5个方案
    print(f"\n{'='*100}")
    print(f"运行5个消融方案...")
    print(f"{'='*100}")

    schemes = {
        'Baseline': ('Heuristic', 'Uniform', 'k-means'),
        '+Opt-Mode': ('Greedy', 'Uniform', 'k-means'),
        '+Opt-Resource': ('Heuristic', 'KKT', 'k-means'),
        '+Opt-Position': ('Heuristic', 'Uniform', 'L-BFGS-B'),
        'Full (Ours)': ('Greedy', 'KKT', 'L-BFGS-B')
    }

    se_results = {}
    for i, (scheme_name, config_tuple) in enumerate(schemes.items(), 1):
        print(f"  [{i}/5] 运行 {scheme_name}...")
        se_vals = run_scheme_quick(config_tuple, user_positions, snr_db_range,
                                   elevation_deg, abs_bandwidth, config)
        se_results[scheme_name] = se_vals

    # 计算增益矩阵
    baseline_se = se_results['Baseline']
    scheme_names = list(schemes.keys())

    gain_matrix = []
    for scheme_name in scheme_names:
        if scheme_name == 'Baseline':
            gains = np.zeros(len(snr_db_range))
        else:
            gains = (se_results[scheme_name] - baseline_se) / baseline_se * 100
        gain_matrix.append(gains)

    gain_matrix = np.array(gain_matrix)
    snr_labels = [str(int(snr)) for snr in snr_db_range]

    # 生成优化版热力图
    print(f"\n{'='*100}")
    print(f"生成优化版热力图...")
    print(f"{'='*100}")

    # 主版本：RdYlGn配色，自动显示数值
    plot_optimized_heatmap(
        gain_matrix, scheme_names, snr_labels,
        'results/figures/ablation_heatmap_optimized.png',
        colormap='RdYlGn',
        show_values='auto',
        add_grid_lines=True
    )

    # 可选：生成其他配色版本
    # plot_optimized_heatmap(
    #     gain_matrix, scheme_names, snr_labels,
    #     'results/figures/ablation_heatmap_coolwarm.png',
    #     colormap='coolwarm',
    #     show_values='sparse',
    #     add_grid_lines=True
    # )

    # 打印统计信息
    print(f"\n{'='*100}")
    print(f"统计信息")
    print(f"{'='*100}")

    for i, scheme_name in enumerate(scheme_names):
        avg_gain = np.mean(gain_matrix[i])
        max_gain = np.max(gain_matrix[i])
        max_snr = snr_db_range[np.argmax(gain_matrix[i])]

        print(f"\n{scheme_name}:")
        print(f"  平均增益: {avg_gain:>6.2f}%")
        print(f"  最大增益: {max_gain:>6.2f}% @ SNR={max_snr:.0f}dB")

    print(f"\n{'='*100}")
    print(f"实验完成！")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
