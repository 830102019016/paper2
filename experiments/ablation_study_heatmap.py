"""
消融实验 - 热力图版本

使用热力图（Heatmap）展示消融实验结果
包含三种热力图设计：
1. 经典增益热力图（方案 × SNR）
2. 多指标热力图（方案 × 性能指标）
3. 组合热力图（主热力图 + 边缘统计）
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

    # 初始化优化器
    mode_selector = GreedySelector() if mode_type == 'Greedy' else HeuristicSelector()
    s2a_allocator = KKTAllocator(config_obj) if resource_type == 'KKT' else UniformAllocator(config_obj)

    # 优化ABS位置
    if position_type == 'k-means':
        a2g_channel = A2GChannel()
        placement = ABSPlacement(
            abs_height_range=(config_obj.abs_height_min, config_obj.abs_height_max),
            height_step=config_obj.abs_height_step
        )
        abs_position, _ = placement.optimize_position_complete(user_positions, a2g_channel)
    else:  # L-BFGS-B
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

    # SNR扫描
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


def plot_heatmap_version_a(data_matrix, scheme_names, snr_labels, save_path):
    """
    方案A：经典增益热力图

    行：方案名称
    列：SNR点
    颜色：增益百分比（相对Baseline）
    """
    # 动态计算图表尺寸（让格子更紧凑）
    n_cols = len(snr_labels)
    n_rows = len(scheme_names)

    # 每个格子的尺寸（英寸）
    cell_width = 0.6   # 缩小格子宽度
    cell_height = 0.7  # 缩小格子高度

    # 计算总尺寸（加上边距）
    fig_width = n_cols * cell_width + 2.5   # 右侧留空给colorbar
    fig_height = n_rows * cell_height + 1.5  # 上下留空给标题和标签

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 绘制热力图
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=50)

    # 设置刻度
    ax.set_xticks(np.arange(len(snr_labels)))
    ax.set_yticks(np.arange(len(scheme_names)))
    ax.set_xticklabels(snr_labels, fontsize=10)
    ax.set_yticklabels(scheme_names, fontsize=11, fontweight='bold')

    # 标签
    ax.set_xlabel('SNR [dB]', fontsize=13, fontweight='bold')
    ax.set_title('Ablation Study: Performance Gain Heatmap (Relative to Baseline)',
                 fontsize=14, fontweight='bold', pad=12)

    # 添加数值标注（字体稍小）
    for i in range(len(scheme_names)):
        for j in range(len(snr_labels)):
            text_color = 'white' if data_matrix[i, j] > 25 or data_matrix[i, j] < -2 else 'black'
            text = ax.text(j, i, f'{data_matrix[i, j]:.1f}%',
                          ha="center", va="center",
                          color=text_color, fontsize=9, fontweight='bold')

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, label='Performance Gain [%]',
                       fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 热力图已保存: {save_path}")
    plt.show()

    return fig


def plot_heatmap_version_b(se_data, scheme_names, save_path):
    """
    方案B：多指标热力图

    行：方案名称
    列：性能指标（SE@低SNR, SE@中SNR, SE@高SNR, 平均SE等）
    颜色：归一化得分
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # 计算多个性能指标
    metrics = []
    metric_names = ['SE@Low\n(0-10dB)', 'SE@Mid\n(10-20dB)', 'SE@High\n(20-30dB)',
                   'Avg SE', 'Max SE', 'Min SE']

    for scheme_se in se_data:
        low_snr = np.mean(scheme_se[:11])
        mid_snr = np.mean(scheme_se[11:21])
        high_snr = np.mean(scheme_se[21:])
        avg_se = np.mean(scheme_se)
        max_se = np.max(scheme_se)
        min_se = np.min(scheme_se)
        metrics.append([low_snr, mid_snr, high_snr, avg_se, max_se, min_se])

    metrics_matrix = np.array(metrics)

    # 归一化（每列独立归一化到0-100）
    normalized_matrix = np.zeros_like(metrics_matrix)
    for j in range(metrics_matrix.shape[1]):
        col = metrics_matrix[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            normalized_matrix[:, j] = (col - col_min) / (col_max - col_min) * 100
        else:
            normalized_matrix[:, j] = 50

    # 绘制热力图
    im = ax.imshow(normalized_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

    # 设置刻度
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(scheme_names)))
    ax.set_xticklabels(metric_names, fontsize=11, fontweight='bold')
    ax.set_yticklabels(scheme_names, fontsize=12, fontweight='bold')

    # 标签
    ax.set_title('Ablation Study: Multi-Metric Performance Heatmap',
                 fontsize=15, fontweight='bold', pad=15)

    # 添加实际数值和归一化得分
    for i in range(len(scheme_names)):
        for j in range(len(metric_names)):
            actual_val = metrics_matrix[i, j]
            norm_val = normalized_matrix[i, j]
            text_color = 'white' if norm_val > 60 else 'black'

            # 显示实际值和归一化分数
            text = ax.text(j, i, f'{actual_val:.2f}\n({norm_val:.0f})',
                          ha="center", va="center",
                          color=text_color, fontsize=9, fontweight='bold')

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, label='Normalized Score [0-100]')
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 方案B保存: {save_path}")
    plt.show()

    return fig


def plot_heatmap_version_c(data_matrix, se_data, scheme_names, snr_labels, save_path):
    """
    方案C：组合热力图（最推荐）

    主热力图 + 右侧平均增益条 + 底部SNR性能曲线
    """
    fig = plt.figure(figsize=(16, 8))

    # 创建网格布局
    gs = fig.add_gridspec(3, 4, height_ratios=[0.1, 1, 0.3], width_ratios=[1, 1, 1, 0.3],
                         hspace=0.05, wspace=0.05)

    # ========== 主热力图 ==========
    ax_main = fig.add_subplot(gs[1, :3])

    im = ax_main.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=50)

    ax_main.set_xticks(np.arange(len(snr_labels)))
    ax_main.set_yticks(np.arange(len(scheme_names)))
    ax_main.set_xticklabels([])  # X轴标签放在底部子图
    ax_main.set_yticklabels(scheme_names, fontsize=13, fontweight='bold')

    ax_main.set_ylabel('Optimization Scheme', fontsize=14, fontweight='bold')

    # 添加数值标注
    for i in range(len(scheme_names)):
        for j in range(len(snr_labels)):
            text_color = 'white' if data_matrix[i, j] > 25 or data_matrix[i, j] < -2 else 'black'
            text = ax_main.text(j, i, f'{data_matrix[i, j]:.1f}',
                               ha="center", va="center",
                               color=text_color, fontsize=9, fontweight='bold')

    # ========== 右侧平均增益条形图 ==========
    ax_right = fig.add_subplot(gs[1, 3])

    avg_gains = np.mean(data_matrix, axis=1)
    colors_bar = ['#d62728' if g < 0 else '#2ca02c' if g > 10 else '#ff7f0e'
                  for g in avg_gains]

    bars = ax_right.barh(np.arange(len(scheme_names)), avg_gains, color=colors_bar,
                         edgecolor='black', linewidth=1.5)

    # 标注数值
    for i, (bar, val) in enumerate(zip(bars, avg_gains)):
        ax_right.text(val + 0.5, i, f'{val:.1f}%',
                     va='center', fontsize=10, fontweight='bold')

    ax_right.set_yticks([])
    ax_right.set_xlabel('Avg Gain [%]', fontsize=11, fontweight='bold')
    ax_right.set_xlim([min(avg_gains)-2, max(avg_gains)+5])
    ax_right.grid(True, alpha=0.3, axis='x')
    ax_right.axvline(x=0, color='black', linestyle='--', linewidth=1)

    # ========== 底部SNR性能曲线 ==========
    ax_bottom = fig.add_subplot(gs[2, :3])

    # 绘制所有方案的SE曲线
    colors_line = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (scheme_name, se_vals) in enumerate(zip(scheme_names, se_data)):
        ax_bottom.plot(np.arange(len(snr_labels)), se_vals,
                      marker='o', markersize=4, linewidth=2,
                      color=colors_line[i], label=scheme_name, alpha=0.8)

    ax_bottom.set_xticks(np.arange(len(snr_labels)))
    ax_bottom.set_xticklabels(snr_labels, fontsize=11)
    ax_bottom.set_xlabel('SNR [dB]', fontsize=14, fontweight='bold')
    ax_bottom.set_ylabel('SE [bits/s/Hz]', fontsize=11, fontweight='bold')
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(loc='upper left', fontsize=9, ncol=5)

    # ========== 顶部标题 ==========
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5,
                 'Ablation Study: Comprehensive Performance Analysis',
                 ha='center', va='center',
                 fontsize=16, fontweight='bold')

    # ========== 颜色条 ==========
    cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Performance Gain [%]')
    cbar.ax.tick_params(labelsize=11)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 方案C保存: {save_path}")
    plt.show()

    return fig


def main():
    """主函数"""

    print("=" * 100)
    print("消融实验 - 热力图版本")
    print("=" * 100)

    # 配置
    snr_db_range = np.arange(10, 21, 1)  # 10-20dB，步长1dB，聚焦关键区间
    elevation_deg = 10
    abs_bandwidth = 1.2e6

    print(f"\n实验配置:")
    print(f"  用户数: {config.N}")
    print(f"  ABS带宽: {abs_bandwidth/1e6:.1f} MHz")
    print(f"  SNR范围: {snr_db_range[0]} ~ {snr_db_range[-1]} dB (步长={snr_db_range[1]-snr_db_range[0]} dB)")
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

    # 计算增益矩阵（相对Baseline），并删除Baseline行
    baseline_se = se_results['Baseline']

    # 只保留优化方案（去掉Baseline）
    scheme_names_all = list(schemes.keys())
    scheme_names = [name for name in scheme_names_all if name != 'Baseline']

    gain_matrix = []
    for scheme_name in scheme_names:
        gains = (se_results[scheme_name] - baseline_se) / baseline_se * 100
        gain_matrix.append(gains)

    gain_matrix = np.array(gain_matrix)
    snr_labels = [str(int(snr)) for snr in snr_db_range]

    # 只生成方案A（经典增益热力图）
    print(f"\n{'='*100}")
    print(f"生成热力图...")
    print(f"{'='*100}")

    plot_heatmap_version_a(gain_matrix, scheme_names, snr_labels,
                          'results/figures/ablation_heatmap_final.png')

    # 打印统计信息
    print(f"\n{'='*100}")
    print(f"性能增益统计 (相对Baseline)")
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
    print(f"  热力图已保存: results/figures/ablation_heatmap_final.png")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
