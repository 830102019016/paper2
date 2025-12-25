"""
SNR性能曲线对比实验：频谱效率 vs SNR

对比方案：
1. Baseline: Heuristic + Uniform + k-means
2. Ours: Greedy + KKT + L-BFGS-B

实验设计：
- SNR范围：-10dB 到 30dB，步长5dB
- 固定参数：仰角10度、用户分布、ABS带宽1.2MHz
- 主要指标：频谱效率（SE, bits/s/Hz）
- 辅助指标：Sum Rate、性能增益、模式分布

输出：
- 图1：2x2多子图布局（SE曲线、增益、吞吐量、模式分布）
- 图2：单主图（适合论文发表）
- 数据：results/data/snr_performance_comparison.npz
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

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
    """
    计算给定ABS位置和SNR下的系统性能

    返回:
        sum_rate: 系统总速率 (bps)
        se: 频谱效率 (bits/s/Hz)
        mode_stats: 模式统计 {'sat': count, 'noma': count, 'oma': count}
        final_rates: 各用户速率
    """
    a2g_channel = A2GChannel()
    s2a_channel = S2AChannel()
    allocator = NOMAAllocator()

    # 1. 计算A2G信道增益
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

    # 3. ABS侧配对
    abs_pairs, _ = allocator.optimal_user_pairing(channel_gains_a2g)

    # 4. 计算A2G速率
    K = len(sat_pairs)
    bandwidth_per_pair = Bd / K

    # NOMA A2G速率
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

    # OMA A2G速率
    a2g_rates_oma = bandwidth_per_pair * np.log2(
        1 + Pd * channel_gains_a2g
    )

    # 5. S2A资源分配
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

    # 6. 计算最终速率
    abs_noma_rates = np.minimum(a2g_rates_noma, s2a_rates)
    abs_oma_rates = np.minimum(a2g_rates_oma, s2a_rates)

    final_rates, selected_modes = mode_selector.select_modes(
        sat_rates, abs_noma_rates, abs_oma_rates, abs_pairs
    )

    # 7. 统计模式分布
    mode_stats = {
        'sat': selected_modes.count('sat'),
        'noma': selected_modes.count('noma'),
        'oma': selected_modes.count('oma_weak') + selected_modes.count('oma_strong')
    }

    # 8. 计算性能指标
    sum_rate = np.sum(final_rates)
    total_bandwidth = config_obj.Bs + Bd
    spectral_efficiency = sum_rate / total_bandwidth

    return sum_rate, spectral_efficiency, mode_stats, final_rates


def run_snr_sweep(user_positions, abs_position,
                  mode_selector, s2a_allocator,
                  snr_db_range, elevation_deg,
                  abs_bandwidth, config_obj, method_name):
    """
    在SNR范围内扫描性能

    返回:
        results: {
            'snr_db': [...],
            'sum_rate': [...],
            'se': [...],
            'mode_stats': [{'sat': x, 'noma': y, 'oma': z}, ...],
            'computation_time': [...]
        }
    """
    print(f"\n{'='*80}")
    print(f"运行 {method_name} 的SNR扫描")
    print(f"{'='*80}")

    N = len(snr_db_range)
    sum_rates = np.zeros(N)
    spectral_efficiencies = np.zeros(N)
    mode_stats_list = []
    computation_times = np.zeros(N)

    # 预计算卫星信息（只需要计算一次，因为仰角固定）
    sat_noma = SatelliteNOMA(config_obj)
    sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)
    allocator = NOMAAllocator()
    sat_pairs, _ = allocator.optimal_user_pairing(sat_channel_gains)

    Pd = config_obj.Pd
    Nd = config_obj.get_abs_noise_power(abs_bandwidth)
    Nsd = config_obj.get_s2a_noise_power(config_obj.Bs)

    for i, snr_db in enumerate(snr_db_range):
        print(f"\n  [{i+1}/{N}] SNR = {snr_db} dB")

        start_time = time.time()

        # 计算卫星速率（随SNR变化）
        snr_linear = 10 ** (snr_db / 10)
        sat_rates, _, sat_power_factors = sat_noma.compute_achievable_rates(
            sat_channel_gains, snr_linear
        )

        # 计算系统性能
        sum_rate, se, mode_stats, _ = compute_system_performance(
            abs_position, user_positions,
            mode_selector, s2a_allocator,
            snr_linear, elevation_deg,
            sat_rates, sat_pairs, sat_power_factors,
            abs_bandwidth, Pd, Nd, Nsd, config_obj
        )

        computation_times[i] = (time.time() - start_time) * 1000  # ms

        sum_rates[i] = sum_rate
        spectral_efficiencies[i] = se
        mode_stats_list.append(mode_stats)

        print(f"    Sum Rate: {sum_rate/1e6:.2f} Mbps")
        print(f"    SE: {se:.3f} bits/s/Hz")
        print(f"    Modes: SAT={mode_stats['sat']}, "
              f"NOMA={mode_stats['noma']}, OMA={mode_stats['oma']}")
        print(f"    Time: {computation_times[i]:.2f} ms")

    return {
        'snr_db': snr_db_range,
        'sum_rate': sum_rates,
        'se': spectral_efficiencies,
        'mode_stats': mode_stats_list,
        'computation_time': computation_times
    }


def plot_snr_performance_comparison(baseline_results, ours_results,
                                     snr_db_range, elevation_deg,
                                     save_path_prefix):
    """
    绘制SNR性能对比图

    方案A：2x2多子图布局（完整版）
    方案B：单主图（论文版）
    """

    # ==================== 方案A：2x2多子图布局 ====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 提取数据
    snr_range = baseline_results['snr_db']
    se_baseline = baseline_results['se']
    se_ours = ours_results['se']
    sum_rate_baseline = baseline_results['sum_rate'] / 1e6  # Mbps
    sum_rate_ours = ours_results['sum_rate'] / 1e6  # Mbps

    # 计算增益
    gain_se = (se_ours - se_baseline) / se_baseline * 100  # %
    gain_sum_rate = (sum_rate_ours - sum_rate_baseline) / sum_rate_baseline * 100

    # ========== 子图(a): 频谱效率 vs SNR ==========
    ax = axes[0, 0]
    ax.plot(snr_range, se_baseline, 'o-', linewidth=2.5, markersize=8,
            label='Baseline (Heuristic+Uniform+k-means)', color='blue')
    ax.plot(snr_range, se_ours, '^-', linewidth=2.5, markersize=8,
            label='Ours (Greedy+KKT+L-BFGS-B)', color='red')

    ax.set_xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=13, fontweight='bold')
    ax.set_title('(a) Spectral Efficiency vs SNR', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim([snr_range[0]-1, snr_range[-1]+1])

    # ========== 子图(b): 性能增益 ==========
    ax = axes[0, 1]
    ax.plot(snr_range, gain_se, 's-', linewidth=2.5, markersize=8,
            label='SE Gain', color='green')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # 在每个点上标注百分比
    for i, (x, y) in enumerate(zip(snr_range, gain_se)):
        if i % 2 == 0:  # 每隔一个点标注，避免拥挤
            ax.annotate(f'{y:.1f}%', xy=(x, y),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=9, color='green',
                       fontweight='bold')

    ax.set_xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Performance Gain (%)', fontsize=13, fontweight='bold')
    ax.set_title('(b) SE Improvement over Baseline', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    ax.set_xlim([snr_range[0]-1, snr_range[-1]+1])

    # ========== 子图(c): 系统吞吐量 ==========
    ax = axes[1, 0]
    ax.plot(snr_range, sum_rate_baseline, 'o-', linewidth=2.5, markersize=8,
            label='Baseline', color='blue')
    ax.plot(snr_range, sum_rate_ours, '^-', linewidth=2.5, markersize=8,
            label='Ours', color='red')

    ax.set_xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sum Rate (Mbps)', fontsize=13, fontweight='bold')
    ax.set_title('(c) System Throughput vs SNR', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim([snr_range[0]-1, snr_range[-1]+1])

    # ========== 子图(d): 模式分布 ==========
    ax = axes[1, 1]

    # 提取模式统计数据（Ours方案）
    sat_counts = [stats['sat'] for stats in ours_results['mode_stats']]
    noma_counts = [stats['noma'] for stats in ours_results['mode_stats']]
    oma_counts = [stats['oma'] for stats in ours_results['mode_stats']]

    # 堆叠面积图
    ax.fill_between(snr_range, 0, sat_counts,
                    alpha=0.7, label='SAT Direct', color='skyblue')
    ax.fill_between(snr_range, sat_counts,
                    np.array(sat_counts) + np.array(noma_counts),
                    alpha=0.7, label='ABS NOMA', color='salmon')
    ax.fill_between(snr_range,
                    np.array(sat_counts) + np.array(noma_counts),
                    np.array(sat_counts) + np.array(noma_counts) + np.array(oma_counts),
                    alpha=0.7, label='ABS OMA', color='lightgreen')

    ax.set_xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Pairs', fontsize=13, fontweight='bold')
    ax.set_title('(d) Transmission Mode Distribution (Ours)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim([snr_range[0]-1, snr_range[-1]+1])
    ax.set_ylim([0, len(ours_results['mode_stats'][0])])

    # 总标题
    fig.suptitle(f'SNR Performance Comparison (Elevation = {elevation_deg}°)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # 保存2x2布局图
    save_path_full = Path(f'{save_path_prefix}_full.png')
    save_path_full.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path_full, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 完整对比图已保存: {save_path_full}")

    # ==================== 方案B：单主图（论文版）====================
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    ax2.plot(snr_range, se_baseline, 'o-', linewidth=3, markersize=10,
             label='Baseline (Heuristic+Uniform+k-means)',
             color='blue', markeredgewidth=1.5, markeredgecolor='darkblue')
    ax2.plot(snr_range, se_ours, '^-', linewidth=3, markersize=10,
             label='Proposed (Greedy+KKT+L-BFGS-B)',
             color='red', markeredgewidth=1.5, markeredgecolor='darkred')

    ax2.set_xlabel('SNR (dB)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=16, fontweight='bold')
    ax2.set_title(f'Spectral Efficiency vs SNR (Elevation = {elevation_deg}°)',
                  fontsize=17, fontweight='bold', pad=15)

    ax2.grid(True, alpha=0.4, linestyle='--', linewidth=1)
    ax2.legend(fontsize=13, loc='upper left', framealpha=0.9,
              edgecolor='black', fancybox=True, shadow=True)

    ax2.set_xlim([snr_range[0]-1, snr_range[-1]+1])
    ax2.tick_params(axis='both', which='major', labelsize=13)

    # 添加增益标注（在中间SNR点）
    mid_idx = len(snr_range) // 2
    mid_snr = snr_range[mid_idx]
    mid_gain = gain_se[mid_idx]
    ax2.annotate(f'Gain: +{mid_gain:.1f}%',
                xy=(mid_snr, se_ours[mid_idx]),
                xytext=(20, 20), textcoords='offset points',
                fontsize=12, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                              color='darkgreen', lw=2))

    plt.tight_layout()

    # 保存单主图
    save_path_paper = Path(f'{save_path_prefix}_paper.png')
    plt.savefig(save_path_paper, dpi=300, bbox_inches='tight')
    print(f"[OK] 论文版主图已保存: {save_path_paper}")

    return fig, fig2


def print_performance_table(baseline_results, ours_results, snr_db_range):
    """
    打印性能对比表格
    """
    print(f"\n{'='*100}")
    print(f"性能对比表格")
    print(f"{'='*100}")
    print(f"{'SNR(dB)':<10} {'SE_Base':<15} {'SE_Ours':<15} {'Gain(%)':<12} "
          f"{'SR_Base(Mbps)':<16} {'SR_Ours(Mbps)':<16} {'Gain(%)':<12}")
    print(f"{'-'*100}")

    for i, snr_db in enumerate(snr_db_range):
        se_base = baseline_results['se'][i]
        se_ours = ours_results['se'][i]
        gain_se = (se_ours - se_base) / se_base * 100

        sr_base = baseline_results['sum_rate'][i] / 1e6
        sr_ours = ours_results['sum_rate'][i] / 1e6
        gain_sr = (sr_ours - sr_base) / sr_base * 100

        print(f"{snr_db:<10} {se_base:<15.3f} {se_ours:<15.3f} {gain_se:<12.2f} "
              f"{sr_base:<16.2f} {sr_ours:<16.2f} {gain_sr:<12.2f}")

    print(f"{'='*100}")

    # 计算平均增益
    avg_gain_se = np.mean((ours_results['se'] - baseline_results['se']) / baseline_results['se'] * 100)
    avg_gain_sr = np.mean((ours_results['sum_rate'] - baseline_results['sum_rate']) / baseline_results['sum_rate'] * 100)

    print(f"\n平均性能增益:")
    print(f"  频谱效率 (SE): +{avg_gain_se:.2f}%")
    print(f"  系统吞吐量 (Sum Rate): +{avg_gain_sr:.2f}%")
    print(f"{'='*100}")


def main():
    """主函数"""

    print("=" * 100)
    print("SNR性能曲线对比实验")
    print("=" * 100)

    # ==================== 配置参数 ====================
    snr_db_range = np.arange(-10, 31, 5)  # -10dB 到 30dB，步长5dB
    elevation_deg = 10
    abs_bandwidth = 1.2e6

    print(f"\n实验配置:")
    print(f"  用户数: {config.N}")
    print(f"  覆盖半径: {config.coverage_radius} m")
    print(f"  ABS带宽: {abs_bandwidth/1e6:.1f} MHz")
    print(f"  卫星带宽: {config.Bs/1e6:.1f} MHz")
    print(f"  总带宽: {(config.Bs + abs_bandwidth)/1e6:.1f} MHz")
    print(f"  SNR范围: {snr_db_range[0]} ~ {snr_db_range[-1]} dB (步长={snr_db_range[1]-snr_db_range[0]} dB)")
    print(f"  仰角: {elevation_deg}°")

    # ==================== 生成用户分布 ====================
    print(f"\n{'='*100}")
    print(f"步骤1: 生成用户分布")
    print(f"{'='*100}")

    dist = UserDistribution(config.N, config.coverage_radius,
                           seed=config.random_seed)
    user_positions = dist.generate_uniform_circle()

    print(f"[OK] 生成 {config.N} 个用户位置")

    # ==================== 优化ABS位置（两种方法）====================
    print(f"\n{'='*100}")
    print(f"步骤2: 优化ABS位置")
    print(f"{'='*100}")

    # Baseline ABS位置（k-means）
    print(f"\n[Baseline] 使用k-means优化ABS位置...")
    a2g_channel = A2GChannel()
    placement = ABSPlacement(
        abs_height_range=(config.abs_height_min, config.abs_height_max),
        height_step=config.abs_height_step
    )
    abs_position_baseline, _ = placement.optimize_position_complete(
        user_positions, a2g_channel
    )
    print(f"  ABS位置: ({abs_position_baseline[0]:.2f}, "
          f"{abs_position_baseline[1]:.2f}, {abs_position_baseline[2]:.0f})")

    # Ours ABS位置（L-BFGS-B）
    print(f"\n[Ours] 使用L-BFGS-B优化ABS位置...")

    # 需要先计算一次卫星信息用于位置优化
    sat_noma = SatelliteNOMA(config)
    sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)
    allocator = NOMAAllocator()
    sat_pairs, _ = allocator.optimal_user_pairing(sat_channel_gains)

    # 使用中间SNR值进行位置优化
    snr_mid = 15  # dB
    snr_linear = 10 ** (snr_mid / 10)
    sat_rates, _, sat_power_factors = sat_noma.compute_achievable_rates(
        sat_channel_gains, snr_linear
    )

    Pd = config.Pd
    Nd = config.get_abs_noise_power(abs_bandwidth)
    Nsd = config.get_s2a_noise_power(config.Bs)

    optimizer = ContinuousPositionOptimizer(config, method='L-BFGS-B')
    ours_mode_selector = GreedySelector()
    ours_s2a_allocator = KKTAllocator(config)

    abs_position_ours, _, _ = optimizer.optimize_position_lbfgsb_pure(
        user_positions, ours_mode_selector, ours_s2a_allocator,
        snr_linear, elevation_deg,
        sat_channel_gains, sat_rates, sat_pairs, sat_power_factors,
        abs_bandwidth, Pd, Nd, Nsd,
        max_iter=20,
        verbose=False
    )
    print(f"  ABS位置: ({abs_position_ours[0]:.2f}, "
          f"{abs_position_ours[1]:.2f}, {abs_position_ours[2]:.0f})")

    # ==================== 运行SNR扫描 ====================
    print(f"\n{'='*100}")
    print(f"步骤3: SNR性能扫描")
    print(f"{'='*100}")

    # Baseline: Heuristic + Uniform + k-means
    baseline_mode_selector = HeuristicSelector()
    baseline_s2a_allocator = UniformAllocator(config)

    baseline_results = run_snr_sweep(
        user_positions, abs_position_baseline,
        baseline_mode_selector, baseline_s2a_allocator,
        snr_db_range, elevation_deg,
        abs_bandwidth, config, "Baseline"
    )

    # Ours: Greedy + KKT + L-BFGS-B
    ours_results = run_snr_sweep(
        user_positions, abs_position_ours,
        ours_mode_selector, ours_s2a_allocator,
        snr_db_range, elevation_deg,
        abs_bandwidth, config, "Ours"
    )

    # ==================== 性能对比表格 ====================
    print_performance_table(baseline_results, ours_results, snr_db_range)

    # ==================== 绘制对比图 ====================
    print(f"\n{'='*100}")
    print(f"步骤4: 绘制性能对比图")
    print(f"{'='*100}")

    save_path_prefix = 'results/figures/snr_performance_comparison'
    plot_snr_performance_comparison(
        baseline_results, ours_results,
        snr_db_range, elevation_deg,
        save_path_prefix
    )

    # ==================== 保存数据 ====================
    print(f"\n{'='*100}")
    print(f"步骤5: 保存实验数据")
    print(f"{'='*100}")

    data_save_path = Path('results/data/snr_performance_comparison.npz')
    data_save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        data_save_path,
        snr_db_range=snr_db_range,
        # Baseline
        baseline_se=baseline_results['se'],
        baseline_sum_rate=baseline_results['sum_rate'],
        baseline_mode_stats=baseline_results['mode_stats'],
        baseline_computation_time=baseline_results['computation_time'],
        baseline_abs_position=abs_position_baseline,
        # Ours
        ours_se=ours_results['se'],
        ours_sum_rate=ours_results['sum_rate'],
        ours_mode_stats=ours_results['mode_stats'],
        ours_computation_time=ours_results['computation_time'],
        ours_abs_position=abs_position_ours,
        # Config
        user_positions=user_positions,
        elevation_deg=elevation_deg,
        abs_bandwidth=abs_bandwidth
    )

    print(f"[OK] 数据已保存: {data_save_path}")

    # ==================== 总结 ====================
    print(f"\n{'='*100}")
    print(f"实验完成总结")
    print(f"{'='*100}")

    avg_gain = np.mean((ours_results['se'] - baseline_results['se']) / baseline_results['se'] * 100)
    max_gain = np.max((ours_results['se'] - baseline_results['se']) / baseline_results['se'] * 100)
    max_gain_snr = snr_db_range[np.argmax((ours_results['se'] - baseline_results['se']) / baseline_results['se'] * 100)]

    print(f"\n关键发现:")
    print(f"  ✓ 平均频谱效率增益: +{avg_gain:.2f}%")
    print(f"  ✓ 最大增益: +{max_gain:.2f}% @ SNR={max_gain_snr} dB")
    print(f"  ✓ SNR范围: {snr_db_range[0]} ~ {snr_db_range[-1]} dB")
    print(f"  ✓ 测试点数: {len(snr_db_range)}")

    print(f"\n生成的文件:")
    print(f"  1. 完整对比图: {save_path_prefix}_full.png")
    print(f"  2. 论文版主图: {save_path_prefix}_paper.png")
    print(f"  3. 实验数据: {data_save_path}")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()
