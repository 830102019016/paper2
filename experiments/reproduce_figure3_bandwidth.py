"""
复现论文Figure 3: Sum rate for E = 10° and different ABS bandwidth values Bd = {0.4, 1.2, 2, 3} MHz

实验设计:
- X轴: SNR [dB], 范围 0-30 dB
- Y轴: Sum Rate [Mbps]
- 固定参数: 仰角 E = 10°
- ABS带宽: Bd ∈ {0.4, 1.2, 2, 3} MHz
- 对比方法: SATCON (Ours) vs SAT-NOMA (Baseline)
- 使用: optimizers 优化算法 (Greedy + KKT + L-BFGS-B)
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


def compute_system_performance(abs_position, user_positions,
                                mode_selector, s2a_allocator,
                                snr_linear, elevation_deg,
                                sat_rates, sat_pairs, sat_power_factors,
                                Bd, Pd, Nd, Nsd, config_obj):
    """
    计算给定配置下的系统性能

    返回:
        sum_rate: 系统总速率 (bps)
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

    # 6. 计算最终速率（含DF约束）
    abs_noma_rates = np.minimum(a2g_rates_noma, s2a_rates)
    abs_oma_rates = np.minimum(a2g_rates_oma, s2a_rates)

    # 7. 最终模式选择
    final_rates, _ = mode_selector.select_modes(
        sat_rates, abs_noma_rates, abs_oma_rates, abs_pairs
    )

    sum_rate = np.sum(final_rates)

    return sum_rate


def run_satcon_method(snr_db_range, elevation_deg, abs_bandwidth, config_obj):
    """
    运行SATCON方法 (Greedy + KKT + L-BFGS-B)

    返回:
        results: {'snr_db': [...], 'sum_rate': [...]}
    """
    print(f"\n{'='*60}")
    print(f"运行SATCON方法 (Bd={abs_bandwidth/1e6:.1f} MHz)")
    print(f"{'='*60}")

    # 生成用户分布（固定随机种子）
    dist = UserDistribution(config_obj.N, config_obj.coverage_radius,
                           seed=config_obj.random_seed)
    user_positions = dist.generate_uniform_circle()

    # 初始化组件
    mode_selector = GreedySelector()
    s2a_allocator = KKTAllocator(config_obj)
    optimizer = ContinuousPositionOptimizer(config_obj, method='L-BFGS-B')

    results = {
        'snr_db': [],
        'sum_rate': []
    }

    Pd = config_obj.Pd
    Nd = config_obj.get_abs_noise_power(abs_bandwidth)
    Nsd = config_obj.get_s2a_noise_power(config_obj.Bs)

    for i, snr_db in enumerate(snr_db_range):
        print(f"  [{i+1}/{len(snr_db_range)}] SNR = {snr_db} dB", end='')

        # 计算卫星传输参数
        sat_noma = SatelliteNOMA(config_obj)
        sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)

        allocator = NOMAAllocator()
        sat_pairs, _ = allocator.optimal_user_pairing(sat_channel_gains)

        snr_linear = 10 ** (snr_db / 10)
        sat_rates, _, sat_power_factors = sat_noma.compute_achievable_rates(
            sat_channel_gains, snr_linear
        )

        # 优化ABS位置
        abs_position, sum_rate, _ = optimizer.optimize_position_lbfgsb_pure(
            user_positions, mode_selector, s2a_allocator,
            snr_linear, elevation_deg,
            sat_channel_gains, sat_rates, sat_pairs, sat_power_factors,
            abs_bandwidth, Pd, Nd, Nsd,
            max_iter=15,
            verbose=False
        )

        results['snr_db'].append(snr_db)
        results['sum_rate'].append(sum_rate)

        print(f" → {sum_rate/1e6:.2f} Mbps")

    return results


def run_baseline_method(snr_db_range, elevation_deg, abs_bandwidth, config_obj):
    """
    运行SAT-NOMA Baseline方法 (仅卫星直达NOMA，不使用ABS)

    返回:
        results: {'snr_db': [...], 'sum_rate': [...]}
    """
    print(f"\n{'='*60}")
    print(f"运行SAT-NOMA Baseline (仅卫星直达)")
    print(f"{'='*60}")

    results = {
        'snr_db': [],
        'sum_rate': []
    }

    for i, snr_db in enumerate(snr_db_range):
        print(f"  [{i+1}/{len(snr_db_range)}] SNR = {snr_db} dB", end='')

        # 计算卫星传输参数
        sat_noma = SatelliteNOMA(config_obj)
        sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)

        snr_linear = 10 ** (snr_db / 10)
        sat_rates, _, _ = sat_noma.compute_achievable_rates(
            sat_channel_gains, snr_linear
        )

        # 仅卫星直达，总速率 = sum(sat_rates)
        sum_rate = np.sum(sat_rates)

        results['snr_db'].append(snr_db)
        results['sum_rate'].append(sum_rate)

        print(f" → {sum_rate/1e6:.2f} Mbps")

    return results


def plot_figure3_style(all_results, elevation_deg, save_path):
    """
    绘制Figure 3风格的图表

    参数:
        all_results: {
            0.4: {'satcon': {...}, 'baseline': {...}},
            1.2: {'satcon': {...}, 'baseline': {...}},
            2.0: {'satcon': {...}, 'baseline': {...}},
            3.0: {'satcon': {...}, 'baseline': {...}}
        }
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # 颜色配置（仿论文）
    colors = {
        0.4: '#1f77b4',   # 蓝色
        1.2: '#ff7f0e',   # 橙色
        2.0: '#2ca02c',   # 绿色
        3.0: '#9467bd'    # 紫色
    }

    markers = {
        'satcon': '*',    # 星形（实心）
        'baseline': 'o'   # 圆圈（空心）
    }

    bandwidth_list = [0.4, 1.2, 2.0, 3.0]

    # 绘制曲线
    for bd in bandwidth_list:
        # SATCON曲线（实心星形）
        satcon_data = all_results[bd]['satcon']
        ax.plot(satcon_data['snr_db'],
               np.array(satcon_data['sum_rate']) / 1e6,  # 转换为 Mbps
               color=colors[bd], marker=markers['satcon'],
               markersize=10, linewidth=2, linestyle='-',
               markerfacecolor=colors[bd], markeredgecolor='black',
               markeredgewidth=0.8,
               label=f'SATCON $B_d$={bd} MHz')

    # SAT-NOMA Baseline（空心圆圈，只画一条）
    baseline_data = all_results[0.4]['baseline']  # 与带宽无关，只画一次
    ax.plot(baseline_data['snr_db'],
           np.array(baseline_data['sum_rate']) / 1e6,
           color='gray', marker=markers['baseline'],
           markersize=8, linewidth=2, linestyle='--',
           markerfacecolor='none', markeredgecolor='gray',
           markeredgewidth=1.5,
           label='SAT-NOMA')

    # 坐标轴设置
    ax.set_xlabel('SNR [dB]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sum Rate [Mbps]', fontsize=14, fontweight='bold')

    # 标题
    title = f'Fig. 3: Sum rate for $E$ = {elevation_deg}° and different ABS bandwidth\n'
    title += r'values $B_d$ = {0.4, 1.2, 2, 3} MHz'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # 图例
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)

    # 坐标轴范围
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 70])  # 根据论文图调整

    # 刻度
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    # 保存图表
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存PNG格式
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Figure 3 已保存: {save_path}")

    # 保存EPS格式（论文用）
    eps_path = Path(str(save_path).replace('figures', 'eps').replace('.png', '.eps'))
    eps_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    print(f"[OK] EPS格式已保存: {eps_path}")

    plt.show()

    return fig, ax


def main():
    """主函数：复现Figure 3"""

    print("=" * 80)
    print("复现论文 Figure 3: Sum Rate vs SNR for Different ABS Bandwidths")
    print("=" * 80)

    # ==================== 配置参数 ====================
    snr_db_range = np.arange(0, 31, 2)  # 0, 2, 4, ..., 30 dB (16个点)
    elevation_deg = 10                   # 固定仰角 10°
    bandwidth_list = [0.4e6, 1.2e6, 2.0e6, 3.0e6]  # 4个带宽

    print(f"\n实验配置:")
    print(f"  SNR范围: {snr_db_range[0]} - {snr_db_range[-1]} dB ({len(snr_db_range)} points)")
    print(f"  仰角: {elevation_deg}°")
    print(f"  ABS带宽: {[bd/1e6 for bd in bandwidth_list]} MHz")
    print(f"  用户数: {config.N}")
    print(f"  覆盖半径: {config.coverage_radius} m")
    print(f"  使用算法: Greedy + KKT + L-BFGS-B (optimizers)")

    # ==================== 运行实验 ====================
    all_results = {}

    # 首先运行Baseline（与带宽无关，只需运行一次）
    print(f"\n{'#'*80}")
    print(f"# 步骤1: 运行SAT-NOMA Baseline (仅卫星直达)")
    print(f"{'#'*80}")
    baseline_results = run_baseline_method(
        snr_db_range, elevation_deg, bandwidth_list[0], config
    )

    # 为每个带宽运行SATCON
    for bd in bandwidth_list:
        print(f"\n{'#'*80}")
        print(f"# 步骤2: 运行SATCON (Bd = {bd/1e6:.1f} MHz)")
        print(f"{'#'*80}")

        # SATCON方法
        satcon_results = run_satcon_method(
            snr_db_range, elevation_deg, bd, config
        )

        all_results[bd/1e6] = {
            'satcon': satcon_results,
            'baseline': baseline_results  # 所有带宽共享同一个baseline
        }

    # ==================== 生成Figure 3 ====================
    print(f"\n{'='*80}")
    print(f"生成 Figure 3")
    print(f"{'='*80}")

    save_path = Path('results/figures/figure3_bandwidth_impact.png')
    plot_figure3_style(all_results, elevation_deg, save_path)

    # ==================== 性能对比总结 ====================
    print(f"\n{'='*80}")
    print(f"性能对比总结 (SNR = 30 dB)")
    print(f"{'='*80}")

    baseline_sum_rate_max = baseline_results['sum_rate'][-1]
    print(f"\nSAT-NOMA Baseline: {baseline_sum_rate_max/1e6:.2f} Mbps")

    for bd in bandwidth_list:
        satcon_sum_rate_max = all_results[bd/1e6]['satcon']['sum_rate'][-1]
        improvement = 100 * (satcon_sum_rate_max / baseline_sum_rate_max - 1)

        print(f"\nSATCON (Bd = {bd/1e6:.1f} MHz):")
        print(f"  Sum Rate: {satcon_sum_rate_max/1e6:.2f} Mbps")
        print(f"  改进:     +{improvement:.2f}%")

    print(f"\n{'='*80}")
    print(f"[OK] Figure 3 复现完成!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
