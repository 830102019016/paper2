"""
消融实验（Ablation Study）：算法模块贡献度分析

实验设计：
通过逐步添加优化模块，分析每个模块对系统性能的贡献

对比5个方案：
1. Baseline: Heuristic + Uniform + k-means (全Baseline)
2. +Opt-Mode: Greedy + Uniform + k-means (只优化模式选择)
3. +Opt-Resource: Heuristic + KKT + k-means (只优化资源分配)
4. +Opt-Position: Heuristic + Uniform + L-BFGS-B (只优化位置)
5. Full (Ours): Greedy + KKT + L-BFGS-B (全优化)

可视化：
- 2x3 六子图布局
- 包含：曲线图、热力图、柱状图、雷达图、散点图
- 高度专业化，适合SCI期刊发表
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
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
    """计算系统性能"""
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
        distance=d_s2a,
        fading_gain=1.0,
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

    a2g_rates_oma = bandwidth_per_pair * np.log2(
        1 + Pd * channel_gains_a2g
    )

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

    return sum_rate, spectral_efficiency


def run_scheme(scheme_name, user_positions, mode_selector_type,
               s2a_allocator_type, position_optimizer_type,
               snr_db_range, elevation_deg, abs_bandwidth, config_obj):
    """运行单个方案"""
    print(f"\n{'='*80}")
    print(f"运行方案: {scheme_name}")
    print(f"  模式选择: {mode_selector_type}")
    print(f"  资源分配: {s2a_allocator_type}")
    print(f"  位置优化: {position_optimizer_type}")
    print(f"{'='*80}")

    # 初始化优化器
    if mode_selector_type == 'Heuristic':
        mode_selector = HeuristicSelector()
    else:  # Greedy
        mode_selector = GreedySelector()

    if s2a_allocator_type == 'Uniform':
        s2a_allocator = UniformAllocator(config_obj)
    else:  # KKT
        s2a_allocator = KKTAllocator(config_obj)

    # 优化ABS位置
    start_time_pos = time.time()

    if position_optimizer_type == 'k-means':
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

    position_time = (time.time() - start_time_pos) * 1000  # ms

    print(f"  ABS位置: ({abs_position[0]:.1f}, {abs_position[1]:.1f}, {abs_position[2]:.0f})")
    print(f"  位置优化耗时: {position_time:.2f} ms")

    # SNR扫描
    sat_noma = SatelliteNOMA(config_obj)
    sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)
    allocator = NOMAAllocator()
    sat_pairs, _ = allocator.optimal_user_pairing(sat_channel_gains)

    Pd = config_obj.Pd
    Nd = config_obj.get_abs_noise_power(abs_bandwidth)
    Nsd = config_obj.get_s2a_noise_power(config_obj.Bs)

    N = len(snr_db_range)
    spectral_efficiencies = np.zeros(N)
    sum_rates = np.zeros(N)
    computation_times = np.zeros(N)

    for i, snr_db in enumerate(snr_db_range):
        start_time = time.time()

        snr_linear = 10 ** (snr_db / 10)
        sat_rates, _, sat_power_factors = sat_noma.compute_achievable_rates(
            sat_channel_gains, snr_linear
        )

        sum_rate, se = compute_system_performance(
            abs_position, user_positions,
            mode_selector, s2a_allocator,
            snr_linear, elevation_deg,
            sat_rates, sat_pairs, sat_power_factors,
            abs_bandwidth, Pd, Nd, Nsd, config_obj
        )

        computation_times[i] = (time.time() - start_time) * 1000

        sum_rates[i] = sum_rate
        spectral_efficiencies[i] = se

        if i % 5 == 0:
            print(f"    SNR={snr_db:2d} dB -> SE={se:.3f} bits/s/Hz, Time={computation_times[i]:.2f} ms")

    avg_comp_time = np.mean(computation_times)
    print(f"  平均计算时间: {avg_comp_time:.2f} ms")
    print(f"  完成！")

    return {
        'snr_db': snr_db_range,
        'se': spectral_efficiencies,
        'sum_rate': sum_rates,
        'abs_position': abs_position,
        'position_time': position_time,
        'avg_comp_time': avg_comp_time
    }


def plot_ablation_study(all_results, snr_db_range, elevation_deg, abs_bandwidth, save_path):
    """
    绘制消融实验结果（2x3六子图布局）
    """
    fig = plt.figure(figsize=(18, 12))

    # 配色方案
    colors = {
        'Baseline': '#1f77b4',              # 蓝色
        '+Opt-Mode': '#ff7f0e',             # 橙色
        '+Opt-Resource': '#2ca02c',         # 绿色
        '+Opt-Position': '#d62728',         # 红色
        'Full (Ours)': '#9467bd'            # 紫色
    }

    markers = {
        'Baseline': 'o',
        '+Opt-Mode': 's',
        '+Opt-Resource': '^',
        '+Opt-Position': 'D',
        'Full (Ours)': '*'
    }

    # ========== (a) SE vs SNR 曲线对比 ==========
    ax1 = plt.subplot(2, 3, 1)

    for scheme_name, results in all_results.items():
        ax1.plot(results['snr_db'], results['se'],
                marker=markers[scheme_name],
                color=colors[scheme_name],
                linewidth=2.5,
                markersize=7 if scheme_name != 'Full (Ours)' else 12,
                markeredgewidth=1.2,
                markeredgecolor='black',
                label=scheme_name,
                alpha=0.9)

    ax1.set_xlabel('SNR [dB]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Spectral Efficiency [bits/s/Hz]', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Performance Comparison', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9, loc='upper left')

    # ========== (b) 增益热力图 ==========
    ax2 = plt.subplot(2, 3, 2)

    baseline_se = all_results['Baseline']['se']

    # 构建增益矩阵（方案 × SNR点）
    schemes = ['Baseline', '+Opt-Mode', '+Opt-Resource', '+Opt-Position', 'Full (Ours)']
    snr_indices = [0, 5, 10, 15, 20, 25, 30]  # 显示关键SNR点

    gain_matrix = []
    for scheme in schemes:
        gains = []
        for idx in snr_indices:
            if scheme == 'Baseline':
                gain = 0.0
            else:
                se_val = all_results[scheme]['se'][idx]
                gain = (se_val - baseline_se[idx]) / baseline_se[idx] * 100
            gains.append(gain)
        gain_matrix.append(gains)

    im = ax2.imshow(gain_matrix, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=50)

    ax2.set_xticks(range(len(snr_indices)))
    ax2.set_xticklabels([f'{snr_db_range[i]}' for i in snr_indices])
    ax2.set_yticks(range(len(schemes)))
    ax2.set_yticklabels(schemes, fontsize=10)

    ax2.set_xlabel('SNR [dB]', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Gain Heatmap [%]', fontsize=13, fontweight='bold')

    # 添加数值标注
    for i in range(len(schemes)):
        for j in range(len(snr_indices)):
            text = ax2.text(j, i, f'{gain_matrix[i][j]:.1f}',
                           ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax2, label='Gain [%]')

    # ========== (c) 增益分解柱状图 ==========
    ax3 = plt.subplot(2, 3, 3)

    # 计算每个模块的独立贡献（在SNR=20dB时）
    idx_20db = 20
    baseline_val = baseline_se[idx_20db]

    contributions = {
        'Mode Selection': (all_results['+Opt-Mode']['se'][idx_20db] - baseline_val) / baseline_val * 100,
        'Resource Alloc': (all_results['+Opt-Resource']['se'][idx_20db] - baseline_val) / baseline_val * 100,
        'Position Optim': (all_results['+Opt-Position']['se'][idx_20db] - baseline_val) / baseline_val * 100,
        'Joint (Full)': (all_results['Full (Ours)']['se'][idx_20db] - baseline_val) / baseline_val * 100
    }

    modules = list(contributions.keys())
    values = list(contributions.values())

    bars = ax3.bar(modules, values, color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    # 添加数值标注
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'+{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax3.set_ylabel('Gain over Baseline [%]', fontsize=12, fontweight='bold')
    ax3.set_title(f'(c) Module Contribution (SNR=20dB)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.set_xticklabels(modules, rotation=15, ha='right', fontsize=10)

    # ========== (d) 雷达图/蜘蛛图 ==========
    ax4 = plt.subplot(2, 3, 4, projection='polar')

    # 评估维度
    categories = ['Low SNR\n(0-10dB)', 'Mid SNR\n(10-20dB)', 'High SNR\n(20-30dB)',
                  'Avg Gain', 'Efficiency']
    N_cat = len(categories)

    # 计算各方案在不同维度的得分
    def compute_scores(results):
        low_snr = np.mean(results['se'][:11])
        mid_snr = np.mean(results['se'][11:21])
        high_snr = np.mean(results['se'][21:])
        avg_gain = np.mean(results['se'])
        efficiency = 1.0 / (results['avg_comp_time'] / 100)  # 归一化
        return [low_snr, mid_snr, high_snr, avg_gain, efficiency]

    angles = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
    angles += angles[:1]

    for scheme_name in ['Baseline', '+Opt-Mode', '+Opt-Resource',
                        '+Opt-Position', 'Full (Ours)']:
        scores = compute_scores(all_results[scheme_name])
        scores += scores[:1]

        ax4.plot(angles, scores, 'o-', linewidth=2,
                label=scheme_name, color=colors[scheme_name])
        ax4.fill(angles, scores, alpha=0.15, color=colors[scheme_name])

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=9)
    ax4.set_ylim(0, None)
    ax4.set_title('(d) Multi-dimensional Performance', fontsize=13,
                  fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax4.grid(True)

    # ========== (e) 计算时间对比 ==========
    ax5 = plt.subplot(2, 3, 5)

    schemes_list = list(all_results.keys())
    comp_times = [all_results[s]['avg_comp_time'] for s in schemes_list]
    pos_times = [all_results[s]['position_time'] for s in schemes_list]

    x = np.arange(len(schemes_list))
    width = 0.35

    bars1 = ax5.bar(x - width/2, comp_times, width, label='Avg Comp Time',
                    color='skyblue', edgecolor='black', linewidth=1.2)
    bars2 = ax5.bar(x + width/2, pos_times, width, label='Position Optim',
                    color='lightcoral', edgecolor='black', linewidth=1.2)

    ax5.set_ylabel('Time [ms]', fontsize=12, fontweight='bold')
    ax5.set_title('(e) Computational Complexity', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(schemes_list, rotation=15, ha='right', fontsize=9)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--')

    # ========== (f) ABS位置分布散点图 ==========
    ax6 = plt.subplot(2, 3, 6)

    for scheme_name, results in all_results.items():
        pos = results['abs_position']
        ax6.scatter(pos[0], pos[1], s=200, marker=markers[scheme_name],
                   color=colors[scheme_name], edgecolors='black', linewidths=1.5,
                   label=scheme_name, alpha=0.8, zorder=5)

        # 标注高度
        ax6.annotate(f'h={pos[2]:.0f}m', xy=(pos[0], pos[1]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, color=colors[scheme_name], fontweight='bold')

    # 绘制覆盖范围
    circle = plt.Circle((0, 0), config.coverage_radius, color='gray',
                        fill=False, linestyle='--', linewidth=2, alpha=0.5)
    ax6.add_patch(circle)

    ax6.set_xlabel('X [m]', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Y [m]', fontsize=12, fontweight='bold')
    ax6.set_title('(f) Optimized ABS Positions', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.legend(fontsize=8, loc='upper right')
    ax6.set_aspect('equal')
    ax6.set_xlim([-config.coverage_radius*1.2, config.coverage_radius*1.2])
    ax6.set_ylim([-config.coverage_radius*1.2, config.coverage_radius*1.2])

    # 总标题
    fig.suptitle(f'Ablation Study: Module Contribution Analysis (Bd={abs_bandwidth/1e6:.1f}MHz, E={elevation_deg}°)',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # 保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 消融实验图表已保存: {save_path}")

    # 显示
    plt.show()

    return fig


def print_summary(all_results, snr_db_range):
    """打印总结表格"""
    print(f"\n{'='*100}")
    print(f"消融实验总结")
    print(f"{'='*100}")

    baseline_se = all_results['Baseline']['se']

    print(f"\n{'方案':<20} {'SE@0dB':<12} {'SE@20dB':<12} {'平均增益':<12} {'计算时间':<15}")
    print(f"{'-'*100}")

    for scheme_name, results in all_results.items():
        se_0db = results['se'][0]
        se_20db = results['se'][20]
        avg_gain = np.mean((results['se'] - baseline_se) / baseline_se * 100)
        comp_time = results['avg_comp_time']

        print(f"{scheme_name:<20} {se_0db:<12.3f} {se_20db:<12.3f} "
              f"{avg_gain:>10.2f}% {comp_time:>12.2f} ms")

    print(f"{'='*100}")


def main():
    """主函数"""

    print("=" * 100)
    print("消融实验（Ablation Study）")
    print("=" * 100)

    # 配置
    snr_db_range = np.arange(0, 31, 1)
    elevation_deg = 10
    abs_bandwidth = 1.2e6

    print(f"\n实验配置:")
    print(f"  用户数: {config.N}")
    print(f"  ABS带宽: {abs_bandwidth/1e6:.1f} MHz")
    print(f"  SNR范围: {snr_db_range[0]} ~ {snr_db_range[-1]} dB")
    print(f"  仰角: {elevation_deg}°")

    # 生成用户分布
    print(f"\n{'='*100}")
    print(f"步骤1: 生成用户分布")
    print(f"{'='*100}")

    dist = UserDistribution(config.N, config.coverage_radius, seed=config.random_seed)
    user_positions = dist.generate_uniform_circle()
    print(f"[OK] 生成 {config.N} 个用户位置")

    # 运行所有方案
    print(f"\n{'='*100}")
    print(f"步骤2: 运行5个消融方案")
    print(f"{'='*100}")

    schemes = {
        'Baseline': ('Heuristic', 'Uniform', 'k-means'),
        '+Opt-Mode': ('Greedy', 'Uniform', 'k-means'),
        '+Opt-Resource': ('Heuristic', 'KKT', 'k-means'),
        '+Opt-Position': ('Heuristic', 'Uniform', 'L-BFGS-B'),
        'Full (Ours)': ('Greedy', 'KKT', 'L-BFGS-B')
    }

    all_results = {}
    for scheme_name, (mode, resource, position) in schemes.items():
        results = run_scheme(
            scheme_name, user_positions,
            mode, resource, position,
            snr_db_range, elevation_deg, abs_bandwidth, config
        )
        all_results[scheme_name] = results

    # 打印总结
    print_summary(all_results, snr_db_range)

    # 绘图
    print(f"\n{'='*100}")
    print(f"步骤3: 生成可视化图表")
    print(f"{'='*100}")

    save_path = 'results/figures/ablation_study.png'
    plot_ablation_study(all_results, snr_db_range, elevation_deg, abs_bandwidth, save_path)

    # 保存数据
    print(f"\n{'='*100}")
    print(f"步骤4: 保存数据")
    print(f"{'='*100}")

    data_path = Path('results/data/ablation_study.npz')
    data_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {'snr_db_range': snr_db_range, 'user_positions': user_positions}
    for scheme_name, results in all_results.items():
        key_prefix = scheme_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')
        save_dict[f'{key_prefix}_se'] = results['se']
        save_dict[f'{key_prefix}_abs_position'] = results['abs_position']

    np.savez(data_path, **save_dict)
    print(f"[OK] 数据已保存: {data_path}")

    print(f"\n{'='*100}")
    print(f"实验完成！")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
