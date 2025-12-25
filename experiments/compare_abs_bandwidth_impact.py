"""
ABS带宽影响分析实验：频谱效率 vs SNR（不同ABS带宽）

复现SATCON论文Figure 2风格：
- 固定仰角E=10°
- 对比不同ABS带宽：Bd = {0.4, 1.2, 2, 3} MHz
- 包含SAT-NOMA基准曲线（Bd=0，纯卫星直达）
- 展示ABS带宽对系统性能的影响

对比方案：
1. SAT-NOMA (Baseline): 纯卫星直达，无ABS
2. SATCON Bd=0.4 MHz: 我们的算法，小带宽
3. SATCON Bd=1.2 MHz: 我们的算法，中带宽
4. SATCON Bd=2 MHz: 我们的算法，大带宽
5. SATCON Bd=3 MHz: 我们的算法，超大带宽

输出：
- 图1：SE vs SNR曲线（论文Figure 2风格）
- 数据：results/data/abs_bandwidth_impact.npz
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.user_distribution import UserDistribution
from src.abs_placement import ABSPlacement
from src.a2g_channel import A2GChannel, S2AChannel
from src.noma_transmission import SatelliteNOMA
from src.power_allocation import NOMAAllocator
from src.optimizers.mode_selector import GreedySelector
from src.optimizers.resource_allocator import KKTAllocator
from src.optimizers.position_optimizer import ContinuousPositionOptimizer


def compute_satcon_performance(abs_position, user_positions,
                                mode_selector, s2a_allocator,
                                snr_linear, elevation_deg,
                                sat_rates, sat_pairs, sat_power_factors,
                                abs_bandwidth, Pd, Nd, Nsd, config_obj):
    """
    计算SATCON系统性能（含ABS）

    返回:
        sum_rate: 系统总速率 (bps)
        se: 频谱效率 (bits/s/Hz)
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
    bandwidth_per_pair = abs_bandwidth / K

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

    final_rates, _ = mode_selector.select_modes(
        sat_rates, abs_noma_rates, abs_oma_rates, abs_pairs
    )

    # 7. 计算性能指标
    sum_rate = np.sum(final_rates)
    total_bandwidth = config_obj.Bs + abs_bandwidth
    spectral_efficiency = sum_rate / total_bandwidth

    return sum_rate, spectral_efficiency


def compute_sat_noma_baseline(snr_linear, elevation_deg, config_obj):
    """
    计算SAT-NOMA基准性能（无ABS）

    返回:
        sum_rate: 系统总速率 (bps)
        se: 频谱效率 (bits/s/Hz)
    """
    sat_noma = SatelliteNOMA(config_obj)
    sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)
    sat_rates, _, _ = sat_noma.compute_achievable_rates(sat_channel_gains, snr_linear)

    sum_rate = np.sum(sat_rates)
    spectral_efficiency = sum_rate / config_obj.Bs  # 只用卫星带宽

    return sum_rate, spectral_efficiency


def run_snr_sweep_for_bandwidth(user_positions, abs_bandwidth_mhz,
                                 snr_db_range, elevation_deg,
                                 config_obj, scheme_name):
    """
    对给定ABS带宽，在SNR范围内扫描性能

    参数:
        abs_bandwidth_mhz: ABS带宽 (MHz)，如果为0则只运行SAT-NOMA

    返回:
        results: {'snr_db': [...], 'se': [...], 'sum_rate': [...]}
    """
    print(f"\n{'='*80}")
    print(f"运行 {scheme_name} (Bd={abs_bandwidth_mhz} MHz)")
    print(f"{'='*80}")

    abs_bandwidth = abs_bandwidth_mhz * 1e6  # 转换为Hz
    N = len(snr_db_range)
    spectral_efficiencies = np.zeros(N)
    sum_rates = np.zeros(N)

    # 如果是SAT-NOMA基准（Bd=0）
    if abs_bandwidth_mhz == 0:
        for i, snr_db in enumerate(snr_db_range):
            snr_linear = 10 ** (snr_db / 10)
            sum_rate, se = compute_sat_noma_baseline(snr_linear, elevation_deg, config_obj)
            sum_rates[i] = sum_rate
            spectral_efficiencies[i] = se

            if i % 2 == 0:  # 每隔一个点打印
                print(f"  SNR={snr_db:2d} dB -> SE={se:.3f} bits/s/Hz")

        return {
            'snr_db': snr_db_range,
            'se': spectral_efficiencies,
            'sum_rate': sum_rates
        }

    # SATCON系统（含ABS）
    # 1. 优化ABS位置（使用中间SNR值）
    print(f"\n  [1/3] 优化ABS位置（使用SNR=15dB）...")

    sat_noma = SatelliteNOMA(config_obj)
    sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)
    allocator = NOMAAllocator()
    sat_pairs, _ = allocator.optimal_user_pairing(sat_channel_gains)

    snr_mid = 15  # dB
    snr_linear = 10 ** (snr_mid / 10)
    sat_rates, _, sat_power_factors = sat_noma.compute_achievable_rates(
        sat_channel_gains, snr_linear
    )

    Pd = config_obj.Pd
    Nd = config_obj.get_abs_noise_power(abs_bandwidth)
    Nsd = config_obj.get_s2a_noise_power(config_obj.Bs)

    optimizer = ContinuousPositionOptimizer(config_obj, method='L-BFGS-B')
    mode_selector = GreedySelector()
    s2a_allocator = KKTAllocator(config_obj)

    abs_position, _, _ = optimizer.optimize_position_lbfgsb_pure(
        user_positions, mode_selector, s2a_allocator,
        snr_linear, elevation_deg,
        sat_channel_gains, sat_rates, sat_pairs, sat_power_factors,
        abs_bandwidth, Pd, Nd, Nsd,
        max_iter=20,
        verbose=False
    )

    print(f"    ABS位置: ({abs_position[0]:.1f}, {abs_position[1]:.1f}, {abs_position[2]:.0f})")

    # 2. SNR扫描
    print(f"\n  [2/3] SNR性能扫描...")

    for i, snr_db in enumerate(snr_db_range):
        snr_linear = 10 ** (snr_db / 10)
        sat_rates, _, sat_power_factors = sat_noma.compute_achievable_rates(
            sat_channel_gains, snr_linear
        )

        sum_rate, se = compute_satcon_performance(
            abs_position, user_positions,
            mode_selector, s2a_allocator,
            snr_linear, elevation_deg,
            sat_rates, sat_pairs, sat_power_factors,
            abs_bandwidth, Pd, Nd, Nsd, config_obj
        )

        sum_rates[i] = sum_rate
        spectral_efficiencies[i] = se

        if i % 2 == 0:  # 每隔一个点打印
            print(f"    SNR={snr_db:2d} dB -> SE={se:.3f} bits/s/Hz, "
                  f"Sum Rate={sum_rate/1e6:.2f} Mbps")

    print(f"\n  [3/3] 完成！")

    return {
        'snr_db': snr_db_range,
        'se': spectral_efficiencies,
        'sum_rate': sum_rates,
        'abs_position': abs_position
    }


def plot_figure2_style(all_results, elevation_deg, save_path):
    """
    绘制SATCON Figure 2风格的曲线图

    参数:
        all_results: 字典，键为方案名称，值为结果数据
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # 定义颜色和标记样式（匹配论文风格）
    styles = {
        'SAT-NOMA': {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'linewidth': 2.5, 'markersize': 8},
        'SATCON Bd=0.4 MHz': {'color': 'orange', 'marker': 's', 'linestyle': '-', 'linewidth': 2.5, 'markersize': 8},
        'SATCON Bd=1.2 MHz': {'color': 'green', 'marker': '^', 'linestyle': '-', 'linewidth': 2.5, 'markersize': 9},
        'SATCON Bd=2 MHz': {'color': 'red', 'marker': 'D', 'linestyle': '-', 'linewidth': 2.5, 'markersize': 8},
        'SATCON Bd=3 MHz': {'color': 'purple', 'marker': 'v', 'linestyle': '-', 'linewidth': 2.5, 'markersize': 9},
    }

    # 绘制曲线（按顺序）
    scheme_order = ['SAT-NOMA', 'SATCON Bd=0.4 MHz', 'SATCON Bd=1.2 MHz',
                   'SATCON Bd=2 MHz', 'SATCON Bd=3 MHz']

    for scheme_name in scheme_order:
        if scheme_name not in all_results:
            continue

        results = all_results[scheme_name]
        style = styles[scheme_name]

        ax.plot(results['snr_db'], results['se'],
               marker=style['marker'],
               color=style['color'],
               linestyle=style['linestyle'],
               linewidth=style['linewidth'],
               markersize=style['markersize'],
               markeredgewidth=1.2,
               markeredgecolor='black',
               label=scheme_name,
               alpha=0.9)

    # 坐标轴设置
    ax.set_xlabel('SNR [dB]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Spectral Efficiency [bits/sec/Hz]', fontsize=16, fontweight='bold')

    # 标题
    title = f'Fig. 2: Average spectral efficiency of MTs, for E = {elevation_deg}° and\n'
    title += r'different ABS bandwidth values $B_d$ = {0.4, 1.2, 2, 3} MHz.'
    ax.set_title(title, fontsize=14, fontweight='normal', pad=15)

    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # 图例
    ax.legend(fontsize=12, loc='upper left', framealpha=0.95,
             edgecolor='black', fancybox=False, shadow=False)

    # 坐标轴范围
    snr_range = all_results['SAT-NOMA']['snr_db']
    ax.set_xlim([snr_range[0], snr_range[-1]])
    ax.set_ylim([0, None])  # y轴从0开始

    # 刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=13)

    # 次网格
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)

    plt.tight_layout()

    # 保存图表
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Figure 2风格图表已保存: {save_path}")

    # 显示图表
    plt.show()

    return fig, ax


def print_summary_table(all_results, snr_points=[0, 10, 20, 30]):
    """
    打印关键SNR点的性能对比表格
    """
    print(f"\n{'='*100}")
    print(f"关键SNR点性能对比 (Spectral Efficiency, bits/s/Hz)")
    print(f"{'='*100}")

    # 表头
    header = f"{'Scheme':<25}"
    for snr in snr_points:
        header += f"SNR={snr}dB{' ':>7}"
    header += f"{'Avg Gain':<12}"
    print(header)
    print(f"{'-'*100}")

    # SAT-NOMA基准
    sat_noma_results = all_results['SAT-NOMA']
    snr_db_range = sat_noma_results['snr_db']
    sat_noma_se = sat_noma_results['se']

    row = f"{'SAT-NOMA (Baseline)':<25}"
    sat_noma_values = []
    for snr in snr_points:
        idx = np.where(snr_db_range == snr)[0]
        if len(idx) > 0:
            se_val = sat_noma_se[idx[0]]
            sat_noma_values.append(se_val)
            row += f"{se_val:>13.3f} "
        else:
            sat_noma_values.append(0)
            row += f"{'N/A':>13} "
    row += f"{'0.00%':>12}"
    print(row)

    # SATCON各方案
    scheme_order = ['SATCON Bd=0.4 MHz', 'SATCON Bd=1.2 MHz',
                   'SATCON Bd=2 MHz', 'SATCON Bd=3 MHz']

    for scheme_name in scheme_order:
        if scheme_name not in all_results:
            continue

        results = all_results[scheme_name]
        row = f"{scheme_name:<25}"

        gains = []
        for i, snr in enumerate(snr_points):
            idx = np.where(snr_db_range == snr)[0]
            if len(idx) > 0:
                se_val = results['se'][idx[0]]
                row += f"{se_val:>13.3f} "

                # 计算增益
                if sat_noma_values[i] > 0:
                    gain = (se_val - sat_noma_values[i]) / sat_noma_values[i] * 100
                    gains.append(gain)
            else:
                row += f"{'N/A':>13} "

        # 平均增益
        avg_gain = np.mean(gains) if gains else 0
        row += f"{avg_gain:>11.2f}%"
        print(row)

    print(f"{'='*100}")


def main():
    """主函数"""

    print("=" * 100)
    print("ABS带宽影响分析实验（SATCON Figure 2风格）")
    print("=" * 100)

    # ==================== 配置参数 ====================
    snr_db_range = np.arange(0, 31, 1)  # 0-30dB，步长1dB（更密集，曲线更平滑）
    elevation_deg = 10  # 固定仰角10度
    abs_bandwidth_values_mhz = [0, 0.4, 1.2, 2, 3]  # 0表示SAT-NOMA基准

    print(f"\n实验配置:")
    print(f"  用户数: {config.N}")
    print(f"  覆盖半径: {config.coverage_radius} m")
    print(f"  卫星带宽: {config.Bs/1e6:.1f} MHz")
    print(f"  ABS带宽值: {abs_bandwidth_values_mhz} MHz")
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

    # ==================== 运行所有方案 ====================
    print(f"\n{'='*100}")
    print(f"步骤2: 运行所有方案的SNR扫描")
    print(f"{'='*100}")

    all_results = {}

    for bd_mhz in abs_bandwidth_values_mhz:
        if bd_mhz == 0:
            scheme_name = 'SAT-NOMA'
        else:
            scheme_name = f'SATCON Bd={bd_mhz} MHz'

        results = run_snr_sweep_for_bandwidth(
            user_positions, bd_mhz,
            snr_db_range, elevation_deg,
            config, scheme_name
        )

        all_results[scheme_name] = results

    # ==================== 性能对比表格 ====================
    print_summary_table(all_results, snr_points=[0, 10, 20, 30])

    # ==================== 绘制Figure 2风格图表 ====================
    print(f"\n{'='*100}")
    print(f"步骤3: 绘制Figure 2风格图表")
    print(f"{'='*100}")

    save_path = 'results/figures/fig2_abs_bandwidth_impact.png'
    plot_figure2_style(all_results, elevation_deg, save_path)

    # ==================== 保存数据 ====================
    print(f"\n{'='*100}")
    print(f"步骤4: 保存实验数据")
    print(f"{'='*100}")

    data_save_path = Path('results/data/abs_bandwidth_impact.npz')
    data_save_path.parent.mkdir(parents=True, exist_ok=True)

    # 准备保存数据
    save_dict = {
        'snr_db_range': snr_db_range,
        'elevation_deg': elevation_deg,
        'abs_bandwidth_values_mhz': abs_bandwidth_values_mhz,
        'user_positions': user_positions
    }

    for scheme_name, results in all_results.items():
        key_prefix = scheme_name.replace(' ', '_').replace('=', '').replace('.', 'p')
        save_dict[f'{key_prefix}_se'] = results['se']
        save_dict[f'{key_prefix}_sum_rate'] = results['sum_rate']
        if 'abs_position' in results:
            save_dict[f'{key_prefix}_abs_position'] = results['abs_position']

    np.savez(data_save_path, **save_dict)
    print(f"[OK] 数据已保存: {data_save_path}")

    # ==================== 总结 ====================
    print(f"\n{'='*100}")
    print(f"实验完成总结")
    print(f"{'='*100}")

    # 计算最优带宽配置（在SNR=20dB时）
    snr_test = 20
    idx = np.where(snr_db_range == snr_test)[0][0]

    best_scheme = None
    best_se = 0
    for scheme_name, results in all_results.items():
        se_val = results['se'][idx]
        if se_val > best_se:
            best_se = se_val
            best_scheme = scheme_name

    print(f"\n关键发现:")
    print(f"  ✓ 测试方案数: {len(all_results)}")
    print(f"  ✓ SNR范围: {snr_db_range[0]} ~ {snr_db_range[-1]} dB")
    print(f"  ✓ 测试点数: {len(snr_db_range)}")
    print(f"  ✓ 最优方案 (SNR={snr_test}dB): {best_scheme} (SE={best_se:.3f} bits/s/Hz)")

    # 计算相对SAT-NOMA的平均增益
    sat_noma_se_avg = np.mean(all_results['SAT-NOMA']['se'])
    print(f"\n  相对SAT-NOMA基准的平均增益:")
    for scheme_name in ['SATCON Bd=0.4 MHz', 'SATCON Bd=1.2 MHz',
                       'SATCON Bd=2 MHz', 'SATCON Bd=3 MHz']:
        if scheme_name in all_results:
            se_avg = np.mean(all_results[scheme_name]['se'])
            gain = (se_avg - sat_noma_se_avg) / sat_noma_se_avg * 100
            print(f"    - {scheme_name}: +{gain:.2f}%")

    print(f"\n生成的文件:")
    print(f"  1. Figure 2风格图表: {save_path}")
    print(f"  2. 实验数据: {data_save_path}")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()
