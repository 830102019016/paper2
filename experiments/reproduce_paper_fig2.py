"""
复现论文Fig.2：频谱效率 vs SNR（不同ABS带宽）

论文设置：
- E = 10°
- Bd = {0.4, 1.2, 2, 3} MHz
- SNR = 0-30 dB
- 对比 SATCON vs SAT-NOMA
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json

from config import config
from src.satcon_system import SATCONSystem
from src.noma_transmission import SatelliteNOMA

# 设置字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def simulate_sat_noma_baseline():
    """
    仿真SAT-NOMA baseline（纯卫星NOMA，无ABS）
    """
    print("="*70)
    print("Running SAT-NOMA Baseline Simulation")
    print("="*70)

    sat_noma = SatelliteNOMA(config)

    snr_db_range = np.arange(0, 32, 2)
    n_realizations = 50
    elevation_deg = 10

    n_snr_points = len(snr_db_range)
    sum_rates_all = np.zeros((n_snr_points, n_realizations))

    for i, snr_db in enumerate(snr_db_range):
        print(f"SNR = {snr_db} dB", end='\r')
        snr_linear = 10 ** (snr_db / 10)

        for r in range(n_realizations):
            seed = config.random_seed + i * n_realizations + r

            # 生成信道增益
            sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)

            # 计算速率（注意：现在返回3个值，包括power_factors）
            sat_rates, _, _ = sat_noma.compute_achievable_rates(sat_channel_gains, snr_linear)
            sum_rates_all[i, r] = np.sum(sat_rates)

    mean_sum_rates = np.mean(sum_rates_all, axis=1)
    mean_se = mean_sum_rates / config.Bs  # 频谱效率

    print(f"\nSAT-NOMA Baseline Complete!")

    return snr_db_range, mean_se


def simulate_satcon_multiple_bandwidth():
    """
    仿真SATCON方案（多个ABS带宽）
    """
    print("\n" + "="*70)
    print("Running SATCON Simulation with Multiple Bandwidths")
    print("="*70)

    snr_db_range = np.arange(0, 32, 2)
    n_realizations = 50
    elevation_deg = 10

    # 论文中测试的ABS带宽
    bd_values = [0.4e6, 1.2e6, 2e6, 3e6]  # MHz

    results = {}

    for bd in bd_values:
        print(f"\n--- Bd = {bd/1e6:.1f} MHz ---")

        satcon = SATCONSystem(config, bd)

        mean_sum_rates, mean_se, _, _ = satcon.simulate_performance(
            snr_db_range=snr_db_range,
            elevation_deg=elevation_deg,
            n_realizations=n_realizations,
            verbose=True
        )

        results[bd] = {
            'mean_se': mean_se,
            'mean_sum_rates': mean_sum_rates
        }

    return snr_db_range, results


def plot_comparison():
    """
    绘制对比图（复现Fig.2）
    """
    # 1. 运行SAT-NOMA baseline
    snr_range_baseline, se_baseline = simulate_sat_noma_baseline()

    # 2. 运行SATCON（多个带宽）
    snr_range, satcon_results = simulate_satcon_multiple_bandwidth()

    # 3. 绘图
    plt.figure(figsize=(10, 6))

    # SAT-NOMA baseline
    plt.plot(snr_range_baseline, se_baseline, 'kx-', linewidth=2, markersize=8,
             label='SAT-NOMA (Our Implementation)', markevery=2)

    # SATCON 不同带宽
    colors = ['b', 'g', 'r', 'm']
    markers = ['o', 's', '^', 'd']
    bd_values = [0.4e6, 1.2e6, 2e6, 3e6]

    for (bd, color, marker) in zip(bd_values, colors, markers):
        se = satcon_results[bd]['mean_se']
        plt.plot(snr_range, se, color=color, marker=marker, linewidth=2,
                 markersize=6, label=f'SATCON Bd={bd/1e6:.1f} MHz',
                 markevery=2)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('SNR [dB]', fontsize=12)
    plt.ylabel('Spectral Efficiency [bits/sec/Hz]', fontsize=12)
    plt.title('Reproduction of Paper Fig.2: Average Spectral Efficiency (E=10°)',
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.xlim([0, 30])
    # 自动调整y轴范围以显示所有曲线
    # plt.ylim([0, 12])  # 论文原图范围（我们的数据超出此范围）

    plt.tight_layout()

    # 保存
    fig_file = 'results/reproduction_fig2.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Figure saved: {fig_file}")

    # 保存数据
    results_data = {
        'snr_db_range': snr_range.tolist(),
        'sat_noma_se': se_baseline.tolist(),
        'satcon_results': {
            f'{bd/1e6:.1f}MHz': {
                'se': satcon_results[bd]['mean_se'].tolist()
            }
            for bd in bd_values
        }
    }

    with open('results/reproduction_fig2_data.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    plt.show()

    # 打印数值对比表
    print("\n" + "="*70)
    print("Numerical Comparison Table")
    print("="*70)
    print(f"{'SNR(dB)':<10} {'SAT-NOMA':<12} {'Bd=0.4':<12} {'Bd=1.2':<12} {'Bd=2.0':<12} {'Bd=3.0':<12}")
    print("-"*70)

    for i, snr in enumerate(snr_range):
        row = f"{snr:<10}"
        row += f"{se_baseline[i]:<12.2f}"
        for bd in bd_values:
            row += f"{satcon_results[bd]['mean_se'][i]:<12.2f}"
        print(row)

    # 关键SNR点对比
    print("\n" + "="*70)
    print("Key Performance Indicators")
    print("="*70)

    key_snrs = [0, 10, 20, 30]
    print("\nSE at Key SNR Points:")
    for snr in key_snrs:
        idx = np.where(snr_range == snr)[0][0]
        print(f"\nSNR = {snr} dB:")
        print(f"  SAT-NOMA:         {se_baseline[idx]:.2f} bits/s/Hz")
        for bd in bd_values:
            se = satcon_results[bd]['mean_se'][idx]
            gain = ((se - se_baseline[idx]) / se_baseline[idx]) * 100
            print(f"  SATCON Bd={bd/1e6:.1f}MHz: {se:.2f} bits/s/Hz (Gain: {gain:+.1f}%)")

    # 最佳带宽分析
    print("\n" + "="*70)
    print("Optimal Bandwidth Analysis")
    print("="*70)

    for snr in key_snrs:
        idx = np.where(snr_range == snr)[0][0]
        se_values = [satcon_results[bd]['mean_se'][idx] for bd in bd_values]
        best_idx = np.argmax(se_values)
        best_bd = bd_values[best_idx]
        best_se = se_values[best_idx]
        print(f"SNR={snr}dB: Best Bd = {best_bd/1e6:.1f} MHz (SE = {best_se:.2f} bits/s/Hz)")


if __name__ == "__main__":
    plot_comparison()

    print("\n" + "="*70)
    print("[DONE] Paper Fig.2 Reproduction Completed!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Compare with original paper Fig.2")
    print("  2. Analyze any differences")
    print("  3. Identify potential causes for discrepancies")
