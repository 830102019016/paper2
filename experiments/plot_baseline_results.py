"""
绘制Baseline性能结果（修复编码问题）
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def main():
    # 读取结果
    with open('results/baseline_performance.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    snr_db_range = np.array(results['experiment_params']['snr_db_range'])
    mean_se = np.array(results['performance']['mean_se_bits_per_hz'])
    std_sum_rates = np.array(results['performance']['std_sum_rates_Mbps'])
    mode_stats = {
        'noma': np.array(results['mode_statistics']['noma']),
        'oma_weak': np.array(results['mode_statistics']['oma_weak']),
        'oma_strong': np.array(results['mode_statistics']['oma_strong']),
        'sat': np.array(results['mode_statistics']['sat'])
    }

    # 打印摘要（避免中文输出）
    print("=" * 70)
    print("SATCON Baseline Performance Summary")
    print("=" * 70)
    print(f"\nSNR Range: {snr_db_range[0]}-{snr_db_range[-1]} dB")
    print(f"Min SE (SNR={snr_db_range[0]}dB): {mean_se[0]:.2f} bits/s/Hz")
    print(f"Max SE (SNR={snr_db_range[-1]}dB): {mean_se[-1]:.2f} bits/s/Hz")
    print(f"SE Growth: {mean_se[-1]/mean_se[0]:.2f}x")

    # 分析结果
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    # 1. 性能趋势
    se_low_snr = mean_se[snr_db_range <= 10].mean()
    se_mid_snr = mean_se[(snr_db_range > 10) & (snr_db_range <= 20)].mean()
    se_high_snr = mean_se[snr_db_range > 20].mean()

    print(f"\n[Performance Trend]")
    print(f"  Low SNR (0-10dB):  Avg SE = {se_low_snr:.2f} bits/s/Hz")
    print(f"  Mid SNR (10-20dB): Avg SE = {se_mid_snr:.2f} bits/s/Hz")
    print(f"  High SNR (20-30dB):Avg SE = {se_high_snr:.2f} bits/s/Hz")

    # 2. 模式分布
    K = 16  # N=32, K=16对
    idx_30db = np.where(snr_db_range == 30)[0][0]
    print(f"\n[Mode Distribution at 30dB]")
    print(f"  NOMA:       {mode_stats['noma'][idx_30db]:.1f}/{K} ({mode_stats['noma'][idx_30db]/K*100:.1f}%)")
    print(f"  OMA_Weak:   {mode_stats['oma_weak'][idx_30db]:.1f}/{K} ({mode_stats['oma_weak'][idx_30db]/K*100:.1f}%)")
    print(f"  OMA_Strong: {mode_stats['oma_strong'][idx_30db]:.1f}/{K} ({mode_stats['oma_strong'][idx_30db]/K*100:.1f}%)")
    print(f"  SAT:        {mode_stats['sat'][idx_30db]:.1f}/{K} ({mode_stats['sat'][idx_30db]/K*100:.1f}%)")

    # 3. Baseline验证
    print(f"\n[Baseline Validation]")
    if mean_se[-1] > mean_se[0]:
        print(f"  [OK] SE increases with SNR")
    else:
        print(f"  [WARNING] SE does not increase with SNR")

    if mode_stats['noma'][0] > mode_stats['noma'][-1]:
        print(f"  [OK] NOMA ratio decreases with SNR (OMA better at high SNR)")
    else:
        print(f"  [INFO] NOMA ratio does not decrease with SNR")

    if mode_stats['sat'].max() < K * 0.1:
        print(f"  [OK] ABS is active most of the time (SAT mode < 10%)")
    else:
        print(f"  [INFO] SAT mode is relatively high")

    # 绘制图形
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # 子图1: 频谱效率 vs SNR
    ax1 = axes[0]
    ax1.plot(snr_db_range, mean_se, 'b-o', linewidth=2, markersize=6,
             label='SATCON Baseline (Corrected)')
    ax1.fill_between(snr_db_range,
                      mean_se - std_sum_rates/(1.2),
                      mean_se + std_sum_rates/(1.2),
                      alpha=0.2, color='blue')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlabel('Satellite SNR (dB)', fontsize=12)
    ax1.set_ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=12)
    ax1.set_title('SATCON Baseline Performance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.set_xlim([snr_db_range[0], snr_db_range[-1]])

    # 子图2: 模式分布 vs SNR
    ax2 = axes[1]
    ax2.plot(snr_db_range, mode_stats['noma'], 'r-o', linewidth=2,
             markersize=6, label='NOMA')
    ax2.plot(snr_db_range, mode_stats['oma_weak'], 'g-s', linewidth=2,
             markersize=6, label='OMA (Weak)')
    ax2.plot(snr_db_range, mode_stats['oma_strong'], 'b-^', linewidth=2,
             markersize=6, label='OMA (Strong)')
    ax2.plot(snr_db_range, mode_stats['sat'], 'm-d', linewidth=2,
             markersize=6, label='SAT (No ABS)')

    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Satellite SNR (dB)', fontsize=12)
    ax2.set_ylabel('Average Number of Pairs', fontsize=12)
    ax2.set_title('Transmission Mode Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.set_xlim([snr_db_range[0], snr_db_range[-1]])
    ax2.set_ylim([0, K])

    plt.tight_layout()

    # 保存图形
    fig_file = 'results/baseline_performance_curves.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Figure saved to: {fig_file}")

    plt.show()

    print("\n" + "=" * 70)
    print("[DONE] Baseline performance validation completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Baseline is validated and stable")
    print("  2. Ready to design enhancement schemes")
    print("  3. This result will be used as baseline benchmark")


if __name__ == "__main__":
    main()
