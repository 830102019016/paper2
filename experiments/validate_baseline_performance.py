"""
Baseline性能验证脚本

参考：Karavolos et al. 2022 - SATCON论文实验设计
目的：
1. 验证修正后baseline在多个SNR点的稳定性
2. 绘制性能曲线（频谱效率 vs SNR）
3. 分析模式分布随SNR变化
4. 保存结果供后续对比

实验设置：
- SNR范围：0-30 dB（参考原论文）
- 步长：2 dB
- 仿真次数：50次MC
- ABS带宽：1.2 MHz（参考原论文）
- 卫星仰角：10°（参考原论文）
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json
from datetime import datetime

from config import config
from src.satcon_system import SATCONSystem

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def run_baseline_validation():
    """运行baseline性能验证"""

    print("=" * 70)
    print("SATCON Baseline 性能验证")
    print("=" * 70)

    # 实验参数（参考SATCON原论文）
    snr_db_range = np.arange(0, 32, 2)  # 0-30 dB, 步长2dB
    n_realizations = 50  # 50次MC（平衡精度和时间）
    elevation_deg = 10  # 卫星仰角10度
    abs_bandwidth = 1.2e6  # ABS带宽1.2 MHz

    print(f"\n实验参数：")
    print(f"  SNR范围: {snr_db_range[0]}-{snr_db_range[-1]} dB")
    print(f"  SNR步长: 2 dB")
    print(f"  SNR点数: {len(snr_db_range)}")
    print(f"  仿真次数: {n_realizations}")
    print(f"  卫星仰角: {elevation_deg}°")
    print(f"  ABS带宽: {abs_bandwidth/1e6:.1f} MHz")
    print(f"  用户数: {config.N}")
    print(f"  覆盖半径: {config.coverage_radius} m")

    # 创建SATCON系统
    satcon = SATCONSystem(config, abs_bandwidth)

    # 运行仿真
    print(f"\n开始仿真...")
    mean_sum_rates, mean_se, std_sum_rates, mode_stats = satcon.simulate_performance(
        snr_db_range=snr_db_range,
        elevation_deg=elevation_deg,
        n_realizations=n_realizations,
        verbose=True
    )

    # 打印结果摘要
    print(f"\n" + "=" * 70)
    print("性能结果摘要")
    print("=" * 70)
    print(f"{'SNR(dB)':<10} {'SE(bits/s/Hz)':<18} {'NOMA':<8} {'OMA_W':<8} {'OMA_S':<8} {'SAT':<8}")
    print("-" * 70)

    for i, snr in enumerate(snr_db_range):
        print(f"{snr:<10} {mean_se[i]:<18.2f} "
              f"{mode_stats['noma'][i]:<8.1f} "
              f"{mode_stats['oma_weak'][i]:<8.1f} "
              f"{mode_stats['oma_strong'][i]:<8.1f} "
              f"{mode_stats['sat'][i]:<8.1f}")

    # 保存数据
    results = {
        'timestamp': datetime.now().isoformat(),
        'experiment_params': {
            'snr_db_range': snr_db_range.tolist(),
            'n_realizations': n_realizations,
            'elevation_deg': elevation_deg,
            'abs_bandwidth_MHz': abs_bandwidth/1e6,
            'n_users': config.N,
            'coverage_radius_m': config.coverage_radius
        },
        'performance': {
            'mean_sum_rates_Mbps': (mean_sum_rates/1e6).tolist(),
            'mean_se_bits_per_hz': mean_se.tolist(),
            'std_sum_rates_Mbps': (std_sum_rates/1e6).tolist()
        },
        'mode_statistics': {
            'noma': mode_stats['noma'].tolist(),
            'oma_weak': mode_stats['oma_weak'].tolist(),
            'oma_strong': mode_stats['oma_strong'].tolist(),
            'sat': mode_stats['sat'].tolist()
        }
    }

    results_file = 'results/baseline_performance.json'
    Path('results').mkdir(exist_ok=True)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 结果已保存到: {results_file}")

    return snr_db_range, mean_se, std_sum_rates, mode_stats


def plot_performance_curves(snr_db_range, mean_se, std_sum_rates, mode_stats):
    """绘制性能曲线"""

    print(f"\n正在绘制性能曲线...")

    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # 子图1: 频谱效率 vs SNR
    ax1 = axes[0]
    ax1.plot(snr_db_range, mean_se, 'b-o', linewidth=2, markersize=6,
             label='SATCON Baseline (修正后)')
    ax1.fill_between(snr_db_range,
                      mean_se - std_sum_rates/(1.2e6),
                      mean_se + std_sum_rates/(1.2e6),
                      alpha=0.2, color='blue')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlabel('Satellite SNR (dB)', fontsize=12)
    ax1.set_ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=12)
    ax1.set_title('SATCON Baseline Performance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.set_xlim([snr_db_range[0], snr_db_range[-1]])

    # 子图2: 模式分布 vs SNR
    ax2 = axes[1]
    K = config.N // 2  # 总配对数

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
    print(f"✓ 图形已保存到: {fig_file}")

    plt.show()


def analyze_results(snr_db_range, mean_se, mode_stats):
    """分析实验结果"""

    print(f"\n" + "=" * 70)
    print("结果分析")
    print("=" * 70)

    # 1. 性能趋势分析
    se_low_snr = mean_se[snr_db_range <= 10].mean()
    se_mid_snr = mean_se[(snr_db_range > 10) & (snr_db_range <= 20)].mean()
    se_high_snr = mean_se[snr_db_range > 20].mean()

    print(f"\n【性能趋势】")
    print(f"  低SNR (0-10dB):   平均SE = {se_low_snr:.2f} bits/s/Hz")
    print(f"  中SNR (10-20dB):  平均SE = {se_mid_snr:.2f} bits/s/Hz")
    print(f"  高SNR (20-30dB):  平均SE = {se_high_snr:.2f} bits/s/Hz")
    print(f"  增长趋势: {se_high_snr/se_low_snr:.2f}x (高/低SNR)")

    # 2. 模式分布分析
    K = config.N // 2
    noma_ratio = mode_stats['noma'] / K * 100
    oma_weak_ratio = mode_stats['oma_weak'] / K * 100
    oma_strong_ratio = mode_stats['oma_strong'] / K * 100
    sat_ratio = mode_stats['sat'] / K * 100

    print(f"\n【模式分布（高SNR: 30dB）】")
    idx_30db = np.where(snr_db_range == 30)[0][0]
    print(f"  NOMA:      {noma_ratio[idx_30db]:.1f}%")
    print(f"  OMA_Weak:  {oma_weak_ratio[idx_30db]:.1f}%")
    print(f"  OMA_Strong:{oma_strong_ratio[idx_30db]:.1f}%")
    print(f"  SAT:       {sat_ratio[idx_30db]:.1f}%")

    # 3. ABS增益分析（需要对比SAT-only性能）
    print(f"\n【Baseline验证结论】")

    # 检查性能合理性
    if mean_se[-1] > mean_se[0]:
        print(f"  ✓ 性能随SNR增长（符合预期）")
    else:
        print(f"  ✗ 警告：性能未随SNR增长")

    # 检查模式分布合理性
    if mode_stats['noma'][0] > mode_stats['noma'][-1]:
        print(f"  ✓ NOMA比例随SNR降低（高SNR时OMA更优，符合预期）")
    else:
        print(f"  ⚠ 注意：NOMA比例未随SNR降低（可能需要检查）")

    # 检查ABS是否起作用
    if mode_stats['sat'].max() < K * 0.1:
        print(f"  ✓ ABS大部分时间激活（SAT模式<10%）")
    else:
        print(f"  ⚠ 注意：SAT模式较多，ABS可能未充分利用")

    print(f"\n【后续建议】")
    print(f"  1. 如果性能曲线稳定且合理 → 可以开始设计增强方案")
    print(f"  2. 如果发现异常 → 需要进一步调试baseline")
    print(f"  3. 保存本次结果作为baseline性能基准")


def main():
    """主函数"""

    # 1. 运行性能验证
    snr_db_range, mean_se, std_sum_rates, mode_stats = run_baseline_validation()

    # 2. 绘制性能曲线
    plot_performance_curves(snr_db_range, mean_se, std_sum_rates, mode_stats)

    # 3. 分析结果
    analyze_results(snr_db_range, mean_se, mode_stats)

    print(f"\n" + "=" * 70)
    print("✓ Baseline性能验证完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
