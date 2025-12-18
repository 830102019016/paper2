"""
Phase 2 快速验证版：多SNR测试（样本量减少）

用于快速验证代码正确性和初步评估效果

配置：
- SNR: 0-30 dB (步长5dB) - 减少测试点
- n_realizations: 10 - 减少样本量
- 运行时间：约5-8分钟

使用场景：
1. 快速验证代码是否正确运行
2. 初步评估位置优化效果
3. 调试和参数调整

注意：最终论文数据请使用完整版本（test_position_optimization_multi_snr.py）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
import json
from datetime import datetime

from config import config
from src.satcon_system import SATCONSystem

# 设置字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def test_position_optimization_quick():
    """快速验证版本"""
    print("=" * 70)
    print("Phase 2 快速验证：多SNR测试（样本量减少）")
    print("=" * 70)

    # 快速测试配置
    snr_db_range = np.arange(0, 32, 5)  # 0, 5, 10, 15, 20, 25, 30 (7个点)
    elevation_deg = 10
    abs_bandwidth = 1.2e6
    n_realizations = 10  # 快速验证

    print(f"\n【快速测试配置】")
    print(f"  用户数: {config.N}")
    print(f"  ABS带宽: {abs_bandwidth/1e6:.1f} MHz")
    print(f"  SNR范围: {snr_db_range[0]}-{snr_db_range[-1]} dB (步长{snr_db_range[1]-snr_db_range[0]}dB)")
    print(f"  测试点数: {len(snr_db_range)}")
    print(f"  仰角: {elevation_deg}°")
    print(f"  实现次数: {n_realizations}")
    print(f"  预计时间: 5-8分钟")

    # 定义3种方案
    schemes = {
        'Baseline': {
            'mode_selector': 'heuristic',
            's2a_allocator': 'uniform',
            'position_optimizer': None,
            'color': '#95a5a6',
            'marker': 'o',
            'linestyle': '--'
        },
        'One-stage': {
            'mode_selector': 'greedy',
            's2a_allocator': 'kkt',
            'position_optimizer': None,
            'color': '#3498db',
            'marker': 's',
            'linestyle': '-.'
        },
        'Two-stage': {
            'mode_selector': 'greedy',
            's2a_allocator': 'kkt',
            'position_optimizer': 'L-BFGS-B',
            'color': '#e74c3c',
            'marker': '^',
            'linestyle': '-'
        }
    }

    results = {}

    # 运行每种方案
    for scheme_name, scheme_config in schemes.items():
        print(f"\n{'=' * 70}")
        print(f"方案: {scheme_name}")
        print(f"  配置: {scheme_config['mode_selector']} + "
              f"{scheme_config['s2a_allocator']} + "
              f"{scheme_config['position_optimizer'] or 'k-means'}")
        print(f"{'=' * 70}")

        # 创建SATCON系统
        satcon = SATCONSystem(
            config,
            abs_bandwidth=abs_bandwidth,
            mode_selector=scheme_config['mode_selector'],
            s2a_allocator=scheme_config['s2a_allocator'],
            position_optimizer=scheme_config['position_optimizer']
        )

        start_time = time.time()

        # 运行仿真
        mean_sum_rates, mean_se, std_sum_rates, mode_stats = satcon.simulate_performance(
            snr_db_range=snr_db_range,
            elevation_deg=elevation_deg,
            n_realizations=n_realizations,
            verbose=True
        )

        elapsed_time = time.time() - start_time

        # 保存结果
        results[scheme_name] = {
            'config': scheme_config,
            'mean_sum_rates': mean_sum_rates,
            'mean_se': mean_se,
            'std_sum_rates': std_sum_rates,
            'mode_stats': mode_stats,
            'elapsed_time': elapsed_time
        }

        print(f"\n【完成】平均SE: {np.mean(mean_se):.2f} bits/s/Hz, "
              f"用时: {elapsed_time/60:.1f}分钟")

    # 快速分析
    print(f"\n{'=' * 70}")
    print("【快速分析】")
    print(f"{'=' * 70}")

    baseline_se = results['Baseline']['mean_se']
    one_stage_se = results['One-stage']['mean_se']
    two_stage_se = results['Two-stage']['mean_se']

    # 平均改进
    avg_improvement_one = np.mean(100 * (one_stage_se / baseline_se - 1))
    avg_improvement_two = np.mean(100 * (two_stage_se / baseline_se - 1))
    avg_improvement_pos = np.mean(100 * (two_stage_se / one_stage_se - 1))

    print(f"\n平均频谱效率:")
    print(f"  Baseline:   {np.mean(baseline_se):.3f} bits/s/Hz")
    print(f"  One-stage:  {np.mean(one_stage_se):.3f} bits/s/Hz (+{avg_improvement_one:.2f}%)")
    print(f"  Two-stage:  {np.mean(two_stage_se):.3f} bits/s/Hz (+{avg_improvement_two:.2f}%)")

    print(f"\n位置优化贡献:")
    print(f"  Two-stage vs One-stage: +{avg_improvement_pos:.2f}%")

    print(f"\n总运行时间:")
    total_time = sum(data['elapsed_time'] for data in results.values())
    print(f"  {total_time/60:.1f} 分钟")

    # 简单可视化
    visualize_quick(results, snr_db_range, abs_bandwidth)

    # 结论
    print(f"\n{'=' * 70}")
    print("【结论】")
    print(f"{'=' * 70}")

    if avg_improvement_pos >= 2.0:
        print(f"  [SUCCESS] 位置优化显著改进性能 (+{avg_improvement_pos:.2f}%)")
        print(f"  建议：运行完整版测试获取论文数据")
    elif avg_improvement_pos >= 0.5:
        print(f"  [OK] 位置优化有改进 (+{avg_improvement_pos:.2f}%)")
        print(f"  建议：运行完整版测试确认效果")
    else:
        print(f"  [WARNING] 位置优化改进较小 (+{avg_improvement_pos:.2f}%)")
        print(f"  建议：考虑Phase 3优化策略（PSO/多起点等）")

    return results


def visualize_quick(results, snr_db_range, abs_bandwidth):
    """快速可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 子图1：频谱效率 vs SNR
    ax1 = axes[0]
    for scheme_name, data in results.items():
        config = data['config']
        ax1.plot(
            snr_db_range, data['mean_se'],
            color=config['color'],
            marker=config['marker'],
            linestyle=config['linestyle'],
            linewidth=2,
            markersize=8,
            label=scheme_name,
            alpha=0.8
        )

    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=12)
    ax1.set_title(f'SE vs SNR (Quick Test)\nBd={abs_bandwidth/1e6:.1f}MHz, n={10}',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=10)

    # 子图2：改进百分比
    ax2 = axes[1]
    baseline_se = results['Baseline']['mean_se']

    for scheme_name, data in results.items():
        if scheme_name == 'Baseline':
            continue
        config = data['config']
        improvement = 100 * (data['mean_se'] / baseline_se - 1)
        ax2.plot(
            snr_db_range, improvement,
            color=config['color'],
            marker=config['marker'],
            linestyle=config['linestyle'],
            linewidth=2,
            markersize=8,
            label=scheme_name,
            alpha=0.8
        )

    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('Improvement over Baseline (%)', fontsize=12)
    ax2.set_title('Relative Improvement (Quick Test)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)

    plt.tight_layout()

    # 保存图片
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    fig_path = results_dir / 'position_optimization_quick.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"\n[OK] 图表已保存到: {fig_path}")

    plt.close()


if __name__ == "__main__":
    print("\n这是快速验证版本（约5-8分钟）")
    print("如需完整测试，请运行: test_position_optimization_multi_snr.py\n")

    results = test_position_optimization_quick()

    print("\n" + "=" * 70)
    print("快速验证完成！")
    print("=" * 70)
    print("\n下一步:")
    print("  1. 如果效果满意，运行完整版测试获取论文数据")
    print("  2. 如果效果不明显，调整参数后重新测试")
    print("  3. 完整版命令: python experiments/test_position_optimization_multi_snr.py")
