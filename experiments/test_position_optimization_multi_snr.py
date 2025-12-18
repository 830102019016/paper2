"""
Phase 2: ABS位置优化多SNR性能对比

对比实验：
1. Baseline: heuristic + uniform + k-means
2. One-stage: greedy + kkt + k-means
3. Two-stage: greedy + kkt + hybrid (位置优化)

测试配置：
- N=32, K=16
- Bd=1.2 MHz
- SNR=0-30 dB (步长2dB)
- n_realizations=50 (提高准确性)

预期结果：
- 绘制频谱效率 vs SNR 曲线（3条）
- 定量分析改进幅度
- 提供论文用图表和数据

运行时间：约30-40分钟
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
from tqdm import tqdm

from config import config
from src.satcon_system import SATCONSystem

# 设置字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def test_position_optimization_multi_snr():
    """
    多SNR点测试：对比3种方案

    方案1（Baseline）：
        - mode_selector='heuristic'
        - s2a_allocator='uniform'
        - position_optimizer=None

    方案2（One-stage）：
        - mode_selector='greedy'
        - s2a_allocator='kkt'
        - position_optimizer=None

    方案3（Two-stage，推荐）：
        - mode_selector='greedy'
        - s2a_allocator='kkt'
        - position_optimizer='hybrid'
    """
    print("=" * 70)
    print("Phase 2: ABS位置优化多SNR性能对比")
    print("=" * 70)

    # 测试配置
    snr_db_range = np.arange(0, 32, 2)  # 0-30 dB
    elevation_deg = 10
    abs_bandwidth = 1.2e6
    n_realizations = 50  # 提高准确性

    print(f"\n【测试配置】")
    print(f"  用户数: {config.N}")
    print(f"  ABS带宽: {abs_bandwidth/1e6:.1f} MHz")
    print(f"  SNR范围: {snr_db_range[0]}-{snr_db_range[-1]} dB (步长{snr_db_range[1]-snr_db_range[0]}dB)")
    print(f"  仰角: {elevation_deg}°")
    print(f"  实现次数: {n_realizations}")

    # 定义3种方案
    schemes = {
        'Baseline (heuristic+uniform+k-means)': {
            'mode_selector': 'heuristic',
            's2a_allocator': 'uniform',
            'position_optimizer': None,
            'color': '#95a5a6',  # 灰色
            'marker': 'o',
            'linestyle': '--'
        },
        'One-stage (greedy+kkt+k-means)': {
            'mode_selector': 'greedy',
            's2a_allocator': 'kkt',
            'position_optimizer': None,
            'color': '#3498db',  # 蓝色
            'marker': 's',
            'linestyle': '-.'
        },
        'Two-stage (greedy+kkt+L-BFGS-B)': {
            'mode_selector': 'greedy',
            's2a_allocator': 'kkt',
            'position_optimizer': 'L-BFGS-B',
            'color': '#e74c3c',  # 红色
            'marker': '^',
            'linestyle': '-'
        }
    }

    results = {}

    # 运行每种方案
    for scheme_name, scheme_config in schemes.items():
        print(f"\n{'=' * 70}")
        print(f"方案: {scheme_name}")
        print(f"  模式选择: {scheme_config['mode_selector']}")
        print(f"  资源分配: {scheme_config['s2a_allocator']}")
        print(f"  位置优化: {scheme_config['position_optimizer'] or 'k-means'}")
        print(f"{'=' * 70}")

        # 创建SATCON系统
        satcon = SATCONSystem(
            config,
            abs_bandwidth=abs_bandwidth,
            mode_selector=scheme_config['mode_selector'],
            s2a_allocator=scheme_config['s2a_allocator'],
            position_optimizer=scheme_config['position_optimizer']
        )

        # 记录开始时间
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

        print(f"\n【完成】")
        print(f"  平均频谱效率: {np.mean(mean_se):.2f} bits/s/Hz")
        print(f"  总运行时间: {elapsed_time/60:.1f} 分钟")

    # 对比分析
    print(f"\n{'=' * 70}")
    print("【对比分析】")
    print(f"{'=' * 70}")

    baseline_se = results['Baseline (heuristic+uniform+k-means)']['mean_se']
    one_stage_se = results['One-stage (greedy+kkt+k-means)']['mean_se']
    two_stage_se = results['Two-stage (greedy+kkt+L-BFGS-B)']['mean_se']

    # 计算改进幅度
    improvement_one_stage = 100 * (one_stage_se / baseline_se - 1)
    improvement_two_stage = 100 * (two_stage_se / baseline_se - 1)
    improvement_position = 100 * (two_stage_se / one_stage_se - 1)

    print(f"\n【平均频谱效率】")
    print(f"  Baseline:   {np.mean(baseline_se):.3f} bits/s/Hz")
    print(f"  One-stage:  {np.mean(one_stage_se):.3f} bits/s/Hz "
          f"(+{np.mean(improvement_one_stage):.2f}% vs Baseline)")
    print(f"  Two-stage:  {np.mean(two_stage_se):.3f} bits/s/Hz "
          f"(+{np.mean(improvement_two_stage):.2f}% vs Baseline)")

    print(f"\n【位置优化贡献】")
    print(f"  Two-stage vs One-stage: +{np.mean(improvement_position):.2f}%")

    # 按SNR区间分析
    print(f"\n【按SNR区间分析】")
    for snr_range_name, snr_indices in [
        ('低SNR (0-10dB)', range(0, 6)),
        ('中SNR (10-20dB)', range(5, 11)),
        ('高SNR (20-30dB)', range(10, 16))
    ]:
        baseline_avg = np.mean(baseline_se[list(snr_indices)])
        two_stage_avg = np.mean(two_stage_se[list(snr_indices)])
        improvement = 100 * (two_stage_avg / baseline_avg - 1)

        print(f"  {snr_range_name}: +{improvement:.2f}%")

    print(f"\n【计算时间】")
    for scheme_name in schemes.keys():
        elapsed = results[scheme_name]['elapsed_time']
        print(f"  {scheme_name}: {elapsed/60:.1f} 分钟")

    # 保存结果
    save_results(results, snr_db_range, abs_bandwidth)

    # 可视化
    visualize_results(results, snr_db_range, abs_bandwidth)

    return results


def save_results(results, snr_db_range, abs_bandwidth):
    """保存结果到JSON"""
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    # 转换numpy数组为列表（JSON序列化）
    results_json = {
        'config': {
            'snr_db_range': snr_db_range.tolist(),
            'abs_bandwidth_MHz': abs_bandwidth / 1e6,
            'n_realizations': 50,
            'elevation_deg': 10
        },
        'schemes': {}
    }

    for scheme_name, data in results.items():
        results_json['schemes'][scheme_name] = {
            'config': data['config'],
            'mean_se': data['mean_se'].tolist(),
            'std_se': (data['std_sum_rates'] / abs_bandwidth).tolist(),
            'elapsed_time_min': data['elapsed_time'] / 60
        }

    # 计算改进幅度
    baseline_se = results['Baseline (heuristic+uniform+k-means)']['mean_se']
    one_stage_se = results['One-stage (greedy+kkt+k-means)']['mean_se']
    two_stage_se = results['Two-stage (greedy+kkt+L-BFGS-B)']['mean_se']

    results_json['comparison'] = {
        'one_stage_vs_baseline_pct': (100 * (one_stage_se / baseline_se - 1)).tolist(),
        'two_stage_vs_baseline_pct': (100 * (two_stage_se / baseline_se - 1)).tolist(),
        'two_stage_vs_one_stage_pct': (100 * (two_stage_se / one_stage_se - 1)).tolist(),
        'avg_improvement': {
            'one_stage_vs_baseline': float(np.mean(100 * (one_stage_se / baseline_se - 1))),
            'two_stage_vs_baseline': float(np.mean(100 * (two_stage_se / baseline_se - 1))),
            'two_stage_vs_one_stage': float(np.mean(100 * (two_stage_se / one_stage_se - 1)))
        }
    }

    results_json['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    json_path = results_dir / 'position_optimization_multi_snr.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 结果已保存到: {json_path}")


def visualize_results(results, snr_db_range, abs_bandwidth):
    """可视化对比结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    schemes = results.keys()

    # 子图1：频谱效率 vs SNR（主图）
    ax1 = axes[0, 0]
    for scheme_name, data in results.items():
        config = data['config']
        ax1.plot(
            snr_db_range, data['mean_se'],
            color=config['color'],
            marker=config['marker'],
            linestyle=config['linestyle'],
            linewidth=2,
            markersize=6,
            label=scheme_name.split('(')[0].strip(),
            alpha=0.8
        )

    ax1.set_xlabel('SNR (dB)', fontsize=11)
    ax1.set_ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=11)
    ax1.set_title(f'Spectral Efficiency vs SNR\n(Bd={abs_bandwidth/1e6:.1f}MHz, E=10°, n=50)',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)

    # 子图2：改进百分比 vs SNR
    ax2 = axes[0, 1]
    baseline_se = results['Baseline (heuristic+uniform+k-means)']['mean_se']

    for scheme_name, data in results.items():
        if 'Baseline' in scheme_name:
            continue
        config = data['config']
        improvement = 100 * (data['mean_se'] / baseline_se - 1)
        ax2.plot(
            snr_db_range, improvement,
            color=config['color'],
            marker=config['marker'],
            linestyle=config['linestyle'],
            linewidth=2,
            markersize=6,
            label=scheme_name.split('(')[0].strip(),
            alpha=0.8
        )

    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('SNR (dB)', fontsize=11)
    ax2.set_ylabel('Improvement over Baseline (%)', fontsize=11)
    ax2.set_title('Relative Improvement vs SNR', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)

    # 子图3：平均频谱效率对比（条形图）
    ax3 = axes[1, 0]
    scheme_names_short = [name.split('(')[0].strip() for name in schemes]
    mean_se_values = [np.mean(data['mean_se']) for data in results.values()]
    colors = [data['config']['color'] for data in results.values()]

    bars = ax3.bar(range(len(schemes)), mean_se_values, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(schemes)))
    ax3.set_xticklabels(scheme_names_short, rotation=15, ha='right')
    ax3.set_ylabel('Avg Spectral Efficiency (bits/s/Hz)', fontsize=11)
    ax3.set_title('Average Performance Comparison', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # 添加数值标注
    for i, (bar, val) in enumerate(zip(bars, mean_se_values)):
        ax3.text(bar.get_x() + bar.get_width()/2, val,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # 子图4：位置优化的边际贡献
    ax4 = axes[1, 1]
    one_stage_se = results['One-stage (greedy+kkt+k-means)']['mean_se']
    two_stage_se = results['Two-stage (greedy+kkt+L-BFGS-B)']['mean_se']
    position_contribution = 100 * (two_stage_se / one_stage_se - 1)

    ax4.plot(snr_db_range, position_contribution,
             color='#27ae60', marker='D', linestyle='-',
             linewidth=2, markersize=6, label='Position Optimization')
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax4.fill_between(snr_db_range, 0, position_contribution,
                     color='#27ae60', alpha=0.2)

    ax4.set_xlabel('SNR (dB)', fontsize=11)
    ax4.set_ylabel('Position Optimization Gain (%)', fontsize=11)
    ax4.set_title('Marginal Contribution of Position Optimization\n(Two-stage vs One-stage)',
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=9)

    plt.tight_layout()

    # 保存图片
    results_dir = Path(__file__).parent.parent / 'results'
    fig_path = results_dir / 'position_optimization_multi_snr.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 图表已保存到: {fig_path}")

    plt.close()


if __name__ == "__main__":
    print("\n提示: 此测试将运行约30-40分钟。")
    print("如需快速验证，可以先减少n_realizations或SNR点数。\n")

    # 可选：询问用户是否继续
    # response = input("是否继续? (y/n): ")
    # if response.lower() != 'y':
    #     print("测试已取消。")
    #     exit(0)

    results = test_position_optimization_multi_snr()

    print("\n" + "=" * 70)
    print("Phase 2 测试完成！")
    print("=" * 70)
    print("\n结果文件:")
    print("  - results/position_optimization_multi_snr.json")
    print("  - results/position_optimization_multi_snr.png")
    print("\n下一步:")
    print("  1. 查看结果图表，分析位置优化的效果")
    print("  2. 如果改进明显（+2-5%），可用于论文")
    print("  3. 如果改进不明显，考虑Phase 3优化（PSO/多起点等）")
