"""
Phase 1: ABS位置连续优化快速验证

对比实验：
1. Baseline: k-means/k-medoids + 网格搜索高度
2. Hybrid: k-means → L-BFGS-B（推荐配置）

固定配置（论文最终方案）：
- mode_selector='greedy'   # 一阶段：贪心模式选择
- s2a_allocator='kkt'      # 一阶段：KKT资源分配
- N=32, K=16
- Bd=1.2 MHz
- SNR=20 dB (单点测试)
- n_realizations=10 (快速验证)

预期结果：
- Hybrid方法相比Baseline改进：+2-5%
- 计算时间：Hybrid约为Baseline的2-3倍
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


def test_position_optimization_single_snr():
    """
    单点SNR测试：对比baseline vs hybrid

    【推荐配置】：
    - mode_selector='greedy'
    - s2a_allocator='kkt'
    - position_optimizer=None/'hybrid'
    """
    print("=" * 70)
    print("Phase 1: ABS位置优化快速验证（单点SNR测试）")
    print("=" * 70)

    # 测试配置
    snr_db = 20
    elevation_deg = 10
    abs_bandwidth = 1.2e6
    n_realizations = 10  # 快速验证

    print(f"\n【测试配置】")
    print(f"  用户数: {config.N}")
    print(f"  ABS带宽: {abs_bandwidth/1e6:.1f} MHz")
    print(f"  SNR: {snr_db} dB")
    print(f"  仰角: {elevation_deg}°")
    print(f"  实现次数: {n_realizations}")
    print(f"  模式选择: greedy（推荐）")
    print(f"  资源分配: kkt（推荐）")

    # 方法列表
    methods = {
        'Baseline (k-means)': None,
        'L-BFGS-B (pure)': 'L-BFGS-B'
    }

    results = {}

    for method_name, position_optimizer in methods.items():
        print(f"\n{'=' * 70}")
        print(f"测试方法: {method_name}")
        print(f"{'=' * 70}")

        # 创建SATCON系统（推荐配置）
        satcon = SATCONSystem(
            config,
            abs_bandwidth=abs_bandwidth,
            mode_selector='greedy',      # 【推荐】一阶段：贪心模式选择
            s2a_allocator='kkt',         # 【推荐】一阶段：KKT资源分配
            position_optimizer=position_optimizer  # 二阶段：位置优化
        )

        # 统计数据
        sum_rates = []
        computation_times = []

        print(f"\n运行 {n_realizations} 次仿真...")

        for r in range(n_realizations):
            seed = config.random_seed + r

            # 记录时间
            start_time = time.time()

            # 单次仿真
            sum_rate, mode_stats = satcon.simulate_single_realization(
                snr_db, elevation_deg, seed
            )

            elapsed_time = time.time() - start_time

            # 记录结果
            sum_rates.append(sum_rate)
            computation_times.append(elapsed_time)

            # 显示进度
            print(f"  [{r+1}/{n_realizations}] Sum Rate: {sum_rate/1e6:.2f} Mbps, "
                  f"Time: {elapsed_time:.2f}s", end='\r')

        print()  # 换行

        # 统计结果
        mean_sum_rate = np.mean(sum_rates)
        std_sum_rate = np.std(sum_rates)
        mean_se = mean_sum_rate / config.Bs
        mean_time = np.mean(computation_times)

        results[method_name] = {
            'mean_sum_rate': mean_sum_rate,
            'std_sum_rate': std_sum_rate,
            'mean_se': mean_se,
            'mean_time': mean_time,
            'sum_rates': sum_rates,
            'computation_times': computation_times
        }

        print(f"\n【结果】")
        print(f"  平均总速率: {mean_sum_rate/1e6:.2f} ± {std_sum_rate/1e6:.2f} Mbps")
        print(f"  频谱效率: {mean_se:.2f} bits/s/Hz")
        print(f"  平均计算时间: {mean_time:.2f}s")

    # 对比分析
    print(f"\n{'=' * 70}")
    print("【对比分析】")
    print(f"{'=' * 70}")

    baseline_rate = results['Baseline (k-means)']['mean_sum_rate']
    lbfgsb_rate = results['L-BFGS-B (pure)']['mean_sum_rate']

    improvement_abs = lbfgsb_rate - baseline_rate
    improvement_pct = 100 * (lbfgsb_rate / baseline_rate - 1)

    baseline_time = results['Baseline (k-means)']['mean_time']
    lbfgsb_time = results['L-BFGS-B (pure)']['mean_time']
    time_ratio = lbfgsb_time / baseline_time

    print(f"\n性能改进:")
    print(f"  Baseline (k-means): {baseline_rate/1e6:.2f} Mbps")
    print(f"  L-BFGS-B (pure):    {lbfgsb_rate/1e6:.2f} Mbps")
    print(f"  改进:               +{improvement_abs/1e6:.2f} Mbps (+{improvement_pct:.2f}%)")

    print(f"\n计算成本:")
    print(f"  Baseline: {baseline_time:.2f}s")
    print(f"  L-BFGS-B: {lbfgsb_time:.2f}s")
    print(f"  比例:     {time_ratio:.2f}×")

    print(f"\n结论:")
    if improvement_pct >= 2.0:
        print(f"  [SUCCESS] L-BFGS-B方法有效改进了系统性能 (+{improvement_pct:.2f}%)")
    elif improvement_pct >= 0.5:
        print(f"  [OK] L-BFGS-B方法略有改进 (+{improvement_pct:.2f}%)")
    else:
        print(f"  [WARNING] L-BFGS-B方法改进不明显 (+{improvement_pct:.2f}%)")

    if time_ratio <= 3.0:
        print(f"  [OK] 计算成本可接受 ({time_ratio:.2f}x baseline)")
    else:
        print(f"  [WARNING] 计算成本较高 ({time_ratio:.2f}x baseline)")

    # 保存结果
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    # 保存JSON
    results_json = {
        'config': {
            'snr_db': snr_db,
            'elevation_deg': elevation_deg,
            'abs_bandwidth_MHz': abs_bandwidth / 1e6,
            'n_realizations': n_realizations,
            'mode_selector': 'greedy',
            's2a_allocator': 'kkt'
        },
        'results': {
            method_name: {
                'mean_sum_rate_Mbps': res['mean_sum_rate'] / 1e6,
                'std_sum_rate_Mbps': res['std_sum_rate'] / 1e6,
                'mean_se_bps_Hz': res['mean_se'],
                'mean_time_s': res['mean_time']
            }
            for method_name, res in results.items()
        },
        'comparison': {
            'improvement_abs_Mbps': improvement_abs / 1e6,
            'improvement_pct': improvement_pct,
            'time_ratio': time_ratio
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    json_path = results_dir / 'position_optimization_test.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 结果已保存到: {json_path}")

    # 可视化
    visualize_results(results, abs_bandwidth)

    return results


def visualize_results(results, abs_bandwidth):
    """可视化对比结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = list(results.keys())
    colors = ['#3498db', '#e74c3c']

    # 子图1：总速率对比
    ax1 = axes[0]
    mean_rates = [results[m]['mean_sum_rate'] / 1e6 for m in methods]
    std_rates = [results[m]['std_sum_rate'] / 1e6 for m in methods]

    bars = ax1.bar(range(len(methods)), mean_rates, yerr=std_rates,
                   color=colors, alpha=0.7, capsize=5)

    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.set_ylabel('Sum Rate (Mbps)')
    ax1.set_title(f'System Performance Comparison\n(Bd={abs_bandwidth/1e6:.1f} MHz, SNR=20dB)')
    ax1.grid(axis='y', alpha=0.3)

    # 添加数值标注
    for i, (bar, val) in enumerate(zip(bars, mean_rates)):
        ax1.text(bar.get_x() + bar.get_width()/2, val,
                f'{val:.2f}', ha='center', va='bottom')

    # 子图2：计算时间对比
    ax2 = axes[1]
    mean_times = [results[m]['mean_time'] for m in methods]

    bars2 = ax2.bar(range(len(methods)), mean_times,
                    color=colors, alpha=0.7)

    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.set_ylabel('Computation Time (s)')
    ax2.set_title('Computation Cost Comparison')
    ax2.grid(axis='y', alpha=0.3)

    # 添加数值标注
    for i, (bar, val) in enumerate(zip(bars2, mean_times)):
        ax2.text(bar.get_x() + bar.get_width()/2, val,
                f'{val:.2f}s', ha='center', va='bottom')

    plt.tight_layout()

    # 保存图片
    results_dir = Path(__file__).parent.parent / 'results'
    fig_path = results_dir / 'position_optimization_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 图表已保存到: {fig_path}")

    plt.close()


if __name__ == "__main__":
    results = test_position_optimization_single_snr()

    print("\n" + "=" * 70)
    print("Phase 1 测试完成！")
    print("=" * 70)
    print("\n下一步:")
    print("  1. 检查改进是否达到预期（+2-5%）")
    print("  2. 如果效果好，进行多SNR点测试")
    print("  3. 考虑调整L-BFGS-B迭代次数（当前20次）")
    print("  4. 可选：实现PSO方法对比")
