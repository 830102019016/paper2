"""
寻找最优优化器组合

目标：系统性地测试所有"模式选择器 + S2A分配器"组合，
     找到在不同SNR下性能最好的配置，用于论文发表对比

测试矩阵：
  模式选择器: heuristic, greedy, exhaustive
  S2A分配器: uniform, kkt, waterfilling
  共 3×3 = 9 种组合

输出：
  1. 性能对比表格（所有组合 vs baseline）
  2. 每个SNR点的最优配置
  3. 推荐的论文对比方案
"""

import numpy as np
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.satcon_system import SATCONSystem


def test_all_combinations():
    """测试所有优化器组合"""

    print("=" * 80)
    print("系统性测试所有优化器组合")
    print("=" * 80)

    # 定义测试矩阵
    mode_selectors = ['heuristic', 'greedy']  # 先不测试 exhaustive（太慢）
    s2a_allocators = ['uniform', 'kkt', 'waterfilling']

    # SNR 测试点
    test_snr = np.array([10, 20, 30])
    n_realizations = 10  # 增加到10次以获得更稳定结果

    # 存储结果
    results = {}

    # 测试所有组合
    total_configs = len(mode_selectors) * len(s2a_allocators)
    current_config = 0

    for mode_sel in mode_selectors:
        for s2a_alloc in s2a_allocators:
            current_config += 1
            config_name = f"{mode_sel}+{s2a_alloc}"

            print(f"\n{'='*80}")
            print(f"[{current_config}/{total_configs}] 测试配置: {config_name}")
            print(f"  模式选择: {mode_sel}")
            print(f"  S2A分配: {s2a_alloc}")
            print(f"{'='*80}")

            try:
                # 创建系统
                start_time = time.time()

                satcon = SATCONSystem(
                    config,
                    abs_bandwidth=1.2e6,
                    mode_selector=mode_sel,
                    s2a_allocator=s2a_alloc
                )

                # 运行仿真
                print(f"\n运行仿真 (SNR={test_snr} dB, {n_realizations}次MC)...")

                mean_rates, mean_se, std_rates, mode_stats = satcon.simulate_performance(
                    snr_db_range=test_snr,
                    elevation_deg=10,
                    n_realizations=n_realizations,
                    verbose=False  # 关闭进度条以减少输出
                )

                elapsed_time = time.time() - start_time

                # 保存结果
                results[config_name] = {
                    'mode_selector': mode_sel,
                    's2a_allocator': s2a_alloc,
                    'se': mean_se,
                    'se_std': np.std(mean_se),
                    'modes': mode_stats,
                    'time': elapsed_time
                }

                # 打印结果
                print(f"\n结果:")
                for i, snr in enumerate(test_snr):
                    print(f"  SNR={snr}dB: SE={mean_se[i]:.3f} bits/s/Hz")
                print(f"  运行时间: {elapsed_time:.2f}s")

            except Exception as e:
                print(f"\n❌ 错误: {config_name} 测试失败")
                print(f"  异常: {e}")
                import traceback
                traceback.print_exc()

    return results, test_snr


def analyze_results(results, test_snr):
    """分析结果并找出最优配置"""

    print("\n\n" + "=" * 80)
    print("完整性能对比表格")
    print("=" * 80)

    # 找到 baseline
    baseline_name = 'heuristic+uniform'
    baseline_se = results[baseline_name]['se'] if baseline_name in results else None

    # 打印表头
    print(f"\n{'配置':<25} {'模式选择':<15} {'S2A分配':<15} ", end='')
    for snr in test_snr:
        print(f"{'SNR='+str(snr)+'dB':<15}", end='')
    print(f"{'平均增益':<15} {'运行时间':<10}")
    print("-" * 140)

    # 打印每个配置的结果
    for name, data in sorted(results.items()):
        se_values = data['se']

        # 计算增益
        if baseline_se is not None:
            gains = [(se - baseline_se[i]) / baseline_se[i] * 100
                     for i, se in enumerate(se_values)]
            avg_gain = np.mean(gains)
        else:
            gains = [0] * len(se_values)
            avg_gain = 0

        # 打印
        print(f"{name:<25} {data['mode_selector']:<15} {data['s2a_allocator']:<15} ", end='')
        for se in se_values:
            print(f"{se:.3f}         ", end='')

        gain_str = f"{avg_gain:+.1f}%" if avg_gain != 0 else "baseline"
        time_str = f"{data['time']:.1f}s"
        print(f"{gain_str:<15} {time_str:<10}")

    print("\n" + "=" * 80)
    print("每个 SNR 点的最优配置")
    print("=" * 80)

    best_configs = {}

    for i, snr in enumerate(test_snr):
        # 找到该 SNR 下 SE 最大的配置
        best_name = max(results.keys(), key=lambda k: results[k]['se'][i])
        best_se = results[best_name]['se'][i]

        # 计算相比 baseline 的增益
        if baseline_se is not None:
            gain = (best_se - baseline_se[i]) / baseline_se[i] * 100
        else:
            gain = 0

        best_configs[snr] = {
            'config': best_name,
            'se': best_se,
            'gain': gain
        }

        print(f"\nSNR = {snr} dB:")
        print(f"  最优配置: {best_name}")
        print(f"  SE: {best_se:.3f} bits/s/Hz")
        print(f"  增益: {gain:+.1f}% (vs baseline)")
        print(f"  模式分布: ", end='')
        modes = results[best_name]['modes']
        print(f"NOMA={modes['noma'][i]:.1f}, "
              f"OMA-W={modes['oma_weak'][i]:.1f}, "
              f"OMA-S={modes['oma_strong'][i]:.1f}, "
              f"SAT={modes['sat'][i]:.1f}")

    print("\n" + "=" * 80)
    print("论文发表建议")
    print("=" * 80)

    # 找到整体表现最好的配置（平均增益最大）
    best_overall = max(
        [k for k in results.keys() if k != baseline_name],
        key=lambda k: np.mean([
            (results[k]['se'][i] - baseline_se[i]) / baseline_se[i] * 100
            for i in range(len(test_snr))
        ])
    )

    avg_gain_overall = np.mean([
        (results[best_overall]['se'][i] - baseline_se[i]) / baseline_se[i] * 100
        for i in range(len(test_snr))
    ])

    print(f"\n1. 推荐的主要对比方案:")
    print(f"   {best_overall}")
    print(f"   平均增益: {avg_gain_overall:+.1f}%")
    print(f"   论文中可以重点展示这个配置 vs baseline")

    print(f"\n2. 分场景对比:")
    for snr, info in best_configs.items():
        if info['config'] != best_overall:
            print(f"   SNR={snr}dB: {info['config']} 表现更好 (增益 {info['gain']:+.1f}%)")

    print(f"\n3. 论文写作建议:")
    print(f"   - 主图：展示 {best_overall} vs baseline 的完整 SNR 曲线")
    print(f"   - 补充表格：展示所有配置在关键 SNR 点的性能对比")
    print(f"   - 分析：解释为什么不同 SNR 下最优配置不同")

    # 找到计算效率最高的配置（性能/时间比）
    efficiency_scores = {}
    for name, data in results.items():
        if name != baseline_name:
            avg_gain = np.mean([
                (data['se'][i] - baseline_se[i]) / baseline_se[i] * 100
                for i in range(len(test_snr))
            ])
            efficiency = avg_gain / data['time'] if data['time'] > 0 else 0
            efficiency_scores[name] = efficiency

    best_efficiency = max(efficiency_scores.keys(), key=lambda k: efficiency_scores[k])

    print(f"\n4. 实时系统推荐:")
    print(f"   {best_efficiency}")
    print(f"   性能/时间比最优 (效率得分: {efficiency_scores[best_efficiency]:.2f})")

    print("\n" + "=" * 80)

    return best_configs, best_overall


def main():
    """主函数"""
    # 运行所有测试
    results, test_snr = test_all_combinations()

    if not results:
        print("\n❌ 没有成功的测试结果")
        return

    # 分析结果
    best_configs, best_overall = analyze_results(results, test_snr)

    # 保存结果到文件
    import json
    output_file = Path(__file__).parent.parent / 'results' / 'optimizer_combinations.json'
    output_file.parent.mkdir(exist_ok=True)

    # 转换为可序列化格式
    serializable_results = {}
    for name, data in results.items():
        serializable_results[name] = {
            'mode_selector': data['mode_selector'],
            's2a_allocator': data['s2a_allocator'],
            'se': data['se'].tolist(),
            'se_std': float(data['se_std']),
            'time': data['time'],
            'modes': {k: v.tolist() for k, v in data['modes'].items()}
        }

    with open(output_file, 'w') as f:
        json.dump({
            'snr_range': test_snr.tolist(),
            'results': serializable_results,
            'best_overall': best_overall
        }, f, indent=2)

    print(f"\n结果已保存到: {output_file}")
    print("\n✓ 测试完成！")


if __name__ == "__main__":
    main()
