"""
对比 Exhaustive 和 Greedy 模式选择器

目标：验证两者是否等价
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.satcon_system import SATCONSystem


def compare_selectors():
    """对比穷举和贪心"""

    print("=" * 80)
    print("对比 Exhaustive vs Greedy 模式选择器")
    print("=" * 80)

    # 测试配置
    test_snr = np.array([10, 20, 30])
    n_realizations = 10  # 减少到10次以加快速度

    print(f"\n测试参数:")
    print(f"  SNR 点: {test_snr}")
    print(f"  蒙特卡洛次数: {n_realizations}")
    print(f"  用户对数 K: {config.N // 2}")
    print(f"  穷举组合数: 4^{config.N // 2} = {4 ** (config.N // 2)}")

    configs = [
        ('greedy', 'uniform', 'Greedy+Uniform'),      # 先测试快的
        ('exhaustive', 'uniform', 'Exhaustive+Uniform')  # 再测试慢的
    ]

    results = {}

    import time

    for mode_sel, s2a_alloc, name in configs:
        print(f"\n{'='*80}")
        print(f"测试: {name}")
        print(f"{'='*80}")

        start_time = time.time()

        satcon = SATCONSystem(
            config,
            abs_bandwidth=1.2e6,
            mode_selector=mode_sel,
            s2a_allocator=s2a_alloc
        )

        print(f"\n开始仿真 (预计每次实现约 {'0.02s' if mode_sel == 'greedy' else '1-2s'})...")

        mean_rates, mean_se, std_rates, mode_stats = satcon.simulate_performance(
            snr_db_range=test_snr,
            elevation_deg=10,
            n_realizations=n_realizations,
            verbose=True
        )

        elapsed = time.time() - start_time

        results[name] = {
            'se': mean_se,
            'modes': mode_stats,
            'time': elapsed
        }

        print(f"\n完成！总用时: {elapsed:.1f}s (平均 {elapsed/(n_realizations*len(test_snr)):.2f}s/实现)")

    # 对比结果
    print("\n" + "=" * 80)
    print("性能对比")
    print("=" * 80)

    exhaustive_se = results['Exhaustive+Uniform']['se']
    greedy_se = results['Greedy+Uniform']['se']

    print(f"\n{'SNR (dB)':<10} {'Exhaustive':<15} {'Greedy':<15} {'差异 (%)':<15} {'等价?':<10}")
    print("-" * 80)

    all_equal = True
    for i, snr in enumerate(test_snr):
        diff_pct = (greedy_se[i] - exhaustive_se[i]) / exhaustive_se[i] * 100
        is_equal = abs(diff_pct) < 0.1  # 容忍 0.1% 的误差

        if not is_equal:
            all_equal = False

        equal_str = "是" if is_equal else "否"
        print(f"{snr:<10} {exhaustive_se[i]:<15.3f} {greedy_se[i]:<15.3f} "
              f"{diff_pct:<15.2f} {equal_str:<10}")

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)

    # 计算时间差异
    greedy_time = results['Greedy+Uniform']['time']
    exhaustive_time = results['Exhaustive+Uniform']['time']
    speedup = exhaustive_time / greedy_time

    print(f"\n时间对比:")
    print(f"  Greedy: {greedy_time:.1f}s")
    print(f"  Exhaustive: {exhaustive_time:.1f}s")
    print(f"  加速比: {speedup:.1f}x")

    if all_equal:
        print("\n" + "=" * 80)
        print("结论: Greedy 和 Exhaustive 性能等价（差异 < 0.1%）")
        print("=" * 80)
        print("\n论文写作建议:")
        print("\n【方案 1：诚实版（推荐）】")
        print("  标题: Greedy Mode Selection for ...")
        print("  算法: 明确说使用贪心算法")
        print("  对比: 与穷举搜索对比，证明贪心达到相同性能但快 {:.0f}x".format(speedup))
        print("  优势: 诚实、有对比实验、突出计算效率")
        print("\n【方案 2：理论+实践版】")
        print("  算法描述: '理论上采用穷举搜索获得全局最优'")
        print("  实现: '由于 pair 间独立性，贪心算法可达到相同性能'")
        print("  验证: '仿真验证贪心与穷举性能等价（表X）'")
        print("  优势: 理论严谨，实践高效")
        print("\n【不推荐：仿真用贪心，论文只写穷举】")
        print("  风险: 审稿人可能要求提供穷举的仿真结果")
        print("  问题: 如果被发现，可能质疑研究诚信")
    else:
        print("\n" + "=" * 80)
        print("结论: Greedy 和 Exhaustive 性能不等价")
        print("=" * 80)
        print("\n差异最大的 SNR 点:")
        max_diff_idx = np.argmax(np.abs((greedy_se - exhaustive_se) / exhaustive_se))
        print(f"  SNR={test_snr[max_diff_idx]}dB: "
              f"差异={(greedy_se[max_diff_idx] - exhaustive_se[max_diff_idx]) / exhaustive_se[max_diff_idx] * 100:.2f}%")
        print("\n论文写作建议:")
        print("  - 必须诚实说明使用的算法")
        print("  - 如果用 Greedy: 标题写 'Greedy'，说明是高效近似")
        print("  - 如果要最优性能: 用 Exhaustive，接受长仿真时间")
        print("  - 可以两者都测试，展示性能-复杂度权衡")


if __name__ == "__main__":
    compare_selectors()
