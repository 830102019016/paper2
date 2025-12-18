"""
对比 Exhaustive 和 Greedy（小规模用户）

使用 N=8 用户（K=4 对），使得穷举搜索可行（256 种组合）
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# 临时修改配置，减少用户数
import config as config_module

# 创建小规模配置（复制原配置对象）
small_config = type('obj', (object,), {})()
for attr in dir(config_module.config):
    if not attr.startswith('_'):
        setattr(small_config, attr, getattr(config_module.config, attr))

# 修改用户数
small_config.N = 8  # 8个用户，4对

from src.satcon_system import SATCONSystem


def compare_selectors_small():
    """对比穷举和贪心（小规模）"""

    print("=" * 80)
    print("对比 Exhaustive vs Greedy（小规模 N=8）")
    print("=" * 80)

    # 测试配置
    test_snr = np.array([10, 20, 30])
    n_realizations = 20

    K = small_config.N // 2
    total_combos = 4 ** K

    print(f"\n测试参数:")
    print(f"  用户数 N: {small_config.N}")
    print(f"  用户对数 K: {K}")
    print(f"  穷举组合数: 4^{K} = {total_combos}")
    print(f"  SNR 点: {test_snr}")
    print(f"  蒙特卡洛次数: {n_realizations}")

    configs = [
        ('greedy', 'uniform', 'Greedy+Uniform'),
        ('exhaustive', 'uniform', 'Exhaustive+Uniform')
    ]

    results = {}
    import time

    for mode_sel, s2a_alloc, name in configs:
        print(f"\n{'='*80}")
        print(f"测试: {name}")
        print(f"{'='*80}")

        start_time = time.time()

        satcon = SATCONSystem(
            small_config,  # 使用小配置
            abs_bandwidth=1.2e6,
            mode_selector=mode_sel,
            s2a_allocator=s2a_alloc
        )

        print(f"\n开始仿真...")

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
    max_diff = 0

    for i, snr in enumerate(test_snr):
        diff_pct = (greedy_se[i] - exhaustive_se[i]) / exhaustive_se[i] * 100
        is_equal = abs(diff_pct) < 0.5  # 容忍 0.5% 的误差（蒙特卡洛波动）

        if not is_equal:
            all_equal = False

        max_diff = max(max_diff, abs(diff_pct))
        equal_str = "是" if is_equal else "否"
        print(f"{snr:<10} {exhaustive_se[i]:<15.3f} {greedy_se[i]:<15.3f} "
              f"{diff_pct:<+15.2f} {equal_str:<10}")

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

    print(f"\n性能差异:")
    print(f"  最大差异: {max_diff:.2f}%")
    print(f"  是否等价（< 0.5%）: {'是' if all_equal else '否'}")

    if all_equal:
        print("\n" + "=" * 80)
        print("结论: Greedy 和 Exhaustive 性能等价")
        print("=" * 80)
        print("\n论文写作建议:")
        print("\n【方案 1：诚实+高效（推荐）】")
        print("  - 标题: 'Efficient Greedy Mode Selection for...'")
        print("  - 算法: 明确使用贪心算法")
        print("  - 验证: 小规模穷举验证（N=8）证明贪心达到全局最优")
        print("  - 大规模: N=32 只能用贪心（穷举不可行）")
        print(f"  - 优势: 诚实 + 加速{speedup:.0f}倍 + 可扩展")

        print("\n【方案 2：理论严谨版】")
        print("  - 问题形式化: '最大化总速率的模式选择'")
        print("  - 理论分析: '由于pair间独立性，贪心可达全局最优'")
        print("  - 证明: '定理1：贪心算法的最优性'")
        print("  - 验证: '仿真验证（表X）'")

        print("\n【方案 3：实用版】")
        print("  - 算法: 'Pair-wise Optimal Mode Selection'")
        print("  - 描述: '对每个pair独立选择最优模式'")
        print("  - 不提'贪心'或'穷举'，强调实用性")

    else:
        print("\n" + "=" * 80)
        print("警告: Greedy 和 Exhaustive 性能不等价")
        print("=" * 80)
        print("\n可能原因:")
        print("  - 蒙特卡洛随机性（增加仿真次数）")
        print("  - pair间存在耦合（需要联合优化）")
        print("\n建议:")
        print("  - 增加仿真次数到 100+ 以减少随机性")
        print("  - 如果仍不等价，必须用穷举（或设计更好算法）")


if __name__ == "__main__":
    compare_selectors_small()
