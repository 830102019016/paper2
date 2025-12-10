# -*- coding: utf-8 -*-
"""
验证 ILP 优化性能

对比 ILP (cvxpy) vs Greedy 的性能差异
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from config import config
from src_enhanced.integer_programming_decision import IntegerProgrammingDecision


def test_ilp_vs_greedy_performance():
    """测试 ILP vs Greedy 性能对比"""
    print("=" * 70)
    print("ILP vs Greedy 性能对比测试")
    print("=" * 70)

    # 创建测试场景
    K = 16
    np.random.seed(42)

    # 模拟配对信息
    pairs = []
    for k in range(K):
        weak_idx = 2 * k
        strong_idx = 2 * k + 1
        pairs.append((weak_idx, strong_idx))

    # 模拟信道增益（制造一些差异性，以便体现 ILP 优势）
    abs_gains = np.random.uniform(0.5, 2.0, 32)
    sat_gains = np.random.uniform(0.3, 1.5, 32)

    # 创建 ILP 决策器
    optimizer = IntegerProgrammingDecision()

    print(f"\n配置:")
    print(f"  配对数量: K = {K}")
    print(f"  用户数量: N = {2*K} = {2*K}")
    print(f"  ABS带宽: {config.Bd_options[1]/1e6:.2f} MHz")
    print(f"  卫星带宽: {config.Bs/1e6:.2f} MHz")
    print(f"  ABS功率: {config.Pd_dBm:.1f} dBm")
    print(f"  卫星功率: 20.0 dBm (测试值)")

    # 1. Greedy 优化
    print(f"\n{'='*70}")
    print("[1] Greedy 优化")
    print(f"{'='*70}")

    modes_greedy, rate_greedy = optimizer.optimize_greedy(
        pairs, abs_gains, sat_gains
    )

    print(f"\n结果:")
    print(f"  总速率: {rate_greedy:.6f} bps/Hz")

    # 统计各模式数量
    mode_names = {0: 'ABS NOMA', 1: 'ABS OMA (weak)', 2: 'ABS OMA (strong)', 3: 'Satellite'}
    mode_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for mode in modes_greedy:
        mode_counts[mode] += 1

    print(f"\n模式分布:")
    for mode, name in mode_names.items():
        print(f"    {name}: {mode_counts[mode]} pairs ({mode_counts[mode]/K*100:.1f}%)")

    # 2. ILP 优化
    print(f"\n{'='*70}")
    print("[2] ILP 优化 (cvxpy)")
    print(f"{'='*70}")

    try:
        modes_ilp, rate_ilp = optimizer.optimize_ilp(
            pairs, abs_gains, sat_gains
        )

        print(f"\n结果:")
        print(f"  总速率: {rate_ilp:.6f} bps/Hz")

        # 统计各模式数量
        mode_counts_ilp = {0: 0, 1: 0, 2: 0, 3: 0}
        for mode in modes_ilp:
            mode_counts_ilp[mode] += 1

        print(f"\n模式分布:")
        for mode, name in mode_names.items():
            print(f"    {name}: {mode_counts_ilp[mode]} pairs ({mode_counts_ilp[mode]/K*100:.1f}%)")

        # 3. 性能对比
        print(f"\n{'='*70}")
        print("[3] 性能对比")
        print(f"{'='*70}")

        improvement = (rate_ilp - rate_greedy) / rate_greedy * 100

        print(f"\n  Greedy 总速率: {rate_greedy:.6f} bps/Hz")
        print(f"  ILP 总速率:    {rate_ilp:.6f} bps/Hz")
        print(f"  性能提升:      {improvement:+.3f}%")

        if improvement > 0:
            print(f"\n  [PASS] ILP 优于 Greedy (提升 {improvement:.3f}%)")
        elif improvement == 0:
            print(f"\n  [INFO] ILP 与 Greedy 性能相同")
        else:
            print(f"\n  [WARN] ILP 性能低于 Greedy (降低 {-improvement:.3f}%)")

        # 4. 决策差异分析
        print(f"\n{'='*70}")
        print("[4] 决策差异分析")
        print(f"{'='*70}")

        diff_count = np.sum(modes_ilp != modes_greedy)
        print(f"\n  决策不同的配对数: {diff_count}/{K} ({diff_count/K*100:.1f}%)")

        if diff_count > 0:
            print(f"\n  差异详情:")
            for k in range(K):
                if modes_ilp[k] != modes_greedy[k]:
                    print(f"    Pair {k}: Greedy={mode_names[modes_greedy[k]]} -> ILP={mode_names[modes_ilp[k]]}")

    except Exception as e:
        print(f"\n  [ERROR] ILP 优化失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print("测试完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_ilp_vs_greedy_performance()
