"""
优化器集成测试脚本

测试可插拔的模式选择器和 S2A 资源分配器是否正常工作

测试方案：
1. Baseline: heuristic + uniform
2. Opt-M: exhaustive + uniform (仅优化模式选择)
3. Opt-S: heuristic + kkt (仅优化S2A分配)
4. Opt-Full: exhaustive + kkt (全优化)
5. Opt-Fast: greedy + kkt (低复杂度全优化)
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.satcon_system import SATCONSystem


def test_single_configuration(mode_selector, s2a_allocator, name):
    """测试单个配置"""
    print(f"\n{'='*60}")
    print(f"测试配置: {name}")
    print(f"  模式选择器: {mode_selector}")
    print(f"  S2A分配器: {s2a_allocator}")
    print(f"{'='*60}")

    # 创建系统
    satcon = SATCONSystem(
        config,
        abs_bandwidth=1.2e6,
        mode_selector=mode_selector,
        s2a_allocator=s2a_allocator
    )

    # 快速测试（3个SNR点，5次实现）
    test_snr = np.array([10, 20, 30])
    print(f"\n运行快速测试 (SNR={test_snr} dB, 5次蒙特卡洛)...")

    mean_rates, mean_se, std_rates, mode_stats = satcon.simulate_performance(
        snr_db_range=test_snr,
        elevation_deg=10,
        n_realizations=5,
        verbose=True
    )

    print(f"\n结果:")
    for i, snr in enumerate(test_snr):
        print(f"  SNR={snr}dB:")
        print(f"    SE: {mean_se[i]:.3f} bits/s/Hz")
        print(f"    模式分布: NOMA={mode_stats['noma'][i]:.1f}, "
              f"OMA-W={mode_stats['oma_weak'][i]:.1f}, "
              f"OMA-S={mode_stats['oma_strong'][i]:.1f}, "
              f"SAT={mode_stats['sat'][i]:.1f}")

    return mean_se


def main():
    """主测试函数"""
    print("=" * 80)
    print("SATCON 优化器集成测试")
    print("=" * 80)

    # 测试配置
    test_configs = [
        ('heuristic', 'uniform', 'Baseline'),
        ('exhaustive', 'uniform', 'Opt-M (仅模式优化)'),
        ('heuristic', 'kkt', 'Opt-S (仅S2A优化)'),
        ('exhaustive', 'kkt', 'Opt-Full (全优化)'),
        ('greedy', 'kkt', 'Opt-Fast (快速优化)')
    ]

    results = {}

    for mode_sel, s2a_alloc, name in test_configs:
        try:
            se_values = test_single_configuration(mode_sel, s2a_alloc, name)
            results[name] = se_values
        except Exception as e:
            print(f"\n错误: {name} 测试失败")
            print(f"  异常: {e}")
            import traceback
            traceback.print_exc()

    # 对比总结
    print("\n" + "=" * 80)
    print("性能对比总结")
    print("=" * 80)

    if results:
        # 获取 baseline 作为参考
        baseline_se = results.get('Baseline', None)

        print(f"\n{'配置':<25} {'SNR=10dB':<15} {'SNR=20dB':<15} {'SNR=30dB':<15}")
        print("-" * 80)

        for name, se_values in results.items():
            se_str = [f"{se:.3f}" for se in se_values]
            print(f"{name:<25} {se_str[0]:<15} {se_str[1]:<15} {se_str[2]:<15}")

        # 增益对比
        if baseline_se is not None:
            print("\n" + "-" * 80)
            print(f"{'配置':<25} {'增益@10dB':<15} {'增益@20dB':<15} {'增益@30dB':<15}")
            print("-" * 80)

            for name, se_values in results.items():
                if name != 'Baseline':
                    gains = [(se - baseline_se[i]) / baseline_se[i] * 100
                             for i, se in enumerate(se_values)]
                    gain_str = [f"+{g:.1f}%" for g in gains]
                    print(f"{name:<25} {gain_str[0]:<15} {gain_str[1]:<15} {gain_str[2]:<15}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
