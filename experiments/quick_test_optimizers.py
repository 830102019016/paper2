"""
优化器快速测试脚本

快速验证优化器是否正常工作，使用较少的仿真次数

测试方案（精简版）：
1. Baseline: heuristic + uniform
2. Opt-S: heuristic + kkt (仅优化S2A分配)
3. Opt-Fast: greedy + kkt (快速全优化)
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.satcon_system import SATCONSystem


def quick_test():
    """快速测试三个关键配置"""
    print("=" * 80)
    print("SATCON 优化器快速测试")
    print("=" * 80)

    # 测试配置（精简版）
    test_configs = [
        ('heuristic', 'uniform', 'Baseline'),
        ('heuristic', 'kkt', 'Opt-S (仅S2A优化)'),
        ('greedy', 'kkt', 'Opt-Fast (快速全优化)')
    ]

    # 快速测试参数
    test_snr = np.array([10, 20, 30])
    n_realizations = 3  # 只运行3次（而不是5次）

    results = {}

    for mode_sel, s2a_alloc, name in test_configs:
        print(f"\n{'='*60}")
        print(f"配置: {name}")
        print(f"  模式选择器: {mode_sel}")
        print(f"  S2A分配器: {s2a_alloc}")
        print(f"{'='*60}")

        try:
            # 创建系统
            satcon = SATCONSystem(
                config,
                abs_bandwidth=1.2e6,
                mode_selector=mode_sel,
                s2a_allocator=s2a_alloc
            )

            print(f"\n运行测试 (SNR={test_snr} dB, {n_realizations}次蒙特卡洛)...")

            # 运行仿真
            mean_rates, mean_se, std_rates, mode_stats = satcon.simulate_performance(
                snr_db_range=test_snr,
                elevation_deg=10,
                n_realizations=n_realizations,
                verbose=True
            )

            # 保存结果
            results[name] = {
                'se': mean_se,
                'modes': mode_stats
            }

            # 打印结果
            print(f"\n结果:")
            for i, snr in enumerate(test_snr):
                print(f"  SNR={snr}dB:")
                print(f"    SE: {mean_se[i]:.3f} bits/s/Hz")
                print(f"    模式分布: NOMA={mode_stats['noma'][i]:.1f}, "
                      f"OMA-W={mode_stats['oma_weak'][i]:.1f}, "
                      f"OMA-S={mode_stats['oma_strong'][i]:.1f}, "
                      f"SAT={mode_stats['sat'][i]:.1f}")

        except Exception as e:
            print(f"\n❌ 错误: {name} 测试失败")
            print(f"  异常: {e}")
            import traceback
            traceback.print_exc()

    # 对比总结
    print("\n" + "=" * 80)
    print("性能对比总结")
    print("=" * 80)

    if results:
        baseline_se = results.get('Baseline', {}).get('se', None)

        print(f"\n{'配置':<30} {'SNR=10dB':<15} {'SNR=20dB':<15} {'SNR=30dB':<15}")
        print("-" * 80)

        for name, data in results.items():
            se_values = data['se']
            se_str = [f"{se:.3f}" for se in se_values]
            print(f"{name:<30} {se_str[0]:<15} {se_str[1]:<15} {se_str[2]:<15}")

        # 增益对比
        if baseline_se is not None and len(baseline_se) > 0:
            print("\n" + "-" * 80)
            print(f"{'配置':<30} {'增益@10dB':<15} {'增益@20dB':<15} {'增益@30dB':<15}")
            print("-" * 80)

            for name, data in results.items():
                if name != 'Baseline':
                    se_values = data['se']
                    gains = [(se - baseline_se[i]) / baseline_se[i] * 100
                             for i, se in enumerate(se_values)]
                    gain_str = [f"+{g:.1f}%" if g >= 0 else f"{g:.1f}%"
                                for g in gains]
                    print(f"{name:<30} {gain_str[0]:<15} {gain_str[1]:<15} {gain_str[2]:<15}")

    print("\n" + "=" * 80)
    print("✓ 测试完成！")
    print("=" * 80)

    # 验证优化器有效性
    print("\n验证:")
    if baseline_se is not None and 'Opt-S (仅S2A优化)' in results:
        opt_s_se = results['Opt-S (仅S2A优化)']['se']
        improvement = any(opt_s_se[i] >= baseline_se[i] - 1e-6
                         for i in range(len(baseline_se)))
        print(f"  ✓ S2A优化器有效: {improvement}")

    if baseline_se is not None and 'Opt-Fast (快速全优化)' in results:
        opt_fast_se = results['Opt-Fast (快速全优化)']['se']
        improvement = any(opt_fast_se[i] >= baseline_se[i] - 1e-6
                         for i in range(len(baseline_se)))
        print(f"  ✓ 快速优化器有效: {improvement}")


if __name__ == "__main__":
    quick_test()
