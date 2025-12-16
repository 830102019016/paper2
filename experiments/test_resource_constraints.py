# -*- coding: utf-8 -*-
"""
测试资源约束集成
验证ResourceConstrainedDecision正确集成到JointOptimizationSATCON

目标：
1. 验证资源约束参数正确传递
2. 验证ABS容量约束生效
3. 验证S2A回程约束生效
4. 验证性能提升
"""
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src_enhanced.joint_satcon_system import JointOptimizationSATCON

def test_integration():
    """测试资源约束集成"""
    print("=" * 80)
    print("测试资源约束集成")
    print("=" * 80)

    # 测试参数
    snr_db = 20
    elevation_deg = 10
    abs_bandwidth = 1.2e6
    n_mc = 10  # 快速测试

    print(f"\n配置:")
    print(f"  SNR: {snr_db} dB")
    print(f"  仰角: {elevation_deg}°")
    print(f"  ABS带宽: {abs_bandwidth/1e6:.1f} MHz")
    print(f"  MC次数: {n_mc}")
    print(f"  用户数: {config.N}")

    # 创建测试场景
    scenarios = {
        '无约束（基线）': {
            'max_abs_users': None,
            'max_s2a_capacity': None
        },
        'ABS容量=8': {
            'max_abs_users': 8,
            'max_s2a_capacity': None
        },
        'ABS容量=4': {
            'max_abs_users': 4,
            'max_s2a_capacity': None
        },
        'S2A回程=20Mbps': {
            'max_abs_users': None,
            'max_s2a_capacity': 20e6  # 20 Mbps
        },
        '双重约束(8+20M)': {
            'max_abs_users': 8,
            'max_s2a_capacity': 20e6
        }
    }

    print(f"\n测试场景:")
    for i, (name, params) in enumerate(scenarios.items(), 1):
        abs_info = f"ABS容量={params['max_abs_users']}" if params['max_abs_users'] else "无ABS约束"
        s2a_info = f"S2A={params['max_s2a_capacity']/1e6:.0f}M" if params['max_s2a_capacity'] else "无S2A约束"
        print(f"  {i}. {name}: {abs_info}, {s2a_info}")

    # 运行测试
    print("\n" + "=" * 80)
    print("开始测试...")
    print("=" * 80)

    results = {}

    for name, params in scenarios.items():
        print(f"\n[测试] {name}")
        print("-" * 60)

        try:
            # 创建系统
            system = JointOptimizationSATCON(
                config, abs_bandwidth,
                use_module1=True,
                use_module2=True,
                use_module3=True,
                max_abs_users=params['max_abs_users'],
                max_s2a_capacity=params['max_s2a_capacity']
            )

            # 运行单次实现测试
            seed = config.random_seed
            sum_rate, mode_stats = system.simulate_single_realization(
                snr_db, elevation_deg, seed, use_joint_optimization=False
            )

            rate_mbps = sum_rate / 1e6
            abs_users = mode_stats['noma'] + mode_stats['oma_weak'] + mode_stats['oma_strong']

            results[name] = {
                'rate': sum_rate,
                'rate_mbps': rate_mbps,
                'mode_stats': mode_stats,
                'abs_users': abs_users
            }

            print(f"  [OK] 总速率: {rate_mbps:.2f} Mbps")
            print(f"  [OK] ABS用户数: {abs_users}/16")
            print(f"  [OK] 模式分布: NOMA={mode_stats['noma']}, "
                  f"OMA_W={mode_stats['oma_weak']}, "
                  f"OMA_S={mode_stats['oma_strong']}, "
                  f"SAT={mode_stats['sat']}")

            # 验证约束是否生效
            if params['max_abs_users'] is not None:
                if abs_users <= params['max_abs_users']:
                    print(f"  [OK] ABS容量约束满足: {abs_users} <= {params['max_abs_users']}")
                else:
                    print(f"  [ERROR] ABS容量约束违反: {abs_users} > {params['max_abs_users']}")

        except Exception as e:
            print(f"  [ERROR] 测试失败: {e}")
            import traceback
            traceback.print_exc()

    # 对比结果
    print("\n" + "=" * 80)
    print("性能对比")
    print("=" * 80)

    baseline_rate = results['无约束（基线）']['rate']

    print(f"\n{'场景':<20} {'速率(Mbps)':<15} {'ABS用户':<12} {'提升(%)':<10}")
    print("-" * 80)

    for name, data in results.items():
        improvement = (data['rate'] - baseline_rate) / baseline_rate * 100
        print(f"{name:<20} {data['rate_mbps']:<15.2f} {data['abs_users']:<12} {improvement:>8.2f}%")

    # 验证结论
    print("\n" + "=" * 80)
    print("测试结论")
    print("=" * 80)

    # 检查1：约束场景速率是否下降
    constrained_lower = all(
        results[name]['rate'] <= baseline_rate
        for name in scenarios.keys()
        if name != '无约束（基线）'
    )

    # 检查2：更严格的约束导致更低的速率
    abs8_rate = results['ABS容量=8']['rate']
    abs4_rate = results['ABS容量=4']['rate']
    constraint_order_correct = abs4_rate <= abs8_rate

    if constrained_lower:
        print("  [OK] 资源约束导致速率下降（符合预期）")
    else:
        print("  [WARNING] 某些约束场景速率高于基线（可能需要检查）")

    if constraint_order_correct:
        print("  [OK] 约束强度与性能成反比（符合预期）")
    else:
        print("  [WARNING] 约束顺序检验失败")

    print("\n最终结论:")
    if constrained_lower and constraint_order_correct:
        print("  [OK] 资源约束集成成功！")
        print("  [OK] 约束按预期工作")
        print("  [OK] 可以开始完整实验")
    else:
        print("  [WARNING] 部分检验未通过，需要进一步调查")

    return results


def test_monte_carlo_with_constraints():
    """运行Monte Carlo测试验证统计显著性"""
    print("\n" + "=" * 80)
    print("Monte Carlo测试（资源约束）")
    print("=" * 80)

    snr_range = [20]  # 只测试20dB
    n_mc = 50
    abs_bandwidth = 1.2e6

    scenarios = {
        'Baseline': {'max_abs_users': None, 'max_s2a_capacity': None},
        'ABS-8': {'max_abs_users': 8, 'max_s2a_capacity': None},
        'S2A-20M': {'max_abs_users': None, 'max_s2a_capacity': 20e6},
    }

    print(f"\n配置: SNR={snr_range[0]}dB, MC={n_mc}次")

    results = {}

    for name, params in scenarios.items():
        print(f"\n运行 {name}...")

        system = JointOptimizationSATCON(
            config, abs_bandwidth,
            use_module1=True,
            use_module2=True,
            use_module3=True,
            max_abs_users=params['max_abs_users'],
            max_s2a_capacity=params['max_s2a_capacity']
        )

        mean_rates, mean_se, std_rates, _ = system.simulate_performance(
            snr_db_range=snr_range,
            elevation_deg=10,
            n_realizations=n_mc,
            verbose=True
        )

        results[name] = {
            'rate': mean_rates[0],
            'std': std_rates[0]
        }

        rate_mbps = mean_rates[0] / 1e6
        std_mbps = std_rates[0] / 1e6
        print(f"  结果: {rate_mbps:.2f}±{std_mbps:.2f} Mbps")

    # 统计检验
    print("\n性能对比:")
    baseline = results['Baseline']['rate']
    for name in ['ABS-8', 'S2A-20M']:
        rate_diff = results[name]['rate'] - baseline
        improvement = rate_diff / baseline * 100
        print(f"  {name} vs Baseline: {improvement:+.2f}%")

    return results


if __name__ == '__main__':
    # 测试1：集成验证
    print("\n" + "=" * 80)
    print("第1部分：集成验证测试")
    print("=" * 80)
    results1 = test_integration()

    # 测试2：Monte Carlo验证（可选，耗时较长）
    run_mc_test = input("\n是否运行Monte Carlo测试？(y/n，预计耗时5分钟): ").lower() == 'y'

    if run_mc_test:
        print("\n" + "=" * 80)
        print("第2部分：Monte Carlo测试")
        print("=" * 80)
        results2 = test_monte_carlo_with_constraints()
    else:
        print("\n[跳过] Monte Carlo测试")

    print("\n" + "=" * 80)
    print("所有测试完成！")
    print("=" * 80)
