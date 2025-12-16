# -*- coding: utf-8 -*-
"""
快速验证脚本 - 直接测试系统性能
不需要复杂的框架，直接运行3个系统对比

配置：3个SNR点 × 10次实现
预计时间：2-3分钟
"""
import numpy as np
import sys
from pathlib import Path
import time

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src_enhanced.joint_satcon_system import JointOptimizationSATCON

def main():
    print("=" * 80)
    print("快速验证 - 测试系统性能")
    print("=" * 80)

    # 实验参数
    snr_range = [10, 20, 30]  # 3个SNR点
    n_realizations = 10        # 10次Monte Carlo
    elevation_deg = 10
    abs_bandwidth = 1.2e6

    print(f"\n配置:")
    print(f"  SNR点: {snr_range}")
    print(f"  Monte Carlo次数: {n_realizations}")
    print(f"  仰角: {elevation_deg}°")
    print(f"  ABS带宽: {abs_bandwidth/1e6:.1f} MHz")
    print(f"  用户数: {config.N}")
    print("\n" + "=" * 80)

    # 创建三个系统进行对比
    systems = {
        'Baseline': {
            'system': JointOptimizationSATCON(
                config, abs_bandwidth,
                use_module1=False,
                use_module2=False,
                use_module3=False
            ),
            'description': '原方案（k-means+独立配对+贪婪决策）'
        },
        'Module1': {
            'system': JointOptimizationSATCON(
                config, abs_bandwidth,
                use_module1=True,
                use_module2=False,
                use_module3=False
            ),
            'description': '仅模块1（梯度位置优化）'
        },
        'Full': {
            'system': JointOptimizationSATCON(
                config, abs_bandwidth,
                use_module1=True,
                use_module2=True,
                use_module3=True
            ),
            'description': '完整方案（所有模块）'
        }
    }

    print("\n测试系统:")
    for name, info in systems.items():
        print(f"  {name}: {info['description']}")

    # 存储结果
    results = {name: {'rates': [], 'se': []} for name in systems.keys()}

    # 运行实验
    print("\n" + "=" * 80)
    print("开始运行实验...")
    print("=" * 80)

    start_time = time.time()

    for name, info in systems.items():
        print(f"\n运行 {name}...")
        system = info['system']

        try:
            mean_rates, mean_se, _, _ = system.simulate_performance(
                snr_db_range=snr_range,
                elevation_deg=elevation_deg,
                n_realizations=n_realizations,
                verbose=False
            )

            results[name]['rates'] = mean_rates
            results[name]['se'] = mean_se

            print(f"  [OK] 完成")
            for i, snr in enumerate(snr_range):
                # 转换为 Mbps (mean_rates 单位是 bps)
                rate_mbps = mean_rates[i] / 1e6
                print(f"    SNR={snr}dB: Rate={rate_mbps:.2f} Mbps, SE={mean_se[i]:.2f} bits/s/Hz")

        except Exception as e:
            print(f"  [ERROR] 错误: {e}")
            import traceback
            traceback.print_exc()

    elapsed_time = time.time() - start_time

    # 打印对比结果
    print("\n" + "=" * 80)
    print("结果对比")
    print("=" * 80)

    baseline_rates = results['Baseline']['rates']
    baseline_se = results['Baseline']['se']

    print(f"\n{'系统':<15} {'SNR(dB)':<10} {'速率(Mbps)':<15} {'频谱效率':<15} {'提升(%)':<10}")
    print("-" * 80)

    for snr_idx, snr in enumerate(snr_range):
        for name in ['Baseline', 'Module1', 'Full']:
            rate = results[name]['rates'][snr_idx] / 1e6  # 转换为 Mbps
            se = results[name]['se'][snr_idx]

            if name == 'Baseline':
                improvement = 0.0
            else:
                improvement = (results[name]['rates'][snr_idx] - baseline_rates[snr_idx]) / baseline_rates[snr_idx] * 100

            print(f"{name:<15} {snr:<10} {rate:<15.2f} {se:<15.2f} {improvement:>8.2f}%")
        print("-" * 80)

    # 计算平均提升
    print("\n平均提升:")
    for name in ['Module1', 'Full']:
        avg_improvement = np.mean([
            (results[name]['rates'][i] - baseline_rates[i]) / baseline_rates[i] * 100
            for i in range(len(snr_range))
        ])
        print(f"  {name}: {avg_improvement:.2f}%")

    # 重点关注20dB的结果
    idx_20db = snr_range.index(20) if 20 in snr_range else -1
    if idx_20db >= 0:
        print(f"\n重点关注 SNR=20dB:")
        print(f"  Baseline: {baseline_rates[idx_20db]/1e6:.2f} Mbps")
        print(f"  Full系统: {results['Full']['rates'][idx_20db]/1e6:.2f} Mbps")
        improvement_20db = (results['Full']['rates'][idx_20db] - baseline_rates[idx_20db]) / baseline_rates[idx_20db] * 100
        print(f"  提升: {improvement_20db:.2f}%")

        # 与预期对比
        expected_improvement = 3.52  # 来自 README.md
        print(f"\n  预期提升: {expected_improvement}%")
        if abs(improvement_20db - expected_improvement) < 1.0:
            print(f"  [OK] 结果符合预期！")
        else:
            print(f"  [WARNING] 结果与预期有差异（可能是随机性）")

    print(f"\n运行时间: {elapsed_time:.1f} 秒")
    print("=" * 80)
    print("\n[OK] 快速验证完成！")

    # 判断结果
    full_improvement = np.mean([
        (results['Full']['rates'][i] - baseline_rates[i]) / baseline_rates[i] * 100
        for i in range(len(snr_range))
    ])

    print("\n结论:")
    if full_improvement > 2.0:
        print(f"  [OK] 性能提升明显 ({full_improvement:.2f}%)，系统工作正常")
        print(f"  [OK] 可以进行下一步：开发资源受限场景（方案A）")
    elif full_improvement > 0.5:
        print(f"  [WARNING] 性能提升较小 ({full_improvement:.2f}%)，建议增加MC次数验证")
    else:
        print(f"  [ERROR] 性能提升不明显 ({full_improvement:.2f}%)，需要检查代码")

    return results

if __name__ == '__main__':
    results = main()
