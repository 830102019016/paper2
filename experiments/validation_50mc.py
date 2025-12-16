# -*- coding: utf-8 -*-
"""
中等规模验证 - 50次MC
用于更准确的性能评估

配置：3个SNR点 × 50次实现
预计时间：5-8分钟
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
    print("中等规模验证 - 50次Monte Carlo")
    print("=" * 80)

    # 实验参数
    snr_range = [10, 20, 30]  # 3个SNR点
    n_realizations = 50        # 50次Monte Carlo（5倍于快速测试）
    elevation_deg = 10
    abs_bandwidth = 1.2e6

    print(f"\n配置:")
    print(f"  SNR点: {snr_range}")
    print(f"  Monte Carlo次数: {n_realizations}")
    print(f"  仰角: {elevation_deg}°")
    print(f"  ABS带宽: {abs_bandwidth/1e6:.1f} MHz")
    print(f"  用户数: {config.N}")
    print(f"  预计时间: 5-8分钟")
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
        'Module2': {
            'system': JointOptimizationSATCON(
                config, abs_bandwidth,
                use_module1=False,
                use_module2=True,
                use_module3=False
            ),
            'description': '仅模块2（联合配对优化）'
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
    results = {name: {'rates': [], 'se': [], 'std': []} for name in systems.keys()}

    # 运行实验
    print("\n" + "=" * 80)
    print("开始运行实验...")
    print("=" * 80)

    start_time = time.time()

    for name, info in systems.items():
        print(f"\n运行 {name}...")
        system = info['system']

        try:
            mean_rates, mean_se, std_rates, _ = system.simulate_performance(
                snr_db_range=snr_range,
                elevation_deg=elevation_deg,
                n_realizations=n_realizations,
                verbose=True  # 显示进度条
            )

            results[name]['rates'] = mean_rates
            results[name]['se'] = mean_se
            results[name]['std'] = std_rates

            print(f"  [OK] 完成")
            for i, snr in enumerate(snr_range):
                # 转换为 Mbps
                rate_mbps = mean_rates[i] / 1e6
                std_mbps = std_rates[i] / 1e6
                print(f"    SNR={snr}dB: Rate={rate_mbps:.2f}±{std_mbps:.2f} Mbps, SE={mean_se[i]:.2f} bits/s/Hz")

        except Exception as e:
            print(f"  [ERROR] 错误: {e}")
            import traceback
            traceback.print_exc()

    elapsed_time = time.time() - start_time

    # 打印详细对比结果
    print("\n" + "=" * 80)
    print("详细结果对比（50次MC）")
    print("=" * 80)

    baseline_rates = results['Baseline']['rates']
    baseline_se = results['Baseline']['se']

    print(f"\n{'系统':<15} {'SNR(dB)':<10} {'速率(Mbps)':<20} {'频谱效率':<15} {'提升(%)':<10}")
    print("-" * 90)

    for snr_idx, snr in enumerate(snr_range):
        for name in ['Baseline', 'Module1', 'Module2', 'Full']:
            rate = results[name]['rates'][snr_idx] / 1e6
            std = results[name]['std'][snr_idx] / 1e6
            se = results[name]['se'][snr_idx]

            if name == 'Baseline':
                improvement = 0.0
            else:
                improvement = (results[name]['rates'][snr_idx] - baseline_rates[snr_idx]) / baseline_rates[snr_idx] * 100

            print(f"{name:<15} {snr:<10} {rate:.2f}±{std:.2f}{'':<10} {se:<15.2f} {improvement:>8.2f}%")
        print("-" * 90)

    # 计算平均提升和置信区间
    print("\n性能提升总结:")
    print("-" * 60)
    for name in ['Module1', 'Module2', 'Full']:
        improvements = [
            (results[name]['rates'][i] - baseline_rates[i]) / baseline_rates[i] * 100
            for i in range(len(snr_range))
        ]
        avg_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)

        # 95%置信区间
        ci_95 = 1.96 * std_improvement / np.sqrt(len(snr_range))

        print(f"  {name:<15}: {avg_improvement:>6.2f}% ± {ci_95:.2f}% (95% CI)")

    # 重点关注20dB的结果
    idx_20db = snr_range.index(20) if 20 in snr_range else -1
    if idx_20db >= 0:
        print(f"\n" + "=" * 60)
        print(f"重点结果 @ SNR=20dB (50次MC)")
        print("=" * 60)

        for name in ['Baseline', 'Module1', 'Module2', 'Full']:
            rate_mbps = results[name]['rates'][idx_20db] / 1e6
            std_mbps = results[name]['std'][idx_20db] / 1e6
            se = results[name]['se'][idx_20db]

            if name == 'Baseline':
                print(f"  {name:<15}: {rate_mbps:.2f}±{std_mbps:.2f} Mbps, SE={se:.2f} bits/s/Hz")
            else:
                improvement = (results[name]['rates'][idx_20db] - baseline_rates[idx_20db]) / baseline_rates[idx_20db] * 100
                print(f"  {name:<15}: {rate_mbps:.2f}±{std_mbps:.2f} Mbps, SE={se:.2f} bits/s/Hz (+{improvement:.2f}%)")

        # 统计显著性检验（t-test）
        print(f"\n统计显著性检验:")

        # 这里我们假设样本间独立（实际上用了共同随机数会有相关性）
        # 简化检验：只看均值差异是否大于标准误差的1.96倍
        for name in ['Module1', 'Module2', 'Full']:
            rate_diff = results[name]['rates'][idx_20db] - baseline_rates[idx_20db]
            # 合并标准差（假设独立）
            combined_std = np.sqrt(results[name]['std'][idx_20db]**2 + results['Baseline']['std'][idx_20db]**2)
            std_error = combined_std / np.sqrt(n_realizations)

            z_score = rate_diff / std_error
            significant = abs(z_score) > 1.96  # 95%置信水平

            if significant:
                print(f"  {name} vs Baseline: [OK] 显著 (z={z_score:.2f})")
            else:
                print(f"  {name} vs Baseline: [WARNING] 不显著 (z={z_score:.2f})")

    print(f"\n运行时间: {elapsed_time/60:.1f} 分钟 ({elapsed_time:.0f} 秒)")
    print("=" * 80)
    print("\n[OK] 中等规模验证完成！")

    # 最终判断
    full_improvement = np.mean([
        (results['Full']['rates'][i] - baseline_rates[i]) / baseline_rates[i] * 100
        for i in range(len(snr_range))
    ])

    print("\n" + "=" * 80)
    print("最终结论:")
    print("=" * 80)
    if full_improvement > 3.0:
        print(f"  [OK] 性能提升显著 ({full_improvement:.2f}%)，系统工作正常")
        print(f"  [OK] 结果与预期一致（预期3.5%左右）")
        print(f"  [OK] 可以开始方案A开发（资源受限架构）")
        print(f"\n  预期效果：")
        print(f"    - 方案A将性能提升从 {full_improvement:.1f}% → 12-15%")
        print(f"    - Module 3贡献从 0% → 8-10%")
        print(f"    - 开发时间：5天")
    elif full_improvement > 1.5:
        print(f"  [WARNING] 性能提升较小 ({full_improvement:.2f}%)")
        print(f"  [WARNING] 低于预期（3.5%），建议检查：")
        print(f"    1. 增加MC次数到100-200次")
        print(f"    2. 检查随机种子是否固定")
        print(f"    3. 验证模块是否正确启用")
    else:
        print(f"  [ERROR] 性能提升不明显 ({full_improvement:.2f}%)")
        print(f"  [ERROR] 需要排查代码问题")

    return results

if __name__ == '__main__':
    results = main()
