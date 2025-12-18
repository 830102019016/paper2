"""
查看Baseline性能验证结果

快速查看和对比保存的baseline性能数据
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def load_results(results_file='results/baseline_performance.json'):
    """加载保存的结果"""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def print_summary(results):
    """打印结果摘要"""
    params = results['experiment_params']
    perf = results['performance']
    modes = results['mode_statistics']

    print("=" * 70)
    print("SATCON Baseline 性能结果")
    print("=" * 70)
    print(f"\n实验参数：")
    print(f"  时间戳: {results['timestamp']}")
    print(f"  SNR范围: {params['snr_db_range'][0]}-{params['snr_db_range'][-1]} dB")
    print(f"  仿真次数: {params['n_realizations']}")
    print(f"  用户数: {params['n_users']}")

    snr_range = np.array(params['snr_db_range'])
    mean_se = np.array(perf['mean_se_bits_per_hz'])

    print(f"\n性能指标：")
    print(f"  最小SE (SNR={snr_range[0]}dB): {mean_se[0]:.2f} bits/s/Hz")
    print(f"  最大SE (SNR={snr_range[-1]}dB): {mean_se[-1]:.2f} bits/s/Hz")
    print(f"  平均SE: {mean_se.mean():.2f} bits/s/Hz")
    print(f"  SE增长: {mean_se[-1]/mean_se[0]:.2f}x")


def plot_quick_view(results):
    """快速可视化"""
    params = results['experiment_params']
    perf = results['performance']
    modes = results['mode_statistics']

    snr_range = np.array(params['snr_db_range'])
    mean_se = np.array(perf['mean_se_bits_per_hz'])

    plt.figure(figsize=(10, 5))
    plt.plot(snr_range, mean_se, 'b-o', linewidth=2, markersize=6,
             label='SATCON Baseline')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Satellite SNR (dB)', fontsize=12)
    plt.ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=12)
    plt.title('SATCON Baseline Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = load_results()
    print_summary(results)
    plot_quick_view(results)
