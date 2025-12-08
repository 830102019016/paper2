"""
复现论文 Figure 2 - SAT-NOMA 基准曲线

目标：
- X轴: SNR 0-30 dB
- Y轴: 频谱效率 (bits/s/Hz)
- 曲线: SAT-NOMA (论文中的紫色×线)

运行方法：
python simulations/fig2_sat_noma.py

输出：
- 图表: results/figures/fig2_baseline_sat_noma.png
- 数据: results/data/fig2_baseline.npz
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.noma_transmission import SatelliteNOMA
from src.utils import setup_plot_style


def plot_figure2_baseline():
    """绘制 Figure 2 的 SAT-NOMA 基准曲线"""
    
    print("=" * 70)
    print("复现论文 Figure 2 - SAT-NOMA 基准曲线")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ==================== 仿真设置 ====================
    sat_noma = SatelliteNOMA(config)
    
    snr_range = config.Ps_dB          # 0-30 dB
    elevation = 10                     # 论文 Fig.2 使用 E=10°
    n_monte_carlo = config.n_monte_carlo  # 默认1000次
    
    print(f"\n仿真配置:")
    print(f"  SNR范围: {snr_range[0]}-{snr_range[-1]} dB ({len(snr_range)} 点)")
    print(f"  仰角: {elevation}°")
    print(f"  Monte Carlo次数: {n_monte_carlo}")
    print(f"  用户数: {config.N}")
    print(f"  配对数: {config.K}")
    print(f"  卫星带宽: {config.Bs/1e6:.1f} MHz")
    
    # ==================== 运行仿真 ====================
    print(f"\n开始仿真...")
    print(f"预计时间: ~{len(snr_range) * n_monte_carlo * 0.001:.1f} 秒")
    print("-" * 70)
    
    mean_sum_rates, mean_se, std_sum_rates = sat_noma.simulate_performance(
        snr_db_range=snr_range,
        elevation_deg=elevation,
        n_realizations=n_monte_carlo,
        verbose=True
    )
    
    print("-" * 70)
    print(f"[OK] 仿真完成")
    
    # ==================== 绘图 ====================
    print(f"\n生成图表...")
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制主曲线
    ax.plot(snr_range, mean_se, 'x-', 
            color='purple', linewidth=2, markersize=8, markevery=2,
            label='SAT-NOMA (Reproduced)', zorder=3)
    
    # 可选：添加置信区间（阴影）
    std_se = std_sum_rates / config.Bs
    ax.fill_between(snr_range, 
                    mean_se - std_se, 
                    mean_se + std_se,
                    alpha=0.2, color='purple', label='±1 std')
    
    # 图表设置
    ax.set_xlabel('SNR [dB]', fontsize=13, fontweight='bold')
    ax.set_ylabel('Spectral Efficiency [bits/sec/Hz]', fontsize=13, fontweight='bold')
    ax.set_title(f'Figure 2 Baseline: SAT-NOMA Performance (E={elevation}°, N={config.N})', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.set_xlim([snr_range[0], snr_range[-1]])
    ax.set_ylim([0, max(mean_se) * 1.15])
    
    # 添加关键点标注
    key_snrs = [10, 20, 30]
    for snr in key_snrs:
        idx = np.argmin(np.abs(snr_range - snr))
        ax.plot(snr, mean_se[idx], 'ro', markersize=6, zorder=4)
        ax.annotate(f'{mean_se[idx]:.2f}', 
                   xy=(snr, mean_se[idx]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    
    # ==================== 保存结果 ====================
    # 创建输出目录
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # 保存图表
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    fig_path = fig_dir / 'fig2_baseline_sat_noma.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 图表已保存: {fig_path}")
    
    # 保存数据
    data_dir = output_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    data_path = data_dir / 'fig2_baseline.npz'
    np.savez(data_path, 
             snr_db=snr_range, 
             spectral_efficiency=mean_se,
             sum_rate_mbps=mean_sum_rates/1e6,
             std_sum_rate=std_sum_rates,
             elevation=elevation,
             n_monte_carlo=n_monte_carlo,
             timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'))
    print(f"[OK] 数据已保存: {data_path}")
    
    # ==================== 性能报告 ====================
    print(f"\n" + "=" * 70)
    print("性能报告")
    print("=" * 70)
    
    # 关键性能指标
    idx_10db = np.argmin(np.abs(snr_range - 10))
    idx_20db = np.argmin(np.abs(snr_range - 20))
    idx_30db = np.argmin(np.abs(snr_range - 30))
    
    print(f"\n频谱效率 (bits/s/Hz):")
    print(f"  @ SNR =  0 dB: {mean_se[0]:.3f} ± {std_se[0]:.3f}")
    print(f"  @ SNR = 10 dB: {mean_se[idx_10db]:.3f} ± {std_se[idx_10db]:.3f}")
    print(f"  @ SNR = 20 dB: {mean_se[idx_20db]:.3f} ± {std_se[idx_20db]:.3f}")
    print(f"  @ SNR = 30 dB: {mean_se[idx_30db]:.3f} ± {std_se[idx_30db]:.3f}")
    
    print(f"\n总速率 (Mbps):")
    print(f"  @ SNR = 10 dB: {mean_sum_rates[idx_10db]/1e6:.2f}")
    print(f"  @ SNR = 20 dB: {mean_sum_rates[idx_20db]/1e6:.2f}")
    print(f"  @ SNR = 30 dB: {mean_sum_rates[idx_30db]/1e6:.2f}")
    
    print(f"\n曲线特性:")
    # 检查单调性
    diffs = np.diff(mean_se)
    monotonic = np.all(diffs > 0)
    print(f"  单调递增: {'[YES]' if monotonic else '[NO]'}")

    # 饱和检查（最后10个点的增长率）
    last_10_growth = np.mean(diffs[-10:])
    print(f"  高SNR增长率: {last_10_growth:.4f} bits/s/Hz per dB")
    print(f"  饱和趋势: {'[明显]' if last_10_growth < 0.1 else '[轻微]'}")
    
    # ==================== 与论文对比 ====================
    print(f"\n与论文对比 (估计值):")
    print(f"  论文Fig.2约: 10dB→3-4, 20dB→6-7, 30dB→9-10 bits/s/Hz")
    print(f"  本次复现:   10dB→{mean_se[idx_10db]:.1f}, "
          f"20dB→{mean_se[idx_20db]:.1f}, 30dB→{mean_se[idx_30db]:.1f}")
    
    # 简单评估
    expected_ranges = {
        10: (3, 4),
        20: (6, 7),
        30: (9, 10)
    }
    
    match_10 = expected_ranges[10][0] <= mean_se[idx_10db] <= expected_ranges[10][1]
    match_20 = expected_ranges[20][0] <= mean_se[idx_20db] <= expected_ranges[20][1]
    match_30 = expected_ranges[30][0] <= mean_se[idx_30db] <= expected_ranges[30][1]

    print(f"\n匹配度评估:")
    print(f"  10dB: {'[OK] 在范围内' if match_10 else '[WARN] 偏差较大'}")
    print(f"  20dB: {'[OK] 在范围内' if match_20 else '[WARN] 偏差较大'}")
    print(f"  30dB: {'[OK] 在范围内' if match_30 else '[WARN] 偏差较大'}")

    if match_10 and match_20 and match_30:
        print(f"\n[SUCCESS] 复现成功！结果与论文高度吻合")
    elif (match_10 or match_20) and (match_20 or match_30):
        print(f"\n[GOOD] 复现基本成功，趋势正确，数值有偏差")
        print(f"  建议：调整Loo模型参数或增加Monte Carlo次数")
    else:
        print(f"\n[WARN] 复现结果与论文存在较大差异")
        print(f"  可能原因：")
        print(f"    1. Loo模型参数不准确（论文未明确给出）")
        print(f"    2. 路径损耗模型简化")
        print(f"    3. Monte Carlo次数不足")
    
    print(f"\n" + "=" * 70)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 70)
    
    # 显示图表
    plt.show()


if __name__ == "__main__":
    plot_figure2_baseline()
