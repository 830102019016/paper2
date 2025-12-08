"""
复现完整的论文 Figure 2

目标：生成5条曲线
1. SAT-NOMA (baseline - Phase 1)
2. SATCON Bd=0.4 MHz
3. SATCON Bd=1.2 MHz
4. SATCON Bd=2 MHz
5. SATCON Bd=3 MHz

运行方法：
python simulations/fig2_complete.py

预计时间：约15-30分钟（取决于Monte Carlo次数）
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.satcon_system import SATCONSystem
from src.utils import setup_plot_style


def load_phase1_baseline():
    """加载Phase 1的SAT-NOMA基准结果"""
    baseline_path = Path(__file__).parent.parent / 'results' / 'data' / 'fig2_baseline.npz'
    
    if baseline_path.exists():
        print("[OK] 加载Phase 1基准数据...")
        data = np.load(baseline_path)
        return data['snr_db'], data['spectral_efficiency']
    else:
        print("[WARN] 未找到Phase 1基准数据，需要先运行: python simulations/fig2_sat_noma.py")
        return None, None


def simulate_satcon_curves(snr_range, elevation, n_mc):
    """
    仿真4条SATCON曲线（不同带宽）
    
    参数:
        snr_range: SNR范围
        elevation: 卫星仰角
        n_mc: Monte Carlo次数
    
    返回:
        satcon_results: dict {bandwidth: spectral_efficiency}
    """
    # 调试模式：只测试单个带宽以加快速度
    # bandwidths = config.Bd_options  # [0.4, 1.2, 2, 3] MHz
    bandwidths = [1.2e6]  # 只测试Bd=1.2MHz
    satcon_results = {}
    
    print(f"\n开始SATCON仿真...")
    print(f"  带宽选项: {[bd/1e6 for bd in bandwidths]} MHz")
    print(f"  SNR范围: {snr_range[0]}-{snr_range[-1]} dB")
    print(f"  Monte Carlo: {n_mc} 次")
    print("-" * 70)
    
    for bd in bandwidths:
        print(f"\n仿真 SATCON (Bd={bd/1e6:.1f} MHz)...")
        
        satcon = SATCONSystem(config, abs_bandwidth=bd)
        mean_rates, mean_se, std_rates, mode_stats = satcon.simulate_performance(
            snr_db_range=snr_range,
            elevation_deg=elevation,
            n_realizations=n_mc,
            verbose=True
        )
        
        satcon_results[bd] = {
            'spectral_efficiency': mean_se,
            'sum_rate': mean_rates,
            'std': std_rates,
            'mode_stats': mode_stats
        }
        
        print(f"  [OK] 完成 Bd={bd/1e6:.1f} MHz")
        print(f"    SE @ 10dB: {mean_se[np.argmin(np.abs(snr_range-10))]:.2f} bits/s/Hz")
        print(f"    SE @ 20dB: {mean_se[np.argmin(np.abs(snr_range-20))]:.2f} bits/s/Hz")
    
    print("-" * 70)
    print("[OK] 所有SATCON曲线仿真完成")
    
    return satcon_results


def plot_complete_figure2(snr_range, baseline_se, satcon_results, save_dir):
    """
    绘制完整的Figure 2（5条曲线）
    
    参数:
        snr_range: SNR范围
        baseline_se: SAT-NOMA频谱效率
        satcon_results: SATCON结果字典
        save_dir: 保存目录
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 1. SAT-NOMA基准（紫色×）
    ax.plot(snr_range, baseline_se, 
           'x-', color='purple', linewidth=2, markersize=8, markevery=3,
           label='SAT-NOMA', zorder=5)
    
    # 2. SATCON曲线（使用实际运行的带宽）
    colors = ['blue', 'green', 'orange', 'red']
    markers = ['o', 's', '^', 'd']
    # 使用实际运行的带宽，而不是硬编码的config.Bd_options
    bandwidths = sorted(satcon_results.keys())

    for i, bd in enumerate(bandwidths):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        se = satcon_results[bd]['spectral_efficiency']
        ax.plot(snr_range, se,
               marker=marker, linestyle='-', color=color,
               linewidth=2, markersize=6, markevery=3,
               label=f'SATCON $B_d$={bd/1e6:.1f}MHz',
               zorder=4)
    
    # 图表设置
    ax.set_xlabel('SNR [dB]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Spectral Efficiency [bits/sec/Hz]', fontsize=14, fontweight='bold')
    ax.set_title('Figure 2: SATCON vs SAT-NOMA Performance', 
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.95, ncol=1)
    ax.set_xlim([snr_range[0], snr_range[-1]])
    
    # 设置y轴范围
    all_se = [baseline_se] + [satcon_results[bd]['spectral_efficiency'] 
                              for bd in bandwidths]
    max_se = max([np.max(se) for se in all_se])
    ax.set_ylim([0, max_se * 1.1])
    
    plt.tight_layout()
    
    # 保存图表
    fig_path = save_dir / 'figures' / 'fig2_complete.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 图表已保存: {fig_path}")
    
    return fig, ax


def save_complete_data(snr_range, baseline_se, satcon_results, save_dir):
    """保存完整仿真数据"""
    data_dict = {
        'snr_db': snr_range,
        'sat_noma_se': baseline_se,
    }
    
    # 添加SATCON数据（使用实际运行的带宽）
    for bd in satcon_results.keys():
        bd_mhz = bd / 1e6
        data_dict[f'satcon_{bd_mhz:.1f}mhz_se'] = satcon_results[bd]['spectral_efficiency']
        data_dict[f'satcon_{bd_mhz:.1f}mhz_sum_rate'] = satcon_results[bd]['sum_rate']
    
    data_dict['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_dict['n_monte_carlo'] = config.n_monte_carlo
    
    data_path = save_dir / 'data' / 'fig2_complete.npz'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(data_path, **data_dict)
    print(f"[OK] 数据已保存: {data_path}")


def print_performance_report(snr_range, baseline_se, satcon_results):
    """打印性能报告"""
    print("\n" + "=" * 70)
    print("性能报告 - Figure 2 完整结果")
    print("=" * 70)
    
    # 关键SNR点
    key_snrs = [10, 20, 30]
    
    print(f"\n频谱效率 (bits/s/Hz) 对比:")
    header = ' '.join([f"{snr}dB".rjust(10) for snr in key_snrs])
    print(f"{'方案':<25} {header}")
    print("-" * 70)
    
    # SAT-NOMA
    baseline_vals = [baseline_se[np.argmin(np.abs(snr_range-snr))] for snr in key_snrs]
    baseline_str = ' '.join([f"{v:>10.3f}" for v in baseline_vals])
    print(f"{'SAT-NOMA':<25} {baseline_str}")
    
    # SATCON（使用实际运行的带宽）
    for bd in sorted(satcon_results.keys()):
        se = satcon_results[bd]['spectral_efficiency']
        satcon_vals = [se[np.argmin(np.abs(snr_range-snr))] for snr in key_snrs]
        label = f"SATCON Bd={bd/1e6:.1f}MHz"
        satcon_str = ' '.join([f"{v:>10.3f}" for v in satcon_vals])
        print(f"{label:<25} {satcon_str}")

    # 性能增益
    print(f"\n" + "=" * 70)
    print("相对SAT-NOMA的性能增益 (%)")
    print("=" * 70)

    for bd in sorted(satcon_results.keys()):
        se = satcon_results[bd]['spectral_efficiency']
        gains = [(se[np.argmin(np.abs(snr_range-snr))] /
                 baseline_se[np.argmin(np.abs(snr_range-snr))] - 1) * 100
                for snr in key_snrs]
        label = f"SATCON Bd={bd/1e6:.1f}MHz"
        gains_str = ' '.join([f"{g:>9.1f}%" for g in gains])
        print(f"{label:<25} {gains_str}")

    # 模式统计
    print(f"\n" + "=" * 70)
    print("决策模式统计 (平均每SNR)")
    print("=" * 70)

    for bd in sorted(satcon_results.keys()):
        mode_stats = satcon_results[bd]['mode_stats']
        print(f"\nSATCON Bd={bd/1e6:.1f}MHz:")
        print(f"  NOMA模式:      {np.mean(mode_stats['noma']):.1f} pairs")
        print(f"  OMA弱用户:     {np.mean(mode_stats['oma_weak']):.1f} pairs")
        print(f"  OMA强用户:     {np.mean(mode_stats['oma_strong']):.1f} pairs")
        print(f"  不传输:        {np.mean(mode_stats['none']):.1f} pairs")


def main():
    """主函数"""
    print("=" * 70)
    print("复现论文 Figure 2 - 完整版（5条曲线）")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 配置
    snr_range = config.Ps_dB
    elevation = 10
    n_mc = config.n_monte_carlo
    
    print(f"\n仿真配置:")
    print(f"  用户数: {config.N}")
    print(f"  SNR范围: {snr_range[0]}-{snr_range[-1]} dB ({len(snr_range)} 点)")
    print(f"  仰角: {elevation}°")
    print(f"  Monte Carlo: {n_mc} 次")
    print(f"  预计时间: ~{len(snr_range) * n_mc * 5 * 0.002:.1f} 分钟")
    
    # 1. 加载Phase 1基准
    snr_baseline, baseline_se = load_phase1_baseline()
    
    if snr_baseline is None:
        print("\n❌ 错误：请先运行Phase 1仿真生成基准数据")
        return
    
    # 验证SNR范围一致
    if not np.array_equal(snr_baseline, snr_range):
        print("[WARN] 警告：SNR范围不一致，将使用Phase 1的SNR范围")
        snr_range = snr_baseline
    
    # 2. 运行SATCON仿真
    satcon_results = simulate_satcon_curves(snr_range, elevation, n_mc)
    
    # 3. 绘制完整图表
    save_dir = Path(__file__).parent.parent / 'results'
    fig, ax = plot_complete_figure2(snr_range, baseline_se, satcon_results, save_dir)
    
    # 4. 保存数据
    save_complete_data(snr_range, baseline_se, satcon_results, save_dir)
    
    # 5. 性能报告
    print_performance_report(snr_range, baseline_se, satcon_results)
    
    print(f"\n" + "=" * 70)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\n=== Figure 2 完整复现完成 ===")
    print("\n输出文件:")
    print(f"  - results/figures/fig2_complete.png")
    print(f"  - results/data/fig2_complete.npz")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    main()
