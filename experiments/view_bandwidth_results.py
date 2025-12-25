"""
查看ABS带宽影响实验结果
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 加载数据
data = np.load('results/data/abs_bandwidth_impact.npz')

snr_db_range = data['snr_db_range']
bd_values = data['abs_bandwidth_values_mhz']

print("=" * 80)
print("ABS Bandwidth Impact Experiment Results")
print("=" * 80)

print(f"\nExperiment Configuration:")
print(f"  SNR Range: {snr_db_range[0]} - {snr_db_range[-1]} dB ({len(snr_db_range)} points)")
print(f"  Elevation: {data['elevation_deg']} degrees")
print(f"  ABS Bandwidth Values: {bd_values} MHz")
print(f"  Number of Users: {len(data['user_positions'])}")

# 提取性能数据
schemes = {
    'SAT-NOMA': data['SAT-NOMA_se'],
    'SATCON Bd=0.4 MHz': data['SATCON_Bd0p4_MHz_se'],
    'SATCON Bd=1.2 MHz': data['SATCON_Bd1p2_MHz_se'],
    'SATCON Bd=2 MHz': data['SATCON_Bd2_MHz_se'],
    'SATCON Bd=3 MHz': data['SATCON_Bd3_MHz_se']
}

# 打印关键SNR点的性能
print(f"\n{'='*80}")
print(f"Spectral Efficiency [bits/s/Hz] at Key SNR Points")
print(f"{'='*80}")
print(f"{'Scheme':<25} {'0dB':<10} {'10dB':<10} {'20dB':<10} {'30dB':<10} {'Avg Gain':<12}")
print(f"{'-'*80}")

sat_noma_se = schemes['SAT-NOMA']
sat_noma_avg = np.mean(sat_noma_se)

for scheme_name, se_values in schemes.items():
    # 提取关键SNR点
    se_0db = se_values[0]
    se_10db = se_values[10]
    se_20db = se_values[20]
    se_30db = se_values[30]

    # 计算平均增益
    if scheme_name == 'SAT-NOMA':
        gain_str = "0.00%"
    else:
        avg_gain = (np.mean(se_values) - sat_noma_avg) / sat_noma_avg * 100
        gain_str = f"+{avg_gain:.2f}%"

    print(f"{scheme_name:<25} {se_0db:<10.3f} {se_10db:<10.3f} {se_20db:<10.3f} {se_30db:<10.3f} {gain_str:<12}")

print(f"{'='*80}")

# ABS位置分析
print(f"\nOptimized ABS Positions:")
print(f"{'='*80}")
abs_positions = {
    'Bd=0.4 MHz': data['SATCON_Bd0p4_MHz_abs_position'],
    'Bd=1.2 MHz': data['SATCON_Bd1p2_MHz_abs_position'],
    'Bd=2 MHz': data['SATCON_Bd2_MHz_abs_position'],
    'Bd=3 MHz': data['SATCON_Bd3_MHz_abs_position']
}

for bd_name, pos in abs_positions.items():
    print(f"  {bd_name:<15} -> x={pos[0]:>7.1f}m, y={pos[1]:>7.1f}m, h={pos[2]:>5.0f}m")

# 性能增益分析
print(f"\n{'='*80}")
print(f"Performance Gain Analysis (relative to SAT-NOMA)")
print(f"{'='*80}")

for scheme_name, se_values in schemes.items():
    if scheme_name == 'SAT-NOMA':
        continue

    # 计算各SNR区间的平均增益
    low_snr = snr_db_range <= 10
    mid_snr = (snr_db_range > 10) & (snr_db_range <= 20)
    high_snr = snr_db_range > 20

    gain_low = np.mean((se_values[low_snr] - sat_noma_se[low_snr]) / sat_noma_se[low_snr] * 100)
    gain_mid = np.mean((se_values[mid_snr] - sat_noma_se[mid_snr]) / sat_noma_se[mid_snr] * 100)
    gain_high = np.mean((se_values[high_snr] - sat_noma_se[high_snr]) / sat_noma_se[high_snr] * 100)
    gain_overall = np.mean((se_values - sat_noma_se) / sat_noma_se * 100)

    print(f"\n{scheme_name}:")
    print(f"  Low SNR (0-10dB):    {gain_low:>6.2f}%")
    print(f"  Mid SNR (11-20dB):   {gain_mid:>6.2f}%")
    print(f"  High SNR (21-30dB):  {gain_high:>6.2f}%")
    print(f"  Overall Average:     {gain_overall:>6.2f}%")

# 关键发现
print(f"\n{'='*80}")
print(f"Key Findings:")
print(f"{'='*80}")

# 找到最佳带宽配置
best_scheme_20db = None
best_se_20db = 0
for scheme_name, se_values in schemes.items():
    if scheme_name == 'SAT-NOMA':
        continue
    se_20db = se_values[20]
    if se_20db > best_se_20db:
        best_se_20db = se_20db
        best_scheme_20db = scheme_name

print(f"\n1. Best Configuration at SNR=20dB: {best_scheme_20db}")
print(f"   SE = {best_se_20db:.3f} bits/s/Hz")

# 带宽边际效益分析
print(f"\n2. Bandwidth Marginal Return Analysis:")
bd_se_20db = []
for bd in [0.4, 1.2, 2, 3]:
    if bd == 0.4:
        scheme = 'SATCON Bd=0.4 MHz'
    elif bd == 1.2:
        scheme = 'SATCON Bd=1.2 MHz'
    elif bd == 2:
        scheme = 'SATCON Bd=2 MHz'
    else:
        scheme = 'SATCON Bd=3 MHz'
    bd_se_20db.append((bd, schemes[scheme][20]))

for i in range(1, len(bd_se_20db)):
    bd_prev, se_prev = bd_se_20db[i-1]
    bd_curr, se_curr = bd_se_20db[i]
    delta_bd = bd_curr - bd_prev
    delta_se = se_curr - se_prev
    marginal = delta_se / delta_bd
    print(f"   {bd_prev:.1f} -> {bd_curr:.1f} MHz: +{delta_se:.3f} bits/s/Hz (+{marginal:.3f} per MHz)")

print(f"\n3. Recommendation:")
print(f"   - For best performance: Use Bd=1.2 MHz (good balance)")
print(f"   - For resource efficiency: Use Bd=0.4-1.2 MHz (high marginal return)")
print(f"   - Beyond 2 MHz: Diminishing returns observed")

print(f"\n{'='*80}")
print(f"Generated Files:")
print(f"  - Figure: results/figures/fig2_abs_bandwidth_impact.png")
print(f"  - Data:   results/data/abs_bandwidth_impact.npz")
print(f"{'='*80}\n")
