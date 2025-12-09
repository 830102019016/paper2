"""
全面调试 SATCON 系统 - 根据论文验证每个步骤
"""
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import config
from src.satcon_system import SATCONSystem
from src.user_distribution import UserDistribution
from src.abs_placement import ABSPlacement
from src.a2g_channel import A2GChannel
from src.noma_transmission import SatelliteNOMA
from src.power_allocation import NOMAAllocator

print("=" * 80)
print("SATCON 全面调试 - 基于论文公式验证")
print("=" * 80)

# 测试参数
snr_db = 20
elevation = 10
seed = 42
Bd = 1.2e6

print(f"\n测试参数:")
print(f"  SNR: {snr_db} dB")
print(f"  仰角: {elevation}°")
print(f"  带宽: {Bd/1e6:.1f} MHz")
print(f"  用户数: {config.N}")

# 步骤1: 生成用户
print(f"\n{'='*80}")
print("步骤1: 生成用户分布")
print("="*80)
dist = UserDistribution(config.N, config.coverage_radius, seed=seed)
user_positions = dist.generate_uniform_circle()
print(f"[OK] 生成 {config.N} 个用户")

# 步骤2: 优化ABS位置
print(f"\n{'='*80}")
print("步骤2: 优化ABS位置")
print("="*80)
a2g = A2GChannel()
abs_placement = ABSPlacement()
abs_position, info = abs_placement.optimize_position_complete(user_positions, a2g)
print(f"[OK] ABS位置: ({abs_position[0]:.1f}, {abs_position[1]:.1f}, {abs_position[2]:.1f})")
print(f"  方法: {info['method']}")

# 步骤3: 卫星NOMA速率
print(f"\n{'='*80}")
print("步骤3: 卫星NOMA传输（论文公式5, 6）")
print("="*80)
sat_noma = SatelliteNOMA(config)
snr_linear = 10 ** (snr_db / 10)
print(f"  SNR线性值: {snr_linear:.6f} (SNR={snr_db}dB)")

sat_gains = sat_noma.compute_channel_gains_with_pathloss(elevation)
print(f"  卫星信道增益范围: {np.min(sat_gains):.2e} ~ {np.max(sat_gains):.2e}")

sat_rates, sat_sum = sat_noma.compute_achievable_rates(sat_gains, snr_linear)

# 详细调试：打印SINR
print(f"  前3个用户的 SNR*Gamma (SINR): {snr_linear * sat_gains[:3]}")
print(f"  前3个用户的 log2(1+SINR): {np.log2(1 + snr_linear * sat_gains[:3])}")

print(f"  卫星总速率: {sat_sum/1e6:.6f} Mbps")
print(f"  卫星频谱效率: {sat_sum/config.Bs:.6f} bits/s/Hz")
print(f"  前3个用户速率: {sat_rates[:3]/1e6} Mbps")

# 检查卫星速率是否合理
if sat_sum < 1e3:
    print(f"  [ERROR] 卫星速率太小！")
else:
    print(f"  [OK] 卫星速率正常")

# 步骤4: 计算A2G信道增益
print(f"\n{'='*80}")
print("步骤4: A2G信道增益（论文公式10）")
print("="*80)
h_abs = abs_position[2]
distances_2d = dist.compute_distances_from_point(user_positions, abs_position)
Nd = config.get_abs_noise_power(Bd)
print(f"  ABS高度: {h_abs:.0f} m")
print(f"  噪声功率 Nd: {Nd:.6e} W")

fading_a2g = a2g.generate_fading(config.N, seed=seed+1)
a2g_gains = np.array([
    a2g.compute_channel_gain(h_abs, r, fading, config.Gd_t_dB, config.Gd_r_dB, Nd)
    for r, fading in zip(distances_2d, fading_a2g)
])
print(f"  A2G信道增益范围: {np.min(a2g_gains):.2e} ~ {np.max(a2g_gains):.2e}")
print(f"  前3个用户增益: {a2g_gains[:3]}")

# 步骤5: ABS NOMA速率计算
print(f"\n{'='*80}")
print("步骤5: ABS NOMA速率（论文第3页，基于公式5,6）")
print("="*80)

allocator = NOMAAllocator()
K = config.N // 2
bandwidth_per_pair = Bd / K

# 用户配对
pairs, paired_gains = allocator.optimal_user_pairing(a2g_gains)
print(f"  配对数: {len(pairs)}")
print(f"  每对带宽: {bandwidth_per_pair/1e6:.3f} MHz")

abs_noma_rates = np.zeros(config.N)
for k in range(min(3, K)):  # 只打印前3对
    weak_idx, strong_idx = pairs[k]
    gamma_weak, gamma_strong = paired_gains[k]

    # 功率分配
    beta_strong, beta_weak = allocator.compute_power_factors(
        gamma_strong, gamma_weak, config.Pd
    )

    # 根据论文公式计算速率
    # 强用户: R_j = (B/K) * log2(1 + β_j * P * Γ_j)
    rate_strong = bandwidth_per_pair * np.log2(1 + beta_strong * config.Pd * gamma_strong)

    # 弱用户: R_i = (B/K) * log2(1 + ((1-β_j)*P*Γ_i) / (β_j*P*Γ_i + 1))
    rate_weak = bandwidth_per_pair * np.log2(
        1 + (beta_weak * config.Pd * gamma_weak) / (beta_strong * config.Pd * gamma_weak + 1)
    )

    abs_noma_rates[weak_idx] = rate_weak
    abs_noma_rates[strong_idx] = rate_strong

    print(f"\n  配对 {k+1}: weak={weak_idx}, strong={strong_idx}")
    print(f"    γ_weak={gamma_weak:.2e}, γ_strong={gamma_strong:.2e}")
    print(f"    β_weak={beta_weak:.3f}, β_strong={beta_strong:.3f}")
    print(f"    R_weak={rate_weak/1e6:.6f} Mbps")
    print(f"    R_strong={rate_strong/1e6:.6f} Mbps")

# 计算所有对
for k in range(K):
    if k >= 3:  # 前3对已经计算过了
        weak_idx, strong_idx = pairs[k]
        gamma_weak, gamma_strong = paired_gains[k]
        beta_strong, beta_weak = allocator.compute_power_factors(
            gamma_strong, gamma_weak, config.Pd
        )
        rate_strong = bandwidth_per_pair * np.log2(1 + beta_strong * config.Pd * gamma_strong)
        rate_weak = bandwidth_per_pair * np.log2(
            1 + (beta_weak * config.Pd * gamma_weak) / (beta_strong * config.Pd * gamma_weak + 1)
        )
        abs_noma_rates[weak_idx] = rate_weak
        abs_noma_rates[strong_idx] = rate_strong

abs_noma_sum = np.sum(abs_noma_rates)
print(f"\n  ABS NOMA总速率: {abs_noma_sum/1e6:.6f} Mbps")
print(f"  ABS NOMA频谱效率: {abs_noma_sum/Bd:.6f} bits/s/Hz")

if abs_noma_sum < 1e3:
    print(f"  [ERROR] ABS NOMA速率太小！")
else:
    print(f"  [OK] ABS NOMA速率正常")

# 步骤6: 混合决策
print(f"\n{'='*80}")
print("步骤6: 混合决策（论文第3页Section III.C）")
print("="*80)

# S2A速率（简化：假设足够大）
s2a_rates = sat_rates * 10
print(f"  S2A速率（简化）: 卫星速率 × 10")

# ABS NOMA速率限制
abs_noma_limited = np.minimum(abs_noma_rates, s2a_rates)

# OMA速率
bandwidth_per_user = Bd / K
abs_oma_rates = bandwidth_per_user * np.log2(1 + config.Pd * a2g_gains)
abs_oma_limited = np.minimum(abs_oma_rates, s2a_rates)

final_rates = sat_rates.copy()
decision_count = {'noma': 0, 'oma_weak': 0, 'oma_strong': 0, 'none': 0}

for k, (weak_idx, strong_idx) in enumerate(pairs):
    R_s_i = sat_rates[weak_idx]
    R_s_j = sat_rates[strong_idx]
    R_dn_i = abs_noma_limited[weak_idx]
    R_dn_j = abs_noma_limited[strong_idx]
    R_do_i = abs_oma_limited[weak_idx]
    R_do_j = abs_oma_limited[strong_idx]

    # 论文决策规则
    if R_s_i < R_dn_i and R_s_j < R_dn_j:
        # 情况1: NOMA双用户
        final_rates[weak_idx] = R_dn_i
        final_rates[strong_idx] = R_dn_j
        decision_count['noma'] += 1
        decision = 'NOMA'
    elif R_s_i < R_do_i and R_s_j >= R_dn_j:
        # 情况2: OMA只给弱用户
        final_rates[weak_idx] = R_do_i
        final_rates[strong_idx] = R_s_j
        decision_count['oma_weak'] += 1
        decision = 'OMA_WEAK'
    elif R_s_i >= R_dn_i and R_s_j < R_do_j:
        # 情况3: OMA只给强用户
        final_rates[weak_idx] = R_s_i
        final_rates[strong_idx] = R_do_j
        decision_count['oma_strong'] += 1
        decision = 'OMA_STRONG'
    else:
        # 情况4: 不传输
        decision_count['none'] += 1
        decision = 'NONE'

    if k < 3:
        print(f"\n  配对 {k+1} 决策: {decision}")
        print(f"    R_s_i={R_s_i/1e6:.6f}, R_dn_i={R_dn_i/1e6:.6f}, R_do_i={R_do_i/1e6:.6f}")
        print(f"    R_s_j={R_s_j/1e6:.6f}, R_dn_j={R_dn_j/1e6:.6f}, R_do_j={R_do_j/1e6:.6f}")
        print(f"    最终: weak={final_rates[weak_idx]/1e6:.6f}, strong={final_rates[strong_idx]/1e6:.6f} Mbps")

final_sum = np.sum(final_rates)
print(f"\n  决策统计: {decision_count}")
print(f"  最终总速率: {final_sum/1e6:.6f} Mbps")
print(f"  最终频谱效率: {final_sum/Bd:.6f} bits/s/Hz")

# 最终诊断
print(f"\n{'='*80}")
print("最终诊断")
print("="*80)

if final_sum < 1e3:
    print("[ERROR] 最终速率太小！")
    print("\n可能原因分析：")
    if sat_sum < 1e3:
        print("  1. 卫星速率异常 - 检查Loo信道模型")
    if abs_noma_sum < 1e3:
        print("  2. ABS NOMA速率异常 - 检查A2G信道增益")
    if decision_count['noma'] == 0 and decision_count['oma_weak'] == 0 and decision_count['oma_strong'] == 0:
        print("  3. 所有配对都选择'不传输' - 检查混合决策条件")
else:
    print(f"[OK] 系统正常工作")
    print(f"  卫星贡献: {sat_sum/1e6:.3f} Mbps")
    print(f"  ABS NOMA潜力: {abs_noma_sum/1e6:.3f} Mbps")
    print(f"  最终总和: {final_sum/1e6:.3f} Mbps")

print("=" * 80)
