"""
调试 KKT 资源分配器

目标：找出为什么 KKT 分配器性能比 uniform 差

调试内容：
1. 检查 KKT 分配的带宽量
2. 对比 uniform 和 KKT 的 S2A 速率
3. 分析是否因为带宽分配不足导致性能下降
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.satcon_system import SATCONSystem
from src.user_distribution import UserDistribution
from src.abs_placement import ABSPlacement
from src.a2g_channel import A2GChannel, S2AChannel
from src.noma_transmission import SatelliteNOMA
from src.power_allocation import NOMAAllocator


def debug_single_realization(snr_db=20):
    """
    调试单次仿真，对比 uniform 和 KKT 的详细行为
    """
    print("=" * 80)
    print(f"调试 KKT vs Uniform (SNR={snr_db}dB)")
    print("=" * 80)

    # 初始化子系统
    sat_noma = SatelliteNOMA(config)
    a2g_channel = A2GChannel()
    s2a_channel = S2AChannel()
    abs_placement = ABSPlacement()
    allocator = NOMAAllocator()

    Bd = 1.2e6
    Nd = config.get_abs_noise_power(Bd)
    Nsd = config.get_s2a_noise_power(config.Bs)

    # 1. 生成用户分布
    seed = config.random_seed
    dist = UserDistribution(config.N, config.coverage_radius, seed=seed)
    user_positions = dist.generate_uniform_circle()

    # 2. 优化ABS位置
    abs_position, _ = abs_placement.optimize_position_complete(
        user_positions, a2g_channel
    )

    # 3. 计算卫星信道增益
    elevation_deg = 10
    sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)

    # 4. 卫星NOMA配对
    sat_pairs, _ = allocator.optimal_user_pairing(sat_channel_gains)
    K = len(sat_pairs)

    # 5. 计算卫星NOMA速率
    snr_linear = 10 ** (snr_db / 10)
    sat_rates, _, sat_power_factors = sat_noma.compute_achievable_rates(
        sat_channel_gains, snr_linear
    )

    # 6. 计算A2G信道增益
    h_abs = abs_position[2]
    distances_2d = dist.compute_distances_from_point(user_positions, abs_position)
    fading_a2g = a2g_channel.generate_fading(config.N, seed=seed+1)
    channel_gains_a2g = np.array([
        a2g_channel.compute_channel_gain(
            h_abs, r, fading, config.Gd_t_dB, config.Gd_r_dB, Nd
        )
        for r, fading in zip(distances_2d, fading_a2g)
    ])

    # 7. 计算 S2A 信道增益
    satellite_altitude = config.satellite_altitude
    elevation_rad = np.deg2rad(elevation_deg)
    d_s2a = (satellite_altitude - h_abs) / np.sin(elevation_rad)
    h_s2a = s2a_channel.compute_channel_gain(
        distance=d_s2a,
        fading_gain=1.0,
        G_tx_dB=config.Gs_t_dB,
        G_rx_dB=config.Gsd_r_dB,
        noise_power=Nsd
    )

    # 8. 计算 A2G 速率
    bandwidth_per_pair = Bd / K

    a2g_rates_noma = np.zeros(2*K)
    for k in range(K):
        weak_idx, strong_idx = sat_pairs[k]
        gamma_weak = channel_gains_a2g[weak_idx]
        gamma_strong = channel_gains_a2g[strong_idx]

        if gamma_weak > gamma_strong:
            weak_idx, strong_idx = strong_idx, weak_idx
            gamma_weak, gamma_strong = gamma_strong, gamma_weak

        beta_strong, beta_weak = allocator.compute_power_factors(
            gamma_strong, gamma_weak, config.Pd
        )

        a2g_rates_noma[strong_idx] = bandwidth_per_pair * np.log2(
            1 + beta_strong * config.Pd * gamma_strong
        )
        a2g_rates_noma[weak_idx] = bandwidth_per_pair * np.log2(
            1 + beta_weak * config.Pd * gamma_weak /
            (beta_strong * config.Pd * gamma_weak + 1)
        )

    a2g_rates_oma = bandwidth_per_pair * np.log2(
        1 + config.Pd * channel_gains_a2g
    )

    print(f"\n基础信息:")
    print(f"  用户对数 K: {K}")
    print(f"  每对带宽: {bandwidth_per_pair/1e6:.2f} MHz")
    print(f"  总 Bs: {config.Bs/1e6:.1f} MHz")
    print(f"  S2A 信道增益 h_s2a: {h_s2a:.6e}")

    # 9. 用 greedy 选择模式（临时决策）
    from src.optimizers.mode_selector import GreedySelector
    mode_selector = GreedySelector()

    _, modes_temp = mode_selector.select_modes(
        sat_rates, a2g_rates_noma, a2g_rates_oma, sat_pairs
    )

    print(f"\n临时模式选择 (greedy):")
    for k, mode in enumerate(modes_temp):
        print(f"  Pair {k}: {mode}")

    # ========== 对比 Uniform 和 KKT ==========

    from src.optimizers.resource_allocator import UniformAllocator, KKTAllocator

    uniform_alloc = UniformAllocator(config)
    kkt_alloc = KKTAllocator(config)

    # Uniform 分配
    b_uniform = uniform_alloc.allocate_bandwidth(
        sat_pairs, modes_temp, a2g_rates_noma, a2g_rates_oma,
        snr_linear, h_s2a, sat_power_factors
    )

    s2a_rates_uniform = uniform_alloc.compute_s2a_rates(
        sat_pairs, b_uniform, snr_linear, h_s2a, sat_power_factors
    )

    # KKT 分配
    b_kkt = kkt_alloc.allocate_bandwidth(
        sat_pairs, modes_temp, a2g_rates_noma, a2g_rates_oma,
        snr_linear, h_s2a, sat_power_factors
    )

    s2a_rates_kkt = kkt_alloc.compute_s2a_rates(
        sat_pairs, b_kkt, snr_linear, h_s2a, sat_power_factors
    )

    print(f"\n{'='*80}")
    print("带宽分配对比")
    print(f"{'='*80}")
    print(f"\n{'Pair':<6} {'Mode':<12} {'Uniform (MHz)':<15} {'KKT (MHz)':<15} {'差异':<10}")
    print("-" * 80)

    for k in range(K):
        diff = (b_kkt[k] - b_uniform[k]) / 1e6
        print(f"{k:<6} {modes_temp[k]:<12} {b_uniform[k]/1e6:>10.2f}     {b_kkt[k]/1e6:>10.2f}     {diff:>+8.2f}")

    print(f"\n总带宽: Uniform={np.sum(b_uniform)/1e6:.2f} MHz, KKT={np.sum(b_kkt)/1e6:.2f} MHz")

    print(f"\n{'='*80}")
    print("S2A 速率对比 (Mbps)")
    print(f"{'='*80}")
    print(f"\n{'User':<6} {'A2G NOMA':<12} {'A2G OMA':<12} {'S2A Uniform':<15} {'S2A KKT':<15} {'瓶颈':<10}")
    print("-" * 80)

    for i in range(2*K):
        bottleneck_uniform = "A2G" if a2g_rates_noma[i] < s2a_rates_uniform[i] else "S2A"
        bottleneck_kkt = "A2G" if a2g_rates_noma[i] < s2a_rates_kkt[i] else "S2A"

        print(f"{i:<6} {a2g_rates_noma[i]/1e6:>8.2f}    {a2g_rates_oma[i]/1e6:>8.2f}    "
              f"{s2a_rates_uniform[i]/1e6:>10.2f}      {s2a_rates_kkt[i]/1e6:>10.2f}      "
              f"{bottleneck_kkt:<10}")

    # 计算最终速率
    abs_noma_uniform = np.minimum(a2g_rates_noma, s2a_rates_uniform)
    abs_oma_uniform = np.minimum(a2g_rates_oma, s2a_rates_uniform)

    abs_noma_kkt = np.minimum(a2g_rates_noma, s2a_rates_kkt)
    abs_oma_kkt = np.minimum(a2g_rates_oma, s2a_rates_kkt)

    # 最终模式选择
    final_rates_uniform, modes_uniform = mode_selector.select_modes(
        sat_rates, abs_noma_uniform, abs_oma_uniform, sat_pairs
    )

    final_rates_kkt, modes_kkt = mode_selector.select_modes(
        sat_rates, abs_noma_kkt, abs_oma_kkt, sat_pairs
    )

    print(f"\n{'='*80}")
    print("最终性能对比")
    print(f"{'='*80}")
    print(f"\nUniform:")
    print(f"  最终模式: {modes_uniform}")
    print(f"  总速率: {np.sum(final_rates_uniform)/1e6:.3f} Mbps")
    print(f"  频谱效率: {np.sum(final_rates_uniform)/config.Bs:.3f} bits/s/Hz")

    print(f"\nKKT:")
    print(f"  最终模式: {modes_kkt}")
    print(f"  总速率: {np.sum(final_rates_kkt)/1e6:.3f} Mbps")
    print(f"  频谱效率: {np.sum(final_rates_kkt)/config.Bs:.3f} bits/s/Hz")

    diff_pct = (np.sum(final_rates_kkt) - np.sum(final_rates_uniform)) / np.sum(final_rates_uniform) * 100
    print(f"\n差异: {diff_pct:+.1f}%")

    # 分析问题
    print(f"\n{'='*80}")
    print("问题诊断")
    print(f"{'='*80}")

    # 检查 KKT 是否分配不足
    insufficient_pairs = []
    for k in range(K):
        if modes_temp[k] != 'sat' and b_kkt[k] < b_uniform[k] * 0.5:
            insufficient_pairs.append(k)

    if insufficient_pairs:
        print(f"\n⚠️  发现 KKT 分配不足的 pair: {insufficient_pairs}")
        for k in insufficient_pairs:
            print(f"  Pair {k} ({modes_temp[k]}): KKT={b_kkt[k]/1e6:.2f}MHz < Uniform={b_uniform[k]/1e6:.2f}MHz")
    else:
        print(f"\n✓ KKT 分配量正常")

    # 检查是否有 S2A 瓶颈
    s2a_bottleneck_count_uniform = sum(1 for i in range(2*K) if s2a_rates_uniform[i] < a2g_rates_noma[i])
    s2a_bottleneck_count_kkt = sum(1 for i in range(2*K) if s2a_rates_kkt[i] < a2g_rates_noma[i])

    print(f"\nS2A 瓶颈用户数:")
    print(f"  Uniform: {s2a_bottleneck_count_uniform}/{2*K}")
    print(f"  KKT: {s2a_bottleneck_count_kkt}/{2*K}")

    if s2a_bottleneck_count_kkt > s2a_bottleneck_count_uniform:
        print(f"\n⚠️  KKT 导致更多 S2A 瓶颈！")


def main():
    """测试不同 SNR"""
    for snr in [10, 20, 30]:
        debug_single_realization(snr)
        print("\n\n")


if __name__ == "__main__":
    main()
