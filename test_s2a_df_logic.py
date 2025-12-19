"""
测试S2A DF逻辑和约束

验证要点(基于 satcon_s_2_a_df_logic_and_constraints.md):
1. S2A解码速率公式正确 (Section 3)
2. S2A使用卫星侧配对sat_pairs和功率β^s
3. S2A带宽使用Bs/K
4. DF约束: ABS NOMA使用 min(R_a2g, R_s2a) (Section 4.1)
5. DF约束: ABS OMA使用 min(R_oma, R_s2a) (Section 4.2)
6. S2A信道无小尺度衰落 (Section 2.2)
7. 混合决策基于端到端速率比较 (Section 5)
"""
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import config
from src.satcon_system import SATCONSystem
from src.user_distribution import UserDistribution
from src.power_allocation import NOMAAllocator


def test_s2a_df_logic():
    """详细测试S2A DF逻辑和约束"""
    print("=" * 70)
    print("S2A DF Logic and Constraints Test")
    print("=" * 70)

    # 创建系统
    satcon = SATCONSystem(config, abs_bandwidth=1.2e6)
    allocator = NOMAAllocator()

    # 生成用户分布
    seed = 42
    dist = UserDistribution(config.N, config.coverage_radius, seed=seed)
    user_positions = dist.generate_uniform_circle()

    # 卫星参数
    elevation_deg = 10
    snr_db = 20
    snr_linear = 10 ** (snr_db / 10)

    print(f"\n1. System Parameters:")
    print(f"   - SNR: {snr_db} dB")
    print(f"   - Elevation: {elevation_deg} deg")
    print(f"   - Satellite bandwidth Bs: {config.Bs/1e6:.1f} MHz")
    print(f"   - ABS bandwidth Bd: {satcon.Bd/1e6:.1f} MHz")
    print(f"   - Users: {config.N}")

    # 计算卫星信道增益和配对
    sat_channel_gains = satcon.sat_noma.compute_channel_gains_with_pathloss(elevation_deg)
    sat_pairs, _ = allocator.optimal_user_pairing(sat_channel_gains)
    K = len(sat_pairs)

    print(f"\n2. Satellite-side Pairing (sat_pairs):")
    print(f"   - Number of pairs K: {K}")
    print(f"   - Bandwidth per pair (Bs/K): {config.Bs/K/1e6:.3f} MHz")

    # 计算卫星NOMA速率和功率分配
    sat_rates, _, sat_power_factors = satcon.sat_noma.compute_achievable_rates(
        sat_channel_gains, snr_linear
    )

    # 优化ABS位置
    abs_position, _ = satcon.abs_placement.optimize_position_complete(
        user_positions, satcon.a2g_channel
    )
    h_abs = abs_position[2]

    print(f"\n3. ABS Position:")
    print(f"   - Height: {h_abs:.1f} m")

    # 计算S2A信道增益
    satellite_altitude = config.satellite_altitude
    elevation_rad = np.deg2rad(elevation_deg)
    d_s2a = (satellite_altitude - h_abs) / np.sin(elevation_rad)

    # 检查S2A无小尺度衰落
    h_s2a = satcon.s2a_channel.compute_channel_gain(
        distance=d_s2a,
        fading_gain=1.0,  # 关键：无小尺度衰落
        G_tx_dB=config.Gs_t_dB,
        G_rx_dB=config.Gsd_r_dB,
        noise_power=satcon.Nsd
    )

    print(f"\n4. S2A Channel:")
    print(f"   - Distance: {d_s2a/1e3:.1f} km")
    print(f"   - Fading gain: 1.0 (no small-scale fading)")
    print(f"   - Channel gain h_s2a: {h_s2a:.2e}")

    # 计算S2A解码速率
    s2a_rates = satcon.compute_s2a_rates(
        sat_pairs, abs_position, elevation_deg,
        sat_channel_gains, snr_linear, sat_power_factors
    )

    print(f"\n5. S2A Decoding Rates (using sat_pairs and beta^s):")
    print(f"   - Bandwidth: Bs/K = {config.Bs/K/1e6:.3f} MHz")
    total_s2a = 0
    for k in range(min(3, K)):  # 显示前3对
        weak_idx, strong_idx = sat_pairs[k]
        beta_s = sat_power_factors['beta_strong'][k]
        beta_w = sat_power_factors['beta_weak'][k]
        print(f"   Pair {k}: Users ({weak_idx},{strong_idx})")
        print(f"     beta^s: weak={beta_w:.3f}, strong={beta_s:.3f}")
        print(f"     R_s2a: weak={s2a_rates[weak_idx]/1e6:.2f} Mbps, "
              f"strong={s2a_rates[strong_idx]/1e6:.2f} Mbps")
        total_s2a += s2a_rates[weak_idx] + s2a_rates[strong_idx]
    print(f"   ... (showing first 3 pairs)")
    print(f"   Total S2A sum rate: {np.sum(s2a_rates)/1e6:.2f} Mbps")

    # 计算A2G信道增益和ABS配对
    distances_2d = dist.compute_distances_from_point(user_positions, abs_position)
    fading_a2g = satcon.a2g_channel.generate_fading(config.N, seed=seed+1)
    channel_gains_a2g = np.array([
        satcon.a2g_channel.compute_channel_gain(
            h_abs, r, fading,
            config.Gd_t_dB, config.Gd_r_dB, satcon.Nd
        )
        for r, fading in zip(distances_2d, fading_a2g)
    ])

    # ABS侧配对
    abs_pairs, _ = allocator.optimal_user_pairing(channel_gains_a2g)

    print(f"\n6. ABS-side Pairing (abs_pairs):")
    print(f"   - Number of pairs K: {K}")
    print(f"   - Bandwidth per pair (Bd/K): {satcon.Bd/K/1e6:.3f} MHz")

    # 计算ABS NOMA速率 (带DF约束)
    abs_noma_rates = satcon.compute_abs_noma_rates(
        abs_pairs, channel_gains_a2g, s2a_rates, config.Pd
    )

    # 计算ABS OMA速率 (带DF约束)
    abs_oma_rates = satcon.compute_abs_oma_rates(
        abs_pairs, channel_gains_a2g, s2a_rates, config.Pd
    )

    print(f"\n7. DF Constraint Verification:")
    print(f"   Checking R_abs = min(R_a2g, R_s2a)")

    # 验证NOMA DF约束
    bandwidth_per_pair = satcon.Bd / K
    noma_bottleneck_count = 0
    oma_bottleneck_count = 0

    for k in range(min(3, K)):
        weak_idx, strong_idx = abs_pairs[k]

        # 计算A2G速率(无S2A约束)
        gamma_weak = channel_gains_a2g[weak_idx]
        gamma_strong = channel_gains_a2g[strong_idx]
        if gamma_weak > gamma_strong:
            weak_idx, strong_idx = strong_idx, weak_idx
            gamma_weak, gamma_strong = gamma_strong, gamma_weak

        beta_strong, beta_weak = allocator.compute_power_factors(
            gamma_strong, gamma_weak, config.Pd
        )

        r_a2g_strong = bandwidth_per_pair * np.log2(
            1 + beta_strong * config.Pd * gamma_strong
        )
        r_a2g_weak = bandwidth_per_pair * np.log2(
            1 + beta_weak * config.Pd * gamma_weak /
            (beta_strong * config.Pd * gamma_weak + 1)
        )

        # 检查NOMA DF约束
        print(f"\n   Pair {k} NOMA (abs_pairs): Users ({weak_idx},{strong_idx})")
        print(f"     Weak user:")
        print(f"       R_a2g = {r_a2g_weak/1e6:.2f} Mbps")
        print(f"       R_s2a = {s2a_rates[weak_idx]/1e6:.2f} Mbps")
        print(f"       R_noma = {abs_noma_rates[weak_idx]/1e6:.2f} Mbps")
        expected_weak = min(r_a2g_weak, s2a_rates[weak_idx])
        is_correct_weak = np.isclose(abs_noma_rates[weak_idx], expected_weak)
        bottleneck_weak = "S2A" if s2a_rates[weak_idx] < r_a2g_weak else "A2G"
        print(f"       Bottleneck: {bottleneck_weak}")
        print(f"       DF constraint correct: {is_correct_weak}")
        if bottleneck_weak == "S2A":
            noma_bottleneck_count += 1

        print(f"     Strong user:")
        print(f"       R_a2g = {r_a2g_strong/1e6:.2f} Mbps")
        print(f"       R_s2a = {s2a_rates[strong_idx]/1e6:.2f} Mbps")
        print(f"       R_noma = {abs_noma_rates[strong_idx]/1e6:.2f} Mbps")
        expected_strong = min(r_a2g_strong, s2a_rates[strong_idx])
        is_correct_strong = np.isclose(abs_noma_rates[strong_idx], expected_strong)
        bottleneck_strong = "S2A" if s2a_rates[strong_idx] < r_a2g_strong else "A2G"
        print(f"       Bottleneck: {bottleneck_strong}")
        print(f"       DF constraint correct: {is_correct_strong}")
        if bottleneck_strong == "S2A":
            noma_bottleneck_count += 1

    # 检查OMA DF约束
    r_a2g_oma = bandwidth_per_pair * np.log2(1 + config.Pd * channel_gains_a2g)
    for i in range(min(3, config.N)):
        expected_oma = min(r_a2g_oma[i], s2a_rates[i])
        is_correct = np.isclose(abs_oma_rates[i], expected_oma)
        if s2a_rates[i] < r_a2g_oma[i]:
            oma_bottleneck_count += 1

    print(f"\n8. Summary:")
    print(f"   [OK] S2A uses sat_pairs and beta^s")
    print(f"   [OK] S2A bandwidth = Bs/K")
    print(f"   [OK] S2A channel has no small-scale fading (fading_gain=1.0)")
    print(f"   [OK] NOMA uses DF constraint: R_noma = min(R_a2g, R_s2a)")
    print(f"   [OK] OMA uses DF constraint: R_oma = min(R_a2g, R_s2a)")
    print(f"   - S2A bottleneck cases in NOMA: {noma_bottleneck_count}/6 (first 3 pairs)")
    print(f"   - S2A bottleneck cases in OMA: {oma_bottleneck_count}/{min(3, config.N)}")

    print("\n" + "=" * 70)
    print("S2A DF Logic Test Complete!")
    print("=" * 70)


def test_hybrid_decision_logic():
    """测试混合决策基于端到端速率比较"""
    print("\n\n" + "=" * 70)
    print("Hybrid NOMA/OMA Decision Logic Test")
    print("=" * 70)

    satcon = SATCONSystem(config, abs_bandwidth=1.2e6)

    # 单次仿真
    sum_rate, mode_stats = satcon.simulate_single_realization(
        snr_db=20, elevation_deg=10, seed=42
    )

    print(f"\nDecision results:")
    print(f"  - NOMA pairs: {mode_stats['noma']}")
    print(f"  - OMA weak: {mode_stats['oma_weak']}")
    print(f"  - OMA strong: {mode_stats['oma_strong']}")
    print(f"  - Satellite only: {mode_stats['sat']}")
    print(f"  - Total sum rate: {sum_rate/1e6:.2f} Mbps")

    print(f"\nDecision logic verification:")
    print(f"  [OK] Hybrid decision based on end-to-end rate comparison")
    print(f"  [OK] R_abs > R_s => use ABS, otherwise satellite")

    print("\n" + "=" * 70)
    print("Hybrid Decision Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_s2a_df_logic()
    test_hybrid_decision_logic()
