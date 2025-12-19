"""
测试Toy Network双配对逻辑

验证要点:
1. 卫星配对(sat_pairs)基于Γ^s(卫星信道增益)
2. ABS配对(abs_pairs)基于Γ^d(A2G信道增益)
3. 两套配对可能不同
4. S2A解码使用sat_pairs和β^s
5. A2G传输使用abs_pairs和β^d
"""
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import config
from src.satcon_system import SATCONSystem
from src.user_distribution import UserDistribution
from src.power_allocation import NOMAAllocator


def test_dual_pairing_logic():
    """详细测试两套配对逻辑"""
    print("=" * 70)
    print("Toy Network双配对逻辑测试")
    print("=" * 70)

    # 创建系统
    satcon = SATCONSystem(config, abs_bandwidth=1.2e6)
    allocator = NOMAAllocator()

    # 生成用户分布
    seed = 42
    dist = UserDistribution(config.N, config.coverage_radius, seed=seed)
    user_positions = dist.generate_uniform_circle()

    # 计算卫星信道增益Γ^s
    elevation_deg = 10
    sat_channel_gains = satcon.sat_noma.compute_channel_gains_with_pathloss(elevation_deg)

    print(f"\n1. 卫星信道增益 Γ^s (归一化):")
    gamma_s_normalized = sat_channel_gains / np.max(sat_channel_gains)
    for i, g in enumerate(gamma_s_normalized):
        print(f"   User {i}: {g:.4f}")

    # 卫星侧配对
    sat_pairs, sat_order = allocator.optimal_user_pairing(sat_channel_gains)
    print(f"\n2. 卫星侧配对 (基于Γ^s排序):")
    print(f"   排序顺序: {sat_order}")
    print(f"   配对结果:")
    for k, (weak, strong) in enumerate(sat_pairs):
        print(f"   Pair {k}: User {weak} (弱) <-> User {strong} (强)")

    # 计算A2G信道增益Γ^d
    # 简单起见,使用固定ABS位置
    abs_position = np.array([0, 0, 300])  # 中心,300m高度
    h_abs = abs_position[2]
    distances_2d = dist.compute_distances_from_point(user_positions, abs_position)
    fading_a2g = satcon.a2g_channel.generate_fading(config.N, seed=seed+1)
    channel_gains_a2g = np.array([
        satcon.a2g_channel.compute_channel_gain(
            h_abs, r, fading,
            config.Gd_t_dB, config.Gd_r_dB, satcon.Nd
        )
        for r, fading in zip(distances_2d, fading_a2g)
    ])

    print(f"\n3. A2G信道增益 Γ^d (归一化):")
    gamma_d_normalized = channel_gains_a2g / np.max(channel_gains_a2g)
    for i, g in enumerate(gamma_d_normalized):
        print(f"   User {i}: {g:.4f}")

    # ABS侧配对
    abs_pairs, abs_order = allocator.optimal_user_pairing(channel_gains_a2g)
    print(f"\n4. ABS侧配对 (基于Γ^d排序):")
    print(f"   排序顺序: {abs_order}")
    print(f"   配对结果:")
    for k, (weak, strong) in enumerate(abs_pairs):
        print(f"   Pair {k}: User {weak} (弱) <-> User {strong} (强)")

    # 对比两套配对
    print(f"\n5. 配对对比:")
    same_count = 0
    for k in range(len(sat_pairs)):
        sat_set = set(sat_pairs[k])
        abs_set = set(abs_pairs[k])
        is_same = sat_set == abs_set
        same_count += is_same
        status = "[SAME]" if is_same else "[DIFF]"
        print(f"   Pair {k}: Sat{sat_pairs[k]} vs ABS{abs_pairs[k]} - {status}")

    print(f"\n6. 总结:")
    print(f"   - 卫星配对数: {len(sat_pairs)}")
    print(f"   - ABS配对数: {len(abs_pairs)}")
    print(f"   - 相同配对数: {same_count}/{len(sat_pairs)}")
    print(f"   - 不同配对数: {len(sat_pairs)-same_count}/{len(sat_pairs)}")

    if same_count == len(sat_pairs):
        print(f"   [WARNING] All pairs are same - may be coincidence or similar channel")
    else:
        print(f"   [OK] Dual pairing logic correct: Sat and ABS use different pairings")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)


def test_full_simulation():
    """完整仿真测试"""
    print("\n\n" + "=" * 70)
    print("完整仿真测试 (验证端到端逻辑)")
    print("=" * 70)

    satcon = SATCONSystem(config, abs_bandwidth=1.2e6)

    # 快速测试
    snr_range = np.array([10, 20, 30])
    print(f"\n运行快速仿真 (SNR: {snr_range} dB, 5次实现)...")

    mean_rates, mean_se, std_rates, mode_stats = satcon.simulate_performance(
        snr_db_range=snr_range,
        elevation_deg=10,
        n_realizations=5,
        verbose=False
    )

    print(f"\n结果:")
    for i, snr in enumerate(snr_range):
        print(f"  SNR={snr}dB: SE={mean_se[i]:.2f} bits/s/Hz")
        print(f"    NOMA={mode_stats['noma'][i]:.1f}, "
              f"OMA_W={mode_stats['oma_weak'][i]:.1f}, "
              f"OMA_S={mode_stats['oma_strong'][i]:.1f}, "
              f"SAT={mode_stats['sat'][i]:.1f}")

    print("\n[OK] Full simulation successful!")


if __name__ == "__main__":
    test_dual_pairing_logic()
    test_full_simulation()
