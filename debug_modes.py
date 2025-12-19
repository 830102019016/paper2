"""
调试脚本：检查为什么两个方法性能不同
"""
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import config
from src.user_distribution import UserDistribution
from src.abs_placement import ABSPlacement
from src.a2g_channel import A2GChannel
from src.noma_transmission import SatelliteNOMA
from src.power_allocation import NOMAAllocator
from src.optimizers.mode_selector import GreedySelector, HeuristicSelector
from src.optimizers.resource_allocator import KKTAllocator, UniformAllocator

# 配置参数
snr_db = 10
elevation_deg = 10
abs_bandwidth = 1.2e6

# 生成用户分布
dist = UserDistribution(config.N, config.coverage_radius, seed=config.random_seed)
user_positions = dist.generate_uniform_circle()

# 计算卫星传输参数
sat_noma = SatelliteNOMA(config)
sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)

allocator = NOMAAllocator()
sat_pairs, _ = allocator.optimal_user_pairing(sat_channel_gains)

snr_linear = 10 ** (snr_db / 10)
sat_rates, _, sat_power_factors = sat_noma.compute_achievable_rates(
    sat_channel_gains, snr_linear
)

print(f"卫星直达速率（sat_rates）:")
print(f"  总速率: {np.sum(sat_rates)/1e6:.2f} Mbps")
print(f"  平均速率: {np.mean(sat_rates)/1e6:.2f} Mbps")
print(f"  这个值是固定的，不应该因为Baseline和我们的算法而改变")

# Baseline方法
baseline_mode_selector = HeuristicSelector()
baseline_s2a_allocator = UniformAllocator(config)

a2g_channel = A2GChannel()
placement = ABSPlacement(
    abs_height_range=(config.abs_height_min, config.abs_height_max),
    height_step=config.abs_height_step
)

abs_position_baseline, _ = placement.optimize_position_complete(user_positions, a2g_channel)

# 我们的方法
ours_mode_selector = GreedySelector()
ours_s2a_allocator = KKTAllocator(config)

# 这里我们使用相同的ABS位置来验证
abs_position_ours = abs_position_baseline.copy()

print(f"\n使用相同的ABS位置: {abs_position_baseline}")
print(f"\n现在检查两个模式选择器是否会给出不同的结果...")

# 直接测试模式选择
# 创建虚拟的ABS速率（全部为0，这样所有用户都应该选择卫星）
abs_noma_rates = np.zeros(len(user_positions))
abs_oma_rates = np.zeros(len(user_positions))

print(f"\n【测试1：ABS速率全为0】")
print(f"在这种情况下，所有用户都应该选择卫星直达")

baseline_rates_1, baseline_modes_1 = baseline_mode_selector.select_modes(
    sat_rates, abs_noma_rates, abs_oma_rates, sat_pairs
)

ours_rates_1, ours_modes_1 = ours_mode_selector.select_modes(
    sat_rates, abs_noma_rates, abs_oma_rates, sat_pairs
)

print(f"\nBaseline总速率: {np.sum(baseline_rates_1)/1e6:.2f} Mbps")
print(f"我们的算法总速率: {np.sum(ours_rates_1)/1e6:.2f} Mbps")
print(f"差异: {(np.sum(ours_rates_1) - np.sum(baseline_rates_1))/1e6:.4f} Mbps")

if np.sum(baseline_rates_1) != np.sum(ours_rates_1):
    print(f"\n[警告] 即使ABS速率为0，两个方法仍然给出不同的结果！")
    print(f"\nBaseline返回的速率:")
    print(baseline_rates_1[:10])
    print(f"\n我们的算法返回的速率:")
    print(ours_rates_1[:10])

    print(f"\nBaseline模式:")
    print(baseline_modes_1)
    print(f"\n我们的算法模式:")
    print(ours_modes_1)
else:
    print(f"\n[OK] 当ABS速率为0时，两个方法给出相同的结果")
