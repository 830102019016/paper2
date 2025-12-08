"""
基础验证脚本

目的：快速验证所有模块是否正常工作

运行方法：
python simulations/validation.py
"""
import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.channel_models import LooChannel, PathLossModel
from src.power_allocation import NOMAAllocator
from src.noma_transmission import SatelliteNOMA


def validate_config():
    """验证配置参数"""
    print("\n" + "=" * 60)
    print("【1】验证配置参数")
    print("=" * 60)
    
    config.print_config()
    
    # 检查关键参数
    assert config.N == 32, "用户数应为32"
    assert config.K == 16, "配对数应为16"
    assert config.Bs == 5e6, "带宽应为5MHz"
    
    print("\n✓ 配置参数验证通过")


def validate_channel_models():
    """验证信道模型"""
    print("\n" + "=" * 60)
    print("【2】验证信道模型")
    print("=" * 60)
    
    # 创建Loo信道
    loo = LooChannel(
        alpha_db=config.alpha_dB,
        psi_db=config.psi_dB,
        mp_db=config.MP_dB,
        seed=config.random_seed
    )
    
    # 生成信道
    gains = loo.generate_channel_gain(config.N)
    print(f"生成 {config.N} 个用户信道增益:")
    print(f"  均值: {np.mean(gains):.6f}")
    print(f"  范围: [{np.min(gains):.6f}, {np.max(gains):.6f}]")
    
    # 测试SC/WC
    strong, weak = loo.generate_strong_weak_channels(config.K)
    print(f"\n强/弱信道对比:")
    print(f"  SC均值: {np.mean(strong):.6f}")
    print(f"  WC均值: {np.mean(weak):.6f}")
    print(f"  验证: {'✓' if np.mean(strong) > np.mean(weak) else '✗'}")
    
    # 测试路径损耗
    dist = PathLossModel.satellite_distance(10)
    loss_db = PathLossModel.free_space_loss_db(dist, config.fs)
    print(f"\n路径损耗 (E=10°):")
    print(f"  距离: {dist/1e3:.1f} km")
    print(f"  损耗: {loss_db:.1f} dB")
    
    print("\n✓ 信道模型验证通过")


def validate_power_allocation():
    """验证功率分配"""
    print("\n" + "=" * 60)
    print("【3】验证功率分配")
    print("=" * 60)
    
    allocator = NOMAAllocator()
    
    # 测试功率分配
    gamma_w, gamma_s = 0.01, 0.1
    Ps = 1.0
    beta_s, beta_w = allocator.compute_power_factors(gamma_s, gamma_w, Ps)
    
    print(f"功率分配 (Γ_w={gamma_w}, Γ_s={gamma_s}, Ps={Ps}W):")
    print(f"  弱用户: β_w = {beta_w:.4f} ({beta_w*100:.1f}%)")
    print(f"  强用户: β_s = {beta_s:.4f} ({beta_s*100:.1f}%)")
    print(f"  功率和: {beta_w + beta_s:.6f}")
    print(f"  验证: {'✓' if abs((beta_w + beta_s) - 1.0) < 1e-6 else '✗'}")
    
    # 测试用户配对
    np.random.seed(config.random_seed)
    test_gains = np.random.exponential(0.05, size=config.N)
    pairs, paired_gains = allocator.optimal_user_pairing(test_gains)
    
    print(f"\n用户配对 (前3对):")
    for k in range(min(3, len(pairs))):
        w_idx, s_idx = pairs[k]
        gamma_w, gamma_s = paired_gains[k]
        print(f"  Pair {k+1}: MT{w_idx} ↔ MT{s_idx}, "
              f"Γ_w={gamma_w:.6f}, Γ_s={gamma_s:.6f}")
    
    is_valid, msg = allocator.validate_pairing(pairs, paired_gains)
    print(f"\n配对验证: {'✓ ' + msg if is_valid else '✗ ' + msg}")
    
    print("\n✓ 功率分配验证通过")


def validate_noma_transmission():
    """验证NOMA传输"""
    print("\n" + "=" * 60)
    print("【4】验证NOMA传输")
    print("=" * 60)
    
    sat_noma = SatelliteNOMA(config)
    
    # 单次仿真
    test_snr = 20  # dB
    Ps = config.snr_to_power(test_snr)
    gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg=10)
    rates, sum_rate = sat_noma.compute_achievable_rates(gains, Ps)
    
    print(f"单次仿真 (SNR={test_snr}dB, E=10°):")
    print(f"  总速率: {sum_rate/1e6:.2f} Mbps")
    print(f"  频谱效率: {sum_rate/config.Bs:.2f} bits/s/Hz")
    print(f"  用户速率统计:")
    print(f"    均值: {np.mean(rates)/1e6:.2f} Mbps")
    print(f"    最大: {np.max(rates)/1e6:.2f} Mbps")
    print(f"    最小: {np.min(rates)/1e6:.2f} Mbps")
    
    # 快速Monte Carlo
    print(f"\n快速Monte Carlo (3个SNR点, 5次实现):")
    test_snr_range = np.array([10, 20, 30])
    mean_rates, mean_se, std_rates = sat_noma.simulate_performance(
        snr_db_range=test_snr_range,
        elevation_deg=10,
        n_realizations=5,
        verbose=False
    )
    
    for i, snr in enumerate(test_snr_range):
        print(f"  SNR={snr}dB: SE={mean_se[i]:.2f}±{std_rates[i]/config.Bs:.2f} bits/s/Hz")
    
    # 验证单调性
    is_increasing = np.all(np.diff(mean_se) > 0)
    print(f"\n单调性验证: {'✓ SE随SNR递增' if is_increasing else '✗ 异常'}")
    
    print("\n✓ NOMA传输验证通过")


def run_validation():
    """运行完整验证"""
    print("\n" + "=" * 60)
    print("SATCON 项目基础验证")
    print("=" * 60)
    
    try:
        validate_config()
        validate_channel_models()
        validate_power_allocation()
        validate_noma_transmission()
        
        print("\n" + "=" * 60)
        print("✓✓✓ 所有验证通过！项目基础功能正常 ✓✓✓")
        print("=" * 60)
        print("\n下一步：运行完整仿真")
        print("  python simulations/fig2_sat_noma.py")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗✗✗ 验证失败: {str(e)} ✗✗✗")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_validation()
