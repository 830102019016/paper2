"""
信道模型实现

包含：
1. LooChannel - 简化Loo模型（陆地移动卫星信道）
2. PathLossModel - 路径损耗模型（自由空间）

论文参考：
- Section II.B: Channel model
- Loo's model distribution with parameters (α, ψ, MP)
"""
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from config import config


class LooChannel:
    """
    简化Loo模型 - 陆地移动卫星(LMS)信道

    论文描述：
    - LoS分量：对数正态分布，参数 (α, ψ)
    - 多径分量(MP)：Rayleigh分布，σ_s^2 = 0.5 * 10^(MP/10)
    - 合成：h_s_i = LoS + MP

    ⚠️ 简化假设（需在论文中明确说明）：
    - 忽略多普勒效应与 Jakes 时变模型
    - 每次 Monte Carlo 仿真采用独立信道实现
    - 忽略空间相关性

    📝 建议论文说明：
    "For simplicity, the Doppler effect is ignored and independent
    channel realizations are assumed in each Monte Carlo simulation."

    参数来源：
    - 论文引用 [5] Table 2，城市环境，L-band，手持天线
    - config.py 中使用估计值
    """
    
    def __init__(self, alpha_db, psi_db, mp_db, seed=None):
        """
        初始化Loo信道模型
        
        参数:
            alpha_db (float): LoS分量平均功率 (dB)
            psi_db (float): LoS分量标准差 (dB)
            mp_db (float): 多径分量功率 (dB)
            seed (int, optional): 随机种子，用于可复现
        """
        self.alpha_db = alpha_db
        self.psi_db = psi_db
        self.mp_db = mp_db
        self.rng = np.random.default_rng(seed)
    
    def generate_channel_gain(self, n_users):
        """
        生成 N 个用户的信道功率增益 |h_s_i|^2
        
        实现逻辑：
        1. LoS分量：对数正态分布
        2. MP分量：Rayleigh幅度的平方 = 指数分布
        3. 合成：power_gain = LoS_power + MP_power
        
        参数:
            n_users (int): 用户数量
        
        返回:
            channel_gains (ndarray): shape (n_users,)，信道功率增益
        """
        # LoS分量功率（对数正态分布）
        # 论文：power of LoS component is log-normally distributed
        alpha_linear = 10 ** (self.alpha_db / 10)
        psi_linear = 10 ** (self.psi_db / 10)
        
        # 对数正态分布：X ~ LogNormal(μ, σ)
        # E[X] = exp(μ + σ²/2), 这里简化为使用 alpha 作为 scale
        los_power = self.rng.lognormal(
            mean=np.log(alpha_linear), 
            sigma=np.log(psi_linear), 
            size=n_users
        )
        
        # 多径分量功率
        # 论文：σ_s^2 = 0.5 * 10^(MP/10)
        # Rayleigh 分布的幅度平方 = 指数分布
        mp_variance = 0.5 * 10 ** (self.mp_db / 10)
        multipath_power = self.rng.exponential(scale=mp_variance, size=n_users)
        
        # 合成信道功率增益
        channel_gains = los_power + multipath_power
        
        return channel_gains
    
    def generate_strong_weak_channels(self, n_pairs):
        """
        生成配对的强/弱信道用户

        论文场景：50% SC (strong channel) + 50% WC (weak channel)
        - SC: LoS situation（视距良好）
        - WC: deep shadowing（深度阴影）

        实现方法：
        - SC用户：增强LoS，减弱MP（增加差距以确保NOMA功率分配正确）
        - WC用户：减弱LoS，保持MP

        参数:
            n_pairs (int): 配对数量 K

        返回:
            strong_gains (ndarray): shape (n_pairs,) 强信道增益
            weak_gains (ndarray): shape (n_pairs,) 弱信道增益
        """
        # 生成 SC 用户（LoS 主导）
        # 增加差距：+8dB instead of +5dB
        self_sc = LooChannel(
            alpha_db=self.alpha_db + 8,   # LoS 更强 (+8dB，增强差距)
            psi_db=max(1, self.psi_db - 1),  # 波动更小 (-1dB)
            mp_db=self.mp_db - 5,         # 多径更弱 (-5dB)
            seed=self.rng.integers(0, 1000000)
        )
        strong_gains = self_sc.generate_channel_gain(n_pairs)

        # 生成 WC 用户（深度阴影）
        # 增加差距：-12dB instead of -10dB
        self_wc = LooChannel(
            alpha_db=self.alpha_db - 12,  # LoS 很弱 (-12dB，增强差距)
            psi_db=self.psi_db + 2,       # 波动更大 (+2dB)
            mp_db=self.mp_db,             # 多径保持不变
            seed=self.rng.integers(0, 1000000)
        )
        weak_gains = self_wc.generate_channel_gain(n_pairs)

        return strong_gains, weak_gains


class PathLossModel:
    """
    路径损耗模型
    
    包含：
    1. 自由空间路径损耗 (FSL)
    2. 卫星距离估算
    
    论文参考：
    - Section II.B: "path loss attenuation using the free space path loss (FSL) model"
    """
    
    @staticmethod
    def free_space_loss(distance, frequency):
        """
        自由空间路径损耗 (FSL)
        
        公式：L_FS = (4π * d * f / c)^2
        
        其中：
        - d: 距离 (m)
        - f: 频率 (Hz)
        - c: 光速 (m/s)
        
        参数:
            distance (float): 传播距离 (m)
            frequency (float): 频率 (Hz)
        
        返回:
            loss (float): 路径损耗（线性值，非dB）
        """
        wavelength = config.c / frequency
        loss = (4 * np.pi * distance / wavelength) ** 2
        return loss
    
    @staticmethod
    def free_space_loss_db(distance, frequency):
        """
        自由空间路径损耗 (dB表示)
        
        公式：L_FS(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
        
        参数:
            distance (float): 传播距离 (m)
            frequency (float): 频率 (Hz)
        
        返回:
            loss_db (float): 路径损耗 (dB)
        """
        loss_linear = PathLossModel.free_space_loss(distance, frequency)
        loss_db = 10 * np.log10(loss_linear)
        return loss_db
    
    @staticmethod
    def satellite_distance(elevation_angle_deg):
        """
        根据仰角估算卫星距离（简化几何模型）
        
        假设：
        - VLEO 高度 h ≈ 500 km
        - 地球曲率忽略（对于 VLEO 误差可接受）
        
        几何关系：d ≈ h / sin(elevation)
        
        参数:
            elevation_angle_deg (float): 卫星仰角 (度)
        
        返回:
            distance (float): 卫星到地面站的距离 (m)
        """
        elevation_rad = np.deg2rad(elevation_angle_deg)
        h_satellite = config.satellite_altitude  # 500 km
        
        # 简化几何：slant range ≈ h / sin(E)
        # 注意：这是近似值，精确计算需考虑地球曲率
        distance = h_satellite / np.sin(elevation_rad)
        
        return distance


# ==================== 测试代码 ====================
def test_channel_model():
    """测试信道模型功能"""
    print("=" * 60)
    print("测试信道模型")
    print("=" * 60)
    
    # 创建 Loo 信道
    loo = LooChannel(
        alpha_db=config.alpha_dB,
        psi_db=config.psi_dB,
        mp_db=config.MP_dB,
        seed=config.random_seed
    )
    
    # 测试 1：生成信道实现
    print(f"\n【测试1：生成信道实现】")
    gains = loo.generate_channel_gain(config.N)
    print(f"生成 {config.N} 个用户的信道增益:")
    print(f"  均值: {np.mean(gains):.6f}")
    print(f"  标准差: {np.std(gains):.6f}")
    print(f"  最小值: {np.min(gains):.6f}")
    print(f"  最大值: {np.max(gains):.6f}")
    print(f"  中位数: {np.median(gains):.6f}")
    
    # 测试 2：SC/WC 分离
    print(f"\n【测试2：强/弱信道对比】")
    strong, weak = loo.generate_strong_weak_channels(config.K)
    print(f"配对数 K = {config.K}")
    print(f"  强信道 SC 均值: {np.mean(strong):.6f}")
    print(f"  弱信道 WC 均值: {np.mean(weak):.6f}")
    print(f"  增益比: {np.mean(strong)/np.mean(weak):.2f}x")
    print(f"  验证: {'✓ SC > WC' if np.mean(strong) > np.mean(weak) else '✗ 失败'}")
    
    # 测试 3：路径损耗
    print(f"\n【测试3：路径损耗计算】")
    test_elevations = [10, 20, 40]
    for elev in test_elevations:
        dist = PathLossModel.satellite_distance(elev)
        loss = PathLossModel.free_space_loss(dist, config.fs)
        loss_db = 10 * np.log10(loss)
        
        print(f"  仰角 E={elev}°:")
        print(f"    距离: {dist/1e3:.1f} km")
        print(f"    路径损耗: {loss_db:.1f} dB")
    
    # 测试 4：可复现性
    print(f"\n【测试4：随机种子可复现性】")
    loo1 = LooChannel(config.alpha_dB, config.psi_dB, config.MP_dB, seed=42)
    loo2 = LooChannel(config.alpha_dB, config.psi_dB, config.MP_dB, seed=42)
    gains1 = loo1.generate_channel_gain(10)
    gains2 = loo2.generate_channel_gain(10)
    
    is_reproducible = np.allclose(gains1, gains2)
    print(f"  两次生成结果相同: {'✓ 是' if is_reproducible else '✗ 否'}")
    
    print("\n" + "=" * 60)
    print("✓ 信道模型测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_channel_model()