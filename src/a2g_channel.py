"""
A2G和S2A信道模型

实现：
1. A2G (Air-to-Ground) 信道 - ABS到地面用户
2. S2A (Satellite-to-Air) 信道 - 卫星到ABS
3. 基于3GPP TR 38.811标准

论文参考：
- Section II.B: A2G channel model
- 公式(1): L^{A2G}(h,r) = 20log(4πy*fd/c) + η_LoS*P_LoS + η_NLoS*(1-P_LoS)
- 引用[16]: elevation angle-based path loss model
"""
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import config


class A2GChannel:
    """
    Air-to-Ground (A2G) 信道模型
    
    基于3GPP TR 38.811 V15.4.0标准
    适用于UAV/ABS到地面用户的通信
    """
    
    def __init__(self, frequency=None, environment='urban'):
        """
        初始化A2G信道模型
        
        参数:
            frequency (float): 频率 (Hz)，默认使用config.fd
            environment (str): 环境类型 ('urban', 'suburban', 'rural')
        """
        self.fd = frequency if frequency is not None else config.fd
        self.environment = environment
        
        # 3GPP TR 38.811 参数（城市环境）
        if environment == 'urban':
            self.eta_los = config.a2g_eta_los      # 1.0 dB
            self.eta_nlos = config.a2g_eta_nlos    # 20.0 dB
            self.a = config.a2g_param_a            # 9.61
            self.b = config.a2g_param_b            # 0.16
        elif environment == 'suburban':
            self.eta_los = 0.1
            self.eta_nlos = 21.0
            self.a = 4.88
            self.b = 0.43
        elif environment == 'rural':
            self.eta_los = 0.06
            self.eta_nlos = 23.0
            self.a = 0.0
            self.b = 0.0
        else:
            raise ValueError(f"未知环境类型: {environment}")
    
    def compute_los_probability(self, h, r):
        """
        计算LoS概率
        
        3GPP模型:
        P_LoS = 1 / (1 + a * exp(-b * [θ - a]))
        
        其中 θ = arctan(h/r) * 180/π (仰角，单位：度)
        
        参数:
            h (float or ndarray): ABS高度 (m)
            r (float or ndarray): 2D水平距离 (m)
        
        返回:
            p_los (float or ndarray): LoS概率 [0, 1]
        """
        # 避免除零
        r = np.maximum(r, 1e-6)
        
        # 计算仰角（度）
        theta_deg = np.arctan2(h, r) * 180 / np.pi
        
        # LoS概率
        if self.environment == 'rural':
            # 农村环境：始终LoS
            p_los = 1.0
        else:
            exponent = -self.b * (theta_deg - self.a)
            # 数值稳定性处理
            exponent = np.clip(exponent, -50, 50)
            p_los = 1.0 / (1.0 + self.a * np.exp(exponent))
        
        return p_los
    
    def compute_pathloss(self, h, r):
        """
        计算A2G路径损耗
        
        论文公式(1):
        L^{A2G}(h,r) = 20*log10(4π*y*fd/c) + η_LoS*P_LoS(h,r) + η_NLoS*(1-P_LoS(h,r))
        
        其中:
        - y = sqrt(r² + h²) 是3D直线距离
        - fd 是频率
        - c 是光速
        
        参数:
            h (float or ndarray): ABS高度 (m)
            r (float or ndarray): 2D水平距离 (m)
        
        返回:
            loss_linear (float or ndarray): 路径损耗（线性值，非dB）
        """
        # 1. 计算3D距离
        y = np.sqrt(r**2 + h**2)
        
        # 2. 自由空间路径损耗（dB）
        # FSPL = 20*log10(4π*y*f/c)
        fspl_db = 20 * np.log10(4 * np.pi * y * self.fd / config.c)
        
        # 3. LoS概率
        p_los = self.compute_los_probability(h, r)
        
        # 4. 环境额外损耗（dB）
        env_loss_db = self.eta_los * p_los + self.eta_nlos * (1 - p_los)
        
        # 5. 总路径损耗（dB）
        total_loss_db = fspl_db + env_loss_db
        
        # 6. 转换为线性值
        loss_linear = 10 ** (total_loss_db / 10)
        
        return loss_linear
    
    def compute_pathloss_db(self, h, r):
        """计算A2G路径损耗（dB表示）"""
        loss_linear = self.compute_pathloss(h, r)
        return 10 * np.log10(loss_linear)
    
    def generate_fading(self, n_users, seed=None):
        """
        生成Rayleigh衰落
        
        论文: A2G信道的多径分量服从Rayleigh分布
        h^d_i ~ CN(0, 1) => |h^d_i|² ~ Exponential(1)
        
        参数:
            n_users (int): 用户数量
            seed (int, optional): 随机种子
        
        返回:
            fading_gains (ndarray): shape (n_users,) 衰落功率增益
        """
        rng = np.random.default_rng(seed)
        
        # 复高斯 CN(0, 1)
        real = rng.normal(0, 1/np.sqrt(2), n_users)
        imag = rng.normal(0, 1/np.sqrt(2), n_users)
        
        # 功率 |h|²
        fading_gains = real**2 + imag**2
        
        return fading_gains
    
    def compute_channel_gain(self, h, r, fading_gain, 
                            G_tx_dB, G_rx_dB, noise_power):
        """
        计算完整的A2G信道增益
        
        Γ^d_l = (G_tx * G_rx) / (L^{A2G} * Nd) * |h^d_l|²
        
        参数:
            h: ABS高度
            r: 2D距离
            fading_gain: 小尺度衰落
            G_tx_dB: 发射天线增益 (dBi)
            G_rx_dB: 接收天线增益 (dBi)
            noise_power: 噪声功率 (W)
        
        返回:
            channel_gain: 信道增益（无量纲）
        """
        # 路径损耗
        path_loss = self.compute_pathloss(h, r)
        
        # 天线增益（转线性）
        G_tx = 10 ** (G_tx_dB / 10)
        G_rx = 10 ** (G_rx_dB / 10)
        
        # 完整信道增益
        channel_gain = (G_tx * G_rx / (path_loss * noise_power)) * fading_gain
        
        return channel_gain


class S2AChannel:
    """
    Satellite-to-Air (S2A) 信道模型
    
    卫星到ABS的链路，使用自由空间路径损耗
    """
    
    def __init__(self, frequency=None):
        """
        初始化S2A信道模型
        
        参数:
            frequency (float): 频率 (Hz)，默认使用config.fs
        """
        self.fs = frequency if frequency is not None else config.fs
    
    def compute_pathloss(self, distance):
        """
        计算S2A路径损耗（自由空间）
        
        与Phase 1的FSL模型相同
        
        参数:
            distance (float): 卫星到ABS的距离 (m)
        
        返回:
            loss_linear (float): 路径损耗（线性值）
        """
        from src.channel_models import PathLossModel
        return PathLossModel.free_space_loss(distance, self.fs)
    
    def compute_pathloss_db(self, distance):
        """计算S2A路径损耗（dB）"""
        loss_linear = self.compute_pathloss(distance)
        return 10 * np.log10(loss_linear)
    
    def compute_channel_gain(self, distance, fading_gain,
                            G_tx_dB, G_rx_dB, noise_power):
        """
        计算S2A信道增益
        
        Λ^{sd} = (Gs_t * Gsd_r) / (L^{FS}_sd * Nsd)
        
        注意：论文公式(7)中没有小尺度衰落项
        
        参数:
            distance: 卫星到ABS距离
            fading_gain: 小尺度衰落（可设为1，如果不考虑）
            G_tx_dB: 卫星发射增益
            G_rx_dB: ABS接收增益
            noise_power: 噪声功率
        
        返回:
            channel_gain: 信道增益
        """
        path_loss = self.compute_pathloss(distance)
        G_tx = 10 ** (G_tx_dB / 10)
        G_rx = 10 ** (G_rx_dB / 10)
        
        channel_gain = (G_tx * G_rx / (path_loss * noise_power)) * fading_gain
        
        return channel_gain


# ==================== 测试代码 ====================
def test_a2g_channel():
    """测试A2G信道模型"""
    print("=" * 60)
    print("测试A2G/S2A信道模型")
    print("=" * 60)
    
    # 测试1：A2G路径损耗
    print(f"\n【测试1：A2G路径损耗】")
    a2g = A2GChannel()
    
    test_heights = [50, 100, 200]
    test_distances = [100, 300, 500]
    
    print(f"频率: {a2g.fd/1e9:.1f} GHz")
    print(f"环境: {a2g.environment}")
    print(f"\n路径损耗测试:")
    print(f"{'高度(m)':<10} {'距离(m)':<10} {'LoS概率':<12} {'损耗(dB)':<12}")
    print("-" * 50)
    
    for h in test_heights:
        for r in test_distances:
            p_los = a2g.compute_los_probability(h, r)
            loss_db = a2g.compute_pathloss_db(h, r)
            print(f"{h:<10} {r:<10} {p_los:<12.3f} {loss_db:<12.2f}")
    
    # 测试2：Rayleigh衰落
    print(f"\n【测试2：Rayleigh衰落生成】")
    fading = a2g.generate_fading(n_users=1000, seed=42)
    print(f"  生成 1000 个衰落实现")
    print(f"  均值: {np.mean(fading):.3f} (理论值=1.0)")
    print(f"  标准差: {np.std(fading):.3f} (理论值=1.0)")
    print(f"  验证: {'✓ 接近理论值' if abs(np.mean(fading)-1.0)<0.1 else '⚠ 偏差'}")
    
    # 测试3：完整信道增益
    print(f"\n【测试3：完整A2G信道增益】")
    h_test = 100  # m
    r_test = 200  # m
    fading_test = 1.0  # 不考虑衰落
    Bd_test = 1.2e6  # 1.2 MHz
    Nd_test = config.get_abs_noise_power(Bd_test)
    
    gamma = a2g.compute_channel_gain(
        h_test, r_test, fading_test,
        config.Gd_t_dB, config.Gd_r_dB, Nd_test
    )
    
    print(f"  高度: {h_test} m")
    print(f"  距离: {r_test} m")
    print(f"  信道增益: {gamma:.6e}")
    print(f"  信道增益(dB): {10*np.log10(gamma):.2f} dB")
    
    # 测试4：S2A信道
    print(f"\n【测试4：S2A信道模型】")
    s2a = S2AChannel()
    
    # 假设卫星在天顶
    sat_distance = config.satellite_altitude  # 500 km
    s2a_loss_db = s2a.compute_pathloss_db(sat_distance)
    
    print(f"  卫星高度: {sat_distance/1e3:.0f} km")
    print(f"  S2A路径损耗: {s2a_loss_db:.2f} dB")
    
    # S2A信道增益
    Nsd = config.get_s2a_noise_power()
    lambda_sd = s2a.compute_channel_gain(
        sat_distance, 1.0,  # 不考虑衰落
        config.Gs_t_dB, config.Gsd_r_dB, Nsd
    )
    
    print(f"  S2A信道增益: {lambda_sd:.6e}")
    print(f"  S2A信道增益(dB): {10*np.log10(lambda_sd):.2f} dB")
    
    print("\n" + "=" * 60)
    print("✓ A2G/S2A信道模型测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_a2g_channel()
