"""
仿真参数配置 - 基于论文 Table I

Karavolos et al., "Satellite Aerial Terrestrial Hybrid NOMA Scheme 
in 6G Networks: An Unsupervised Learning Approach", IEEE 2022
"""
import numpy as np

class SimulationConfig:
    """系统参数配置类"""
    
    # ==================== 基础配置 ====================
    # 用户配置
    N = 32                          # 总用户数
    K = N // 2                      # NOMA配对数 (K pairs)
    
    # 卫星参数
    fs = 1.625e9                    # 卫星下行频率 (Hz) - L-band
    Bs = 5e6                        # 卫星带宽 (Hz) - 5 MHz
    Ps_dB = np.arange(0, 31, 1)     # 卫星发射SNR (dB) - Figure 2 的 X 轴
    Gs_t_dB = 24                    # 卫星发射天线增益 (dBi)
    Gs_r_dB = 0                     # MT接收天线增益 (dBi) - 手持天线
    Ts_dBK = 25.7                   # 系统接收噪声温度 (dBK)
    
    # ==================== 物理常数 ====================
    c = 3e8                         # 光速 (m/s)
    k_boltzmann = 1.38e-23          # 玻尔兹曼常数 (J/K)
    
    # ==================== 信道模型参数 ====================
    # Loo模型参数（简化版 - Phase 1）
    # 参考论文：城市环境，L-band，手持天线
    # 注意：这些是调整后的值，以匹配论文Figure 2的性能曲线
    # 原始估计值 alpha=-15dB, MP=-10dB 导致信道增益过低
    alpha_dB = -2                   # LoS分量平均衰减 (dB) [调整后]
    psi_dB = 3                      # LoS分量标准差 (dB)
    MP_dB = 0                       # 多径分量功率 (dB) [调整后]
    f_doppler_max = 40e3            # 最大多普勒频移 (Hz) - 论文给出
    
    # ==================== 仿真控制 ====================
    n_monte_carlo = 1000            # Monte Carlo仿真次数
                                    # 注意：先用1000调试，后期可增加到10000
    random_seed = 42                # 随机种子（确保可复现）
    
    # ==================== 用户分布（Phase 2 使用）====================
    coverage_radius = 500           # ABS覆盖半径 (m)
    sc_ratio = 0.5                  # 强信道(SC)用户比例 - 论文设定50%
    
    # ==================== 卫星轨道（Phase 3 使用）====================
    satellite_altitude = 500e3      # VLEO卫星高度 (m) - 约500km
    
    # ==================== 辅助方法 ====================
    @classmethod
    def get_noise_power(cls):
        """
        计算噪声功率 Ns = k * T * B
        
        返回:
            Ns: 噪声功率 (W)
        """
        T_kelvin = 10 ** (cls.Ts_dBK / 10)  # dBK转K
        Ns = cls.k_boltzmann * T_kelvin * cls.Bs
        return Ns
    
    @classmethod
    def snr_to_power(cls, snr_db):
        """
        将SNR(dB)转换为实际发射功率(W)
        
        SNR = Ps / Ns  =>  Ps = Ns * 10^(SNR_dB/10)
        
        参数:
            snr_db: 信噪比 (dB)
        
        返回:
            Ps: 发射功率 (W)
        """
        Ns = cls.get_noise_power()
        Ps = Ns * (10 ** (snr_db / 10))
        return Ps
    
    @classmethod
    def power_to_snr(cls, power_watt):
        """
        将功率(W)转换为SNR(dB)
        
        参数:
            power_watt: 功率 (W)
        
        返回:
            snr_db: 信噪比 (dB)
        """
        Ns = cls.get_noise_power()
        snr_db = 10 * np.log10(power_watt / Ns)
        return snr_db
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("=" * 60)
        print("系统配置参数 (基于论文 Table I)")
        print("=" * 60)
        print(f"\n【用户配置】")
        print(f"  总用户数 N: {cls.N}")
        print(f"  配对数 K: {cls.K}")
        
        print(f"\n【卫星参数】")
        print(f"  下行频率: {cls.fs/1e9:.3f} GHz")
        print(f"  带宽: {cls.Bs/1e6:.1f} MHz")
        print(f"  发射天线增益: {cls.Gs_t_dB} dBi")
        print(f"  接收天线增益: {cls.Gs_r_dB} dBi")
        print(f"  噪声温度: {cls.Ts_dBK} dBK")
        print(f"  噪声功率: {cls.get_noise_power():.3e} W")
        
        print(f"\n【信道模型】")
        print(f"  Loo模型 - α: {cls.alpha_dB} dB")
        print(f"  Loo模型 - ψ: {cls.psi_dB} dB")
        print(f"  Loo模型 - MP: {cls.MP_dB} dB")
        print(f"  最大多普勒: {cls.f_doppler_max/1e3:.0f} kHz")
        
        print(f"\n【仿真设置】")
        print(f"  SNR范围: {cls.Ps_dB[0]}-{cls.Ps_dB[-1]} dB")
        print(f"  Monte Carlo次数: {cls.n_monte_carlo}")
        print(f"  随机种子: {cls.random_seed}")
        print("=" * 60)

# 创建全局配置实例
config = SimulationConfig()

# 测试代码
if __name__ == "__main__":
    config.print_config()
    
    # 测试SNR转换
    print("\n【测试SNR-功率转换】")
    test_snr = 20  # dB
    test_power = config.snr_to_power(test_snr)
    recovered_snr = config.power_to_snr(test_power)
    
    print(f"  SNR: {test_snr} dB")
    print(f"  功率: {test_power:.6e} W")
    print(f"  恢复SNR: {recovered_snr:.2f} dB")
    print(f"  验证: {'✓ 通过' if abs(recovered_snr - test_snr) < 0.01 else '✗ 失败'}")