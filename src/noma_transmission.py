"""
卫星NOMA传输系统实现

实现：
1. 完整信道增益计算（小尺度衰落 + 路径损耗）
2. NOMA可达速率计算（论文公式5, 6）
3. Monte Carlo性能仿真

论文参考：
- Section III.B: Satellite NOMA transmission
- 公式(3): Γ_l = (Gs_t * Gs_r) / (L_FS * Ns) * |h_l|^2
- 公式(5): R_j = (Bs/K) * log2(1 + β_j * Ps * Γ_j)
- 公式(6): R_i = (Bs/K) * log2(1 + (1-β_j)*Ps*Γ_i / (β_j*Ps*Γ_i + 1))
"""
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm  # 进度条

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from config import config
from src.power_allocation import NOMAAllocator
from src.channel_models import LooChannel, PathLossModel


class SatelliteNOMA:
    """
    卫星NOMA传输系统
    
    功能：
    1. 信道增益计算（含路径损耗）
    2. NOMA速率计算
    3. 性能仿真（Monte Carlo）
    """
    
    def __init__(self, config_obj):
        """
        初始化卫星NOMA系统
        
        参数:
            config_obj: 配置对象（来自config.py）
        """
        self.config = config_obj
        self.allocator = NOMAAllocator()
        
        # 创建信道模型
        self.loo_channel = LooChannel(
            alpha_db=config_obj.alpha_dB,
            psi_db=config_obj.psi_dB,
            mp_db=config_obj.MP_dB,
            seed=config_obj.random_seed
        )
    
    def compute_channel_gains_with_pathloss(self, elevation_deg):
        """
        计算包含路径损耗的完整信道增益

        论文公式(3)：Γ_l = (Gs_t * Gs_r) / (L_FS_l * Ns) * |h_s_l|^2

        **关键理解**：论文的SNR定义是"接收SNR"，而不是"发射SNR"！
        - 在速率公式中：R = B * log2(1 + SNR * Γ)
        - 这里的SNR是无量纲的接收信噪比
        - Γ已经归一化，所以SNR*Γ就是实际的SINR

        参数:
            elevation_deg (float): 卫星仰角 (度)

        返回:
            channel_gains (ndarray): shape (N,) 归一化信道增益
        """
        # 1. 小尺度衰落（Loo模型）
        fading_gains = self.loo_channel.generate_channel_gain(self.config.N)

        # 2. 大尺度路径损耗
        distance = PathLossModel.satellite_distance(elevation_deg)
        path_loss = PathLossModel.free_space_loss(distance, self.config.fs)

        # 3. 天线增益（线性值）
        G_tx = 10 ** (self.config.Gs_t_dB / 10)
        G_rx = 10 ** (self.config.Gs_r_dB / 10)

        # 4. 噪声功率
        Ns = self.config.get_noise_power()

        # 5. 归一化信道增益（论文公式3）
        # Γ = (Gtx * Grx / (L_FS * Ns)) * |h|^2
        channel_gains = (G_tx * G_rx / (path_loss * Ns)) * fading_gains

        return channel_gains

    def compute_achievable_rates(self, channel_gains, snr_linear, verbose=False):
        """
        计算 SAT-NOMA 可达速率

        流程：
        1. 用户配对（最优配对策略）
        2. 功率分配（最优分配因子）
        3. 速率计算（论文公式5, 6）

        参数:
            channel_gains (ndarray): shape (2K,) 信道增益
            snr_linear (float): 接收SNR（线性值，无量纲）
            verbose (bool): 是否输出详细信息

        返回:
            rates (ndarray): shape (2K,) 每个用户的速率 (bps)
            sum_rate (float): 总速率 (bps)
        """
        K = len(channel_gains) // 2
        bandwidth_per_pair = self.config.Bs / K  # 每对用户分配的带宽

        # 1. 用户配对
        pairs, paired_gains = self.allocator.optimal_user_pairing(channel_gains)

        # 2. 计算每对的速率
        rates = np.zeros(len(channel_gains))

        for k in range(K):
            weak_idx, strong_idx = pairs[k]
            gamma_weak, gamma_strong = paired_gains[k]

            # 2.1 功率分配
            # 注意：这里传入的是SNR而不是功率
            beta_strong, beta_weak = self.allocator.compute_power_factors(
                gamma_strong, gamma_weak, snr_linear, verbose=False
            )

            # 2.2 强用户速率 - 论文公式(5)
            # R_j = B/K * log2(1 + β_j * SNR * Γ_j)
            rate_strong = bandwidth_per_pair * np.log2(
                1 + beta_strong * snr_linear * gamma_strong
            )

            # 2.3 弱用户速率 - 论文公式(6)
            interference = beta_strong * snr_linear * gamma_weak
            signal = beta_weak * snr_linear * gamma_weak
            rate_weak = bandwidth_per_pair * np.log2(
                1 + signal / (interference + 1)
            )

            # 2.4 存储速率
            rates[weak_idx] = rate_weak
            rates[strong_idx] = rate_strong

        # 3. 计算总速率
        sum_rate = np.sum(rates)

        return rates, sum_rate
    
    def simulate_performance(self, snr_db_range, elevation_deg=10, 
                            n_realizations=100, verbose=True):
        """
        Monte Carlo 仿真系统性能
        
        过程：
        1. 遍历每个SNR点
        2. 对每个SNR，进行n_realizations次独立信道实现
        3. 统计平均性能
        
        参数:
            snr_db_range (ndarray): SNR范围 (dB)
            elevation_deg (float): 卫星仰角 (度)
            n_realizations (int): Monte Carlo 仿真次数
            verbose (bool): 是否显示进度
        
        返回:
            mean_sum_rates (ndarray): shape (len(snr_range),) 平均总速率
            mean_spectral_efficiency (ndarray): shape (len(snr_range),) 平均频谱效率
            std_sum_rates (ndarray): 总速率标准差（可选）
        """
        n_snr_points = len(snr_db_range)
        sum_rates_all = np.zeros((n_snr_points, n_realizations))
        
        if verbose:
            print(f"开始 Monte Carlo 仿真...")
            print(f"  SNR 点数: {n_snr_points}")
            print(f"  每点仿真次数: {n_realizations}")
            print(f"  总仿真次数: {n_snr_points * n_realizations}")
        
        # 使用 tqdm 进度条
        snr_iterator = tqdm(enumerate(snr_db_range), total=n_snr_points, 
                           desc="仿真进度", disable=not verbose)
        
        for i, snr_db in snr_iterator:
            # 将 SNR(dB) 转换为线性值
            snr_linear = 10 ** (snr_db / 10)

            for r in range(n_realizations):
                # 每次实现独立生成信道
                # 使用不同的随机种子确保独立性
                self.loo_channel.rng = np.random.default_rng(
                    self.config.random_seed + i * n_realizations + r
                )

                # 生成信道增益
                channel_gains = self.compute_channel_gains_with_pathloss(elevation_deg)

                # 计算速率（使用SNR线性值）
                _, sum_rate = self.compute_achievable_rates(channel_gains, snr_linear)
                sum_rates_all[i, r] = sum_rate
            
            # 更新进度条信息
            if verbose:
                current_mean = np.mean(sum_rates_all[i, :])
                snr_iterator.set_postfix({
                    'SNR': f'{snr_db}dB',
                    'Mean SE': f'{current_mean/self.config.Bs:.2f} bits/s/Hz'
                })
        
        # 统计结果
        mean_sum_rates = np.mean(sum_rates_all, axis=1)
        std_sum_rates = np.std(sum_rates_all, axis=1)
        mean_spectral_efficiency = mean_sum_rates / self.config.Bs  # bits/s/Hz
        
        if verbose:
            print(f"仿真完成")
        
        return mean_sum_rates, mean_spectral_efficiency, std_sum_rates


# ==================== 测试代码 ====================
def test_noma_transmission():
    """测试 NOMA 传输系统"""
    print("=" * 60)
    print("测试卫星 NOMA 传输系统")
    print("=" * 60)
    
    sat_noma = SatelliteNOMA(config)
    
    # 测试 1：信道增益计算
    print(f"\n【测试1：信道增益计算】")
    test_elevations = [10, 20, 40]
    for elev in test_elevations:
        gains = sat_noma.compute_channel_gains_with_pathloss(elev)
        print(f"  仰角 E={elev}°:")
        print(f"    均值: {np.mean(gains):.6e}")
        print(f"    中位数: {np.median(gains):.6e}")
        print(f"    最大/最小: {np.max(gains)/np.min(gains):.2f}x")
    
    # 测试 2：单次速率计算
    print(f"\n【测试2：单次速率计算】")
    test_snr = 20  # dB
    Ps_test = config.snr_to_power(test_snr)
    
    channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg=10)
    rates, sum_rate = sat_noma.compute_achievable_rates(channel_gains, Ps_test)
    
    print(f"  SNR: {test_snr} dB")
    print(f"  发射功率: {Ps_test:.6e} W")
    print(f"  总速率: {sum_rate/1e6:.2f} Mbps")
    print(f"  频谱效率: {sum_rate/config.Bs:.2f} bits/s/Hz")
    print(f"  用户速率统计:")
    print(f"    均值: {np.mean(rates)/1e6:.2f} Mbps")
    print(f"    最大: {np.max(rates)/1e6:.2f} Mbps")
    print(f"    最小: {np.min(rates)/1e6:.2f} Mbps")
    
    # 测试 3：小规模 Monte Carlo
    print(f"\n【测试3：小规模 Monte Carlo (快速验证)】")
    test_snr_range = np.array([0, 10, 20, 30])
    mean_rates, mean_se, std_rates = sat_noma.simulate_performance(
        snr_db_range=test_snr_range,
        elevation_deg=10,
        n_realizations=10,  # 仅10次快速测试
        verbose=True
    )
    
    print(f"\n结果摘要:")
    for i, snr in enumerate(test_snr_range):
        print(f"  SNR={snr:2d}dB: SE={mean_se[i]:.2f}±{std_rates[i]/config.Bs:.2f} bits/s/Hz")
    
    # 测试 4：验证单调性
    print(f"\n【测试4：验证性能单调性】")
    is_increasing = np.all(np.diff(mean_se) > 0)
    print(f"  频谱效率随SNR递增: {'✓ 是' if is_increasing else '✗ 否'}")
    
    print("\n" + "=" * 60)
    print("✓ NOMA 传输系统测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_noma_transmission()
