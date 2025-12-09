"""
SATCON完整系统 - 卫星+ABS协作NOMA

实现论文Section III的完整SATCON方案：
1. 卫星NOMA传输
2. ABS位置优化
3. ABS NOMA/OMA传输
4. 混合决策规则（4种情况）

论文参考：
- Section III: SATCON
- Section III.C: ABS hybrid NOMA/OMA transmission
"""
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.user_distribution import UserDistribution
from src.abs_placement import ABSPlacement
from src.a2g_channel import A2GChannel, S2AChannel
from src.noma_transmission import SatelliteNOMA
from src.power_allocation import NOMAAllocator
from src.channel_models import PathLossModel


class SATCONSystem:
    """
    SATCON完整系统：卫星-空中-地面混合NOMA方案
    """
    
    def __init__(self, config_obj, abs_bandwidth):
        """
        初始化SATCON系统
        
        参数:
            config_obj: 配置对象
            abs_bandwidth: ABS带宽 (Hz)
        """
        self.config = config_obj
        self.Bd = abs_bandwidth
        
        # 初始化子系统
        self.sat_noma = SatelliteNOMA(config_obj)
        self.a2g_channel = A2GChannel()
        self.s2a_channel = S2AChannel()
        self.abs_placement = ABSPlacement()
        self.allocator = NOMAAllocator()
        
        # 噪声功率
        self.Nd = config_obj.get_abs_noise_power(abs_bandwidth)
        self.Nsd = config_obj.get_s2a_noise_power()
    
    def compute_abs_noma_rates(self, user_positions, abs_position, 
                               channel_gains_a2g, total_power):
        """
        计算ABS NOMA传输的可达速率
        
        流程：
        1. 根据A2G信道增益配对用户
        2. 计算NOMA功率分配
        3. 计算各用户可达速率
        
        参数:
            user_positions: 用户位置
            abs_position: ABS位置
            channel_gains_a2g: A2G信道增益
            total_power: ABS总功率
        
        返回:
            rates_noma: NOMA速率
            pairs: 用户配对
        """
        K = len(channel_gains_a2g) // 2
        bandwidth_per_pair = self.Bd / K
        
        # 用户配对（基于A2G增益）
        pairs, paired_gains = self.allocator.optimal_user_pairing(channel_gains_a2g)
        
        rates_noma = np.zeros(len(channel_gains_a2g))
        
        for k in range(K):
            weak_idx, strong_idx = pairs[k]
            gamma_weak, gamma_strong = paired_gains[k]
            
            # 功率分配
            beta_strong, beta_weak = self.allocator.compute_power_factors(
                gamma_strong, gamma_weak, total_power
            )
            
            # 速率计算
            # 注意：channel_gain已经包含噪声归一化 (G/(L*Nd))
            # 强用户：SNR = beta_strong * P * gamma_strong
            rate_strong = bandwidth_per_pair * np.log2(
                1 + beta_strong * total_power * gamma_strong
            )

            # 弱用户：SINR = signal / (interference + 1)
            # +1 是因为 channel_gain 已归一化，分母中的 Nd/Nd = 1
            interference = beta_strong * total_power * gamma_weak
            signal = beta_weak * total_power * gamma_weak
            rate_weak = bandwidth_per_pair * np.log2(
                1 + signal / (interference + 1)
            )
            
            rates_noma[weak_idx] = rate_weak
            rates_noma[strong_idx] = rate_strong
        
        return rates_noma, pairs
    
    def compute_abs_oma_rates(self, channel_gains_a2g, total_power):
        """
        计算ABS OMA传输的可达速率
        
        OMA：每个用户独占时频资源
        
        参数:
            channel_gains_a2g: A2G信道增益
            total_power: ABS总功率
        
        返回:
            rates_oma: OMA速率
        """
        K = len(channel_gains_a2g) // 2
        bandwidth_per_user = self.Bd / K  # 每个用户分配的带宽
        
        # OMA：全功率分配给单个用户
        # 注意：channel_gain已包含噪声归一化
        rates_oma = bandwidth_per_user * np.log2(
            1 + total_power * channel_gains_a2g
        )
        
        return rates_oma
    
    def hybrid_decision(self, sat_rates, abs_noma_rates, abs_oma_rates, 
                       s2a_rates, pairs):
        """
        混合NOMA/OMA决策规则
        
        论文Section III.C的4种情况：
        
        定义：
        - R^s: 卫星直传速率
        - R^{dn}: ABS NOMA速率（受S2A限制）
        - R^{do}: ABS OMA速率（受S2A限制）
        
        决策规则：
        1. R^s_i < R^{dn}_i AND R^s_j < R^{dn}_j
           → 使用NOMA，两用户都受益
        
        2. R^s_i < R^{do}_i AND R^s_j >= R^{dn}_j
           → 使用OMA只传给弱用户i
        
        3. R^s_i >= R^{dn}_i AND R^s_j < R^{do}_j
           → 使用OMA只传给强用户j
        
        4. Otherwise
           → 不传输，用户直接用卫星
        
        参数:
            sat_rates: 卫星速率
            abs_noma_rates: ABS NOMA速率
            abs_oma_rates: ABS OMA速率
            s2a_rates: S2A链路速率（限制ABS转发速率）
            pairs: 用户配对
        
        返回:
            final_rates: 最终速率
            modes: 每对的传输模式
        """
        final_rates = sat_rates.copy()
        modes = []
        
        # 考虑S2A链路限制
        abs_noma_limited = np.minimum(abs_noma_rates, s2a_rates)
        abs_oma_limited = np.minimum(abs_oma_rates, s2a_rates)
        
        for k, (weak_idx, strong_idx) in enumerate(pairs):
            R_s_i = sat_rates[weak_idx]
            R_s_j = sat_rates[strong_idx]
            R_dn_i = abs_noma_limited[weak_idx]
            R_dn_j = abs_noma_limited[strong_idx]
            R_do_i = abs_oma_limited[weak_idx]
            R_do_j = abs_oma_limited[strong_idx]
            
            # 情况1：NOMA双用户
            if R_s_i < R_dn_i and R_s_j < R_dn_j:
                final_rates[weak_idx] = R_dn_i
                final_rates[strong_idx] = R_dn_j
                modes.append('noma')
            
            # 情况2：OMA只给弱用户
            elif R_s_i < R_do_i and R_s_j >= R_dn_j:
                final_rates[weak_idx] = R_do_i
                final_rates[strong_idx] = R_s_j
                modes.append('oma_weak')
            
            # 情况3：OMA只给强用户  
            elif R_s_i >= R_dn_i and R_s_j < R_do_j:
                final_rates[weak_idx] = R_s_i
                final_rates[strong_idx] = R_do_j
                modes.append('oma_strong')
            
            # 情况4：不传输
            else:
                # 保持卫星速率
                modes.append('none')
        
        return final_rates, modes
    
    def simulate_single_realization(self, snr_db, elevation_deg, seed):
        """
        单次仿真实现
        
        完整流程：
        1. 生成用户分布
        2. 优化ABS位置
        3. 计算卫星NOMA速率
        4. 计算ABS NOMA/OMA速率
        5. 执行混合决策
        6. 返回最终速率
        
        参数:
            snr_db: 卫星SNR (dB)
            elevation_deg: 卫星仰角 (度)
            seed: 随机种子
        
        返回:
            sum_rate: 总速率
            mode_stats: 决策模式统计
        """
        # 1. 生成用户分布
        dist = UserDistribution(self.config.N, self.config.coverage_radius, seed=seed)
        user_positions = dist.generate_uniform_circle()
        
        # 2. 优化ABS位置
        abs_position, _ = self.abs_placement.optimize_position_complete(
            user_positions, self.a2g_channel
        )
        
        # 3. 卫星NOMA速率
        snr_linear = 10 ** (snr_db / 10)
        sat_channel_gains = self.sat_noma.compute_channel_gains_with_pathloss(elevation_deg)
        sat_rates, _ = self.sat_noma.compute_achievable_rates(sat_channel_gains, snr_linear)
        
        # 4. A2G信道增益
        h_abs = abs_position[2]
        distances_2d = dist.compute_distances_from_point(user_positions, abs_position)
        
        fading_a2g = self.a2g_channel.generate_fading(self.config.N, seed=seed+1)
        channel_gains_a2g = np.array([
            self.a2g_channel.compute_channel_gain(
                h_abs, r, fading, 
                self.config.Gd_t_dB, self.config.Gd_r_dB, self.Nd
            )
            for r, fading in zip(distances_2d, fading_a2g)
        ])
        
        # 5. ABS NOMA/OMA速率
        abs_noma_rates, pairs = self.compute_abs_noma_rates(
            user_positions, abs_position, channel_gains_a2g, self.config.Pd
        )
        abs_oma_rates = self.compute_abs_oma_rates(channel_gains_a2g, self.config.Pd)
        
        # 6. S2A链路速率（简化：假设足够大，不限制）
        # 实际应计算S2A信道增益和可达速率
        s2a_rates = sat_rates * 10  # 简化：S2A容量远大于用户速率
        
        # 7. 混合决策
        final_rates, modes = self.hybrid_decision(
            sat_rates, abs_noma_rates, abs_oma_rates, s2a_rates, pairs
        )
        
        sum_rate = np.sum(final_rates)
        
        # 统计模式
        mode_stats = {
            'noma': modes.count('noma'),
            'oma_weak': modes.count('oma_weak'),
            'oma_strong': modes.count('oma_strong'),
            'none': modes.count('none')
        }
        
        return sum_rate, mode_stats
    
    def simulate_performance(self, snr_db_range, elevation_deg=10, 
                            n_realizations=100, verbose=True):
        """
        Monte Carlo性能仿真
        
        参数:
            snr_db_range: SNR范围
            elevation_deg: 卫星仰角
            n_realizations: 仿真次数
            verbose: 是否显示进度
        
        返回:
            mean_sum_rates, mean_se, std_sum_rates, mode_statistics
        """
        n_snr_points = len(snr_db_range)
        sum_rates_all = np.zeros((n_snr_points, n_realizations))
        mode_stats_all = {
            'noma': np.zeros((n_snr_points, n_realizations)),
            'oma_weak': np.zeros((n_snr_points, n_realizations)),
            'oma_strong': np.zeros((n_snr_points, n_realizations)),
            'none': np.zeros((n_snr_points, n_realizations))
        }
        
        if verbose:
            print(f"SATCON仿真 (Bd={self.Bd/1e6:.1f}MHz)...")
        
        snr_iterator = tqdm(enumerate(snr_db_range), total=n_snr_points,
                           desc=f"Bd={self.Bd/1e6:.1f}MHz", disable=not verbose)
        
        for i, snr_db in snr_iterator:
            for r in range(n_realizations):
                seed = self.config.random_seed + i * n_realizations + r
                
                sum_rate, mode_stats = self.simulate_single_realization(
                    snr_db, elevation_deg, seed
                )
                
                sum_rates_all[i, r] = sum_rate
                for mode in mode_stats_all.keys():
                    mode_stats_all[mode][i, r] = mode_stats[mode]
            
            if verbose:
                current_se = np.mean(sum_rates_all[i, :]) / self.Bd
                snr_iterator.set_postfix({'SE': f'{current_se:.2f} bits/s/Hz'})
        
        # 统计结果
        mean_sum_rates = np.mean(sum_rates_all, axis=1)
        std_sum_rates = np.std(sum_rates_all, axis=1)
        mean_se = mean_sum_rates / self.Bd
        
        # 模式统计
        mean_mode_stats = {mode: np.mean(counts, axis=1) 
                          for mode, counts in mode_stats_all.items()}
        
        return mean_sum_rates, mean_se, std_sum_rates, mean_mode_stats


# ==================== 测试代码 ====================
def test_satcon_system():
    """测试SATCON系统"""
    print("=" * 60)
    print("测试SATCON完整系统")
    print("=" * 60)
    
    # 创建SATCON系统
    satcon = SATCONSystem(config, abs_bandwidth=1.2e6)
    
    print(f"\n系统配置:")
    print(f"  用户数: {config.N}")
    print(f"  ABS带宽: {satcon.Bd/1e6:.1f} MHz")
    print(f"  ABS功率: {config.Pd_dBm} dBm")
    
    # 快速测试
    print(f"\n快速测试 (3个SNR点, 5次实现)...")
    test_snr = np.array([10, 20, 30])
    
    mean_rates, mean_se, std_rates, mode_stats = satcon.simulate_performance(
        snr_db_range=test_snr,
        elevation_deg=10,
        n_realizations=5,
        verbose=True
    )
    
    print(f"\n结果摘要:")
    for i, snr in enumerate(test_snr):
        print(f"  SNR={snr}dB: SE={mean_se[i]:.2f} bits/s/Hz")
        print(f"    模式: NOMA={mode_stats['noma'][i]:.1f}, "
              f"OMA_W={mode_stats['oma_weak'][i]:.1f}, "
              f"OMA_S={mode_stats['oma_strong'][i]:.1f}, "
              f"NONE={mode_stats['none'][i]:.1f}")
    
    print("\n" + "=" * 60)
    print("✓ SATCON系统测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_satcon_system()
