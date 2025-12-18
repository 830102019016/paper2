"""
SATCON完整系统 - 卫星+ABS协作NOMA

实现论文Section III的完整SATCON方案：
1. 卫星NOMA传输
2. ABS位置优化
3. ABS NOMA/OMA传输
4. 混合决策规则（4种情况）

✅ 【已完成的关键修正】（2024-12）：
1. 配对机制：只使用卫星配对（sat_pairs），ABS不重新配对
2. OMA带宽：整个 Bd/K 给单个用户（不是 Bd/2K）
3. S2A约束：考虑 Decode-and-Forward 瓶颈（R = min(R_a2g, R_s2a)）
4. 决策对照：端到端速率比较（卫星直达 vs ABS转发）

论文参考：
- Section III: SATCON
- Section III.C: ABS hybrid NOMA/OMA transmission

相关文档：
- docs/Baseline修正计划_v2_正确版.md - 详细修正说明
- docs/satcon_决策逻辑总结.md - SATCON决策逻辑
- docs/baseline_assumptions.md - 模块假设与合规性
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

    【关键修正】本实现已按照SATCON原论文正确逻辑修正：
    - 唯一配对：只有卫星NOMA配对（sat_pairs），ABS基于此配对做决策
    - OMA带宽：Bd/K（整个pair带宽）给单个用户
    - S2A约束：ABS转发速率受 S2A 解码速率限制（DF瓶颈）
    - 混合决策：端到端速率比较（R_s vs R_dn/R_do）

    依赖模块（已同步修正）：
    - user_distribution.py: 用户分布（均匀圆形）
    - abs_placement.py: ABS位置优化
    - channel_models.py: Loo信道模型（已明确简化假设）
    - a2g_channel.py: A2G/S2A信道（S2A无小尺度衰落）
    - noma_transmission.py: 卫星NOMA（已统一SNR定义）
    - power_allocation.py: 功率分配（含保护性机制）
    """
    
    def __init__(self, config_obj, abs_bandwidth,
                 mode_selector='heuristic', s2a_allocator='uniform',
                 position_optimizer=None):
        """
        初始化SATCON系统

        【推荐配置】（论文最终方案，一阶段+二阶段优化）：
            mode_selector='greedy'           # 一阶段：贪心模式选择
            s2a_allocator='kkt'              # 一阶段：KKT资源分配
            position_optimizer='L-BFGS-B'    # 二阶段：纯L-BFGS-B位置优化

        参数:
            config_obj: 配置对象
            abs_bandwidth: ABS带宽 (Hz)
            mode_selector: 模式选择器类型
                - 'heuristic': 启发式规则（baseline）
                - 'exhaustive': 穷举搜索（全局最优）
                - 'greedy': 贪心算法（快速近似）【推荐】
            s2a_allocator: S2A资源分配器类型
                - 'uniform': 均分分配（baseline）
                - 'kkt': KKT最优分配【推荐】
                - 'waterfilling': Water-filling算法
            position_optimizer: ABS位置优化器类型（新增）
                - None: 使用传统k-means/k-medoids（baseline）
                - 'L-BFGS-B': 纯梯度优化（从固定起点）【推荐】
        """
        self.config = config_obj
        self.Bd = abs_bandwidth

        # 初始化子系统
        self.sat_noma = SatelliteNOMA(config_obj)
        self.a2g_channel = A2GChannel()
        self.s2a_channel = S2AChannel()
        self.abs_placement = ABSPlacement()
        self.allocator = NOMAAllocator()

        # 【新增】初始化优化器
        from src.optimizers.mode_selector import (
            HeuristicSelector, ExhaustiveSelector, GreedySelector
        )
        from src.optimizers.resource_allocator import (
            UniformAllocator, KKTAllocator, WaterFillingAllocator
        )

        # 模式选择器
        if mode_selector == 'heuristic':
            self.mode_selector = HeuristicSelector()
        elif mode_selector == 'exhaustive':
            self.mode_selector = ExhaustiveSelector()
        elif mode_selector == 'greedy':
            self.mode_selector = GreedySelector()
        else:
            raise ValueError(f"Unknown mode_selector: {mode_selector}")

        # S2A分配器
        if s2a_allocator == 'uniform':
            self.s2a_allocator = UniformAllocator(config_obj)
        elif s2a_allocator == 'kkt':
            self.s2a_allocator = KKTAllocator(config_obj)
        elif s2a_allocator == 'waterfilling':
            self.s2a_allocator = WaterFillingAllocator(config_obj)
        else:
            raise ValueError(f"Unknown s2a_allocator: {s2a_allocator}")

        # 噪声功率
        self.Nd = config_obj.get_abs_noise_power(abs_bandwidth)
        # 【关键修正】S2A噪声应基于卫星带宽Bs（论文Table I: Nsd = κTsdBs）
        self.Nsd = config_obj.get_s2a_noise_power(config_obj.Bs)

        # 【新增】位置优化器（二阶段优化）
        self.position_optimizer_type = position_optimizer
        if position_optimizer is not None:
            from src.optimizers.position_optimizer import ContinuousPositionOptimizer
            self.position_optimizer = ContinuousPositionOptimizer(
                config_obj, method=position_optimizer
            )
        else:
            self.position_optimizer = None
    
    def compute_abs_noma_rates(self, sat_pairs, channel_gains_a2g,
                               s2a_rates, total_power):
        """
        计算ABS NOMA传输的可达速率（Decode-and-Forward）

        【关键修正】：
        1. 使用卫星配对（sat_pairs），不重新配对
        2. 考虑S2A解码约束（DF瓶颈）

        参数:
            sat_pairs: 卫星配对 [(i,j), ...]
            channel_gains_a2g: A2G信道增益 [N]
            s2a_rates: S2A链路速率 [N]（卫星→ABS解码速率）
            total_power: ABS功率 Pd

        返回:
            rates_noma: NOMA转发速率 [N]（已考虑S2A约束）
        """
        K = len(sat_pairs)
        bandwidth_per_pair = self.Bd / K  # 每对一个时隙

        rates_noma = np.zeros(2*K)

        for k in range(K):
            # 【修正点1】使用卫星配对
            weak_idx, strong_idx = sat_pairs[k]
            gamma_weak = channel_gains_a2g[weak_idx]
            gamma_strong = channel_gains_a2g[strong_idx]

            # 确保弱-强顺序（基于A2G信道）
            if gamma_weak > gamma_strong:
                weak_idx, strong_idx = strong_idx, weak_idx
                gamma_weak, gamma_strong = gamma_strong, gamma_weak

            # ABS重新编码NOMA（用A2G信道的功率分配）
            beta_strong, beta_weak = self.allocator.compute_power_factors(
                gamma_strong, gamma_weak, total_power
            )

            # 计算A2G速率（ABS → 用户）
            rate_a2g_strong = bandwidth_per_pair * np.log2(
                1 + beta_strong * total_power * gamma_strong
            )
            rate_a2g_weak = bandwidth_per_pair * np.log2(
                1 + beta_weak * total_power * gamma_weak /
                (beta_strong * total_power * gamma_weak + 1)
            )

            # 【修正点2】考虑S2A瓶颈（DF约束）
            # ABS必须先从卫星解码，再转发
            rate_s2a_weak = s2a_rates[weak_idx]
            rate_s2a_strong = s2a_rates[strong_idx]

            # 取瓶颈（min）
            rates_noma[weak_idx] = min(rate_a2g_weak, rate_s2a_weak)
            rates_noma[strong_idx] = min(rate_a2g_strong, rate_s2a_strong)

        return rates_noma
    
    def compute_abs_oma_rates(self, sat_pairs, channel_gains_a2g,
                              s2a_rates, total_power):
        """
        计算ABS OMA传输的可达速率

        【关键修正】：
        1. 使用sat_pairs（知道配对关系）
        2. 整个pair带宽给单个用户（不是Bd/2K）
        3. 考虑S2A约束

        参数:
            sat_pairs: 卫星配对
            channel_gains_a2g: A2G信道增益 [N]
            s2a_rates: S2A链路速率 [N]
            total_power: ABS功率 Pd

        返回:
            rates_oma: OMA转发速率 [N]（已考虑S2A约束）
        """
        K = len(sat_pairs)

        # 【关键修正1】每对占用 Bd/K 带宽
        # OMA时，整个 Bd/K 都给一个用户
        bandwidth_per_pair = self.Bd / K  # 不是 Bd/(2K) ！

        # 计算A2G速率（全功率 + 全带宽）
        rates_a2g = bandwidth_per_pair * np.log2(
            1 + total_power * channel_gains_a2g
        )

        # 【关键修正2】考虑S2A约束
        # 即使是OMA，ABS也需要从卫星解码
        rates_oma = np.minimum(rates_a2g, s2a_rates)

        return rates_oma
    
    def hybrid_decision(self, sat_rates, abs_noma_rates, abs_oma_rates,
                       sat_pairs):
        """
        混合NOMA/OMA决策规则（逐对，4条规则）

        【修正点3】对照关系正确：
        - 对照：R_s（卫星直达，端到端）
        - 候选：R_dn, R_do（ABS转发，已含S2A约束，端到端）

        参数:
            sat_rates: 卫星直达速率 [N]
            abs_noma_rates: ABS NOMA速率 [N]（已含S2A约束）
            abs_oma_rates: ABS OMA速率 [N]（已含S2A约束）
            sat_pairs: 配对 [(i,j), ...]

        返回:
            final_rates: 最终速率 [N]
            modes: 每对的传输模式
        """
        final_rates = sat_rates.copy()
        modes = []

        for k, (weak_idx, strong_idx) in enumerate(sat_pairs):
            R_s_i = sat_rates[weak_idx]
            R_s_j = sat_rates[strong_idx]
            R_dn_i = abs_noma_rates[weak_idx]
            R_dn_j = abs_noma_rates[strong_idx]
            R_do_i = abs_oma_rates[weak_idx]
            R_do_j = abs_oma_rates[strong_idx]

            # 【正确对照】端到端速率比较

            # 规则1：NOMA双用户
            if R_s_i < R_dn_i and R_s_j < R_dn_j:
                final_rates[weak_idx] = R_dn_i
                final_rates[strong_idx] = R_dn_j
                modes.append('noma')

            # 规则2：OMA只给弱用户
            elif R_s_i < R_do_i and R_s_j >= R_dn_j:
                final_rates[weak_idx] = R_do_i
                final_rates[strong_idx] = R_s_j
                modes.append('oma_weak')

            # 规则3：OMA只给强用户
            elif R_s_i >= R_dn_i and R_s_j < R_do_j:
                final_rates[weak_idx] = R_s_i
                final_rates[strong_idx] = R_do_j
                modes.append('oma_strong')

            # 规则4：不转发
            else:
                modes.append('sat')

        return final_rates, modes

    def compute_s2a_rates(self, sat_pairs, abs_position, elevation_deg,
                         sat_channel_gains, snr_linear, sat_power_factors):
        """
        计算S2A链路速率（卫星→ABS）- 修正版

        【关键修正】（2024-12）：
        - S2A使用卫星带宽Bs（不是Bd）
        - 计算公式：R_s2a = (Bs/K) * log2(1 + SNR * |h_s2a|^2)
        - 参考论文Eq.(7)：S2A只考虑路径损耗，无小尺度衰落
        - 【最新修正】直接复用卫星侧的功率分配因子β

        物理意义：
        - ABS需要从卫星接收并解码NOMA信号
        - 每对占用 Bs/K 带宽（卫星下行带宽）
        - S2A信道增益计算：自由空间路径损耗（仰角相关）

        参数:
            sat_pairs: 卫星配对
            abs_position: ABS位置 [x, y, h]
            elevation_deg: 卫星仰角
            sat_channel_gains: 卫星信道增益（各用户）- 未使用但保留接口
            snr_linear: 卫星SNR（线性）
            sat_power_factors: 卫星侧功率分配因子 {'beta_strong': [K], 'beta_weak': [K]}

        返回:
            s2a_rates: S2A解码速率 [N]（基于Bs带宽）
        """
        K = len(sat_pairs)
        Bs_per_pair = self.config.Bs / K  # 关键：每对的卫星带宽（S2A使用Bs，不是Bd）

        # 计算卫星到ABS的距离（基于仰角）
        # 假设ABS在地面中心，高度为abs_position[2]
        h_abs = abs_position[2]
        satellite_altitude = self.config.satellite_altitude

        # 根据仰角计算斜距
        elevation_rad = np.deg2rad(elevation_deg)
        # d = (H_sat - h_abs) / sin(elevation)
        d_s2a = (satellite_altitude - h_abs) / np.sin(elevation_rad)

        # 计算S2A信道增益（ABS到卫星）
        # 根据论文Eq.(7)：只考虑路径损耗，无小尺度衰落
        h_s2a = self.s2a_channel.compute_channel_gain(
            distance=d_s2a,
            fading_gain=1.0,  # 论文假设：S2A无小尺度衰落
            G_tx_dB=self.config.Gs_t_dB,  # 卫星发射增益 = 24 dBi
            G_rx_dB=self.config.Gsd_r_dB,  # 【修正】ABS接收卫星增益 = 9 dBi（论文Table I）
            noise_power=self.Nsd  # S2A噪声功率（已修正为基于Bs）
        )

        # 【关键修正4】S2A NOMA SIC解码 - 直接复用卫星侧β（2024-12）
        # 卫星发送NOMA复合信号，ABS需要SIC解码
        # 使用卫星计算的功率分配因子（保证一致性）
        s2a_rates = np.zeros(2*K)

        for k in range(K):
            weak_idx, strong_idx = sat_pairs[k]

            # 【修正】直接使用卫星侧的β（不再重新计算）
            beta_strong = sat_power_factors['beta_strong'][k]
            beta_weak = sat_power_factors['beta_weak'][k]

            # S2A NOMA解码速率（论文公式 Eq.8, Eq.9）
            # 强用户：R_sd_j = (Bs/K) * log2(1 + β_j * Ps * Λ^sd)
            rate_s2a_strong = Bs_per_pair * np.log2(
                1 + beta_strong * snr_linear * h_s2a
            )

            # 弱用户：R_sd_i = (Bs/K) * log2(1 + (1-β_j) * Ps * Λ^sd / (β_j * Ps * Λ^sd + 1))
            rate_s2a_weak = Bs_per_pair * np.log2(
                1 + beta_weak * snr_linear * h_s2a /
                (beta_strong * snr_linear * h_s2a + 1)
            )

            s2a_rates[strong_idx] = rate_s2a_strong
            s2a_rates[weak_idx] = rate_s2a_weak

        return s2a_rates

    def simulate_single_realization(self, snr_db, elevation_deg, seed):
        """
        单次仿真（支持可插拔优化器）

        【流程修正】：
        1. 计算 A2G 速率（不含 S2A 约束）
        2. 使用优化器选择模式和分配 S2A 资源
        3. 计算最终速率（含 S2A 约束）

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

        # 2. 计算卫星信道增益（需要提前计算，用于位置优化）
        sat_channel_gains = self.sat_noma.compute_channel_gains_with_pathloss(elevation_deg)

        # 3. 【唯一的配对】卫星NOMA配对
        sat_pairs, _ = self.allocator.optimal_user_pairing(sat_channel_gains)
        K = len(sat_pairs)

        # 4. 计算卫星NOMA速率（基于sat_pairs）并获取功率分配因子
        snr_linear = 10 ** (snr_db / 10)
        sat_rates, _, sat_power_factors = self.sat_noma.compute_achievable_rates(sat_channel_gains, snr_linear)

        # 5. 【关键修改】优化ABS位置
        if self.position_optimizer is not None:
            # 使用纯L-BFGS-B优化（从固定起点开始）
            abs_position, _, _ = self.position_optimizer.optimize_position_lbfgsb_pure(
                user_positions,
                mode_selector=self.mode_selector,
                s2a_allocator=self.s2a_allocator,
                snr_linear=snr_linear,
                elevation_deg=elevation_deg,
                sat_channel_gains=sat_channel_gains,
                sat_rates=sat_rates,
                sat_pairs=sat_pairs,
                sat_power_factors=sat_power_factors,
                Bd=self.Bd,
                Pd=self.config.Pd,
                Nd=self.Nd,
                Nsd=self.Nsd,
                max_iter=20,
                verbose=False
            )
        else:
            # 使用传统几何聚类方法（k-means，作为baseline）
            abs_position, _ = self.abs_placement.optimize_position_complete(
                user_positions, self.a2g_channel
            )

        # 6. 计算A2G信道增益
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

        # 7. 计算 S2A 信道增益（用于资源分配优化）
        satellite_altitude = self.config.satellite_altitude
        elevation_rad = np.deg2rad(elevation_deg)
        d_s2a = (satellite_altitude - h_abs) / np.sin(elevation_rad)
        h_s2a = self.s2a_channel.compute_channel_gain(
            distance=d_s2a,
            fading_gain=1.0,
            G_tx_dB=self.config.Gs_t_dB,
            G_rx_dB=self.config.Gsd_r_dB,
            noise_power=self.Nsd
        )

        # 8. 计算 A2G 速率（不含 S2A 约束）
        # 这些是 ABS → 用户的速率（假设 S2A 无瓶颈）
        bandwidth_per_pair = self.Bd / K

        # NOMA A2G 速率（不含 S2A 约束）
        a2g_rates_noma = np.zeros(2*K)
        for k in range(K):
            weak_idx, strong_idx = sat_pairs[k]
            gamma_weak = channel_gains_a2g[weak_idx]
            gamma_strong = channel_gains_a2g[strong_idx]

            if gamma_weak > gamma_strong:
                weak_idx, strong_idx = strong_idx, weak_idx
                gamma_weak, gamma_strong = gamma_strong, gamma_weak

            beta_strong, beta_weak = self.allocator.compute_power_factors(
                gamma_strong, gamma_weak, self.config.Pd
            )

            a2g_rates_noma[strong_idx] = bandwidth_per_pair * np.log2(
                1 + beta_strong * self.config.Pd * gamma_strong
            )
            a2g_rates_noma[weak_idx] = bandwidth_per_pair * np.log2(
                1 + beta_weak * self.config.Pd * gamma_weak /
                (beta_strong * self.config.Pd * gamma_weak + 1)
            )

        # OMA A2G 速率（不含 S2A 约束）
        a2g_rates_oma = bandwidth_per_pair * np.log2(
            1 + self.config.Pd * channel_gains_a2g
        )

        # 9. 【新增】使用 S2A 分配器优化带宽分配
        # 先用启发式/贪心选择模式（临时决策）
        _, modes_temp = self.mode_selector.select_modes(
            sat_rates, a2g_rates_noma, a2g_rates_oma, sat_pairs
        )

        # 根据临时模式分配 S2A 带宽
        b_allocated = self.s2a_allocator.allocate_bandwidth(
            sat_pairs, modes_temp, a2g_rates_noma, a2g_rates_oma,
            snr_linear, h_s2a, sat_power_factors
        )

        # 计算 S2A 速率（基于优化的带宽分配）
        s2a_rates = self.s2a_allocator.compute_s2a_rates(
            sat_pairs, b_allocated, snr_linear, h_s2a, sat_power_factors
        )

        # 10. 计算 ABS 最终速率（含 S2A 约束）
        abs_noma_rates = np.minimum(a2g_rates_noma, s2a_rates)
        abs_oma_rates = np.minimum(a2g_rates_oma, s2a_rates)

        # 11. 【新增】使用模式选择器做最终决策
        final_rates, modes = self.mode_selector.select_modes(
            sat_rates, abs_noma_rates, abs_oma_rates, sat_pairs
        )

        # 12. 统计
        mode_stats = {
            'noma': modes.count('noma'),
            'oma_weak': modes.count('oma_weak'),
            'oma_strong': modes.count('oma_strong'),
            'sat': modes.count('sat')
        }

        return np.sum(final_rates), mode_stats
    
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
            'sat': np.zeros((n_snr_points, n_realizations))
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
                current_se = np.mean(sum_rates_all[i, :]) / self.config.Bs
                snr_iterator.set_postfix({'SE': f'{current_se:.2f} bits/s/Hz'})
        
        # 统计结果
        mean_sum_rates = np.mean(sum_rates_all, axis=1)
        std_sum_rates = np.std(sum_rates_all, axis=1)
        mean_se = mean_sum_rates / self.config.Bs
        
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
              f"SAT={mode_stats['sat'][i]:.1f}")
    
    print("\n" + "=" * 60)
    print("SATCON系统测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_satcon_system()
