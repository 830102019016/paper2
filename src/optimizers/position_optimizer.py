"""
ABS位置连续优化器

本模块实现基于通信性能的ABS位置优化算法，替代传统的几何聚类方法。

实现方法：
1. L-BFGS-B: 梯度优化（数值梯度 + 边界约束）
2. Hybrid: k-means初始化 → L-BFGS-B精细优化（推荐）

优化目标：
    max_{p_d = [x, y, h]} Σ_{u∈U} R_u(p_d)

其中 R_u(p_d) 是用户u在ABS位置p_d下的最终速率（经过模式选择和资源分配）

技术要点：
- 热启动：从k-means/k-medoids结果开始
- 数值梯度：有限差分法（目标函数不可微）
- 边界约束：高度、覆盖范围
- 缓存优化：固定小尺度衰落（减少随机性）

论文创新点：
- 直接优化通信性能（非几何距离）
- 联合优化位置、模式选择、资源分配
- 相比k-means，预期增益：+2-5%
"""

import numpy as np
from scipy.optimize import minimize
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from config import config
from src.user_distribution import UserDistribution
from src.abs_placement import ABSPlacement
from src.a2g_channel import A2GChannel, S2AChannel
from src.noma_transmission import SatelliteNOMA
from src.power_allocation import NOMAAllocator


class ContinuousPositionOptimizer:
    """
    ABS位置连续优化器（基于通信性能）

    【推荐配置】（论文最终方案）：
        method='L-BFGS-B' (梯度优化)
        mode_selector=GreedySelector
        s2a_allocator=KKTAllocator

    【其他配置】（可选，用于对比）：
        method='PSO' (粒子群优化，未实现)
        method='GA' (遗传算法，未实现)
    """

    def __init__(self, config_obj, method='L-BFGS-B'):
        """
        初始化位置优化器

        参数:
            config_obj: 配置对象（来自config.py）
            method: 优化方法
                - 'L-BFGS-B': 拟牛顿法（推荐）
                - 'PSO': 粒子群优化（未实现）
                - 'GA': 遗传算法（未实现）
        """
        self.config = config_obj
        self.method = method

        # 初始化子模块
        self.a2g_channel = A2GChannel()
        self.s2a_channel = S2AChannel()
        self.sat_noma = SatelliteNOMA(config_obj)
        self.allocator = NOMAAllocator()

        # 优化统计
        self.n_evaluations = 0
        self.optimization_history = []

    def _compute_a2g_channel_gains(self, abs_position, user_positions,
                                   fading_gains, Bd, Nd):
        """
        计算A2G信道增益（给定ABS位置）

        参数:
            abs_position: ABS位置 [x, y, h]
            user_positions: 用户位置 [N, 3]
            fading_gains: 小尺度衰落 [N]（固定，避免随机性）
            Bd: ABS带宽 (Hz)
            Nd: 噪声功率 (W)

        返回:
            channel_gains_a2g: A2G信道增益 [N]
        """
        x_abs, y_abs, h_abs = abs_position

        # 计算2D距离
        distances_2d = np.linalg.norm(
            user_positions[:, :2] - np.array([x_abs, y_abs]), axis=1
        )

        # 计算A2G信道增益
        channel_gains_a2g = np.array([
            self.a2g_channel.compute_channel_gain(
                h_abs, r, fading,
                self.config.Gd_t_dB, self.config.Gd_r_dB, Nd
            )
            for r, fading in zip(distances_2d, fading_gains)
        ])

        return channel_gains_a2g

    def _compute_s2a_channel_gain(self, abs_position, elevation_deg, Nsd):
        """
        计算S2A信道增益（给定ABS位置）

        参数:
            abs_position: ABS位置 [x, y, h]
            elevation_deg: 卫星仰角 (度)
            Nsd: S2A噪声功率 (W)

        返回:
            h_s2a: S2A信道增益（标量）
        """
        h_abs = abs_position[2]
        satellite_altitude = self.config.satellite_altitude

        # 根据仰角计算斜距
        elevation_rad = np.deg2rad(elevation_deg)
        d_s2a = (satellite_altitude - h_abs) / np.sin(elevation_rad)

        # 计算S2A信道增益（无小尺度衰落）
        h_s2a = self.s2a_channel.compute_channel_gain(
            distance=d_s2a,
            fading_gain=1.0,  # 论文假设：S2A无小尺度衰落
            G_tx_dB=self.config.Gs_t_dB,
            G_rx_dB=self.config.Gsd_r_dB,
            noise_power=Nsd
        )

        return h_s2a

    def _objective_function(self, abs_position, **kwargs):
        """
        目标函数：计算给定ABS位置下的负系统总速率

        流程（复用satcon_system.py的逻辑）：
        1. 计算A2G信道增益（基于abs_position）
        2. 计算S2A信道增益（基于abs_position）
        3. 计算A2G速率（NOMA/OMA，不含S2A约束）
        4. 使用s2a_allocator分配S2A带宽
        5. 计算最终速率（含S2A约束）
        6. 使用mode_selector选择模式
        7. 返回负总速率（scipy最小化）

        参数:
            abs_position: ABS位置 [x, y, h]（优化变量）
            **kwargs: 其他参数（固定）
                - user_positions: 用户位置 [N, 3]
                - fading_a2g: A2G小尺度衰落 [N]（固定）
                - sat_rates: 卫星直达速率 [N]
                - sat_pairs: 卫星配对 [(i,j), ...]
                - sat_power_factors: 卫星功率分配因子
                - snr_linear: 卫星SNR（线性）
                - elevation_deg: 卫星仰角
                - mode_selector: 模式选择器
                - s2a_allocator: S2A资源分配器
                - Bd: ABS带宽
                - Pd: ABS功率
                - Nd: A2G噪声功率
                - Nsd: S2A噪声功率

        返回:
            neg_sum_rate: 负总速率（用于最小化）
        """
        # 解包参数
        user_positions = kwargs['user_positions']
        fading_a2g = kwargs['fading_a2g']
        sat_rates = kwargs['sat_rates']
        sat_pairs = kwargs['sat_pairs']
        sat_power_factors = kwargs['sat_power_factors']
        snr_linear = kwargs['snr_linear']
        elevation_deg = kwargs['elevation_deg']
        mode_selector = kwargs['mode_selector']
        s2a_allocator = kwargs['s2a_allocator']
        Bd = kwargs['Bd']
        Pd = kwargs['Pd']
        Nd = kwargs['Nd']
        Nsd = kwargs['Nsd']

        # 统计评估次数
        self.n_evaluations += 1

        try:
            # 1. 计算A2G信道增益
            channel_gains_a2g = self._compute_a2g_channel_gains(
                abs_position, user_positions, fading_a2g, Bd, Nd
            )

            # 2. 计算S2A信道增益
            h_s2a = self._compute_s2a_channel_gain(
                abs_position, elevation_deg, Nsd
            )

            # 3. 计算A2G速率（不含S2A约束）
            K = len(sat_pairs)
            bandwidth_per_pair = Bd / K

            # NOMA A2G速率
            a2g_rates_noma = np.zeros(2*K)
            for k in range(K):
                weak_idx, strong_idx = sat_pairs[k]
                gamma_weak = channel_gains_a2g[weak_idx]
                gamma_strong = channel_gains_a2g[strong_idx]

                if gamma_weak > gamma_strong:
                    weak_idx, strong_idx = strong_idx, weak_idx
                    gamma_weak, gamma_strong = gamma_strong, gamma_weak

                beta_strong, beta_weak = self.allocator.compute_power_factors(
                    gamma_strong, gamma_weak, Pd
                )

                a2g_rates_noma[strong_idx] = bandwidth_per_pair * np.log2(
                    1 + beta_strong * Pd * gamma_strong
                )
                a2g_rates_noma[weak_idx] = bandwidth_per_pair * np.log2(
                    1 + beta_weak * Pd * gamma_weak /
                    (beta_strong * Pd * gamma_weak + 1)
                )

            # OMA A2G速率
            a2g_rates_oma = bandwidth_per_pair * np.log2(
                1 + Pd * channel_gains_a2g
            )

            # 4. 使用S2A分配器（临时模式选择）
            _, modes_temp = mode_selector.select_modes(
                sat_rates, a2g_rates_noma, a2g_rates_oma, sat_pairs
            )

            # 5. 分配S2A带宽
            b_allocated = s2a_allocator.allocate_bandwidth(
                sat_pairs, modes_temp, a2g_rates_noma, a2g_rates_oma,
                snr_linear, h_s2a, sat_power_factors
            )

            # 6. 计算S2A速率
            s2a_rates = s2a_allocator.compute_s2a_rates(
                sat_pairs, b_allocated, snr_linear, h_s2a, sat_power_factors
            )

            # 7. 计算ABS最终速率（含S2A约束）
            abs_noma_rates = np.minimum(a2g_rates_noma, s2a_rates)
            abs_oma_rates = np.minimum(a2g_rates_oma, s2a_rates)

            # 8. 最终模式选择
            final_rates, _ = mode_selector.select_modes(
                sat_rates, abs_noma_rates, abs_oma_rates, sat_pairs
            )

            # 9. 计算总速率
            sum_rate = np.sum(final_rates)

            # 记录优化历史（每10次记录一次，避免过多）
            if self.n_evaluations % 10 == 0:
                self.optimization_history.append({
                    'position': abs_position.copy(),
                    'sum_rate': sum_rate
                })

            # 返回负值（scipy最小化）
            return -sum_rate

        except Exception as e:
            # 异常处理：返回极大值（表示不可行解）
            print(f"  [Warning] Objective function error at {abs_position}: {e}")
            return 1e12

    def optimize_position_lbfgsb(self, user_positions,
                                 mode_selector, s2a_allocator,
                                 snr_linear, elevation_deg,
                                 sat_channel_gains, sat_rates, sat_pairs,
                                 sat_power_factors,
                                 Bd, Pd, Nd, Nsd,
                                 initial_position=None,
                                 max_iter=20):
        """
        L-BFGS-B优化ABS位置

        参数:
            user_positions: 用户位置 [N, 3]
            mode_selector: 模式选择器
            s2a_allocator: S2A资源分配器
            snr_linear: 卫星SNR（线性）
            elevation_deg: 卫星仰角
            sat_channel_gains: 卫星信道增益 [N]
            sat_rates: 卫星直达速率 [N]
            sat_pairs: 卫星配对
            sat_power_factors: 卫星功率分配因子
            Bd: ABS带宽
            Pd: ABS功率
            Nd: A2G噪声功率
            Nsd: S2A噪声功率
            initial_position: 初始位置 [x, y, h]（如None则用k-means）
            max_iter: 最大迭代次数

        返回:
            optimal_position: 最优位置 [x, y, h]
            optimal_rate: 最优总速率
            info: 优化详情
        """
        # 如果没有初始位置，使用固定起点（覆盖区域中心 + 中等高度）
        if initial_position is None:
            # 用户质心作为水平位置
            user_center_xy = np.mean(user_positions[:, :2], axis=0)
            # 中等高度
            initial_height = (self.config.abs_height_min + self.config.abs_height_max) / 2
            initial_position = np.array([user_center_xy[0], user_center_xy[1], initial_height])

        # 生成固定的小尺度衰落（避免随机性）
        np.random.seed(self.config.random_seed)
        fading_a2g = self.a2g_channel.generate_fading(
            len(user_positions), seed=self.config.random_seed
        )

        # 定义边界约束
        coverage_radius = self.config.coverage_radius
        bounds = [
            (-coverage_radius, coverage_radius),  # x坐标
            (-coverage_radius, coverage_radius),  # y坐标
            (self.config.abs_height_min, self.config.abs_height_max)  # 高度
        ]

        # 打包参数
        kwargs = {
            'user_positions': user_positions,
            'fading_a2g': fading_a2g,
            'sat_rates': sat_rates,
            'sat_pairs': sat_pairs,
            'sat_power_factors': sat_power_factors,
            'snr_linear': snr_linear,
            'elevation_deg': elevation_deg,
            'mode_selector': mode_selector,
            's2a_allocator': s2a_allocator,
            'Bd': Bd,
            'Pd': Pd,
            'Nd': Nd,
            'Nsd': Nsd
        }

        # 重置统计
        self.n_evaluations = 0
        self.optimization_history = []

        # 计算初始速率
        initial_rate = -self._objective_function(initial_position, **kwargs)

        # 创建带参数的目标函数
        def objective_with_params(abs_position):
            return self._objective_function(abs_position, **kwargs)

        # L-BFGS-B优化
        result = minimize(
            fun=objective_with_params,
            x0=initial_position,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': 1e-6,
                'gtol': 1e-5,
                'disp': False
            }
        )

        optimal_position = result.x
        optimal_rate = -result.fun

        # 优化详情
        info = {
            'initial_position': initial_position,
            'initial_rate': initial_rate,
            'optimal_position': optimal_position,
            'optimal_rate': optimal_rate,
            'improvement_abs': optimal_rate - initial_rate,
            'improvement_pct': 100 * (optimal_rate / initial_rate - 1),
            'n_iterations': result.nit,
            'n_evaluations': self.n_evaluations,
            'success': result.success,
            'message': result.message,
            'history': self.optimization_history
        }

        return optimal_position, optimal_rate, info

    def optimize_position_lbfgsb_pure(self, user_positions,
                                      mode_selector, s2a_allocator,
                                      snr_linear, elevation_deg,
                                      sat_channel_gains, sat_rates, sat_pairs,
                                      sat_power_factors,
                                      Bd, Pd, Nd, Nsd,
                                      max_iter=20,
                                      verbose=False):
        """
        纯L-BFGS-B优化（从固定起点开始，无k-means初始化）

        【推荐配置】（论文最终方案）：
            mode_selector=GreedySelector
            s2a_allocator=KKTAllocator
            max_iter=20

        初始位置：
            x, y = 用户质心
            h = 中等高度（275m）

        参数: 同optimize_position_lbfgsb

        返回:
            optimal_position: 最优位置 [x, y, h]
            optimal_rate: 最优总速率
            info: 优化详情
        """
        if verbose:
            print(f"  [L-BFGS-B Optimization] Starting...")

        # 直接使用L-BFGS-B（initial_position=None会自动使用固定起点）
        optimal_position, optimal_rate, opt_info = self.optimize_position_lbfgsb(
            user_positions, mode_selector, s2a_allocator,
            snr_linear, elevation_deg,
            sat_channel_gains, sat_rates, sat_pairs, sat_power_factors,
            Bd, Pd, Nd, Nsd,
            initial_position=None,  # 使用固定起点
            max_iter=max_iter
        )

        if verbose:
            print(f"  [Initial] Position: "
                  f"({opt_info['initial_position'][0]:.1f}, "
                  f"{opt_info['initial_position'][1]:.1f}, "
                  f"{opt_info['initial_position'][2]:.0f})")
            print(f"  [Initial] Rate: {opt_info['initial_rate']/1e6:.2f} Mbps")
            print(f"  [Optimal] Position: "
                  f"({optimal_position[0]:.1f}, {optimal_position[1]:.1f}, {optimal_position[2]:.0f})")
            print(f"  [Optimal] Rate: {optimal_rate/1e6:.2f} Mbps")
            print(f"  [Improvement] +{opt_info['improvement_abs']/1e6:.2f} Mbps "
                  f"(+{opt_info['improvement_pct']:.2f}%)")
            print(f"  [Convergence] Iterations: {opt_info['n_iterations']}, "
                  f"Evaluations: {opt_info['n_evaluations']}")

        return optimal_position, optimal_rate, opt_info


# ==================== 单元测试 ====================

if __name__ == "__main__":
    """简单测试"""
    from src.optimizers.mode_selector import GreedySelector
    from src.optimizers.resource_allocator import KKTAllocator

    print("=" * 60)
    print("位置优化器单元测试")
    print("=" * 60)

    # 配置
    snr_db = 20
    elevation_deg = 10
    abs_bandwidth = 1.2e6

    # 生成用户分布
    dist = UserDistribution(config.N, config.coverage_radius, seed=42)
    user_positions = dist.generate_uniform_circle()

    print(f"\n测试配置:")
    print(f"  用户数: {config.N}")
    print(f"  ABS带宽: {abs_bandwidth/1e6:.1f} MHz")
    print(f"  SNR: {snr_db} dB")
    print(f"  仰角: {elevation_deg}°")

    # 计算卫星信息
    sat_noma = SatelliteNOMA(config)
    sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)

    allocator = NOMAAllocator()
    sat_pairs, _ = allocator.optimal_user_pairing(sat_channel_gains)

    snr_linear = 10 ** (snr_db / 10)
    sat_rates, _, sat_power_factors = sat_noma.compute_achievable_rates(
        sat_channel_gains, snr_linear
    )

    # 创建优化器
    mode_selector = GreedySelector()
    s2a_allocator = KKTAllocator(config)

    Pd = config.Pd
    Nd = config.get_abs_noise_power(abs_bandwidth)
    Nsd = config.get_s2a_noise_power(config.Bs)

    # 测试位置优化
    print(f"\n【测试：混合位置优化】")
    optimizer = ContinuousPositionOptimizer(config, method='L-BFGS-B')

    optimal_pos, optimal_rate, info = optimizer.optimize_position_hybrid(
        user_positions, mode_selector, s2a_allocator,
        snr_linear, elevation_deg,
        sat_channel_gains, sat_rates, sat_pairs, sat_power_factors,
        abs_bandwidth, Pd, Nd, Nsd,
        max_iter=10,
        verbose=True
    )

    print(f"\n结果摘要:")
    print(f"  最优位置: ({optimal_pos[0]:.2f}, {optimal_pos[1]:.2f}, {optimal_pos[2]:.0f})")
    print(f"  最优速率: {optimal_rate/1e6:.2f} Mbps")
    print(f"  改进: +{info['improvement_abs']/1e6:.2f} Mbps ({info['improvement_pct']:.2f}%)")
    print(f"  收敛: {'✓' if info['success'] else '✗'}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
