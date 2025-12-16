# -*- coding: utf-8 -*-
"""
联合优化SATCON系统

集成三个增强模块：
1. 基于梯度的ABS位置优化（模块1）
2. 联合用户配对优化（模块2）
3. 基于整数规划的混合决策（模块3）

核心创新：交替优化框架
- 原始方法：顺序优化（位置 -> 配对 -> 决策）
- 新方法：带反馈循环的迭代联合优化

算法：
1. 初始化ABS位置（使用梯度优化器）
2. 重复直到收敛：
   a. 固定位置，优化联合配对
   b. 固定配对，优化混合决策
   c. 固定配对/决策，优化位置
   d. 检查收敛
3. 返回最终解

作者：SATCON Enhancement Project
日期：2025-12-10
"""
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.user_distribution import UserDistribution
from src.a2g_channel import A2GChannel, S2AChannel
from src.noma_transmission import SatelliteNOMA
from src.power_allocation import NOMAAllocator

# 导入增强模块
from src_enhanced.gradient_position_optimizer import GradientPositionOptimizer
from src_enhanced.joint_pairing_optimizer import JointPairingOptimizer
from src_enhanced.integer_programming_decision import IntegerProgrammingDecision
from src_enhanced.resource_constrained_decision import ResourceConstrainedDecision


class JointOptimizationSATCON:
    """
    联合优化SATCON系统

    将三个增强模块组合成统一框架，
    通过迭代优化实现全局系统性能
    """

    def __init__(self, config_obj, abs_bandwidth,
                 use_module1=True, use_module2=True, use_module3=True,
                 max_abs_users=None, max_s2a_capacity=None, enforce_fairness=False):
        """
        初始化联合优化SATCON系统

        参数：
            config_obj: 配置对象
            abs_bandwidth: ABS带宽 (Hz)
            use_module1: 启用梯度位置优化
            use_module2: 启用联合配对优化
            use_module3: 启用整数规划决策
            max_abs_users: ABS最大同时服务用户对数（资源约束）
            max_s2a_capacity: S2A回程最大容量，单位bps（资源约束）
            enforce_fairness: 是否强制公平性约束
        """
        self.config = config_obj
        self.Bd = abs_bandwidth

        # 模块标志（用于消融研究）
        self.use_module1 = use_module1
        self.use_module2 = use_module2
        self.use_module3 = use_module3

        # 资源约束参数（方案A）
        self.max_abs_users = max_abs_users
        self.max_s2a_capacity = max_s2a_capacity
        self.enforce_fairness = enforce_fairness

        # 初始化基础系统
        self.sat_noma = SatelliteNOMA(config_obj)
        self.a2g_channel = A2GChannel()
        self.s2a_channel = S2AChannel()
        self.allocator = NOMAAllocator()

        # 初始化增强模块
        if use_module1:
            self.gradient_optimizer = GradientPositionOptimizer(config_obj)

        if use_module2:
            self.pairing_optimizer = JointPairingOptimizer(config_obj)

        # 总是初始化决策优化器（即使基线也需要）
        # 如果有资源约束，使用ResourceConstrainedDecision
        if max_abs_users is not None or max_s2a_capacity is not None:
            self.decision_optimizer = ResourceConstrainedDecision(
                max_abs_users=max_abs_users,
                max_s2a_capacity=max_s2a_capacity,
                use_ilp=True,
                enforce_fairness=enforce_fairness
            )
        else:
            self.decision_optimizer = IntegerProgrammingDecision()

        # 如果模块禁用则回退到原始方法
        from src.abs_placement import ABSPlacement
        self.abs_placement = ABSPlacement()

        # 噪声功率
        self.Nd = config_obj.get_abs_noise_power(abs_bandwidth)
        self.Nsd = config_obj.get_s2a_noise_power()

        # 优化参数
        self.max_iterations = 10
        self.convergence_threshold = 1e-3  # 0.1% 改进

    def compute_channel_gains(self, user_positions, abs_position,
                             elevation_deg, snr_db, seed):
        """
        计算所有信道增益（卫星和A2G）

        参数：
            user_positions: 用户位置
            abs_position: ABS位置 [x, y, h]
            elevation_deg: 卫星仰角
            snr_db: 卫星SNR
            seed: 衰落的随机种子

        返回：
            sat_gains: 卫星信道增益 [N]
            a2g_gains: A2G信道增益 [N]
            sat_rates: 卫星NOMA速率 [N]
        """
        # 卫星信道增益
        snr_linear = 10 ** (snr_db / 10)
        sat_gains = self.sat_noma.compute_channel_gains_with_pathloss(elevation_deg)
        sat_rates, _ = self.sat_noma.compute_achievable_rates(sat_gains, snr_linear)

        # A2G信道增益
        h_abs = abs_position[2]
        distances_2d = np.sqrt((user_positions[:, 0] - abs_position[0])**2 +
                               (user_positions[:, 1] - abs_position[1])**2)

        fading_a2g = self.a2g_channel.generate_fading(self.config.N, seed=seed)
        a2g_gains = np.array([
            self.a2g_channel.compute_channel_gain(
                h_abs, r, fading,
                self.config.Gd_t_dB, self.config.Gd_r_dB, self.Nd
            )
            for r, fading in zip(distances_2d, fading_a2g)
        ])

        return sat_gains, a2g_gains, sat_rates

    def optimize_position(self, user_positions, a2g_fading, initial_position=None):
        """
        优化ABS位置

        如果启用则使用模块1（梯度优化器），
        否则回退到k-means

        参数：
            user_positions: 用户位置
            a2g_fading: A2G衰落系数（固定以确保公平性）
            initial_position: 初始猜测 [x, y, h]

        返回：
            optimal_position: 优化的ABS位置 [x, y, h]
        """
        if self.use_module1:
            # 模块1：基于梯度的优化
            optimal_pos, _ = self.gradient_optimizer.optimize(
                user_positions, a2g_fading, initial_guess=initial_position
            )
            return optimal_pos
        else:
            # 回退：原始k-means + 高度优化
            abs_xy, _, _ = self.abs_placement.optimize_xy_position(user_positions)
            abs_h, _ = self.abs_placement.optimize_height(
                abs_xy, user_positions, self.a2g_channel
            )
            return np.array([abs_xy[0], abs_xy[1], abs_h])

    def optimize_pairing(self, sat_gains, a2g_gains):
        """
        优化用户配对

        如果启用则使用模块2（联合配对），
        否则回退到独立配对

        参数：
            sat_gains: 卫星信道增益
            a2g_gains: A2G信道增益

        返回：
            sat_pairs: 卫星配对 [(i,j), ...]
            abs_pairs: ABS配对 [(m,n), ...]
        """
        if self.use_module2:
            # 模块2：联合配对优化
            sat_pairs, abs_pairs, _, _ = self.pairing_optimizer.optimize_greedy_with_local_search(
                sat_gains, a2g_gains
            )
            return sat_pairs, abs_pairs
        else:
            # 回退：独立配对
            sat_pairs, _ = self.allocator.optimal_user_pairing(sat_gains)
            abs_pairs, _ = self.allocator.optimal_user_pairing(a2g_gains)
            return sat_pairs, abs_pairs

    def optimize_decision(self, sat_pairs, abs_pairs, sat_gains, a2g_gains, Ps_dB):
        """
        优化混合NOMA/OMA决策

        如果启用则使用模块3（整数规划），
        否则回退到贪婪决策

        参数：
            sat_pairs: 卫星配对
            abs_pairs: ABS配对
            sat_gains: 卫星信道增益
            a2g_gains: A2G信道增益
            Ps_dB: 卫星功率 (dB)

        返回：
            decisions: 每对的模式决策
            final_rate: 总系统速率
        """
        if self.use_module3:
            # 模块3：整数规划决策
            decisions, final_rate, _ = self.decision_optimizer.optimize(
                sat_pairs, abs_pairs, sat_gains, a2g_gains,
                Ps_dB, self.config.Pd, self.config.Bs, self.Bd
            )
            return decisions, final_rate
        else:
            # 回退：原始贪婪决策
            decisions, final_rate, _ = self.decision_optimizer.optimize_greedy(
                sat_pairs, abs_pairs, sat_gains, a2g_gains,
                Ps_dB, self.config.Pd, self.config.Bs, self.Bd
            )
            return decisions, final_rate

    def compute_system_rate(self, user_positions, abs_position,
                           elevation_deg, snr_db, seed):
        """
        计算给定配置的总系统速率

        参数：
            user_positions: 用户位置
            abs_position: ABS位置
            elevation_deg: 卫星仰角
            snr_db: 卫星SNR
            seed: 随机种子

        返回：
            total_rate: 总系统速率
            info: 详细信息
        """
        # 计算信道增益
        sat_gains, a2g_gains, sat_rates = self.compute_channel_gains(
            user_positions, abs_position, elevation_deg, snr_db, seed
        )

        # 优化配对
        sat_pairs, abs_pairs = self.optimize_pairing(sat_gains, a2g_gains)

        # 优化决策
        decisions, total_rate = self.optimize_decision(
            sat_pairs, abs_pairs, sat_gains, a2g_gains, snr_db
        )

        info = {
            'sat_pairs': sat_pairs,
            'abs_pairs': abs_pairs,
            'decisions': decisions,
            'mode_counts': {
                'noma': decisions.count('noma'),
                'oma_weak': decisions.count('oma_weak'),
                'oma_strong': decisions.count('oma_strong'),
                'sat': decisions.count('sat')
            }
        }

        return total_rate, info

    def optimize_joint(self, user_positions, elevation_deg, snr_db, seed,
                      verbose=False):
        """
        使用交替算法的联合优化

        算法：
        1. 初始化ABS位置
        2. 重复：
           a. 固定位置，优化配对 + 决策
           b. 固定配对/决策，优化位置
           c. 检查收敛
        3. 返回最佳解

        参数：
            user_positions: 用户位置
            elevation_deg: 卫星仰角
            snr_db: 卫星SNR
            seed: 随机种子
            verbose: 打印调试信息

        返回：
            best_rate: 最佳达到速率
            best_config: 最佳配置
        """
        # 生成固定的A2G衰落以确保公平比较
        a2g_fading = self.a2g_channel.generate_fading(self.config.N, seed=seed)

        # 步骤1：初始化位置
        abs_position = self.optimize_position(user_positions, a2g_fading)

        best_rate = 0
        best_config = None

        # 步骤2：交替优化
        for iteration in range(self.max_iterations):
            # 使用当前位置计算系统速率
            current_rate, info = self.compute_system_rate(
                user_positions, abs_position, elevation_deg, snr_db, seed
            )

            if verbose:
                print(f"  迭代 {iteration+1}: 速率={current_rate/1e6:.2f} Mbps, "
                      f"位置=[{abs_position[0]:.1f}, {abs_position[1]:.1f}, {abs_position[2]:.0f}]")

            # 更新最佳
            if current_rate > best_rate:
                best_rate = current_rate
                best_config = {
                    'abs_position': abs_position.copy(),
                    'sat_pairs': info['sat_pairs'],
                    'abs_pairs': info['abs_pairs'],
                    'decisions': info['decisions'],
                    'mode_counts': info['mode_counts']
                }

                improvement = (current_rate - best_rate) / best_rate if best_rate > 0 else float('inf')

                # 检查收敛
                if iteration > 0 and improvement < self.convergence_threshold:
                    if verbose:
                        print(f"  在迭代 {iteration+1} 收敛")
                    break

            # 基于当前配对/决策更新位置
            # （实际上，这需要更复杂的方法
            #  考虑位置对配对/决策的影响）
            # 目前，我们执行单次优化
            if iteration == 0 and self.use_module1:
                # 尝试精炼位置
                abs_position_new = self.optimize_position(
                    user_positions, a2g_fading, initial_position=abs_position
                )

                # 检查精炼是否有帮助
                new_rate, _ = self.compute_system_rate(
                    user_positions, abs_position_new, elevation_deg, snr_db, seed
                )

                if new_rate > current_rate:
                    abs_position = abs_position_new
                    if verbose:
                        print(f"    位置已精炼：速率提升到 {new_rate/1e6:.2f} Mbps")

        return best_rate, best_config

    def simulate_single_realization(self, snr_db, elevation_deg, seed,
                                   use_joint_optimization=True):
        """
        单次实现仿真

        参数：
            snr_db: 卫星SNR (dB)
            elevation_deg: 卫星仰角（度）
            seed: 随机种子
            use_joint_optimization: 使用交替优化

        返回：
            sum_rate: 总系统速率
            mode_stats: 模式统计
        """
        # 生成用户分布
        dist = UserDistribution(self.config.N, self.config.coverage_radius, seed=seed)
        user_positions = dist.generate_uniform_circle()

        if use_joint_optimization:
            # 联合优化
            sum_rate, config_info = self.optimize_joint(
                user_positions, elevation_deg, snr_db, seed, verbose=False
            )
            mode_stats = config_info['mode_counts']
        else:
            # 简单的单次优化
            a2g_fading = self.a2g_channel.generate_fading(self.config.N, seed=seed)
            abs_position = self.optimize_position(user_positions, a2g_fading)
            sum_rate, info = self.compute_system_rate(
                user_positions, abs_position, elevation_deg, snr_db, seed
            )
            mode_stats = info['mode_counts']

        return sum_rate, mode_stats

    def simulate_performance(self, snr_db_range, elevation_deg=10,
                           n_realizations=100, use_joint_optimization=True,
                           verbose=True):
        """
        蒙特卡洛性能仿真

        参数：
            snr_db_range: SNR值数组 (dB)
            elevation_deg: 卫星仰角
            n_realizations: 蒙特卡洛实现次数
            use_joint_optimization: 使用交替优化
            verbose: 显示进度条

        返回：
            mean_sum_rates: 平均总速率 [n_snr]
            mean_se: 平均频谱效率 [n_snr]
            std_sum_rates: 标准差 [n_snr]
            mode_statistics: 模式使用统计
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
            modules_enabled = []
            if self.use_module1:
                modules_enabled.append("M1:梯度位置")
            if self.use_module2:
                modules_enabled.append("M2:联合配对")
            if self.use_module3:
                modules_enabled.append("M3:整数规划")
            if not modules_enabled:
                modules_enabled.append("基线")

            module_str = "+".join(modules_enabled)
            print(f"联合SATCON仿真 [{module_str}]")
            print(f"  Bd={self.Bd/1e6:.1f}MHz, 联合优化={use_joint_optimization}")

        snr_iterator = tqdm(enumerate(snr_db_range), total=n_snr_points,
                           desc="仿真中", disable=not verbose)

        for i, snr_db in snr_iterator:
            for r in range(n_realizations):
                seed = self.config.random_seed + i * n_realizations + r

                sum_rate, mode_stats = self.simulate_single_realization(
                    snr_db, elevation_deg, seed, use_joint_optimization
                )

                sum_rates_all[i, r] = sum_rate
                for mode in mode_stats_all.keys():
                    if mode in mode_stats:
                        mode_stats_all[mode][i, r] = mode_stats[mode]

            if verbose:
                current_se = np.mean(sum_rates_all[i, :]) / self.Bd
                snr_iterator.set_postfix({'频谱效率': f'{current_se:.2f} bits/s/Hz'})

        # 统计
        mean_sum_rates = np.mean(sum_rates_all, axis=1)
        std_sum_rates = np.std(sum_rates_all, axis=1)
        mean_se = mean_sum_rates / self.Bd

        mean_mode_stats = {mode: np.mean(counts, axis=1)
                          for mode, counts in mode_stats_all.items()}

        return mean_sum_rates, mean_se, std_sum_rates, mean_mode_stats


# ==================== 测试代码 ====================
def test_joint_satcon_system():
    """测试联合优化SATCON系统"""
    print("=" * 70)
    print("测试联合优化SATCON系统")
    print("=" * 70)

    # 测试配置
    test_snr = np.array([10, 20, 30])
    n_real = 5

    # 创建用于对比的系统
    systems = {
        '基线（原始）': JointOptimizationSATCON(
            config, 1.2e6,
            use_module1=False, use_module2=False, use_module3=False
        ),
        '模块1（梯度位置）': JointOptimizationSATCON(
            config, 1.2e6,
            use_module1=True, use_module2=False, use_module3=False
        ),
        '模块2（联合配对）': JointOptimizationSATCON(
            config, 1.2e6,
            use_module1=False, use_module2=True, use_module3=False
        ),
        '模块3（整数规划）': JointOptimizationSATCON(
            config, 1.2e6,
            use_module1=False, use_module2=False, use_module3=True
        ),
        '完整系统（全部）': JointOptimizationSATCON(
            config, 1.2e6,
            use_module1=True, use_module2=True, use_module3=True
        )
    }

    print(f"\n测试配置：")
    print(f"  SNR点：{test_snr}")
    print(f"  实现次数：{n_real}")
    print(f"  系统数：{len(systems)}")

    results = {}

    for name, system in systems.items():
        print(f"\n{'-'*70}")
        print(f"Testing: {name}")
        print(f"{'-'*70}")

        mean_rates, mean_se, _, mode_stats = system.simulate_performance(
            test_snr, elevation_deg=10, n_realizations=n_real, verbose=True
        )

        results[name] = {
            'mean_rates': mean_rates,
            'mean_se': mean_se,
            'mode_stats': mode_stats
        }

    # Comparison
    print(f"\n{'='*70}")
    print("Performance Comparison")
    print(f"{'='*70}")

    baseline_rates = results['Baseline (Original)']['mean_rates']

    print(f"\n{'System':<25} {'SE@10dB':<12} {'SE@20dB':<12} {'SE@30dB':<12} {'Gain@20dB'}")
    print("-" * 70)

    for name, data in results.items():
        se = data['mean_se']
        gain = (data['mean_rates'][1] - baseline_rates[1]) / baseline_rates[1] * 100

        print(f"{name:<25} {se[0]:<12.2f} {se[1]:<12.2f} {se[2]:<12.2f} {gain:>7.2f}%")

    print("\n" + "=" * 70)
    print(" Joint Optimization SATCON Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    test_joint_satcon_system()
