# -*- coding: utf-8 -*-
"""
基于梯度的ABS位置优化器

优化目标：最大化系统总速率
方法：L-BFGS-B梯度优化

与k-means的对比：
- 原始方法：min sum ||u_i - p||^2 (几何距离)
- 新方法：max sum R_i(p, h) (系统速率)

作者：SATCON Enhancement Project
日期：2025-12-10
"""
import numpy as np
from scipy.optimize import minimize
import sys
from pathlib import Path

# 将项目根目录添加到路径
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.a2g_channel import A2GChannel


class GradientPositionOptimizer:
    """
    基于梯度的ABS位置优化器

    核心创新：
    - 从几何优化（k-means）转向性能驱动优化
    - 联合优化3D位置 [x, y, h]
    - 考虑信道物理特性（路径损耗、LOS概率）
    """

    def __init__(self, config_obj):
        """
        初始化优化器

        参数：
            config_obj: 系统配置对象
        """
        self.config = config_obj
        self.a2g_channel = A2GChannel()

        # 优化边界
        self.bounds_xy = (-config_obj.coverage_radius, config_obj.coverage_radius)
        self.bounds_h = (50, 500)  # 高度范围：50m - 500m

    def compute_total_rate(self, pos_3d, user_positions, a2g_gains_fading):
        """
        计算给定ABS位置下的系统总速率

        这是优化的目标函数

        参数：
            pos_3d: [x, y, h] ABS 3D位置
            user_positions: [N, 2] 用户位置数组
            a2g_gains_fading: [N] A2G小尺度衰落（预生成，固定）

        返回：
            total_rate: 系统总速率（bps）

        物理意义：
            - 将ABS放置在位置 (x, y, h)
            - 计算每个用户的可达速率
            - 求和得到总速率
        """
        x, y, h = pos_3d
        total_rate = 0.0

        for i, u in enumerate(user_positions):
            # 计算2D距离（水平距离）
            r = np.sqrt((x - u[0])**2 + (y - u[1])**2)

            # 使用默认带宽（1.2 MHz）
            Bd = self.config.Bd_options[1]  # 1.2 MHz

            # 计算A2G信道增益（包括路径损耗和小尺度衰落）
            gamma = self.a2g_channel.compute_channel_gain(
                h=h,
                r=r,
                fading_gain=a2g_gains_fading[i],
                G_tx_dB=self.config.Gd_t_dB,
                G_rx_dB=self.config.Gd_r_dB,
                noise_power=self.config.get_abs_noise_power(Bd)
            )

            # 计算OMA速率（简化版本，作为位置优化的代理）
            # 注意：实际系统使用NOMA，但在位置优化中使用OMA作为代理
            rate = Bd * np.log2(1 + self.config.Pd * gamma)
            total_rate += rate

        return total_rate

    def optimize(self, user_positions, a2g_gains_fading, initial_guess=None):
        """
        优化ABS位置

        使用L-BFGS-B算法进行基于梯度的优化

        参数：
            user_positions: [N, 2] 用户位置
            a2g_gains_fading: [N] A2G小尺度衰落
            initial_guess: [3] 初始位置猜测（可选）
                          如果未提供，使用用户质心 + 默认高度

        返回：
            optimal_position: [3] 最优ABS位置 [x, y, h]
            optimization_info: dict 优化信息
                - success: 是否成功收敛
                - iterations: 迭代次数
                - final_rate: 最优速率
                - initial_rate: 初始速率
                - improvement: 绝对改进
        """
        # 初始猜测：用户质心 + 默认高度100m
        if initial_guess is None:
            center_xy = np.mean(user_positions[:, :2], axis=0)
            initial_guess = [center_xy[0], center_xy[1], 100.0]

        # 定义目标函数（最小化负速率 = 最大化速率）
        def objective(pos_3d):
            rate = self.compute_total_rate(pos_3d, user_positions, a2g_gains_fading)
            return -rate  # 取负值用于最小化

        # 边界约束
        bounds = [
            (self.bounds_xy[0], self.bounds_xy[1]),  # x范围
            (self.bounds_xy[0], self.bounds_xy[1]),  # y范围
            (self.bounds_h[0], self.bounds_h[1])     # h范围
        ]

        # L-BFGS-B优化（带边界的拟牛顿法）
        result = minimize(
            objective,
            x0=initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 100,      # 最大迭代次数
                'ftol': 1e-6         # 函数值收敛容差
            }
        )

        # 收集优化信息
        optimization_info = {
            'success': result.success,
            'iterations': result.nit,
            'final_rate': -result.fun,  # 恢复正值
            'initial_rate': -objective(initial_guess),
            'improvement': (-result.fun) - (-objective(initial_guess)),
            'message': result.message
        }

        return result.x, optimization_info


# ==================== 测试代码 ====================
def test_gradient_optimizer():
    """测试梯度位置优化器"""
    from src.user_distribution import UserDistribution
    from src.abs_placement import ABSPlacement

    print("=" * 60)
    print("测试梯度位置优化器")
    print("=" * 60)

    # 生成测试用户（32个用户，均匀分布在500m半径圆内）
    dist = UserDistribution(config.N, config.coverage_radius, seed=42)
    user_positions = dist.generate_uniform_circle()

    print(f"\n用户分布：")
    print(f"  用户数量：{config.N}")
    print(f"  覆盖半径：{config.coverage_radius}m")
    print(f"  质心：({np.mean(user_positions[:, 0]):.2f}, {np.mean(user_positions[:, 1]):.2f})")

    # 生成A2G衰落（固定以确保公平比较）
    a2g_channel = A2GChannel()
    a2g_fading = a2g_channel.generate_fading(config.N, seed=42)

    # 创建优化器
    optimizer = GradientPositionOptimizer(config)

    # 优化
    print(f"\n开始梯度优化...")
    optimal_pos, info = optimizer.optimize(user_positions, a2g_fading)

    print(f"\n梯度优化结果：")
    print(f"  最优位置：({optimal_pos[0]:.2f}, {optimal_pos[1]:.2f}, {optimal_pos[2]:.0f}m)")
    print(f"  收敛状态：{'成功' if info['success'] else '失败'}")
    print(f"  迭代次数：{info['iterations']}")
    print(f"  初始速率：{info['initial_rate']/1e6:.2f} Mbps")
    print(f"  最优速率：{info['final_rate']/1e6:.2f} Mbps")
    print(f"  改进幅度：{info['improvement']/1e6:.2f} Mbps (+{info['improvement']/info['initial_rate']*100:.1f}%)")

    # 与k-means对比
    print(f"\n" + "=" * 60)
    print("与原始k-means方法对比")
    print("=" * 60)

    kmeans_placement = ABSPlacement()
    kmeans_xy, method, cost = kmeans_placement.optimize_xy_position(user_positions)
    kmeans_h, _ = kmeans_placement.optimize_height(kmeans_xy, user_positions, a2g_channel)
    kmeans_pos = [kmeans_xy[0], kmeans_xy[1], kmeans_h]

    # 计算k-means位置的速率
    kmeans_rate = optimizer.compute_total_rate(kmeans_pos, user_positions, a2g_fading)

    print(f"\nk-means结果：")
    print(f"  位置：({kmeans_pos[0]:.2f}, {kmeans_pos[1]:.2f}, {kmeans_pos[2]:.0f}m)")
    print(f"  速率：{kmeans_rate/1e6:.2f} Mbps")

    print(f"\n性能对比：")
    print(f"  梯度优化 vs k-means：")
    print(f"    绝对增益：{(info['final_rate'] - kmeans_rate)/1e6:.2f} Mbps")
    print(f"    相对增益：{(info['final_rate'] - kmeans_rate)/kmeans_rate*100:.1f}%")

    # 验证成功标准
    print(f"\n" + "=" * 60)
    if info['success'] and info['final_rate'] > kmeans_rate:
        print("[通过] 梯度位置优化器测试通过")
        print(f"  - 优化成功收敛")
        print(f"  - 性能优于k-means (+{(info['final_rate'] - kmeans_rate)/kmeans_rate*100:.1f}%)")
    else:
        print("[失败] 测试失败")
        if not info['success']:
            print(f"  - 优化未收敛：{info['message']}")
        if info['final_rate'] <= kmeans_rate:
            print(f"  - 性能未优于k-means")
    print("=" * 60)

    return optimal_pos, info, kmeans_pos, kmeans_rate


if __name__ == "__main__":
    test_gradient_optimizer()
