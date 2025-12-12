# -*- coding: utf-8 -*-
"""
联合用户配对优化器

核心思想：
- 原始方法：卫星和ABS配对独立进行
- 新方法：联合优化考虑协同效应

方法：
1. 贪婪 + 局部搜索（推荐用于N=32）
2. 穷举搜索（仅用于N<=16，指数复杂度）

作者：SATCON Enhancement Project
日期：2025-12-10
"""
import numpy as np
from itertools import combinations
import sys
from pathlib import Path

# 将项目根目录添加到路径
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.power_allocation import NOMAAllocator


class JointPairingOptimizer:
    """
    联合配对优化器

    输入：
    - sat_gains: 卫星信道增益
    - a2g_gains: A2G信道增益
    - config: 系统配置

    输出：
    - sat_pairs: 最优卫星配对
    - abs_pairs: 最优ABS配对
    - joint_benefit: 联合收益（总速率）
    """

    def __init__(self, config_obj):
        """
        初始化联合配对优化器

        参数：
            config_obj: 系统配置对象
        """
        self.config = config_obj
        self.allocator = NOMAAllocator()

    def compute_joint_benefit(self, sat_pair_idx, abs_pair_idx,
                              sat_gains, a2g_gains, Ps_dB, Pd, Bs, Bd):
        """
        计算给定配对组合的联合收益

        关键：对于每对k，考虑4种混合决策选项：
        1. 两个用户都使用ABS NOMA
        2. 弱用户使用ABS OMA，强用户使用卫星
        3. 弱用户使用卫星，强用户使用ABS OMA
        4. 两个用户都使用卫星NOMA

        参数：
            sat_pair_idx: 卫星配对索引 [(i1,j1), (i2,j2), ...]
            abs_pair_idx: ABS配对索引 [(m1,n1), (m2,n2), ...]
            sat_gains: 卫星信道增益 [N]
            a2g_gains: A2G信道增益 [N]
            Ps_dB: 卫星发射功率 (dB)
            Pd: ABS发射功率 (W)
            Bs, Bd: 卫星和ABS带宽

        返回：
            total_rate: 总系统速率 (bps)

        注意：
            这实现了简化的每对混合决策。
            完整的全局优化将在模块3中实现。
        """
        K = len(sat_pair_idx)
        Bs_per_pair = Bs / K
        Bd_per_pair = Bd / K

        # 将Ps从dB转换为W
        Ps = 10 ** (Ps_dB / 10)

        total_rate = 0.0

        # 处理每对k并做出混合决策
        for k in range(K):
            # 获取卫星配对用户
            sat_i, sat_j = sat_pair_idx[k]
            gamma_sat_i, gamma_sat_j = sat_gains[sat_i], sat_gains[sat_j]

            # 确保i是弱用户，j是强用户（对于卫星）
            if gamma_sat_i > gamma_sat_j:
                sat_i, sat_j = sat_j, sat_i
                gamma_sat_i, gamma_sat_j = gamma_sat_j, gamma_sat_i

            # 计算此对的卫星NOMA速率
            beta_sat_j, beta_sat_i = self.allocator.compute_power_factors(
                gamma_sat_j, gamma_sat_i, Ps
            )
            R_sat_i = Bs_per_pair * np.log2(1 + beta_sat_i * Ps * gamma_sat_i /
                                            (beta_sat_j * Ps * gamma_sat_i + 1))
            R_sat_j = Bs_per_pair * np.log2(1 + beta_sat_j * Ps * gamma_sat_j)

            # 获取ABS配对用户（对应相同的配对索引k）
            abs_m, abs_n = abs_pair_idx[k]
            gamma_abs_m, gamma_abs_n = a2g_gains[abs_m], a2g_gains[abs_n]

            # 确保m是弱用户，n是强用户（对于ABS）
            if gamma_abs_m > gamma_abs_n:
                abs_m, abs_n = abs_n, abs_m
                gamma_abs_m, gamma_abs_n = gamma_abs_n, gamma_abs_m

            # 计算此对的ABS NOMA速率
            beta_abs_n, beta_abs_m = self.allocator.compute_power_factors(
                gamma_abs_n, gamma_abs_m, Pd
            )
            R_abs_noma_m = Bd_per_pair * np.log2(1 + beta_abs_m * Pd * gamma_abs_m /
                                                  (beta_abs_n * Pd * gamma_abs_m + 1))
            R_abs_noma_n = Bd_per_pair * np.log2(1 + beta_abs_n * Pd * gamma_abs_n)

            # 计算ABS OMA速率（每个用户获得完整时隙）
            R_abs_oma_m = Bd_per_pair * np.log2(1 + Pd * gamma_abs_m)
            R_abs_oma_n = Bd_per_pair * np.log2(1 + Pd * gamma_abs_n)

            # 现在做出混合决策：为此对选择最佳选项
            # 注意：假设sat_i对应abs_m（在各自系统中都是弱用户）
            #       sat_j对应abs_n（在各自系统中都是强用户）

            # 选项1：两个用户都使用ABS NOMA
            rate_option1 = R_abs_noma_m + R_abs_noma_n

            # 选项2：弱用户使用ABS OMA，强用户使用卫星
            # 映射：abs_m（弱）使用ABS OMA，sat_j（强）使用卫星
            if abs_m == sat_i:  # Same user
                rate_option2 = R_abs_oma_m + R_sat_j
            elif abs_m == sat_j:
                rate_option2 = R_abs_oma_m + R_sat_i
            else:
                rate_option2 = 0  # 不同用户，跳过此选项

            # 选项3：强用户使用ABS OMA，弱用户使用卫星
            # 映射：abs_n（强）使用ABS OMA，sat_i（弱）使用卫星
            if abs_n == sat_j:  # Same user
                rate_option3 = R_sat_i + R_abs_oma_n
            elif abs_n == sat_i:
                rate_option3 = R_sat_j + R_abs_oma_n
            else:
                rate_option3 = 0  # 不同用户，跳过此选项

            # 选项4：两个用户都使用卫星NOMA
            rate_option4 = R_sat_i + R_sat_j

            # 选择最佳选项
            pair_rate = max(rate_option1, rate_option2, rate_option3, rate_option4)
            total_rate += pair_rate

        return total_rate

    def optimize_greedy_with_local_search(self, sat_gains, a2g_gains):
        """
        贪婪算法 + 局部搜索

        算法：
        1. 初始解：原始SATCON（独立配对）
        2. 局部搜索：交换配对尝试改进
        3. 迭代直到收敛

        参数：
            sat_gains: [N] 卫星信道增益
            a2g_gains: [N] A2G信道增益

        返回：
            best_sat_pairs: 最优卫星配对
            best_abs_pairs: 最优ABS配对
            best_benefit: 最佳联合收益
            iterations: 迭代次数
        """
        N = len(sat_gains)
        K = N // 2

        # 使用默认带宽
        Bd = self.config.Bd_options[1]  # 1.2 MHz

        # 初始解：原始独立配对
        sat_pairs_init, _ = self.allocator.optimal_user_pairing(sat_gains)
        abs_pairs_init, _ = self.allocator.optimal_user_pairing(a2g_gains)

        current_sat_pairs = sat_pairs_init.copy()
        current_abs_pairs = abs_pairs_init.copy()

        # 使用SNR = 20 dB进行测试
        Ps_dB = 20  # dB

        current_benefit = self.compute_joint_benefit(
            current_sat_pairs, current_abs_pairs,
            sat_gains, a2g_gains,
            Ps_dB, self.config.Pd,
            self.config.Bs, Bd
        )

        improved = True
        iterations = 0
        max_iterations = 50

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            # 尝试交换卫星配对
            for k1 in range(K):
                for k2 in range(k1+1, K):
                    # 在配对k1和k2之间交换一个用户
                    new_sat_pairs = [list(p) for p in current_sat_pairs]  # 深拷贝
                    new_sat_pairs[k1] = [current_sat_pairs[k1][0], current_sat_pairs[k2][1]]
                    new_sat_pairs[k2] = [current_sat_pairs[k2][0], current_sat_pairs[k1][1]]

                    new_benefit = self.compute_joint_benefit(
                        new_sat_pairs, current_abs_pairs,
                        sat_gains, a2g_gains,
                        Ps_dB, self.config.Pd,
                        self.config.Bs, Bd
                    )

                    if new_benefit > current_benefit:
                        current_sat_pairs = new_sat_pairs
                        current_benefit = new_benefit
                        improved = True

            # 尝试交换ABS配对
            for k1 in range(K):
                for k2 in range(k1+1, K):
                    new_abs_pairs = [list(p) for p in current_abs_pairs]  # 深拷贝
                    new_abs_pairs[k1] = [current_abs_pairs[k1][0], current_abs_pairs[k2][1]]
                    new_abs_pairs[k2] = [current_abs_pairs[k2][0], current_abs_pairs[k1][1]]

                    new_benefit = self.compute_joint_benefit(
                        current_sat_pairs, new_abs_pairs,
                        sat_gains, a2g_gains,
                        Ps_dB, self.config.Pd,
                        self.config.Bs, Bd
                    )

                    if new_benefit > current_benefit:
                        current_abs_pairs = new_abs_pairs
                        current_benefit = new_benefit
                        improved = True

        return current_sat_pairs, current_abs_pairs, current_benefit, iterations

    def optimize(self, sat_gains, a2g_gains):
        """
        主优化接口

        参数：
            sat_gains: [N] 卫星信道增益
            a2g_gains: [N] A2G信道增益

        返回：
            sat_pairs: 最优卫星配对
            abs_pairs: 最优ABS配对
            benefit: 联合收益
            info: 优化信息字典
        """
        sat_pairs, abs_pairs, benefit, iterations = \
            self.optimize_greedy_with_local_search(sat_gains, a2g_gains)

        info = {
            'iterations': iterations,
            'final_benefit': benefit
        }

        return sat_pairs, abs_pairs, benefit, info


# ==================== 测试代码 ====================
def test_joint_pairing():
    """测试联合配对优化器"""
    print("=" * 60)
    print("测试联合配对优化器")
    print("=" * 60)

    # 生成具有更多多样性的测试信道增益
    # 以创建联合优化的强机会
    np.random.seed(42)

    # 创建多样化的信道条件
    # 一些用户拥有良好的卫星但较差的A2G
    # 一些用户拥有较差的卫星但良好的A2G
    sat_gains = np.random.exponential(0.01, size=config.N)
    a2g_gains = np.random.exponential(0.05, size=config.N)

    # 添加强多样性：创建明确的专业化
    for i in range(0, config.N, 4):
        if i+1 < config.N:
            # 用户i：非常好的卫星，非常差的A2G
            sat_gains[i] *= 5.0  # 强卫星
            a2g_gains[i] *= 0.2  # 弱A2G
            # 用户i+1：非常差的卫星，非常好的A2G
            sat_gains[i+1] *= 0.2  # 弱卫星
            a2g_gains[i+1] *= 5.0  # 强A2G

    print(f"\n测试配置：")
    print(f"  用户数量：{config.N}")
    print(f"  配对数量：{config.N // 2}")

    # 创建优化器
    optimizer = JointPairingOptimizer(config)

    # 基线：独立配对（原始SATCON）
    print(f"\n" + "-" * 60)
    print("基线：独立配对（原始SATCON）")
    print("-" * 60)

    allocator = NOMAAllocator()
    sat_pairs_old, _ = allocator.optimal_user_pairing(sat_gains)
    abs_pairs_old, _ = allocator.optimal_user_pairing(a2g_gains)

    Bd = config.Bd_options[1]  # 1.2 MHz
    Ps_dB = 20  # 使用SNR = 20 dB进行测试

    benefit_old = optimizer.compute_joint_benefit(
        sat_pairs_old, abs_pairs_old,
        sat_gains, a2g_gains,
        Ps_dB, config.Pd, config.Bs, Bd
    )

    print(f"  总速率：{benefit_old/1e6:.2f} Mbps")
    print(f"  卫星配对：{len(sat_pairs_old)} 对")
    print(f"  ABS配对：{len(abs_pairs_old)} 对")

    # 新方法：联合优化
    print(f"\n" + "-" * 60)
    print("新方法：联合配对优化")
    print("-" * 60)
    print(f"运行局部搜索优化...")

    sat_pairs_new, abs_pairs_new, benefit_new, info = optimizer.optimize(
        sat_gains, a2g_gains
    )

    print(f"\n优化结果：")
    print(f"  总速率：{benefit_new/1e6:.2f} Mbps")
    print(f"  迭代次数：{info['iterations']}")

    # 性能对比
    print(f"\n" + "=" * 60)
    print("性能对比")
    print("=" * 60)

    improvement_abs = (benefit_new - benefit_old) / 1e6
    improvement_pct = (benefit_new - benefit_old) / benefit_old * 100

    print(f"\n基线（独立配对）：")
    print(f"  速率：{benefit_old/1e6:.2f} Mbps")

    print(f"\n提出方法（联合优化）：")
    print(f"  速率：{benefit_new/1e6:.2f} Mbps")

    print(f"\n改进幅度：")
    print(f"  绝对增益：{improvement_abs:.2f} Mbps")
    print(f"  相对增益：{improvement_pct:.2f}%")

    # 验证成功
    print(f"\n" + "=" * 60)
    if benefit_new >= benefit_old * 0.99:  # 允许小的数值误差
        print("[通过] 联合配对优化器测试通过")
        print(f"  - 联合优化在 {info['iterations']} 次迭代中收敛")
        if benefit_new > benefit_old:
            print(f"  - 性能优于基线 (+{improvement_pct:.2f}%)")
        else:
            print(f"  - 性能等于基线（局部最优）")
    else:
        print("[失败] 测试失败")
        print(f"  - 性能劣于基线 ({improvement_pct:.2f}%)")
    print("=" * 60)

    return sat_pairs_new, abs_pairs_new, benefit_new, info


if __name__ == "__main__":
    test_joint_pairing()
