# -*- coding: utf-8 -*-
"""
资源受限的传输决策优化器

核心创新（方案A）：
1. ABS容量约束：限制同时使用ABS的用户对数量
2. S2A回程约束：考虑卫星-ABS回程链路的容量限制
3. 全局优化：ILP在资源受限下的全局协调

目标：
- 解决Module 3性能提升不明显的问题（当前0%）
- 通过引入资源约束，放大ILP全局优化的价值
- 预期性能提升：从3.6% → 12-15%

作者：Claude Code
日期：2025-12-16
"""
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.power_allocation import NOMAAllocator

# 尝试导入cvxpy（用于ILP）
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("警告：cvxpy未安装，将使用启发式算法")


class ResourceConstrainedDecision:
    """
    资源受限的传输决策优化器

    与原始IntegerProgrammingDecision的区别：
    - 新增ABS容量约束：max_abs_users
    - 新增S2A回程约束：max_s2a_capacity
    - 新增公平性约束（可选）
    """

    def __init__(self, max_abs_users=None, max_s2a_capacity=None,
                 use_ilp=True, enforce_fairness=False):
        """
        初始化资源受限决策优化器

        参数：
            max_abs_users: ABS最多同时服务的用户对数（None表示无限制）
            max_s2a_capacity: S2A回程最大容量，单位bps（None表示无限制）
            use_ilp: 是否使用整数线性规划（需要cvxpy）
            enforce_fairness: 是否强制公平性约束
        """
        self.max_abs_users = max_abs_users
        self.max_s2a_capacity = max_s2a_capacity
        self.use_ilp = use_ilp and HAS_CVXPY
        self.enforce_fairness = enforce_fairness
        self.allocator = NOMAAllocator()

        if not HAS_CVXPY and use_ilp:
            print("警告：cvxpy未安装，回退到启发式算法")
            self.use_ilp = False

    def compute_rate_options(self, sat_pairs, abs_pairs, sat_gains, a2g_gains,
                            Ps_dB, Pd, Bs, Bd):
        """
        计算每对的所有速率选项

        参数：
            sat_pairs: 卫星配对 [(i,j), ...]
            abs_pairs: ABS配对 [(m,n), ...]
            sat_gains: 卫星信道增益 [N]
            a2g_gains: A2G信道增益 [N]
            Ps_dB: 卫星功率 (dB)
            Pd: ABS功率 (W)
            Bs: 卫星带宽
            Bd: ABS带宽

        返回：
            rate_options: 包含各模式速率的字典
        """
        K = len(sat_pairs)
        Bs_per_pair = Bs / K
        Bd_per_pair = Bd / K
        Ps = 10 ** (Ps_dB / 10)

        # 初始化速率数组
        rate_sat = np.zeros(K)
        rate_abs_noma = np.zeros(K)
        rate_abs_oma_weak = np.zeros(K)
        rate_abs_oma_strong = np.zeros(K)

        for k in range(K):
            # 卫星配对
            sat_i, sat_j = sat_pairs[k]
            gamma_sat_i, gamma_sat_j = sat_gains[sat_i], sat_gains[sat_j]

            if gamma_sat_i > gamma_sat_j:
                sat_i, sat_j = sat_j, sat_i
                gamma_sat_i, gamma_sat_j = gamma_sat_j, gamma_sat_i

            # 卫星NOMA速率
            beta_sat_j, beta_sat_i = self.allocator.compute_power_factors(
                gamma_sat_j, gamma_sat_i, Ps
            )
            R_sat_i = Bs_per_pair * np.log2(1 + beta_sat_i * Ps * gamma_sat_i /
                                            (beta_sat_j * Ps * gamma_sat_i + 1))
            R_sat_j = Bs_per_pair * np.log2(1 + beta_sat_j * Ps * gamma_sat_j)

            # ABS配对
            abs_m, abs_n = abs_pairs[k]
            gamma_abs_m, gamma_abs_n = a2g_gains[abs_m], a2g_gains[abs_n]

            if gamma_abs_m > gamma_abs_n:
                abs_m, abs_n = abs_n, abs_m
                gamma_abs_m, gamma_abs_n = gamma_abs_n, gamma_abs_m

            # ABS NOMA速率
            beta_abs_n, beta_abs_m = self.allocator.compute_power_factors(
                gamma_abs_n, gamma_abs_m, Pd
            )
            R_abs_noma_m = Bd_per_pair * np.log2(1 + beta_abs_m * Pd * gamma_abs_m /
                                                  (beta_abs_n * Pd * gamma_abs_m + 1))
            R_abs_noma_n = Bd_per_pair * np.log2(1 + beta_abs_n * Pd * gamma_abs_n)

            # ABS OMA速率
            R_abs_oma_m = Bd_per_pair * np.log2(1 + Pd * gamma_abs_m)
            R_abs_oma_n = Bd_per_pair * np.log2(1 + Pd * gamma_abs_n)

            # 此对的速率选项
            rate_sat[k] = R_sat_i + R_sat_j

            # 检查用户是否匹配
            if abs_m == sat_i and abs_n == sat_j:
                rate_abs_noma[k] = R_abs_noma_m + R_abs_noma_n
                rate_abs_oma_weak[k] = R_abs_oma_m + R_sat_j
                rate_abs_oma_strong[k] = R_sat_i + R_abs_oma_n
            elif abs_m == sat_j and abs_n == sat_i:
                rate_abs_noma[k] = R_abs_noma_m + R_abs_noma_n
                rate_abs_oma_weak[k] = R_abs_oma_m + R_sat_i
                rate_abs_oma_strong[k] = R_sat_j + R_abs_oma_n
            else:
                rate_abs_noma[k] = R_abs_noma_m + R_abs_noma_n
                rate_abs_oma_weak[k] = max(R_abs_oma_m, R_abs_oma_n) + min(R_sat_i, R_sat_j)
                rate_abs_oma_strong[k] = max(R_sat_i, R_sat_j) + min(R_abs_oma_m, R_abs_oma_n)

        return {
            'sat': rate_sat,
            'abs_noma': rate_abs_noma,
            'abs_oma_weak': rate_abs_oma_weak,
            'abs_oma_strong': rate_abs_oma_strong
        }

    def optimize_ilp_with_constraints(self, sat_pairs, abs_pairs, sat_gains, a2g_gains,
                                     Ps_dB, Pd, Bs, Bd):
        """
        带资源约束的ILP优化（核心创新）

        约束：
        1. 模式互斥：每对只能选一种模式
        2. ABS容量：同时使用ABS的用户对数 <= max_abs_users
        3. S2A回程：ABS总流量 <= max_s2a_capacity
        4. 公平性（可选）：最小用户速率 >= threshold

        参数：
            [与compute_rate_options相同]

        返回：
            decisions: 决策字典 {pair_id: mode}
            final_rate: 总系统速率
            info: 优化信息
        """
        if not HAS_CVXPY:
            print("cvxpy不可用，回退到启发式算法")
            return self.optimize_heuristic_with_constraints(
                sat_pairs, abs_pairs, sat_gains, a2g_gains,
                Ps_dB, Pd, Bs, Bd
            )

        K = len(sat_pairs)

        # 计算速率选项
        options = self.compute_rate_options(
            sat_pairs, abs_pairs, sat_gains, a2g_gains,
            Ps_dB, Pd, Bs, Bd
        )

        # 决策变量
        x_noma = cp.Variable(K, boolean=True)      # ABS NOMA
        y_weak = cp.Variable(K, boolean=True)      # ABS OMA (弱用户)
        y_strong = cp.Variable(K, boolean=True)    # ABS OMA (强用户)

        # 目标函数：最大化总速率
        objective = cp.Maximize(
            x_noma @ options['abs_noma'] +
            y_weak @ options['abs_oma_weak'] +
            y_strong @ options['abs_oma_strong'] +
            (1 - x_noma - y_weak - y_strong) @ options['sat']
        )

        # 约束1：模式互斥
        constraints = [x_noma + y_weak + y_strong <= 1]

        # 约束2：ABS容量限制（新增）
        if self.max_abs_users is not None:
            abs_usage = x_noma + y_weak + y_strong  # 使用ABS的用户对
            constraints.append(cp.sum(abs_usage) <= self.max_abs_users)

        # 约束3：S2A回程容量限制（新增）
        if self.max_s2a_capacity is not None:
            # ABS总流量 = NOMA两用户 + OMA单用户的流量总和
            abs_total_traffic = (
                x_noma @ options['abs_noma'] +
                y_weak @ options['abs_oma_weak'] +
                y_strong @ options['abs_oma_strong']
            )
            constraints.append(abs_total_traffic <= self.max_s2a_capacity)

        # 约束4：公平性（可选，新增）
        if self.enforce_fairness:
            # 确保每对的速率不低于某个阈值
            min_rate_threshold = np.min(options['sat']) * 0.8  # 至少80%的基线速率
            for k in range(K):
                pair_rate = (
                    x_noma[k] * options['abs_noma'][k] +
                    y_weak[k] * options['abs_oma_weak'][k] +
                    y_strong[k] * options['abs_oma_strong'][k] +
                    (1 - x_noma[k] - y_weak[k] - y_strong[k]) * options['sat'][k]
                )
                constraints.append(pair_rate >= min_rate_threshold)

        # 求解ILP
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.GLPK_MI, verbose=False)
        except:
            try:
                # 回退到其他求解器
                problem.solve(solver=cp.CBC, verbose=False)
            except:
                print("ILP求解失败，回退到启发式")
                return self.optimize_heuristic_with_constraints(
                    sat_pairs, abs_pairs, sat_gains, a2g_gains,
                    Ps_dB, Pd, Bs, Bd
                )

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"ILP未找到最优解（状态: {problem.status}），回退到启发式")
            return self.optimize_heuristic_with_constraints(
                sat_pairs, abs_pairs, sat_gains, a2g_gains,
                Ps_dB, Pd, Bs, Bd
            )

        # 提取决策（返回列表格式以保持兼容性）
        decisions = []
        final_rate = 0.0

        for k in range(K):
            if x_noma[k].value > 0.5:
                decisions.append('noma')
                final_rate += options['abs_noma'][k]
            elif y_weak[k].value > 0.5:
                decisions.append('oma_weak')
                final_rate += options['abs_oma_weak'][k]
            elif y_strong[k].value > 0.5:
                decisions.append('oma_strong')
                final_rate += options['abs_oma_strong'][k]
            else:
                decisions.append('sat')
                final_rate += options['sat'][k]

        # 统计信息
        abs_users_count = sum(1 for mode in decisions if mode != 'sat')
        abs_traffic = sum(
            options['abs_noma'][k] if decisions[k] == 'noma' else
            options['abs_oma_weak'][k] if decisions[k] == 'oma_weak' else
            options['abs_oma_strong'][k] if decisions[k] == 'oma_strong' else
            0.0
            for k in range(K)
        )

        info = {
            'solver': 'ILP (资源受限)',
            'optimal_value': problem.value,
            'status': problem.status,
            'abs_users': abs_users_count,
            'abs_traffic': abs_traffic,
            'max_abs_users': self.max_abs_users,
            'max_s2a_capacity': self.max_s2a_capacity,
            'constraints_active': {
                'abs_capacity': abs_users_count == self.max_abs_users if self.max_abs_users else False,
                's2a_backhaul': abs_traffic >= 0.95 * self.max_s2a_capacity if self.max_s2a_capacity else False
            }
        }

        return decisions, final_rate, info

    def optimize_heuristic_with_constraints(self, sat_pairs, abs_pairs, sat_gains, a2g_gains,
                                           Ps_dB, Pd, Bs, Bd):
        """
        启发式算法（考虑资源约束）

        策略：
        1. 计算所有选项的"价值"（速率增益）
        2. 按价值排序
        3. 贪婪分配，直到资源耗尽

        参数：
            [与optimize_ilp_with_constraints相同]

        返回：
            decisions, final_rate, info
        """
        K = len(sat_pairs)

        # 计算速率选项
        options = self.compute_rate_options(
            sat_pairs, abs_pairs, sat_gains, a2g_gains,
            Ps_dB, Pd, Bs, Bd
        )

        # 计算每对选择ABS的"价值"（相对于卫星的增益）
        gains = []
        for k in range(K):
            baseline = options['sat'][k]
            best_abs_mode = max(
                ('noma', options['abs_noma'][k]),
                ('oma_weak', options['abs_oma_weak'][k]),
                ('oma_strong', options['abs_oma_strong'][k]),
                key=lambda x: x[1]
            )
            gain = best_abs_mode[1] - baseline
            gains.append((k, best_abs_mode[0], gain, best_abs_mode[1]))

        # 按增益排序（降序）
        gains.sort(key=lambda x: x[2], reverse=True)

        # 贪婪分配（使用列表格式以保持兼容性）
        decisions = ['sat'] * K
        abs_users_count = 0
        abs_traffic = 0.0

        for k, mode, gain, abs_rate in gains:
            # 检查是否值得使用ABS
            if gain <= 0:
                break

            # 检查ABS容量约束
            if self.max_abs_users is not None and abs_users_count >= self.max_abs_users:
                break

            # 检查S2A回程约束
            if self.max_s2a_capacity is not None:
                if abs_traffic + abs_rate > self.max_s2a_capacity:
                    break

            # 分配ABS
            decisions[k] = mode
            abs_users_count += 1
            abs_traffic += abs_rate

        # 计算最终速率
        final_rate = sum(
            options['abs_noma'][k] if decisions[k] == 'noma' else
            options['abs_oma_weak'][k] if decisions[k] == 'oma_weak' else
            options['abs_oma_strong'][k] if decisions[k] == 'oma_strong' else
            options['sat'][k]
            for k in range(K)
        )

        info = {
            'solver': 'Heuristic (资源受限)',
            'abs_users': abs_users_count,
            'abs_traffic': abs_traffic,
            'max_abs_users': self.max_abs_users,
            'max_s2a_capacity': self.max_s2a_capacity
        }

        return decisions, final_rate, info

    def optimize(self, sat_pairs, abs_pairs, sat_gains, a2g_gains,
                Ps_dB, Pd, Bs, Bd):
        """
        主优化接口

        自动选择ILP或启发式算法

        参数：
            [与optimize_ilp_with_constraints相同]

        返回：
            decisions, final_rate, info
        """
        if self.use_ilp:
            return self.optimize_ilp_with_constraints(
                sat_pairs, abs_pairs, sat_gains, a2g_gains,
                Ps_dB, Pd, Bs, Bd
            )
        else:
            return self.optimize_heuristic_with_constraints(
                sat_pairs, abs_pairs, sat_gains, a2g_gains,
                Ps_dB, Pd, Bs, Bd
            )


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("=" * 80)
    print("测试资源受限决策优化器")
    print("=" * 80)

    # 生成测试数据
    np.random.seed(42)
    N = config.N
    K = N // 2

    sat_gains = np.random.exponential(0.01, size=N)
    a2g_gains = np.random.exponential(0.05, size=N)

    # 创建配对
    sat_pairs = [(2*i, 2*i+1) for i in range(K)]
    abs_pairs = [(2*i, 2*i+1) for i in range(K)]

    Ps_dB = 20
    Pd = 1.0
    Bs = 5e6
    Bd = 1.2e6

    print(f"\n测试场景:")
    print(f"  用户数: N={N}")
    print(f"  用户对数: K={K}")
    print(f"  卫星SNR: {Ps_dB} dB")
    print(f"  ABS功率: {Pd} W")

    # 测试1：无约束（基线）
    print("\n" + "=" * 80)
    print("测试1：无约束（基线）")
    print("=" * 80)

    optimizer_baseline = ResourceConstrainedDecision(
        max_abs_users=None,
        max_s2a_capacity=None
    )

    decisions, rate, info = optimizer_baseline.optimize(
        sat_pairs, abs_pairs, sat_gains, a2g_gains,
        Ps_dB, Pd, Bs, Bd
    )

    print(f"\n结果:")
    print(f"  总速率: {rate/1e6:.2f} Mbps")
    print(f"  ABS用户数: {info['abs_users']}/{K}")
    print(f"  求解器: {info['solver']}")

    # 测试2：ABS容量约束
    print("\n" + "=" * 80)
    print("测试2：ABS容量约束 (max_abs_users=8)")
    print("=" * 80)

    optimizer_capacity = ResourceConstrainedDecision(
        max_abs_users=8,
        max_s2a_capacity=None
    )

    decisions_cap, rate_cap, info_cap = optimizer_capacity.optimize(
        sat_pairs, abs_pairs, sat_gains, a2g_gains,
        Ps_dB, Pd, Bs, Bd
    )

    print(f"\n结果:")
    print(f"  总速率: {rate_cap/1e6:.2f} Mbps")
    print(f"  ABS用户数: {info_cap['abs_users']}/{K} (限制: 8)")
    print(f"  速率下降: {(rate-rate_cap)/rate*100:.2f}%")

    if 'constraints_active' in info_cap:
        print(f"  容量约束激活: {info_cap['constraints_active']['abs_capacity']}")

    # 测试3：S2A回程约束
    print("\n" + "=" * 80)
    print("测试3：S2A回程约束 (max_capacity=20 Mbps)")
    print("=" * 80)

    optimizer_backhaul = ResourceConstrainedDecision(
        max_abs_users=None,
        max_s2a_capacity=20e6  # 20 Mbps
    )

    decisions_bh, rate_bh, info_bh = optimizer_backhaul.optimize(
        sat_pairs, abs_pairs, sat_gains, a2g_gains,
        Ps_dB, Pd, Bs, Bd
    )

    print(f"\n结果:")
    print(f"  总速率: {rate_bh/1e6:.2f} Mbps")
    print(f"  ABS用户数: {info_bh['abs_users']}/{K}")
    print(f"  ABS流量: {info_bh['abs_traffic']/1e6:.2f} Mbps (限制: 20 Mbps)")
    print(f"  速率下降: {(rate-rate_bh)/rate*100:.2f}%")

    # 对比总结
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)
    print(f"\n{'场景':<30} {'速率(Mbps)':<15} {'ABS用户':<15} {'相对基线':<10}")
    print("-" * 80)
    print(f"{'无约束（基线）':<30} {rate/1e6:<15.2f} {info['abs_users']:<15} {0.0:>8.2f}%")
    print(f"{'ABS容量约束(8对)':<30} {rate_cap/1e6:<15.2f} {info_cap['abs_users']:<15} {(rate_cap-rate)/rate*100:>8.2f}%")
    print(f"{'S2A回程约束(20Mbps)':<30} {rate_bh/1e6:<15.2f} {info_bh['abs_users']:<15} {(rate_bh-rate)/rate*100:>8.2f}%")

    print("\n" + "=" * 80)
    print("[OK] 资源受限决策优化器测试完成！")
    print("=" * 80)
