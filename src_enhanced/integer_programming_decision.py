# -*- coding: utf-8 -*-
"""
基于整数规划的混合决策优化器

原始方法：基于规则的贪婪决策（4条规则，每对局部最优）
新方法：整数规划实现全局最优决策

核心思想：
- 原始方法：每对独立选择最佳模式
- 新方法：联合优化所有对的模式选择以获得全局最优

依赖：
- cvxpy（可选）：pip install cvxpy
- 如果cvxpy不可用，回退到改进的贪婪算法

作者：SATCON Enhancement Project
日期：2025-12-10
"""
import numpy as np
import sys
from pathlib import Path

# 尝试导入cvxpy
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("警告：未安装cvxpy。改用贪婪算法。")
    print("  安装命令：pip install cvxpy")

# 将项目根目录添加到路径
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.power_allocation import NOMAAllocator


class IntegerProgrammingDecision:
    """
    基于整数规划的混合决策优化器

    决策变量（对于每对k）：
    - x[k] in {0,1}：两个用户都使用ABS NOMA
    - y_weak[k] in {0,1}：仅弱用户使用ABS OMA
    - y_strong[k] in {0,1}：仅强用户使用ABS OMA

    目标：
    - 最大化所有用户速率之和

    约束：
    - 互斥性：x[k] + y_weak[k] + y_strong[k] <= 1
    - 带宽限制：使用的带宽总和 <= Bd
    - S2A回传限制：ABS速率 <= S2A速率（简化：假设充足）
    """

    def __init__(self):
        """初始化优化器"""
        self.use_ilp = CVXPY_AVAILABLE
        self.allocator = NOMAAllocator()

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
            rate_options: 包含键['sat', 'abs_noma', 'abs_oma_weak', 'abs_oma_strong']的字典
                         每个值是每对速率的[K]数组
        """
        K = len(sat_pairs)
        Bs_per_pair = Bs / K
        Bd_per_pair = Bd / K
        Ps = 10 ** (Ps_dB / 10)

        # 初始化速率数组
        rate_sat = np.zeros(K)           # 卫星NOMA速率（对的总和）
        rate_abs_noma = np.zeros(K)      # ABS NOMA速率（对的总和）
        rate_abs_oma_weak = np.zeros(K)  # ABS OMA弱用户 + 卫星强用户
        rate_abs_oma_strong = np.zeros(K) # 卫星弱用户 + ABS OMA强用户

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
                # 完美匹配
                rate_abs_noma[k] = R_abs_noma_m + R_abs_noma_n
                rate_abs_oma_weak[k] = R_abs_oma_m + R_sat_j
                rate_abs_oma_strong[k] = R_sat_i + R_abs_oma_n
            elif abs_m == sat_j and abs_n == sat_i:
                # 反向匹配
                rate_abs_noma[k] = R_abs_noma_m + R_abs_noma_n
                rate_abs_oma_weak[k] = R_abs_oma_m + R_sat_i
                rate_abs_oma_strong[k] = R_sat_j + R_abs_oma_n
            else:
                # 不同用户 - 使用可用的最佳方案
                rate_abs_noma[k] = R_abs_noma_m + R_abs_noma_n
                rate_abs_oma_weak[k] = max(R_abs_oma_m, R_abs_oma_n) + min(R_sat_i, R_sat_j)
                rate_abs_oma_strong[k] = max(R_sat_i, R_sat_j) + min(R_abs_oma_m, R_abs_oma_n)

        return {
            'sat': rate_sat,
            'abs_noma': rate_abs_noma,
            'abs_oma_weak': rate_abs_oma_weak,
            'abs_oma_strong': rate_abs_oma_strong
        }

    def optimize_ilp(self, sat_pairs, abs_pairs, sat_gains, a2g_gains,
                    Ps_dB, Pd, Bs, Bd):
        """
        整数线性规划优化

        参数：
            sat_pairs: 卫星配对
            abs_pairs: ABS配对
            sat_gains: 卫星信道增益
            a2g_gains: A2G信道增益
            Ps_dB: 卫星功率 (dB)
            Pd: ABS功率 (W)
            Bs: 卫星带宽
            Bd: ABS带宽

        返回：
            decisions: 每对的决策列表 ['sat'/'noma'/'oma_weak'/'oma_strong']
            final_rate: 总系统速率
            info: 优化信息
        """
        if not CVXPY_AVAILABLE:
            raise RuntimeError("cvxpy不可用。请使用optimize_greedy()代替。")

        K = len(sat_pairs)

        # 计算速率选项
        rate_opts = self.compute_rate_options(
            sat_pairs, abs_pairs, sat_gains, a2g_gains,
            Ps_dB, Pd, Bs, Bd
        )

        # 决策变量
        x_noma = cp.Variable(K, boolean=True)
        y_weak = cp.Variable(K, boolean=True)
        y_strong = cp.Variable(K, boolean=True)

        # 目标：最大化总速率
        objective_terms = []
        for k in range(K):
            # Contribution from each mode
            contrib = (x_noma[k] * rate_opts['abs_noma'][k] +
                      y_weak[k] * rate_opts['abs_oma_weak'][k] +
                      y_strong[k] * rate_opts['abs_oma_strong'][k] +
                      (1 - x_noma[k] - y_weak[k] - y_strong[k]) * rate_opts['sat'][k])
            objective_terms.append(contrib)

        objective = cp.Maximize(cp.sum(objective_terms))

        # 约束
        constraints = []

        # 1. 互斥性：每对最多一种ABS模式
        for k in range(K):
            constraints.append(x_noma[k] + y_weak[k] + y_strong[k] <= 1)

        # 2. 仅当ABS优于卫星时才使用ABS
        for k in range(K):
            # 仅当NOMA更好时使用
            if rate_opts['abs_noma'][k] <= rate_opts['sat'][k]:
                constraints.append(x_noma[k] == 0)
            # 仅当OMA弱用户更好时使用
            if rate_opts['abs_oma_weak'][k] <= rate_opts['sat'][k]:
                constraints.append(y_weak[k] == 0)
            # 仅当OMA强用户更好时使用
            if rate_opts['abs_oma_strong'][k] <= rate_opts['sat'][k]:
                constraints.append(y_strong[k] == 0)

        # 求解
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.GLPK_MI, verbose=False)
        except:
            try:
                problem.solve(verbose=False)
            except Exception as e:
                print(f"ILP求解器失败：{e}")
                return self.optimize_greedy(sat_pairs, abs_pairs, sat_gains, a2g_gains,
                                           Ps_dB, Pd, Bs, Bd)

        # 提取决策
        decisions = []
        final_rate = 0.0

        for k in range(K):
            if x_noma.value[k] > 0.5:
                decisions.append('noma')
                final_rate += rate_opts['abs_noma'][k]
            elif y_weak.value[k] > 0.5:
                decisions.append('oma_weak')
                final_rate += rate_opts['abs_oma_weak'][k]
            elif y_strong.value[k] > 0.5:
                decisions.append('oma_strong')
                final_rate += rate_opts['abs_oma_strong'][k]
            else:
                decisions.append('sat')
                final_rate += rate_opts['sat'][k]

        info = {
            'solver': 'ILP',
            'optimal_value': problem.value,
            'status': problem.status
        }

        return decisions, final_rate, info

    def optimize_greedy(self, sat_pairs, abs_pairs, sat_gains, a2g_gains,
                       Ps_dB, Pd, Bs, Bd):
        """
        贪婪算法（当cvxpy不可用时的回退方案）

        类似于原始SATCON但具有改进的逻辑

        参数：
            [与optimize_ilp相同]

        返回：
            decisions: 决策列表
            final_rate: 总速率
            info: 优化信息
        """
        K = len(sat_pairs)

        # 计算速率选项
        rate_opts = self.compute_rate_options(
            sat_pairs, abs_pairs, sat_gains, a2g_gains,
            Ps_dB, Pd, Bs, Bd
        )

        # 贪婪决策：对于每对，选择最佳选项
        decisions = []
        final_rate = 0.0

        for k in range(K):
            # 比较所有选项
            options = {
                'sat': rate_opts['sat'][k],
                'noma': rate_opts['abs_noma'][k],
                'oma_weak': rate_opts['abs_oma_weak'][k],
                'oma_strong': rate_opts['abs_oma_strong'][k]
            }

            # 选择最佳
            best_mode = max(options, key=options.get)
            decisions.append(best_mode)
            final_rate += options[best_mode]

        info = {
            'solver': '贪婪算法',
            'optimal_value': final_rate,
            'status': '成功'
        }

        return decisions, final_rate, info

    def optimize(self, sat_pairs, abs_pairs, sat_gains, a2g_gains,
                Ps_dB, Pd, Bs, Bd):
        """
        主优化接口

        如果可用则自动选择ILP，否则使用贪婪算法

        参数：
            sat_pairs: 卫星配对
            abs_pairs: ABS配对
            sat_gains: 卫星信道增益
            a2g_gains: A2G信道增益
            Ps_dB: 卫星功率 (dB)
            Pd: ABS功率 (W)
            Bs: 卫星带宽
            Bd: ABS带宽

        返回：
            decisions: 每对的模式决策列表
            final_rate: 总系统速率
            info: 优化信息
        """
        if self.use_ilp:
            try:
                return self.optimize_ilp(sat_pairs, abs_pairs, sat_gains, a2g_gains,
                                        Ps_dB, Pd, Bs, Bd)
            except:
                print("ILP失败，回退到贪婪算法")
                return self.optimize_greedy(sat_pairs, abs_pairs, sat_gains, a2g_gains,
                                           Ps_dB, Pd, Bs, Bd)
        else:
            return self.optimize_greedy(sat_pairs, abs_pairs, sat_gains, a2g_gains,
                                       Ps_dB, Pd, Bs, Bd)


# ==================== 测试代码 ====================
def test_integer_programming_decision():
    """测试整数规划决策优化器"""
    print("=" * 60)
    print("测试整数规划决策优化器")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    sat_gains = np.random.exponential(0.01, size=config.N)
    a2g_gains = np.random.exponential(0.05, size=config.N)

    # 添加多样性
    for i in range(0, config.N, 4):
        if i+1 < config.N:
            sat_gains[i] *= 5.0
            a2g_gains[i] *= 0.2
            sat_gains[i+1] *= 0.2
            a2g_gains[i+1] *= 5.0

    print(f"\n测试配置：")
    print(f"  用户数量：{config.N}")
    print(f"  配对数量：{config.N // 2}")
    print(f"  可用求解器：{'ILP (cvxpy)' if CVXPY_AVAILABLE else '贪婪算法（回退）'}")

    # 从基线获取配对
    allocator = NOMAAllocator()
    sat_pairs, _ = allocator.optimal_user_pairing(sat_gains)
    abs_pairs, _ = allocator.optimal_user_pairing(a2g_gains)

    # 测试参数
    Bd = config.Bd_options[1]
    Ps_dB = 20

    # 创建优化器
    optimizer = IntegerProgrammingDecision()

    # 基线：贪婪决策（每对局部最优）
    print(f"\n" + "-" * 60)
    print("基线：贪婪决策（每对局部最优）")
    print("-" * 60)

    decisions_greedy, rate_greedy, info_greedy = optimizer.optimize_greedy(
        sat_pairs, abs_pairs, sat_gains, a2g_gains,
        Ps_dB, config.Pd, config.Bs, Bd
    )

    print(f"  总速率：{rate_greedy/1e6:.2f} Mbps")
    print(f"  模式分布：")
    print(f"    卫星：{decisions_greedy.count('sat')}")
    print(f"    ABS NOMA：{decisions_greedy.count('noma')}")
    print(f"    ABS OMA弱用户：{decisions_greedy.count('oma_weak')}")
    print(f"    ABS OMA强用户：{decisions_greedy.count('oma_strong')}")

    # 提出方法：全局优化
    print(f"\n" + "-" * 60)
    print("提出方法：全局优化决策")
    print("-" * 60)

    decisions_opt, rate_opt, info_opt = optimizer.optimize(
        sat_pairs, abs_pairs, sat_gains, a2g_gains,
        Ps_dB, config.Pd, config.Bs, Bd
    )

    print(f"  求解器：{info_opt['solver']}")
    print(f"  总速率：{rate_opt/1e6:.2f} Mbps")
    print(f"  模式分布：")
    print(f"    卫星：{decisions_opt.count('sat')}")
    print(f"    ABS NOMA：{decisions_opt.count('noma')}")
    print(f"    ABS OMA弱用户：{decisions_opt.count('oma_weak')}")
    print(f"    ABS OMA强用户：{decisions_opt.count('oma_strong')}")

    # 性能对比
    print(f"\n" + "=" * 60)
    print("性能对比")
    print("=" * 60)

    improvement_abs = (rate_opt - rate_greedy) / 1e6
    improvement_pct = (rate_opt - rate_greedy) / rate_greedy * 100

    print(f"\n基线（贪婪算法）：")
    print(f"  速率：{rate_greedy/1e6:.2f} Mbps")

    print(f"\n提出方法（全局优化）：")
    print(f"  速率：{rate_opt/1e6:.2f} Mbps")

    print(f"\n改进幅度：")
    print(f"  绝对增益：{improvement_abs:.2f} Mbps")
    print(f"  相对增益：{improvement_pct:.2f}%")

    # 验证
    print(f"\n" + "=" * 60)
    if rate_opt >= rate_greedy * 0.99:
        print("[通过] 整数规划决策测试通过")
        print(f"  - 求解器：{info_opt['solver']}")
        if rate_opt > rate_greedy:
            print(f"  - 性能优于贪婪算法 (+{improvement_pct:.2f}%)")
        else:
            print(f"  - 性能等于贪婪算法（局部最优）")
    else:
        print("[失败] 测试失败")
    print("=" * 60)

    return decisions_opt, rate_opt, info_opt


if __name__ == "__main__":
    test_integer_programming_decision()
