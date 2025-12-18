"""
S2A 资源分配优化器

本模块实现三种 S2A 带宽分配策略：
1. UniformAllocator: 原 baseline 均分策略（Bs/K）
2. KKTAllocator: KKT 最优分配（消除 DF 瓶颈）
3. WaterFillingAllocator: Water-filling 算法（信道自适应）

核心思想：
- 只给 ABS 转发的 pair 分配 S2A 带宽
- 分配量使得 S2A 速率 ≥ A2G 速率（消除 DF 瓶颈）
- 若总需求超过 Bs，按比例缩减
"""

import numpy as np
from abc import ABC, abstractmethod


class S2AAllocator(ABC):
    """S2A 带宽分配器抽象基类"""

    def __init__(self, config_obj):
        """
        初始化分配器

        参数:
            config_obj: 配置对象（来自 config.py）
        """
        self.config = config_obj

    @abstractmethod
    def allocate_bandwidth(self, sat_pairs, modes, a2g_rates_noma, a2g_rates_oma,
                          snr_linear, h_s2a, sat_power_factors):
        """
        分配 S2A 带宽

        参数:
            sat_pairs: 卫星配对 [(weak_idx, strong_idx), ...]
            modes: 选择的模式 ['sat'/'noma'/'oma_weak'/'oma_strong']
            a2g_rates_noma: ABS NOMA的A2G速率（未考虑S2A约束）[N]
            a2g_rates_oma: ABS OMA的A2G速率（未考虑S2A约束）[N]
            snr_linear: 卫星SNR（线性）
            h_s2a: S2A信道增益（标量，ABS到卫星）
            sat_power_factors: 卫星侧功率分配因子 {'beta_strong': [K], 'beta_weak': [K]}

        返回:
            b_allocated: 各 pair 的带宽分配 [K]（单位：Hz）
        """
        pass

    def compute_s2a_rates(self, sat_pairs, b_allocated, snr_linear, h_s2a,
                         sat_power_factors):
        """
        计算 S2A 速率（给定带宽分配）

        参数:
            sat_pairs: 卫星配对
            b_allocated: 带宽分配 [K]
            snr_linear: 卫星SNR（线性）
            h_s2a: S2A信道增益
            sat_power_factors: 功率分配因子

        返回:
            s2a_rates: S2A速率 [N]
        """
        K = len(sat_pairs)
        N = 2 * K
        s2a_rates = np.zeros(N)

        for k in range(K):
            weak_idx, strong_idx = sat_pairs[k]

            # 带宽（可能为0）
            b_k = b_allocated[k]

            if b_k <= 0:
                # 无带宽分配，S2A速率为0（强制使用卫星直达）
                s2a_rates[weak_idx] = 0
                s2a_rates[strong_idx] = 0
                continue

            # 功率分配因子
            beta_strong = sat_power_factors['beta_strong'][k]
            beta_weak = sat_power_factors['beta_weak'][k]

            # S2A NOMA解码速率（类似 satcon_system.py:298-306）
            # 强用户：R_sd_j = b_k * log2(1 + β_j * SNR * h_s2a)
            rate_s2a_strong = b_k * np.log2(
                1 + beta_strong * snr_linear * h_s2a
            )

            # 弱用户：R_sd_i = b_k * log2(1 + β_i * SNR * h_s2a / (β_j * SNR * h_s2a + 1))
            rate_s2a_weak = b_k * np.log2(
                1 + beta_weak * snr_linear * h_s2a /
                (beta_strong * snr_linear * h_s2a + 1)
            )

            s2a_rates[strong_idx] = rate_s2a_strong
            s2a_rates[weak_idx] = rate_s2a_weak

        return s2a_rates


class UniformAllocator(S2AAllocator):
    """
    均分分配器（Baseline）

    策略：固定均分 Bs/K，与模式无关

    问题：
    - SAT 模式的 pair 仍占用 S2A 带宽（浪费）
    - 未考虑 A2G 瓶颈（可能过度分配）
    """

    def allocate_bandwidth(self, sat_pairs, modes, a2g_rates_noma, a2g_rates_oma,
                          snr_linear, h_s2a, sat_power_factors):
        """
        均分分配

        实现：完全复制原 satcon_system.py 的逻辑
        """
        K = len(sat_pairs)
        b_allocated = np.ones(K) * (self.config.Bs / K)
        return b_allocated


class KKTAllocator(S2AAllocator):
    """
    KKT 最优分配器

    策略：
    1. SAT 模式的 pair 不分配 S2A 带宽（b_k = 0）
    2. 其他模式：分配刚好消除 DF 瓶颈的带宽
       - 目标：R_s2a ≈ R_a2g（使 min(R_a2g, R_s2a) = R_a2g）
    3. 若总需求超过 Bs，按比例缩减

    数学推导：
    - NOMA: R_s2a(b_k) = b_k * log2(1 + SNR * h_s2a)
    - 目标: R_s2a(b_k) = R_a2g
    - 反解: b_k = R_a2g / log2(1 + SNR * h_s2a)

    优势：
    - 消除 DF 瓶颈
    - 避免带宽浪费
    - O(K) 复杂度
    """

    def allocate_bandwidth(self, sat_pairs, modes, a2g_rates_noma, a2g_rates_oma,
                          snr_linear, h_s2a, sat_power_factors):
        """
        KKT 最优分配

        核心逻辑：
        1. 根据模式确定目标速率（A2G 瓶颈）
        2. 反解所需带宽
        3. 按比例缩减（若超过 Bs）
        """
        K = len(sat_pairs)
        b_optimal = np.zeros(K)

        # 数值稳定性保护
        eps = 1e-10

        for k in range(K):
            mode = modes[k]

            # 规则1：SAT 模式不需要 S2A 带宽
            if mode == 'sat':
                b_optimal[k] = 0
                continue

            weak_idx, strong_idx = sat_pairs[k]

            # 功率分配因子
            beta_strong = sat_power_factors['beta_strong'][k]
            beta_weak = sat_power_factors['beta_weak'][k]

            # 计算有效 SNR
            snr_eff = snr_linear * h_s2a

            # 规则2：根据模式确定瓶颈速率
            if mode == 'noma':
                # NOMA：两用户都需要满足
                # 弱用户的有效 SNR（SIC 后）
                snr_weak_eff = beta_weak * snr_eff / (1 + beta_strong * snr_eff)

                # 强用户的有效 SNR
                snr_strong_eff = beta_strong * snr_eff

                # 目标速率
                R_target_weak = a2g_rates_noma[weak_idx]
                R_target_strong = a2g_rates_noma[strong_idx]

                # 反解所需带宽（取二者中更严格的约束）
                # b_k = R / log2(1 + SNR_eff)
                if snr_weak_eff > eps:
                    b_weak = R_target_weak / np.log2(1 + snr_weak_eff)
                else:
                    b_weak = 1e12  # 极大值（表示无法满足）

                if snr_strong_eff > eps:
                    b_strong = R_target_strong / np.log2(1 + snr_strong_eff)
                else:
                    b_strong = 1e12

                # 取最严格约束（最大带宽需求）
                b_optimal[k] = max(b_weak, b_strong)

            elif mode == 'oma_weak':
                # OMA 弱用户：ABS只转发弱用户，但S2A仍是NOMA传输
                # 需要满足S2A对弱用户的解码速率
                R_target = a2g_rates_oma[weak_idx]

                # S2A 弱用户有效 SNR（NOMA SIC）
                snr_oma_eff = beta_weak * snr_eff / (1 + beta_strong * snr_eff)

                if snr_oma_eff > eps:
                    b_optimal[k] = R_target / np.log2(1 + snr_oma_eff)
                else:
                    b_optimal[k] = 1e12

            elif mode == 'oma_strong':
                # OMA 强用户：ABS只转发强用户，但S2A仍是NOMA传输
                # 需要满足S2A对强用户的解码速率
                R_target = a2g_rates_oma[strong_idx]

                # S2A 强用户有效 SNR（NOMA直接解码）
                snr_oma_eff = beta_strong * snr_eff

                if snr_oma_eff > eps:
                    b_optimal[k] = R_target / np.log2(1 + snr_oma_eff)
                else:
                    b_optimal[k] = 1e12

            # 保护：限制最大值
            if b_optimal[k] > 1e10:
                b_optimal[k] = self.config.Bs / K  # 回退到均分

        # 规则3：若总需求超过 Bs，按比例缩减
        total_demand = np.sum(b_optimal)
        if total_demand > self.config.Bs:
            scaling_factor = self.config.Bs / total_demand
            b_optimal *= scaling_factor

        return b_optimal


class WaterFillingAllocator(S2AAllocator):
    """
    Water-filling 分配器

    策略：
    1. SAT 模式的 pair 不分配带宽
    2. 其他模式：优先分配给"信道增益×目标速率"最大的 pair
    3. 迭代分配直到耗尽 Bs

    适用场景：
    - S2A 信道增益差异大（仰角不同、遮挡等）
    - 需要考虑边际效应递减

    算法步骤：
    1. 计算各 pair 的"优先级"（速率/带宽斜率）
    2. 按优先级降序分配
    3. 动态更新优先级（边际效应）

    复杂度：O(K log K)（排序）
    """

    def allocate_bandwidth(self, sat_pairs, modes, a2g_rates_noma, a2g_rates_oma,
                          snr_linear, h_s2a, sat_power_factors):
        """
        Water-filling 分配

        简化版本：
        1. 计算各 pair 的初始斜率（速率/带宽）
        2. 按斜率分配（优先满足斜率大的）
        """
        K = len(sat_pairs)
        b_allocated = np.zeros(K)

        # 计算各 pair 的斜率（速率/带宽比）
        slopes = np.zeros(K)

        for k in range(K):
            mode = modes[k]

            if mode == 'sat':
                slopes[k] = 0  # 不参与分配
                continue

            weak_idx, strong_idx = sat_pairs[k]

            # 功率分配因子
            beta_strong = sat_power_factors['beta_strong'][k]
            beta_weak = sat_power_factors['beta_weak'][k]

            # 有效 SNR
            snr_eff = snr_linear * h_s2a

            # 计算斜率（初始速率/带宽比）
            # 对于 NOMA：取弱用户的斜率（通常是瓶颈）
            if mode == 'noma':
                snr_weak_eff = beta_weak * snr_eff / (1 + beta_strong * snr_eff)
                slopes[k] = np.log2(1 + snr_weak_eff) if snr_weak_eff > 0 else 0

            else:  # OMA
                slopes[k] = np.log2(1 + snr_eff) if snr_eff > 0 else 0

        # 按斜率降序排序
        sorted_indices = np.argsort(slopes)[::-1]

        # 分配带宽
        remaining_bandwidth = self.config.Bs

        for k in sorted_indices:
            if slopes[k] <= 0:
                break  # 跳过不需要分配的 pair

            mode = modes[k]
            weak_idx, strong_idx = sat_pairs[k]

            # 计算理想分配量（类似 KKT）
            if mode == 'noma':
                R_target = max(a2g_rates_noma[weak_idx], a2g_rates_noma[strong_idx])
            elif mode == 'oma_weak':
                R_target = a2g_rates_oma[weak_idx]
            else:  # oma_strong
                R_target = a2g_rates_oma[strong_idx]

            # 所需带宽
            b_needed = R_target / slopes[k] if slopes[k] > 0 else 0

            # 分配（取可用带宽和需求的最小值）
            b_allocated[k] = min(b_needed, remaining_bandwidth)
            remaining_bandwidth -= b_allocated[k]

            if remaining_bandwidth <= 0:
                break

        return b_allocated


# ==================== 辅助函数 ====================

def compare_allocators(sat_pairs, modes, a2g_rates_noma, a2g_rates_oma,
                      snr_linear, h_s2a, sat_power_factors, config_obj):
    """
    对比三种分配器的性能

    用于调试和验证

    返回:
        dict: 各分配器的结果
    """
    allocators = {
        'Uniform': UniformAllocator(config_obj),
        'KKT': KKTAllocator(config_obj),
        'WaterFilling': WaterFillingAllocator(config_obj)
    }

    results = {}
    for name, allocator in allocators.items():
        b_allocated = allocator.allocate_bandwidth(
            sat_pairs, modes, a2g_rates_noma, a2g_rates_oma,
            snr_linear, h_s2a, sat_power_factors
        )

        # 计算 S2A 速率
        s2a_rates = allocator.compute_s2a_rates(
            sat_pairs, b_allocated, snr_linear, h_s2a, sat_power_factors
        )

        results[name] = {
            'b_allocated': b_allocated,
            's2a_rates': s2a_rates,
            'total_bandwidth': np.sum(b_allocated)
        }

    return results


# ==================== 单元测试 ====================

if __name__ == "__main__":
    """简单测试"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import config

    print("=" * 60)
    print("资源分配器单元测试")
    print("=" * 60)

    # 构造测试数据（K=2 对）
    sat_pairs = [(0, 1), (2, 3)]
    modes = ['noma', 'oma_weak']  # 第1对NOMA，第2对OMA弱用户

    # A2G 速率（假设值）
    a2g_rates_noma = np.array([1e6, 2e6, 1.5e6, 2.5e6])  # bits/s
    a2g_rates_oma = np.array([1.8e6, 3e6, 2e6, 3.5e6])

    # S2A 参数
    snr_db = 20
    snr_linear = 10 ** (snr_db / 10)
    h_s2a = 0.5  # 假设信道增益

    # 功率分配因子
    sat_power_factors = {
        'beta_strong': np.array([0.7, 0.6]),
        'beta_weak': np.array([0.3, 0.4])
    }

    print(f"\n输入数据:")
    print(f"  Pairs: {sat_pairs}")
    print(f"  Modes: {modes}")
    print(f"  SNR: {snr_db} dB")
    print(f"  h_s2a: {h_s2a}")
    print(f"  Total Bs: {config.Bs/1e6:.1f} MHz")

    # 对比三种分配器
    results = compare_allocators(
        sat_pairs, modes, a2g_rates_noma, a2g_rates_oma,
        snr_linear, h_s2a, sat_power_factors, config
    )

    print(f"\n结果对比:")
    for name, result in results.items():
        print(f"\n  {name}:")
        print(f"    Bandwidth allocation: {result['b_allocated']/1e6} MHz")
        print(f"    Total bandwidth: {result['total_bandwidth']/1e6:.2f} MHz "
              f"(<= {config.Bs/1e6:.1f} MHz)")
        print(f"    S2A rates: {result['s2a_rates']/1e6} Mbps")

    # 验证约束
    print(f"\n验证:")
    for name, result in results.items():
        total = result['total_bandwidth']
        is_valid = total <= config.Bs + 1e-6
        print(f"  {name}: 总带宽 <= Bs: {is_valid} "
              f"({total/1e6:.2f} <= {config.Bs/1e6:.1f} MHz)")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
