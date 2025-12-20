"""
传输模式选择优化器

本模块实现三种模式选择策略：
1. HeuristicSelector: 原 baseline 启发式规则（4条if-else）
2. ExhaustiveSelector: 穷举搜索（全局最优）
3. GreedySelector: 贪心算法（快速近似）

传输模式：
- 'sat': 卫星直达（两用户都不通过ABS）
- 'noma': ABS NOMA转发（两用户都通过ABS NOMA）
- 'oma_weak': ABS OMA转发弱用户，强用户走卫星
- 'oma_strong': ABS OMA转发强用户，弱用户走卫星
"""

import numpy as np
from abc import ABC, abstractmethod
from itertools import product


class ModeSelector(ABC):
    """模式选择器抽象基类"""

    @abstractmethod
    def select_modes(self, sat_rates, abs_noma_rates, abs_oma_rates, abs_pairs):
        """
        选择传输模式

        参数:
            sat_rates: 卫星直达速率 [N]
            abs_noma_rates: ABS NOMA速率 [N]（已含S2A约束）
            abs_oma_rates: ABS OMA速率 [N]（已含S2A约束）
            abs_pairs: ABS配对 [(weak_idx, strong_idx), ...]
                      注：基于A2G信道增益(Γ^d)配对，与SAT配对可能不同

        返回:
            final_rates: 最终速率 [N]
            modes: 选择的模式列表 ['sat'/'noma'/'oma_weak'/'oma_strong']
        """
        pass


class HeuristicSelector(ModeSelector):
    """
    启发式模式选择器（Baseline）

    实现原 SATCON 论文的 4 条启发式规则：
    1. 若两用户都通过 ABS NOMA 更好 → NOMA
    2. 若只有弱用户通过 ABS OMA 更好 → OMA-W
    3. 若只有强用户通过 ABS OMA 更好 → OMA-S
    4. 否则 → SAT（卫星直达）

    问题：
    - 规则顺序依赖（先检查NOMA再检查OMA）
    - 无法保证局部或全局最优
    """

    def select_modes(self, sat_rates, abs_noma_rates, abs_oma_rates, abs_pairs):
        """
        启发式规则决策

        实现逻辑：完全复制原 satcon_system.py 的 hybrid_decision 方法

        注：虽然原方法基于SAT-pairs，但根据路线A，应基于ABS-pairs决策
             因为NOMA/OMA转发是ABS行为，资源分配基于ABS配对
        """
        final_rates = sat_rates.copy()
        modes = []

        for k, (weak_idx, strong_idx) in enumerate(abs_pairs):
            R_s_i = sat_rates[weak_idx]
            R_s_j = sat_rates[strong_idx]
            R_dn_i = abs_noma_rates[weak_idx]
            R_dn_j = abs_noma_rates[strong_idx]
            R_do_i = abs_oma_rates[weak_idx]
            R_do_j = abs_oma_rates[strong_idx]

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


class ExhaustiveSelector(ModeSelector):
    """
    穷举搜索模式选择器（全局最优）

    算法：
    1. 枚举所有可能的模式组合（4^K 种）
    2. 计算每种组合的总速率
    3. 选择总速率最大的组合

    复杂度：O(4^K)
    - K=4: 256 种组合（可接受）
    - K=5: 1024 种组合（仍可接受）
    - K>6: 建议使用贪心算法

    优势：
    - 保证全局最优
    - 提供性能上界（用于对比其他算法）
    """

    def select_modes(self, sat_rates, abs_noma_rates, abs_oma_rates, abs_pairs):
        """
        穷举搜索全局最优模式

        实现细节：
        1. 使用 itertools.product 生成所有组合
        2. 对每种组合计算总速率
        3. 记录最优解

        注：基于ABS-pairs搜索，因为NOMA/OMA是ABS的物理行为
        """
        K = len(abs_pairs)
        N = len(sat_rates)

        # 所有可能的模式
        modes_options = ['sat', 'noma', 'oma_weak', 'oma_strong']

        best_sum_rate = 0
        best_modes = None
        best_rates = None

        # 穷举所有模式组合（4^K 种）
        total_combinations = 4 ** K

        # 【保护机制】如果组合数过多（K > 6），降级为贪心
        if K > 6:
            if not hasattr(self, '_downgraded'):
                print(f"  [ExhaustiveSelector] K={K} 过大，自动降级为贪心算法")
                print(f"  (穷举需要 {total_combinations:,} 种组合，不可行)")
                self._downgraded = True

            # 降级为贪心
            return GreedySelector().select_modes(
                sat_rates, abs_noma_rates, abs_oma_rates, abs_pairs
            )

        # 只在组合数较多时显示警告
        if total_combinations > 1000 and not hasattr(self, '_warned'):
            print(f"  [ExhaustiveSelector] 穷举 {total_combinations} 种组合 (K={K})...")
            self._warned = True

        for mode_combo in product(modes_options, repeat=K):
            # 计算该组合下的速率
            rates = np.zeros(N)

            for k, (weak_idx, strong_idx) in enumerate(abs_pairs):
                mode = mode_combo[k]

                if mode == 'sat':
                    # 卫星直达
                    rates[weak_idx] = sat_rates[weak_idx]
                    rates[strong_idx] = sat_rates[strong_idx]

                elif mode == 'noma':
                    # ABS NOMA转发
                    rates[weak_idx] = abs_noma_rates[weak_idx]
                    rates[strong_idx] = abs_noma_rates[strong_idx]

                elif mode == 'oma_weak':
                    # OMA只给弱用户
                    rates[weak_idx] = abs_oma_rates[weak_idx]
                    rates[strong_idx] = sat_rates[strong_idx]

                elif mode == 'oma_strong':
                    # OMA只给强用户
                    rates[weak_idx] = sat_rates[weak_idx]
                    rates[strong_idx] = abs_oma_rates[strong_idx]

            # 计算总速率
            total_rate = np.sum(rates)

            # 更新最优解
            if total_rate > best_sum_rate:
                best_sum_rate = total_rate
                best_modes = list(mode_combo)
                best_rates = rates.copy()

        return best_rates, best_modes


class GreedySelector(ModeSelector):
    """
    贪心模式选择器（快速近似）

    算法：
    1. 对每个 pair，独立计算 4 种模式的收益（pair总速率）
    2. 选择收益最大的模式

    复杂度：O(K)
    - 每个 pair 只需 4 次比较
    - 远快于穷举（适合实时系统）

    性能：
    - 通常接近全局最优（pair 间耦合较弱时）
    - 最坏情况：可能次优（当 pair 间资源竞争激烈时）

    优势：
    - 低复杂度（线性时间）
    - 易于理解和实现
    - 适合实时决策
    """

    def select_modes(self, sat_rates, abs_noma_rates, abs_oma_rates, abs_pairs):
        """
        贪心选择：逐对选择最优模式

        实现细节：
        1. 计算每个 ABS-pair 在 4 种模式下的总速率
        2. 选择总速率最大的模式

        注：基于ABS-pairs进行贪心决策，每个pair独立选择最优模式
             这是物理可实现的，因为ABS的NOMA/OMA资源分配基于ABS配对
        """
        final_rates = sat_rates.copy()
        modes = []

        for k, (weak_idx, strong_idx) in enumerate(abs_pairs):
            # 计算各模式的收益（pair总速率）
            gains = {
                'sat': sat_rates[weak_idx] + sat_rates[strong_idx],
                'noma': abs_noma_rates[weak_idx] + abs_noma_rates[strong_idx],
                'oma_weak': abs_oma_rates[weak_idx] + sat_rates[strong_idx],
                'oma_strong': sat_rates[weak_idx] + abs_oma_rates[strong_idx]
            }

            # 选择最优模式
            best_mode = max(gains, key=gains.get)
            modes.append(best_mode)

            # 更新速率
            if best_mode == 'noma':
                final_rates[weak_idx] = abs_noma_rates[weak_idx]
                final_rates[strong_idx] = abs_noma_rates[strong_idx]

            elif best_mode == 'oma_weak':
                final_rates[weak_idx] = abs_oma_rates[weak_idx]
                # 强用户保持卫星速率（已在初始化时设置）

            elif best_mode == 'oma_strong':
                # 弱用户保持卫星速率（已在初始化时设置）
                final_rates[strong_idx] = abs_oma_rates[strong_idx]

            # 'sat' 模式不需要修改（已在初始化时设置）

        return final_rates, modes


# ==================== 辅助函数 ====================

def compare_selectors(sat_rates, abs_noma_rates, abs_oma_rates, abs_pairs):
    """
    对比三种选择器的性能

    用于调试和验证

    参数:
        abs_pairs: ABS配对（基于A2G信道增益）

    返回:
        dict: 各选择器的结果
    """
    selectors = {
        'Heuristic': HeuristicSelector(),
        'Exhaustive': ExhaustiveSelector(),
        'Greedy': GreedySelector()
    }

    results = {}
    for name, selector in selectors.items():
        final_rates, modes = selector.select_modes(
            sat_rates, abs_noma_rates, abs_oma_rates, abs_pairs
        )
        results[name] = {
            'sum_rate': np.sum(final_rates),
            'modes': modes,
            'rates': final_rates
        }

    return results


# ==================== 单元测试 ====================

if __name__ == "__main__":
    """简单测试"""
    print("=" * 60)
    print("模式选择器单元测试")
    print("=" * 60)

    # 构造测试数据（K=2 对，N=4 用户）
    abs_pairs = [(0, 1), (2, 3)]  # ABS配对（基于A2G信道增益）

    # 假设速率（bits/s）
    sat_rates = np.array([1.0, 2.0, 1.5, 2.5])  # 卫星直达
    abs_noma_rates = np.array([1.2, 2.3, 1.3, 2.8])  # ABS NOMA
    abs_oma_rates = np.array([1.8, 3.0, 2.0, 3.5])  # ABS OMA

    print(f"\n输入数据:")
    print(f"  ABS Pairs: {abs_pairs}")
    print(f"  SAT rates: {sat_rates}")
    print(f"  NOMA rates: {abs_noma_rates}")
    print(f"  OMA rates: {abs_oma_rates}")

    # 对比三种选择器
    results = compare_selectors(sat_rates, abs_noma_rates, abs_oma_rates, abs_pairs)

    print(f"\n结果对比:")
    for name, result in results.items():
        print(f"\n  {name}:")
        print(f"    Sum rate: {result['sum_rate']:.2f} bits/s")
        print(f"    Modes: {result['modes']}")
        print(f"    Final rates: {result['rates']}")

    # 验证穷举 >= 其他方法
    exhaustive_rate = results['Exhaustive']['sum_rate']
    heuristic_rate = results['Heuristic']['sum_rate']
    greedy_rate = results['Greedy']['sum_rate']

    print(f"\n验证:")
    print(f"  穷举 >= 启发式: {exhaustive_rate >= heuristic_rate - 1e-6} "
          f"({exhaustive_rate:.2f} >= {heuristic_rate:.2f})")
    print(f"  穷举 >= 贪心: {exhaustive_rate >= greedy_rate - 1e-6} "
          f"({exhaustive_rate:.2f} >= {greedy_rate:.2f})")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
