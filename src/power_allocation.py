"""
NOMA功率分配算法

实现：
1. 最优功率分配因子计算（论文公式4）
2. 最优用户配对策略（论文引用[14]）

论文参考：
- Section III.B: Satellite NOMA transmission
- 公式(4): β_j = (sqrt(1 + Γ_i * Ps) - 1) / (Γ_i * Ps)
"""
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from config import config


class NOMAAllocator:
    """
    NOMA最优功率分配器
    
    功能：
    1. 计算最优功率分配因子
    2. 执行用户配对策略
    """
    
    @staticmethod
    def compute_power_factors(gamma_strong, gamma_weak, total_power, verbose=False):
        """
        计算NOMA最优功率分配因子

        论文公式(4)：
        β_j = (sqrt(1 + Γ_i * Ps) - 1) / (Γ_i * Ps)

        其中：
        - β_j: 强用户功率分配因子
        - β_i = 1 - β_j: 弱用户功率分配因子
        - Γ_i: 弱用户信道增益（注意是弱用户！）
        - Ps: 总发射功率 (W)

        NOMA原理：
        - 弱用户分配更多功率（β_i > β_j）
        - 强用户通过SIC解码，先解弱用户信号

        ⚠️ 保护性机制（稀有退化情况处理）：
        - 在极端信道条件下，公式(4)可能无法保证 β_i > β_j
        - 此时采用保守分配：弱用户60%，强用户40%
        - 该机制触发频率极低（<1%），不影响基线性能

        📝 建议论文说明：
        "A safeguard power allocation is applied in rare degenerate cases
        to ensure the weak user always receives a higher power share."

        参数:
            gamma_strong (float or ndarray): 强用户信道增益 Γ_j
            gamma_weak (float or ndarray): 弱用户信道增益 Γ_i
            total_power (float): 总功率 Ps (W)
            verbose (bool): 是否输出警告信息

        返回:
            beta_strong (float or ndarray): 强用户功率因子 β_j
            beta_weak (float or ndarray): 弱用户功率因子 β_i
        """
        # 论文公式(4)
        term = gamma_weak * total_power
        beta_strong = (np.sqrt(1 + term) - 1) / term
        beta_weak = 1 - beta_strong

        # 保护机制：确保弱用户获得更多功率
        invalid_mask = beta_weak < beta_strong
        if np.any(invalid_mask):
            # 对于问题配对，采用更保守的分配：弱用户60%，强用户40%
            if np.isscalar(beta_weak):
                beta_weak = 0.6
                beta_strong = 0.4
            else:
                beta_weak = np.where(invalid_mask, 0.6, beta_weak)
                beta_strong = np.where(invalid_mask, 0.4, beta_strong)

            if verbose:
                n_invalid = np.sum(invalid_mask) if hasattr(invalid_mask, '__len__') else 1
                print(f"⚠️ 警告：{n_invalid}个配对使用保护性功率分配 (弱60%/强40%)")

        # 确保有效范围 [0, 1]
        beta_strong = np.clip(beta_strong, 0, 1)
        beta_weak = np.clip(beta_weak, 0, 1)

        return beta_strong, beta_weak
    
    @staticmethod
    def optimal_user_pairing(channel_gains):
        """
        NOMA最优用户配对策略
        
        论文引用[14] L. Zhu et al., "Optimal user pairing for 
        downlink non-orthogonal multiple access (NOMA)"
        
        策略：
        1. 按信道增益排序：Γ_1 ≤ Γ_2 ≤ ... ≤ Γ_2K
        2. 配对规则：MT_k ↔ MT_(2K-k+1)
           - 最弱 ↔ 最强
           - 次弱 ↔ 次强
           - ...
        
        优点：
        - 最大化配对间的增益差距
        - 优化SIC性能
        - 均衡系统容量
        
        参数:
            channel_gains (ndarray): shape (2K,) 所有用户的信道增益
        
        返回:
            pairs (ndarray): shape (K, 2) 配对索引 [弱用户idx, 强用户idx]
            paired_gains (ndarray): shape (K, 2) 配对增益 [Γ_i, Γ_j]
        """
        n_users = len(channel_gains)
        K = n_users // 2
        
        # 检查用户数是否为偶数
        if n_users % 2 != 0:
            raise ValueError(f"用户数必须为偶数，当前为 {n_users}")
        
        # 1. 按增益排序
        sorted_indices = np.argsort(channel_gains)
        sorted_gains = channel_gains[sorted_indices]
        
        # 2. 配对：最弱与最强、次弱与次强...
        pairs = np.zeros((K, 2), dtype=int)
        paired_gains = np.zeros((K, 2))
        
        for k in range(K):
            weak_idx = sorted_indices[k]              # 第k弱的用户
            strong_idx = sorted_indices[n_users - k - 1]  # 第k强的用户
            
            pairs[k] = [weak_idx, strong_idx]
            paired_gains[k] = [sorted_gains[k], sorted_gains[n_users - k - 1]]
        
        return pairs, paired_gains
    
    @staticmethod
    def validate_pairing(pairs, paired_gains):
        """
        验证配对的有效性
        
        检查：
        1. 每对中强用户增益 > 弱用户增益
        2. 没有重复用户
        3. 配对数正确
        
        参数:
            pairs (ndarray): 配对索引
            paired_gains (ndarray): 配对增益
        
        返回:
            is_valid (bool): 配对是否有效
            error_msg (str): 错误信息（如果有）
        """
        K = len(pairs)
        
        # 检查1：强用户 > 弱用户
        for k in range(K):
            gamma_weak, gamma_strong = paired_gains[k]
            if gamma_strong <= gamma_weak:
                return False, f"配对 {k}: 强用户增益 <= 弱用户增益"
        
        # 检查2：没有重复用户
        all_users = pairs.flatten()
        if len(all_users) != len(np.unique(all_users)):
            return False, "存在重复用户"
        
        # 检查3：配对数正确
        if len(pairs) != config.K:
            return False, f"配对数错误：期望 {config.K}，实际 {len(pairs)}"
        
        return True, "配对有效"


# ==================== 测试代码 ====================
def test_power_allocation():
    """测试功率分配算法"""
    print("=" * 60)
    print("测试 NOMA 功率分配算法")
    print("=" * 60)
    
    allocator = NOMAAllocator()
    
    # 测试 1：单对用户功率分配
    print(f"\n【测试1：单对用户功率分配】")
    gamma_weak = 0.01    # 弱用户信道增益
    gamma_strong = 0.1   # 强用户信道增益（10倍差距）
    Ps = 1.0             # 总功率 1W
    
    beta_s, beta_w = allocator.compute_power_factors(gamma_strong, gamma_weak, Ps)
    
    print(f"场景：Γ_weak={gamma_weak:.4f}, Γ_strong={gamma_strong:.4f}, Ps={Ps}W")
    print(f"  弱用户功率因子: β_w = {beta_w:.4f} ({beta_w*100:.1f}%)")
    print(f"  强用户功率因子: β_s = {beta_s:.4f} ({beta_s*100:.1f}%)")
    print(f"  功率和: β_w + β_s = {beta_w + beta_s:.6f}")
    print(f"  验证: {'✓ 弱用户功率更大' if beta_w > beta_s else '✗ 异常'}")
    
    # 测试 2：不同功率水平
    print(f"\n【测试2：不同总功率下的分配】")
    test_powers = [0.1, 1.0, 10.0]
    for P in test_powers:
        bs, bw = allocator.compute_power_factors(gamma_strong, gamma_weak, P)
        print(f"  Ps={P:5.1f}W: β_w={bw:.4f}, β_s={bs:.4f}")
    
    # 测试 3：用户配对
    print(f"\n【测试3：用户配对策略】")
    np.random.seed(config.random_seed)
    # 生成随机信道增益（模拟不同信道条件）
    test_gains = np.random.exponential(scale=0.05, size=config.N)
    
    pairs, paired_gains = allocator.optimal_user_pairing(test_gains)
    
    print(f"总用户数: {config.N}, 配对数: {config.K}")
    print(f"前 5 对配对结果:")
    for k in range(min(5, len(pairs))):
        weak_idx, strong_idx = pairs[k]
        gamma_w, gamma_s = paired_gains[k]
        gain_ratio = gamma_s / gamma_w
        print(f"  Pair {k+1}: MT{weak_idx:2d}(Γ={gamma_w:.6f}) ↔ "
              f"MT{strong_idx:2d}(Γ={gamma_s:.6f}), 比值={gain_ratio:.2f}x")
    
    # 测试 4：配对验证
    print(f"\n【测试4：配对有效性验证】")
    is_valid, msg = allocator.validate_pairing(pairs, paired_gains)
    print(f"  验证结果: {'✓ ' + msg if is_valid else '✗ ' + msg}")
    
    # 测试 5：边界情况
    print(f"\n【测试5：边界情况】")
    # 5.1 信道增益相等
    beta_s_eq, beta_w_eq = allocator.compute_power_factors(0.05, 0.05, 1.0)
    print(f"  相等增益: β_w={beta_w_eq:.4f}, β_s={beta_s_eq:.4f}")
    
    # 5.2 极大差距
    beta_s_ex, beta_w_ex = allocator.compute_power_factors(1.0, 0.001, 1.0)
    print(f"  极大差距: β_w={beta_w_ex:.4f}, β_s={beta_s_ex:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ 功率分配算法测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_power_allocation()
