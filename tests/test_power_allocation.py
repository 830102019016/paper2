"""
测试功率分配模块

运行方法：
pytest tests/test_power_allocation.py -v
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.power_allocation import NOMAAllocator
from config import config


class TestNOMAAllocator:
    """测试NOMA功率分配器"""
    
    def test_power_factor_sum(self):
        """测试功率因子和为1"""
        allocator = NOMAAllocator()
        gamma_w, gamma_s = 0.01, 0.1
        Ps = 1.0
        
        beta_s, beta_w = allocator.compute_power_factors(gamma_s, gamma_w, Ps)
        
        assert abs((beta_s + beta_w) - 1.0) < 1e-6, "功率因子之和应为1"
    
    def test_weak_user_more_power(self):
        """测试弱用户分配更多功率"""
        allocator = NOMAAllocator()
        gamma_w, gamma_s = 0.01, 0.1  # 10倍差距
        Ps = 1.0
        
        beta_s, beta_w = allocator.compute_power_factors(gamma_s, gamma_w, Ps)
        
        assert beta_w > beta_s, "弱用户应分配更多功率"
        assert beta_w > 0.5, "弱用户功率应大于50%"
    
    def test_power_factor_range(self):
        """测试功率因子在有效范围"""
        allocator = NOMAAllocator()
        gamma_w, gamma_s = 0.01, 0.1
        Ps = 1.0
        
        beta_s, beta_w = allocator.compute_power_factors(gamma_s, gamma_w, Ps)
        
        assert 0 <= beta_s <= 1, "beta_s应在[0,1]范围"
        assert 0 <= beta_w <= 1, "beta_w应在[0,1]范围"
    
    def test_pairing_correctness(self):
        """测试用户配对正确性"""
        allocator = NOMAAllocator()
        # 生成随机信道增益
        np.random.seed(42)
        gains = np.random.exponential(0.05, size=config.N)
        
        pairs, paired_gains = allocator.optimal_user_pairing(gains)
        
        # 检查配对数
        assert len(pairs) == config.K, f"配对数应为{config.K}"
        
        # 检查每对中强用户增益大于弱用户
        for k in range(config.K):
            gamma_w, gamma_s = paired_gains[k]
            assert gamma_s >= gamma_w, f"配对{k}: 强用户增益应≥弱用户"
    
    def test_pairing_no_duplicates(self):
        """测试配对中无重复用户"""
        allocator = NOMAAllocator()
        np.random.seed(42)
        gains = np.random.exponential(0.05, size=config.N)
        
        pairs, _ = allocator.optimal_user_pairing(gains)
        
        all_users = pairs.flatten()
        unique_users = np.unique(all_users)
        
        assert len(all_users) == len(unique_users), "存在重复用户"
        assert len(unique_users) == config.N, "未包含所有用户"
    
    def test_pairing_validation(self):
        """测试配对验证功能"""
        allocator = NOMAAllocator()
        np.random.seed(42)
        gains = np.random.exponential(0.05, size=config.N)
        
        pairs, paired_gains = allocator.optimal_user_pairing(gains)
        is_valid, msg = allocator.validate_pairing(pairs, paired_gains)
        
        assert is_valid, f"配对应该有效，但: {msg}"
    
    def test_edge_case_equal_gains(self):
        """测试边界情况：相等增益"""
        allocator = NOMAAllocator()
        gamma = 0.05
        Ps = 1.0
        
        beta_s, beta_w = allocator.compute_power_factors(gamma, gamma, Ps)
        
        # 相等增益时，功率分配应接近均等
        assert abs(beta_s - 0.5) < 0.1, "相等增益应接近均等分配"
        assert abs(beta_w - 0.5) < 0.1, "相等增益应接近均等分配"


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行功率分配单元测试")
    print("=" * 60)
    
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_all_tests()
