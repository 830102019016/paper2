"""
测试NOMA传输系统

运行方法：
pytest tests/test_noma.py -v
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.noma_transmission import SatelliteNOMA
from config import config


class TestSatelliteNOMA:
    """测试卫星NOMA传输系统"""
    
    @pytest.fixture
    def sat_noma(self):
        """创建测试用的NOMA系统"""
        return SatelliteNOMA(config)
    
    def test_channel_gains_positive(self, sat_noma):
        """测试信道增益为正"""
        gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg=10)
        assert np.all(gains > 0), "信道增益必须为正"
    
    def test_channel_gains_shape(self, sat_noma):
        """测试信道增益形状"""
        gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg=10)
        assert gains.shape == (config.N,), f"增益形状应为({config.N},)"
    
    def test_rates_positive(self, sat_noma):
        """测试速率为正"""
        gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg=10)
        Ps = config.snr_to_power(20)
        rates, sum_rate, _ = sat_noma.compute_achievable_rates(gains, Ps)

        assert np.all(rates >= 0), "速率必须非负"
        assert sum_rate >= 0, "总速率必须非负"

    def test_sum_rate_equals_sum_of_rates(self, sat_noma):
        """测试总速率等于各用户速率之和"""
        gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg=10)
        Ps = config.snr_to_power(20)
        rates, sum_rate, _ = sat_noma.compute_achievable_rates(gains, Ps)

        calculated_sum = np.sum(rates)
        assert abs(sum_rate - calculated_sum) < 1e-6, "总速率应等于用户速率之和"

    def test_rate_increases_with_snr(self, sat_noma):
        """测试速率随SNR增加"""
        gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg=10)

        Ps1 = config.snr_to_power(10)
        Ps2 = config.snr_to_power(20)

        _, sum_rate1, _ = sat_noma.compute_achievable_rates(gains, Ps1)
        _, sum_rate2, _ = sat_noma.compute_achievable_rates(gains, Ps2)

        assert sum_rate2 > sum_rate1, "高SNR应产生更高速率"

    def test_spectral_efficiency_range(self, sat_noma):
        """测试频谱效率在合理范围"""
        gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg=10)
        Ps = config.snr_to_power(20)
        _, sum_rate, _ = sat_noma.compute_achievable_rates(gains, Ps)
        
        se = sum_rate / config.Bs
        # 频谱效率应在合理范围（0-20 bits/s/Hz）
        assert 0 < se < 20, f"频谱效率异常: {se:.2f} bits/s/Hz"
    
    def test_simulation_output_shape(self, sat_noma):
        """测试仿真输出形状"""
        test_snr = np.array([0, 10, 20])
        mean_rates, mean_se, std_rates = sat_noma.simulate_performance(
            snr_db_range=test_snr,
            elevation_deg=10,
            n_realizations=5,
            verbose=False
        )
        
        assert len(mean_rates) == len(test_snr), "输出长度应匹配SNR点数"
        assert len(mean_se) == len(test_snr), "频谱效率长度应匹配"
        assert len(std_rates) == len(test_snr), "标准差长度应匹配"
    
    def test_simulation_monotonicity(self, sat_noma):
        """测试仿真结果单调性"""
        test_snr = np.array([0, 10, 20, 30])
        mean_rates, mean_se, _ = sat_noma.simulate_performance(
            snr_db_range=test_snr,
            elevation_deg=10,
            n_realizations=10,
            verbose=False
        )
        
        # 频谱效率应随SNR递增（允许小幅波动）
        diffs = np.diff(mean_se)
        assert np.sum(diffs > 0) >= len(diffs) - 1, "频谱效率应大致递增"


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行NOMA传输单元测试")
    print("=" * 60)
    
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_all_tests()
