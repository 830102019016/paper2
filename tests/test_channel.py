"""
测试信道模型模块

运行方法：
pytest tests/test_channel.py -v
或
python tests/test_channel.py
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.channel_models import LooChannel, PathLossModel
from config import config


class TestLooChannel:
    """测试Loo信道模型"""
    
    def test_channel_shape(self):
        """测试输出形状"""
        loo = LooChannel(config.alpha_dB, config.psi_dB, config.MP_dB, seed=42)
        gains = loo.generate_channel_gain(100)
        assert gains.shape == (100,), "信道增益形状错误"
    
    def test_channel_positive(self):
        """测试信道增益为正"""
        loo = LooChannel(config.alpha_dB, config.psi_dB, config.MP_dB, seed=42)
        gains = loo.generate_channel_gain(100)
        assert np.all(gains > 0), "信道增益必须为正值"
    
    def test_channel_statistics(self):
        """测试信道统计特性"""
        loo = LooChannel(-15, 3, -10, seed=42)
        gains = loo.generate_channel_gain(10000)
        
        mean_gain = np.mean(gains)
        # 检查均值在合理范围（经验值）
        assert 0.0001 < mean_gain < 1.0, f"均值异常: {mean_gain}"
    
    def test_strong_weak_separation(self):
        """测试强弱信道分离"""
        loo = LooChannel(-15, 3, -10, seed=42)
        strong, weak = loo.generate_strong_weak_channels(100)
        
        assert len(strong) == 100, "强信道用户数错误"
        assert len(weak) == 100, "弱信道用户数错误"
        assert np.mean(strong) > np.mean(weak), "强信道应该优于弱信道"
    
    def test_reproducibility(self):
        """测试可复现性"""
        loo1 = LooChannel(config.alpha_dB, config.psi_dB, config.MP_dB, seed=42)
        loo2 = LooChannel(config.alpha_dB, config.psi_dB, config.MP_dB, seed=42)
        
        gains1 = loo1.generate_channel_gain(50)
        gains2 = loo2.generate_channel_gain(50)
        
        assert np.allclose(gains1, gains2), "相同种子应产生相同结果"


class TestPathLossModel:
    """测试路径损耗模型"""
    
    def test_free_space_loss(self):
        """测试自由空间损耗计算"""
        dist = 1000e3  # 1000 km
        loss = PathLossModel.free_space_loss(dist, config.fs)
        loss_db = 10 * np.log10(loss)
        
        # FSL应该很大（>100 dB）
        assert loss_db > 100, f"路径损耗异常: {loss_db:.1f} dB"
        assert loss_db < 200, f"路径损耗过大: {loss_db:.1f} dB"
    
    def test_satellite_distance(self):
        """测试卫星距离计算"""
        # 测试不同仰角
        elevations = [10, 30, 90]
        distances = [PathLossModel.satellite_distance(e) for e in elevations]
        
        # 距离应该随仰角增加而减小
        assert distances[0] > distances[1] > distances[2], "距离应随仰角递减"
        
        # 90度仰角距离应约等于卫星高度
        assert abs(distances[2] - config.satellite_altitude) < 1000, \
            "天顶距离应约等于卫星高度"
    
    def test_loss_increases_with_distance(self):
        """测试损耗随距离增加"""
        dist1 = 500e3
        dist2 = 1000e3
        
        loss1 = PathLossModel.free_space_loss(dist1, config.fs)
        loss2 = PathLossModel.free_space_loss(dist2, config.fs)
        
        assert loss2 > loss1, "距离增加，损耗应增加"


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行信道模型单元测试")
    print("=" * 60)
    
    # 运行pytest
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_all_tests()