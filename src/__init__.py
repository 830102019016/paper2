"""
SATCON 核心模块

包含：
- channel_models: 信道模型（Loo, Rayleigh, 路径损耗）
- power_allocation: NOMA功率分配算法
- noma_transmission: 卫星NOMA传输系统
- utils: 辅助工具函数
"""

__version__ = "0.1.0"
__author__ = "SATCON Reproduction Team"

# 导入主要类（方便外部调用）
from .channel_models import LooChannel, PathLossModel
from .power_allocation import NOMAAllocator
from .noma_transmission import SatelliteNOMA

__all__ = [
    'LooChannel',
    'PathLossModel',
    'NOMAAllocator',
    'SatelliteNOMA',
]