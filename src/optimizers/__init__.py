"""
SATCON 优化器模块

本模块提供 SATCON 系统的优化算法实现：
1. 模式选择优化器（mode_selector.py）
2. S2A 资源分配优化器（resource_allocator.py）

使用示例：
    from src.optimizers.mode_selector import ExhaustiveSelector, GreedySelector
    from src.optimizers.resource_allocator import KKTAllocator

    mode_selector = ExhaustiveSelector()
    s2a_allocator = KKTAllocator(config)
"""

from .mode_selector import (
    ModeSelector,
    HeuristicSelector,
    ExhaustiveSelector,
    GreedySelector
)

from .resource_allocator import (
    S2AAllocator,
    UniformAllocator,
    KKTAllocator,
    WaterFillingAllocator
)

__all__ = [
    'ModeSelector',
    'HeuristicSelector',
    'ExhaustiveSelector',
    'GreedySelector',
    'S2AAllocator',
    'UniformAllocator',
    'KKTAllocator',
    'WaterFillingAllocator'
]
