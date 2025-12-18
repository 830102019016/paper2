# ABS位置连续优化 - 快速使用指南

## 📋 目录

1. [快速开始](#快速开始)
2. [推荐配置](#推荐配置)
3. [实验脚本](#实验脚本)
4. [常见问题](#常见问题)

---

## 🚀 快速开始

### 1. 运行单点测试（Phase 1验证）

```bash
python experiments/test_position_optimization.py
```

**输出**:
- 结果JSON: `results/position_optimization_test.json`
- 对比图表: `results/position_optimization_comparison.png`

---

## ⚙️ 推荐配置

### Baseline配置（原方案，无位置优化）

```python
from src.satcon_system import SATCONSystem
from config import config

satcon_baseline = SATCONSystem(
    config,
    abs_bandwidth=1.2e6,
    mode_selector='heuristic',   # Baseline模式选择
    s2a_allocator='uniform',     # Baseline资源分配
    position_optimizer=None      # 使用k-means几何聚类
)
```

### 推荐配置（一阶段 + 二阶段优化）

```python
satcon_optimized = SATCONSystem(
    config,
    abs_bandwidth=1.2e6,
    mode_selector='greedy',      # ✓ 一阶段：贪心模式选择
    s2a_allocator='kkt',         # ✓ 一阶段：KKT资源分配
    position_optimizer='hybrid'  # ✓ 二阶段：混合位置优化
)
```

### 仿真运行

```python
import numpy as np

# 单次仿真
sum_rate, mode_stats = satcon_optimized.simulate_single_realization(
    snr_db=20,
    elevation_deg=10,
    seed=42
)

# Monte Carlo仿真
mean_rates, mean_se, std_rates, mode_stats = satcon_optimized.simulate_performance(
    snr_db_range=np.arange(0, 31, 2),
    elevation_deg=10,
    n_realizations=50,
    verbose=True
)
```

---

## 📊 实验脚本

### 1. Phase 1: 单点快速验证

**脚本**: `experiments/test_position_optimization.py`

**配置**:
- SNR: 20 dB（单点）
- 实现次数: 10次
- 对比: Baseline vs Hybrid

**运行时间**: ~3分钟

**预期结果**: +0.5-2% 改进

---

### 2. Phase 2: 多SNR性能对比（推荐下一步）

**脚本**: 需要创建 `experiments/test_position_optimization_multi_snr.py`

**建议配置**:
```python
snr_db_range = np.arange(0, 32, 2)  # 0-30 dB
n_realizations = 50                  # 增加样本量

# 对比3种方案:
# 1. Baseline: heuristic + uniform + k-means
# 2. One-stage: greedy + kkt + k-means
# 3. Two-stage: greedy + kkt + hybrid
```

**运行时间**: ~30分钟

**预期结果**:
- 绘制频谱效率 vs SNR 曲线
- 定量分析改进幅度

---

## 🔧 参数调整

### 位置优化器参数

在 `src/optimizers/position_optimizer.py` 中：

```python
# 调整L-BFGS-B迭代次数
optimal_pos, optimal_rate, info = optimizer.optimize_position_hybrid(
    ...,
    max_iter=20,  # 默认20次，可增加到50次
    verbose=True  # 打印优化详情
)
```

### 边界约束

在 `config.py` 中：

```python
# ABS位置约束
abs_height_min = 50      # 最小高度 (m)
abs_height_max = 500     # 最大高度 (m)
coverage_radius = 500    # 覆盖半径 (m)
```

---

## ❓ 常见问题

### Q1: 改进效果不明显（<1%）怎么办？

**可能原因**:
1. 样本量太少（建议 n≥50）
2. k-means初始点已经很优
3. L-BFGS-B迭代次数不足

**解决方案**:
```python
# 增加迭代次数
satcon = SATCONSystem(..., position_optimizer='hybrid')

# 在position_optimizer.py中修改max_iter=50
```

### Q2: 如何对比不同优化器组合？

```python
# 方法1: Baseline
satcon_1 = SATCONSystem(
    config, 1.2e6,
    mode_selector='heuristic',
    s2a_allocator='uniform',
    position_optimizer=None
)

# 方法2: 只用一阶段优化
satcon_2 = SATCONSystem(
    config, 1.2e6,
    mode_selector='greedy',
    s2a_allocator='kkt',
    position_optimizer=None
)

# 方法3: 一阶段 + 二阶段（推荐）
satcon_3 = SATCONSystem(
    config, 1.2e6,
    mode_selector='greedy',
    s2a_allocator='kkt',
    position_optimizer='hybrid'
)

# 分别运行仿真并对比
```

### Q3: 如何查看优化过程？

```python
# 启用verbose模式
optimizer = ContinuousPositionOptimizer(config, method='L-BFGS-B')

optimal_pos, optimal_rate, info = optimizer.optimize_position_hybrid(
    ...,
    verbose=True  # 打印优化详情
)

# 查看优化信息
print(f"初始位置: {info['initial_position']}")
print(f"初始速率: {info['initial_rate']/1e6:.2f} Mbps")
print(f"最优位置: {info['optimal_position']}")
print(f"最优速率: {info['optimal_rate']/1e6:.2f} Mbps")
print(f"改进: +{info['improvement_pct']:.2f}%")
print(f"迭代次数: {info['n_iterations']}")
print(f"函数评估次数: {info['n_evaluations']}")
```

### Q4: 计算时间太长怎么办？

**优化建议**:

1. **减少实现次数**（调试时）:
   ```python
   n_realizations = 10  # 快速验证
   ```

2. **减少迭代次数**:
   ```python
   max_iter = 10  # 默认20次
   ```

3. **使用并行化**（未实现，可扩展）

### Q5: 如何保存优化结果？

结果自动保存在 `results/` 目录：

```
results/
├── position_optimization_test.json          # 测试结果（JSON）
├── position_optimization_comparison.png     # 对比图表
└── (其他实验结果...)
```

---

## 📚 详细文档

- **实现总结**: `docs/位置优化实现总结_Phase1.md`
- **设计方案**: `docs/satcon_框架下的算法创新空间总结.md`
- **源代码**: `src/optimizers/position_optimizer.py`

---

## 🎯 下一步

### 立即行动

1. ✅ 运行 Phase 1 测试（已完成）
   ```bash
   python experiments/test_position_optimization.py
   ```

2. 📊 创建 Phase 2 多SNR测试
   - 复制 `test_position_optimization.py`
   - 修改为多SNR点测试
   - 运行并绘制曲线

3. 📈 分析结果
   - 检查改进幅度是否达到预期（+2-5%）
   - 如不满意，调整参数或尝试PSO

### 可选扩展

- **实现PSO优化器**（更强的全局搜索能力）
- **多起点策略**（k-means + 4个角点）
- **自适应迭代次数**（根据收敛情况动态调整）

---

## 📞 支持

如有问题，请参考：
- `docs/位置优化实现总结_Phase1.md`（详细文档）
- 代码注释（每个函数都有详细说明）
- 单元测试（运行查看示例）

---

**版本**: v1.0
**更新**: 2025-12-18
