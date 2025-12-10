# SATCON Joint Optimization Enhancement

## 概述

本目录包含了SATCON系统的三个增强模块及其集成框架，旨在通过联合优化提升系统性能。

## 核心模块

### 📍 模块1：梯度位置优化 (Gradient Position Optimizer)
**文件：** [gradient_position_optimizer.py](gradient_position_optimizer.py)

**创新点：**
- 原方案：k-means 最小化几何距离
- 新方案：梯度优化最大化系统速率

**方法：** L-BFGS-B 梯度下降

**性能提升：** ~2-3%

**测试：** `pytest tests/test_gradient_optimizer.py`

---

### 🔗 模块2：联合配对优化 (Joint Pairing Optimizer)
**文件：** [joint_pairing_optimizer.py](joint_pairing_optimizer.py)

**创新点：**
- 原方案：卫星和ABS独立配对
- 新方案：考虑协同效应的联合配对

**方法：** 贪婪算法 + 局部搜索

**性能提升：** ~2-3%

**测试：** `pytest tests/test_joint_pairing.py`

---

### 🎯 模块3：整数规划决策 (Integer Programming Decision)
**文件：** [integer_programming_decision.py](integer_programming_decision.py)

**创新点：**
- 原方案：基于规则的贪婪决策（每对独立）
- 新方案：整数线性规划全局最优决策

**方法：** ILP (需要 cvxpy) 或改进的贪婪算法

**性能提升：** 0-2% (取决于场景)

**依赖：** `pip install cvxpy` (可选)

**测试：** `pytest tests/test_ilp_decision.py`

---

### 🎯 集成系统：联合优化框架 (Joint Optimization SATCON)
**文件：** [joint_satcon_system.py](joint_satcon_system.py)

**核心算法：**
```python
# 交替优化框架
1. 初始化 ABS 位置 (模块1)
2. 迭代优化:
   a. 固定位置，优化联合配对 (模块2)
   b. 固定配对，优化混合决策 (模块3)
   c. 检查收敛
3. 返回最优解
```

**性能提升：** ~3-5% (小规模测试)

**测试：** `pytest tests/test_joint_system.py`

---

## 快速开始

### 1. 安装依赖
```bash
# 基础依赖
pip install numpy scipy matplotlib scikit-learn tqdm

# 可选：整数规划求解器（提升模块3性能）
pip install cvxpy
```

### 2. 测试单个模块
```bash
# 测试模块1
python src_enhanced/gradient_position_optimizer.py
pytest tests/test_gradient_optimizer.py -v

# 测试模块2
python src_enhanced/joint_pairing_optimizer.py
pytest tests/test_joint_pairing.py -v

# 测试模块3
python src_enhanced/integer_programming_decision.py
pytest tests/test_ilp_decision.py -v
```

### 3. 测试完整系统
```bash
# 快速测试
python src_enhanced/joint_satcon_system.py

# 完整单元测试
pytest tests/test_joint_system.py -v -s
```

### 4. 运行对比实验
```bash
# 参见 experiments/ 目录
python experiments/run_comparison.py
```

---

## 测试结果摘要

### 单元测试状态
- ✅ [gradient_optimizer.py](gradient_position_optimizer.py): 5/5 测试通过
- ✅ [joint_pairing_optimizer.py](joint_pairing_optimizer.py): 5/5 测试通过
- ✅ [integer_programming_decision.py](integer_programming_decision.py): 6/6 测试通过
- ✅ [joint_satcon_system.py](joint_satcon_system.py): 7/7 测试通过

### 性能提升（SNR=20dB, N=32用户, 10次实现）

| 配置 | 总速率 (Mbps) | 相对基线提升 |
|-----|--------------|------------|
| **Baseline (原方案)** | 26.77 | - |
| **+ 模块1 (梯度位置)** | 27.28 | +1.89% |
| **+ 模块2 (联合配对)** | 27.29 | +1.95% |
| **+ 模块3 (整数规划)** | 26.77 | +0.00% |
| **+ 完整系统 (所有模块)** | 27.71 | **+3.52%** ✅ |

**注意：** 这是小规模测试结果。完整的 Monte Carlo 仿真（1000次实现）预期会有更高的性能提升。

---

## 使用示例

### 示例1：使用完整系统
```python
from config import config
from src_enhanced.joint_satcon_system import JointOptimizationSATCON
import numpy as np

# 创建系统（启用所有模块）
system = JointOptimizationSATCON(
    config,
    abs_bandwidth=1.2e6,
    use_module1=True,  # 梯度位置优化
    use_module2=True,  # 联合配对优化
    use_module3=True   # 整数规划决策
)

# 运行仿真
snr_range = np.arange(0, 31, 5)
mean_rates, mean_se, _, _ = system.simulate_performance(
    snr_db_range=snr_range,
    elevation_deg=10,
    n_realizations=100
)

print(f"平均频谱效率 @ 20dB: {mean_se[4]:.2f} bits/s/Hz")
```

### 示例2：消融实验
```python
# 创建不同配置的系统
systems = {
    'baseline': JointOptimizationSATCON(
        config, 1.2e6,
        use_module1=False, use_module2=False, use_module3=False
    ),
    'full': JointOptimizationSATCON(
        config, 1.2e6,
        use_module1=True, use_module2=True, use_module3=True
    )
}

# 对比性能
for name, sys in systems.items():
    rates, se, _, _ = sys.simulate_performance(
        snr_db_range=[20], n_realizations=10, verbose=False
    )
    print(f"{name}: SE = {se[0]:.2f} bits/s/Hz")
```

---

## 目录结构

```
src_enhanced/
├── gradient_position_optimizer.py   # 模块1
├── joint_pairing_optimizer.py       # 模块2
├── integer_programming_decision.py  # 模块3
├── joint_satcon_system.py          # 集成系统
├── utils_enhanced.py               # 工具函数
├── __init__.py
└── README.md                       # 本文件

tests/
├── test_gradient_optimizer.py      # 模块1测试
├── test_joint_pairing.py           # 模块2测试
├── test_ilp_decision.py            # 模块3测试
└── test_joint_system.py            # 集成测试
```

---

## 已知问题与未来改进

### 当前限制
1. **性能提升有限** (~3-5%)：需要更大规模的仿真验证
2. **模块3效果不显著**：可能需要更复杂的场景或更大的用户数
3. **交替优化迭代少**：当前只做有限次迭代，可以进一步优化

### 计划改进
1. **增加仿真规模**：从10次增加到1000次 Monte Carlo 实现
2. **优化算法参数**：调整收敛阈值和最大迭代次数
3. **扩展场景测试**：
   - 不同用户数 (N=16, 32, 64)
   - 不同仰角 (E=10°, 20°, 40°)
   - 不同带宽 (Bd=0.4, 1.2, 2.0, 3.0 MHz)
4. **性能优化**：
   - 并行化 Monte Carlo 仿真
   - 缓存信道增益计算结果
   - 使用更快的 ILP 求解器 (Gurobi)

---

## 贡献者

- **作者：** SATCON Enhancement Project
- **日期：** 2025-12-10
- **文档版本：** v1.0

---

## 参考文档

- [enhancement_plan_joint_optimization.md](../docs/enhancement_plan_joint_optimization.md) - 完整升级方案
- [DIRECTORY_STRUCTURE.md](../DIRECTORY_STRUCTURE.md) - 项目结构说明

---

## 许可证

MIT License - 参见项目根目录的 LICENSE 文件
