# SATCON论文复现项目

## 论文信息
**标题**: Satellite Aerial Terrestrial Hybrid NOMA Scheme in 6G Networks: An Unsupervised Learning Approach

**作者**: Michail Karavolos et al.

**会议**: IEEE (2022)

## 项目目标

### Phase 1: 基础验证 (当前阶段)
- ✅ 实现简化Loo信道模型
- ✅ 实现卫星NOMA传输机制
- ✅ 验证功率分配公式
- ✅ 复现Figure 2的SAT-NOMA基准曲线

### Phase 2: ABS集成 (计划中)
- ⬜ k-means/k-medoids位置优化
- ⬜ A2G信道模型
- ⬜ 混合NOMA/OMA决策
- ⬜ 完整Figure 2复现

### Phase 3: 精细化 (计划中)
- ⬜ SGP4卫星轨道仿真
- ⬜ 精确Loo模型参数
- ⬜ 大规模Monte Carlo
- ⬜ 其他性能指标

## 快速开始

### 1. 环境配置
```bash
# 创建虚拟环境 (推荐)
conda create -n satcon python=3.10
conda activate satcon

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行基础验证
```bash
# 测试信道模型
python src/channel_models.py

# 测试功率分配
python src/power_allocation.py

# 测试NOMA传输
python src/noma_transmission.py

# 运行完整验证
python simulations/validation.py
```

### 3. 复现Figure 2
```bash
python simulations/fig2_sat_noma.py
```

输出：
- 图表: `results/figures/fig2_baseline_sat_noma.png`
- 数据: `results/data/fig2_baseline.npz`

### 4. 运行测试
```bash
# 运行所有测试
pytest tests/ -v

# 运行单个测试
pytest tests/test_channel.py -v
```

## 项目结构
```
satcon_reproduction/
├── config.py              # 仿真参数 (Table I)
├── src/                   # 核心代码
│   ├── channel_models.py
│   ├── power_allocation.py
│   └── noma_transmission.py
├── tests/                 # 单元测试
├── simulations/           # 仿真脚本
├── results/               # 输出结果
└── docs/                  # 文档
```

## 系统参数

详见 `config.py` 和 `docs/parameters_reference.md`

关键参数：
- 用户数: N = 32
- 卫星频率: 1.625 GHz
- 卫星带宽: 5 MHz
- SNR范围: 0-30 dB
- Monte Carlo次数: 1000 (可调整)

## 性能基准

**预期结果** (Figure 2, E=10°):
- SE @ 10dB: ~3-4 bits/s/Hz
- SE @ 20dB: ~6-7 bits/s/Hz
- SE @ 30dB: ~9-10 bits/s/Hz

## 故障排查

详见 `docs/troubleshooting.md`

常见问题：
1. **曲线数值偏差**: 调整Loo模型参数
2. **仿真速度慢**: 减少Monte Carlo次数
3. **内存不足**: 使用流式计算

## 开发笔记

详见 `docs/implementation_notes.md`

## 许可证

本项目仅用于学术研究和学习目的。

## 联系方式

如有问题，请查看 `docs/troubleshooting.md` 或联系开发者。