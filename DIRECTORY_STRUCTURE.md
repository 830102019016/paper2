# 项目目录结构说明

**创建时间:** 2025-12-10
**用途:** 联合优化方案开发

---

## 完整目录树

```
satcon_reproduction/
│
├── config.py                           # 全局配置（共享）
├── README.md                           # 项目说明
├── requirements.txt                    # 依赖包
├── DIRECTORY_STRUCTURE.md              # 本文档
│
├── src/                                # ✅ Baseline核心代码（保留不变）
│   ├── __init__.py
│   ├── channel_models.py               # Loo信道、路径损耗（共享）
│   ├── power_allocation.py             # NOMA功率分配（共享）
│   ├── noma_transmission.py            # 卫星NOMA传输
│   ├── abs_placement.py                # k-means/k-medoids位置优化
│   ├── a2g_channel.py                  # A2G/S2A信道（共享）
│   ├── user_distribution.py            # 用户分布生成（共享）
│   ├── satcon_system.py                # 原SATCON完整系统
│   └── utils.py                        # 工具函数
│
├── src_enhanced/                       # 🆕 联合优化新模块
│   ├── __init__.py
│   ├── gradient_position_optimizer.py  # 模块1：梯度位置优化
│   ├── joint_pairing_optimizer.py      # 模块2：联合配对优化
│   ├── integer_programming_decision.py # 模块3：整数规划决策
│   ├── joint_satcon_system.py          # 完整联合优化系统
│   └── utils_enhanced.py               # 新增工具函数
│
├── simulations/                        # ✅ Baseline仿真脚本（保留）
│   ├── fig2_complete.py                # 复现论文Figure 2
│   ├── fig2_sat_noma.py                # SAT-NOMA baseline
│   └── validation.py                   # 快速验证
│
├── experiments/                        # 🆕 对比实验脚本
│   ├── __init__.py
│   ├── config_comparison.yaml          # 实验配置
│   ├── run_comparison.py               # 主对比实验
│   ├── run_ablation.py                 # 消融实验
│   ├── run_scalability.py              # 可扩展性实验
│   └── generate_paper_figures.py       # 生成论文图表
│
├── tests/                              # ✅ 单元测试（保留 + 扩展）
│   ├── __init__.py
│   ├── test_channel.py                 # baseline测试
│   ├── test_noma.py                    # baseline测试
│   ├── test_power_allocation.py        # baseline测试
│   ├── test_gradient_optimizer.py      # 🆕 新模块测试
│   ├── test_joint_pairing.py           # 🆕 新模块测试
│   └── test_ilp_decision.py            # 🆕 新模块测试
│
├── results/                            # 结果存储（分层组织）
│   ├── baseline/                       # ✅ Baseline结果
│   │   ├── figures/
│   │   │   ├── fig2_complete.png
│   │   │   └── user_distribution_test.png
│   │   └── data/
│   │       └── fig2_baseline.npz
│   │
│   ├── comparison/                     # 🆕 对比实验结果
│   │   ├── figures/                    # 对比图表
│   │   ├── data/                       # 对比数据
│   │   └── tables/                     # 对比表格
│   │
│   └── proposed/                       # 🆕 新方案结果
│       ├── figures/
│       └── data/
│
├── docs/                               # 文档
│   ├── enhancement_plan_joint_optimization.md  # 详细实施计划
│   └── (其他文档待添加)
│
├── notebooks/                          # 🆕 Jupyter笔记本（可选）
│   └── (待添加)
│
├── scripts/                            # 🆕 辅助脚本
│   └── (待添加)
│
└── MDPI_template_ACS/                  # LaTeX模板
```

---

## 模块职责划分

### src/ (Baseline - 不修改)
- **用途:** 保持原SATCON论文的实现，确保可复现性
- **原则:** 只读，不修改
- **共享模块:** `channel_models.py`, `user_distribution.py`, `a2g_channel.py`

### src_enhanced/ (New - 联合优化)
- **用途:** 实现3个新优化模块
- **依赖:** 可以调用 `src/` 的共享模块
- **独立性:** `src/` 不依赖 `src_enhanced/`

### experiments/ (对比实验)
- **用途:** 运行baseline vs proposed的对比实验
- **调用:** 同时调用 `src/` 和 `src_enhanced/`

---

## 文件状态说明

| 状态 | 说明 |
|------|------|
| ✅ | 已存在，保持不变 |
| 🆕 | 新创建，待实现 |
| 📝 | 占位文件，需填充代码 |

---

## 快速导航

### 实现新模块
1. 编辑 `src_enhanced/gradient_position_optimizer.py`
2. 编辑 `src_enhanced/joint_pairing_optimizer.py`
3. 编辑 `src_enhanced/integer_programming_decision.py`

### 运行实验
```bash
# Baseline
python simulations/fig2_complete.py

# 对比实验
python experiments/run_comparison.py

# 消融实验
python experiments/run_ablation.py
```

### 查看结果
```bash
# Baseline结果
ls results/baseline/figures/

# 对比结果
ls results/comparison/figures/
```

---

## 下一步

1. **本周:** 实现 `src_enhanced/` 的3个模块
2. **下周:** 实现 `experiments/run_comparison.py`
3. **后续:** 运行完整实验，生成论文图表

详细计划见: `docs/enhancement_plan_joint_optimization.md`
