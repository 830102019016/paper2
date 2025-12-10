# -*- coding: utf-8 -*-
"""
快速测试版本 - 验证实验脚本

小规模测试：3个SNR点 × 10次实现
用于快速验证脚本是否正常工作

运行时间：约2-3分钟
"""
import numpy as np
import sys
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from experiments.run_comparison import ComparisonFramework

# 创建测试配置
test_config = {
    'monte_carlo': {
        'n_realizations': 10,  # 快速测试：只用10次
        'n_realizations_ablation': 5,
        'random_seed': 42
    },
    'scenarios': {
        'snr_range': [10, 20, 30],  # 只测试3个SNR点
        'elevation_angle': 10,
        'abs_bandwidth': 1.2e6,
        'user_number': 32
    },
    'baselines': [
        {
            'name': 'Original-SATCON',
            'description': '原方案',
            'enabled': True,
            'module_flags': {
                'use_module1': False,
                'use_module2': False,
                'use_module3': False
            }
        },
        {
            'name': 'SATCON+M1',
            'description': '仅模块1',
            'enabled': True,
            'module_flags': {
                'use_module1': True,
                'use_module2': False,
                'use_module3': False
            }
        },
        {
            'name': 'Proposed-Full',
            'description': '完整方案',
            'enabled': True,
            'module_flags': {
                'use_module1': True,
                'use_module2': True,
                'use_module3': True
            }
        }
    ],
    'experiments': {
        'main': {'enabled': True},
        'ablation': {'enabled': False}  # 快速测试跳过消融实验
    },
    'output': {
        'results_dir': 'results/test_quick',
        'figures_dir': 'results/figures/test_quick',
        'data_dir': 'results/data/test_quick',
        'tables_dir': 'results/tables/test_quick'
    }
}

# 保存测试配置
test_config_file = Path('experiments/config_test_quick.yaml')
with open(test_config_file, 'w', encoding='utf-8') as f:
    yaml.dump(test_config, f)

print("=" * 80)
print("快速测试 - 验证实验脚本")
print("=" * 80)
print("\n配置:")
print(f"  SNR点: {test_config['scenarios']['snr_range']}")
print(f"  Monte Carlo次数: {test_config['monte_carlo']['n_realizations']}")
print(f"  系统数: {len([b for b in test_config['baselines'] if b['enabled']])}")
print(f"  预计时间: 2-3分钟")
print("\n" + "=" * 80)

# 创建并运行
framework = ComparisonFramework(str(test_config_file))
framework.create_systems()
framework.run_main_experiment()
framework.plot_main_results()
framework.print_summary()

print("\n✓ 快速测试完成！")
print("如果一切正常，可以运行完整的大规模实验：")
print("  python experiments/run_comparison.py")
