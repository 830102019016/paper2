# -*- coding: utf-8 -*-
"""
简单验证 ILP 优化功能

测试 cvxpy 是否正常工作
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

try:
    import cvxpy as cp
    print("[OK] cvxpy 已安装")
    print(f"    版本: {cp.__version__}")

    # 简单测试 cvxpy 功能
    print("\n测试 cvxpy 基本功能...")
    x = cp.Variable(boolean=True)
    y = cp.Variable(boolean=True)

    objective = cp.Maximize(3*x + 2*y)
    constraints = [x + y <= 1]

    problem = cp.Problem(objective, constraints)
    result = problem.solve()

    print(f"  目标值: {result}")
    print(f"  x = {x.value}, y = {y.value}")
    print("[PASS] cvxpy 工作正常")

except ImportError as e:
    print(f"[FAIL] cvxpy 未安装: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[FAIL] cvxpy 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("测试 Module 3 的 ILP 实现")
print("="*70)

# 运行 pytest
import subprocess
result = subprocess.run(
    ["python", "-m", "pytest", "tests/test_ilp_decision.py::test_ilp_optimization", "-v"],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print(result.stderr)

if result.returncode == 0:
    print("\n[PASS] ILP 测试全部通过！")
else:
    print("\n[FAIL] ILP 测试失败")
    sys.exit(1)
