"""
NOMA用户配对可视化

展示卫星NOMA传输中的用户配对关系：
- 强用户（Strong）和弱用户（Weak）
- 配对连线
- 信道增益差异

目的：清楚展示NOMA配对机制和用户分组
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.user_distribution import UserDistribution
from src.noma_transmission import SatelliteNOMA
from src.power_allocation import NOMAAllocator


def plot_user_pairing(user_positions, sat_pairs, channel_gains,
                      abs_position=None, save_path=None):
    """
    绘制用户配对关系图

    参数:
        user_positions: 用户位置 [N, 3]
        sat_pairs: 用户配对 [(weak_idx, strong_idx), ...]
        channel_gains: 信道增益 [N]
        abs_position: ABS位置（可选）
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    coverage_radius = config.coverage_radius

    # 1. 绘制覆盖区域
    circle = plt.Circle((0, 0), coverage_radius,
                       color='lightgray', fill=True, alpha=0.2,
                       label=f'Coverage Area (R={coverage_radius}m)')
    ax.add_patch(circle)

    # 2. 确定每个用户的类型（强/弱/未配对）
    user_types = {}  # 0: 未配对, 1: 弱用户, 2: 强用户
    for weak_idx, strong_idx in sat_pairs:
        # 根据信道增益确定强弱
        if channel_gains[weak_idx] > channel_gains[strong_idx]:
            weak_idx, strong_idx = strong_idx, weak_idx

        user_types[weak_idx] = 'weak'
        user_types[strong_idx] = 'strong'

    # 3. 绘制配对连线（先画，这样不会覆盖用户点）
    for pair_idx, (idx1, idx2) in enumerate(sat_pairs):
        weak_idx = idx1 if channel_gains[idx1] <= channel_gains[idx2] else idx2
        strong_idx = idx2 if weak_idx == idx1 else idx1

        # 使用不同颜色表示不同的配对组
        color = plt.cm.tab20(pair_idx % 20)

        ax.plot([user_positions[weak_idx, 0], user_positions[strong_idx, 0]],
               [user_positions[weak_idx, 1], user_positions[strong_idx, 1]],
               color=color, linewidth=2, alpha=0.6, zorder=2)

        # 在连线中点添加配对编号
        mid_x = (user_positions[weak_idx, 0] + user_positions[strong_idx, 0]) / 2
        mid_y = (user_positions[weak_idx, 1] + user_positions[strong_idx, 1]) / 2
        ax.text(mid_x, mid_y, f'P{pair_idx+1}',
               fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                        edgecolor=color, alpha=0.8),
               zorder=3)

    # 4. 绘制用户点（按类型分组）
    weak_users = [i for i, t in user_types.items() if t == 'weak']
    strong_users = [i for i, t in user_types.items() if t == 'strong']

    if weak_users:
        ax.scatter(user_positions[weak_users, 0], user_positions[weak_users, 1],
                  c='red', marker='o', s=150, alpha=0.8,
                  edgecolors='darkred', linewidths=2,
                  label=f'Weak Users ({len(weak_users)})', zorder=4)

    if strong_users:
        ax.scatter(user_positions[strong_users, 0], user_positions[strong_users, 1],
                  c='green', marker='s', s=150, alpha=0.8,
                  edgecolors='darkgreen', linewidths=2,
                  label=f'Strong Users ({len(strong_users)})', zorder=4)

    # 5. 添加用户索引标签
    for i in range(len(user_positions)):
        if i in user_types:
            ax.text(user_positions[i, 0], user_positions[i, 1],
                   f'{i}', fontsize=9, ha='center', va='center',
                   color='white', fontweight='bold', zorder=5)

    # 6. 绘制ABS位置（如果提供）
    if abs_position is not None:
        ax.scatter(abs_position[0], abs_position[1],
                  c='orange', marker='^', s=400,
                  edgecolors='black', linewidths=2.5,
                  label=f'ABS (h={abs_position[2]:.0f}m)', zorder=6)

    # 7. 图表设置
    ax.set_xlabel('X (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=13, fontweight='bold')
    ax.set_title('NOMA User Pairing and Channel Conditions',
                fontsize=15, fontweight='bold', pad=20)

    # 添加统计信息
    channel_gains_db = 10 * np.log10(channel_gains)
    weak_gains = [channel_gains_db[i] for i in weak_users]
    strong_gains = [channel_gains_db[i] for i in strong_users]

    info_text = (f'Total Users: {len(user_positions)}\n'
                f'Pairs: {len(sat_pairs)}\n\n'
                f'Weak Users:\n'
                f'  Avg Gain: {np.mean(weak_gains):.1f} dB\n'
                f'Strong Users:\n'
                f'  Avg Gain: {np.mean(strong_gains):.1f} dB\n\n'
                f'Gain Difference:\n'
                f'  {np.mean(strong_gains) - np.mean(weak_gains):.1f} dB')

    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    # 图例
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axis('equal')
    ax.set_xlim([-coverage_radius*1.2, coverage_radius*1.2])
    ax.set_ylim([-coverage_radius*1.2, coverage_radius*1.2])

    plt.tight_layout()

    # 保存图表
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] 配对关系图已保存: {save_path}")

    return fig, ax


def plot_pairing_comparison(user_positions, sat_pairs, channel_gains,
                            abs_position_baseline, abs_position_ours,
                            save_path=None):
    """
    对比两种方法的配对关系（并排显示）

    参数:
        user_positions: 用户位置
        sat_pairs: 用户配对
        channel_gains: 信道增益
        abs_position_baseline: Baseline ABS位置
        abs_position_ours: 我们的算法ABS位置
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    coverage_radius = config.coverage_radius

    # 确定用户类型
    user_types = {}
    for idx1, idx2 in sat_pairs:
        weak_idx = idx1 if channel_gains[idx1] <= channel_gains[idx2] else idx2
        strong_idx = idx2 if weak_idx == idx1 else idx1
        user_types[weak_idx] = 'weak'
        user_types[strong_idx] = 'strong'

    weak_users = [i for i, t in user_types.items() if t == 'weak']
    strong_users = [i for i, t in user_types.items() if t == 'strong']

    # 绘制两个子图
    for ax, abs_pos, title in [(ax1, abs_position_baseline, 'Baseline: Heuristic + Uniform + k-means'),
                                (ax2, abs_position_ours, 'Our Algorithm: Greedy + KKT + L-BFGS-B')]:

        # 覆盖区域
        circle = plt.Circle((0, 0), coverage_radius,
                           color='lightgray', fill=True, alpha=0.2)
        ax.add_patch(circle)

        # 配对连线
        for pair_idx, (idx1, idx2) in enumerate(sat_pairs):
            weak_idx = idx1 if channel_gains[idx1] <= channel_gains[idx2] else idx2
            strong_idx = idx2 if weak_idx == idx1 else idx1

            color = plt.cm.tab20(pair_idx % 20)
            ax.plot([user_positions[weak_idx, 0], user_positions[strong_idx, 0]],
                   [user_positions[weak_idx, 1], user_positions[strong_idx, 1]],
                   color=color, linewidth=1.5, alpha=0.5, zorder=2)

        # 用户点
        if weak_users:
            ax.scatter(user_positions[weak_users, 0], user_positions[weak_users, 1],
                      c='red', marker='o', s=120, alpha=0.8,
                      edgecolors='darkred', linewidths=2,
                      label=f'Weak Users ({len(weak_users)})', zorder=4)

        if strong_users:
            ax.scatter(user_positions[strong_users, 0], user_positions[strong_users, 1],
                      c='green', marker='s', s=120, alpha=0.8,
                      edgecolors='darkgreen', linewidths=2,
                      label=f'Strong Users ({len(strong_users)})', zorder=4)

        # ABS位置
        ax.scatter(abs_pos[0], abs_pos[1],
                  c='orange', marker='^', s=400,
                  edgecolors='black', linewidths=2.5,
                  label=f'ABS (h={abs_pos[2]:.0f}m)', zorder=6)

        # ABS位置标注
        ax.text(abs_pos[0], abs_pos[1] - 80,
               f'({abs_pos[0]:.1f}, {abs_pos[1]:.1f})',
               fontsize=10, ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

        # 图表设置
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axis('equal')
        ax.set_xlim([-coverage_radius*1.2, coverage_radius*1.2])
        ax.set_ylim([-coverage_radius*1.2, coverage_radius*1.2])

    plt.tight_layout()

    # 保存图表
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] 配对对比图已保存: {save_path}")

    return fig, (ax1, ax2)


def main():
    """主函数：生成配对可视化图"""

    print("=" * 80)
    print("NOMA用户配对可视化")
    print("=" * 80)

    # 配置参数
    elevation_deg = 10

    print(f"\n系统配置:")
    print(f"  用户数: {config.N}")
    print(f"  覆盖半径: {config.coverage_radius} m")
    print(f"  卫星仰角: {elevation_deg} deg")

    # 生成用户分布
    print(f"\n{'='*80}")
    print(f"步骤1: 生成用户分布")
    print(f"{'='*80}")

    dist = UserDistribution(config.N, config.coverage_radius,
                           seed=config.random_seed)
    user_positions = dist.generate_uniform_circle()

    print(f"[OK] 生成 {config.N} 个用户位置")

    # 计算卫星信道增益和配对
    print(f"\n{'='*80}")
    print(f"步骤2: 计算信道增益和用户配对")
    print(f"{'='*80}")

    sat_noma = SatelliteNOMA(config)
    sat_channel_gains = sat_noma.compute_channel_gains_with_pathloss(elevation_deg)

    allocator = NOMAAllocator()
    sat_pairs, pairing_cost = allocator.optimal_user_pairing(sat_channel_gains)

    print(f"[OK] 用户配对完成")
    print(f"  配对数: {len(sat_pairs)}")

    # 显示配对详情
    print(f"\n配对详情:")
    channel_gains_db = 10 * np.log10(sat_channel_gains)
    for i, (idx1, idx2) in enumerate(sat_pairs):
        weak_idx = idx1 if sat_channel_gains[idx1] <= sat_channel_gains[idx2] else idx2
        strong_idx = idx2 if weak_idx == idx1 else idx1

        gain_diff = channel_gains_db[strong_idx] - channel_gains_db[weak_idx]
        print(f"  Pair {i+1}: User {weak_idx} (Weak, {channel_gains_db[weak_idx]:.1f}dB) "
              f"<-> User {strong_idx} (Strong, {channel_gains_db[strong_idx]:.1f}dB) "
              f"[Diff: {gain_diff:.1f}dB]")

    # 计算ABS位置（用于可选显示）
    from src.abs_placement import ABSPlacement
    from src.a2g_channel import A2GChannel

    a2g_channel = A2GChannel()
    placement = ABSPlacement(
        abs_height_range=(config.abs_height_min, config.abs_height_max),
        height_step=config.abs_height_step
    )
    abs_position, _ = placement.optimize_position_complete(user_positions, a2g_channel)

    # 生成配对可视化图
    print(f"\n{'='*80}")
    print(f"步骤3: 生成配对可视化图")
    print(f"{'='*80}")

    # 图1: 基本配对图
    save_path1 = Path('results/figures/user_pairing.png')
    plot_user_pairing(user_positions, sat_pairs, sat_channel_gains,
                     abs_position=abs_position, save_path=save_path1)

    # 图2: 配对对比图（需要两个ABS位置）
    # 这里使用相同位置作为示例，实际使用时可以传入不同的ABS位置
    print(f"\n如需生成配对对比图，请运行 compare_abs_positions.py 后")
    print(f"使用生成的两个ABS位置调用 plot_pairing_comparison 函数")

    # 总结
    print(f"\n{'='*80}")
    print(f"[OK] 配对可视化完成")
    print(f"{'='*80}")
    print(f"\n生成的图表:")
    print(f"  1. 配对关系图: {save_path1}")
    print(f"\n图例说明:")
    print(f"  - 红色圆圈: 弱用户（信道增益较低）")
    print(f"  - 绿色方块: 强用户（信道增益较高）")
    print(f"  - 彩色连线: NOMA配对关系")
    print(f"  - P1, P2, ...: 配对编号")
    print(f"  - 橙色三角: ABS位置")

    print(f"\n{'='*80}")

    plt.show()


if __name__ == "__main__":
    main()
