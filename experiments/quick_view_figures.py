"""
快速查看已保存的实验图表
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import sys

def view_figure(image_path, title=None):
    """显示图表"""
    if not Path(image_path).exists():
        print(f"[Error] 图表不存在: {image_path}")
        return False

    img = mpimg.imread(image_path)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return True

def main():
    """主函数"""

    print("=" * 80)
    print("快速查看实验图表")
    print("=" * 80)

    # 可用的图表
    figures = {
        '1': {
            'path': 'results/figures/fig2_abs_bandwidth_impact.png',
            'title': 'ABS Bandwidth Impact (Figure 2 Style)',
            'desc': 'ABS带宽影响分析 (论文Figure 2风格)'
        },
        '2': {
            'path': 'results/figures/comparison_baseline_3d.png',
            'title': 'Baseline Method - 3D Visualization',
            'desc': 'Baseline方法 - 3D用户分布和ABS位置'
        },
        '3': {
            'path': 'results/figures/comparison_ours_3d.png',
            'title': 'Our Algorithm - 3D Visualization',
            'desc': '我们的算法 - 3D用户分布和ABS位置'
        }
    }

    # 如果命令行有参数，直接显示指定图表
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in figures:
            fig_info = figures[choice]
            print(f"\n显示: {fig_info['desc']}")
            if view_figure(fig_info['path'], fig_info['title']):
                print(f"[OK] 图表已显示")
            return
        else:
            print(f"[Error] 无效的选项: {choice}")
            print("请使用: python quick_view_figures.py [1|2|3|all]")
            return

    # 交互式选择
    print("\n可用的图表:")
    for key, fig_info in figures.items():
        status = "✓" if Path(fig_info['path']).exists() else "✗"
        print(f"  [{key}] {status} {fig_info['desc']}")
    print(f"  [all] 显示所有图表")
    print(f"  [q] 退出")

    choice = input("\n请选择要查看的图表 (1/2/3/all/q): ").strip().lower()

    if choice == 'q':
        print("退出")
        return

    if choice == 'all':
        # 显示所有图表
        for key, fig_info in sorted(figures.items()):
            print(f"\n[{key}] {fig_info['desc']}")
            if view_figure(fig_info['path'], fig_info['title']):
                print(f"[OK] 图表已显示")
            else:
                print(f"[Skip] 图表不存在，跳过")
    elif choice in figures:
        # 显示单个图表
        fig_info = figures[choice]
        print(f"\n显示: {fig_info['desc']}")
        if view_figure(fig_info['path'], fig_info['title']):
            print(f"[OK] 图表已显示")
    else:
        print(f"[Error] 无效的选项: {choice}")

if __name__ == "__main__":
    main()
