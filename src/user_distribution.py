"""
用户空间分布模型

实现：
1. 均匀圆形分布
2. SC/WC信道条件分配
3. 用户位置可视化

论文参考：
- Section II.A: N个MTs均匀分布在半径R=500m的圆形区域
- Section IV: 50% SC (strong channel) + 50% WC (weak channel)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import config


class UserDistribution:
    """用户空间分布生成器"""
    
    def __init__(self, n_users, radius, seed=None):
        """
        初始化用户分布
        
        参数:
            n_users (int): 用户数量 N
            radius (float): 覆盖半径 (m)
            seed (int, optional): 随机种子
        """
        self.n_users = n_users
        self.radius = radius
        self.rng = np.random.default_rng(seed)
    
    def generate_uniform_circle(self):
        """
        在圆形区域内均匀分布用户
        
        论文：MTs uniformly distributed within the ABS coverage area W
        
        算法：
        1. 生成随机角度 θ ~ U(0, 2π)
        2. 生成随机半径 r ~ sqrt(U(0, R²))  # 注意：需要sqrt确保均匀
        3. 转换为笛卡尔坐标 (x, y)
        4. 地面用户：z = 0
        
        返回:
            positions (ndarray): shape (N, 3) 用户坐标 [x, y, z]
        """
        # 极坐标采样
        theta = 2 * np.pi * self.rng.random(self.n_users)
        r = self.radius * np.sqrt(self.rng.random(self.n_users))  # sqrt确保均匀
        
        # 转笛卡尔坐标
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros(self.n_users)  # 地面用户
        
        positions = np.column_stack([x, y, z])
        return positions
    
    def assign_channel_conditions(self, positions, sc_ratio=0.5):
        """
        分配SC/WC信道条件
        
        论文场景：50% SC (LoS situation) + 50% WC (deep shadowing)
        
        实现方式：随机分配（不基于位置）
        - SC用户：信道条件好，Loo模型参数增强
        - WC用户：信道条件差，Loo模型参数减弱
        
        参数:
            positions (ndarray): 用户位置
            sc_ratio (float): SC用户比例 (默认0.5)
        
        返回:
            channel_types (ndarray): shape (N,) 信道类型
                                    0=WC (weak channel)
                                    1=SC (strong channel)
        """
        n_users = len(positions)
        n_sc = int(n_users * sc_ratio)
        
        # 随机选择SC用户
        all_indices = np.arange(n_users)
        sc_indices = self.rng.choice(all_indices, size=n_sc, replace=False)
        
        # 分配类型
        channel_types = np.zeros(n_users, dtype=int)  # 默认WC
        channel_types[sc_indices] = 1  # SC用户
        
        return channel_types
    
    def compute_distances_from_point(self, positions, point):
        """
        计算用户到指定点的2D距离
        
        参数:
            positions (ndarray): shape (N, 3) 用户坐标
            point (ndarray): shape (3,) 参考点坐标
        
        返回:
            distances (ndarray): shape (N,) 2D距离 (m)
        """
        # 只考虑x, y（水平距离）
        distances = np.linalg.norm(positions[:, :2] - point[:2], axis=1)
        return distances
    
    def visualize_distribution(self, positions, channel_types=None, 
                              abs_position=None, save_path=None):
        """
        可视化用户分布
        
        参数:
            positions (ndarray): 用户坐标
            channel_types (ndarray, optional): 信道类型
            abs_position (ndarray, optional): ABS位置
            save_path (str, optional): 保存路径
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制覆盖区域
        circle = plt.Circle((0, 0), self.radius, 
                           color='lightgray', fill=True, alpha=0.3,
                           label=f'Coverage (R={self.radius}m)')
        ax.add_patch(circle)
        
        # 绘制用户
        if channel_types is not None:
            # 区分SC/WC
            sc_mask = channel_types == 1
            wc_mask = channel_types == 0
            
            ax.scatter(positions[sc_mask, 0], positions[sc_mask, 1],
                      c='green', marker='o', s=50, alpha=0.7,
                      label=f'SC Users ({np.sum(sc_mask)})', zorder=3)
            ax.scatter(positions[wc_mask, 0], positions[wc_mask, 1],
                      c='red', marker='x', s=50, alpha=0.7,
                      label=f'WC Users ({np.sum(wc_mask)})', zorder=3)
        else:
            ax.scatter(positions[:, 0], positions[:, 1],
                      c='blue', marker='o', s=50, alpha=0.7,
                      label=f'Users ({len(positions)})', zorder=3)
        
        # 绘制ABS
        if abs_position is not None:
            ax.scatter(abs_position[0], abs_position[1],
                      c='orange', marker='^', s=300, 
                      edgecolors='black', linewidths=2,
                      label=f'ABS (h={abs_position[2]:.0f}m)', zorder=4)
        
        # 图表设置
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'User Distribution (N={self.n_users})', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, loc='upper right')
        ax.axis('equal')
        ax.set_xlim([-self.radius*1.2, self.radius*1.2])
        ax.set_ylim([-self.radius*1.2, self.radius*1.2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")
        
        return fig, ax


# ==================== 测试代码 ====================
def test_user_distribution():
    """测试用户分布模块"""
    print("=" * 60)
    print("测试用户空间分布")
    print("=" * 60)
    
    # 创建分布生成器
    dist = UserDistribution(
        n_users=config.N,
        radius=config.coverage_radius,
        seed=config.random_seed
    )
    
    # 测试1：生成均匀分布
    print(f"\n【测试1：生成均匀圆形分布】")
    positions = dist.generate_uniform_circle()
    print(f"生成 {len(positions)} 个用户位置")
    print(f"  X范围: [{np.min(positions[:, 0]):.1f}, {np.max(positions[:, 0]):.1f}] m")
    print(f"  Y范围: [{np.min(positions[:, 1]):.1f}, {np.max(positions[:, 1]):.1f}] m")
    print(f"  Z范围: [{np.min(positions[:, 2]):.1f}, {np.max(positions[:, 2]):.1f}] m")
    
    # 验证：所有用户在圆内
    distances_from_origin = np.linalg.norm(positions[:, :2], axis=1)
    all_inside = np.all(distances_from_origin <= config.coverage_radius)
    print(f"  所有用户在圆内: {'✓ 是' if all_inside else '✗ 否'}")
    print(f"  最远用户距离: {np.max(distances_from_origin):.1f} m")
    
    # 测试2：分配信道条件
    print(f"\n【测试2：分配SC/WC信道条件】")
    channel_types = dist.assign_channel_conditions(positions, sc_ratio=0.5)
    n_sc = np.sum(channel_types == 1)
    n_wc = np.sum(channel_types == 0)
    print(f"  SC用户: {n_sc} ({n_sc/len(positions)*100:.1f}%)")
    print(f"  WC用户: {n_wc} ({n_wc/len(positions)*100:.1f}%)")
    print(f"  比例验证: {'✓ 接近50%' if abs(n_sc - config.N/2) <= 2 else '⚠ 偏差较大'}")
    
    # 测试3：距离计算
    print(f"\n【测试3：距离计算】")
    test_point = np.array([100, 100, 0])
    distances = dist.compute_distances_from_point(positions, test_point)
    print(f"  测试点: ({test_point[0]}, {test_point[1]}, {test_point[2]})")
    print(f"  距离统计:")
    print(f"    均值: {np.mean(distances):.1f} m")
    print(f"    最小: {np.min(distances):.1f} m")
    print(f"    最大: {np.max(distances):.1f} m")
    
    # 测试4：可视化
    print(f"\n【测试4：生成可视化】")
    output_dir = Path(__file__).parent.parent / 'results' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / 'user_distribution_test.png'
    
    fig, ax = dist.visualize_distribution(
        positions, 
        channel_types=channel_types,
        abs_position=np.array([0, 0, 100]),  # 测试ABS位置
        save_path=save_path
    )
    
    print("\n" + "=" * 60)
    print("✓ 用户分布模块测试完成")
    print("=" * 60)
    
    plt.show()


if __name__ == "__main__":
    test_user_distribution()
