"""
ABS位置优化

实现：
1. k-means聚类定位（k=1）
2. k-medoids聚类定位（k=1）
3. 最优高度搜索
4. 选择成本最小的方案

论文参考：
- Section III.A: ABS placement process
- 公式(2): arg min_{p∈W} Σ ||u_i - p||²
- 高度优化: ∂L^{A2G}(h,r)/∂h = 0
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import config


def simple_kmedoids_k1(X):
    """
    简单的k-medoids实现（k=1情况）

    对于k=1，medoid就是到所有其他点距离和最小的那个点

    参数:
        X (ndarray): shape (n_samples, n_features) 数据点

    返回:
        center (ndarray): shape (n_features,) medoid点的坐标
        inertia (float): 总距离平方和（与sklearn KMedoids兼容）
    """
    n_samples = X.shape[0]

    # 计算每个点到所有其他点的距离平方和
    min_cost = float('inf')
    best_idx = 0

    for i in range(n_samples):
        # 计算点i到所有点的距离平方和
        distances_sq = np.sum((X - X[i]) ** 2, axis=1)
        cost = np.sum(distances_sq)

        if cost < min_cost:
            min_cost = cost
            best_idx = i

    center = X[best_idx]
    inertia = min_cost

    return center, inertia


class ABSPlacement:
    """ABS位置优化器"""
    
    def __init__(self, abs_height_range=(50, 500), height_step=10):
        """
        初始化ABS位置优化器
        
        参数:
            abs_height_range (tuple): ABS高度范围 (min, max) 单位m
            height_step (float): 高度搜索步长 (m)
        """
        self.h_min, self.h_max = abs_height_range
        self.height_step = height_step
    
    def optimize_xy_position(self, user_positions):
        """
        优化ABS的水平位置 (x, y)
        
        方法：
        1. k-means (k=1): 找质心
        2. k-medoids (k=1): 找中位点
        3. 比较成本，选择更优方案
        
        论文公式(2):
        arg min_{p ∈ W} Σ_{u_i ∈ U} ||u_i - p||²
        
        参数:
            user_positions (ndarray): shape (N, 3) 用户坐标
        
        返回:
            xy_position (ndarray): shape (2,) 最优(x, y)坐标
            method (str): 'kmeans' or 'kmedoids'
            cost (float): 最小成本
        """
        # 只使用x, y坐标（忽略z）
        xy_coords = user_positions[:, :2]
        
        # 1. k-means (k=1)
        kmeans = KMeans(n_clusters=1, random_state=config.random_seed, n_init=10)
        kmeans.fit(xy_coords)
        pos_kmeans = kmeans.cluster_centers_[0]
        cost_kmeans = kmeans.inertia_  # 总距离平方和
        
        # 2. k-medoids (k=1) - 使用简单实现
        pos_kmedoids, cost_kmedoids = simple_kmedoids_k1(xy_coords)
        
        # 3. 选择成本更低的
        if cost_kmeans < cost_kmedoids:
            xy_position = pos_kmeans
            method = 'kmeans'
            cost = cost_kmeans
        else:
            xy_position = pos_kmedoids
            method = 'kmedoids'
            cost = cost_kmedoids
        
        return xy_position, method, cost
    
    def optimize_height(self, xy_position, user_positions, a2g_channel):
        """
        优化ABS高度以最小化与最远用户的路径损耗
        
        论文方法：
        1. 找到距离ABS(x,y)位置最远的用户
        2. 优化高度h使得到该用户的路径损耗最小
        3. 求解: ∂L^{A2G}(h, r_i) / ∂h = 0
        
        简化实现：
        - 数值搜索：遍历高度范围，找到路径损耗最小的h
        
        参数:
            xy_position (ndarray): ABS的(x, y)位置
            user_positions (ndarray): 用户坐标
            a2g_channel: A2G信道模型实例
        
        返回:
            h_optimal (float): 最优高度 (m)
            min_loss (float): 最小路径损耗 (线性值)
        """
        # 1. 计算所有用户到ABS(x,y)的2D距离
        distances_2d = np.linalg.norm(
            user_positions[:, :2] - xy_position, axis=1
        )
        
        # 2. 找到最远用户
        farthest_idx = np.argmax(distances_2d)
        r_max = distances_2d[farthest_idx]
        
        # 3. 网格搜索最优高度
        heights = np.arange(self.h_min, self.h_max + self.height_step, 
                           self.height_step)
        losses = np.zeros(len(heights))
        
        for i, h in enumerate(heights):
            # 计算到最远用户的路径损耗
            losses[i] = a2g_channel.compute_pathloss(h, r_max)
        
        # 4. 找到最小损耗对应的高度
        min_idx = np.argmin(losses)
        h_optimal = heights[min_idx]
        min_loss = losses[min_idx]
        
        return h_optimal, min_loss
    
    def optimize_position_complete(self, user_positions, a2g_channel):
        """
        完整优化ABS位置 (x, y, z)
        
        流程：
        1. 优化水平位置 (x, y)
        2. 优化高度 z = h
        
        参数:
            user_positions (ndarray): shape (N, 3) 用户坐标
            a2g_channel: A2G信道模型实例
        
        返回:
            abs_position (ndarray): shape (3,) 最优ABS位置 [x, y, h]
            info (dict): 优化详情
        """
        # 步骤1：优化(x, y)
        xy_pos, method, xy_cost = self.optimize_xy_position(user_positions)
        
        # 步骤2：优化高度h
        h_optimal, min_loss = self.optimize_height(
            xy_pos, user_positions, a2g_channel
        )
        
        # 合成完整位置
        abs_position = np.array([xy_pos[0], xy_pos[1], h_optimal])
        
        # 收集优化信息
        info = {
            'xy_position': xy_pos,
            'height': h_optimal,
            'method': method,
            'xy_cost': xy_cost,
            'min_pathloss': min_loss,
            'min_pathloss_db': 10 * np.log10(min_loss)
        }
        
        return abs_position, info
    
    def compute_coverage_quality(self, abs_position, user_positions, a2g_channel):
        """
        计算ABS覆盖质量指标
        
        指标：
        - 平均路径损耗
        - 最大路径损耗
        - 路径损耗标准差
        
        参数:
            abs_position (ndarray): ABS位置
            user_positions (ndarray): 用户位置
            a2g_channel: A2G信道模型
        
        返回:
            quality_metrics (dict): 质量指标
        """
        h = abs_position[2]
        distances_2d = np.linalg.norm(
            user_positions[:, :2] - abs_position[:2], axis=1
        )
        
        # 计算所有用户的路径损耗
        pathlosses = np.array([
            a2g_channel.compute_pathloss(h, r) for r in distances_2d
        ])
        pathlosses_db = 10 * np.log10(pathlosses)
        
        quality_metrics = {
            'mean_pathloss_db': np.mean(pathlosses_db),
            'max_pathloss_db': np.max(pathlosses_db),
            'min_pathloss_db': np.min(pathlosses_db),
            'std_pathloss_db': np.std(pathlosses_db),
            'mean_distance_2d': np.mean(distances_2d),
            'max_distance_2d': np.max(distances_2d)
        }
        
        return quality_metrics


# ==================== 测试代码 ====================
def test_abs_placement():
    """测试ABS位置优化"""
    print("=" * 60)
    print("测试ABS位置优化")
    print("=" * 60)
    
    # 生成测试用户
    from src.user_distribution import UserDistribution
    from src.a2g_channel import A2GChannel
    
    dist = UserDistribution(config.N, config.coverage_radius, seed=42)
    user_positions = dist.generate_uniform_circle()
    
    print(f"\n生成 {config.N} 个用户")
    print(f"  覆盖半径: {config.coverage_radius} m")
    
    # 创建A2G信道模型（用于高度优化）
    a2g = A2GChannel()
    
    # 创建ABS位置优化器
    placement = ABSPlacement(
        abs_height_range=(config.abs_height_min, config.abs_height_max),
        height_step=config.abs_height_step
    )
    
    # 测试1：优化(x, y)位置
    print(f"\n【测试1：优化ABS水平位置】")
    xy_pos, method, cost = placement.optimize_xy_position(user_positions)
    print(f"  方法: {method}")
    print(f"  位置: ({xy_pos[0]:.2f}, {xy_pos[1]:.2f})")
    print(f"  成本: {cost:.2f} m²")
    
    # 验证：位置应该接近用户质心
    user_center = np.mean(user_positions[:, :2], axis=0)
    distance_to_center = np.linalg.norm(xy_pos - user_center)
    print(f"  用户质心: ({user_center[0]:.2f}, {user_center[1]:.2f})")
    print(f"  距离质心: {distance_to_center:.2f} m")
    
    # 测试2：优化高度
    print(f"\n【测试2：优化ABS高度】")
    h_optimal, min_loss = placement.optimize_height(xy_pos, user_positions, a2g)
    print(f"  最优高度: {h_optimal:.0f} m")
    print(f"  最小路径损耗: {10*np.log10(min_loss):.2f} dB")
    print(f"  高度范围: {config.abs_height_min}-{config.abs_height_max} m")
    
    # 测试3：完整优化
    print(f"\n【测试3：完整位置优化】")
    abs_position, info = placement.optimize_position_complete(
        user_positions, a2g
    )
    print(f"  最优位置: ({abs_position[0]:.2f}, {abs_position[1]:.2f}, {abs_position[2]:.0f})")
    print(f"  优化方法: {info['method']}")
    print(f"  路径损耗: {info['min_pathloss_db']:.2f} dB")
    
    # 测试4：覆盖质量评估
    print(f"\n【测试4：覆盖质量评估】")
    quality = placement.compute_coverage_quality(abs_position, user_positions, a2g)
    print(f"  平均路径损耗: {quality['mean_pathloss_db']:.2f} dB")
    print(f"  最大路径损耗: {quality['max_pathloss_db']:.2f} dB")
    print(f"  路径损耗标准差: {quality['std_pathloss_db']:.2f} dB")
    print(f"  平均2D距离: {quality['mean_distance_2d']:.2f} m")
    
    # 可视化
    print(f"\n【测试5：可视化ABS位置】")
    fig, ax = dist.visualize_distribution(
        user_positions,
        abs_position=abs_position,
        save_path='results/figures/abs_placement_test.png'
    )
    
    print("\n" + "=" * 60)
    print("✓ ABS位置优化测试完成")
    print("=" * 60)
    
    plt.show()


if __name__ == "__main__":
    test_abs_placement()
