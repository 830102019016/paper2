"""
辅助工具函数

包含：
1. 单位转换
2. 绘图工具
3. 数据保存/加载
4. 性能指标计算
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==================== 单位转换 ====================
def db_to_linear(value_db):
    """dB转线性值"""
    return 10 ** (value_db / 10)


def linear_to_db(value_linear):
    """线性值转dB"""
    return 10 * np.log10(value_linear)


def dbm_to_watt(power_dbm):
    """dBm转瓦特"""
    return 10 ** ((power_dbm - 30) / 10)


def watt_to_dbm(power_watt):
    """瓦特转dBm"""
    return 10 * np.log10(power_watt) + 30


# ==================== 绘图工具 ====================
def setup_plot_style():
    """设置绘图风格"""
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['grid.alpha'] = 0.3


def save_figure(fig, filename, output_dir='results/figures', dpi=300):
    """
    保存图表
    
    参数:
        fig: matplotlib图表对象
        filename: 文件名
        output_dir: 输出目录
        dpi: 分辨率
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"✓ 图表已保存: {filepath}")


# ==================== 数据处理 ====================
def save_simulation_data(data_dict, filename, output_dir='results/data'):
    """
    保存仿真数据
    
    参数:
        data_dict: 数据字典
        filename: 文件名
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    np.savez(filepath, **data_dict)
    print(f"✓ 数据已保存: {filepath}")


def load_simulation_data(filename, data_dir='results/data'):
    """
    加载仿真数据
    
    参数:
        filename: 文件名
        data_dir: 数据目录
    
    返回:
        data: numpy数组字典
    """
    filepath = Path(data_dir) / filename
    data = np.load(filepath)
    print(f"✓ 数据已加载: {filepath}")
    return data


# ==================== 性能指标 ====================
def compute_spectral_efficiency(sum_rate, bandwidth):
    """
    计算频谱效率
    
    参数:
        sum_rate: 总速率 (bps)
        bandwidth: 带宽 (Hz)
    
    返回:
        se: 频谱效率 (bits/s/Hz)
    """
    return sum_rate / bandwidth


def compute_energy_efficiency(sum_rate, total_power):
    """
    计算能量效率
    
    参数:
        sum_rate: 总速率 (bps)
        total_power: 总功率 (W)
    
    返回:
        ee: 能量效率 (bits/J)
    """
    return sum_rate / total_power


def compute_fairness_index(rates):
    """
    计算Jain公平性指数
    
    FI = (sum(r_i))^2 / (N * sum(r_i^2))
    
    参数:
        rates: 用户速率数组
    
    返回:
        fi: 公平性指数 [0, 1]
    """
    n = len(rates)
    numerator = np.sum(rates) ** 2
    denominator = n * np.sum(rates ** 2)
    return numerator / denominator if denominator > 0 else 0


# ==================== 统计工具 ====================
def compute_confidence_interval(data, confidence=0.95):
    """
    计算置信区间
    
    参数:
        data: 数据数组
        confidence: 置信度
    
    返回:
        mean, lower, upper
    """
    from scipy import stats
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean, mean - interval, mean + interval


if __name__ == "__main__":
    print("工具函数模块")
    print("=" * 40)
    
    # 测试单位转换
    print("\n【单位转换测试】")
    print(f"20 dB = {db_to_linear(20):.2f} (线性)")
    print(f"100 (线性) = {linear_to_db(100):.2f} dB")
    print(f"30 dBm = {dbm_to_watt(30):.3f} W")
    
    # 测试公平性指数
    print("\n【公平性指数测试】")
    rates1 = np.array([1, 1, 1, 1])  # 完全公平
    rates2 = np.array([4, 0, 0, 0])  # 完全不公平
    print(f"均等速率: FI = {compute_fairness_index(rates1):.3f}")
    print(f"极端不均: FI = {compute_fairness_index(rates2):.3f}")
    
    print("\n✓ 工具函数测试完成")
