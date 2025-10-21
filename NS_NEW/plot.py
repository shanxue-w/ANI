import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,        # 全局字体大小
    "axes.labelsize": 14,   # 坐标轴标签字体大小
    "xtick.labelsize": 14,  # x 轴刻度字体大小
    "ytick.labelsize": 14,  # y 轴刻度字体大小
    "legend.fontsize": 14,  # 图例字体大小
})

file_path_2th = '2th/traj_error.txt'
file_path_4th = '4th/traj_error.txt'
# file_path_A   = '2th/traj_error_1e4.txt'
file_path_A   = 'baseline/traj_error.txt'

data_2th = pd.read_csv(file_path_2th, sep='\t')
data_2th = data_2th['avg_relative_error']
data_2th = data_2th[0:400]
data_4th = pd.read_csv(file_path_4th, sep='\t')
data_4th = data_4th['avg_relative_error']
data_4th = data_4th[0:400]
data_A   = pd.read_csv(file_path_A, sep='\t')
data_A   = data_A['avg_relative_error']
data_A   = data_A[0:400]

# 设置图表大小
plt.figure(figsize=(8, 6))

# 绘制数据
plt.plot(data_A, label='FNO', linestyle='-.', color='#1f77b4', linewidth=2, marker='^', markersize=2)
plt.plot(data_2th, label='ANI-2', linestyle='-', color='#2ca02c', linewidth=2, marker='o', markersize=2)
plt.plot(data_4th, label='ANI-4', linestyle='--', color='#733497', linewidth=2, marker='s', markersize=2)

# 添加标题和标签
# plt.title('Comparison of Average Relative Error', fontsize=14)
plt.xlabel('Time Index', fontsize=14)
plt.ylabel('Average Relative Error', fontsize=14)

# 使用对数坐标轴
# plt.yscale('log')

# 添加网格线
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 添加图例
plt.legend()

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('ns_traj_error_model.pdf', format='pdf', bbox_inches='tight', dpi=300)
