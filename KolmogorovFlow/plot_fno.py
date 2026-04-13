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
file_path_les = '2th/traj_error_les.txt'
file_path_fno = 'FNO_baseline/traj_error.txt'
# file_path_opt = '2th/traj_error_opt.txt'

data_2th = pd.read_csv(file_path_2th, sep='\t')
data_2th = data_2th['avg_relative_error']
data_2th = data_2th[0:400]
data_4th = pd.read_csv(file_path_4th, sep='\t')
data_4th = data_4th['avg_relative_error']
data_4th = data_4th[0:400]
data_les = pd.read_csv(file_path_les, sep='\t')
data_les = data_les['avg_relative_error']
data_les = data_les[0:400]
data_fno = pd.read_csv(file_path_fno, sep='\t')
data_fno = data_fno['avg_relative_error']
data_fno = data_fno[0:400]
# data_opt = pd.read_csv(file_path_opt, sep='\t')
# data_opt = data_opt['avg_relative_error']
# data_opt = data_opt[0:400]

# 设置图表大小
plt.figure(figsize=(8, 6))

# 绘制数据
plt.plot(data_les, label='LES',  color='#0072B2', linewidth=2)
plt.plot(data_2th, label='ANI-2',  color='#009E73', linewidth=2)
plt.plot(data_4th, label='ANI-4',  color='#E69F00', linewidth=2)
plt.plot(data_fno, label='FNO',  color='#D55E00', linewidth=2)
# plt.plot(data_opt, label='Optimized LES', linestyle=':', color='orange', linewidth=1.5, marker='x', markersize=2)

# 添加标题和标签
# plt.title('Comparison of Average Relative Error', fontsize=14)
plt.xlabel('Time Index', fontsize=14)
plt.ylabel('Average Relative Error', fontsize=14)

# 使用对数坐标轴
plt.yscale('log')

# 添加网格线
plt.grid(True, alpha=0.3)

# 添加图例
plt.legend(fontsize=14)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('ns_traj_error_fno.pdf', format='pdf', bbox_inches='tight', dpi=300)
