import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,        # 全局字体大小
    "axes.labelsize": 14,   # 坐标轴标签字体大小
    "xtick.labelsize": 14,  # x 轴刻度字体大小
    "ytick.labelsize": 14,  # y 轴刻度字体大小
    "legend.fontsize": 14,  # 图例字体大小
})

# 读取数据
data_2th = pd.read_csv('2th_new/traj_error.txt', sep='\t').iloc[0:101]
data_4th = pd.read_csv('4th_new/traj_error.txt', sep='\t').iloc[0:101]
data_A   = pd.read_csv('2th_new/traj_error_A.txt', sep='\t').iloc[0:101]

plt.figure(figsize=(8, 6))

# 不对阴影做平滑
for data, label, color in [
    (data_A,   'Pretrained', '#1f77b4'),
    (data_2th, 'ANI-2', '#2ca02c'),
    (data_4th, 'ANI-4', '#733497')
]:
    plt.plot(data['t_index'], data['avg_relative_error'], label=label, color=color, linewidth=2)
    plt.fill_between(data['t_index'],
                     data['avg_relative_error'] - data['std_relative_error'],
                     data['avg_relative_error'] + data['std_relative_error'],
                     color=color, alpha=0.15, linewidth=0)  # 阴影不平滑

# plt.title('Comparison of Average L2 Error', fontsize=14)
plt.xlabel('Time Index', fontsize=14)
plt.ylabel('Average L2 Error', fontsize=14)
# plt.yscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('traj_error_new.pdf', format='pdf', bbox_inches='tight', dpi=300)
# plt.show()
plt.close()
