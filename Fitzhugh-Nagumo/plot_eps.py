import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "font.size": 14,        # 全局字体大小
    "axes.labelsize": 14,   # 坐标轴标签字体大小
    "xtick.labelsize": 14,  # x 轴刻度字体大小
    "ytick.labelsize": 14,  # y 轴刻度字体大小
    "legend.fontsize": 14,  # 图例字体大小
})

# 配置
cases = [4, 4.5, 5, 5.5, 6]  # 横坐标
methods = ['A', '2th', '4th']  # 文件读取名
plot_labels = {'A': 'Pretrained', '2th': 'ANI-2', '4th': 'ANI-4'}  # 图中显示名
colors = ['#1f77b4', '#2ca02c', '#733497']

file_pattern = {
    'A':   '2th_new/traj_error_A_{}.txt',
    '2th': '2th_new/traj_error_{}.txt',
    '4th': '4th_new/traj_error_{}.txt'
}

# 存储数据
values = {m: [] for m in methods}

# 读取每个 case 的第100个数据
for c in cases:
    for m in methods:
        df = pd.read_csv(file_pattern[m].format(c), sep='\t')
        val = df.loc[df['t_index'] == 99, 'avg_relative_error'].values[0]
        values[m].append(val)

# 画图
x = np.array(cases)

plt.figure(figsize=(8, 6))
for i, m in enumerate(methods):
    plt.plot(x, values[m], marker='o', color=colors[i], label=plot_labels[m], linewidth=2)

plt.xticks(x, cases)
plt.xlabel(r'$\epsilon$', fontsize=14)
plt.ylabel('Average Relative Error (t=1s)', fontsize=14)
# plt.title('Comparison at t=1s')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('line_comparison_t99_new.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()
