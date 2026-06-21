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
data_FNO = pd.read_csv('FNO_baseline/traj_error.txt', sep='\t').iloc[0:101]

plt.figure(figsize=(8, 6))

# 不对阴影做平滑
for data, label, color in [
    (data_A,   'Pretrained', '#0072B2'),
    (data_2th, 'ANI-2', '#009E73'),
    (data_4th, 'ANI-4', '#E69F00'),
    (data_FNO, 'FNO', '#D55E00')
]:
    plt.plot(data['t_index'], data['avg_relative_error'], label=label, color=color, linewidth=2)
    plt.fill_between(data['t_index'],
                     data['avg_relative_error'] - data['std_relative_error'],
                     data['avg_relative_error'] + data['std_relative_error'],
                     color=color, alpha=0.15, linewidth=0)  # 阴影不平滑

# plt.title('Comparison of Average L2 Error', fontsize=14)
plt.xlabel('Time Step', fontsize=14)
plt.ylabel(r'Mean relative $L^2$ error', fontsize=14)
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('traj_error_new_response.pdf', format='pdf', bbox_inches='tight', dpi=300)
# plt.show()
plt.close()
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# def mm_to_inch(mm):
#     return mm / 25.4

# # Choose one:
# # width_mm = 88    # one-column figure
# width_mm = 180     # two-column figure, safer for multi-curve plots
# height_mm = 115

# plt.rcParams.update({
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
#     "font.size": 7,
#     "axes.labelsize": 7,
#     "xtick.labelsize": 6.5,
#     "ytick.labelsize": 6.5,
#     "legend.fontsize": 6.5,
#     "axes.linewidth": 0.6,
#     "xtick.major.width": 0.6,
#     "ytick.major.width": 0.6,
#     "xtick.major.size": 3,
#     "ytick.major.size": 3,
#     "pdf.fonttype": 42,   # keep text editable in PDF
#     "ps.fonttype": 42,
#     "axes.unicode_minus": False,
# })

# data_2th = pd.read_csv('2th_new/traj_error.txt', sep='\t').iloc[0:101]
# data_4th = pd.read_csv('4th_new/traj_error.txt', sep='\t').iloc[0:101]
# data_A   = pd.read_csv('2th_new/traj_error_A.txt', sep='\t').iloc[0:101]
# # data_FNO = pd.read_csv('FNO_baseline/traj_error.txt', sep='\t').iloc[0:101]

# fig, ax = plt.subplots(figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)))

# series = [
#     (data_A,   'Pretrained prior', '#0072B2', '-',  'o'),
#     (data_2th, 'ANI-2',            '#009E73', '--', 's'),
#     (data_4th, 'ANI-4',            '#E69F00', '-.', '^'),
#     # (data_FNO, 'FNO',             '#D55E00', ':',  'D'),
# ]

# for data, label, color, linestyle, marker in series:
#     x = data['t_index'].to_numpy()
#     mean = data['avg_relative_error'].to_numpy()
#     sd = data['std_relative_error'].to_numpy()

#     # Avoid invalid lower band on log scale
#     lower = np.maximum(mean - sd, np.finfo(float).tiny)
#     upper = mean + sd

#     ax.plot(
#         x, mean,
#         label=label,
#         color=color,
#         linestyle=linestyle,
#         linewidth=1.2,
#         marker=marker,
#         markersize=2.2,
#         markevery=20,
#     )

#     ax.fill_between(
#         x, lower, upper,
#         color=color,
#         alpha=0.15,
#         linewidth=0,
#     )

# ax.set_xlabel('Time index')
# ax.set_ylabel(r'Mean relative $L^2$ error')
# ax.set_yscale('log')

# ax.grid(True, which='major', linewidth=0.4, alpha=0.3)
# ax.legend(frameon=False)

# fig.tight_layout()
# fig.savefig('traj_error_new.pdf', format='pdf', bbox_inches='tight')
# plt.close(fig)