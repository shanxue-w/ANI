# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata

# plt.rcParams.update({
#     "font.size": 14,        # 全局字体大小
#     "axes.labelsize": 14,   # 坐标轴标签字体大小
#     "xtick.labelsize": 14,  # x 轴刻度字体大小
#     "ytick.labelsize": 14,  # y 轴刻度字体大小
#     "legend.fontsize": 14,  # 图例字体大小
# })

# # 文件名和方法名对应
# files = {
#     "Pretrained": "2th_new/pred_error_2th_A.txt",
#     "ANI-2": "2th_new/pred_error_2th.txt",
#     "ANI-4": "4th_new/pred_error_4th.txt"
# }

# fig, axes = plt.subplots(1, len(files), figsize=(15, 4), constrained_layout=True)

# # 网格用于插值
# grid_a = np.linspace(0.6, 0.8, 20)
# grid_b = np.linspace(0.7, 0.9, 20)
# A_grid, B_grid = np.meshgrid(grid_a, grid_b)

# for ax, (method, fname) in zip(axes, files.items()):
#     data = np.loadtxt(fname, delimiter=",")
#     a, b, err = data[:,0], data[:,1], data[:,2]

#     # log10压缩 error
#     # err_log = np.log10(np.clip(err, 1e-6, 1))
#     err_clip = np.clip(err, 0, 1)
#     # err_clip = err

#     # cubic插值生成平滑网格
#     ERR_grid = griddata((a, b), err_clip, (A_grid, B_grid), method='cubic')

#     # 绘制二维热力图
#     im = ax.imshow(ERR_grid, origin='lower', 
#                    extent=[grid_a[0], grid_a[-1], grid_b[0], grid_b[-1]],
#                    aspect='auto', cmap='viridis')

#     ax.set_title(method, fontsize=14)
#     ax.set_xlabel("a", fontsize=14)
#     ax.set_ylabel("b", fontsize=14)
#     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="error")

# # plt.show()
# plt.savefig("ab_error.pdf", format='pdf', bbox_inches='tight', dpi=300)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# 文件名和方法名对应
files = {
    "Baseline": "NEW_baseline/pred_error.txt",
    "ANI-2": "2th_new/pred_error_2th.txt",
    "ANI-4": "4th_new/pred_error_4th.txt"
}

fig, axes = plt.subplots(1, len(files), figsize=(15, 4), constrained_layout=True)

# 网格用于插值
grid_a = np.linspace(0.6, 0.8, 20)
grid_b = np.linspace(0.7, 0.9, 20)
A_grid, B_grid = np.meshgrid(grid_a, grid_b)

for ax, (method, fname) in zip(axes, files.items()):
    # 假设数据格式为: a, b, err
    data = np.loadtxt(fname, delimiter=",")
    a, b, err = data[:,0], data[:,1], data[:,2]

    # ---------------------------------------------------------
    # 修改核心逻辑：
    # 不使用 clip，而是筛选出“有效点”（err <= 1）。
    # 将 > 1 的点从输入中移除，griddata 会自动填补这些空缺。
    # ---------------------------------------------------------
    
    # 1. 创建掩码，保留 err <= 1 的数据
    mask = err <= 1
    
    # 2. 应用掩码，只取出有效的数据点坐标和值
    a_valid = a[mask]
    b_valid = b[mask]
    err_valid = err[mask]

    # 3. cubic 插值
    # 这里使用的是 a_valid, b_valid。
    # 此时，原本 error > 1 的区域变成了一块“空白”，
    # cubic 算法会根据这块空白周围的有效点，平滑地计算出该区域的插值。
    ERR_grid = griddata((a_valid, b_valid), err_valid, (A_grid, B_grid), method='cubic')

    # 绘制二维热力图
    # 建议加上 vmin 和 vmax 锁定颜色范围，这样对比更清晰（例如锁定在0-1）
    im = ax.imshow(ERR_grid, origin='lower', 
                   extent=[grid_a[0], grid_a[-1], grid_b[0], grid_b[-1]],
                   aspect='auto', cmap='viridis')

    ax.set_title(method, fontsize=14)
    ax.set_xlabel("a", fontsize=14)
    ax.set_ylabel("b", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="error")

# plt.show()
plt.savefig("ab_error_interpolated_newbaseline.pdf", format='pdf', bbox_inches='tight', dpi=300)