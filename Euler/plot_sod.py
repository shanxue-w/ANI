import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 18,        # 全局字体大小
    "axes.labelsize": 22,   # 坐标轴标签字体大小
    "xtick.labelsize": 18,  # x 轴刻度字体大小
    "ytick.labelsize": 18,  # y 轴刻度字体大小
    "legend.fontsize": 18,  # 图例字体大小
})

Location, rho_sod, u_sod, p_sod, e_sod = np.loadtxt(
    'Exact/res0.csv', 
    delimiter=',', 
    skiprows=1, 
    unpack=True
)

rho_2th = np.load("2th/rho_sod_2th.npy")
u_2th = np.load("2th/u_sod_2th.npy")
p_2th = np.load("2th/p_sod_2th.npy")

rho_tvd = np.load("2th/rho_sod_tvd.npy")
u_tvd = np.load("2th/u_sod_tvd.npy")
p_tvd = np.load("2th/p_sod_tvd.npy")

rho_4th = np.load("4th/rho_sod_4th.npy")
u_4th = np.load("4th/u_sod_4th.npy")
p_4th = np.load("4th/p_sod_4th.npy")


error_rho_4th = np.abs(rho_4th - rho_sod)
error_u_4th   = np.abs(u_4th - u_sod)
error_p_4th   = np.abs(p_4th - p_sod)

error_rho_2th = np.abs(rho_2th - rho_sod)
error_u_2th   = np.abs(u_2th - u_sod)
error_p_2th   = np.abs(p_2th - p_sod)

error_rho_tvd = np.abs(rho_tvd - rho_sod)
error_u_tvd   = np.abs(u_tvd - u_sod)
error_p_tvd   = np.abs(p_tvd - p_sod)

# x = np.arange(len(error_rho_4th)) 
x = np.linspace(0, 1, len(error_rho_4th))

fig, axs = plt.subplots(1, 3, figsize=(22, 7))

# Density
axs[0].plot(x, error_rho_tvd, label="TVD", color="#0072B2", linewidth=3)
axs[0].plot(x, error_rho_2th, label="ANI-2", color="#009E73", linewidth=3)
axs[0].plot(x, error_rho_4th, label="ANI-4", color="#E69F00", linewidth=3)
axs[0].set_ylabel(r"$|\rho - \rho_{\mathrm{ref}}|$", fontsize=22)
axs[0].set_xlabel('x', fontsize=22)
# axs[0].set_title("Density Error")
axs[0].grid(True, alpha=0.3)
axs[0].legend()

# Velocity
axs[1].plot(x, error_u_tvd, label="TVD", color="#0072B2", linewidth=3)
axs[1].plot(x, error_u_2th, label="ANI-2", color="#009E73", linewidth=3)
axs[1].plot(x, error_u_4th, label="ANI-4", color="#E69F00", linewidth=3)
axs[1].set_ylabel(r"$|u - u_{\mathrm{ref}}|$", fontsize=22)
axs[1].set_xlabel('x', fontsize=22)
axs[1].set_ylim([-4e-5, 8.5e-4])
axs[1].set_yticks(np.arange(0.0, 8e-4, 1.3e-4))
# axs[1].set_title("Velocity Error")
axs[1].grid(True, alpha=0.3)
axs[1].legend()

# Pressure
axs[2].plot(x, error_p_tvd, label="TVD", color="#0072B2", linewidth=3)
axs[2].plot(x, error_p_2th, label="ANI-2", color="#009E73", linewidth=3)
axs[2].plot(x, error_p_4th, label="ANI-4", color="#E69F00", linewidth=3)
axs[2].set_ylabel(r"$|p - p_{\mathrm{ref}}|$", fontsize=22)
axs[2].set_xlabel('x', fontsize=22)
# axs[2].set_title("Pressure Error")
axs[2].grid(True, alpha=0.3)
axs[2].legend()

plt.tight_layout()
plt.savefig("error_comparison_sod.pdf", format='pdf', bbox_inches='tight', dpi=300)
# plt.show()




from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator

colors =['#0072B2', '#009E73', '#E69F00', '#D55E00', '#CC79A7']  # True + three models

def plot_models_side_by_side(rho_ref, u_ref, p_ref,
                             rho_models, u_models, p_models,
                             labels, axs):
    # axs 是 1x3 array
    
    # ρ
    axs[0].plot(x, rho_ref, color=colors[0], label='Reference', linewidth=3)
    for i, rho_model in enumerate(rho_models):
        axs[0].plot(x, rho_model, color=colors[i+1], linestyle='--', label=labels[i], linewidth=3)
    axs[0].set_ylabel(r'$\rho$', fontsize=22)
    axs[0].set_xlabel('x', fontsize=22)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    axins = inset_axes(axs[0], width="35%", height="35%", loc="lower left")  
    # 在放大图里重新画曲线
    axins.plot(x, rho_ref, color=colors[0])
    for i, rho_model in enumerate(rho_models):
        axins.plot(x, rho_model, color=colors[i+1], linestyle='--', linewidth=3)
    # 设置放大区域
    # x1, x2 = 0.5, 0.6   # 放大横坐标区间
    # y1, y2 = 0.3, 0.4   # 放大纵坐标区间（你可以调整范围）
    x1, x2 = 0.65, 0.75   # 放大横坐标区间
    y1, y2 = 0.2, 0.5   # 放大纵坐标区间（你可以调整范围）
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(False)
    axins.set_xticks([])
    axins.set_yticks([])
    # 用标记框连接主图和插图
    mark_inset(axs[0], axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # u
    axs[1].plot(x, u_ref, color=colors[0], label='Reference', linewidth=3)
    for i, u_model in enumerate(u_models):
        axs[1].plot(x, u_model, color=colors[i+1], linestyle='--', label=labels[i], linewidth=3)
    axs[1].set_ylabel(r'$u$', fontsize=22)
    axs[1].set_xlabel('x', fontsize=22)
    # axs[1].set_ylim([-0.05, 1.45])
    # axs[1].set_ylim([-0.10, 0.10])
    # axs[1].set_yticks(np.arange(0.0, 1.4, 0.3))
    axs[1].grid(True, alpha=0.3)
    # axs[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    # axs[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axs[1].legend()

    axins = inset_axes(axs[1], width="35%", height="35%", loc="lower left")
    axins.plot(x, u_ref, color=colors[0])
    for i, u_model in enumerate(u_models):
        axins.plot(x, u_model, color=colors[i+1], linestyle='--', linewidth=3)
    # x1, x2 = 0.31, 0.36
    # y1, y2 = 1.3, 1.4
    x1, x2 = 0.47, 0.57
    y1, y2 = 0.85, 0.95
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(False)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(axs[1], axins, loc1=2, loc2=4, fc="none", ec="0.5", linewidth=3)
    
    # p
    axs[2].plot(x, p_ref, color=colors[0], label='Reference', linewidth=3)
    for i, p_model in enumerate(p_models):
        axs[2].plot(x, p_model, color=colors[i+1], linestyle='--', label=labels[i], linewidth=3)
    axs[2].set_ylabel(r'$p$', fontsize=22)
    axs[2].set_xlabel('x', fontsize=22)
    # axs[2].set_ylim([0.90, 1.10])
    axs[2].grid(True, alpha=0.3)
    # axs[2].xaxis.set_major_locator(MaxNLocator(nbins=5))
    # axs[2].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axs[2].legend()

    # axins = inset_axes(axs[2], width="35%", height="35%", loc="lower left")
    # axins.plot(x, p_ref, color=colors[0])
    # for i, p_model in enumerate(p_models):
    #     axins.plot(x, p_model, color=colors[i+1], linestyle='--', linewidth=3)
    # x1, x2 = 0.3, 0.4
    # y1, y2 = 0.4, 0.5
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.grid(False)
    # axins.set_xticks([])
    # axins.set_yticks([])
    # mark_inset(axs[2], axins, loc1=2, loc2=4, fc="none", ec="0.5")

    for ax in axs:
        ax.set_aspect('auto')
        ax.xaxis.set_major_locator(MultipleLocator(0.1))  # 统一 x 轴主刻度间隔
        ax.yaxis.set_major_locator(MultipleLocator(0.2))  # 统一 y 轴主刻度间隔

# 创建 1行3列子图
fig, axs = plt.subplots(1, 3, figsize=(22, 7))

plot_models_side_by_side(
    rho_sod, u_sod, p_sod,
    rho_models=[rho_tvd, rho_2th, rho_4th],
    u_models=[u_tvd, u_2th, u_4th],
    p_models=[p_tvd, p_2th, p_4th],
    labels=['TVD', 'ANI-2', 'ANI-4'],
    axs=axs
)


plt.tight_layout()
# plt.show()
plt.savefig("solution_comparison_sod.pdf", format='pdf', bbox_inches='tight', dpi=300)

fig = plt.figure(figsize=(2, 2))
plt.plot(rho_sod, color="#0072B2")
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig("data_1_sod.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

plt.figure(figsize=(2, 2))
plt.plot(u_sod, color="#0072B2")
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig("data_2_sod.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

plt.figure(figsize=(2, 2))
plt.plot(p_sod, color="#0072B2")
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig("data_3_sod.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.close()



data_2th_sod = np.load("2th/sod_weno.npy")
T, N, _ = data_2th_sod.shape

rho = data_2th_sod[:, :, 0]     # [T, N]
u   = data_2th_sod[:, :, 1]
p   = data_2th_sod[:, :, 2]

def plot_clean(field, filename, cmap="viridis"):
    plt.figure(figsize=(4, 4))
    plt.imshow(field, aspect="auto", origin="lower", cmap=cmap, interpolation='nearest')
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()

plot_clean(rho, "rho_clean.pdf", cmap="gray")
plot_clean(u, "u_clean.pdf", cmap="gray")
plot_clean(p, "p_clean.pdf", cmap="gray")

