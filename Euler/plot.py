import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 18,        # 全局字体大小
    "axes.labelsize": 22,   # 坐标轴标签字体大小
    "xtick.labelsize": 18,  # x 轴刻度字体大小
    "ytick.labelsize": 18,  # y 轴刻度字体大小
    "legend.fontsize": 18,  # 图例字体大小
})

# Load error data
error_rho_4th = np.load("4th/error_rho_sod_4th.npy")
error_u_4th = np.load("4th/error_u_sod_4th.npy")
error_p_4th = np.load("4th/error_p_sod_4th.npy")

# error_tvd = np.load("error_tvd.npy")    # shape (128,3)
# error_2th = np.load("error_2th.npy")    # shape (128,3)

error_rho_2th = np.load("2th/error_rho_sod_2th.npy")
error_u_2th = np.load("2th/error_u_sod_2th.npy")
error_p_2th = np.load("2th/error_p_sod_2th.npy")

error_rho_tvd = np.load("2th/error_rho_sod_tvd.npy")
error_u_tvd = np.load("2th/error_u_sod_tvd.npy")
error_p_tvd = np.load("2th/error_p_sod_tvd.npy")

# x = np.arange(len(error_rho_4th)) 
x = np.linspace(0, 1, len(error_rho_4th))

fig, axs = plt.subplots(1, 3, figsize=(22, 7))

# Density
axs[0].plot(x, error_rho_tvd, label="TVD", color="#1f77b4", linewidth=2)
axs[0].plot(x, error_rho_2th, label="ANI-2", color="#2ca02c", linewidth=2)
axs[0].plot(x, error_rho_4th, label="ANI-4", color="#733497", linewidth=2)
axs[0].set_ylabel(r"$|\rho - \rho_{\mathrm{ref}}|$", fontsize=22)
axs[0].set_xlabel('x', fontsize=22)
# axs[0].set_title("Density Error")
axs[0].grid(True)
axs[0].legend()

# Velocity
axs[1].plot(x, error_u_tvd, label="TVD", color="#1f77b4", linewidth=2)
axs[1].plot(x, error_u_2th, label="ANI-2", color="#2ca02c", linewidth=2)
axs[1].plot(x, error_u_4th, label="ANI-4", color="#733497", linewidth=2)
axs[1].set_ylabel(r"$|u - u_{\mathrm{ref}}|$", fontsize=22)
axs[1].set_xlabel('x', fontsize=22)
# axs[1].set_ylim([-4e-5, 8.5e-4])
# axs[1].set_yticks(np.arange(0.0, 8e-4, 1.3e-4))
# axs[1].set_title("Velocity Error")
axs[1].grid(True)
axs[1].legend()

# Pressure
axs[2].plot(x, error_p_tvd, label="TVD", color="#1f77b4", linewidth=2)
axs[2].plot(x, error_p_2th, label="ANI-2", color="#2ca02c", linewidth=2)
axs[2].plot(x, error_p_4th, label="ANI-4", color="#733497", linewidth=2)
axs[2].set_ylabel(r"$|p - p_{\mathrm{ref}}|$", fontsize=22)
axs[2].set_xlabel('x', fontsize=22)
# axs[2].set_title("Pressure Error")
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.savefig("error_comparison_sod.pdf", format='pdf', bbox_inches='tight', dpi=300)
# plt.show()

rho_sod = np.load("2th/rho_sod.npy")
u_sod = np.load("2th/u_sod.npy")
p_sod = np.load("2th/p_sod.npy")

rho_2th = np.load("2th/rho_sod_2th.npy")
u_2th = np.load("2th/u_sod_2th.npy")
p_2th = np.load("2th/p_sod_2th.npy")

rho_tvd = np.load("2th/rho_sod_tvd.npy")
u_tvd = np.load("2th/u_sod_tvd.npy")
p_tvd = np.load("2th/p_sod_tvd.npy")

rho_4th = np.load("4th/rho_sod_4th.npy")
u_4th = np.load("4th/u_sod_4th.npy")
p_4th = np.load("4th/p_sod_4th.npy")

# def plot_row(rho_ref, u_ref, p_ref, rho_model, u_model, p_model, label, ax_row):
#     ax_row[0].plot(x, rho_ref, color='tab:blue', label=r'$\rho$', linewidth=2)
#     ax_row[0].plot(x, rho_model, color='tab:red', linestyle='--', label=f'{label}', linewidth=2)
#     ax_row[0].set_ylabel(r'$\rho$', fontsize=18)
#     ax_row[0].grid(True)
    
#     ax_row[1].plot(x, u_ref, color='tab:blue', label=r'$u$', linewidth=2)
#     ax_row[1].plot(x, u_model, color='tab:red', linestyle='--', label=f'{label}', linewidth=2)
#     # ax_row[1].set_ylim([-0.1, 0.1])
#     ax_row[1].set_ylabel(r'$u$', fontsize=18)
#     ax_row[1].grid(True)
    
#     ax_row[2].plot(x, p_ref, color='tab:blue', label=r'$p$', linewidth=2)
#     ax_row[2].plot(x, p_model, color='tab:red', linestyle='--', label=f'{label}', linewidth=2)
#     # ax_row[2].set_ylim([0.9, 1.10])
#     ax_row[2].set_ylabel(r'$p$', fontsize=18)
#     ax_row[2].grid(True)
    
#     for ax in ax_row:
#         ax.set_xlabel('x', fontsize=18)
#         ax.legend()


# fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# plot_row(rho_sod, u_sod, p_sod, rho_tvd, u_tvd, p_tvd, 'tvd', axes[0])
# plot_row(rho_sod, u_sod, p_sod, rho_2th, u_2th, p_2th, 'ANI-2', axes[1])
# plot_row(rho_sod, u_sod, p_sod, rho_4th, u_4th, p_4th, 'ANI-4', axes[2])

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator

colors = ["#1f77b4", "#2ca02c", "#733497", "#ff7f0e"]  # True + three models

def plot_models_side_by_side(rho_ref, u_ref, p_ref,
                             rho_models, u_models, p_models,
                             labels, axs):
    # axs 是 1x3 array
    
    # ρ
    axs[0].plot(x, rho_ref, color=colors[0], label='True')
    for i, rho_model in enumerate(rho_models):
        axs[0].plot(x, rho_model, color=colors[i+1], linestyle='--', label=labels[i], linewidth=2)
    axs[0].set_ylabel(r'$\rho$', fontsize=22)
    axs[0].set_xlabel('x', fontsize=22)
    axs[0].grid(True)
    # axs[0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    # axs[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axs[0].legend()
    
    axins = inset_axes(axs[0], width="35%", height="35%", loc="lower left")  
    # 在放大图里重新画曲线
    axins.plot(x, rho_ref, color=colors[0])
    for i, rho_model in enumerate(rho_models):
        axins.plot(x, rho_model, color=colors[i+1], linestyle='--', linewidth=2)
    # # 设置放大区域
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
    axs[1].plot(x, u_ref, color=colors[0], label='True')
    for i, u_model in enumerate(u_models):
        axs[1].plot(x, u_model, color=colors[i+1], linestyle='--', label=labels[i], linewidth=2)
    axs[1].set_ylabel(r'$u$', fontsize=22)
    axs[1].set_xlabel('x', fontsize=22)
    # axs[1].set_ylim([-0.05, 1.45])
    # axs[1].set_yticks(np.arange(0.0, 1.4, 0.3))
    axs[1].grid(True)
    # axs[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    # axs[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axs[1].legend()

    axins = inset_axes(axs[1], width="35%", height="35%", loc="lower left")
    axins.plot(x, u_ref, color=colors[0])
    for i, u_model in enumerate(u_models):
        axins.plot(x, u_model, color=colors[i+1], linestyle='--', linewidth=2)
    # x1, x2 = 0.31, 0.36
    # y1, y2 = 1.3, 1.4
    x1, x2 = 0.47, 0.57
    y1, y2 = 0.85, 0.95
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.grid(False)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(axs[1], axins, loc1=2, loc2=4, fc="none", ec="0.5", linewidth=2)
    
    # p
    axs[2].plot(x, p_ref, color=colors[0], label='True')
    for i, p_model in enumerate(p_models):
        axs[2].plot(x, p_model, color=colors[i+1], linestyle='--', label=labels[i], linewidth=2)
    axs[2].set_ylabel(r'$p$', fontsize=22)
    axs[2].set_xlabel('x', fontsize=22)
    # axs[2].set_ylim([0.9, 1.10])
    axs[2].grid(True)
    # axs[2].xaxis.set_major_locator(MaxNLocator(nbins=5))
    # axs[2].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axs[2].legend()

    # axins = inset_axes(axs[2], width="35%", height="35%", loc="lower left")
    # axins.plot(x, p_ref, color=colors[0])
    # for i, p_model in enumerate(p_models):
    #     axins.plot(x, p_model, color=colors[i+1], linestyle='--', linewidth=2)
    # x1, x2 = 0.3, 0.4
    # y1, y2 = 0.4, 0.5
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.grid(False)
    # axins.set_xticks([])
    # axins.set_yticks([])
    # mark_inset(axs[2], axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # for ax in axs:
    #     ax.set_aspect('auto')
    #     ax.xaxis.set_major_locator(MultipleLocator(0.1))  # 统一 x 轴主刻度间隔
    #     ax.yaxis.set_major_locator(MultipleLocator(0.2))  # 统一 y 轴主刻度间隔

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

# plt.show()

plt.tight_layout()
# plt.show()
plt.savefig("solution_comparison_sod.pdf", format='pdf', bbox_inches='tight', dpi=300)

fig = plt.figure(figsize=(2, 2))
plt.plot(rho_sod, color="#1f77b4")
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig("data_1_sod.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

plt.figure(figsize=(2, 2))
plt.plot(u_sod, color="#1f77b4")
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig("data_2_sod.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

plt.figure(figsize=(2, 2))
plt.plot(p_sod, color="#1f77b4")
plt.axis("off")
plt.gca().set_frame_on(False)
plt.tight_layout()
plt.savefig("data_3_sod.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True)
plt.close()



# data_2th_sod = np.load("2th/sod_2th.npy")
# T, N, _ = data_2th_sod.shape

# rho = data_2th_sod[:, :, 0]     # [T, N]
# u   = data_2th_sod[:, :, 1]
# p   = data_2th_sod[:, :, 2]

# def plot_clean(field, filename, cmap="viridis"):
#     plt.figure(figsize=(4, 4))
#     plt.imshow(field, aspect="auto", origin="lower", cmap=cmap, interpolation='nearest')
#     plt.axis('off')  # 关闭坐标轴
#     plt.tight_layout(pad=0)
#     plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0, format='pdf')
#     plt.close()

# plot_clean(rho, "rho_2th_clean_sod.pdf", cmap="viridis")
# plot_clean(u, "u_2th_clean_sod.pdf", cmap="plasma")
# plot_clean(p, "p_2th_clean_sod.pdf", cmap="inferno")



def pri_to_con(pri, gamma=1.4):
    # pri: [N, 3]  rho, u, p
    rho = pri[:, 0]
    u = pri[:, 1]
    p = pri[:, 2]
    cons1 = rho
    cons2 = rho * u
    cons3 = p / (gamma - 1) + 0.5 * rho * u**2
    cons = np.stack([cons1, cons2, cons3], axis=-1)
    return cons

def compute_global(pri_traj, gamma=1.4):
    """
    pri_traj: (T, N, 3)  -> (rho, u, p)
    return: M(t), P(t), E(t)  (all shape (T,))
    """
    T, N, _ = pri_traj.shape
    dx = 1.0 / N
    cons_traj = np.array([pri_to_con(pri_traj[t], gamma) for t in range(T)])  # (T, N, 3)
    rho, mom, E = cons_traj[:, :, 0], cons_traj[:, :, 1], cons_traj[:, :, 2]

    M = np.sum(rho, axis=1) * dx
    P = np.sum(mom, axis=1) * dx
    Etot = np.sum(E, axis=1) * dx
    return M, P, Etot

pri_sod_ref = np.load("2th/sod_weno.npy")   # shape (T, N, 3)
pri_sod_2th = np.load("2th/sod_2th.npy")
pri_sod_4th = np.load("4th/sod_4th.npy")
pri_sod_tvd = np.load("2th/sod_tvd.npy")

M_ref, P_ref, E_ref = compute_global(pri_sod_ref)
M_2th, P_2th, E_2th = compute_global(pri_sod_2th)
M_4th, P_4th, E_4th = compute_global(pri_sod_4th)
M_tvd, P_tvd, E_tvd = compute_global(pri_sod_tvd)

# convert to relative error respect to ref
err_M_2th = np.abs(M_2th - M_ref) 
err_P_2th = np.abs(P_2th - P_ref)
err_E_2th = np.abs(E_2th - E_ref)

err_M_4th = np.abs(M_4th - M_ref)
err_P_4th = np.abs(P_4th - P_ref) 
err_E_4th = np.abs(E_4th - E_ref)

err_M_tvd = np.abs(M_tvd - M_ref)
err_P_tvd = np.abs(P_tvd - P_ref)
err_E_tvd = np.abs(E_tvd - E_ref)

time = 2e-3 * np.arange(len(M_ref))
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

axs[0].plot(time, err_M_tvd, label='TVD', linewidth=2)
axs[0].plot(time, err_M_2th, label='ANI-2', linewidth=2)
axs[0].plot(time, err_M_4th, label='ANI-4', linewidth=2)
axs[0].set_ylabel(r'Error in Mass')
axs[0].set_xlabel('Time')
axs[0].set_yscale('log')
axs[0].grid(True)
axs[0].legend()
axs[0].set_title('Mass Conservation Error')


axs[1].plot(time, err_P_tvd, label='TVD', linewidth=2)
axs[1].plot(time, err_P_2th, label='ANI-2', linewidth=2)
axs[1].plot(time, err_P_4th, label='ANI-4', linewidth=2)
axs[1].set_ylabel(r'Error in Momentum')
axs[1].set_xlabel('Time')
axs[1].set_yscale('log')
axs[1].grid(True)
axs[1].legend()
axs[1].set_title('Momentum Conservation Error')

axs[2].plot(time, err_E_tvd, label='TVD', linewidth=2)
axs[2].plot(time, err_E_2th, label='ANI-2', linewidth=2)
axs[2].plot(time, err_E_4th, label='ANI-4', linewidth=2)
axs[2].set_ylabel(r'Error in Energy')
axs[2].set_xlabel('Time')
axs[2].set_yscale('log')
axs[2].grid(True)
axs[2].legend()
axs[2].set_title('Energy Conservation Error')

plt.tight_layout()
plt.savefig("conservation_error_sod.pdf", format='pdf', bbox_inches='tight', dpi=300)