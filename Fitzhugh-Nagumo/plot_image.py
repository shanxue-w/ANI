import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import cm

plt.rcParams.update({
    "font.size": 14,        # 全局字体大小
    "axes.labelsize": 14,   # 坐标轴标签字体大小
    "xtick.labelsize": 14,  # x 轴刻度字体大小
    "ytick.labelsize": 14,  # y 轴刻度字体大小
    "legend.fontsize": 14,  # 图例字体大小
})
# 假设你已经有这些变量 (numpy 2D arrays)
# u_true_t0, u_2th_t0, u_4th_t0, u_Pretrained_t0
# v_true_t0, v_2th_t0, v_4th_t0, v_Pretrained_t0
# 同理 t=1s 的：u_true_t1, ...

x = np.linspace(0, 1, 128+1)[:-1]
y = np.linspace(0, 1, 128+1)[:-1]
X,Y = np.meshgrid(x, y, indexing='ij')

u_true_t0 = np.load('2th_new/u0_true.npy')
v_true_t0 = np.load('2th_new/v0_true.npy')

u_2th_t0 = np.load('2th_new/u0_2th.npy')
v_2th_t0 = np.load('2th_new/v0_2th.npy')

u_4th_t0 = np.load('4th_new/u0_4th.npy')
v_4th_t0 = np.load('4th_new/v0_4th.npy')

u_Pretrained_t0 = np.load('2th_new/u0_A.npy')
v_Pretrained_t0 = np.load('2th_new/v0_A.npy')

u_true_t1 = np.load('2th_new/u99_true.npy')
v_true_t1 = np.load('2th_new/v99_true.npy')

u_2th_t1 = np.load('2th_new/u99_2th.npy')
v_2th_t1 = np.load('2th_new/v99_2th.npy')

u_4th_t1 = np.load('4th_new/u99_4th.npy')
v_4th_t1 = np.load('4th_new/v99_4th.npy')

u_Pretrained_t1 = np.load('2th_new/u99_A.npy')
v_Pretrained_t1 = np.load('2th_new/v99_A.npy')

# fig, ax = plt.subplots(figsize=(4, 4))
fig = plt.figure(figsize=(4, 4), frameon=False)
ax = fig.add_axes([0, 0, 1, 1]) 
ax.contourf(u_4th_t1, levels=100, cmap='viridis', origin='lower')
# ax.contourf(u_4th_t1, levels=100, cmap='gray', origin='lower')
# plt.axis('off')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.rasterized(True)
ax.axis('off')
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
ax.set_rasterized(True)
plt.savefig("u_4th_t1_contourf_clean.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True, pad_inches=0,)
plt.close()

methods = ["True", "Pretrained", "ANI-2", "ANI-4"]

def plot_uv_grid(u_list, v_list, t_label, methods):
    fig = plt.figure(figsize=(3*len(methods), 5))
    gs = fig.add_gridspec(2, len(methods)+1, width_ratios=[1]*len(methods)+[0.05])

    # 统一颜色范围
    umin, umax = np.min(u_list), np.max(u_list)
    vmin, vmax = np.min(v_list), np.max(v_list)

    # 第一行：u
    axes_u = []
    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs[0, col])
        im_u = ax.contourf(X, Y, u_list[col], cmap='viridis', origin='lower',
                           levels=100, vmin=umin, vmax=umax)
        ax.set_rasterized(True)
        ax.set_title(f"{method} - u")
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')
        axes_u.append(ax)
    cax_u = fig.add_subplot(gs[0, -1])
    norm  = mcolors.Normalize(vmin=umin, vmax=umax)
    sm    = cm.ScalarMappable(norm=norm, cmap=im_u.cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax_u, label="u value")
    # fig.colorbar(im_u, cax=cax_u, label="u value")

    # 第二行：v
    axes_v = []
    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs[1, col])
        im_v = ax.contourf(X, Y, v_list[col], cmap='viridis', origin='lower',
                           levels=100, vmin=vmin, vmax=vmax)
        ax.set_rasterized(True)
        ax.set_title(f"{method} - v")
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')
        axes_v.append(ax)
    cax_v = fig.add_subplot(gs[1, -1])
    norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm    = cm.ScalarMappable(norm=norm, cmap=im_v.cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax_v, label="v value")
    # fig.colorbar(im_v, cax=cax_v, label="v value")

    plt.tight_layout()
    plt.savefig(f"uv_grid_{t_label}_new.pdf", format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

# t=0s
plot_uv_grid(
    [u_true_t0, u_Pretrained_t0, u_2th_t0, u_4th_t0],
    [v_true_t0, v_Pretrained_t0, v_2th_t0, v_4th_t0],
    "0.5s",
    methods
)

# t=1s
plot_uv_grid(
    [u_true_t1, u_Pretrained_t1, u_2th_t1, u_4th_t1],
    [v_true_t1, v_Pretrained_t1, v_2th_t1, v_4th_t1],
    "1s",
    methods
)

import matplotlib.ticker as mticker
def plot_uv_separate(u_true, u_pred_list, v_true, v_pred_list, methods, t_label, X, Y):
    """
    methods: ["Pretrained", "ANI-2", "ANI-4"]  # 不包含True
    u_true, v_true: [ny, nx]
    u_pred_list, v_pred_list: list of predicted fields, len = len(methods)
    """

    fig = plt.figure(figsize=(4.5*(len(methods)+1), 8))
    gs = fig.add_gridspec(2, len(methods)+2, width_ratios=[1]*(len(methods)+1)+[0.05])
    # axes = gs.subplots()

    # 共用 u 的颜色范围
    umin, umax = min(u_true.min(), *(p.min() for p in u_pred_list)), \
                max(u_true.max(), *(p.max() for p in u_pred_list))
    uerr_max = max(np.abs(u_true - p).max() for p in u_pred_list)

    # 第一行：True + Predictions
    # im0 = axes[0,0].contourf(X, Y, u_true, levels=100, cmap="viridis", vmin=umin, vmax=umax)
    # axes[0,0].set_title("True - u")
    # axes[0,0].set_rasterized(True)
    # axes[0,0].set_aspect('equal', adjustable='box')
    # axes[0,0].axis("off")
    ax = fig.add_subplot(gs[0, 0])
    im_u = ax.contourf(X, Y, u_true, levels=100, cmap="viridis", vmin=umin, vmax=umax)
    ax.set_title("True - u")
    ax.set_rasterized(True)
    ax.set_aspect('equal', adjustable='box')
    ax.axis("off")


    for j, (pred, method) in enumerate(zip(u_pred_list, methods), start=1):
        ax = fig.add_subplot(gs[0, j])
        im_u = ax.contourf(X, Y, pred, cmap='viridis', origin='lower',
                           levels=100, vmin=umin, vmax=umax)
        ax.set_rasterized(True)
        ax.set_title(f"{method} - u")
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')

    cax_u = fig.add_subplot(gs[0, -1])
    norm  = mcolors.Normalize(vmin=umin, vmax=umax)
    sm    = cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax_u, label="u value")
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.tick_params(left=False, right=True, labelleft=False, labelright=True)
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # 第二行：Error
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")  # 第一个空
    ax.set_aspect('equal', adjustable='box')
    for j, (pred, method) in enumerate(zip(u_pred_list, methods), start=1):
        err = np.abs(u_true - pred)
        ax = fig.add_subplot(gs[1, j])
        im_u = ax.contourf(X, Y, err, cmap='viridis', origin='lower',
                           levels=100, vmin=0, vmax=uerr_max)
        ax.set_rasterized(True)
        ax.set_title(f"Error {method} - u")
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')

    cax_u = fig.add_subplot(gs[1, -1])
    norm  = mcolors.Normalize(vmin=0, vmax=uerr_max)
    sm    = cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    fig.colorbar(sm, cax=cax_u, label="|u error|")

    plt.tight_layout()
    plt.savefig(f"u_grid_{t_label}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


    fig = plt.figure(figsize=(4.5*(len(methods)+1), 8))
    gs = fig.add_gridspec(2, len(methods)+2, width_ratios=[1]*(len(methods)+1)+[0.05])
    # axes = gs.subplots()

    # 共用 u 的颜色范围
    v_min, v_max = min(v_true.min(), *(p.min() for p in v_pred_list)), \
                max(v_true.max(), *(p.max() for p in v_pred_list))
    v_err_max = max(np.abs(v_true - p).max() for p in v_pred_list)

    # 第一行：True + Predictions
    # im0 = axes[0,0].contourf(X, Y, u_true, levels=100, cmap="viridis", vmin=umin, vmax=umax)
    # axes[0,0].set_title("True - u")
    # axes[0,0].set_rasterized(True)
    # axes[0,0].set_aspect('equal', adjustable='box')
    # axes[0,0].axis("off")
    ax = fig.add_subplot(gs[0, 0])
    im_v_ = ax.contourf(X, Y, v_true, levels=100, cmap="viridis", vmin=v_min, vmax=v_max)
    ax.set_title("True - v")
    ax.set_rasterized(True)
    ax.set_aspect('equal', adjustable='box')
    ax.axis("off")


    for j, (pred, method) in enumerate(zip(v_pred_list, methods), start=1):
        ax = fig.add_subplot(gs[0, j])
        im_v_ = ax.contourf(X, Y, pred, cmap='viridis', origin='lower',
                           levels=100, vmin=v_min, vmax=v_max)
        ax.set_rasterized(True)
        ax.set_title(f"{method} - v")
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')

    cax_v = fig.add_subplot(gs[0, -1])
    norm  = mcolors.Normalize(vmin=v_min, vmax=v_max)
    sm    = cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax_v, label="u value")
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.tick_params(left=False, right=True, labelleft=False, labelright=True)

    # 第二行：Error
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")  # 第一个空
    ax.set_aspect('equal', adjustable='box')
    for j, (pred, method) in enumerate(zip(v_pred_list, methods), start=1):
        err = np.abs(v_true - pred)
        ax = fig.add_subplot(gs[1, j])
        im_v = ax.contourf(X, Y, err, cmap='viridis', origin='lower',
                           levels=100, vmin=0, vmax=v_err_max)
        ax.set_rasterized(True)
        ax.set_title(f"Error {method} - v")
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')

    cax_v = fig.add_subplot(gs[1, -1])
    norm  = mcolors.Normalize(vmin=0, vmax=v_err_max)
    sm    = cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    fig.colorbar(sm, cax=cax_v, label="|v error|")

    plt.tight_layout()
    plt.savefig(f"v_grid_{t_label}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

methods = ["Pretrained", "ANI-2", "ANI-4"]
plot_uv_separate(
    u_true_t1, [u_Pretrained_t1, u_2th_t1, u_4th_t1],
    v_true_t1, [v_Pretrained_t1, v_2th_t1, v_4th_t1],
    methods,
    "1.0s",
    X, Y
)

methods = ["Pretrained", "ANI-2", "ANI-4"]

def plot_error_grid(u_true, v_true, u_methods, v_methods, t_label, methods,
                    signed=False, cmap_abs='viridis', cmap_signed='viridis'):
    """
    u_true, v_true: 2D numpy arrays
    u_methods, v_methods: lists of numpy arrays [u_2th, u_4th, u_Pretrained]
    signed: True → signed error; False → absolute error
    Each subplot has its own colorbar.
    """
    # compute error
    if signed:
        err_u = [m - u_true for m in u_methods]
        err_v = [m - v_true for m in v_methods]
        cmap = cmap_signed
    else:
        err_u = [np.abs(m - u_true) for m in u_methods]
        err_v = [np.abs(m - v_true) for m in v_methods]
        cmap = cmap_abs

    fig = plt.figure(figsize=(3*len(methods), 5))
    gs = fig.add_gridspec(2, len(methods)+1, width_ratios=[1]*len(methods)+[0.05])

    # 统一颜色范围
    umin, umax = np.min(err_u), np.max(err_u)
    vmin, vmax = np.min(err_v), np.max(err_v)

    # 第一行：u
    axes_u = []
    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs[0, col])
        im_u = ax.contourf(X, Y, err_u[col], cmap='viridis', origin='lower',
                           levels=100, vmin=umin, vmax=umax)
        ax.set_rasterized(True)
        ax.set_title(f"{method} — {'u error'}")
        ax.axis('off')
        ax.set_aspect('equal')
        axes_u.append(ax)
    cax_u = fig.add_subplot(gs[0, -1])
    # fig.colorbar(im_u, cax=cax_u)
    norm = mcolors.Normalize(vmin=umin, vmax=umax)
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)  # 用相同 norm
    sm.set_array([])  # 必须设置 array 才能 colorbar
    fig.colorbar(sm, cax=cax_u)

    # 第二行：v
    axes_v = []
    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs[1, col])
        im_v = ax.contourf(X, Y, err_v[col], cmap='viridis', origin='lower',
                           levels=100, vmin=vmin, vmax=vmax)
        ax.set_rasterized(True)
        ax.set_title(f"{method} — {'v error'}")
        ax.axis('off')
        ax.set_aspect('equal')
        axes_v.append(ax)
    cax_v = fig.add_subplot(gs[1, -1])
    # fig.colorbar(im_v, cax=cax_v)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)  # 用相同 norm
    sm.set_array([])  # 必须设置 array
    fig.colorbar(sm, cax=cax_v)

    # fig, axes = plt.subplots(2, len(methods), figsize=(4*len(methods), 8))
    # for col, method in enumerate(methods):
    #     # u
    #     # im_u = axes[0, col].imshow(err_u[col], origin='lower', cmap=cmap, aspect='auto')
    #     im_u = axes[0, col].contourf(X, Y, err_u[col], cmap=cmap, origin='lower', levels=20)
    #     axes[0, col].set_title(f"{method} — {'u error'}")
    #     axes[0, col].axis('off')
    #     plt.colorbar(im_u, ax=axes[0, col], fraction=0.046, pad=0.04)

    #     # v
    #     # im_v = axes[1, col].imshow(err_v[col], origin='lower', cmap=cmap, aspect='auto')
    #     im_v = axes[1, col].contourf(X, Y, err_v[col], cmap=cmap, origin='lower', levels=20)
    #     axes[1, col].set_title(f"{method} — {'v error'}")
    #     axes[1, col].axis('off')
    #     plt.colorbar(im_v, ax=axes[1, col], fraction=0.046, pad=0.04)

    # plt.tight_layout(rect=[0,0,1,0.95])
    # plt.show()
    plt.savefig(f"error_grid_{t_label}_new.pdf", format='pdf', bbox_inches='tight', dpi=300)

plot_error_grid(
    u_true_t0, v_true_t0,
    [u_Pretrained_t0, u_2th_t0, u_4th_t0],
    [v_Pretrained_t0, v_2th_t0, v_4th_t0],
    "0.5s",
    methods
)

plot_error_grid(
    u_true_t1, v_true_t1,
    [u_Pretrained_t1, u_2th_t1, u_4th_t1],
    [v_Pretrained_t1, v_2th_t1, v_4th_t1],
    "1s",
    methods
)