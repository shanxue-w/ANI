import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import cm

import os 
os.makedirs("../../results/Fitzhugh-Nagumo", exist_ok=True)

plt.rcParams.update({
    "font.size": 14,       
    "axes.labelsize": 14,  
    "xtick.labelsize": 14, 
    "ytick.labelsize": 14, 
    "legend.fontsize": 14, 
})

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

fig, ax = plt.subplots(figsize=(4, 4))
ax.contourf(u_4th_t1, levels=100, cmap='viridis', origin='lower')
ax.axis('off')
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
ax.set_rasterized(True)
plt.savefig("../../results/Fitzhugh-Nagumo/u_4th_t1_contourf_clean.pdf", format='pdf', bbox_inches='tight', dpi=300, transparent=True, pad_inches=0,)
plt.close()

methods = ["True", "Pretrained", "ANI-2", "ANI-4"]

import matplotlib.ticker as mticker
def plot_uv_separate(u_true, u_pred_list, v_true, v_pred_list, methods, t_label, X, Y):
    fig = plt.figure(figsize=(4.5*(len(methods)+1), 8))
    gs = fig.add_gridspec(2, len(methods)+2, width_ratios=[1]*(len(methods)+1)+[0.05])

    umin, umax = min(u_true.min(), *(p.min() for p in u_pred_list)), \
                max(u_true.max(), *(p.max() for p in u_pred_list))
    uerr_max = max(np.abs(u_true - p).max() for p in u_pred_list)

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

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")  
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
    plt.savefig(f"../../results/Fitzhugh-Nagumo/u_grid_{t_label}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


    fig = plt.figure(figsize=(4.5*(len(methods)+1), 8))
    gs = fig.add_gridspec(2, len(methods)+2, width_ratios=[1]*(len(methods)+1)+[0.05])
    v_min, v_max = min(v_true.min(), *(p.min() for p in v_pred_list)), \
                max(v_true.max(), *(p.max() for p in v_pred_list))
    v_err_max = max(np.abs(v_true - p).max() for p in v_pred_list)

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

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")  
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
    plt.savefig(f"../../results/Fitzhugh-Nagumo/v_grid_{t_label}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

methods = ["Pretrained", "ANI-2", "ANI-4"]
plot_uv_separate(
    u_true_t1, [u_Pretrained_t1, u_2th_t1, u_4th_t1],
    v_true_t1, [v_Pretrained_t1, v_2th_t1, v_4th_t1],
    methods,
    "1.0s",
    X, Y
)