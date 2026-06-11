# Plot helpers for 2th/3th/4th/resnet outputs
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

def plot_error(ANI_2th, ANI_4th, resnet, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(resnet[1:201], label='Baseline', color="#0072B2", linewidth=2)
    plt.plot(ANI_2th[1:201], label='ANI-2', color="#009E73", linewidth=2)
    # plt.plot(ANI_3th[:1000], label='ANI_3th')
    plt.plot(ANI_4th[1:201], label='ANI-4', color="#E69F00", linewidth=2)
    plt.legend()
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.yscale('log')
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

abs_ANI_2th = np.loadtxt('2th_h32/rel_test_errors_small.txt')
abs_ANI_4th = np.loadtxt('4th_h32/rel_test_errors_small.txt')
abs_resnet    = np.loadtxt('NeuralRK4_h32/rel_test_errors_small_test.txt')

plot_error(abs_ANI_2th, abs_ANI_4th, abs_resnet, 'rel_error_200_new.pdf')

def plot_trajectory(traj, filename, title='Trajectory'):
    # 3D trajectory
    x = traj[:10001, 0]
    y = traj[:10001, 1]
    z = traj[:10001, 2]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(x, y, z, color="#0072B2", linewidth=2)
    ax.set_xlabel("x", fontsize=14, labelpad=5)
    ax.set_ylabel("y", fontsize=14, labelpad=5)
    ax.set_zlabel("z", fontsize=14, labelpad=5)
    # ax.tick_params(axis='both', which='major', direction='in', labelsize=8)
    # ax.tick_params(axis='z', which='major', direction='in', labelsize=10)


    ax.grid(True, color='0.8', linestyle='--', linewidth=0.5)  # light gray dashed grid

    # ax.view_init(elev=30, azim=45)  # adjust view_init for best angle

    plt.tight_layout()
    # plt.savefig(filename, format='pdf')
    plt.savefig(
        filename,
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.02
    )
    plt.close()


lorenz_basedata = np.load('dataset/lorenz_true_plot.npy')
plot_trajectory(lorenz_basedata, 'lorenz_true_plot.pdf')

lorenz_truedata = np.load('dataset/lorenz_stenflo_plot.npy')
plot_trajectory(lorenz_truedata, 'lorenz_stenflo_plot.pdf', title='Lorenz Stenflo True Trajectory')

lorenz_4thdata = np.load('dataset/lorenz_4th_plot.npy')
plot_trajectory(lorenz_4thdata, 'lorenz_4th_plot.pdf', title='Lorenz Stenflo 4th Order Neural ODE')

lorenz_2thdata = np.load('dataset/lorenz_2th_plot.npy')
plot_trajectory(lorenz_2thdata, 'lorenz_2th_plot.pdf', title='Lorenz Stenflo 2th Order Neural ODE')

lorenz_baselinedata = np.load('dataset/lorenz_base_plot.npy')
plot_trajectory(lorenz_baselinedata, 'lorenz_base_plot.pdf', title='Lorenz Stenflo Baseline')


cm = 1 / 2.54
fig = plt.figure(figsize=(2*cm, 2*cm), frameon=False)
ax = fig.add_axes([0, 0, 1, 1], projection='3d') 
ax.plot(lorenz_truedata[:,0], lorenz_truedata[:,1], lorenz_truedata[:,2], color="#0072B2", linewidth=0.2) 
# ax.plot(lorenz_truedata[:,0], lorenz_truedata[:,1], lorenz_truedata[:,2], color="gray")
ax.set_axis_off()
ax.grid(False)
ax.set_proj_type('ortho')
xyz_min = lorenz_truedata.min(axis=0)
xyz_max = lorenz_truedata.max(axis=0)
ax.set_xlim(xyz_min[0], xyz_max[0])
ax.set_ylim(xyz_min[1], xyz_max[1])
ax.set_zlim(xyz_min[2], xyz_max[2])
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
plt.savefig("data_3d_new.pdf", format='pdf', bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
plt.close()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter


def mm_to_inch(mm):
    return mm / 25.4


# ============================================================
# Nature-style global settings
# ============================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,

    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# ============================================================
# Styles
# ============================================================
COLORS = {
    "Baseline": "#0072B2",
    "ANI-2": "#009E73",
    "ANI-4": "#E69F00",
}

LINESTYLES = {
    "Baseline": "-",
    "ANI-2": "--",
    "ANI-4": "-.",
}

MARKERS = {
    "Baseline": "o",
    "ANI-2": "s",
    "ANI-4": "^",
}


# ============================================================
# Helpers
# ============================================================
def add_panel_label(ax, label, x=-0.16, y=1.08):
    """
    For 2D axes.
    """
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left"
    )


def add_panel_label_3d(ax, label, x=-0.12, y=1.02):
    """
    For 3D axes.
    """
    ax.text2D(
        x, y, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left"
    )


def style_3d_axes(ax):
    ax.set_xlabel("x", labelpad=-10)
    ax.set_ylabel("y", labelpad=-10)
    ax.set_zlabel("z", labelpad=-10)

    ax.tick_params(axis="both", pad=-5)
    ax.tick_params(axis="z", pad=-5)

    ax.grid(True, linestyle="--", linewidth=0.30, alpha=0.30)
    ax.view_init(elev=28, azim=-60)


def set_3d_limits(ax, lims):
    ax.set_xlim(lims[0][0], lims[0][1])
    ax.set_ylim(lims[1][0], lims[1][1])
    ax.set_zlim(lims[2][0], lims[2][1])


def get_shared_limits(*arrays, pad_ratio=0.03):
    """
    Compute shared 3D plotting limits from the first three coordinates
    of each trajectory. This also handles 4D Lorenz--Stenflo data.
    """
    xyz_arrays = []

    for arr in arrays:
        arr = np.asarray(arr)

        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(
                f"Each trajectory must have shape [N, d] with d >= 3, but got {arr.shape}."
            )

        xyz_arrays.append(arr[:, :3])

    all_data = np.concatenate(xyz_arrays, axis=0)

    mins = all_data.min(axis=0)
    maxs = all_data.max(axis=0)

    spans = maxs - mins
    spans = np.where(spans == 0, 1.0, spans)

    mins = mins - pad_ratio * spans
    maxs = maxs + pad_ratio * spans

    return [(mins[i], maxs[i]) for i in range(3)]


# ============================================================
# Panel a: relative error
# ============================================================
def plot_error_panel(ax, ani2, ani4, baseline):
    steps = np.arange(1, 201)

    series = [
        ("Baseline", baseline[1:201]),
        ("ANI-2", ani2[1:201]),
        ("ANI-4", ani4[1:201]),
    ]

    for label, y in series:
        ax.plot(
            steps,
            y,
            label=label,
            color=COLORS[label],
            linestyle=LINESTYLES[label],
            linewidth=0.9,
            marker=MARKERS[label],
            markersize=1.7,
            markevery=25,
        )
    ax.set_xlabel("Time Step")
    ax.set_ylabel(r"Relative $L^2$ error")
    ax.set_yscale("log")

    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False, handlelength=2.5)

    add_panel_label(ax, "a")


# ============================================================
# 3D trajectory panels
# ============================================================
def plot_traj_panel(ax, traj, title, lims, label):
    traj = np.asarray(traj)

    if traj.ndim != 2 or traj.shape[1] < 3:
        raise ValueError(
            f"Trajectory must have shape [N, d] with d >= 3, but got {traj.shape}."
        )

    x = traj[:10001, 0]
    y = traj[:10001, 1]
    z = traj[:10001, 2]

    ax.plot3D(x, y, z, color="#0072B2", linewidth=0.65)

    style_3d_axes(ax)
    set_3d_limits(ax, lims)
    ax.set_title(title, pad=2)

    try:
        ax.set_box_aspect((1, 1, 0.8), zoom=1.12)
    except TypeError:
        ax.set_box_aspect((1, 1, 0.8))
    except Exception:
        pass

    add_panel_label_3d(ax, label)


# ============================================================
# Main
# ============================================================
def main():
    # ---------------- Load data ----------------
    abs_ani2 = np.loadtxt("2th_h32/rel_test_errors_small.txt")
    abs_ani4 = np.loadtxt("4th_h32/rel_test_errors_small.txt")
    abs_base = np.loadtxt("NeuralRK4_h32/rel_test_errors_small_test.txt")

    lorenz_ref = np.load("dataset/lorenz_stenflo_plot.npy")
    lorenz_prior = np.load("dataset/lorenz_true_plot.npy")
    lorenz_base = np.load("dataset/lorenz_base_plot.npy")
    lorenz_ani4 = np.load("dataset/lorenz_4th_plot.npy")

    lims = get_shared_limits(lorenz_ref, lorenz_prior, lorenz_base, lorenz_ani4)

    width_mm = 180
    height_mm = 105

    fig = plt.figure(figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)))

    # Top row
    ax_a = fig.add_axes([0.085, 0.555, 0.430, 0.375])
    ax_b = fig.add_axes([0.555, 0.500, 0.425, 0.465], projection="3d")

    # Bottom row: right edge now reaches about 0.975
    ax_c = fig.add_axes([0.025, 0.070, 0.300, 0.375], projection="3d")
    ax_d = fig.add_axes([0.350, 0.070, 0.300, 0.375], projection="3d")
    ax_e = fig.add_axes([0.675, 0.070, 0.300, 0.375], projection="3d")

    plot_error_panel(ax_a, abs_ani2, abs_ani4, abs_base)
    plot_traj_panel(ax_b, lorenz_ref, "Reference", lims, "b")
    plot_traj_panel(ax_c, lorenz_prior, "Prior", lims, "c")
    plot_traj_panel(ax_d, lorenz_base, "Additive gray-box baseline", lims, "d")
    plot_traj_panel(ax_e, lorenz_ani4, "ANI-4", lims, "e")

    fig.savefig(
        "lorenz_combined_nature.pdf",
        format="pdf",
        bbox_inches=None,
        dpi=600,
    )

    fig.savefig(
        "lorenz_combined_nature.png",
        format="png",
        dpi=600,
        bbox_inches=None,
    )

    plt.close(fig)

    print("Saved: lorenz_combined_nature.pdf")
    print("Saved: lorenz_combined_nature.png")


if __name__ == "__main__":
    main()