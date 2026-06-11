import numpy as np
import matplotlib.pyplot as plt


class A():
    def __init__(self):
        super(A, self).__init__()

    def F(self, u):
        u0 = u[..., 0:1]
        u1 = u[..., 1:2]

        new_u0 = u1 
        new_u1 = -9.80665 * np.sin(u0)

        return np.concatenate([new_u0, new_u1], axis=-1)

    def single_step(self, u, parameters=None, t=None, dt=None):
        k1 = dt * self.F(u)
        k2 = dt * self.F(u + 1/3 * k1)
        k3 = dt * self.F(u + 2/3 * k2)
        k4 = dt * self.F(u + 1/12 * k1 + 1/3 * k2 - 1/12 * k3)
        k5 = dt * self.F(u - 1/16 * k1 + 9/8 * k2 - 3/16 * k3 - 3/8 * k4)
        k6 = dt * self.F(u + 9/8 * k2 - 3/8 * k3 - 3/4 * k4 + 1/2 * k5)
        k7 = dt * self.F(u + 9/44 * k1 - 9/11 * k2 + 63/44 * k3 + 18/11 * k4 - 16/11 * k6)
        return u + 11/120 * k1 + 27/40 * k3 + 27/40 * k4 - 4/15 * k5 - 4/15 * k6 + 11/120 * k7


plt.rcParams.update({
    "font.size": 14,        # 全局字体大小
    "axes.labelsize": 14,   # 坐标轴标签字体大小
    "xtick.labelsize": 14,  # x 轴刻度字体大小
    "ytick.labelsize": 14,  # y 轴刻度字体大小
    "legend.fontsize": 14,  # 图例字体大小
})

data_2th_rel = np.loadtxt("2th_test/rel_test_errors_small.txt")
data_3th_rel = np.loadtxt("3th/rel_test_errors_small.txt")
data_4th_rel = np.loadtxt("4th_test/rel_test_errors_small.txt")
data_6th_rel = np.loadtxt("6th_test/rel_test_errors_small.txt")
data_neural_ode_rel = np.loadtxt("neural_RK4/rel_test_errors_baseline.txt")
data_neural_RK_rel = np.loadtxt("neural_RK4/rel_test_errors_baseline.txt")

# 只有errors，shape 都是 (1001, )
plt.figure(figsize=(8, 6))
plt.plot(data_neural_ode_rel[1:], label="Neural-RK6", color="#0072B2", linewidth=2)
plt.plot(data_2th_rel[1:], label="ANI-2", color="#009E73", linewidth=2)#E69F00
# plt.plot(data_3th_rel[1:], label="3th Split", linewidth=2)
plt.plot(data_4th_rel[1:], label="ANI-4", color="#E69F00", linewidth=2)
plt.plot(data_6th_rel[1:], label="ANI-6", color="#D55E00", linewidth=2)

plt.yscale("log")
# plt.title("Relative Test Errors")
plt.xlabel("Time Step", fontsize=14)
plt.ylabel("Relative Error", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig("rel_test_errors_test_baseline.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()


data_2th_abs = np.loadtxt("2th_test/abs_test_errors_small.txt")
data_3th_abs = np.loadtxt("3th/abs_test_errors_small.txt")
data_4th_abs = np.loadtxt("4th_test/abs_test_errors_small.txt")
data_6th_abs = np.loadtxt("6th_test/abs_test_errors_small.txt")
data_neural_ode_abs = np.loadtxt("neural_RK4/abs_test_errors_baseline.txt")
data_neural_RK_abs = np.loadtxt("neural_RK4/abs_test_errors_baseline.txt")

plt.figure(figsize=(8, 6))
plt.plot(data_2th_abs[1:, 0:1], label="ANI-2", linewidth=2)
# plt.plot(data_3th_abs[1:, 0:1], label="3th Split", linewidth=2)
plt.plot(data_4th_abs[1:, 0:1], label="ANI-4", linewidth=2)
plt.plot(data_6th_abs[1:, 0:1], label="ANI-6", linewidth=2)
plt.plot(data_neural_ode_abs[1:, 0:1], label="Neural RK6", linewidth=2)

plt.yscale("log")
# plt.title(r"Absolute $\theta$ Errors")
plt.xlabel("Time Step")
plt.ylabel(r"$\theta$ Error")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig("abs_theta_errors_test_baseline.pdf", format='pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(data_2th_abs[1:, 1:2], label="ANI-2", linewidth=2)
# plt.plot(data_3th_abs[1:, 1:2], label="3th Split", linewidth=2)
plt.plot(data_4th_abs[1:, 1:2], label="ANI-4", linewidth=2)
plt.plot(data_6th_abs[1:, 1:2], label="ANI-6", linewidth=2)
plt.plot(data_neural_ode_abs[1:, 1:2], label="Neural RK6", linewidth=2)

plt.yscale("log")
# plt.title(r"Absolute $\theta$ Errors")
plt.xlabel("Time Step", fontsize=14)
plt.ylabel(r"$\omega$ Error", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig("abs_omega_errors_test_baseline.pdf", format='pdf', bbox_inches='tight')
plt.close()

u_omega = np.load("2th_test/traj_test.npy")
# u_omega_2th = np.load("2th_test/u_2th_test.npy")
# u_omega_4th = np.load("4th_test/u_4th_test.npy")
u_omega_6th = np.load("6th_test/u_6th_L.npy")
# u_omega_neural_ode = np.load("neural_RK4/u_baseline.npy")
# print(u_omega.shape, u_omega_2th.shape, u_omega_4th.shape, u_omega_6th.shape, u_omega_neural_ode.shape)
# [1001, 2] plot (\theta, \omega)

A_test = A()
u_omega_test = []
u_temp = u_omega[0, :]
for i in range(u_omega.shape[0]):
    u_omega_test.append(u_temp)
    u_temp = A_test.single_step(u_temp, dt=1e-1)
u_omega_test = np.array(u_omega_test)
plt.figure(figsize=(8, 6))

# True and ANI-6 用实线
plt.plot(u_omega[:, 0], u_omega[:, 1], label="True", color='tab:blue', linewidth=2.5)
plt.plot(u_omega_6th[:, 0], u_omega_6th[:, 1], linestyle='--', label="ANI-6",
         color="tab:orange", linewidth=2.5)

plt.plot(u_omega_test[:, 0], u_omega_test[:, 1], 'o', label="Prior",
         color="tab:green", markersize=2.5, linewidth=2.5)



plt.xlabel(r"$\theta$", fontsize=14)
plt.ylabel(r"$\omega$", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig("traj_test_baseline.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# cm = 1 / 2.54

# # plt.figure(figsize=(8, 8))
# plt.figure(figsize=(2 * cm, 2 * cm))
# # no axix, only u_true[：, 0]
# # plt.plot(u_omega[:, 0], u_omega[:, 1], color="gray")
# plt.plot(u_omega[:, 0], u_omega[:, 1], color='#1f77b4', linewidth=0.2)
# plt.axis("off")
# plt.gca().set_frame_on(False)
# plt.tight_layout()
# plt.savefig("data_traj.pdf", format='pdf', bbox_inches='tight', pad_inches=0, dpi=300)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter, LogLocator


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
    "Neural-RK6": "#0072B2",
    "ANI-2": "#009E73",
    "ANI-4": "#E69F00",
    "ANI-6": "#D55E00",
    "True": "#0072B2",
    "Prior": "#009E73",
}

LINESTYLES = {
    "Neural-RK6": "-",
    "ANI-2": "--",
    "ANI-4": "-.",
    "ANI-6": ":",
    "True": "-",
    "Prior": "None",
}

MARKERS = {
    "Neural-RK6": "o",
    "ANI-2": "s",
    "ANI-4": "^",
    "ANI-6": "D",
}


# ============================================================
# Helpers
# ============================================================
def add_panel_label(ax, label, x=-0.08, y=1.08):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left"
    )


# ============================================================
# Panel a: relative error
# ============================================================
def plot_error_panel(ax):
    data_2th_rel = np.loadtxt("2th_test/rel_test_errors_small.txt")
    data_4th_rel = np.loadtxt("4th_test/rel_test_errors_small.txt")
    data_6th_rel = np.loadtxt("6th_test/rel_test_errors_small.txt")
    data_neural_rk_rel = np.loadtxt("neural_RK4/rel_test_errors_baseline.txt")

    steps = np.arange(1, len(data_neural_rk_rel))

    series = [
        ("Neural-RK6", data_neural_rk_rel[1:]),
        ("ANI-2", data_2th_rel[1:]),
        ("ANI-4", data_4th_rel[1:]),
        ("ANI-6", data_6th_rel[1:]),
    ]

    for label, y in series:
        ax.plot(
            steps,
            y,
            label=label,
            color=COLORS[label],
            linestyle=LINESTYLES[label],
            linewidth=1.0,
            marker=MARKERS[label],
            markersize=1.8,
            markevery=max(1, len(steps) // 12),
        )

    ax.set_yscale("log")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Relative error")

    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False, handlelength=2.5)

    add_panel_label(ax, "a")


# ============================================================
# Panel b: phase portrait
# ============================================================
def plot_phase_panel(ax):
    # True trajectory
    u_omega = np.load("2th_test/traj_test.npy")
    # ANI-6 trajectory
    u_omega_6th = np.load("6th_test/u_6th_L.npy")

    # Prior rollout
    A_test = A()   # make sure class A is already defined/imported
    u_omega_prior = []
    u_temp = u_omega[0, :].copy()
    for _ in range(u_omega.shape[0]):
        u_omega_prior.append(u_temp.copy())
        u_temp = A_test.single_step(u_temp, dt=1e-1)
    u_omega_prior = np.array(u_omega_prior)

    # Prior
    ax.plot(
        u_omega_prior[:, 0], u_omega_prior[:, 1],
        label="Prior",
        color=COLORS["Prior"],
        linestyle="None",
        marker="o",
        markersize=1.4,
        markevery=max(1, len(u_omega_prior) // 140),
    )

    # True
    ax.plot(
        u_omega[:, 0], u_omega[:, 1],
        label="True",
        color=COLORS["True"],
        linestyle="-",
        linewidth=1.1,
    )

    # ANI-6
    ax.plot(
        u_omega_6th[:, 0], u_omega_6th[:, 1],
        label="ANI-6",
        color="#D55E00",
        linestyle="--",
        linewidth=1.1,
    )


    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\omega$")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False, loc="best", handlelength=2.5)

    add_panel_label(ax, "b")


# ============================================================
# Main
# ============================================================
def main():
    # Nature double-column width
    width_mm = 180
    height_mm = 72

    fig = plt.figure(figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)))

    gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        width_ratios=[1.0, 1.05],
        wspace=0.15,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    plot_error_panel(ax_a)
    plot_phase_panel(ax_b)

    # Fill the canvas manually; do not use bbox_inches="tight"
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.155,
        top=0.930,
        wspace=0.22,
    )

    fig.savefig(
        "pendulum_combined_nature.pdf",
        format="pdf",
        dpi=600,
    )

    fig.savefig(
        "pendulum_combined_nature.png",
        format="png",
        dpi=600,
    )

    plt.close(fig)

    print("Saved: pendulum_combined_nature.pdf")
    print("Saved: pendulum_combined_nature.png")

if __name__ == "__main__":
    main()