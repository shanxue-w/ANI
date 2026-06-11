import numpy as np
import pandas as pd
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
# Style definitions
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
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left"
    )


# ============================================================
# Panel a: error versus epsilon
# ============================================================
def plot_panel_a(ax):
    cases = [4, 4.5, 5, 5.5, 6]

    file_pattern = {
        "Baseline": "NEW_baseline/traj_error_{}.txt",
        "ANI-2": "2th_new/traj_error_{}.txt",
        "ANI-4": "4th_new/traj_error_{}.txt",
    }

    for method in ["Baseline", "ANI-2", "ANI-4"]:
        means = []
        for c in cases:
            df = pd.read_csv(file_pattern[method].format(c), sep="\t")
            row = df.loc[df["t_index"] == 99]
            if len(row) == 0:
                raise ValueError(f"t_index=99 not found in {file_pattern[method].format(c)}")
            means.append(float(row["avg_relative_error"].values[0]))

        ax.plot(
            cases,
            means,
            label=method,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            linewidth=1.1,
            markersize=3.0,
        )

    ax.set_xticks(cases)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"Mean relative $L^2$ error at $t=1\,\mathrm{s}$")

    ax.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.45)
    ax.legend(frameon=False, loc="upper right", handlelength=2.5)

    add_panel_label(ax, "a")


# ============================================================
# Panel b: temporal error accumulation
# ============================================================
def plot_panel_b(ax):
    data_ani2 = pd.read_csv("2th_new/traj_error.txt", sep="\t").iloc[0:101]
    data_ani4 = pd.read_csv("4th_new/traj_error.txt", sep="\t").iloc[0:101]
    data_base = pd.read_csv("NEW_baseline/traj_error_base.txt", sep="\t").iloc[0:101]

    series = [
        ("ANI-2", data_ani2),
        ("ANI-4", data_ani4),
        ("Baseline", data_base),
    ]

    for label, df in series:
        x = df["t_index"].to_numpy()
        y = df["avg_relative_error"].to_numpy()

        ax.plot(
            x,
            y,
            label=label,
            color=COLORS[label],
            linestyle=LINESTYLES[label],
            marker=MARKERS[label],
            linewidth=1.1,
            markersize=2.2,
            markevery=20,
        )

    ax.set_xlabel("Time Step")
    ax.set_ylabel(r"Mean relative $L^2$ error")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    ax.grid(True, linewidth=0.35, alpha=0.3)
    ax.legend(frameon=False, loc="upper left", handlelength=2.5)

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
        width_ratios=[1.0, 1.0],
        wspace=0.28,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    plot_panel_a(ax_a)
    plot_panel_b(ax_b)

    fig.savefig(
        "baseline_ani_comparison_nature.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=600,
    )

    fig.savefig(
        "baseline_ani_comparison_nature.png",
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close(fig)

    print("Saved: baseline_ani_comparison_nature.pdf")
    print("Saved: baseline_ani_comparison_nature.png")


if __name__ == "__main__":
    main()