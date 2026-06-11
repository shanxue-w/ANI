import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def mm_to_inch(mm):
    return mm / 25.4


# ============================================================
# Nature-style global settings
# ============================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

    # Final-size fonts for a 180-mm-wide double-column figure
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

    # Editable text in PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


COLORS = {
    "Pretrained": "#0072B2",
    "ANI-2": "#009E73",
    "ANI-4": "#E69F00",
}

LINESTYLES = {
    "Pretrained": "-",
    "ANI-2": "--",
    "ANI-4": "-.",
}

MARKERS = {
    "Pretrained": "o",
    "ANI-2": "s",
    "ANI-4": "^",
}

USE_LOG_Y_PANEL_A = False


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
        ha="left",
    )


def centers_to_edges(x):
    x = np.asarray(x)
    if len(x) == 1:
        dx = 0.5
        return np.array([x[0] - dx, x[0] + dx])

    mids = 0.5 * (x[:-1] + x[1:])
    first = x[0] - 0.5 * (x[1] - x[0])
    last = x[-1] + 0.5 * (x[-1] - x[-2])
    return np.concatenate([[first], mids, [last]])


def load_grid_data(fname):
    """
    Expected file format:
        a,b,error
    """
    data = np.loadtxt(fname, delimiter=",")
    a = data[:, 0]
    b = data[:, 1]
    err = data[:, 2]

    a_unique = np.unique(a)
    b_unique = np.unique(b)

    Z = np.full((len(b_unique), len(a_unique)), np.nan)

    a_id = {v: i for i, v in enumerate(a_unique)}
    b_id = {v: i for i, v in enumerate(b_unique)}

    for ai, bi, ei in zip(a, b, err):
        Z[b_id[bi], a_id[ai]] = ei

    return a_unique, b_unique, Z


# ============================================================
# Panel a
# ============================================================
def plot_panel_a(ax):
    data = {
        "Pretrained": pd.read_csv("2th_new/traj_error_A.txt", sep="\t").iloc[0:101],
        "ANI-2": pd.read_csv("2th_new/traj_error.txt", sep="\t").iloc[0:101],
        "ANI-4": pd.read_csv("4th_new/traj_error.txt", sep="\t").iloc[0:101],
    }

    for label, df in data.items():
        x = df["t_index"].to_numpy()
        mean = df["avg_relative_error"].to_numpy()
        sd = df["std_relative_error"].to_numpy()

        if USE_LOG_Y_PANEL_A:
            lower = np.maximum(mean - sd, np.finfo(float).tiny)
        else:
            lower = np.maximum(mean - sd, 0.0)

        upper = mean + sd

        ax.plot(
            x, mean,
            label=label,
            color=COLORS[label],
            linestyle=LINESTYLES[label],
            marker=MARKERS[label],
            linewidth=1.1,
            markersize=2.2,
            markevery=20,
        )

        ax.fill_between(
            x, lower, upper,
            color=COLORS[label],
            alpha=0.16,
            linewidth=0,
        )

    if USE_LOG_Y_PANEL_A:
        ax.set_yscale("log")

    ax.set_xlabel("Time Step")
    ax.set_ylabel(r"Mean relative $L^2$ error")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.grid(True, linewidth=0.35, alpha=0.3)
    ax.legend(frameon=False, loc="upper left", handlelength=2.5)

    add_panel_label(ax, "a")


# ============================================================
# Panel b: no error bars
# ============================================================
def plot_panel_b(ax):
    cases = [4, 4.5, 5, 5.5, 6]

    file_pattern = {
        "Pretrained": "2th_new/traj_error_A_{}.txt",
        "ANI-2": "2th_new/traj_error_{}.txt",
        "ANI-4": "4th_new/traj_error_{}.txt",
    }

    for method in ["Pretrained", "ANI-2", "ANI-4"]:
        means = []

        for c in cases:
            df = pd.read_csv(file_pattern[method].format(c), sep="\t")
            row = df.loc[df["t_index"] == 99]

            if len(row) == 0:
                raise ValueError(
                    f"t_index=99 not found in {file_pattern[method].format(c)}"
                )

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

    add_panel_label(ax, "b")


# ============================================================
# One heatmap
# ============================================================
def plot_one_heatmap(
    ax,
    cax,
    fname,
    title,
    show_ylabel=False,
    show_yticklabels=False,
    show_cbar_label=False,
):
    a_unique, b_unique, Z = load_grid_data(fname)

    # Keep original evaluated grid; no smoothing/interpolation.
    Z = np.clip(Z, 0.0, 1.0)

    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)

    a_edges = centers_to_edges(a_unique)
    b_edges = centers_to_edges(b_unique)

    im = ax.pcolormesh(
        a_edges,
        b_edges,
        Z,
        cmap="viridis",
        shading="flat",
        vmin=vmin,
        vmax=vmax,
        rasterized=False,
    )

    ax.set_title(title, pad=2)
    ax.set_xlabel(r"$a$")

    if show_ylabel:
        ax.set_ylabel(r"$b$")
    else:
        ax.set_ylabel("")

    if not show_yticklabels:
        ax.tick_params(axis="y", labelleft=False, left=False)

    ax.set_xticks(np.linspace(0.60, 0.80, 5))
    ax.set_yticks(np.linspace(0.70, 0.90, 5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.set_xlim(a_edges[0], a_edges[-1])
    ax.set_ylim(b_edges[0], b_edges[-1])
    ax.set_aspect("equal", adjustable="box")

    cb = plt.colorbar(im, cax=cax, format=FormatStrFormatter("%.2f"))
    cb.locator = MaxNLocator(nbins=4)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=6, width=0.6, length=2.2, pad=1)

    if show_cbar_label:
        cb.set_label("Relative error", fontsize=6.5)
    else:
        cb.set_label("")

    return im


# ============================================================
# Panel c
# ============================================================
def plot_panel_c(fig, bottom_spec):
    titles_and_files = [
        ("Pretrained", "2th_new/pred_error_2th_A.txt"),
        ("ANI-2", "2th_new/pred_error_2th.txt"),
        ("ANI-4", "4th_new/pred_error_4th.txt"),
    ]

    axes = []

    for i, (title, fname) in enumerate(titles_and_files):
        inner = bottom_spec[0, i].subgridspec(
            1,
            2,
            width_ratios=[1.0, 0.045],
            wspace=0.08,
        )

        ax = fig.add_subplot(inner[0, 0])
        cax = fig.add_subplot(inner[0, 1])

        plot_one_heatmap(
            ax=ax,
            cax=cax,
            fname=fname,
            title=title,
            show_ylabel=(i == 0),
            show_yticklabels=(i == 0),
            show_cbar_label=(i == 2),
        )

        axes.append(ax)

    add_panel_label(axes[0], "c", x=-0.42, y=1.08)


# ============================================================
# Main
# ============================================================
def main():
    width_mm = 180
    height_mm = 118   # compact; previous 142 was too tall

    fig = plt.figure(figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)))

    outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.00, 0.82],
        hspace=0.38,   # compact; previous 0.68 was too loose
    )

    # Top row
    gs_top = outer[0].subgridspec(
        1,
        2,
        width_ratios=[1.0, 1.0],
        wspace=0.30,
    )

    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])

    plot_panel_a(ax_a)
    plot_panel_b(ax_b)

    # Bottom row: three separated heatmap blocks
    gs_bottom = outer[1].subgridspec(
        1,
        3,
        wspace=0.34,   # compact but still separated
    )

    plot_panel_c(fig, gs_bottom)

    fig.savefig(
        "fhn_combined_nature_compact.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=600,
    )

    fig.savefig(
        "fhn_combined_nature_compact.png",
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close(fig)

    print("Saved: fhn_combined_nature_compact.pdf")
    print("Saved: fhn_combined_nature_compact.png")


if __name__ == "__main__":
    main()