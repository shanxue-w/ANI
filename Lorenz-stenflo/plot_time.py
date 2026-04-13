import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("chaos_metrics_data.csv")

required = {"Metric", "K", "Method", "Value"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

agg = (
    df.groupby(["Metric", "K", "Method"], as_index=False)["Value"]
      .mean()
)


metrics = list(agg["Metric"].unique())

methods_all = ["Reference", "Baseline", "ANI-2", "ANI-4"]
methods_all = [m for m in methods_all if m in agg["Method"].unique()]

custom_palette = {
    "Reference": "#333333",
    "Baseline":  "#1f77b4",
    "ANI-2":     "#ff7f0e",
    "ANI-4":     "#2ca02c",
}
line_styles = {
    "Reference": "-",
    "Baseline":  (0, (2, 2)),  # dashed
    "ANI-2":     "-",
    "ANI-4":     "-",
}

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.linewidth"] = 1.2

def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

n = len(metrics)
fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharex=True)
if n == 1:
    axes = [axes]

for i, metric in enumerate(metrics):
    ax = axes[i]
    sub = agg[agg["Metric"] == metric]

    for m in methods_all:
        s = sub[sub["Method"] == m].sort_values("K")
        if s.empty:
            continue
        ax.plot(
            s["K"].values,
            s["Value"].values,
            label=m,
            color=custom_palette.get(m, None),
            linestyle=line_styles.get(m, "-"),
            linewidth=2.5,
            alpha=0.95,
        )

    ax.set_title(metric, fontweight="bold", pad=12)
    ax.set_xlabel("K (Simulation Steps)")
    if i == 0:
        ax.set_ylabel("Metric Value")
    ax.grid(True, linestyle="--", alpha=0.3)
    despine(ax)
    ax.legend(frameon=False, loc="best")

plt.tight_layout()
plt.savefig("chaos_metrics_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig("chaos_metrics_comparison.pdf", bbox_inches="tight")
plt.show()

wide = agg.pivot(index=["Metric", "K"], columns="Method", values="Value").reset_index()

if "Reference" not in wide.columns:
    raise ValueError("No 'Reference' method found in data; cannot compute differences.")

diff_methods = [m for m in ["Baseline", "ANI-2", "ANI-4"] if m in wide.columns]

# Keep points where Reference and compared methods exist
keep_cols = ["Metric", "K", "Reference"] + diff_methods
wide2 = wide[keep_cols].dropna(subset=["Reference"])

# Absolute difference
diff_long_list = []
for m in diff_methods:
    tmp = wide2[["Metric", "K", "Reference", m]].dropna(subset=[m]).copy()
    tmp["Method"] = m
    tmp["Abs_Diff"] = (tmp[m] - tmp["Reference"]).abs()
    diff_long_list.append(tmp[["Metric", "K", "Method", "Abs_Diff"]])

diff_long = pd.concat(diff_long_list, ignore_index=True)

fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharex=True)
if n == 1:
    axes = [axes]

for i, metric in enumerate(metrics):
    ax = axes[i]
    sub = diff_long[diff_long["Metric"] == metric]

    for m in diff_methods:
        s = sub[sub["Method"] == m].sort_values("K")
        if s.empty:
            continue
        ax.plot(
            s["K"].values,
            s["Abs_Diff"].values,
            label=m,
            color=custom_palette.get(m, None),
            linestyle=line_styles.get(m, "-"),
            linewidth=2.5,
            alpha=0.95,
        )

    ax.set_title(metric, fontweight="bold", pad=12)
    ax.set_xlabel("K (Simulation Steps)")
    if i == 0:
        ax.set_ylabel(r"Abs. Difference $|\mathrm{Method}-\mathrm{Ref}|$")
    ax.grid(True, linestyle="--", alpha=0.3)
    despine(ax)
    ax.legend(frameon=False, loc="best")

plt.tight_layout()
plt.savefig("chaos_metrics_diff_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig("chaos_metrics_diff_comparison.pdf", bbox_inches="tight")
plt.show()