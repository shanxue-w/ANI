"""
Compare relative rollout errors for hidden-dim variants (20 vs 32).
Original folders 2th/, 4th/, NeuralRK4/ are unchanged; this script reads
the parallel dirs 2th_h20, 2th_h32, 4th_h20, 4th_h32, NeuralRK4_h20, NeuralRK4_h32.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})


def load_rel(path):
    p = os.path.join(HERE, path)
    return np.loadtxt(p)


def main():
    # Baseline RK4 MLP
    r20 = load_rel("NeuralRK4_h20/rel_test_errors_small_test.txt")
    r32 = load_rel("NeuralRK4_h32/rel_test_errors_small_test.txt")
    # ANI-2
    a2_20 = load_rel("2th_h20/rel_test_errors_small.txt")
    a2_32 = load_rel("2th_h32/rel_test_errors_small.txt")
    # ANI-4
    a4_20 = load_rel("4th_h20/rel_test_errors_small.txt")
    a4_32 = load_rel("4th_h32/rel_test_errors_small.txt")

    n = 200
    plt.figure(figsize=(9, 6))
    plt.plot(r20[1:n], label="Baseline h=20", color="#332288", linewidth=2)
    plt.plot(r32[1:n], label="Baseline h=32", color="#88CCEE", linewidth=2)
    plt.plot(a2_20[1:n], label="ANI-2 h=20", color="#117733", linewidth=2)
    plt.plot(a2_32[1:n], label="ANI-2 h=32", color="#44AA99", linewidth=2)
    plt.plot(a4_20[1:n], label="ANI-4 h=20", color="#CC6677", linewidth=2)
    plt.plot(a4_32[1:n], label="ANI-4 h=32", color="#AA4499", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Relative error")
    plt.yscale("log")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(ncol=2)
    plt.tight_layout()
    out = os.path.join(HERE, "rel_error_hidden_ablation.pdf")
    plt.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved", out)


if __name__ == "__main__":
    main()
