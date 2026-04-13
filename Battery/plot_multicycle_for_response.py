#!/usr/bin/env python3
"""
Build figures and LaTeX/CSV tables from eval_multicycle_standalone.py outputs,
for rebuttal / supplementary statistical validation (multiple held-out cycles).

Inputs (produced by eval_multicycle_standalone.py --out_csv base):

  base_per_cycle.csv    — one row per discharge cycle
  base_per_battery.csv  — one row per cell (optional)

Example::

    cd Battery
    python eval_multicycle_standalone.py --model ani4 \\
        --checkpoint 4th/best_ani4_model.pt \\
        --data dataset/processed_battery_data_rollout.pt \\
        --out_csv figures/battery_eval

    python plot_multicycle_for_response.py \\
        --per_cycle figures/battery_eval_per_cycle.csv \\
        --per_battery figures/battery_eval_per_battery.csv \\
        --out_dir figures/battery_response_stats \\
        --model_label "ANI-4"

Outputs in ``out_dir``:

  - ``multicycle_boxplot.pdf`` — box/violin of MSE, MAE, RMSE across cycles
  - ``multicycle_rmse_vs_cycle_index.pdf`` — RMSE vs normalized cycle index
  - ``summary_table.tex`` — booktabs table for main text / response
  - ``summary_wide.csv`` — single-row summary for slides
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def read_csv_dicts(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(d: Dict[str, Any], key: str) -> float:
    return float(d[key])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cycle", required=True, help="*_per_cycle.csv from eval script")
    ap.add_argument("--per_battery", default="", help="*_per_battery.csv (optional)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_label", default="Model", help="Short name for figure/table captions")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_csv_dicts(args.per_cycle)
    if not rows:
        raise SystemExit(f"No rows in {args.per_cycle}")

    norm_c = np.array([to_float(r, "norm_cycle_feature") for r in rows])
    mse = np.array([to_float(r, "mse") for r in rows])
    mae = np.array([to_float(r, "mae") for r in rows])
    rmse = np.array([to_float(r, "rmse") for r in rows])
    n_cycles = len(rows)
    cell = rows[0].get("battery_key", "—")

    # ----- summary stats (match eval script: sample std, ddof=1) -----
    def mean_std(x: np.ndarray) -> tuple[float, float]:
        m = float(np.mean(x))
        s = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        return m, s

    m_mse, s_mse = mean_std(mse)
    m_mae, s_mae = mean_std(mae)
    m_rmse, s_rmse = mean_std(rmse)

    # ----- wide CSV -----
    wide_path = out / "summary_wide.csv"
    with open(wide_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "battery_key",
                "model_label",
                "n_held_out_cycles",
                "mse_mean",
                "mse_std",
                "mae_mean",
                "mae_std",
                "rmse_mean",
                "rmse_std",
            ]
        )
        w.writerow(
            [
                cell,
                args.model_label,
                n_cycles,
                m_mse,
                s_mse,
                m_mae,
                s_mae,
                m_rmse,
                s_rmse,
            ]
        )
    print(f"Wrote {wide_path}")

    # ----- LaTeX table -----
    tex_path = out / "summary_table.tex"
    lines = [
        r"% Paste into response / paper; requires \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{args.model_label}: rollout error on NASA cell {cell}, aggregating all held-out test discharge cycles ($n={n_cycles}$).}}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & mean $\pm$ std & unit \\",
        r"\midrule",
        rf"MSE  & ${m_mse:.4e} \pm {s_mse:.4e}$ & V$^2$ \\",
        rf"MAE  & ${m_mae:.4e} \pm {s_mae:.4e}$ & V \\",
        rf"RMSE & ${m_rmse:.4e} \pm {s_rmse:.4e}$ & V \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:battery_multicycle_stats}",
        r"\end{table}",
        "",
    ]
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {tex_path}")

    # ----- per-cycle LaTeX (optional appendix) -----
    tex_cycle = out / "per_cycle_table_appendix.tex"
    with open(tex_cycle, "w", encoding="utf-8") as tf:
        tf.write(
            r"% Full per-cycle breakdown (supplementary). Requires booktabs."
            "\n\\begin{table}[t]\n\\centering\n\\small\n"
        )
        tf.write(
            rf"\caption{{Per held-out cycle metrics ({args.model_label}, cell {cell}).}}"
            "\n\\begin{tabular}{rrrrr}\n\\toprule\n"
        )
        tf.write(
            r"Norm. cycle index & $n_{\mathrm{steps}}$ & MSE & MAE & RMSE \\"
            "\n\\midrule\n"
        )
        order = np.argsort(norm_c)
        for i in order:
            tf.write(
                f"{norm_c[i]:.4f} & {int(to_float(rows[i], 'n_steps'))} & "
                f"{mse[i]:.4e} & {mae[i]:.4e} & {rmse[i]:.4e} \\\\\n"
            )
        tf.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Wrote {tex_cycle}")

    # ----- Figure: box plot -----
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "figure.figsize": (6.5, 3.2),
        }
    )
    fig, ax = plt.subplots(1, 1)
    data = [mse, mae, rmse]
    labels = ["MSE\n(V$^2$)", "MAE\n(V)", "RMSE\n(V)"]
    try:
        ax.boxplot(
            data,
            tick_labels=labels,
            showmeans=True,
            meanline=True,
            widths=0.55,
        )
    except TypeError:
        ax.boxplot(
            data,
            labels=labels,
            showmeans=True,
            meanline=True,
            widths=0.55,
        )
    ax.set_ylabel("Error")
    ax.set_title(
        rf"{args.model_label} — {cell}: distribution over $n={n_cycles}$ held-out cycles"
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    p_box = out / "multicycle_boxplot.pdf"
    fig.savefig(p_box, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p_box}")

    # ----- Figure: RMSE vs norm_cycle -----
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.2))
    order = np.argsort(norm_c)
    ax.plot(
        norm_c[order],
        rmse[order],
        "o-",
        color="#0072B2",
        markersize=5,
        linewidth=1.2,
        label="Per-cycle RMSE",
    )
    ax.axhline(m_rmse, color="#D55E00", linestyle="--", linewidth=1.5, label="Mean RMSE")
    ax.fill_between(
        [norm_c.min(), norm_c.max()],
        m_rmse - s_rmse,
        m_rmse + s_rmse,
        color="#D55E00",
        alpha=0.15,
        label=r"Mean $\pm$ 1 std",
    )
    ax.set_xlabel("Normalized discharge cycle index (aging proxy)")
    ax.set_ylabel("RMSE (V)")
    ax.set_title(rf"{args.model_label} — {cell}: RMSE vs. cycle index ($n={n_cycles}$)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    p_rmse = out / "multicycle_rmse_vs_cycle_index.pdf"
    fig.savefig(p_rmse, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p_rmse}")

    # ----- optional: read per_battery for multi-cell paragraph -----
    if args.per_battery and os.path.isfile(args.per_battery):
        bat = read_csv_dicts(args.per_battery)
        txt_path = out / "multi_cell_blurb.txt"
        with open(txt_path, "w", encoding="utf-8") as bf:
            bf.write("Per-cell summary rows (for copy-paste):\n\n")
            for r in bat:
                bf.write(
                    f"  {r.get('battery_key')}: n_cycles={r.get('n_cycles_evaluated')}, "
                    f"RMSE mean±std = {r.get('rmse_mean')} ± {r.get('rmse_std')}\n"
                )
        print(f"Wrote {txt_path}")

    print("\nDone. Suggested response sentence (EN):")
    print(
        '  "On NASA cell '
        + str(cell)
        + ", we report autoregressive rollout errors aggregated over all "
        + str(n_cycles)
        + " held-out test discharge cycles: RMSE = "
        + f"{m_rmse:.4e} ± {s_rmse:.4e} V (mean ± std across cycles), "
        + 'with analogous statistics for MAE/MSE (see supplementary table/figure)."'
    )


if __name__ == "__main__":
    main()
