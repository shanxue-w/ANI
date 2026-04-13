"""
Generate ``processed_*_rollout.pt`` for each NASA cell (B0005–B0018) that is present.
Run from ``Battery/dataset`` with ``*.mat`` files in the current directory:

    python build_all_nasa_pt.py

Outputs default to ``processed_B0005_rollout.pt``, etc., same schema as
``processed_battery_data_rollout.pt``.
"""
import os

from data import export_processed_rollout_pt

DEFAULT_CELLS = [
    ("B0005.mat", "B0005", "processed_B0005_rollout.pt"),
    ("B0006.mat", "B0006", "processed_B0006_rollout.pt"),
    ("B0007.mat", "B0007", "processed_B0007_rollout.pt"),
    ("B0018.mat", "B0018", "processed_B0018_rollout.pt"),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    for mat_name, key, out_name in DEFAULT_CELLS:
        mat_path = os.path.join(here, mat_name)
        if not os.path.isfile(mat_path):
            print(f"[skip] {mat_name} not found")
            continue
        out_path = os.path.join(here, out_name)
        print(f"\n=== Building {key} -> {out_name} ===")
        export_processed_rollout_pt(
            mat_path=mat_path,
            save_path=out_path,
            battery_key=key,
            train_ratio=0.8,
            val_ratio=0.1,
            dt_min=1e-3,
            min_points=20,
            rollout_len=5,
            do_reload_check=False,
        )


if __name__ == "__main__":
    main()
