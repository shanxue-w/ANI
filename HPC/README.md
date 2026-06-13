# HPC Reproduction Notes

This folder is a small two-language reproduction example for coupling an ANI
correction network with a compiled C++ prior. The reference dynamics are the
Van der Pol system, while the prior is a harmonic-oscillator step map written in
C++/Eigen. The Python scripts train and export TorchScript correctors, and
`combined.cc` runs C++ rollouts against a small-step RK4 reference.

## Files

- `dataset/data.py` generates flow-map pairs `(X, Y, dt)` for the Van der Pol
  reference system. The default dataset uses `dt_flow = 1e-1` and
  `dt_small = 1e-3`.
- `prior_oscillator_eigen.cpp` is the C++/Eigen prior used by the Lie-corrector
  training path.
- `lie_splitting_solver.py` calls the compiled prior, trains a residual
  corrector on prior outputs, and exports `lie_corrector_phi.ts`.
- `ani_pytorch_solver.py` trains the ANI Strang-style model with a differentiable
  PyTorch proxy prior. It exports both `ani_strang_full.ts` and
  `ani_corrector_only.ts`.
- `combined.cc` is the C++ rollout/evaluation driver using LibTorch.
- `plot.py` converts rollout CSV files into PDF trajectory plots.

## Dependencies

Install the repository Python environment first from the repository root:

```bash
pip install -e .
pip install -r requirements.txt
```

For the C++ pieces, make sure a C++17 compiler, Eigen headers, and LibTorch are
available. On many Linux systems Eigen is under `/usr/include/eigen3`; set
`LIBTORCH` to the directory containing `include/` and `lib/`, for example:

```bash
export LIBTORCH=/usr/local/libtorch
```

The checked-in `CMakeLists.txt` is only a local helper. The direct build commands
below match the current source filenames.

## Reuse Existing Artifacts

If the generated artifacts are already present, reproduce the rollout CSVs and
PDFs directly:

```bash
cd HPC

./combined --mode lie --ts lie_corrector_phi.ts --out loss_lie.csv --steps 200
./combined --mode ani --ts ani_corrector_only.ts --out loss_ani.csv --steps 200

python plot.py --csv loss_lie.csv --out ref_lie.pdf --model Lie
python plot.py --csv loss_ani.csv --out ref_ani2.pdf --model ANI2
```

For `--mode ani`, use `ani_corrector_only.ts`: `combined.cc` applies the two
half prior steps in C++ and calls only the TorchScript corrector. The full
TorchScript file `ani_strang_full.ts` is useful when running the complete ANI
Strang map directly from Python/TorchScript, not for this C++ rollout command.

## Regenerate From Source

From the repository root, generate the dataset:

```bash
cd HPC/dataset
python data.py
cd ..
```

This writes:

```text
dataset/toy_hpc_ode_dataset_dt_flow_1e-1_dt_small_1e-3.npz
```

Build the C++/Eigen prior:

```bash
g++ -O3 -std=c++17 -I /usr/include/eigen3 \
  prior_oscillator_eigen.cpp -o prior_oscillator_eigen
```

Train and export the Lie corrector:

```bash
python lie_splitting_solver.py \
  --dataset dataset/toy_hpc_ode_dataset_dt_flow_1e-1_dt_small_1e-3.npz \
  --prior_exec ./prior_oscillator_eigen \
  --out_ts lie_corrector_phi.ts
```

Train and export the ANI Strang model:

```bash
python ani_pytorch_solver.py \
  --dataset dataset/toy_hpc_ode_dataset_dt_flow_1e-1_dt_small_1e-3.npz \
  --out_ts_full ani_strang_full.ts \
  --out_ts_corrector ani_corrector_only.ts
```

Rebuild the C++ rollout driver:

```bash
g++ -O3 -std=c++17 combined.cc -o combined \
  -I /usr/include/eigen3 \
  -I ${LIBTORCH}/include \
  -I ${LIBTORCH}/include/torch/csrc/api/include \
  -L ${LIBTORCH}/lib -ltorch_cpu -ltorch -lc10 \
  -Wl,-rpath,${LIBTORCH}/lib
```

Then rerun the rollout and plotting commands from the previous section.

For a quick smoke test, pass a smaller value such as `--epochs 5` to the Python
training scripts. For reproducing the saved artifacts, use the default epoch
settings unless you intentionally change the training budget.
