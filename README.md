# Alternating Neural Integrators (ANI)

## System Requirements

The codebase has been tested in a standard Linux environment and is intended for research use on workstation- or server-level hardware. This repository has been tested on Ubuntu 22.04 LTS with Python 3.11 and MATLAB R2024a.

### Operating system
The repository is expected to run on:
- Ubuntu 22.04 LTS or later
- Other Linux environments with comparable Python and MATLAB support

Windows and macOS may also work for parts of the repository, but the full workflow has primarily been developed and tested in Linux-based environments.

### Hardware
- A CUDA-capable GPU is recommended for training ANI models
- CPU-only execution is possible for some scripts, but training and large-scale experiments may be significantly slower
- At least **16 GB RAM** is recommended for small to moderate experiments; more memory may be required for larger PDE datasets and training runs

### Software dependencies
The repository requires:
- Python 3.11
- Packages listed in `requirements.txt`
- MATLAB for parts of the symbolic regression / post-processing workflow

Install the Python environment with:
```bash
pip install -r requirements.txt
pip install -e .
```

Some experiments call MATLAB from the terminal via commands of the form:
```bash
matlab -batch "script_name"
```

If `matlab -batch` is not available in your environment, please run the required MATLAB scripts manually.


### Installation time
On a typical research workstation with a stable Python environment, installation usually takes approximately **30 minutes**.

## Quick Demo: Lorenz--Stenflo

We provide a **complete demo** in the `Lorenz-stenflo` folder. This demo includes:

- the **gray-box baseline**,
- the **ANI model**,
- and the **symbolic regression** pipeline.

A small demo dataset is provided in the `Lorenz-stenflo` example for testing the full pipeline. In most cases, the full workflow can be launched with:

```bash
bash run.sh
```

The demo is designed to run automatically, including MATLAB scripts through terminal commands of the form:

```bash
matlab -batch "script_name"
```

If `matlab -batch` is not available in your environment, please run the required MATLAB scripts manually.

### Demo runtime

For the provided **Lorenz--Stenflo** demo, the expected runtime is approximately **3 hours**, depending on hardware, MATLAB availability, and whether intermediate results are already cached.

### Expected output
Running `bash run.sh` in `Lorenz-stenflo` will generate:
- trained ANI model checkpoints,
- baseline results,
- symbolic regression outputs,
- figures and evaluation summaries saved to the corresponding output folders.

## Repository Structure

Each top-level folder corresponds to a specific differential equation or PDE system. The internal structure is generally organized as follows:

### `Problem/`
Contains scripts for visualization, post-processing, and figure generation. These scripts typically use trained models or saved outputs to reproduce plots and analyses.

### `Problem/method/`
Contains the main training and testing scripts for ANI under a given numerical method, discretization, or resolution. Each script typically corresponds to one experiment setting.

- **Note:** In `NS_NEW`, the training script is named `parallel.py` because of its parallel implementation.

### `Problem/dataset/`
Contains dataset-generation scripts as well as the generated data files used for training and testing.

**The data used to build each dataset can be found in the corresponding `dataset` folder of each experiment directory.**

## Contents

The repository currently includes the following experiments:

- **Pendulum**  
  ANI for the damped pendulum ODE.

- **Lorenz-stenflo**  
  ANI for the Lorenz--Stenflo ODE, including a demo for the gray-box baseline, ANI, and symbolic regression.

- **Glycolytic**  
  ANI for the glycolytic oscillator ODE.

- **CompoundPendulum**  
  ANI for the damped compound pendulum ODE.

- **Morris**  
  ANI for the Morris--Lecar neuron model.

- **Kan**  
  ANI for the multi-tropical predator--prey ODE system.

- **Euler**  
  ANI for the one-dimensional compressible Euler equations.

- **Fitzhugh-Nagumo**  
  Implementation and experiments for the corresponding reaction-diffusion benchmark.

- **NS_NEW**  
  ANI for the two-dimensional incompressible Navier--Stokes equations, including additional trade-off experiments.

- **Kolmogorov**  
  ANI for the Kolmogorov flow problem.

- **Battery**  
  ANI for the battery degradation experiment based on cell **B0005**.

- **BASIN**  
  ANI for the hydrology experiment based on the **CAMELS** dataset.

## Data Access

This GitHub repository contains the source code only. The datasets used in this work are hosted separately.

- [Zenodo Link 1](https://doi.org/10.5281/zenodo.17412715)
- [Zenodo Link 2](https://doi.org/10.5281/zenodo.17412698)
- [Zenodo Link 3](https://doi.org/10.5281/zenodo.19482734)

## Running on custom data

For experiments following the common template, users can:
1. place raw data or generated trajectories in the corresponding `Problem/dataset/` directory,
2. modify the dataset generation or loading script in that folder,
3. run the training/testing scripts in `Problem/method/`.

Because different experiments use different state variables and preprocessing pipelines, users should adapt the dataset scripts for their own problem setting.


## Running Experiments

After installation, experiments can typically be run from within the corresponding folder using the provided Python or shell scripts.

For the **Lorenz--Stenflo** demo, the recommended entry point is:

```bash
bash run.sh
```

This script is intended to reproduce the full pipeline, including:
1. the gray-box baseline,
2. ANI training/testing,
3. symbolic regression.

If your system does not support calling MATLAB from the command line via `matlab -batch`, please execute the relevant `.m` files manually.


## Notes

- Please install dependencies with `pip install -r requirements.txt` before running the code.
- Most ODE experiments share a common implementation template.
- The repository is organized so that training, testing, dataset generation, and plotting are separated clearly.
- For reproducibility, please make sure all required dependencies are installed before running experiments.