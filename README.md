# Alternating Neural Integrators (ANI)

This repository contains the implementation of **Alternating Neural Integrators (ANI)** and the numerical experiments used to evaluate ANI on a range of ordinary differential equations (ODEs) and partial differential equations (PDEs).

Most ODE experiments in this repository follow a unified code template. To ensure that all experiments run correctly, we recommend first installing the dependencies and then installing the repository in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```
After installation, experiment scripts can be run directly from the repository without additional configuration.

## Quick Demo: Lorenz--Stenflo

We provide a **complete demo** in the `Lorenz-stenflo` folder. This demo includes:

- the **gray-box baseline**,
- the **ANI model**,
- and the **symbolic regression** pipeline.

In most cases, the full workflow can be launched with:

```bash
bash run.sh
```

The demo is designed to run automatically, including MATLAB scripts through terminal commands of the form:

```bash
matlab -batch "script_name"
```

If `matlab -batch` is not available in your environment, please run the required MATLAB scripts manually.

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

- **Allen-Cahn-new**  
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

- [Zenodo Link 1](https://zenodo.org/records/17412715?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImZiMGNjZDk4LWExNTMtNGM4YS05Yzc5LTJmYWI4MzVmODY5MCIsImRhdGEiOnt9LCJyYW5kb20iOiJjMTJiODRiYWFiYWZhZTVmYTU1NDZiMzVlNTE1ODgwMCJ9.9eJxZvjbyJNErJ1PbaAFLrT125mgY7paS_P_kXdibbehwLg9aTTYw-hStZpSh2P-K2-tPTuSSFty-Xh_qajE-Q)
- [Zenodo Link 2](https://zenodo.org/records/17412698?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjliYmQ2ZTQ0LWYyMjQtNGVjMS04NDc1LTM1YjM5YzQ0NmZkOCIsImRhdGEiOnt9LCJyYW5kb20iOiIxYWE4MWMyMzQzZDIyYjY3NDAxZjEzNThlMzNhZTc4ZiJ9.5___hNAhSjTrFlyrdkM6WKgziyw0UwyMkttc-R0HVFcAtreUruzKY14CxZpLf90BuhWSdjJepm5k1ZAxUdzfpg)
- [Zenodo Link 3](https://zenodo.org/records/19482734?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjgxNzk2NGZhLTc1NmQtNDM2Mi1hNTgwLTViMzgwMWZhOTMzOSIsImRhdGEiOnt9LCJyYW5kb20iOiIxYTkwYjlhMDM3NzY5MWQzM2UwZmNlN2Q2OTlhMjczZCJ9.cFybQkKN6f28_ut_TqiV_8iu5OC23R4F_zRwC6oFXi2K1vrQ3eVW0EMPI9dGVBLw7Fem33q_dATvrWJ0oTzyMg)



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