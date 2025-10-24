## Basic

This repository contains the implementation of the Alternating Neural Integrators (ANI) model and numerical experiments of using ANI for solving various differential equations.

Most of the ODE experiments share a unified template, so it is important to first install the package in editable mode to ensure that all experiments run smoothly:
```bash
pip install -e .
```
After installation, you can run the experiment scripts directly from the repository without additional configuration.

## Contents

Each folder corresponds to a specific differential equation or PDE system. The structure inside each folder is organized as follows:

* `Problem/`
  Contains scripts for visualization and plotting. These scripts typically use trained models or saved data to produce figures and analysis.

* `Problem/method/`
  Contains the training and testing scripts (`.py` files) for the ANI model with a specific numerical method or resolution. Each file corresponds to either a training run or a test run.
  - **Note:** For `NS_NEW`, the training script is named `parallel.py` due to its parallel implementation.

* `Problem/dataset/`
  Contains scripts for dataset generation as well as the generated data files. These scripts are used to create the training and testing data used by the ANI models.

If you want, I can also make a **tree-style diagram** showing `Problem/`, `method/`, `dataset/`, and plotting scripts, so it’s visually easier to understand. Do you want me to add that?


- `Pendulum`: Implementation and experiments of ANI on the damped pendulum ODE.
- `Lorenz-stenflo`: Implementation and experiments of ANI on the Lorenz-Stenflo ODE.
- `Glycolytic`: Implementation and experiments of ANI on the glycolytic oscillator ODE.
- `CompoundPendulum`: Implementation and experiments of ANI on the damped compound pendulum ODE.
- `Morris`: Implementation and experiments of ANI on the Morris-Lecar neuron model ODE.
- `Kan`: Implementation and experiments of ANI on the Multi-tropical predator-prey ODE system.
- `Euler`: Implementation and experiments of ANI on the 1D compressible Euler equations.
- `Allen-Cahn-new`: Implementation and experiments of ANI on the Fitzhugh-Nagumo equation.
- `NS_NEW`: Implementation and experiments of ANI on the 2D incompressible Navier-Stokes equations.
- `Kolmogorov`: Implementation and experiments of ANI on the Kolmogorov flow problem.


**The data used to generate the dataset can be found in the `dataset` folder of each experiment directory.**