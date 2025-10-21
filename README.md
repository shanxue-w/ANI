## Basic

This repository contains the implementation of the Alternating Neural Integrators (ANI) model and numerical experiments of using ANI for solving various differential equations.

Most of the ODE experiments share a unified template, so it is important to first install the package in editable mode to ensure that all experiments run smoothly:
```bash
pip install -e .
```
After installation, you can run the experiment scripts directly from the repository without additional configuration.

## Contents
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