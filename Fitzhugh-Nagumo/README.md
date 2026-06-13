# FitzHugh-Nagumo

This folder contains the FitzHugh-Nagumo reaction-diffusion experiments used by
ANI. The implementation is organized around three pieces:

- `dataset/data.py` generates the reference trajectories with a spectral ETDRK4
  solver. The main one-step training data are saved at interval `h = 0.01`, and
  the `data_dt` variant also stores variable horizons `0.01`, `0.005`, and
  `0.0025`.
- `basemodel/base.py` trains the `FNO_ETDRK4` neural propagator. Its
  `forward(x, dts)` interface is step-size aware: the passed `dts` values are
  used to recompute and cache the ETDRK4 coefficients for each requested step.
- `2th_new/ANI_2th.py` and `4th_new/ANI_4th.py` load the pretrained
  `FNO_ETDRK4` checkpoint from `basemodel/best_model.pth`, freeze its
  parameters, and use it as the `N0` prior inside the ANI composition.

## Fractional Prior Steps

ANI-2 and ANI-4 call the frozen prior at fractional substep sizes such as
`h/2` and `h/4`. This is intentional for this experiment. The prior is not used
only as a fixed black-box map at one saved interval; it is an FNO-ETDRK4
step-map prior whose public interface accepts the target substep size through
`dts`. The ETDRK4 part of the architecture is evaluated with the supplied
substep, while the learned FNO component supplies the neural approximation to
the nonlinear update within that structured step.

This makes the fractional calls consistent with the way the prior is defined
and pretrained. In ANI, the frozen prior provides a physics-structured proposal
at each substep, and the trainable correction network is optimized around the
full composed update to refine the resulting one-step map. Therefore the
fractional FNO-ETDRK4 calls should be interpreted as callable prior steps inside
the ANI integrator rather than as a standalone exact solver requirement.
