import numpy as np
import torch


def rhs_full(state: np.ndarray, mu: float = 1.0) -> np.ndarray:
    """
    Full vector field:
        dx/dt = y
        dy/dt = mu * (1 - x^2) * y - x
    """
    x, y = state[..., 0], state[..., 1]
    dx = y
    dy = mu * (1.0 - x ** 2) * y - x
    return np.stack([dx, dy], axis=-1)


def rk4_step(state: np.ndarray, dt: float, mu: float = 1.0) -> np.ndarray:
    """
    Standard RK4 step for the full system.
    """
    k1 = rhs_full(state, mu)
    k2 = rhs_full(state + 0.5 * dt * k1, mu)
    k3 = rhs_full(state + 0.5 * dt * k2, mu)
    k4 = rhs_full(state + dt * k3, mu)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def generate_dataset(
    num_trajectories: int = 1000,
    T: float = 1e-1,
    dt_small: float = 1e-3,
    dt_flow: float = 1e-1,
    mu: float = 1.0,
    x0_range=(-2.0, 2.0),
    y0_range=(-2.0, 2.0),
    seed: int = 0,
):
    """
    Generate a dataset of flow-map pairs for the ODE

        dx/dt = y
        dy/dt = mu (1 - x^2) y - x

    using a small time step dt_small for "exact" trajectories, and
    a coarser flow-map step dt_flow for the learning target.

    Returns
    -------
    X : torch.Tensor, shape (N, 2)
        States at time t.
    Y : torch.Tensor, shape (N, 2)
        Corresponding states at time t + dt_flow.
    dt : float
        The flow-map step size (dt_flow).
    """
    rng = np.random.default_rng(seed)

    steps_per_flow = int(round(dt_flow / dt_small))
    num_steps = int(round(T / dt_small))
    if steps_per_flow <= 0:
        raise ValueError("dt_flow must be larger than dt_small.")

    X_list = []
    Y_list = []

    for _ in range(num_trajectories):
        x0 = rng.uniform(*x0_range)
        y0 = rng.uniform(*y0_range)
        state = np.array([x0, y0], dtype=np.float64)

        t = 0.0
        for step in range(0, num_steps, steps_per_flow):
            # state at time t
            x_t = state.copy()

            # advance by dt_flow using small RK4 steps
            for _ in range(steps_per_flow):
                state = rk4_step(state, dt_small, mu=mu)
                t += dt_small

            x_tp = state.copy()

            X_list.append(x_t)
            Y_list.append(x_tp)

    X = torch.from_numpy(np.stack(X_list, axis=0))
    Y = torch.from_numpy(np.stack(Y_list, axis=0))
    return X, Y, dt_flow


if __name__ == "__main__":
    X, Y, dt = generate_dataset()
    np.savez(
        "toy_hpc_ode_dataset_dt_flow_1e-1_dt_small_1e-3.npz",
        X=X.numpy(),
        Y=Y.numpy(),
        dt=dt,
    )
