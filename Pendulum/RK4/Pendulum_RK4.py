import numpy as np
import matplotlib.pyplot as plt

def ode_function(t, y):
    alpha = 0.1
    beta = 9.80665
    z = np.zeros_like(y)
    z[0] = y[1]
    z[1] = - alpha * y[1] - beta * np.sin(y[0])
    return z

def RK4(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

def plot_trajectories(NN_traj, true_traj, title="Trajectories Comparison", i = 0):
    plt.figure(figsize=(10, 6))
    plt.plot(NN_traj[:, 0], NN_traj[:, 1], label='NN Prediction', color='blue')
    plt.plot(true_traj[:, 0], true_traj[:, 1], label='True Trajectory', color='orange', linestyle='--')
    plt.xlabel('Theta (rad)')
    plt.ylabel('Theta Dot (rad/s)')
    plt.title(title)
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(f"trajectory_comparison_small_{i}.png")
    plt.close()

if __name__ == "__main__":
    dt = 5e-2
    test_trajectory = np.load("../dataset/test_trajectories_fixed.npy")
    j = 0
    abs_test_errors = np.array([])
    rel_test_errors = np.array([])
    for traj in test_trajectory:
        # print(traj.shape)
        u0 = traj[0, :2]
        u_lists = [u0]
        for i in range(1000):
            u0 = RK4(ode_function, 0, u0, dt)
            u_lists.append(u0)
        u_lists = np.array(u_lists)
        plot_trajectories(u_lists, traj[:, :2], f"Trajectory Comparison {j}", j)
        j += 1

        abs_error = np.abs(u_lists - traj[:, :2])
        rel_error = np.linalg.norm(u_lists - traj[:, :2], axis=1) / np.linalg.norm(traj[:, :2], axis=1)
        abs_test_errors = np.append(abs_test_errors, abs_error)
        rel_test_errors = np.append(rel_test_errors, rel_error)
    np.savetxt("abs_test_errors_RK4.txt", abs_test_errors.reshape(-1, 2))
    np.savetxt("rel_test_errors_RK4.txt", rel_test_errors)
