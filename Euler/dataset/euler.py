import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4  # 理想气体常数

# ------------------ 原始/守恒量互转 ------------------
def primitive_to_conserved(rho, u, p):
    E = p / (gamma - 1) + 0.5 * rho * u**2
    return np.stack([rho, rho * u, E], axis=0)

def conserved_to_primitive(U):
    rho = np.maximum(U[0], 1e-12)
    u = U[1] / rho
    E = U[2]
    kinetic = 0.5 * rho * u**2
    p = (gamma - 1) * (E - kinetic)
    p = np.maximum(p, 1e-12)          # 强制正压
    return rho, u, p

# ------------------ 通量 ------------------
def flux(U):
    rho, u, p = conserved_to_primitive(U)
    E = U[2]
    return np.stack([rho * u, rho * u**2 + p, u * (E + p)], axis=0)

# ------------------ 最大波速 ------------------
def max_wave_speed(U):
    rho, u, p = conserved_to_primitive(U)
    c = np.sqrt(np.maximum(gamma * p / rho, 0))
    return np.max(np.abs(u) + c)

def eigen_decomposition(U_avg):
    rho, u, p = conserved_to_primitive(U_avg)
    H = (U_avg[2] + p) / rho
    c = np.sqrt(gamma * p / rho)

    R = np.array([
        [1,           1,         1],
        [u - c,       u,       u + c],
        [H - u*c,     0.5*u**2, H + u*c]
    ])
    Rinv = np.linalg.inv(R)
    return R, Rinv, c



# ------------------ 你的 WENO 实现（完全不变） ------------------
def WENO(a, u, dx, f):
    u = np.pad(u, (3, 3), mode='wrap')  # 改成零梯度边界
    nx = len(u) - 6
    I = np.arange(3, nx + 3)

    # 负方向重构
    vmm, vm, v, vp, vpp = u[I - 2], u[I - 1], u[I], u[I + 1], u[I + 2]
    p0n = (2*vmm - 7*vm + 11*v) / 6
    p1n = (-vm + 5*v + 2*vp) / 6
    p2n = (2*v + 5*vp - vpp) / 6
    B0n = 13/12*(vmm - 2*vm + v)**2 + 1/4*(vmm - 4*vm + 3*v)**2
    B1n = 13/12*(vm - 2*v + vp)**2 + 1/4*(vm - vp)**2
    B2n = 13/12*(v - 2*vp + vpp)**2 + 1/4*(3*v - 4*vp + vpp)**2
    d0n, d1n, d2n = 1.0/10.0, 3.0/5.0, 3.0/10.0
    epsilon = 1e-20
    alpha0n = d0n / (epsilon + B0n)**2
    alpha1n = d1n / (epsilon + B1n)**2
    alpha2n = d2n / (epsilon + B2n)**2
    alpha_sum_n = alpha0n + alpha1n + alpha2n
    w0n, w1n, w2n = alpha0n / alpha_sum_n, alpha1n / alpha_sum_n, alpha2n / alpha_sum_n
    hn = w0n * p0n + w1n * p1n + w2n * p2n

    # 正方向重构
    vmm, vm, v, vp, vpp = u[I - 1], u[I], u[I + 1], u[I + 2], u[I + 3]
    p0p = (-vmm + 5*vm + 2*v) / 6
    p1p = (2*vm + 5*v - vp) / 6
    p2p = (11*v - 7*vp + 2*vpp) / 6
    B0p = 13/12*(vmm - 2*vm + v)**2 + 1/4*(vmm - 4*vm + 3*v)**2
    B1p = 13/12*(vm - 2*v + vp)**2 + 1/4*(vm - vp)**2
    B2p = 13/12*(v - 2*vp + vpp)**2 + 1/4*(3*v - 4*vp + vpp)**2
    d0p, d1p, d2p = 1.0/10.0, 3.0/5.0, 3.0/10.0
    alpha0p = d0p / (epsilon + B0p)**2
    alpha1p = d1p / (epsilon + B1p)**2
    alpha2p = d2p / (epsilon + B2p)**2
    alpha_sum_p = alpha0p + alpha1p + alpha2p
    w0p, w1p, w2p = alpha0p / alpha_sum_p, alpha1p / alpha_sum_p, alpha2p / alpha_sum_p
    hp = w0p * p0p + w1p * p1p + w2p * p2p

    LF = 0.5 * (f(hn) + f(hp) - a * (hp - hn))
    res = (LF - np.roll(LF, 1)) / dx
    return -res

# ------------------ 你的 WENO_flux_full（完全不变） ------------------
def WENO_flux_full(U, dx):
    res = np.zeros_like(U)
    a = max_wave_speed(U)

    for i in range(U.shape[0]):  # 对每个分量做特征分解
        def flux_i(u_i):
            U_mod = U.copy()
            U_mod[i] = u_i
            return flux(U_mod)[i]
        
        res[i] = WENO(a, U[i], dx, flux_i)
    
    return res

def RK4_step(U, dx, dt):
    def rhs(U):
        return WENO_flux_full(U, dx)
    
    k1 = rhs(U)
    k2 = rhs(U + 0.5 * dt * k1)
    k3 = rhs(U + 0.5 * dt * k2)
    k4 = rhs(U + dt * k3)
    return U + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)


def solve_euler():
    nx = 500
    x = np.linspace(0, 1, nx, endpoint=False)
    dx = x[1] - x[0]
    t = 0.0
    t_end = 0.2
    CFL = 0.4

    # 周期条件下简化Sod激波管初始条件（你也可以换成光滑波）
    rho_L, u_L, p_L = 1.0, 0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1

    rho = 1 + 0.2 * np.sin(2 * np.pi * x)
    u   = 0.7 * np.ones_like(x)
    p   = np.ones_like(x)

    U = primitive_to_conserved(rho, u, p)

    while t < t_end:
        a = max_wave_speed(U)
        dt = CFL * dx / a
        if t + dt > t_end:
            dt = t_end - t
        U = RK4_step(U, dx, dt)
        t += dt
        print(f"[Euler] t = {t:.5f}")

    rho, u, p = conserved_to_primitive(U)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(x, rho, label="Density")
    plt.xlabel("x"); plt.ylabel("Density"); plt.grid(True)
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(x, u, label="Velocity")
    plt.xlabel("x"); plt.ylabel("Velocity"); plt.grid(True)
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(x, p, label="Pressure")
    plt.xlabel("x"); plt.ylabel("Pressure"); plt.grid(True)
    plt.legend()

    plt.suptitle("1D Euler Equations Solution (Periodic BC)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    solve_euler()
