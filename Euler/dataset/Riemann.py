import numpy as np
import matplotlib.pyplot as plt

from FD1D_Euler import FD

def fun_Pri(x):
    N = len(x)
    res = np.zeros((N,3))
    for i in range(N):
        if x[i]<0.5:
            res[i,0]=1.0
            res[i,1]=0.0
            res[i,2]=1.0
        else:
            res[i,0]=0.125
            res[i,1]=0.0
            res[i,2]=0.1
    return res

N = 128 * 2
t_end = 1e-1

FD1 = FD(N)
FD1.t_end = t_end 
# FD1.CFL = 0.0001
FD1.CFL = 0.5
FD1.bc = 1
FD1.Init(fun_Pri)
FD1.Solve()

Ng = FD1.Ng
# fig, ax = plt.subplots()
# ax.plot(FD1.xc, FD1.Pris[Ng:N+Ng,0], "rs", label="FD")
# ax.legend()
# fig.savefig("res.pdf")

# 三张图，分别画rho,u,p
fig, ax = plt.subplots(3,1)
ax[0].plot(FD1.xc, FD1.Pris[Ng:N+Ng,0], label="FD")
ax[1].plot(FD1.xc, FD1.Pris[Ng:N+Ng,1], label="FD")
ax[2].plot(FD1.xc, FD1.Pris[Ng:N+Ng,2], label="FD")
ax[0].legend()
ax[1].legend()
ax[2].legend()
fig.savefig("res.png")


