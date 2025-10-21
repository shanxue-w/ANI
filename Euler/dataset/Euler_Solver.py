import torch
import numpy as np # Keep numpy for initial data generation if needed, or convert everything to torch

def Con2Pri(Con, gamma=1.4):
    # Con: [batch, N, 3]
    rho, m, E = Con[:,:,0], Con[:,:,1], Con[:,:,2]
    p = (gamma-1.0)*(E - m**2/rho/2.0)
    return torch.stack([rho, m/rho, p], dim=2) # Stacking along a new dimension for [batch, N, 3]

def Pri2Con(Pri, gamma=1.4):
    # Pri: [batch, N, 3]
    rho, v, p = Pri[:,:,0], Pri[:,:,1], Pri[:,:,2]
    E = p/(gamma-1.0) + rho*v**2/2.0
    return torch.stack([rho, rho*v, E], dim=2)

def maxLambda(Pri, gamma=1.4):
    # Pri: [batch, N, 3]
    rho, v, p = Pri[:,:,0], Pri[:,:,1], Pri[:,:,2]
    return torch.abs(v) + torch.sqrt(gamma*p/rho)

def Flux(Pri, gamma=1.4):
    # Pri: [batch, N, 3]
    rho, v, p = Pri[:,:,0], Pri[:,:,1], Pri[:,:,2]
    E = p/(gamma-1.0) + rho*v**2/2.0
    return torch.stack([rho*v, rho*v**2+p, (E+p)*v], dim=2)

def WENO(u):
    stencil_A = torch.tensor([1.0/3.0,-7.0/6.0, 11.0/6.0], device=u.device, dtype=u.dtype)
    stencil_B = torch.tensor([-1.0/6.0, 5.0/6.0, 1.0/3.0], device=u.device, dtype=u.dtype)
    stencil_C = torch.tensor([1.0/3.0, 5.0/6.0, -1.0/6.0], device=u.device, dtype=u.dtype)
    gamma = torch.tensor([1.0/10.0, 3.0/5.0, 3.0/10.0], device=u.device, dtype=u.dtype)
    epsilon = 1e-20

    # u shape: [batch_size, 5, 3]
    # Approximation from three different stencils
    res_A = stencil_A[0]*u[:,0,:] + stencil_A[1]*u[:,1,:] + stencil_A[2]*u[:,2,:] # [batch_size, 3]
    res_B = stencil_B[0]*u[:,1,:] + stencil_B[1]*u[:,2,:] + stencil_B[2]*u[:,3,:] # [batch_size, 3]
    res_C = stencil_C[0]*u[:,2,:] + stencil_C[1]*u[:,3,:] + stencil_C[2]*u[:,4,:] # [batch_size, 3]

    # Compute smoothness indicators
    # beta_A shape: [batch_size]
    beta_A = 13.0/12.0*(u[:,0,:]-2.0*u[:,1,:]+u[:,2,:])**2 + 1.0/4.0*(u[:,0,:]-4.0*u[:,1,:]+3*u[:,2,:])**2
    beta_B = 13.0/12.0*(u[:,1,:]-2.0*u[:,2,:]+u[:,3,:])**2 + 1.0/4.0*(u[:,1,:]-u[:,3,:])**2
    beta_C = 13.0/12.0*(u[:,2,:]-2.0*u[:,3,:]+u[:,4,:])**2 + 1.0/4.0*(3.0*u[:,2,:]-4.0*u[:,3,:]+u[:,4,:])**2
    
    # Sum over the components (3) to get a single scalar for each batch element
    beta_A = beta_A.sum(dim=-1) # [batch_size]
    beta_B = beta_B.sum(dim=-1)
    beta_C = beta_C.sum(dim=-1)

    # Compute weights
    tmp_A = gamma[0]/(epsilon+beta_A)**2 # [batch_size]
    tmp_B = gamma[1]/(epsilon+beta_B)**2
    tmp_C = gamma[2]/(epsilon+beta_C)**2
    
    tmp = tmp_A+tmp_B+tmp_C # [batch_size]
    
    # Expand dims for broadcasting with res_A, res_B, res_C
    weight_A = (tmp_A/tmp).unsqueeze(-1) # [batch_size, 1]
    weight_B = (tmp_B/tmp).unsqueeze(-1)
    weight_C = (tmp_C/tmp).unsqueeze(-1)
    
    return (weight_A*res_A + weight_B*res_B + weight_C*res_C) # [batch_size, 3]


class FD:
    def __init__(self, N, batch_size=1, device='cuda', dtype=torch.float64):
        self.Nx = N
        self.Ng = 3 # ghost cell
        self.bc = 0 # periodic
        self.test_Convergence = False
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        self.x_beg = 0.0
        self.x_end = 1.0
        self.t = 0.0
        self.t_end = 1.0
        self.CFL = 0.4

        self.Flux_func = Flux 
        self.maxLambda = maxLambda 

        # Initialize tensors on the specified device and dtype
        self.Cons = torch.zeros((self.batch_size, self.Nx+2*self.Ng, 3), device=self.device, dtype=self.dtype)
        self.Pris = torch.zeros((self.batch_size, self.Nx+2*self.Ng, 3), device=self.device, dtype=self.dtype)
        self.tmp_Cons = torch.zeros((self.batch_size, self.Nx+2*self.Ng, 3), device=self.device, dtype=self.dtype)

        self.Numerical_flux = torch.zeros((self.batch_size, self.Nx+1, 3), device=self.device, dtype=self.dtype)
        self.RHS = torch.zeros((self.batch_size, self.Nx, 3), device=self.device, dtype=self.dtype)

    def Init(self, Pri_func): # 根据Pri_fun初始化self.u
        Nx = self.Nx
        Ng = self.Ng
        self.dx = (self.x_end - self.x_beg)/Nx
        self.xl = torch.linspace(self.x_beg, self.x_end - self.dx, Nx, device=self.device, dtype=self.dtype)
        self.xr = self.xl + self.dx
        self.xc = self.xl + 0.5*self.dx

        x_batch = self.xc.unsqueeze(0).repeat(self.batch_size, 1)  # [batch, Nx]
        initial_pris = Pri_func(x_batch)  # [batch, Nx, 3]

        self.Pris[:, Ng:Nx+Ng, :] = initial_pris
        self.Cons[:, Ng:Nx+Ng, :] = Pri2Con(initial_pris)

    def BC(self):
        Ng = self.Ng
        Nx = self.Nx
        if self.bc==0: # periodic
            self.Cons[:, 0:Ng, :] = self.Cons[:, Nx:Nx+Ng, :]
            self.Pris[:, 0:Ng, :] = self.Pris[:, Nx:Nx+Ng, :]
            self.Cons[:, Nx+Ng:Nx+2*Ng, :] = self.Cons[:, Ng:2*Ng, :]
            self.Pris[:, Nx+Ng:Nx+2*Ng, :] = self.Pris[:, Ng:2*Ng, :]

        if self.bc==1: # fixed boundary conditions
            self.Cons[:, 0:Ng, :] = self.Cons[:, Ng, :].unsqueeze(1)
            self.Pris[:, 0:Ng, :] = self.Pris[:, Ng, :].unsqueeze(1)
            self.Cons[:, Nx+Ng:Nx+2*Ng, :] = self.Cons[:, Nx+Ng-1, :].unsqueeze(1)
            self.Pris[:, Nx+Ng:Nx+2*Ng, :] = self.Pris[:, Nx+Ng-1, :].unsqueeze(1)

        if self.bc==2: # reflective
            for i in range(Ng):
                self.Cons[:, i, :] = self.Cons[:, 2*Ng-1-i, :]
                self.Pris[:, i, :] = self.Pris[:, 2*Ng-1-i, :]
                self.Cons[:, i, 1] *= -1 # Invert momentum
                self.Pris[:, i, 1] *= -1

            for i in range(Nx+Ng, Nx+2*Ng):
                self.Cons[:, i, :] = self.Cons[:, 2*(Nx+Ng)-1-i, :]
                self.Pris[:, i, :] = self.Pris[:, 2*(Nx+Ng)-1-i, :]
                self.Cons[:, i, 1] *= -1
                self.Pris[:, i, 1] *= -1

    def enforce_physical_bounds(self):
        Ng, Nx = self.Ng, self.Nx

        # Clamp conservative variables（只对内部区域）
        rho = self.Cons[:, Ng:Ng+Nx, 0]
        E   = self.Cons[:, Ng:Ng+Nx, 2]
        m   = self.Cons[:, Ng:Ng+Nx, 1]

        rho[rho < 1e-6] = 1e-6
        kinetic = 0.5 * m**2 / rho
        internal = E - kinetic
        internal[internal < 1e-8] = 1e-8
        E[:] = internal + kinetic

        # 更新 self.Cons，重新计算 self.Pris
        self.Cons[:, Ng:Ng+Nx, 0] = rho
        self.Cons[:, Ng:Ng+Nx, 2] = E
        self.Pris[:, Ng:Ng+Nx, :] = Con2Pri(self.Cons[:, Ng:Ng+Nx, :])


    def Compute_flux(self):
        flux = self.Flux_func(self.Pris) # [batch, N_total, 3]
        alpha = self.maxLambda(self.Pris) # [batch, N_total]

        for i in range(self.Nx+1):
            alpha_i = torch.max(alpha, dim=1)[0] # [batch_size]

            flux_i_stencil = flux[:, i:i+5, :] # [batch_size, 5, 3]
            Cons_i_stencil = self.Cons[:, i:i+5, :] # [batch_size, 5, 3]
            flux_1 = WENO(0.5*(flux_i_stencil + alpha_i.view(-1, 1, 1) * Cons_i_stencil)) # [batch_size, 3]

            flux_i_reversed = torch.stack([flux[:, i+5, :], flux[:, i+4, :], flux[:, i+3, :], flux[:, i+2, :], flux[:, i+1, :]], dim=1) # [batch_size, 5, 3]
            Cons_i_reversed = torch.stack([self.Cons[:, i+5, :], self.Cons[:, i+4, :], self.Cons[:, i+3, :], self.Cons[:, i+2, :], self.Cons[:, i+1, :]], dim=1)

            flux_2 = WENO(0.5*(flux_i_reversed - alpha_i.view(-1, 1, 1) * Cons_i_reversed)) # [batch_size, 3]


            self.Numerical_flux[:, i, :] = flux_1 + flux_2

    def Compute_RHS(self):
        Nx = self.Nx
        Ng = self.Ng
        self.BC() # boundary condition
        self.Compute_flux()
        self.RHS[:, :, :] = (-self.Numerical_flux[:, 1:Nx+1, :] + self.Numerical_flux[:, 0:Nx, :])/self.dx

        if torch.any(torch.isnan(self.RHS)):
            print("nan in RHS")
            print(self.RHS)
            exit(-1) # Consider a more robust error handling for batch processing

    def EulerStep(self, dt, flag):
        self.Compute_RHS()
        Ng = self.Ng
        Nx = self.Nx
        self.Cons[:, Ng:Ng+Nx, :] += dt*self.RHS

        self.enforce_physical_bounds()

        if torch.any(torch.isnan(self.Cons)):
            print("nan in Cons")
            exit(-1)
        if flag == True:
            self.Pris[:, Ng:Ng+Nx, :] = Con2Pri(self.Cons[:, Ng:Ng+Nx, :])
        if torch.any(torch.isnan(self.Pris)):
            print("Pris has nan!")
            print(self.Pris)
            exit(-1)

    def SSPRK3(self, dt):
        Ng = self.Ng
        Nx = self.Nx
        Cons_tmp = self.Cons.clone() # clone is important

        # step 1
        self.EulerStep(dt, True)
        self.t += dt

        # step 2
        # Note: self.Cons is already updated from step 1
        self.EulerStep(dt, False)
        self.Cons = 0.75*Cons_tmp + 0.25*self.Cons
        self.Pris[:, Ng:Ng+Nx, :] = Con2Pri(self.Cons[:, Ng:Ng+Nx, :])
        self.t -= 0.5*dt # Revert time for next step's dt calculation if needed, or if dt is constant

        # step 3
        self.EulerStep(dt, False)
        self.Cons = (Cons_tmp + 2.0*self.Cons)/3.0
        self.Pris[:, Ng:Ng+Nx, :] = Con2Pri(self.Cons[:, Ng:Ng+Nx, :])
        self.t -= 0.5*dt # Revert time again

    def Compute_dt(self):
        Ng = self.Ng
        Nx = self.Nx
        # alpha_max now needs to be the maximum across all batches and all relevant cells
        alpha_max = torch.max(self.maxLambda(self.Pris[:, Ng:Ng+Nx, :]))
        return self.CFL*self.dx/alpha_max

    def Solve(self):
        self.t = 0.0
        while self.t < self.t_end:
            dt = self.Compute_dt()
            if self.test_Convergence == True:
                dt = min(dt, self.dx**(5.0/3.0)) # Use native Python min for scalar dt
            if self.t + dt > self.t_end:
                dt = self.t_end - self.t
            self.SSPRK3(dt)
            # self.EulerStep(dt, True) # If using only Euler step
            self.t += dt # SSPRK3 already updates self.t, this line is redundant if SSPRK3 updates it correctly.
                          # I've modified SSPRK3 to reflect correct time updates.
            print(f"[FD] t = {self.t:.5f}, dt = {dt:.5e}") # Print current time and dt for debugging

    def Compute_Err(self, Pri_exact, idx):
        Ng = self.Ng
        Nx = self.Nx
        dx = self.dx

        Pri = self.Pris[:, Ng:Ng+Nx, :] # [batch_size, Nx, 3]
        
        # Pri_exact needs to be adjusted to return [batch_size, Nx, 3] or [Nx, 3] for broadcasting
        # Assuming Pri_exact returns [Nx, 3] for a given time
        Pri_e_single_batch = Pri_exact(self.xc, self.t_end).to(self.device, self.dtype) # [Nx, 3]
        Pri_e = Pri_e_single_batch.unsqueeze(0).repeat(self.batch_size, 1, 1) # [batch_size, Nx, 3]

        self.err = torch.abs(Pri[:,:,idx]-Pri_e[:,:,idx]) # [batch_size, Nx]
        errL1 = torch.mean(self.err, dim=1) # [batch_size]
        errL2 = torch.sqrt(torch.mean(self.err**2, dim=1)) # [batch_size]
        errLinf = torch.max(self.err, dim=1)[0] # [batch_size]
        return torch.stack([errL1, errL2, errLinf], dim=1) # [batch_size, 3] (L1, L2, Linf for each batch)
    
    def Get_Cons(self):
        return self.Cons[:, self.Ng:self.Nx+self.Ng, :]