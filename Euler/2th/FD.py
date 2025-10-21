import numpy as np

def Con2Pri(Con, gamma=1.4):
    rho, m, E = Con[:,0], Con[:,1], Con[:,2]
    p = (gamma-1.0)*(E - m**2/rho/2.0)
    return np.array([rho, m/rho, p]).transpose()

def Pri2Con(Pri, gamma=1.4):
    rho, v, p = Pri[:,0], Pri[:,1], Pri[:,2]
    E = p/(gamma-1.0) + rho*v**2/2.0
    return np.array([rho, rho*v, E]).transpose()

def maxLambda(Pri, gamma=1.4):
    rho, v, p = Pri[:,0], Pri[:,1], Pri[:,2]
    return np.abs(v) + np.sqrt(gamma*p/rho)

def Flux(Pri, gamma=1.4):
    rho, v, p = Pri[:,0], Pri[:,1], Pri[:,2]
    E = p/(gamma-1.0) + rho*v**2/2.0
    return np.array([rho*v, rho*v**2+p, (E+p)*v]).transpose()

def WENO(u):
    # return u[2,:]
    stencil_A = [ 1.0/3.0,-7.0/6.0, 11.0/6.0]
    stencil_B = [-1.0/6.0, 5.0/6.0,  1.0/3.0]
    stencil_C = [ 1.0/3.0, 5.0/6.0, -1.0/6.0]
    gamma = [1.0/10.0, 3.0/5.0, 3.0/10.0]
    # epsilon=1e-6
    epsilon=1e-20
    # Approximation from three different stencils
    res_A = stencil_A[0]*u[0,:]+stencil_A[1]*u[1,:]+stencil_A[2]*u[2,:]
    res_B = stencil_B[0]*u[1,:]+stencil_B[1]*u[2,:]+stencil_B[2]*u[3,:]
    res_C = stencil_C[0]*u[2,:]+stencil_C[1]*u[3,:]+stencil_C[2]*u[4,:]
    # Compute smoothness indicators
    beta_A = 13.0/12.0*(u[0,:]-2.0*u[1,:]+u[2,:])**2 + 1.0/4.0*(u[0,:]-4.0*u[1,:]+3*u[2,:])**2
    beta_B = 13.0/12.0*(u[1,:]-2.0*u[2,:]+u[3,:])**2 + 1.0/4.0*(u[1,:]-u[3,:])**2
    beta_C = 13.0/12.0*(u[2,:]-2.0*u[3,:]+u[4,:])**2 + 1.0/4.0*(3.0*u[2,:]-4.0*u[3,:]+u[4,:])**2
    # Compute weights
    tmp_A = gamma[0]/(epsilon+beta_A)**2
    tmp_B = gamma[1]/(epsilon+beta_B)**2
    tmp_C = gamma[2]/(epsilon+beta_C)**2
    tmp=tmp_A+tmp_B+tmp_C
    weight_A=tmp_A/tmp
    weight_B=tmp_B/tmp
    weight_C=tmp_C/tmp
    return (weight_A*res_A+weight_B*res_B+weight_C*res_C)

class FD:
    def __init__(self, N):
        self.Nx = N
        self.Ng = 3 # ghost cell
        self.bc = 0 # periodic
        self.test_Convergence = False

        self.x_beg = 0
        self.x_end = 1.0
        self.t = 0
        self.t_end = 1.0
        self.CFL = 0.4

        self.Flux_func = Flux 
        self.maxLambda = maxLambda  

        self.Cons = np.zeros((self.Nx+2*self.Ng, 3))
        self.Pris = np.zeros((self.Nx+2*self.Ng, 3))
        self.tmp_Cons = np.zeros((self.Nx+2*self.Ng, 3))

        self.Numerical_flux = np.zeros((self.Nx+1, 3))
        self.RHS = np.zeros((self.Nx, 3))

    def Init(self, Pri_func): # 根据Pri_fun初始化self.u
        Nx = self.Nx
        Ng = self.Ng
        self.dx = (self.x_end - self.x_beg)/Nx
        self.xl = self.x_beg + np.linspace(0, Nx-1, Nx)*self.dx
        self.xr = self.xl + self.dx
        self.xc = self.xl + 0.5*self.dx

        self.Pris[Ng:Nx+Ng,:] = Pri_func(self.xc)
        self.Cons[Ng:Nx+Ng,:] = Pri2Con(self.Pris[Ng:Nx+Ng,:])

    def BC(self):
        Ng = self.Ng
        Nx = self.Nx
        if self.bc==0:
            # periodic
            self.Cons[0:Ng,:] = self.Cons[Nx:Nx+Ng,:]
            self.Pris[0:Ng,:] = self.Pris[Nx:Nx+Ng,:]
            self.Cons[Nx+Ng:Nx+2*Ng,:] = self.Cons[Ng:2*Ng,:]
            self.Pris[Nx+Ng:Nx+2*Ng,:] = self.Pris[Ng:2*Ng,:]

        if self.bc==1:
            self.Cons[0:Ng,:] = self.Cons[Ng,:]
            self.Pris[0:Ng,:] = self.Pris[Ng,:]
            self.Cons[Nx+Ng:Nx+2*Ng,:] = self.Cons[Nx+Ng-1,:]
            self.Pris[Nx+Ng:Nx+2*Ng,:] = self.Pris[Nx+Ng-1,:]

        if self.bc==2: # reflective
            for i in range(Ng):
                self.Cons[i,:] = self.Cons[2*Ng-1-i,:]
                self.Pris[i,:] = self.Pris[2*Ng-1-i,:]
                self.Cons[i,1] *= -1
                self.Pris[i,1] *= -1

            for i in range(Nx+Ng, Nx+2*Ng):
                self.Cons[i,:] = self.Cons[2*(Nx+Ng)-1-i,:]
                self.Pris[i,:] = self.Pris[2*(Nx+Ng)-1-i,:]
                self.Cons[i,1] *= -1
                self.Pris[i,1] *= -1

    def Compute_flux(self):
        flux = self.Flux_func(self.Pris)
        alpha = self.maxLambda(self.Pris)

        for i in range(self.Nx+1):
            # alpha_i = np.max(alpha[i:i+6])
            alpha_i = np.max(alpha)

            flux_i = flux[i:i+5,:]
            Cons_i = self.Cons[i:i+5,:]
            flux_1 = WENO(0.5*(flux_i+alpha_i*Cons_i))

            idx = range(i+5,i,-1)
            flux_i = flux[idx,:]
            Cons_i = self.Cons[idx,:]
            flux_2 = WENO(0.5*(flux_i-alpha_i*Cons_i))

            self.Numerical_flux[i] = flux_1 + flux_2

    def enforce_physical_bounds(self):
        Ng, Nx = self.Ng, self.Nx

        # Clamp conservative variables（只对内部区域）
        rho = self.Cons[Ng:Ng+Nx, 0]
        E   = self.Cons[Ng:Ng+Nx, 2]
        m   = self.Cons[Ng:Ng+Nx, 1]

        rho[rho < 1e-6] = 1e-6
        kinetic = 0.5 * m**2 / rho
        internal = E - kinetic
        internal[internal < 1e-8] = 1e-8
        # E[:] = internal + kinetic

        # 更新 self.Cons，重新计算 self.Pris
        self.Cons[Ng:Ng+Nx, 0] = rho
        self.Cons[Ng:Ng+Nx, 2] = E
        self.Pris[Ng:Ng+Nx, :] = Con2Pri(self.Cons[Ng:Ng+Nx, :])

    def Compute_RHS(self):
        Nx = self.Nx
        Ng = self.Ng
        self.BC() # boundary condition
        self.Compute_flux()
        self.RHS[:,:] = (-self.Numerical_flux[1:Nx+1,:]+self.Numerical_flux[0:Nx,:])/self.dx

        if np.any(np.isnan(self.RHS)):
            print("nan in RHS")
            # print(self.RHS)

    def EulerStep(self, dt, flag):
        self.Compute_RHS()
        Ng = self.Ng
        Nx = self.Nx
        self.Cons[Ng:Ng+Nx,:] += dt*self.RHS

        self.enforce_physical_bounds()

        if np.any(np.isnan(self.Cons)):
            print("nan in Cons")
            # exit(-1)
            return False
        if flag == True:
            self.Pris[Ng:Ng+Nx,:] = Con2Pri(self.Cons[Ng:Ng+Nx,:])
        if np.any(np.isnan(self.Pris)):
            print("Pris has nan!")
            print(self.Pris)
            # exit(-1)
            return False
        return True

    def SSPRK3(self, dt):
        Ng = self.Ng
        Nx = self.Nx
        Cons_tmp = np.copy(self.Cons) # copy is important , = is just a reference
        # step 1
        status1 = self.EulerStep(dt, True)
        self.t += dt
        # step 2
        status2 = self.EulerStep(dt, False)
        self.Cons = 0.75*Cons_tmp+0.25*self.Cons
        self.Pris[Ng:Ng+Nx,:] = Con2Pri(self.Cons[Ng:Ng+Nx,:])
        self.t -= 0.5*dt
        # step 3
        status3 = self.EulerStep(dt, False)
        self.Cons = (Cons_tmp + 2.0*self.Cons)/3.0
        self.Pris[Ng:Ng+Nx,:] = Con2Pri(self.Cons[Ng:Ng+Nx,:])
        self.t -= 0.5*dt

        if status1 and status2 and status3:
            return True
        else:
            print("SSPRK3 failed, retrying...")
            return False

    def Compute_dt(self):
        Ng = self.Ng
        Nx = self.Nx
        alpha_max = np.max(self.maxLambda(self.Pris[Ng:Ng+Nx,:]))
        return self.CFL*self.dx/alpha_max

    def Solve(self):
        self.t = 0
        while self.t<self.t_end:
            dt = self.Compute_dt()
            if dt < 1e-12:
                print(f"Warning: dt={dt} is too small, stopping simulation.")
                return False
            if self.test_Convergence == True:
                dt = np.min((dt, self.dx**(5.0/3.0)))# fourth order
            if self.t+dt>self.t_end:
                dt = self.t_end-self.t
            status = self.SSPRK3(dt)
            if not status:
                return False
            # self.EulerStep(dt, True)
            self.t += dt
            # print("t = %1.2e, dt = %1.2e"%(self.t, dt))
        return True
        
    def Compute_Err(self, Pri_exact, idx):
        Ng = self.Ng
        Nx = self.Nx
        dx = self.dx

        Pri = self.Pris[Ng:Ng+Nx,:]
        Pri_e = Pri_exact(self.xc, self.t_end)
        self.err = np.abs(Pri[:,idx]-Pri_e[:,idx])
        errL1 = np.mean(self.err)
        errL2 = np.sqrt(np.mean(self.err**2))
        errLinf = np.max(self.err)
        return np.array([errL1, errL2, errLinf])
    

