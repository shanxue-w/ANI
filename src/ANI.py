import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np

torch.set_default_dtype(torch.float64)

# class FiLM(nn.Module):
#     def __init__(self, time_dim=1, hidden_dim=128):
#         super().__init__()
#         self.gamma_layer = nn.Sequential(
#             nn.Linear(time_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )
#         self.beta_layer = nn.Sequential(
#             nn.Linear(time_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )

#     def forward(self, x, t):
#         gamma = self.gamma_layer(t)
#         beta = self.beta_layer(t)
#         return gamma * x + beta

class FiLM(nn.Module):
    def __init__(self, time_dim=1, hidden_dim=128):
        super().__init__()
        self.gamma = nn.Linear(time_dim, hidden_dim)
        self.beta = nn.Linear(time_dim, hidden_dim)

    def forward(self, x, t):
        return self.gamma(t) * x + self.beta(t)

class FiLM_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=6, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        self.in_layer = nn.Linear(input_dim - 1, hidden_dim)

        self.film_layers = nn.ModuleList()
        self.hidden_layers_list = nn.ModuleList()

        # 保存中间 hidden_dim 状态以确保最终输出维度正确
        current_dim = hidden_dim

        # 上升阶段
        for _ in range(int((hidden_layers - 1) / 2)):
            next_dim = current_dim * 2
            self.hidden_layers_list.append(nn.Linear(current_dim, next_dim))
            self.film_layers.append(FiLM(time_dim=1, hidden_dim=next_dim))
            current_dim = next_dim

        # 下降阶段
        for _ in range(int((hidden_layers - 1) / 2)):
            next_dim = current_dim // 2
            self.hidden_layers_list.append(nn.Linear(current_dim, next_dim))
            self.film_layers.append(FiLM(time_dim=1, hidden_dim=next_dim))
            current_dim = next_dim

        # 输出层结构分支
        if hidden_layers % 2 == 0:
            self.output_layer = nn.Linear(current_dim, output_dim)
            self.final_fc1 = None
            self.final_fc2 = None
        else:
            self.final_fc1 = nn.Linear(current_dim, current_dim // 2)
            self.final_fc2 = nn.Linear(current_dim // 2, output_dim)
            self.output_layer = None

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x_feat = x[:, :-1]
        t = x[:, -1:].detach()

        h = F.gelu(self.in_layer(x_feat))

        for layer, film in zip(self.hidden_layers_list, self.film_layers):
            h = layer(h)
            h = film(h, t)
            h = F.gelu(h)

        if self.output_layer is not None:
            return self.output_layer(h)
        else:
            h = F.gelu(self.final_fc1(h))
            return self.final_fc2(h)



class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int = 6, hidden_dim: int = 128):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        # self.norms.append(nn.BatchNorm1d(hidden_dim))

        for i in range(int((hidden_layers - 1)/2)):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim*2))
            hidden_dim *= 2
            # self.norms.append(nn.BatchNorm1d(hidden_dim))

        for i in range(int((hidden_layers - 1)/2)):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim//2))
            hidden_dim //= 2
            # self.norms.append(nn.BatchNorm1d(hidden_dim))

        # 最后一两层（不加BatchNorm）
        if hidden_layers % 2 == 0:
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            self.final_fc1 = nn.Linear(hidden_dim, hidden_dim//2)
            self.final_fc2 = nn.Linear(hidden_dim//2, output_dim)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.layers.apply(lambda x: nn.init.xavier_uniform_(x.weight) if isinstance(x, nn.Linear) else None)
        self.layers.apply(lambda x: nn.init.zeros_(x.bias) if isinstance(x, nn.Linear) else None)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # x = self.norms[i](x)
            x = F.gelu(x)

        if self.hidden_layers % 2 == 0:
            x = self.output_layer(x)
        else:
            x = F.gelu(self.final_fc1(x))
            x = self.final_fc2(x)
        return x

class DeepONet(nn.Module):
    def __init__(self, branch: MLP, trunk: MLP):
        super(DeepONet, self).__init__()
        self.branch = branch
        self.trunk = trunk

    
    def forward(self, u: torch.Tensor, t: torch.Tensor):
        # u is multidimeninsional, t is 1D
        # u: (n, d), t: (n, 1)
        # branch: (n, d) -> (n, d)
        # trunk: (n, 1) -> (n, d)
        return u + t * (self.branch(u) * self.trunk(t))

class FILM_ResidualBlockWithT(nn.Module):
    def __init__(self, mlp: FiLM_MLP):
        super(FILM_ResidualBlockWithT, self).__init__()
        # self.mlp = mlp
        self.film_mlp = mlp
        # self.mlp = MLP(input_dim, output_dim, hidden_layers, hidden_dim)

    @abstractmethod
    def forward(self, x):
        # y = self.mlp(x)
        dts = x[..., -1:]  # 假设最后一个维度是时间步长
        u   = x[..., 0:-1]
        y = self.film_mlp(x)
        out = u + dts * y  # 假设第一个维度是 u(t)
        return out

class ResidualBlockWithT(nn.Module):
    def __init__(self, mlp: MLP):
        super(ResidualBlockWithT, self).__init__()
        # self.mlp = mlp
        self.mlp = mlp
        # self.mlp = MLP(input_dim, output_dim, hidden_layers, hidden_dim)

    @abstractmethod
    def forward(self, x):
        # y = self.mlp(x)
        dts = x[..., -1:]  # 假设最后一个维度是时间步长
        u   = x[..., 0:-1]
        y = self.mlp(x)
        out = u + dts * y  # 假设第一个维度是 u(t)
        return out

class RK4(nn.Module):
    def __init__(self, mlp: nn.Module):  # 可以是你自定义的 MLP
        super(RK4, self).__init__()
        self.mlp = mlp

    def forward(self, x, dts):
        u = x[..., 0:1]
        k1 = self.mlp(x)
        temp = u + 0.5 * dts * k1
        k2 = self.mlp(torch.cat((temp, x[..., 1:]), dim=-1))
        temp = u + 0.5 * dts * k2
        k3 = self.mlp(torch.cat((temp, x[..., 1:]), dim=-1))
        k4 = self.mlp(torch.cat((u + dts * k3, x[..., 1:]), dim=-1))
        u = u + (dts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return u

# N0 是一个基类，是构造一个 u(t) -> u(t+delta_t) 的映射，这个我们有一个精确的数值方法或者解析解
class N0(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def single_step(self, u: torch.Tensor, parameters=None, t=None, dt=None, mu=None, sigma=None) -> torch.Tensor:
        """
        并行推进多个状态 u_i(t_i) → u_i(t_i + dt_i)
        u: (n, d)
        t: float 或 (n,)
        dt: float 或 (n,)
        返回 u_next: (n, d)
        """
        pass

    def evolve(self, u0: torch.Tensor, parameters, t0, dt):
        return self.single_step(u0, parameters=parameters, t=t0, dt=dt)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cdouble))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cdouble)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(1, self.width) # input channel is 1: (a(x))

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.mlps = nn.Sequential(
            nn.Linear(1, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, 4 * self.width),
        )

    def forward(self, x, dts):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)         # [B, N, in_dim + 1]
        x = self.fc0(x)                          # [B, N, width]
        x = x.permute(0, 2, 1)                   # [B, width, N]
        x = F.pad(x, [0, self.padding])          # zero padding

        # FiLM modulation parameters
        films = self.mlps(dts)                   # [B, 4 * width]
        films = films.view(-1, 4, self.width)    # [B, 4, width]

        # 1st block
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        gamma, beta = films[:, 0].unsqueeze(-1), films[:, 1].unsqueeze(-1)
        x = gamma * x + beta
        x = F.gelu(x)

        # 2nd block
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        gamma, beta = films[:, 2].unsqueeze(-1), films[:, 3].unsqueeze(-1)
        x = gamma * x + beta
        x = F.gelu(x)

        # 3rd block
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # 4th block
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)  # [B, N, width]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float64)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex128))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex128))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.complex128,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, film_input=1, in_channels=1, out_channels=1):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2+in_channels, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

        self.instancenorm = nn.InstanceNorm2d(self.width, affine=False)

        self.film_mlp = nn.Sequential(
            nn.Linear(film_input, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 4 * self.width * 2)  # 输出4层，每层width个gamma和width个beta
        )


    def apply_film(self, x, gamma, beta):
        # x: [B, C, H, W], gamma, beta: [B, C, 1, 1]
        return gamma * x + beta

    def forward(self, x, dt=None):
        batchsize = x.shape[0]
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = F.pad(x, [0, self.padding, 0, self.padding], mode='circular')

        # Compute FiLM parameters
        if dt is None:
            # default gamma=1, beta=0
            gamma_beta = torch.cat([
                torch.ones(batchsize, 4, self.width, device=x.device),
                torch.zeros(batchsize, 4, self.width, device=x.device)
            ], dim=-1)  # [B,4,width*2]
            gamma_beta = gamma_beta.view(batchsize, 4, 2, self.width)
        else:
            if dt.dim() == 0:
                dt_in = dt.unsqueeze(0).repeat(batchsize, 1)
            elif dt.dim() == 1:
                dt_in = dt.view(batchsize, 1)
            else:
                dt_in = dt
            gamma_beta = self.film_mlp(dt_in)  # [B, 4*width*2]
            gamma_beta = gamma_beta.view(batchsize, 4, 2, self.width)  # [B, layer, (gamma/beta), channel]

        # split gamma and beta per layer
        gamma0, beta0 = gamma_beta[:, 0, 0], gamma_beta[:, 0, 1]
        gamma1, beta1 = gamma_beta[:, 1, 0], gamma_beta[:, 1, 1]
        gamma2, beta2 = gamma_beta[:, 2, 0], gamma_beta[:, 2, 1]
        gamma3, beta3 = gamma_beta[:, 3, 0], gamma_beta[:, 3, 1]

        # reshape for broadcasting
        gamma0 = gamma0.unsqueeze(-1).unsqueeze(-1)
        beta0  = beta0.unsqueeze(-1).unsqueeze(-1)
        gamma1 = gamma1.unsqueeze(-1).unsqueeze(-1)
        beta1  = beta1.unsqueeze(-1).unsqueeze(-1)
        gamma2 = gamma2.unsqueeze(-1).unsqueeze(-1)
        beta2  = beta2.unsqueeze(-1).unsqueeze(-1)
        gamma3 = gamma3.unsqueeze(-1).unsqueeze(-1)
        beta3  = beta3.unsqueeze(-1).unsqueeze(-1)

        # Layer 0
        # x normalization by layer
        x = self.instancenorm(x) 
        x_film = self.apply_film(x, gamma0, beta0)
        x1 = self.conv0(x_film)
        x2 = self.w0(x_film)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 1
        x = self.instancenorm(x)
        x_film = self.apply_film(x, gamma1, beta1)
        x1 = self.conv1(x_film)
        x2 = self.w1(x_film)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 2
        x = self.instancenorm(x)
        x_film = self.apply_film(x, gamma2, beta2)
        x1 = self.conv2(x_film)
        x2 = self.w2(x_film)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 3
        x = self.instancenorm(x)
        x_film = self.apply_film(x, gamma3, beta3)
        x1 = self.conv3(x_film)
        x2 = self.w3(x_film)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float64)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float64)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class ANIBASE(nn.Module):
    def __init__(self, N0_SCHEME: N0, input_dim: int, output_dim: int, hidden_layers: int = 4, hidden_dim: int = 128):
        super(ANIBASE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim

        self.N0_SCHEME = N0_SCHEME
        self.mlp = MLP(input_dim, output_dim, hidden_layers, hidden_dim)
        self.branch = MLP(input_dim, output_dim, hidden_layers, hidden_dim)
        self.trunk = MLP(1, output_dim, hidden_layers, hidden_dim)
        self.deeponet = DeepONet(self.branch, self.trunk)
        self.rk4 = RK4(self.mlp)
        self.film_mlp = FiLM_MLP(input_dim, output_dim, hidden_layers, hidden_dim)
        self.residual_block = ResidualBlockWithT(self.mlp)
        self.film_residual_block = FILM_ResidualBlockWithT(self.film_mlp)

    @abstractmethod
    def get_mu_and_sigma(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    @abstractmethod
    def modify_input(self, x):
        pass

    @abstractmethod
    def modify_output(self, x):
        pass

    @abstractmethod
    def SplitConditions(self, x):
        pass
    
    @abstractmethod
    def forward(self, x):
        # x = self.N0_SCHEME.single_step(x, x[..., -1])
        u, parameters, unused, dts = self.SplitConditions(x)
        # u = self.modify_input(u)
        u = self.N0_SCHEME.single_step(u, parameters=parameters, dt=dts/2.0, mu=self.mu, sigma=self.sigma)
        # input_RK = torch.cat([u, unused], dim=-1)
        # u = self.rk4(input_RK, dts)
        u = self.residual_block(torch.cat([u, unused, dts], dim=-1))
        u = self.N0_SCHEME.single_step(u, parameters=parameters, dt=dts/2.0, mu=self.mu, sigma=self.sigma)
        # u = self.modify_output(u)
        return u

    @abstractmethod
    def predict(self, x):
        # x = self.N0_SCHEME.single_step(x, x[..., -1])
        u, parameters, unused, dts = self.SplitConditions(x)
        u = self.modify_input(u)
        u = self.N0_SCHEME.single_step(u, parameters=parameters, dt=dts/2.0, mu=self.mu, sigma=self.sigma)
        # input_RK = torch.cat([u, unused], dim=-1)
        # u = self.rk4(input_RK, dts)
        u = self.residual_block(torch.cat([u, unused, dts], dim=-1))
        u = self.N0_SCHEME.single_step(u, parameters=parameters, dt=dts/2.0, mu=self.mu, sigma=self.sigma)
        u = self.modify_output(u)
        return u

class LinearCombination(nn.Module):
    def __init__(self, input_dim):
        super(LinearCombination, self).__init__()
        self.linear = nn.Linear(2 * input_dim, input_dim, bias=False)

    def forward(self, u1, u2):
        # 拼接成 [u1 | u2]，形状为 (batch_size, 2 * input_dim)
        combined = torch.cat([u1, u2], dim=-1)
        return self.linear(combined)

class ScalarLinearCombination(nn.Module):
    def __init__(self, a: float = 0.5, b: float = 0.5):
        super(ScalarLinearCombination, self).__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float64))  # 初始值可设为任意

    def forward(self, u1, u2):
        return self.a * u1 + self.b * u2
