import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class DoublePendulumODE(nn.Module):
    def __init__(self):
        super().__init__()
        # 可学习参数
        self.m1 = nn.Parameter(torch.tensor(1.0))
        self.m2 = nn.Parameter(torch.tensor(1.0))
        self.L1 = nn.Parameter(torch.tensor(1.0))
        self.L2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, state):
        theta1, omega1, theta2, omega2 = torch.unbind(state, dim=1)
        m1, m2, L1, L2 = self.m1, self.m2, self.L1, self.L2
        delta = theta1 - theta2
        sin_d = torch.sin(delta)
        cos_d = torch.cos(delta)
        g = 9.8

        # Denominators
        den1 = L1 * (m1 + m2 - m2 * cos_d**2)
        den2 = L2 * (m1 + m2 - m2 * cos_d**2)

        # dtheta/dt = omega
        dtheta1 = omega1
        dtheta2 = omega2

        # domega1/dt
        domega1 = (-L1 * m2 * sin_d * cos_d * omega1**2
                   - L2 * m2 * sin_d * omega2**2
                   - g * (m1 + m2) * torch.sin(theta1)
                   + g * m2 * torch.sin(theta2) * cos_d) / den1

        # domega2/dt
        domega2 = (L1 * (m1 + m2) * sin_d * omega1**2
                   + L2 * m2 * sin_d * cos_d * omega2**2
                   + g * (m1 * torch.sin(theta1) * cos_d - m1 * torch.sin(theta2))
                   + g * m2 * (torch.sin(theta1) * cos_d - torch.sin(theta2))) / den2

        return torch.stack([dtheta1, domega1, dtheta2, domega2], dim=-1)

def rk4_step(func, y, dt):
    k1 = func(y)
    k2 = func(y + dt / 2 * k1)
    k3 = func(y + dt / 2 * k2)
    k4 = func(y + dt * k3)
    return y + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)


batch_size = 16
dt = 1e-2
num_epochs = 1000
lr = 1e-3

data_input = np.load('../dataset/pinn_input.npy')
data_input = data_input[:, 0:4]
data_output = np.load('../dataset/pinn_output.npy')
inputs = torch.tensor(data_input, dtype=torch.float64)
targets = torch.tensor(data_output, dtype=torch.float64)

dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = DoublePendulumODE()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_inputs, batch_targets in loader:
        optimizer.zero_grad()
        pred = rk4_step(model, batch_inputs, dt)
        loss = loss_fn(pred, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_inputs.size(0)
    
    if epoch % 100 == 0 or epoch == num_epochs - 1:
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

learned_params = {
    'm1': model.m1.item(),
    'm2': model.m2.item(),
    'L1': model.L1.item(),
    'L2': model.L2.item()
}

print(learned_params)