
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IzhikevichNet(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super().__init__()
        self.W1 = nn.Parameter(torch.rand(hidden_neurons, input_neurons, device=device) * 0.05)
        self.W2 = nn.Parameter(torch.rand(output_neurons, hidden_neurons, device=device) * 0.05)
        self.gain1 = nn.Parameter(torch.tensor(1.0, device=device))
        self.gain2 = nn.Parameter(torch.tensor(1.0, device=device))

        self.a = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = 8.0
        self.dt = 1.0  # ms
        self.steps = 100

    def adaptive_log_compress(self, x, gain):
        return torch.sign(x) * torch.log1p(gain * torch.abs(x) + 1e-6)

    def simulate(self, v_input, target_output):
        v_vec = torch.zeros(self.steps, self.W1.shape[0] + self.W2.shape[0], device=device)
        u_vec = torch.zeros_like(v_vec)

        # Initialize v at rest
        v_vec[0] = torch.tensor([-65.0] * v_vec.shape[1], device=device)
        u_vec[0] = self.b * v_vec[0]

        loss = 0.0

        for t in range(1, self.steps):
            input_current = v_input

            I_hidden_raw = self.W1 @ input_current
            I_hidden = self.adaptive_log_compress(I_hidden_raw, self.gain1)

            I_output_raw = self.W2 @ I_hidden
            I_output = self.adaptive_log_compress(I_output_raw, self.gain2)

            total_input = torch.cat((I_hidden, I_output))
            v_prev = v_vec[t-1]
            u_prev = u_vec[t-1]

            v = v_prev + self.dt * (0.04 * v_prev ** 2 + 5 * v_prev + 0.140 - u_prev + total_input)
            u = u_prev + self.dt * self.a * (self.b * v_prev - u_prev)

            # Spike reset logic
            spiked = v >= 30
            v[spiked] = self.c
            u[spiked] = u[spiked] + self.d

            v_vec[t] = v
            u_vec[t] = u

            if t == self.steps - 1:
                loss = nn.MSELoss()(I_output, target_output)

        return loss, v_vec, u_vec

# Configuration
input_neurons = 2
hidden_neurons = 3
output_neurons = 2

model = IzhikevichNet(input_neurons, hidden_neurons, output_neurons).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

v_input = torch.tensor([-0.8, 0.8], dtype=torch.float32, device=device)
target_output = torch.tensor([0.3, -0.3], dtype=torch.float32, device=device)

loss_history = []
for epoch in range(100):
    optimizer.zero_grad()
    loss, v_vec, u_vec = model.simulate(v_input, target_output)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Gain1 = {model.gain1.item():.3f}, Gain2 = {model.gain2.item():.3f}")

# Plot loss
plt.plot(loss_history)
plt.title("Loss Over Time (Izhikevich Dynamics)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
