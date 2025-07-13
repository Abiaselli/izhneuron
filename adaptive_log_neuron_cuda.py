
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Ensure we use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AdaptiveLogNeuronNet(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super().__init__()
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        self.W1 = nn.Parameter(torch.rand(hidden_neurons, input_neurons, device=device) * 0.05)
        self.W2 = nn.Parameter(torch.rand(output_neurons, hidden_neurons, device=device) * 0.05)

        # Learnable compression gains
        self.gain1 = nn.Parameter(torch.tensor(1.0, device=device))
        self.gain2 = nn.Parameter(torch.tensor(1.0, device=device))

    def adaptive_log_compress(self, x, gain):
        return torch.sign(x) * torch.log1p(gain * torch.abs(x) + 1e-6)

    def forward(self, v_input):
        I_hidden_raw = self.W1 @ v_input
        I_hidden = self.adaptive_log_compress(I_hidden_raw, self.gain1)

        I_output_raw = self.W2 @ I_hidden
        I_output = self.adaptive_log_compress(I_output_raw, self.gain2)

        return I_hidden, I_output

# Configuration
input_neurons = 2
hidden_neurons = 3
output_neurons = 2
steps = 100

# Initialize model
model = AdaptiveLogNeuronNet(input_neurons, hidden_neurons, output_neurons).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Input voltage vector
v_input = torch.tensor([-0.8, 0.8], dtype=torch.float32, device=device)
target_output = torch.tensor([0.3, -0.3], dtype=torch.float32, device=device)

# Training loop
loss_history = []
for epoch in range(steps):
    optimizer.zero_grad()
    _, I_output = model(v_input)

    loss = nn.MSELoss()(I_output, target_output)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Gain1 = {model.gain1.item():.3f}, Gain2 = {model.gain2.item():.3f}")

# Plot loss
plt.plot(loss_history)
plt.title("Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

# Final output and gains
print("Final Output:", I_output.detach().cpu().numpy())
print("Final Gains:", model.gain1.item(), model.gain2.item())
