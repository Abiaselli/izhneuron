
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IzhikevichNet(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super().__init__()
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        self.W1 = nn.Parameter(torch.rand(hidden_neurons, input_neurons, device=device) * 0.05)
        self.W2 = nn.Parameter(torch.rand(output_neurons, hidden_neurons, device=device) * 0.05)
        self.W_feedback = nn.Parameter(torch.rand(hidden_neurons, output_neurons, device=device) * 0.02)

        self.gain1 = nn.Parameter(torch.tensor(1.0, device=device))
        self.gain2 = nn.Parameter(torch.tensor(1.0, device=device))

        self.a = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = 8.0
        self.dt = 1.0
        self.steps = 100

    def adaptive_log_compress(self, x, gain):
        return torch.sign(x) * torch.log1p(gain * torch.abs(x) + 1e-6)

    def simulate(self, v_input, target_output):
        total_neurons = self.hidden_neurons + self.output_neurons
        v_vec = torch.zeros(self.steps, total_neurons, device=device)
        u_vec = torch.zeros_like(v_vec)

        v_vec[0] = torch.tensor([-65.0] * total_neurons, device=device)
        u_vec[0] = self.b * v_vec[0]

        loss = 0.0

        for t in range(1, self.steps):
            I_hidden_raw = self.W1 @ v_input
            I_hidden = self.adaptive_log_compress(I_hidden_raw, self.gain1)

            I_output_raw = self.W2 @ I_hidden
            I_output = self.adaptive_log_compress(I_output_raw, self.gain2)

            # Feedback inhibition from output to hidden
            feedback_signal = self.W_feedback @ I_output
            I_hidden = I_hidden - feedback_signal


            total_input = torch.cat((I_hidden, I_output))
            v_prev = v_vec[t - 1]
            u_prev = u_vec[t - 1]

            v = v_prev + self.dt * (0.04 * v_prev ** 2 + 5 * v_prev + 0.140 - u_prev + total_input)
            u = u_prev + self.dt * self.a * (self.b * v_prev - u_prev)

            spiked = v >= 30
            v[spiked] = self.c
            u[spiked] += self.d

            v_vec[t] = v
            u_vec[t] = u

            if t == self.steps - 1:
                loss = nn.MSELoss()(I_output, target_output)

        return loss, v_vec, u_vec

# Structural growth: add neuron if loss plateaus
def grow_hidden_layer(model):
    hidden_neurons = model.W1.shape[0]
    input_neurons = model.W1.shape[1]
    output_neurons = model.W2.shape[0]

    # New weights with correct shapes
    new_W1_row = nn.Parameter(torch.rand(1, input_neurons, device=device) * 0.05)
    new_W2_col = nn.Parameter(torch.rand(output_neurons, 1, device=device) * 0.05)
    new_feedback = nn.Parameter(torch.rand(1, output_neurons, device=device) * 0.02)

    print("🔧 Growing structure:")
    print(f"  W1: {model.W1.shape} + {new_W1_row.shape}")
    print(f"  W2: {model.W2.shape} + {new_W2_col.T.shape}")
    print(f"  W_feedback: {model.W_feedback.shape} + {new_feedback.shape}")

    with torch.no_grad():
        # Concatenate row to W1
        model.W1 = nn.Parameter(torch.cat([model.W1, new_W1_row], dim=0))

        # Ensure new_W2_col shape matches model.W2 (output_neurons x hidden_neurons)
        model.W2 = nn.Parameter(torch.cat([model.W2, new_W2_col], dim=1))

        # Ensure new_feedback shape matches W_feedback (hidden_neurons x output_neurons)
        model.W_feedback = nn.Parameter(torch.cat([model.W_feedback, new_feedback], dim=0))

        model.hidden_neurons += 1

    print(f"🧠 New hidden neuron added. Total: {model.hidden_neurons}")


# Configuration
input_neurons = 2
hidden_neurons = 3
output_neurons = 2

model = IzhikevichNet(input_neurons, hidden_neurons, output_neurons).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

v_input = torch.tensor([-0.8, 0.8], dtype=torch.float32, device=device)
target_output = torch.tensor([0.3, -0.3], dtype=torch.float32, device=device)

loss_history = []
prev_loss = None
for epoch in range(100):
    optimizer.zero_grad()
    loss, v_vec, u_vec = model.simulate(v_input, target_output)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Hidden Neurons = {model.hidden_neurons}")
        if prev_loss is not None and abs(prev_loss - loss.item()) < 1e-4:
            grow_hidden_layer(model)
        prev_loss = loss.item()

# Plot loss
plt.plot(loss_history)
plt.title("Loss Over Time with Feedback Inhibition + Growth")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
