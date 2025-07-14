
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Layer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.rand(out_features, in_features, device=device) * 0.05)
        self.gain = nn.Parameter(torch.tensor(1.0, device=device))
        self.feedback = nn.Parameter(torch.rand(out_features, in_features, device=device) * 0.02)

    def adaptive_log_compress(self, x):
        return torch.sign(x) * torch.log1p(self.gain * torch.abs(x) + 1e-6)

    def forward(self, x, feedback_inhibition=None):
        out = self.W @ x
        if feedback_inhibition is not None:
            out = out - feedback_inhibition
        return self.adaptive_log_compress(out)

class MultiLayerNetwork(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super().__init__()
        self.layers = nn.ModuleList([
            Layer(input_neurons, hidden_neurons),
            Layer(hidden_neurons, output_neurons)
        ])
        self.hidden_neurons = [hidden_neurons]
        self.output_neurons = output_neurons
        self.a, self.b, self.c, self.d = 0.02, 0.2, -65.0, 8.0
        self.dt = 1.0
        self.steps = 100

    def simulate(self, v_input, target_output):
        total_neurons = sum(self.hidden_neurons) + self.output_neurons
        v_vec = torch.zeros(self.steps, total_neurons, device=device)
        u_vec = torch.zeros_like(v_vec)
        v_vec[0] = torch.tensor([-65.0] * total_neurons, device=device)
        u_vec[0] = self.b * v_vec[0]
        loss = 0.0

        for t in range(1, self.steps):
            x = v_input
            outputs = []
            for i, layer in enumerate(self.layers):
                feedback = None
                if layer.feedback.shape[1] == x.shape[0]:
                    feedback = layer.feedback @ x
                x = layer(x, feedback)
                outputs.append(x)

            full_input = torch.cat(outputs, dim=0)
            v_prev, u_prev = v_vec[t - 1], u_vec[t - 1]
            v = v_prev + self.dt * (0.04 * v_prev ** 2 + 5 * v_prev + 0.140 - u_prev + full_input)
            u = u_prev + self.dt * self.a * (self.b * v_prev - u_prev)
            spiked = v >= 30
            v[spiked], u[spiked] = self.c, u[spiked] + self.d
            v_vec[t], u_vec[t] = v, u

            if t == self.steps - 1:
                loss = nn.MSELoss()(outputs[-1], target_output)

        return loss, v_vec, u_vec

    def grow_hidden_layer(self, new_hidden_neurons):
        hidden_layer = self.layers[0]
        next_layer = self.layers[1]

        W1_old = hidden_layer.W.data
        new_W1 = torch.rand(W1_old.shape[0] + new_hidden_neurons, W1_old.shape[1], device=device) * 0.05
        new_W1[:W1_old.shape[0], :] = W1_old
        hidden_layer.W = nn.Parameter(new_W1)

        fb_old = hidden_layer.feedback.data
        new_fb = torch.rand(new_W1.shape[0], new_W1.shape[1], device=device) * 0.02
        new_fb[:fb_old.shape[0], :] = fb_old
        hidden_layer.feedback = nn.Parameter(new_fb)

        W2_old = next_layer.W.data
        new_W2 = torch.rand(W2_old.shape[0], W2_old.shape[1] + new_hidden_neurons, device=device) * 0.05
        new_W2[:, :W2_old.shape[1]] = W2_old
        next_layer.W = nn.Parameter(new_W2)

        self.hidden_neurons[0] += new_hidden_neurons
        print(f"🧠 Grew hidden neurons from {W1_old.shape[0]} → {new_W1.shape[0]}")

# Parameters
input_neurons, hidden_neurons, output_neurons = 2, 3, 2
model = MultiLayerNetwork(input_neurons, hidden_neurons, output_neurons).to(device)

v_input = torch.tensor([-0.8, 0.8], dtype=torch.float32, device=device)
target_output = torch.tensor([0.3, -0.3], dtype=torch.float32, device=device)

loss_history = []
prev_loss = None
growth_cooldown = 0
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(150):
    optimizer.zero_grad()
    loss, v_vec, u_vec = model.simulate(v_input, target_output)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Hidden Neurons = {model.hidden_neurons[-1]}")
        if growth_cooldown > 0:
            growth_cooldown -= 1
        elif prev_loss is not None and abs(prev_loss - loss.item()) < 1e-4:
            model.grow_hidden_layer(new_hidden_neurons=1)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            growth_cooldown = 5
        prev_loss = loss.item()

# Plot loss
plt.plot(loss_history)
plt.title("Loss Over Time with Multi-Layer Growth")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
