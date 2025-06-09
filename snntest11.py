import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List

# ---------- Surrogate Gradient Spiking Mechanisms ----------
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 30.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        #torch.autograd.set_detect_anomaly(True)
        input, = ctx.saved_tensors
        grad = grad_output.clone()
        return grad * 1.0 / (1.0 + torch.abs(input - 30.0))**2

surrogate_spike = SurrogateSpike.apply

# ---------- Differentiable Izhikevich Neuron Layer ----------
class IzhikevichLayer(nn.Module):
    def __init__(self, num_neurons, batch_size, device='cuda', mode='surrogate', threshold=30.0, scale=10.0):
        super().__init__()
        self.N = num_neurons
        self.B = batch_size
        self.device = device
        self.mode = mode
        self.threshold = threshold
        self.scale = scale

        self.a = nn.Parameter(torch.full((self.B, self.N), 0.02, device=device))
        self.b = nn.Parameter(torch.full((self.B, self.N), 0.2, device=device))
        self.c = nn.Parameter(torch.full((self.B, self.N), -65.0, device=device))
        self.d = nn.Parameter(torch.full((self.B, self.N), 8.0, device=device))
        self.bias = nn.Parameter(torch.full((self.B, self.N), 98.0, device=device))

        self.register_buffer("v", torch.full((self.B, self.N), -65.0, device=device))
        self.register_buffer("u", torch.zeros(self.B, self.N, device=device))
        self.register_buffer("last_spike_time", torch.full((self.B, self.N), -float('inf'), device=device))

    def reset(self):
        with torch.no_grad():
            self.v.fill_(-65.0)
            self.u.copy_(self.b.detach() * self.v.detach())
            self.last_spike_time.fill_(-float('inf'))

    def forward(self, input_current, t, coeffs):
        #torch.autograd.set_detect_anomaly(True)
        v = self.v.detach().clone().requires_grad_(True)
        u = self.u.detach().clone().requires_grad_(True)

        v_sq = v ** 2
        terms = torch.stack([v_sq, v, torch.ones_like(v)], dim=-1)
        dv = torch.einsum('bni,i->bn', terms, coeffs) - u + input_current + self.bias
        du = self.a * (self.b * v - u)

        v_new = v + dv
        u_new = u + du

        if self.mode == 'sigmoid':
            fired = torch.sigmoid((v_new - self.threshold) * self.scale)
        else:
            fired = surrogate_spike(v_new)

        v_post = torch.where(fired > 0.5, self.c, v_new)
        u_post = torch.where(fired > 0.5, u_new + self.d, u_new)

        with torch.no_grad():
            self.v.copy_(v_post)
            self.u.copy_(u_post)
            self.last_spike_time[fired > 0.5] = t

        return fired


# ---------- Trainable SNN Network ----------
class TrainableSNN(nn.Module):
    def __init__(self, layer_sizes: List[int], batch_size: int, mode='surrogate', device='cuda'):
        super().__init__()
        self.B = batch_size
        self.device = device
        self.coeffs = torch.tensor([0.04, 5.0, 140.0], device=device)
        self.layers = nn.ModuleList([
            IzhikevichLayer(layer_sizes[i], batch_size, device=device, mode=mode)
            for i in range(len(layer_sizes))
        ]).to(device)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(batch_size, layer_sizes[i], layer_sizes[i+1]) * 0.5)
            for i in range(len(layer_sizes) - 1)
        ]).to(device)

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward(self, input_series, steps=40):
        #torch.autograd.set_detect_anomaly(True)
        self.reset()
        x = input_series.to(self.device)
        spike_accum = torch.zeros(steps, self.B, self.layers[-1].N, device=self.device)

        for t in range(steps):
            s = self.layers[0](x[t], t, self.coeffs)

            for i in range(1, len(self.layers)):
                w = self.weights[i - 1]
                s = torch.bmm(s.unsqueeze(1), w.clone()).squeeze(1)
                s_in = s.clone()  # preserve autograd history
                s = self.layers[i](s_in, t, self.coeffs)
                if i == len(self.layers) - 1:
                    spike_accum[t] = s
        return spike_accum

# ---------- Simple Binary Pattern Task ----------
def generate_dataset(n=40):
    patterns = torch.tensor([[0, 0, 1], [1, 1, 0]])
    idxs = torch.randint(0, 2, (n,))
    X = patterns[idxs]
    Y = idxs.clone()
    return X, Y

class BitPatternEncoder:
    def __init__(self, T, input_size, strength=20.0):
        self.T = T
        self.N = input_size
        self.strength = strength

    def encode(self, bit_array):
        B = bit_array.shape[0]
        spike_train = torch.zeros(self.T, B, self.N)
        for b in range(B):
            for n in range(self.N):
                if bit_array[b, n] == 1:
                    spike_train[:, b, n] = self.strength
        return spike_train

class SpikeCountDecoder:
    def decode(self, output_spikes):
        return output_spikes.sum(dim=0).argmax(dim=1)

# ---------- Training Script ----------
def train_snn(mode='surrogate'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T = 1, 40
    model = TrainableSNN([3, 6, 2], batch_size=B, mode=mode, device=device)
    encoder = BitPatternEncoder(T, 3)
    decoder = SpikeCountDecoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    EPOCHS = 20
    SAMPLES = 30
    accuracy_curve = []

    for epoch in range(EPOCHS):
        X, Y = generate_dataset(SAMPLES)
        correct = 0
        total_loss = 0
        optimizer.zero_grad()
        for i in range(SAMPLES):
            x = X[i].unsqueeze(0)
            y = Y[i].unsqueeze(0).to(device)
            input_series = encoder.encode(x).to(device)
            optimizer.zero_grad()
            output_spikes = model(input_series, steps=T)
            spike_counts = output_spikes.sum(dim=0)
            loss = loss_fn(spike_counts, y)
            loss.backward(retain_graph=True)
            optimizer.step()

            pred = decoder.decode(output_spikes)
            correct += int(pred.item() == y.item())
            total_loss += loss
        total_loss.backward(retain_graph=True)  # Only ONE backward call
        optimizer.step()

        acc = correct / SAMPLES * 100
        accuracy_curve.append(acc)
        print(f"[{mode}] Epoch {epoch+1:02d} - Accuracy: {acc:.2f}% | Loss: {total_loss:.4f}")

    return accuracy_curve

# ---------- Run Both Modes and Plot ----------
acc_surrogate = train_snn(mode='surrogate')
acc_sigmoid = train_snn(mode='sigmoid')

plt.plot(acc_surrogate, label='Surrogate Spike')
plt.plot(acc_sigmoid, label='Sigmoid Spike')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("SNN Accuracy - Surrogate vs. Sigmoid")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
