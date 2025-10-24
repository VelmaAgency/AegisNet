# snn_t48.py - Spiking Neural Networks for T48 anomaly detection
import torch
import torch.nn as nn
from typing import Tuple

class SNN(nn.Module):
    """T48 SNN for temporal anomaly detection in IIoT."""
    def __init__(self, input_size: int = 128, hidden_dim: int = 16, beta: float = 0.85, theta: float = 1.0):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.beta = beta  # Leak factor
        self.theta = theta  # Spike threshold
        self.v = torch.zeros(hidden_dim)  # Neuron potential

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Neuron potential: V(t) = V(t-1) + I(t) - β V(t-1); spike if V(t) > θ."""
        try:
            I = self.fc1(x)  # Input current
            self.v = self.v + I - self.beta * self.v
            spikes = (self.v > self.theta).float()
            self.v = self.v * (1 - spikes)  # Reset spiked neurons
            output = self.fc2(spikes)  # Anomaly score
            return output, spikes
        except Exception as e:
            print(f"SNN error: {e}")
            return torch.zeros(1), torch.zeros(self.v.size())

# Example usage
if __name__ == "__main__":
    snn = SNN()
    input_tensor = torch.rand(1, 128)  # Mock telemetry
    score, spikes = snn(input_tensor)
    print(f"Anomaly score: {score.item()}, Spikes: {spikes.sum().item()}")
