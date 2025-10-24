# in_vehicle_security.py - SNN for in-vehicle anomalies (CAN bus, BLESA/KNOB)
from snn_t48 import SNN
import torch

def detect_can_anomaly(data: torch.Tensor) -> float:
    """SNN auto-encoder for CAN bus/Bluetooth vulns (BLESA/KNOB)."""
    snn = SNN(input_size=data.size(1))
    score, _ = snn(data)
    return score.item()  # Anomaly if >0.93

# Example
if __name__ == "__main__":
    data = torch.rand(1, 64)  # Mock CAN packet
    score = detect_can_anomaly(data)
    print(f"CAN anomaly score: {score}")
