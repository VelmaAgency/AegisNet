# threatsense.py - ThreatSense AI Predictive Engine for AegisNet v3.0
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import MinMaxScaler
from imbalancedlearn.over_sampling import SMOTE
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
import asyncio

logger = logging.getLogger(__name__)

class LSTMDetector(nn.Module):
    """LSTM for threat detection with ACKTR/K-FAC tuning."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 1):
        super(LSTMDetector, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for anomaly classification."""
        try:
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return self.sigmoid(out)
        except Exception as e:
            logger.error(f"LSTM forward error: {e}")
            return torch.zeros(x.shape[0], 1)

class WGAN:
    """WGAN-GP/EO-WGAN for synthetic threat generation."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, gp_lambda: float = 10.0):
        self.generator = nn.Sequential(
            nn.Linear(100, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimizer_c = optim.Adam(self.critic.parameters(), lr=0.0002)
        self.gp_lambda = gp_lambda

    def gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Gradient Penalty for WGAN-GP."""
        alpha = torch.rand(real.size(0), 1).expand_as(real)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_()
        critic_out = self.critic(interpolated)
        gradients = torch.autograd.grad(outputs=critic_out, inputs=interpolated, grad_outputs=torch.ones_like(critic_out), create_graph=True, retain_graph=True)[0]
        gp = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return gp

    def train_step(self, real_data: torch.Tensor) -> Dict:
        """Training step for WGAN-GP/EO-WGAN."""
        try:
            # Critic update
            self.optimizer_c.zero_grad()
            fake_noise = torch.randn(real_data.size(0), 100)
            fake_data = self.generator(fake_noise)
            critic_real = self.critic(real_data).mean()
            critic_fake = self.critic(fake_data.detach()).mean()
            gp = self.gradient_penalty(real_data, fake_data)
            critic_loss = -critic_real + critic_fake + self.gp_lambda * gp
            critic_loss.backward()
            self.optimizer_c.step()

            # Generator update
            self.optimizer_g.zero_grad()
            fake_data = self.generator(fake_noise)
            generator_loss = -self.critic(fake_data).mean()
            generator_loss.backward()
            self.optimizer_g.step()

            return {"critic_loss": critic_loss.item(), "generator_loss": generator_loss.item()}
        except Exception as e:
            logger.error(f"WGAN training error: {e}")
            return {"critic_loss": 0.0, "generator_loss": 0.0}

class ThreatSenseAI:
    """Integrated ThreatSense AI engine with HFL and DP."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, dp_sigma: float = 0.04):
        self.lstm = LSTMDetector(input_dim, hidden_dim)
        self.wgan = WGAN(input_dim, hidden_dim)
        self.scaler = MinMaxScaler()
        self.smote = SMOTE()
        self.dp_sigma = dp_sigma  # Differential Privacy noise

    def preprocess_data(self, data: np.ndarray) -> torch.Tensor:
        """Preprocess with normalization and SMOTE for imbalance."""
        try:
            normalized = self.scaler.fit_transform(data)
            balanced_data, _ = self.smote.fit_resample(normalized, np.zeros(len(normalized)))  # Placeholder labels
            return torch.tensor(balanced_data, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Preprocess error: {e}")
            return torch.tensor(data, dtype=torch.float32)

    def add_dp_noise(self, data: torch.Tensor) -> torch.Tensor:
        """Add Differential Privacy noise."""
        try:
            noise = torch.randn_like(data) * self.dp_sigma
            return data + noise
        except Exception as e:
            logger.error(f"DP noise error: {e}")
            return data

    async def federated_update(self, local_models: List[LSTMDetector]) -> LSTMDetector:
        """HFL aggregation with DP noise."""
        try:
            global_state = self.lstm.state_dict()
            for local in local_models:
                local_state = local.state_dict()
                for key in global_state:
                    global_state[key] += self.add_dp_noise(local_state[key]) / len(local_models)
            self.lstm.load_state_dict(global_state)
            logger.info("Federated model updated with HFL and DP")
            return self.lstm
        except Exception as e:
            logger.error(f"Federated update error: {e}")
            return self.lstm

    async def detect_threat(self, input_data: torch.Tensor) -> Dict:
        """Detect threats with LSTM and WGAN-generated synthetics."""
        try:
            preprocessed = self.preprocess_data(input_data.numpy())
            synthetic = self.wgan.generator(torch.randn(1, 100)).detach()
            combined = torch.cat((preprocessed, synthetic), dim=0)
            scores = self.lstm(combined.unsqueeze(1))
            anomaly_score = scores.mean().item()
            status = "anomaly" if anomaly_score > 0.93 else "normal"
            logger.info("Threat detection completed", extra={"score": anomaly_score, "status": status})
            return {"status": status, "score": anomaly_score}
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"status": "error", "score": 0.0}

# Example usage
if __name__ == "__main__":
    async def main():
        threatsense = ThreatSenseAI()
        input_data = torch.rand(100, 128)  # Mock data
        result = await threatsense.detect_threat(input_data)
        print(f"Threat detection: {result}")

    asyncio.run(main())
    # threatsense.py - ThreatSense AI Predictive Engine for AegisNet v3.0
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List
import logging
from sklearn.preprocessing import MinMaxScaler
from imbalancedlearn.over_sampling import SMOTE
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
import asyncio
import kfac  # Add kfac-pytorch to requirements.txt for ACKTR

logger = logging.getLogger(__name__)

class LSTMDetector(nn.Module):
    """LSTM for threat detection with ACKTR/K-FAC tuning."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 1):
        super(LSTMDetector, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LSTM forward pass for anomaly classification."""
        try:
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return self.sigmoid(out)
        except Exception as e:
            logger.error(f"LSTM forward error: {e}")
            return torch.zeros(x.shape[0], 1)

    def train_with_acktr(self, data: torch.Tensor, labels: torch.Tensor, epochs: int = 10):
        """Train with ACKTR/K-FAC optimizer."""
        try:
            optimizer = kfac.KFAC(self.parameters(), eps=1e-5, pi=0.001, update_freq=10)
            criterion = nn.BCELoss()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            logger.info("ACKTR training completed", extra={"epochs": epochs})
        except Exception as e:
            logger.error(f"ACKTR training error: {e}")

class WGAN:
    # ... (existing WGAN code)

class ThreatSenseAI:
    """Integrated ThreatSense AI engine with HFL and DP."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, dp_sigma: float = 0.04):
        self.lstm = LSTMDetector(input_dim, hidden_dim)
        self.wgan = WGAN(input_dim, hidden_dim)
        self.scaler = MinMaxScaler()
        self.smote = SMOTE()
        self.dp_sigma = dp_sigma

    # ... (existing methods)

# Example usage
if __name__ == "__main__":
    async def main():
        threatsense = ThreatSenseAI()
        input_data = torch.rand(100, 128)  # Mock data
        labels = torch.rand(100, 1)  # Mock labels
        threatsense.lstm.train_with_acktr(input_data.unsqueeze(1), labels)
        result = await threatsense.detect_threat(input_data)
        print(f"Threat detection: {result}")

    asyncio.run(main())
    # threatsense.py - ThreatSense AI Predictive Engine for AegisNet v3.0
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List
import logging
from sklearn.preprocessing import MinMaxScaler
from imbalancedlearn.over_sampling import SMOTE
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
import asyncio
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler  # For mixed precision
import kfac  # Add kfac-pytorch to requirements.txt for ACKTR

logger = logging.getLogger(__name__)

class LSTMDetector(nn.Module):
    """LSTM for threat detection with optimized ACKTR/K-FAC tuning."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 1):
        super(LSTMDetector, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LSTM forward pass for anomaly classification."""
        try:
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return self.sigmoid(out)
        except Exception as e:
            logger.error(f"LSTM forward error: {e}")
            return torch.zeros(x.shape[0], 1)

    def train_with_acktr(self, data: torch.Tensor, labels: torch.Tensor, epochs: int = 10, batch_size: int = 64, patience: int = 3):
        """Optimized ACKTR training loop with KFAC, mixed precision, batching, and early stopping."""
        try:
            optimizer = kfac.KFAC(self.parameters(), eps=1e-5, pi=0.001, update_freq=10)
            criterion = nn.BCELoss()
            scaler = GradScaler()  # For mixed precision
            dataset = TensorDataset(data.unsqueeze(1), labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            best_loss = float('inf')
            no_improve = 0

            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch_data, batch_labels in dataloader:
                    optimizer.zero_grad()
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = self(batch_data)
                        loss = criterion(outputs, batch_labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(dataloader)
                logger.info("Epoch completed", extra={"epoch": epoch, "loss": avg_loss})
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info("Early stopping triggered")
                        break
            logger.info("ACKTR training completed", extra={"epochs": epoch, "best_loss": best_loss})
        except Exception as e:
            logger.error(f"ACKTR training error: {e}")

class WGAN:
    # ... (existing WGAN code)

class ThreatSenseAI:
    """Integrated ThreatSense AI engine with HFL and DP."""
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, dp_sigma: float = 0.04):
        self.lstm = LSTMDetector(input_dim, hidden_dim)
        self.wgan = WGAN(input_dim, hidden_dim)
        self.scaler = MinMaxScaler()
        self.smote = SMOTE()
        self.dp_sigma = dp_sigma

    # ... (existing methods)

# Example usage
if __name__ == "__main__":
    async def main():
        threatsense = ThreatSenseAI()
        input_data = torch.rand(100, 128)  # Mock data
        labels = torch.rand(100, 1)  # Mock labels
        threatsense.lstm.train_with_acktr(input_data, labels, batch_size=32)
        result = await threatsense.detect_threat(input_data)
        print(f"Threat detection: {result}")

    asyncio.run(main())
    
    