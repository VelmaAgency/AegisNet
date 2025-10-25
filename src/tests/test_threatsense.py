# test_threatsense.py - Comprehensive Unit Tests for threatsense.py
import pytest
import torch
from threatsense import LSTMDetector, WGAN, ThreatSenseAI

@pytest.fixture
def lstm_detector():
    return LSTMDetector()

@pytest.fixture
def wgan():
    return WGAN()

@pytest.fixture
def threatsense_ai():
    return ThreatSenseAI()

def test_lstm_forward(lstm_detector):
    """Test LSTM forward pass."""
    data = torch.rand(1, 1, 128)
    output = lstm_detector(data)
    assert output.shape == (1, 1)
    assert 0 <= output.item() <= 1

def test_lstm_train_acktr(lstm_detector):
    """Test ACKTR training."""
    data = torch.rand(100, 1, 128)
    labels = torch.rand(100, 1)
    lstm_detector.train_with_acktr(data, labels, epochs=1, batch_size=50)

def test_wgan_gp(wgan):
    """Test WGAN gradient penalty."""
    real = torch.rand(2, 128)
    fake = torch.rand(2, 128)
    gp = wgan.gradient_penalty(real, fake)
    assert gp.item() >= 0

def test_wgan_train_step(wgan):
    """Test WGAN training step."""
    real_data = torch.rand(2, 128)
    result = wgan.train_step(real_data)
    assert "critic_loss" in result
    assert "generator_loss" in result

def test_threatsense_preprocess(threatsense_ai):
    """Test data preprocessing."""
    data = np.random.rand(100, 128)
    preprocessed = threatsense_ai.preprocess_data(data)
    assert preprocessed.shape == (100, 128)

def test_threatsense_dp_noise(threatsense_ai):
    """Test DP noise addition."""
    data = torch.rand(128)
    noisy = threatsense_ai.add_dp_noise(data)
    assert noisy.shape == data.shape

def test_threatsense_federated_update(threatsense_ai):
    """Test HFL update."""
    local_models = [LSTMDetector() for _ in range(2)]
    updated_model = asyncio.run(threatsense_ai.federated_update(local_models))
    assert isinstance(updated_model, LSTMDetector)

def test_threatsense_detect_threat(threatsense_ai):
    """Test threat detection."""
    input_data = torch.rand(128)
    result = asyncio.run(threatsense_ai.detect_threat(input_data))
    assert "status" in result and "score" in result

if __name__ == "__main__":
    pytest.main(["-v"])