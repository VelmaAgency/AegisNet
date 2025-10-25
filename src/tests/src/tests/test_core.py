# src/tests/test_core.py - Unit tests for core.py
import pytest
import torch
import yara
from core import DBAbstraction, APTShield, VoiceIntentDetector, filter_prompt, BioTriad

@pytest.fixture
def db_abstraction():
    """Fixture for DBAbstraction."""
    return DBAbstraction(dbs=["SQLite", "Postgres"])

@pytest.fixture
def apt_shield():
    """Fixture for APTShield."""
    return APTShield()

@pytest.fixture
def voice_detector():
    """Fixture for VoiceIntentDetector."""
    return VoiceIntentDetector()

@pytest.fixture
def bio_triad():
    """Fixture for BioTriad."""
    return BioTriad()

def test_db_abstraction_audit_query_valid(db_abstraction):
    """Test valid DB query auditing."""
    assert db_abstraction.audit_query("SELECT * FROM users") == True

def test_db_abstraction_audit_query_injection(db_abstraction):
    """Test SQL injection detection."""
    assert db_abstraction.audit_query("DROP TABLE users") == False

def test_apt_shield_scan_data_cl0p(apt_shield):
    """Test YARA scan for Cl0p/MOVEit."""
    data = b"MOVEit.DMZ X-siLock-Comment"
    matches = apt_shield.scan_data(data)
    assert "MOVEit_Transfer_exploit_webshell_aspx" in matches

def test_apt_shield_scan_data_xworm(apt_shield):
    """Test YARA scan for XWorm."""
    data = b"xworm StartDDos"
    matches = apt_shield.scan_data(data)
    assert "win_xworm_w0" in matches or "Windows_Trojan_Xworm_732e6c12" in matches

def test_apt_shield_error_handling(apt_shield, monkeypatch):
    """Test YARA error handling."""
    def mock_match(*args, **kwargs):
        raise yara.Error("Invalid rule")
    monkeypatch.setattr(apt_shield.rules, "match", mock_match)
    assert apt_shield.scan_data(b"test") == []

def test_voice_detector_deepfake(voice_detector):
    """Test deepfake detection."""
    audio = torch.ones(100) * 0.95  # Above threshold
    assert voice_detector.detect_deepfake(audio) > 0.93
    audio = torch.ones(100) * 0.5  # Below threshold
    assert voice_detector.detect_deepfake(audio) == 0.0

def test_filter_prompt_sanitization():
    """Test prompt injection filtering."""
    assert filter_prompt("Safe <input>") == "Safe &lt;input&gt;"
    assert filter_prompt("malicious code") == ""

def test_bio_triad_planarian_healing(bio_triad):
    """Test PlanarianHealing recovery."""
    data = torch.rand(128)
    assert bio_triad.planarian_healing(data, anomaly_score=0.95, recovery_time=1.0) == True
    assert bio_triad.planarian_healing(data, anomaly_score=0.5, recovery_time=1.0) == False

def test_bio_triad_neoblast_hardening(bio_triad):
    """Test Neoblast hardening with adversarial noise."""
    data = torch.rand(128)
    hardened = bio_triad.neoblast_hardening(data, threats=["deepfake"])
    assert torch.all(hardened >= 0) and torch.all(hardened <= 1)  # Clamped for deepfake

# Example usage
if __name__ == "__main__":
    pytest.main(["-v"])