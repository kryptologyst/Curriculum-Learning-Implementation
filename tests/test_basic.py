"""Basic tests for curriculum learning implementation."""

import pytest
import torch
import torch.nn as nn

from src.data.curriculum import BasicDifficultyCurriculum
from src.models.simple_cnn import SimpleCNN
from src.utils.device import get_device, set_seed


def test_basic_difficulty_curriculum():
    """Test basic difficulty curriculum strategy."""
    curriculum = BasicDifficultyCurriculum(
        num_stages=3,
        noise_levels=[0.0, 0.1, 0.2],
        occlusion_sizes=[0, 5, 10],
    )
    
    # Test stage calculation
    stage = curriculum.get_stage(0, 10)
    assert stage == 0
    
    stage = curriculum.get_stage(5, 10)
    assert stage == 1
    
    stage = curriculum.get_stage(9, 10)
    assert stage == 2
    
    # Test curriculum application
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32)
    targets = torch.randint(0, 10, (batch_size,))
    
    modified_inputs, modified_targets = curriculum.apply_curriculum((inputs, targets), stage=1)
    
    assert modified_inputs.shape == inputs.shape
    assert modified_targets.shape == targets.shape
    assert torch.equal(modified_targets, targets)


def test_simple_cnn():
    """Test SimpleCNN model."""
    model = SimpleCNN(num_classes=10, dropout=0.2, hidden_dim=128)
    
    # Test forward pass
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32)
    outputs = model(inputs)
    
    assert outputs.shape == (batch_size, 10)
    
    # Test features extraction
    features = model.get_features(inputs)
    assert features.shape == (batch_size, 128)
    
    # Test confidence calculation
    confidence = model.get_confidence(inputs)
    assert confidence.shape == (batch_size,)
    assert torch.all(confidence >= 0) and torch.all(confidence <= 1)


def test_device_management():
    """Test device management utilities."""
    device = get_device("cpu")
    assert device.type == "cpu"
    
    # Test seeding
    set_seed(42)
    
    # Test device info
    from src.utils.device import get_device_info
    device_info = get_device_info()
    assert "cuda_available" in device_info
    assert "mps_available" in device_info
    assert "cpu_count" in device_info


def test_curriculum_stage_info():
    """Test curriculum stage information."""
    curriculum = BasicDifficultyCurriculum(num_stages=3)
    
    for stage in range(3):
        stage_info = curriculum.get_stage_info(stage)
        assert "stage" in stage_info
        assert "noise_level" in stage_info
        assert "occlusion_size" in stage_info
        assert "difficulty" in stage_info


if __name__ == "__main__":
    pytest.main([__file__])
