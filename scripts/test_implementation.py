#!/usr/bin/env python3
"""Quick test script for curriculum learning implementation."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.data.curriculum import BasicDifficultyCurriculum
from src.models.simple_cnn import SimpleCNN
from src.utils.device import get_device, set_seed


def test_basic_functionality():
    """Test basic functionality of the curriculum learning implementation."""
    print("🧪 Testing Curriculum Learning Implementation")
    print("=" * 50)
    
    # Set random seed
    set_seed(42)
    
    # Get device
    device = get_device("cpu")  # Use CPU for testing
    print(f"✅ Using device: {device}")
    
    # Test curriculum strategy
    print("\n📚 Testing Curriculum Strategy...")
    curriculum = BasicDifficultyCurriculum(
        num_stages=3,
        noise_levels=[0.0, 0.1, 0.2],
        occlusion_sizes=[0, 5, 10],
    )
    
    print(f"✅ Curriculum strategy created: {curriculum.__class__.__name__}")
    print(f"✅ Number of stages: {curriculum.num_stages}")
    
    # Test stage calculation
    stage = curriculum.get_stage(0, 10)
    assert stage == 0, f"Expected stage 0, got {stage}"
    print(f"✅ Stage calculation works: epoch 0 -> stage {stage}")
    
    # Test model creation
    print("\n🤖 Testing Model Creation...")
    model = SimpleCNN(num_classes=10, dropout=0.2, hidden_dim=128)
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {model.__class__.__name__}")
    print(f"✅ Total parameters: {total_params:,}")
    
    # Test forward pass
    print("\n🔄 Testing Forward Pass...")
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32).to(device)
    outputs = model(inputs)
    
    assert outputs.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {outputs.shape}"
    print(f"✅ Forward pass works: input {inputs.shape} -> output {outputs.shape}")
    
    # Test curriculum application
    print("\n🎯 Testing Curriculum Application...")
    targets = torch.randint(0, 10, (batch_size,))
    batch = (inputs, targets)
    
    for stage in range(curriculum.num_stages):
        modified_batch = curriculum.apply_curriculum(batch, stage)
        modified_inputs, modified_targets = modified_batch
        
        assert modified_inputs.shape == inputs.shape, f"Input shape mismatch at stage {stage}"
        assert torch.equal(modified_targets, targets), f"Targets changed at stage {stage}"
        
        stage_info = curriculum.get_stage_info(stage)
        print(f"✅ Stage {stage}: noise={stage_info['noise_level']:.1f}, occlusion={stage_info['occlusion_size']}")
    
    # Test confidence calculation
    print("\n🎯 Testing Confidence Calculation...")
    confidence = model.get_confidence(inputs)
    assert confidence.shape == (batch_size,), f"Expected confidence shape ({batch_size},), got {confidence.shape}"
    assert torch.all(confidence >= 0) and torch.all(confidence <= 1), "Confidence values should be in [0, 1]"
    print(f"✅ Confidence calculation works: shape {confidence.shape}")
    
    # Test features extraction
    print("\n🔍 Testing Features Extraction...")
    features = model.get_features(inputs)
    assert features.shape == (batch_size, 128), f"Expected features shape ({batch_size}, 128), got {features.shape}"
    print(f"✅ Features extraction works: shape {features.shape}")
    
    print("\n🎉 All tests passed! Curriculum learning implementation is working correctly.")
    print("\n📋 Summary:")
    print(f"  - Curriculum strategy: {curriculum.__class__.__name__}")
    print(f"  - Model: {model.__class__.__name__}")
    print(f"  - Parameters: {total_params:,}")
    print(f"  - Device: {device}")
    print(f"  - Stages: {curriculum.num_stages}")
    
    return True


def test_original_implementation():
    """Test the original implementation for comparison."""
    print("\n🔄 Testing Original Implementation...")
    print("=" * 50)
    
    try:
        # Import original implementation
        from scripts.run_original import SimpleCNN as OriginalCNN, curriculum_learning_step
        
        # Test original model
        model = OriginalCNN()
        print(f"✅ Original model created: {model.__class__.__name__}")
        
        # Test original curriculum step
        data = torch.randn(2, 3, 32, 32)
        modified_data = curriculum_learning_step(data, step=2)
        
        assert modified_data.shape == data.shape, "Original curriculum step should preserve shape"
        print("✅ Original curriculum step works")
        
        # Test forward pass
        outputs = model(data)
        assert outputs.shape == (2, 10), f"Expected shape (2, 10), got {outputs.shape}"
        print("✅ Original forward pass works")
        
        print("✅ Original implementation is working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error testing original implementation: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Curriculum Learning Implementation Test Suite")
    print("=" * 60)
    print("Author: kryptologyst")
    print("GitHub: https://github.com/kryptologyst")
    print("=" * 60)
    
    # Test modern implementation
    modern_success = test_basic_functionality()
    
    # Test original implementation
    original_success = test_original_implementation()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"  Modern Implementation: {'✅ PASS' if modern_success else '❌ FAIL'}")
    print(f"  Original Implementation: {'✅ PASS' if original_success else '❌ FAIL'}")
    
    if modern_success and original_success:
        print("\n🎉 All tests passed! Both implementations are working correctly.")
        print("\n🚀 Ready to run experiments:")
        print("  - python src/experiment.py")
        print("  - python scripts/run_original.py")
        print("  - streamlit run demo/streamlit_demo.py")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
